"""Initial artwork becomes conforming subcell spherical material boundaries."""
import math
import unittest
import numpy as np

from raster_engine import Simulation
from mesh_geometry import icosphere,geometry,build_locator,locate_points
from material_initialization import (build_initial_material,_sample_scores,
                                     _source_categories)


def unit(x):
    return x/np.linalg.norm(x,axis=-1,keepdims=True)


class Artwork:
    def __init__(self,category,width=128,rotation=None):
        self.w=width;self.h=width//2
        lon,lat=np.meshgrid((np.arange(self.w)+.5)*2*np.pi/self.w-np.pi,
                           np.pi/2-(np.arange(self.h)+.5)*np.pi/self.h)
        self.xyz=np.column_stack((np.cos(lat).ravel()*np.cos(lon).ravel(),
                                 np.cos(lat).ravel()*np.sin(lon).ravel(),np.sin(lat).ravel()))
        self.plate,self.initial_crust,craton=category(self.xyz)
        self.plate=np.asarray(self.plate,np.int16)
        self.initial_crust=np.asarray(self.initial_crust,np.uint8)
        self.parcel_patch=np.arange(len(self.xyz))
        self.parcel_craton=np.asarray(craton,np.int64)
        edges=np.pi/2-np.arange(self.h+1)*np.pi/self.h
        self.cell_area=np.repeat((np.sin(edges[:-1])-np.sin(edges[1:]))*2*np.pi/self.w*6371.**2,self.w)
        self.rotation=np.eye(3) if rotation is None else rotation

    def _indices(self,points):
        return Simulation._indices(self,points@self.rotation.T)

    def _sample_coordinates(self,points):
        return Simulation._sample_coordinates(self,points@self.rotation.T)


def cap(points):
    axis=unit(np.array([.43,-.27,.8]))
    kind=(points@axis>.28).astype(np.uint8)
    owner=np.where(kind>0,1,0)
    return owner,kind,np.full(len(points),-1)


def intricate(points):
    x,y,z=points.T
    kind=np.where(x+.18*np.sin(5*y-2*z)>.03,1,0)
    kind=np.where((kind>0)&(z>.22)&(x>.3),2,kind)
    # Two touching cratons with the same owner/kind remain different material
    # identities. Non-sequential IDs expose accidental regrouping at import.
    craton=np.where(kind==2,np.where(y>0,91,17),-1)
    owner=np.where(kind==0,0,np.where(y+.13*np.sin(7*z)>.07,1,2))
    return owner,kind,craton


class MaterialInitializationTests(unittest.TestCase):
    def test_complete_partition_is_closed_and_covers_sphere(self):
        partition=build_initial_material(Artwork(intricate),icosphere(2))
        native=geometry(partition['vertices'],partition['faces'])
        self.assertEqual(len(native['vertices'])-len(native['edge_vertices'])+len(native['faces']),2)
        self.assertTrue(np.all(native['edge_faces']>=0))
        self.assertAlmostEqual(partition['area_km2'].sum()/(4*math.pi*6371.**2),1.,places=13)
        points=unit(np.random.default_rng(192).normal(size=(4000,3)))
        cells,_=locate_points(points,build_locator(partition['vertices'],partition['faces']))
        self.assertTrue(np.all(cells>=0))

    def test_seed_cells_match_all_three_categories_and_preserve_craton_ids(self):
        source=Artwork(intricate)
        partition=build_initial_material(source,icosphere(2))
        cells=partition['source_seed_cells']
        np.testing.assert_array_equal(partition['face_kind'],source.initial_crust[cells])
        np.testing.assert_array_equal(partition['face_owner'],source.plate[cells])
        np.testing.assert_array_equal(partition['face_craton'],source.parcel_craton[cells])
        self.assertEqual(set(partition['face_craton']),{-1,17,91})
        self.assertGreater(partition['diagnostics']['corrected_seed_cells'],0)

    def test_unmodified_artwork_and_repeatable_topology(self):
        source=Artwork(intricate)
        original=[a.copy() for a in (source.plate,source.initial_crust,source.parcel_craton)]
        mesh=icosphere(2)
        first=build_initial_material(source,mesh)
        second=build_initial_material(source,mesh)
        for key in ('vertices','faces','face_owner','face_kind','face_craton','source_seed_cells'):
            np.testing.assert_array_equal(first[key],second[key])
        for before,after in zip(original,(source.plate,source.initial_crust,source.parcel_craton)):
            np.testing.assert_array_equal(before,after)

    def test_initial_contours_and_identity_rotate_across_poles(self):
        mesh=icosphere(2)
        q,_=np.linalg.qr(np.random.default_rng(33).normal(size=(3,3)))
        q[:,0]*=np.linalg.det(q)
        first=build_initial_material(Artwork(intricate),mesh)
        moved=build_initial_material(Artwork(intricate,rotation=q),geometry(mesh['vertices']@q,mesh['faces']))
        for key in ('faces','face_owner','face_kind','face_craton','source_seed_cells'):
            np.testing.assert_array_equal(first[key],moved[key])
        np.testing.assert_allclose(first['vertices']@q,moved['vertices'],atol=3e-13)
        np.testing.assert_allclose(first['area_km2'],moved['area_km2'],rtol=2e-8,atol=1e-8)

    def test_unreffined_contours_agree_with_shared_p1_category_winners(self):
        source=Artwork(intricate)
        mesh=icosphere(2)
        partition=build_initial_material(source,mesh,refine_boundaries=False)
        table,codes=_source_categories(source,source.plate,source.initial_crust)
        keys,weights=_sample_scores(source,mesh['vertices'],np.arange(len(table)),codes)
        dense=np.zeros((len(mesh['vertices']),len(table)))
        for i in range(4):np.add.at(dense,(np.arange(len(keys)),keys[:,i]),weights[:,i])
        points=unit(np.random.default_rng(10).normal(size=(3000,3)))
        cell,bary=locate_points(points,build_locator(mesh['vertices'],mesh['faces']))
        winner=np.argmax(np.einsum('ni,nij->nj',bary,dense[mesh['faces'][cell]]),axis=1)
        fitted,_=locate_points(points,build_locator(partition['vertices'],partition['faces']))
        observed=np.column_stack((partition['face_owner'][fitted],partition['face_kind'][fitted],partition['face_craton'][fitted]))
        np.testing.assert_array_equal(observed,table[winner])

    def test_coast_cuts_triangle_interiors_and_improves_small_circle_geometry(self):
        source=Artwork(cap,width=256)
        mesh=icosphere(3)
        axis=unit(np.array([.43,-.27,.8]))
        old_kind=source.initial_crust[source._indices(mesh['xyz'])]
        old_edges=mesh['edge_vertices'][old_kind[mesh['edge_faces'][:,0]]!=old_kind[mesh['edge_faces'][:,1]]]
        old_points=mesh['vertices'][np.unique(old_edges)]
        error=lambda points:np.sqrt(np.mean((np.arcsin(points@axis)-np.arcsin(.28))**2))
        errors=[]
        for refine in (False,True):
            partition=build_initial_material(source,mesh,refine_boundaries=refine)
            fitted=geometry(partition['vertices'],partition['faces'])
            kind=partition['face_kind']
            coast=fitted['edge_vertices'][kind[fitted['edge_faces'][:,0]]!=kind[fitted['edge_faces'][:,1]]]
            coast_points=fitted['vertices'][np.unique(coast)]
            errors.append(error(coast_points))
            # Most contour vertices lie in the interiors of old faces/edges;
            # they are not old mesh corners with a cosmetically blurred map.
            corners=np.max(coast_points@mesh['vertices'].T,axis=1)>1-1e-12
            self.assertLess(np.mean(corners),.05)
            self.assertLess(errors[-1],error(old_points)*(.3 if refine else .7))
            expected=2*np.pi*6371.**2*(1-.28)
            self.assertLess(abs(partition['area_km2'][kind>0].sum()/expected-1),.006)
        self.assertLess(errors[1],errors[0])

    def test_adaptive_refinement_is_local_and_center_only_island_survives(self):
        mesh=icosphere(1)
        axis=mesh['xyz'][13]
        def island(points):
            kind=(points@axis>np.cos(.065)).astype(np.uint8)
            return kind,kind,np.full(len(points),-1)
        source=Artwork(island,width=512)
        coarse=build_initial_material(source,mesh,refine_boundaries=False)
        refined=build_initial_material(source,mesh)
        self.assertFalse(np.any(coarse['face_kind']))
        self.assertTrue(np.any(refined['face_kind']))
        self.assertGreater(refined['diagnostics']['interior_feature_points'],0)
        self.assertLess(refined['diagnostics']['working_faces'],len(mesh['faces'])*2)
        self.assertGreater(refined['diagnostics']['green_control_faces'],0)

    def test_uniform_artwork_adds_no_faces_and_needs_no_craton_ledger(self):
        for kind in (0,1,2):
            def uniform(points):
                return np.zeros(len(points),int),np.full(len(points),kind),np.full(len(points),-1)
            source=Artwork(uniform,width=48)
            del source.parcel_craton,source.parcel_patch
            mesh=icosphere(2)
            partition=build_initial_material(source,mesh)
            self.assertEqual(len(partition['faces']),len(mesh['faces']))
            np.testing.assert_array_equal(partition['vertices'],mesh['vertices'])
            np.testing.assert_array_equal(partition['faces'],mesh['faces'])
            self.assertTrue(np.all(partition['face_kind']==kind))
            self.assertTrue(np.all(partition['face_craton']==(0 if kind==2 else -1)))

    def test_sparse_score_storage_does_not_scale_with_craton_count(self):
        source=Artwork(intricate)
        codes=np.arange(len(source.xyz))
        points=icosphere(3)['vertices']
        keys,weights=_sample_scores(source,points,codes,codes)
        self.assertEqual(keys.shape,(len(points),4))
        self.assertEqual(weights.shape,(len(points),4))
        np.testing.assert_allclose(weights.sum(axis=1),1.,atol=2e-12)


if __name__=='__main__':unittest.main()
