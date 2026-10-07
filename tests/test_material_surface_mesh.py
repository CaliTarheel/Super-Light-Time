"""Connected spherical material transport, exact coverage and enduring state."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import material_surface as material
from mesh_geometry import icosphere
from ridge_geometry import rotate


def two_faces():
    vertices = np.array([[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    faces = np.array([[3,0,1],[3,1,2]])
    return vertices, faces


def patch(**kwargs):
    vertices, faces = two_faces()
    return material.initialize_surface(vertices,faces,np.array([0,0]),np.array([2,2]),
                                       face_id=np.array([41,42]),**kwargs)


def frozen_material(surface):
    return deepcopy({key:surface[key] for key in ('face_id','face_kind','face_region',
                    'reference_area_km2','columns','provenance')})


class MaterialSurfaceMeshTests(unittest.TestCase):
    def assert_tree_equal(self, first, second):
        if isinstance(first, np.ndarray):
            self.assertEqual(first.dtype, second.dtype)
            np.testing.assert_array_equal(first,second)
        elif isinstance(first, dict):
            self.assertEqual(set(first),set(second))
            for key in first:
                self.assert_tree_equal(first[key],second[key])
        else:
            self.assertEqual(first,second)

    def test_connected_closed_continent_has_shared_vertices_and_exact_sphere_area(self):
        mesh=icosphere(2)
        count=len(mesh['faces'])
        s=material.initialize_surface(mesh['vertices'],mesh['faces'],np.zeros(count,int),np.ones(count,int))
        self.assertEqual(len(s['vertices']),len(mesh['vertices']))
        self.assertLess(len(s['vertices']),len(s['faces']))
        self.assertAlmostEqual(float(s['area_km2'].sum())/(4*np.pi*6371.**2),1.,places=13)
        edges=np.sort(np.concatenate([s['faces'][:,[0,1]],s['faces'][:,[1,2]],s['faces'][:,[2,0]]]),axis=1)
        _,counts=np.unique(edges,axis=0,return_counts=True)
        np.testing.assert_array_equal(counts,2)
        np.testing.assert_array_equal(s['area_km2'],s['reference_area_km2'])

    def test_ocean_filter_preserves_input_face_ids_kinds_columns_and_provenance(self):
        mesh=icosphere(1); count=len(mesh['faces'])
        kinds=np.arange(count)%4
        thickness=30.+np.arange(count)/100.
        ids=10_000+np.arange(count)
        birth=np.arange(count,dtype=float)*2.
        s=material.initialize_surface(mesh['vertices'],mesh['faces'],np.zeros(count,int),kinds,
            face_id=ids,columns={'thickness_km':thickness},provenance={'birth_myr':birth})
        keep=kinds>0
        np.testing.assert_array_equal(s['face_kind'],kinds[keep])
        np.testing.assert_array_equal(s['face_id'],ids[keep])
        np.testing.assert_array_equal(s['columns']['thickness_km'],thickness[keep])
        np.testing.assert_array_equal(s['provenance']['birth_myr'],birth[keep])
        thickness[:]=-999.; birth[:]=-999.
        self.assertTrue(np.all(s['columns']['thickness_km']>0))
        self.assertTrue(np.all(s['provenance']['birth_myr']>=0))

    def test_fault_split_duplicates_only_shared_side_vertices_without_displacement(self):
        s=patch(columns={'thickness_km':np.array([39.,42.])},
                provenance={'craton_id':np.array([77,77])})
        old=frozen_material(s)
        triangles=s['vertices'][s['faces']].copy()
        original_vertices=s['vertices'].copy()
        self.assertEqual(len(original_vertices),4)
        result=material.reassign_owners(s,np.array([0,3]))
        self.assertEqual(result,dict(duplicated_vertices=2,changed_faces=1))
        self.assertEqual(len(s['vertices']),6)
        np.testing.assert_array_equal(s['vertices'][:4],original_vertices)
        np.testing.assert_array_equal(s['vertices'][s['faces']],triangles)
        np.testing.assert_array_equal(s['vertex_owner'][s['faces']],np.repeat([[0],[3]],3,axis=1))
        self.assert_tree_equal(frozen_material(s),old)
        before=deepcopy(s)
        self.assertEqual(material.reassign_owners(s,np.array([0,3])),dict(duplicated_vertices=0,changed_faces=0))
        self.assert_tree_equal(s,before)

    def test_initial_owner_or_region_boundary_has_independent_coincident_sides(self):
        vertices,faces=two_faces()
        for owners,regions in [(np.array([0,1]),np.array([0,0])),
                                (np.array([0,0]),np.array([6,7]))]:
            s=material.initialize_surface(vertices,faces,owners,np.ones(2,int),face_region=regions)
            self.assertEqual(len(s['vertices']),6)
            self.assertEqual(len(np.intersect1d(s['faces'][0],s['faces'][1])),0)
            np.testing.assert_array_equal(s['vertices'][s['faces']],vertices[faces])

    def test_many_polar_rotations_preserve_craton_geometry_area_and_all_material_state(self):
        mesh=icosphere(2); count=len(mesh['faces'])
        s=material.initialize_surface(mesh['vertices'],mesh['faces'],np.full(count,5),np.full(count,2),
            columns={'thickness_km':np.full(count,42.),'heat_m':np.linspace(0,50,count)},
            provenance={'birth_myr':np.zeros(count),'source_region':np.arange(count)%7})
        old=frozen_material(s)
        source=s['vertices'].copy()
        query=material.face_centres(s)
        initial_area=s['area_km2'].copy()
        omega=np.array([.007,.028,-.013])
        for _ in range(100):
            material.advect_surface(s,{5:omega},2.)
        np.testing.assert_allclose(np.linalg.norm(s['vertices'],axis=1),1.,atol=3e-15)
        np.testing.assert_allclose(s['vertices']@s['vertices'].T,source@source.T,atol=1e-13)
        np.testing.assert_allclose(s['area_km2'],initial_area,rtol=2e-13,atol=1e-8)
        self.assert_tree_equal(frozen_material(s),old)
        moved_query=rotate(query,omega*200.)
        hits=material.sample_surface(s,moved_query)
        np.testing.assert_array_equal(hits['query_index'],np.arange(count))
        np.testing.assert_array_equal(hits['face_id'],np.arange(count))
        self.assertEqual(len(hits['uncovered_query_index']),0)
        material.advect_surface(s,{5:omega},-200.)
        np.testing.assert_allclose(s['vertices'],source,atol=2e-14)

    def test_separated_owners_can_move_independently_without_straining_any_triangle(self):
        s=patch()
        material.reassign_owners(s,np.array([0,1]))
        before=s['vertices'][s['faces']].copy()
        reference=s['reference_area_km2'].copy()
        omegas=np.array([[0.,.08,0.],[.04,0.,-.06]])
        material.advect_surface(s,omegas,10.)
        after=s['vertices'][s['faces']]
        for face,owner in enumerate(s['face_owner']):
            np.testing.assert_allclose(after[face],rotate(before[face],omegas[owner]*10.),atol=2e-15)
            np.testing.assert_allclose(after[face]@after[face].T,before[face]@before[face].T,atol=2e-15)
        np.testing.assert_allclose(s['area_km2'],reference,rtol=2e-15)
        np.testing.assert_array_equal(s['reference_area_km2'],reference)

    def test_exact_triangle_sampling_leaves_a_real_open_hole_unfilled(self):
        mesh=icosphere(2); count=len(mesh['faces'])
        kinds=np.ones(count,int); kinds[[0,7,119]]=0
        s=material.initialize_surface(mesh['vertices'],mesh['faces'],np.zeros(count,int),kinds)
        centres=np.sum(mesh['vertices'][mesh['faces']],axis=1)
        centres/=np.linalg.norm(centres,axis=1)[:,None]
        hits=material.sample_surface(s,centres)
        np.testing.assert_array_equal(hits['uncovered_query_index'],[0,7,119])
        np.testing.assert_array_equal(hits['face_id'],hits['query_index'])
        np.testing.assert_allclose(hits['weights'].sum(axis=1),1.,atol=1e-14)
        self.assertTrue(np.all(hits['weights']>=-1e-12))

    def test_internal_edges_and_overlapping_patches_return_every_candidate(self):
        s=patch()
        midpoint=np.array([[0.,1.,1.]])/np.sqrt(2.)
        hits=material.sample_surface(s,midpoint)
        np.testing.assert_array_equal(hits['query_index'],[0,0])
        np.testing.assert_array_equal(hits['face_id'],[41,42])
        vertices,faces=two_faces()
        material.append_surface(s,vertices,faces,np.array([7,7]),np.array([3,3]),face_id=np.array([81,82]))
        hits=material.sample_surface(s,midpoint)
        np.testing.assert_array_equal(hits['face_id'],[41,42,81,82])
        np.testing.assert_array_equal(hits['owner'],[0,0,7,7])
        np.testing.assert_array_equal(hits['kind'],[2,2,3,3])

    def test_append_arc_patch_keeps_internal_connectivity_without_welding_old_crust(self):
        s=patch(columns={'thickness_km':np.array([39.,42.])})
        old=frozen_material(s); old_vertices=s['vertices'].copy(); old_faces=s['faces'].copy()
        vertices,faces=two_faces()
        result=material.append_surface(s,vertices,faces,np.array([0,0]),np.array([3,3]),
                                        columns={'thickness_km':np.array([12.,14.])})
        np.testing.assert_array_equal(result['face_indices'],[2,3])
        np.testing.assert_array_equal(result['face_ids'],[43,44])
        self.assertEqual(result['added_vertices'],4)
        np.testing.assert_array_equal(s['faces'][:2],old_faces)
        np.testing.assert_array_equal(s['vertices'][:4],old_vertices)
        self.assertEqual(len(np.intersect1d(s['faces'][2],s['faces'][3])),2)
        self.assertEqual(len(np.intersect1d(s['faces'][:2],s['faces'][2:])),0)
        for key in ('face_id','face_kind','reference_area_km2'):
            np.testing.assert_array_equal(s[key][:2],old[key])
        np.testing.assert_array_equal(s['columns']['thickness_km'],[39.,42.,12.,14.])

    def test_invalid_append_and_missing_angular_velocity_are_atomic(self):
        s=patch(columns={'thickness_km':np.array([39.,42.])})
        before=deepcopy(s); vertices,faces=two_faces()
        with self.assertRaises(ValueError):
            material.append_surface(s,vertices,faces,np.zeros(2,int),np.ones(2,int))
        self.assert_tree_equal(s,before)
        with self.assertRaises(ValueError):
            material.append_surface(s,vertices,faces,np.zeros(2,int),np.ones(2,int),
                                     face_id=np.array([41,99]),columns={'thickness_km':np.ones(2)})
        self.assert_tree_equal(s,before)
        with self.assertRaises(ValueError): material.advect_surface(s,{},2.)
        self.assert_tree_equal(s,before)

    def test_geometry_refresh_updates_area_but_never_reference_or_columns(self):
        s=patch(columns={'thickness_km':np.array([39.,42.])})
        old=frozen_material(s); old_area=s['area_km2'].copy()
        s['vertices']=rotate(s['vertices'],[.7,-.3,.4])
        refreshed=material.refresh_geometry(s)
        np.testing.assert_allclose(refreshed,old_area,rtol=2e-15)
        self.assert_tree_equal(frozen_material(s),old)
        s['vertices'][0]*=2.
        with self.assertRaises(ValueError): material.refresh_geometry(s)

    def test_geometry_and_owner_changes_reject_stale_search_indices(self):
        s=patch(); query=material.face_centres(s)
        index=material.build_surface_locator(s)
        material.reassign_owners(s,np.array([0,1]))
        with self.assertRaises(ValueError): material.sample_surface(s,query,locator=index)
        index=material.build_surface_locator(s)
        material.advect_surface(s,{0:[0.,.1,0.],1:[0.,0.,.1]},2.)
        with self.assertRaises(ValueError): material.sample_surface(s,query,locator=index)

    def test_state_round_trips_through_existing_plain_checkpoint_encoding(self):
        from checkpoint import _encode,_decode
        s=patch(columns={'thickness_km':np.array([39.,42.])},provenance={'name':np.array(['old','new'])})
        material.reassign_owners(s,np.array([0,8]))
        arrays={}; tree=_encode(s,arrays)
        tree=json.loads(json.dumps(tree,allow_nan=False))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'surface.npz'; np.savez_compressed(path,**arrays)
            with np.load(path,allow_pickle=False) as archive:
                restored=_decode(tree,archive)
        self.assert_tree_equal(s,restored)
        velocity={0:[.01,.02,0.],8:[0.,-.03,.01]}
        material.advect_surface(s,velocity,2.); material.advect_surface(restored,velocity,2.)
        self.assert_tree_equal(s,restored)

    def test_empty_ocean_surface_reports_every_query_uncovered(self):
        mesh=icosphere(0); count=len(mesh['faces'])
        s=material.initialize_surface(mesh['vertices'],mesh['faces'],np.zeros(count,int),np.zeros(count,int))
        self.assertEqual(s['vertices'].shape,(0,3)); self.assertEqual(s['faces'].shape,(0,3))
        material.advect_surface(s,{},2.)
        hits=material.sample_surface(s,np.eye(3))
        self.assertEqual(len(hits['face_index']),0)
        np.testing.assert_array_equal(hits['uncovered_query_index'],[0,1,2])


if __name__=='__main__': unittest.main()
