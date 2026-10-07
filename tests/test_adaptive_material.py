"""Geometry, conservative transfers and reversible material adaptation."""
import unittest
import numpy as np

import adaptive_material as adaptive
from mesh_geometry import icosphere,geometry,build_locator,locate_points


def normalized(points):return points/np.linalg.norm(points,axis=-1,keepdims=True)


def refine(mesh,**kwargs):
    n=len(mesh['faces'])
    return adaptive.refine(mesh['vertices'],mesh['faces'],np.zeros(n,int),np.ones(n,int),
        desired_edge_km=kwargs.pop('desired_edge_km',500.),**kwargs)


def coarsen(result,**kwargs):
    return adaptive.coarsen(result['vertices'],result['faces'],result['face_owner'],result['face_kind'],
        result['registry'],face_level=result['face_level'],categories=result['categories'],**kwargs)


class AdaptiveMaterialTests(unittest.TestCase):
    def test_closed_global_refinement_preserves_coverage_area_and_existing_vertices(self):
        mesh=icosphere(2)
        result=refine(mesh)
        np.testing.assert_array_equal(result['vertices'][:len(mesh['vertices'])],mesh['vertices'])
        refined=geometry(result['vertices'],result['faces'])
        self.assertEqual(len(refined['vertices'])-len(refined['edge_vertices'])+len(refined['faces']),2)
        self.assertTrue(np.all(refined['edge_faces']>=0))
        self.assertAlmostEqual(result['area_km2'].sum()/mesh['area_km2'].sum(),1.,places=13)
        points=normalized(np.random.default_rng(43).normal(size=(4000,3)))
        child,_=locate_points(points,build_locator(result['vertices'],result['faces']))
        parent,_=locate_points(points,build_locator(mesh['vertices'],mesh['faces']))
        np.testing.assert_array_equal(result['source_face'][child],parent)
        sums=np.bincount(result['source_face'],weights=result['area_fraction'],minlength=len(mesh['faces']))
        np.testing.assert_allclose(sums,1.,atol=3e-16)

    def test_local_belt_refines_protected_neighbors_without_straining_craton(self):
        mesh=icosphere(2)
        protected=np.ones(len(mesh['faces']),bool);protected[10]=False
        result=refine(mesh,protected=protected)
        refined=geometry(result['vertices'],result['faces'])
        self.assertTrue(np.all(refined['edge_faces']>=0))
        self.assertEqual(result['diagnostics']['split_edges'],3)
        self.assertEqual(result['diagnostics']['protected_subdivided_faces'],3)
        self.assertEqual(len(result['faces']),len(mesh['faces'])+6)
        np.testing.assert_array_equal(result['vertices'][:len(mesh['vertices'])],mesh['vertices'])
        # Only four affected source families occupy the enduring registry.
        self.assertEqual(len(result['registry']['input_faces']),4)

    def test_face_budget_and_level_limits_include_all_green_closure(self):
        mesh=icosphere(2);n=len(mesh['faces'])
        for extra in (0,1,2,3,11,60):
            result=refine(mesh,max_faces=n+extra)
            self.assertLessEqual(len(result['faces']),n+extra)
            self.assertTrue(np.all(geometry(result['vertices'],result['faces'])['edge_faces']>=0))
        level=np.ones(n,int);level[0]=0
        result=refine(mesh,face_level=level,max_level=1)
        # Every edge requested by face zero touches a neighbor already at max.
        self.assertEqual(len(result['faces']),n)
        self.assertEqual(result['diagnostics']['split_edges'],0)
        with self.assertRaises(ValueError):refine(mesh,max_faces=n-1)

    def test_repeated_generations_reduce_lengths_in_requested_region(self):
        mesh=icosphere(1);original=mesh['edge_length'].max();level=np.zeros(len(mesh['faces']),int)
        for _ in range(3):
            result=refine(mesh,desired_edge_km=1.,face_level=level,max_level=3)
            mesh=geometry(result['vertices'],result['faces']);level=result['face_level']
        self.assertLess(mesh['edge_length'].max(),original*.14)
        self.assertTrue(np.all(level==3))

    def test_category_owner_and_craton_interfaces_are_never_mixed_on_split(self):
        mesh=icosphere(1);n=len(mesh['faces'])
        owners=np.arange(n)%3;kinds=1+np.arange(n)%2
        categories=np.column_stack((np.arange(n)//3,100+np.arange(n)%2))
        result=adaptive.refine(mesh['vertices'],mesh['faces'],owners,kinds,desired_edge_km=1.,categories=categories)
        for key,original in [('face_owner',owners),('face_kind',kinds),('categories',categories)]:
            np.testing.assert_array_equal(result[key],original[result['source_face']])

    def test_exact_registered_undo_restores_parent_connectivity_and_id(self):
        mesh=icosphere(2);n=len(mesh['faces']);ids=np.arange(n)+7000
        result=refine(mesh,face_ids=ids)
        child_ids=np.arange(len(result['faces']))+10000
        adaptive.bind_registry(result['registry'],child_ids)
        restored=coarsen(result,face_ids=child_ids)
        np.testing.assert_array_equal(restored['faces'],mesh['faces'])
        np.testing.assert_array_equal(restored['restored_parent_id'],ids)
        self.assertEqual(restored['diagnostics']['coarsened_families'],n)
        np.testing.assert_array_equal(restored['vertices'],result['vertices'])

    def test_mass_and_volume_are_conserved_through_refine_and_weighted_merge(self):
        mesh=icosphere(1);n=len(mesh['faces']);rng=np.random.default_rng(5)
        mass=mesh['area_km2']*rng.uniform(.5,1.9,n)
        thickness=rng.uniform(7,65,n)
        result=refine(mesh)
        child_mass=adaptive.transfer_extensive(mass,result)
        child_thickness=adaptive.transfer_intensive(thickness,result)
        self.assertAlmostEqual(child_mass.sum()/mass.sum(),1.,places=14)
        self.assertAlmostEqual(np.sum(child_mass*child_thickness)/np.sum(mass*thickness),1.,places=14)
        # New subcell column variation must be volume-averaged when coarsening.
        child_thickness+=rng.uniform(-1,1,len(child_mass))
        restored=coarsen(result)
        new_mass=adaptive.transfer_extensive(child_mass,restored)
        new_thickness=adaptive.transfer_intensive(child_thickness,restored,weights=child_mass)
        np.testing.assert_allclose(new_mass,mass,rtol=4e-16)
        self.assertAlmostEqual(np.sum(new_mass*new_thickness)/np.sum(child_mass*child_thickness),1.,places=14)

    def test_active_family_blocks_neighbor_closure_instead_of_creating_t_junction(self):
        mesh=icosphere(1);result=refine(mesh)
        active=np.zeros(len(result['faces']),bool);active[0]=True
        restored=coarsen(result,active=active)
        self.assertEqual(len(restored['faces']),len(result['faces']))
        self.assertGreater(restored['diagnostics']['rejected']['neighbor_closure'],0)
        self.assertTrue(np.all(geometry(restored['vertices'],restored['faces'])['edge_faces']>=0))

    def test_detached_patch_can_coarsen_while_other_patch_stays_active(self):
        mesh=icosphere(1)
        original_faces=mesh['faces'][[0,40]]
        result=adaptive.refine(mesh['vertices'],original_faces,[0,1],[1,2],desired_edge_km=1.)
        active=result['source_face']==0
        restored=coarsen(result,active=active)
        self.assertEqual(len(restored['faces']),5)
        self.assertEqual(restored['diagnostics']['coarsened_families'],1)
        np.testing.assert_array_equal(restored['faces'][-1],original_faces[1])

    def test_changed_sibling_owner_kind_category_or_missing_child_prevents_undo(self):
        mesh=icosphere(1);f=mesh['faces'][:1]
        result=adaptive.refine(mesh['vertices'],f,[0],[1],desired_edge_km=1.,categories=[11])
        for field in ('face_owner','face_kind','categories'):
            changed={key:(value.copy() if isinstance(value,np.ndarray) else value) for key,value in result.items()}
            changed[field][0]+=1
            restored=coarsen(changed)
            self.assertEqual(len(restored['faces']),4)
            self.assertEqual(restored['diagnostics']['rejected']['category'],1)
        missing=adaptive.coarsen(result['vertices'],result['faces'][:-1],result['face_owner'][:-1],
            result['face_kind'][:-1],result['registry'],face_ids=np.arange(3),categories=result['categories'][:-1])
        self.assertEqual(len(missing['faces']),3)
        self.assertEqual(missing['diagnostics']['rejected']['incomplete'],1)

    def test_deformed_midpoint_is_not_flattened_by_coarsening(self):
        mesh=icosphere(1);result=refine(mesh)
        result['vertices'][len(mesh['vertices'])]=normalized(result['vertices'][len(mesh['vertices'])]+[0,.001,0])
        restored=coarsen(result)
        self.assertGreater(restored['diagnostics']['rejected']['footprint'],0)
        np.testing.assert_array_equal(restored['vertices'],result['vertices'])
        self.assertEqual(len(restored['faces']),len(result['faces']))

    def test_refine_and_undo_are_pole_rotation_invariant(self):
        mesh=icosphere(2)
        q,_=np.linalg.qr(np.random.default_rng(2).normal(size=(3,3)));q[:,0]*=np.linalg.det(q)
        first=refine(mesh,max_faces=len(mesh['faces'])+130)
        rotated=geometry(mesh['vertices']@q,mesh['faces'])
        second=refine(rotated,max_faces=len(mesh['faces'])+130)
        np.testing.assert_array_equal(first['faces'],second['faces'])
        np.testing.assert_allclose(first['vertices']@q,second['vertices'],atol=5e-16)
        np.testing.assert_allclose(first['area_fraction'],second['area_fraction'],atol=3e-15)
        np.testing.assert_array_equal(coarsen(first)['faces'],coarsen(second)['faces'])

    def test_chart_corner_scales_preserve_exact_spherical_material_coordinates(self):
        mesh=icosphere(1);result=refine(mesh)
        rng=np.random.default_rng(6);chosen=rng.integers(0,len(result['faces']),100)
        bary=rng.uniform(.1,1,(len(chosen),3));bary/=bary.sum(axis=1)[:,None]
        point=normalized(np.einsum('ni,nij->nj',bary,result['vertices'][result['faces'][chosen]]))
        inherited=np.einsum('ni,nij->nj',bary/result['corner_radial_scale'][chosen],result['corner_barycentric'][chosen])
        inherited/=inherited.sum(axis=1)[:,None]
        expected=normalized(np.einsum('ni,nij->nj',inherited,mesh['vertices'][mesh['faces'][result['source_face'][chosen]]]))
        np.testing.assert_allclose(point,expected,atol=4e-16)

    def test_empty_and_noop_registry_are_safe(self):
        result=adaptive.refine(np.empty((0,3)),np.empty((0,3),int),[],[],desired_edge_km=1.)
        self.assertEqual(result['faces'].shape,(0,3))
        self.assertEqual(coarsen(result)['faces'].shape,(0,3))
        mesh=icosphere(1);result=refine(mesh,desired_edge_km=np.inf)
        self.assertEqual(len(result['registry']['input_faces']),0)
        np.testing.assert_array_equal(coarsen(result)['faces'],mesh['faces'])


if __name__=='__main__':unittest.main()
