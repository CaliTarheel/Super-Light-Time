"""Finite native continental margins preserve geometry and real ocean gaps."""
from copy import deepcopy
import unittest
import numpy as np
import continental_margin as margin
import native_frame_sampling as sampling
import mesh_geometry as geometry
from orientation import orient_frame, rotation_matrix
from tests.test_material_reconstruction import continuous_frame, patch_geometry
from tests.test_arc_surface_envelope import probes
from tests.test_material_initialization import Artwork, cap
from material_initialization import build_initial_material


def unit(x): return x/np.linalg.norm(x, axis=-1, keepdims=True)


def source_frame():
    source = continuous_frame()
    source.update(arc_material_version=1, arc_surface_version=2,
        material_arc_id=np.zeros(2, np.int64), material_arc_basal_m=np.zeros(2))
    source['material_height_m'][:] = 1200.
    return source


class ContinentalMarginTests(unittest.TestCase):
    def test_profile_explicit_slope_shelf_coast_and_interior(self):
        x = np.array([0., .35, .75, 1., 2.])
        values = margin.profile_height(1000., -5000., x)
        np.testing.assert_allclose(values, [-5000., -180., -20., 1000., 1000.])
        self.assertEqual(float(margin.profile_height(1000., -5000., 0.)), -5000.)
        self.assertTrue(np.all(np.diff(margin.profile_height(1000., -5000., np.linspace(0,1,100))) >= 0))
        self.assertLessEqual(margin.profile_height(-1000., -5000., np.array([.4,.6])).max(), -1000.)

    def test_saved_legacy_untouched_and_explicit_upgrade_is_idempotent(self):
        old = source_frame(); before = deepcopy(old)
        new = margin.upgrade_frame(old)
        self.assertNotIn('continental_margin_version', old)
        self.assertTrue(new['continental_margin_revision']['physical_arrays_unchanged'])
        for key, value in old.items():
            if isinstance(value, np.ndarray):
                self.assertIs(new[key], value)
                np.testing.assert_array_equal(value, before[key])
        self.assertEqual(margin.upgrade_frame(new)['continental_margin_revision'],new['continental_margin_revision'])
        invalid = dict(old, continental_margin_version=2)
        with self.assertRaises(ValueError): margin.upgrade_frame(invalid)

    def test_true_open_edge_converges_to_same_ocean_without_image_resolution(self):
        old = source_frame(); new = margin.upgrade_frame(old)
        context = sampling.prepare(new)
        a, b = probes(old, 1e-8)
        legacy_jump = np.abs(sampling.sample_frame(old,a)['elevation']-sampling.sample_frame(old,b)['elevation'])
        corrected_jump = np.abs(sampling.sample_frame(new,a,context)['elevation']-sampling.sample_frame(new,b,context)['elevation'])
        self.assertGreater(legacy_jump.max(),4000.)
        self.assertLess(corrected_jump.max(),1e-3)
        result = sampling.sample_frame(new, np.array([[1.,0.,0.],[0.,1.,0.]]), context)
        self.assertEqual(result['elevation'][0],1200.)
        self.assertEqual(result['material_face'][1],-1)

    def test_duplicated_coincident_fault_is_not_a_false_coast(self):
        v,f = patch_geometry()
        # Each face owns private vertex indices, as after a plate split.
        v = v[f].reshape(-1,3); f = np.arange(6).reshape(2,3)
        prepared = margin.prepare(v,f,np.ones(2,np.uint8),np.zeros(2,int))
        self.assertEqual(prepared['open_edge_count'],4)
        middle=unit(v[0]+v[2])
        d=margin.distance_km(prepared,np.array([middle,middle]),np.array([0,1]))
        self.assertTrue(np.all(d>1000.))
        self.assertEqual(prepared['components'][0],prepared['components'][1])

    def test_same_facing_coincident_sheets_keep_their_own_free_coasts(self):
        v,f = patch_geometry()
        vertices = np.r_[v,v]
        faces = np.r_[f,f+len(v)]
        prepared = margin.prepare(vertices,faces,np.ones(4,np.uint8),np.zeros(4,int))
        self.assertEqual(prepared['open_edge_count'],8)
        self.assertEqual(len(np.unique(prepared['components'])),2)
        self.assertNotEqual(prepared['components'][0],prepared['components'][2])
        midpoint = unit(v[f[0,0]]+v[f[0,1]])
        d = margin.distance_km(prepared,np.array([midpoint,midpoint]),np.array([0,2]))
        np.testing.assert_allclose(d,0.,atol=1e-8)
        baseline = margin.prepare(v,f,np.ones(2,np.uint8),np.zeros(2,int))
        np.testing.assert_allclose(prepared['component_width_km'],baseline['component_width_km'][0],rtol=1e-12)

    def test_same_directed_repeated_index_edge_is_not_an_internal_pair(self):
        import continental_contacts
        v,f = patch_geometry()
        duplicated = np.repeat(f[:1],2,axis=0)
        result = continental_contacts.free_outline(v,duplicated)
        self.assertEqual(len(result['a']),6)
        self.assertEqual(len(result['joins']),0)

    def test_actual_shared_internal_edge_is_paired_before_coincident_overlay(self):
        v,f = patch_geometry()
        # A separate triangle overlays half of a connected two-triangle
        # sheet. The real internal diagonal belongs to that original sheet;
        # the overlying triangle still has its own finite diagonal toe.
        vertices=np.r_[v,v[f[0]]]
        faces=np.r_[f,np.array([[len(v),len(v)+1,len(v)+2]])]
        prepared=margin.prepare(vertices,faces,np.ones(3,np.uint8),np.zeros(3,int))
        self.assertEqual(prepared['open_edge_count'],7)
        self.assertEqual(prepared['components'][0],prepared['components'][1])
        self.assertNotEqual(prepared['components'][0],prepared['components'][2])

    def test_differently_subdivided_coincident_fault_cancels_exact_intervals(self):
        v,_=patch_geometry();mid=unit(v[0]+v[2])
        vertices=np.r_[v[:3],mid[None],v[[0,2,3]]]
        faces=np.array([[0,1,3],[3,1,2],[4,5,6]])
        prepared=margin.prepare(vertices,faces,np.ones(3,np.uint8),np.zeros(3,int))
        self.assertEqual(prepared['open_edge_count'],4)
        self.assertEqual(prepared['contact_diagnostics']['partial_contact_pairs'],2)
        self.assertEqual(len(np.unique(prepared['components'])),1)
        height,_=margin.apply(prepared,mid[None],np.array([0]),np.array([1000.]),np.array([-5000.]))
        self.assertEqual(height[0],1000.)
        # A one-metre physical opening must not be snapped closed.
        moved=vertices.copy();normal=unit(np.cross(v[0],v[2]))
        moved[4:]=unit(moved[4:]+normal[None]*1e-3/6371.)
        gap=margin.prepare(moved,faces,np.ones(3,np.uint8),np.zeros(3,int))
        self.assertEqual(gap['contact_diagnostics']['partial_contact_pairs'],0)
        self.assertEqual(gap['open_edge_count'],7)

    def test_partial_contact_preserves_the_actual_uncovered_coast_interval(self):
        import continental_contacts
        v,_=patch_geometry();mid=unit(v[0]+v[2])
        vertices=np.r_[v,mid[None]]
        faces=np.array([[0,1,2],[0,4,3]])
        result=continental_contacts.free_outline(vertices,faces)
        self.assertEqual(result['diagnostics']['partial_contact_pairs'],1)
        self.assertEqual(len(result['a']),5)
        normal=unit(np.cross(v[0],v[2]))
        on_original=((np.abs(result['a']@normal)<1e-12)&(np.abs(result['b']@normal)<1e-12))
        self.assertEqual(np.count_nonzero(on_original),1)
        a,b=result['a'][on_original][0],result['b'][on_original][0]
        np.testing.assert_allclose(a,v[2],atol=1e-14)
        np.testing.assert_allclose(b,mid,atol=1e-14)

    def test_empty_ocean_has_no_margin_and_no_material(self):
        prepared=margin.prepare(np.empty((0,3)),np.empty((0,3),int),np.empty(0,np.uint8),np.empty(0,int))
        self.assertEqual(prepared['open_edge_count'],0)
        self.assertEqual(len(prepared['component_width_km']),0)

    def test_small_craton_fragment_retains_emerged_core_and_geometry(self):
        vertices=unit(np.array([[1.,-.004,-.004],[1.,.004,-.004],[1.,0.,.004]]))
        faces=np.array([[0,1,2]])
        old=vertices.copy()
        prepared=margin.prepare(vertices,faces,np.array([2]),np.array([0]))
        center=unit(vertices.sum(axis=0))[None]
        value,fraction=margin.apply(prepared,center,np.array([0]),np.array([800.]),np.array([-5000.]))
        self.assertEqual(value[0],800.)
        self.assertEqual(fraction[0],1.)
        self.assertLess(prepared['component_width_km'][0],15.)
        np.testing.assert_array_equal(vertices,old)

    def test_profile_width_and_heights_do_not_jump_at_conforming_refinement(self):
        import adaptive_material
        vertices=unit(np.array([[1.,-.012,-.012],[1.,.012,-.012],[1.,0.,.012]]))
        faces=np.array([[0,1,2]])
        first=margin.prepare(vertices,faces,np.array([1]),np.array([0]))
        refined=adaptive_material.refine(vertices,faces,np.ones(1,int),np.ones(1,np.uint8),
            desired_edge_km=10.,max_level=1,max_faces=100)
        second=margin.prepare(refined['vertices'],refined['faces'],np.ones(len(refined['faces']),np.uint8),
                              np.zeros(len(refined['faces']),int))
        np.testing.assert_allclose(first['component_width_km'],second['component_width_km'],rtol=2e-13)
        bary=np.random.default_rng(7).dirichlet(np.ones(3),100)
        query=unit(bary@vertices)
        child,_=geometry.locate_points(query,geometry.build_locator(refined['vertices'],refined['faces']))
        h1,_=margin.apply(first,query,np.zeros(100,int),np.full(100,900.),np.full(100,-5000.))
        h2,_=margin.apply(second,query,child,np.full(100,900.),np.full(100,-5000.))
        np.testing.assert_allclose(h1,h2,atol=2e-7)

    def test_actual_hole_remains_ocean_and_has_its_own_native_shelf(self):
        # A square ring has a genuine inner open outline, not missing samples.
        outer=np.array([[-.15,-.15],[.15,-.15],[.15,.15],[-.15,.15]])
        inner=outer*.2
        v=unit(np.column_stack((np.ones(8),np.r_[outer,inner])))
        f=[]
        for i in range(4):
            j=(i+1)%4
            f.extend(((i,j,4+j),(i,4+j,4+i)))
        f=np.asarray(f)
        source=source_frame()
        source.update(material_vertices=v,material_faces=f,material_face_id=np.arange(8),
            material_owner=np.ones(8,int),material_kind=np.ones(8,np.uint8),
            material_height_m=np.full(8,1000.),material_erosion_rate_m_myr=np.zeros(8),
            material_arc_id=np.zeros(8,int),material_arc_basal_m=np.zeros(8))
        new=margin.upgrade_frame(source); context=sampling.prepare(new)
        self.assertEqual(context['continental_margin']['open_edge_count'],8)
        result=sampling.sample_frame(new,np.array([[1.,0.,0.]]),context)
        self.assertEqual(result['material_face'][0],-1)
        self.assertLess(result['elevation'][0],-2000.)
        a,b=probes(source,1e-8)
        self.assertLess(np.max(np.abs(sampling.sample_frame(new,a,context)['elevation']-
                                        sampling.sample_frame(new,b,context)['elevation'])),1e-3)

    def test_overlap_is_envelope_not_new_trough_and_is_pole_invariant(self):
        source=source_frame(); v=source['material_vertices'].copy(); f=source['material_faces'].copy()
        angle=.13; rotation=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
        source['material_vertices']=np.r_[v,v@rotation.T]
        source['material_faces']=np.r_[f,f+len(v)]
        for key in ('material_owner','material_kind','material_height_m','material_erosion_rate_m_myr','material_arc_id','material_arc_basal_m'):
            source[key]=np.r_[source[key],source[key]]
        source['material_face_id']=np.arange(4)
        new=margin.upgrade_frame(source); a,b=probes(source,1e-8)
        context=sampling.prepare(new)
        heights=sampling.sample_frame(new,np.r_[a,b],context)['elevation']
        self.assertLess(np.max(np.abs(heights[:len(a)]-heights[len(a):])),.02)
        pose=dict(yaw=175.,pitch=89.,roll=40.)
        moved=orient_frame(new,pose)
        actual=sampling.sample_frame(moved,np.r_[a,b]@rotation_matrix(pose))['elevation']
        np.testing.assert_allclose(actual,heights,atol=1e-6)

    def test_finer_initial_coast_samples_source_not_old_straight_edge(self):
        source=Artwork(cap,width=512); mesh=geometry.icosphere(2)
        axis=unit(np.array([.43,-.27,.8])); errors=[]; counts=[]
        for levels in (1,3):
            fitted=build_initial_material(source,mesh,boundary_refinement_levels=levels)
            native=geometry.geometry(fitted['vertices'],fitted['faces'])
            mask=fitted['face_kind'][native['edge_faces'][:,0]]!=fitted['face_kind'][native['edge_faces'][:,1]]
            a,b=native['vertices'][native['edge_vertices'][mask]].transpose(1,0,2)
            mid=unit(a+b)
            errors.append(np.sqrt(np.mean((np.arcsin(mid@axis)-np.arcsin(.28))**2)))
            counts.append(len(fitted['faces']))
            self.assertLess(abs(fitted['diagnostics']['global_area_error_fraction']),1e-12)
        self.assertLess(errors[1],errors[0]*.5)
        self.assertLess(counts[1],len(mesh['faces'])*4**3)


if __name__=='__main__': unittest.main()
