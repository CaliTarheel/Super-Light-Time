"""Versioned stack sampling, independent of support generation or plate motion."""
from copy import deepcopy
import unittest
import numpy as np

import native_frame_sampling as sampling
import mesh_geometry
from tests.test_collision_contacts import fixture,saved,unit,RADIUS
import crustal_structure


def physical_frame(*,margins=True):
    state=fixture()
    frame=saved(state,margins=margins)
    top=frame['material_owner']==0
    frame.update(collision_surface_version=1,
                 material_collision_support_m=np.where(top,650.,0.))
    frame['material_collision_load_thickness_km']=frame['material_collision_support_m']/crustal_structure.AIRY_M_PER_KM
    frame['material_height_m']=np.where(top,1000.,4300.)
    frame['collision_contacts']=[dict(id=1,top_sheet=1,under_sheet=2)]
    return frame


class NativeCollisionStackSamplingTests(unittest.TestCase):
    def test_interior_uses_selected_column_plus_explicit_support(self):
        frame=physical_frame();before=deepcopy(frame)
        result=sampling.sample_frame(frame,np.array([[1.,0.,0.]]))
        self.assertEqual(result['plate'][0],0)
        self.assertAlmostEqual(result['selected_sheet_height_m'][0],1000.)
        self.assertAlmostEqual(result['physical_stack_support_m'][0],650.)
        self.assertAlmostEqual(result['raw_elevation_m'][0],1650.)
        self.assertAlmostEqual(result['collision_surface_offset_m'][0],0.)
        for key,value in before.items():
            if isinstance(value,np.ndarray):np.testing.assert_array_equal(frame[key],value)

    def test_version_gate_keeps_previous_envelope_semantics(self):
        old=physical_frame();del old['collision_surface_version'];del old['material_collision_support_m'];del old['material_collision_load_thickness_km']
        result=sampling.sample_frame(old,np.array([[1.,0.,0.]]))
        self.assertEqual(result['plate'][0],0)
        self.assertGreater(result['elevation'][0],4000.)
        self.assertGreater(result['collision_surface_offset_m'][0],3000.)
        self.assertNotIn('raw_elevation_m',result)

    def test_components_sum_and_new_heights_are_unclipped(self):
        frame=physical_frame();frame['material_height_m'][:]=12000.
        result=sampling.sample_frame(frame,np.array([[1.,0.,0.]]))
        self.assertAlmostEqual(result['elevation'][0],12650.)
        np.testing.assert_array_equal(result['clipping_delta_m'],0.)
        np.testing.assert_array_equal(result['raw_elevation_m'],result['selected_sheet_height_m']+
                                      result['physical_stack_support_m']+result['display_thermal_support_m'])

    def test_both_contact_edges_have_convergent_one_sided_limits(self):
        for margins in (False,True):
            frame=physical_frame(margins=margins)
            for edge in (-200.,200.):
                differences=[]
                for distance in (.01,.001):
                    points=unit(np.array([[1.,(edge-distance)/RADIUS,0.],[1.,(edge+distance)/RADIUS,0.]]))
                    result=sampling.sample_frame(frame,points)
                    differences.append(float(abs(np.diff(result['elevation'])[0])))
                self.assertLess(differences[1],max(1e-7,differences[0]*.12))
                self.assertLess(differences[1],.1)

    def test_transitive_order_survives_absent_middle_sheet(self):
        frame=physical_frame()
        frame['material_collision_sheet'][frame['material_owner']==0]=3
        frame['material_collision_sheet'][frame['material_owner']==1]=1
        frame['collision_contacts']=[dict(id=1,top_sheet=3,under_sheet=2),dict(id=2,top_sheet=2,under_sheet=1)]
        result=sampling.sample_frame(frame,np.array([[1.,0.,0.]]))
        self.assertEqual(result['plate'][0],0)
        self.assertAlmostEqual(result['elevation'][0],1650.)

    def test_live_prepared_context_matches_saved_path_and_hit_permutation(self):
        frame=physical_frame();points=unit(np.array([[1.,0.,0.],[1.,.02,0.],[1.,.08,0.]]))
        context=sampling.prepare(frame)
        cells,_=mesh_geometry.locate_points(points,context['locator'])
        hits=mesh_geometry.locate_points(points,context['material_locator'],all_hits=True)
        live=sampling.select_exposed_material(context,points,cells,np.zeros(len(points),np.int32),tuple(a[::-1] for a in hits))
        restored=sampling.sample_frame(frame,points)
        np.testing.assert_array_equal(live['material_face'],restored['material_face'])
        np.testing.assert_allclose(live['raw_elevation_m'],restored['elevation'],rtol=0,atol=1e-12)

    def test_true_gap_stays_ocean(self):
        frame=saved(fixture(separated=True),margins=True)
        frame.update(collision_surface_version=1,material_collision_support_m=np.zeros(len(frame['material_faces'])))
        frame['material_collision_load_thickness_km']=np.zeros(len(frame['material_faces']))
        result=sampling.sample_frame(frame,unit([1.,275./RADIUS,0.])[None])
        self.assertEqual(result['material_face'][0],-1)
        self.assertEqual(result['physical_stack_support_m'][0],0.)

    def test_invalid_or_ungated_support_fails_loudly(self):
        for change in ('ungated','missing','negative','shape','nonfinite','unknown'):
            frame=physical_frame()
            if change=='ungated':del frame['collision_surface_version']
            elif change=='missing':del frame['material_collision_support_m']
            elif change=='negative':frame['material_collision_support_m'][0]=-1.
            elif change=='shape':frame['material_collision_support_m']=frame['material_collision_support_m'][:-1]
            elif change=='nonfinite':frame['material_collision_support_m'][0]=np.nan
            elif change=='unknown':frame['collision_surface_version']=2
            with self.subTest(change=change),self.assertRaises(ValueError):sampling.prepare(frame)


if __name__=='__main__':unittest.main()
