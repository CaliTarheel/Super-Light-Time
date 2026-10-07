"""Fixed-volume birth succeeds only where slope, column and geography permit."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
import arc_birth_footprint as footprint
import arc_birth_profile as profile
import arc_cohort_footprint as cohorts
import arc_emplacement_geometry as geometry
import native_arc_material as arcs
from test_native_processes import ocean_fixture
from test_arc_source_geometry import add, positions_in_cell

SOURCE_50 = 278.7753550551505
BASEMENT_50 = -6370.9114166


def world():
    s=ocean_fixture()
    s.native_arc_birth_profile_version=1
    s.native_arc_footprint_version=1
    return s


class ArcBirthFootprintTests(unittest.TestCase):
    def test_public_native_frame_and_typed_checkpoint_keep_physical_footprint_policy(self):
        import checkpoint
        import mesh_history
        import native_frame_sampling
        s=world();point=positions_in_cell(s)[:1];add(s,point,[SOURCE_50])
        s._rasterize();s._boundaries();s._update_surface_domains()
        frame=s.snapshot();mesh_history.arrays(frame)
        sampled=native_frame_sampling.sample_frame(frame,point)
        self.assertTrue(np.isfinite(sampled['elevation']).all())
        self.assertEqual(frame['arc_footprint_version'],1)
        compatibility=dict(engine_sha256='footprint-fixture',auxiliary_sources_sha256={},numpy_version=np.__version__)
        manifest=dict(run_id='footprint-fixture',config=s.config,frames=[dict(time_myr=0.)],frame_count=1,state='paused')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.npz';checkpoint.write_checkpoint(path,s,manifest,compatibility)
            restored,_=checkpoint.read_checkpoint(path,compatibility,type(s))
        self.assertEqual(restored.native_arc_footprint_version,1)
        np.testing.assert_array_equal(restored.material_surface['vertices'],s.material_surface['vertices'])
        np.testing.assert_array_equal(restored.structure['thickness_km'],s.structure['thickness_km'])
        self.assertEqual(restored.native_arc_source_placements,s.native_arc_source_placements)
        for state in (s,restored):add(state,point,[100.])
        np.testing.assert_array_equal(restored.structure['thickness_km'],s.structure['thickness_km'])
        np.testing.assert_array_equal(restored.native_arc_pending['area'],s.native_arc_pending['area'])
        self.assertEqual(restored.native_arc_source_placements,s.native_arc_source_placements)

    def test_saved_50_myr_source_volume_funds_a_submarine_feasible_profile(self):
        centre=np.array([1.,0.,0.]);strike=np.array([0.,1.,0.])
        result=footprint.propose(lambda area:arcs._patch(centre,strike,area),SOURCE_50,BASEMENT_50)
        selected=result['profile'];diagnostic=selected['diagnostics']
        self.assertGreater(result['search']['nominal_profile']['maximum_slope_deg'],60.)
        self.assertTrue(diagnostic['admissible'])
        self.assertLessEqual(diagnostic['maximum_slope_deg'],20.)
        self.assertGreater(diagnostic['actual_area_km2'],SOURCE_50)
        self.assertLess(diagnostic['actual_area_km2'],SOURCE_50*25./8.)
        self.assertAlmostEqual(float(selected['area_km2']@selected['thickness_km']),25.*SOURCE_50,places=8)
        self.assertTrue(np.all((selected['thickness_km']>=8.)&(selected['thickness_km']<=75.)))
        self.assertLess(selected['height_m'].max(),0.,'A successful protoarc need not be an invented emergent island.')
        np.testing.assert_array_equal(result['plan']['vertices'][0],centre)
        self.assertLessEqual(result['search']['evaluations'],footprint.BISECTION_STEPS+2)

    def test_insufficient_volume_stays_pending_without_a_thin_apron(self):
        s=world();point=positions_in_cell(s)[:1]
        vertices=s.material_surface['vertices'].copy()
        report=add(s,point,[100.])
        self.assertEqual(report['new_faces'],0)
        self.assertEqual(report['added_volume_km3'],0.)
        self.assertEqual(report['pending_magma_volume_km3'],2500.)
        self.assertEqual(report['emplacement_geometry']['geometry_evaluations'],0)
        search=report['emplacement_geometry']['sources'][0]['footprint_capacity']
        self.assertFalse(search['admissible'])
        self.assertEqual(search['maximum_footprint_area_km2'],312.5)
        np.testing.assert_array_equal(s.material_surface['vertices'],vertices)
        np.testing.assert_array_equal(s.native_arc_pending['xyz'],point)

    def test_actual_emplacement_closes_source_physical_volume_and_origin_ledgers(self):
        s=world();point=positions_in_cell(s)[:1]
        report=add(s,point,[SOURCE_50])
        self.assertEqual(report['new_faces'],24)
        self.assertEqual(report['pending_area_km2'],0.)
        self.assertAlmostEqual(report['added_area_km2'],SOURCE_50,places=9)
        self.assertGreater(report['physical_footprint_area_added_km2'],SOURCE_50)
        self.assertAlmostEqual(float(s.mass.sum()),SOURCE_50,places=8)
        actual=float(s.material_surface['area_km2']@s.structure['thickness_km'])
        self.assertAlmostEqual(actual,25.*SOURCE_50,places=7)
        self.assertAlmostEqual(float(s.mass@s.structure['added_volume_km_per_reference_km2']),actual,places=7)
        self.assertTrue(cohorts.contained(s.material_surface,point).all())
        self.assertEqual(len(s.native_arc_source_placements),1)
        self.assertAlmostEqual(s.native_arc_source_placements[0]['volume_km3'],actual,places=7)
        measured=report['emplacement_geometry']['sources'][0]['accepted_footprint']
        self.assertTrue(measured['admissible'])
        self.assertLessEqual(measured['material_obstruction_km2']+measured['foreign_owner_obstruction_km2'],measured['admission_tolerance_km2'])
        frame=dict(arcs.snapshot_fields(s),arc_material_diagnostics=report,time_myr=float(s.t))
        profile.validate_frame(frame);geometry.validate_frame(frame)

    def test_valid_ocean_source_cannot_spread_across_foreign_plate_water(self):
        s=world();context=geometry.prepare(s)
        vertices=s.native_mesh['vertices']
        context['owner']['owner_slots']=np.array([0,1])
        context['owner']['scores']=np.vstack((1.+vertices[:,1],1.-vertices[:,1]))
        point=arcs._unit([1.,2./6371.,0.])
        factory=lambda amount:footprint.propose(
            lambda physical_area:arcs._patch(point,[0.,0.,1.],physical_area),amount,BASEMENT_50)['plan']
        result=geometry.admit(context,factory,SOURCE_50,0)
        self.assertGreater(result['diagnostics']['requested_footprint']['foreign_owner_obstruction_km2'],1.)
        self.assertEqual(result['accepted_area_km2'],0.)
        self.assertEqual(result['pending_area_km2'],SOURCE_50)
        self.assertIsNone(result['plan'])

    def test_unconnected_nearby_origins_cannot_pool_volume_to_force_birth(self):
        s=world();points=positions_in_cell(s)
        report=add(s,points,[100.,100.])
        self.assertEqual(report['new_faces'],0)
        self.assertEqual(report['pending_magma_volume_km3'],5000.)
        self.assertEqual(len(s.native_arc_pending['area']),2)
        np.testing.assert_array_equal(s.native_arc_pending['xyz'],points)

    def test_new_schema_checks_footprint_volume_and_explicit_legacy_is_unchanged(self):
        s=world();point=positions_in_cell(s)[:1];report=add(s,point,[SOURCE_50])
        frame=dict(arcs.snapshot_fields(s),arc_material_diagnostics=report,time_myr=float(s.t))
        for change in (lambda f:f.pop('arc_footprint_version'),
                       lambda f:f.update(arc_footprint_version=True),
                       lambda f:f['arc_footprint_policy'].update(maximum_area_ratio=4.),
                       lambda f:f['arc_material_diagnostics']['emplacement_geometry']['sources'][0]['profile_capacity'].update(actual_volume_km3=1.),
                       lambda f:f['arc_material_diagnostics']['emplacement_geometry']['sources'][0]['profile_capacity'].update(minimum_column_km=7.9)):
            altered=deepcopy(frame);change(altered)
            with self.assertRaises(ValueError):geometry.validate_frame(altered)
        old=world();old.native_arc_footprint_version=0
        pending=add(old,positions_in_cell(old)[:1],[SOURCE_50])
        self.assertEqual(pending['new_faces'],0)
        self.assertEqual(pending['pending_magma_volume_km3'],25.*SOURCE_50)
        self.assertNotIn('arc_footprint_version',arcs.snapshot_fields(old))


if __name__=='__main__':unittest.main()
