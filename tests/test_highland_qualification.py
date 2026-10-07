"""Qualification policy stays separate from geological interpretation."""
import unittest
import numpy as np

import highland_qualification as q


class HighlandQualificationTests(unittest.TestCase):
    def sample(self,**changes):
        row=dict(time_myr=500.,continental_fraction=.65,emergent_land_fraction=.6,
            ocean_fraction=.35,continent_count=6,largest_continent_fraction=.2,
            ocean_basin_count=1,largest_ocean_basin_fraction=.35,active_plates=8,
            mixed_plates=4,passive_margin_length_km=10000.,active_coast_length_km=8000.,
            boundary_length_km=dict(ridge=10000.,subduction=8000.,transform=4000.,collision=2000.,rift=1000.),
            total_boundary_mechanism_length_km=25000.,mean_plate_speed_cm_yr=2.,
            max_plate_speed_cm_yr=5.,stalled_trench_length_km=0.,
            rift_system_phases={'active':2,'broken_through':1},rift_breakthrough_count=1,
            trench_system_phases={'mature':3},breakoff_count=1,ocean_created_km2=1e6,
            ocean_consumed_km2=8e5,accreted_km2=2e5,
            slab_ledger=dict(area_relative_residual=1e-12,mass_relative_residual=2e-12))
        row.update(changes)
        return row

    def test_component_areas_use_real_adjacency_and_area(self):
        mask=np.array([True,True,False,True])
        edges=np.array([[0,1],[1,2],[2,3]],np.int32)
        result=q.component_areas(mask,edges,np.array([2.,3.,4.,5.]))
        np.testing.assert_array_equal(result,[5.,5.])

    def test_clean_run_passes_without_worldbuilding_advisories(self):
        first=self.sample(time_myr=0.)
        last=self.sample(time_myr=500.)
        result=q.evaluate([first,last],requested_duration_myr=500.)
        self.assertTrue(result['passed'])
        self.assertEqual(result['failures'],[])
        self.assertEqual(result['advisories'],[])

    def test_hard_gates_are_numerical_and_topological(self):
        bad=self.sample(time_myr=400.,ocean_fraction=.01,active_plates=1,
            slab_ledger=dict(area_relative_residual=1e-4,mass_relative_residual=0.))
        result=q.evaluate([bad],requested_duration_myr=500.)
        self.assertFalse(result['passed'])
        self.assertIn('ocean_floor_lost',result['failures'])
        self.assertIn('plate_system_collapsed',result['failures'])
        self.assertIn('slab_inventory_failed',result['failures'])
        self.assertIn('terminal_time_short',result['failures'])

    def test_worldbuilding_concerns_are_advisories_not_fake_calibration_failures(self):
        first=self.sample(time_myr=450.,largest_continent_fraction=.6,
            rift_breakthrough_count=0,ocean_created_km2=0.,max_plate_speed_cm_yr=.01,
            stalled_trench_length_km=100.,boundary_length_km=dict(
                ridge=0.,subduction=10.,transform=0.,collision=0.,rift=0.))
        last=dict(first,time_myr=500.)
        result=q.evaluate([first,last],requested_duration_myr=500.)
        self.assertTrue(result['passed'])
        for flag in ('supercontinent_dominance','no_continental_breakthrough_observed',
                     'no_new_ocean_observed','recent_near_stagnation','large_stalled_trench_share'):
            self.assertIn(flag,result['advisories'])

    def test_qualification_config_uses_reviewed_highland_initial_metadata(self):
        config=q.qualification_config(41,duration_myr=20.,sample_myr=5.,
            width=64,mesh_level=3,coast_geometry_level=3,mechanics_nodes=256)
        self.assertEqual(config['physics_profile'],'reviewed_v1')
        self.assertNotIn('rift_traction',config)
        self.assertNotIn('continental_lifecycle',config)
        self.assertNotIn('primordial_subduction',config)
        self.assertEqual(config['height'],32)


if __name__=='__main__':
    unittest.main()
