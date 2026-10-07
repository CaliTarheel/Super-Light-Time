"""Declared ring trench around one supercontinent carrier plate."""
from copy import deepcopy
from pathlib import Path
import math
import tempfile
import unittest

import numpy as np

import checkpoint
import force_rifting
import native_engine
import subduction_audit
import supercontinent_ring as ring

CONFIG = dict(width=48, height=24, mesh_level=4, coast_geometry_level=2, plate_count=4,
              mechanics_nodes=128, seed=37, physics_profile='reviewed_v1',
              primordial_subduction={'enabled': True}, supercontinent_ring={'enabled': True})
OVERLOADED = dict(CONFIG, primordial_subduction={'enabled': True, 'initial_slab_depth_km': 200.},
                  trench_persistence='attached_slab_v1',
                  force_limit_rifting={'enabled': True, 'rift_width_km': 10., 'breakup_stretch': 2.,
                                       'ocean_breakup_opening_km': 10.})


class RingConfigurationTests(unittest.TestCase):
    def test_defaults_ranges_and_pole(self):
        self.assertEqual(ring.normalize(None), ring.DEFAULTS)
        self.assertFalse(ring.normalize(None)['enabled'])
        value = ring.normalize(dict(enabled=True, pole_lat_lon_deg=[10, 20], ring_radius_deg=100))
        self.assertEqual(value['pole_lat_lon_deg'], [10., 20.])
        self.assertEqual(value['ring_radius_deg'], 100.)
        for bad in (dict(enabled=1), dict(ring_radius_deg=10.), dict(ring_radius_deg=True),
                    dict(pole_lat_lon_deg=[95, 0]), dict(pole_lat_lon_deg=[1]),
                    dict(far_ocean_age_myr=float('nan')), dict(unknown=1), []):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ring.normalize(bad)

    def test_engine_requires_reviewed_profile_slabs_and_excludes_primordial_ocean(self):
        with self.assertRaisesRegex(ValueError, 'primordial_subduction'):
            native_engine.validate_config(dict(CONFIG, primordial_subduction={'enabled': False}))
        with self.assertRaisesRegex(ValueError, 'reviewed_v1'):
            native_engine.validate_config(dict(CONFIG, physics_profile='legacy'))
        with self.assertRaisesRegex(ValueError, 'alternative'):
            native_engine.validate_config(dict(CONFIG, primordial_ocean={'enabled': True}))


class RingWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = native_engine.Simulation(dict(CONFIG, supercontinent_ring=None,
                                                     primordial_subduction={'enabled': True}))
        cls.world = native_engine.Simulation(CONFIG)

    def slots(self, s):
        carrier = int(np.flatnonzero(s.active & (s.plate_uid == s.supercontinent_ring_carrier_uid))[0])
        over = int(np.flatnonzero(s.active & (s.plate_uid == s.supercontinent_ring_overriding_uid))[0])
        return carrier, over

    def test_one_carrier_owns_all_land_and_the_skirt(self):
        s = self.world
        carrier, over = self.slots(s)
        self.assertEqual(sorted(np.flatnonzero(s.active)), sorted((carrier, over)))
        self.assertTrue(np.all(s.parcel_plate == carrier))
        self.assertEqual(s.supercontinent_ring_diagnostics['merged_continental_plate_uids'], [3, 4])
        # Land geometry and material inventory are exactly those of the ordinary import.
        np.testing.assert_array_equal(s.kind, self.baseline.kind)
        np.testing.assert_allclose(s.mass, self.baseline.mass, rtol=0, atol=0)
        pole = np.asarray(s.supercontinent_ring_diagnostics['pole_xyz'])
        inside = s.xyz@pole > 0.
        # Control cells follow the circle up to one cell of staircase.
        agree = np.mean((s.plate == carrier) == inside)
        self.assertGreater(agree, .97)
        self.assertGreater(s.supercontinent_ring_diagnostics['minimum_skirt_km'], 300.)

    def test_every_ring_contact_subducts_the_carrier_skirt(self):
        s = self.world
        carrier, over = self.slots(s)
        rows = [r for r in s.trench_systems if r.get('initial_subduction')]
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row['downgoing_plate_uid'], s.supercontinent_ring_carrier_uid)
            self.assertEqual(row['overriding_plate_uid'], s.supercontinent_ring_overriding_uid)
            self.assertGreater(row['slab_retained_excess_mass_kg'], 0.)
        self.assertEqual(int(s.polarity[carrier, over]), carrier)
        diagnostics = s.primordial_subduction_diagnostics
        self.assertEqual(diagnostics['initial_subducting_ocean_plate_uids'], [s.supercontinent_ring_carrier_uid])
        # The ring is (up to its staircase) a great circle.
        self.assertGreater(diagnostics['initial_trench_length_km'], 2*math.pi*6371.)

    def test_ring_pull_largely_cancels_in_the_rigid_balance(self):
        s = self.world
        # A uniform ring exerts no net torque, so solved speeds are tiny
        # compared with the ordinary coastal-subduction startup.
        self.assertLess(s.physics_profile_diagnostics['initial_max_speed_cm_yr'], .05)
        self.assertLess(s.physics_profile_diagnostics['initial_max_speed_cm_yr'],
                        self.baseline.physics_profile_diagnostics['initial_max_speed_cm_yr'])

    def test_cooling_ages_follow_declaration(self):
        s = self.world
        carrier, over = self.slots(s)
        wet = s.crust == 0
        far = wet & (s.plate == over)
        np.testing.assert_allclose(s.age[far & (s.support[over] > 1-1e-9)], 100.)
        skirt = s.age[wet & (s.plate == carrier)]
        self.assertGreaterEqual(skirt.min(), 100.-1e-9)
        self.assertLessEqual(skirt.max(), 150.+1e-9)

    def test_default_ring_tears_the_continent_before_the_ocean(self):
        import plate_limit_analysis
        s = self.world
        carrier, _ = self.slots(s)
        report = plate_limit_analysis.plate_mechanisms(s, carrier)
        self.assertTrue(report['continent_first'])
        worst, ocean = report['any'], report['ocean_only']
        self.assertGreater(worst['cut_continental_fraction'], .8)
        self.assertEqual(ocean['cut_continental_fraction'], 0.)
        self.assertGreater(worst['loading_ratio'], 3*ocean['loading_ratio'])
        self.assertLess(ocean['loading_ratio'], 1.)
        self.assertLess(report['loads']['unresolved_fraction_of_gross'], .01)
        if worst['fails']:
            self.assertGreater(worst['predicted_opening_cm_yr_mean'], 0.)

    def test_small_ring_that_cuts_the_continent_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'reaches within'):
            native_engine.Simulation(dict(CONFIG, supercontinent_ring=dict(enabled=True, ring_radius_deg=60.)))

    def test_checkpoint_round_trip_and_short_step(self):
        s = deepcopy(self.world)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'ring.npz'
            checkpoint.write_checkpoint(path, s, {'config': s.config}, {})
            restored, _ = checkpoint.read_checkpoint(path, {}, native_engine.Simulation)
        self.assertEqual(restored.supercontinent_ring_diagnostics, s.supercontinent_ring_diagnostics)
        self.assertEqual(ring.snapshot(restored)['supercontinent_ring_version'], ring.VERSION)
        s.step(.02)
        self.assertAlmostEqual(s.t, .02)


class RingAuditTests(unittest.TestCase):
    def test_initial_ring_is_loaded_stationary_and_cancelling(self):
        s = native_engine.Simulation(dict(OVERLOADED, force_limit_rifting=None))
        report = subduction_audit.audit(s)
        summary = report['summary']
        self.assertAlmostEqual(summary['loaded_stationary_fraction'], 1.)
        self.assertAlmostEqual(summary['pull_stationary_fraction'], 1.)
        self.assertGreater(summary['total_slab_pull_n'], 0.)
        self.assertEqual(summary['polarity_mismatch_km'], 0.)
        self.assertEqual(summary['complex_plates'], ['Supercontinent'])
        plate, = report['plates']
        self.assertGreater(plate['cancellation'], .99)
        # The ring is a great circle: the smooth length is close to 2 pi R.
        self.assertAlmostEqual(plate['smooth_length_km'], 2*np.pi*6371., delta=.05*2*np.pi*6371.)
        self.assertGreater(plate['trace_inflation'], 1.1)


class RingForceRiftingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = native_engine.Simulation(OVERLOADED)
        cls.stepped = deepcopy(cls.world)
        cls.mass = float(cls.stepped.mass.sum())
        cls.stepped.step(.5)

    def test_overloaded_ring_tears_the_continent_and_conserves_material(self):
        s = self.stepped
        rifts = [e for e in s.events if e['type'] == 'rift']
        self.assertEqual(len(rifts), 1)
        details = rifts[0]['details']
        self.assertEqual(details['setting'], 'continental')
        self.assertEqual(details['loading']['cause'], 'force_limit')
        self.assertGreaterEqual(details['loading']['loading_ratio'], 1.)
        self.assertEqual(s.force_rifting_diagnostics['commits'], 1)
        self.assertEqual(int(s.active.sum()), 3)
        self.assertAlmostEqual(float(s.mass.sum()), self.mass, delta=1e-6*self.mass)
        # Every craton group stays on one plate; both daughters carry continent.
        for group in np.unique(s.parcel_craton[s.parcel_craton >= 0]):
            self.assertEqual(len(np.unique(s.parcel_plate[s.parcel_craton == group])), 1)
        owners = np.unique(s.parcel_plate)
        self.assertEqual(len(owners), 2)


if __name__ == '__main__':
    unittest.main()
