"""Lite basin loading uses solved motion and declarations, without slab state."""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import backarc
import effective_subduction
import native_engine
import physics_profile
from tests.test_backarc import fixture, sea_anchor_fixture, attached


def effective_fixture(upper_km_myr=3., convergence=50.):
    s = sea_anchor_fixture(upper_km_myr, convergence=convergence)
    s.backarc_driver_version = backarc.EFFECTIVE_DRIVER_VERSION
    s.subduction_response_version = 0
    s.effective_subduction_version = effective_subduction.VERSION
    s.slab_memory_version = 0
    s.normal_partition_version = 1
    s.config['effective_subduction'] = dict(enabled=True, force_n_per_m=5e12)
    s.effective_subduction_settings = effective_subduction.normalize(s.config['effective_subduction'])
    s.trench_id = np.array([7])
    s.trench_maturity = np.ones(1)
    s.trench_systems = [dict(id=7, phase='mature', maturity=1.,
        downgoing_plate_uid=30, overriding_plate_uid=20,
        effective_subduction=dict(version=1, force_n_per_m=5e12))]
    s.trench_retreat_speed[:] = 0.
    relative = np.cross(s.omega[s.bq]-s.omega[s.bp], s.bmid)*backarc.RADIUS_KM
    s.normal_speed = np.sum(relative*s.bn, axis=1)
    return s


class EffectiveBackarcLoadingTests(unittest.TestCase):
    def test_away_motion_loads_at_solved_speed_without_velocity_or_retreat_mutation(self):
        for speed in (3., 12.):
            with self.subTest(speed=speed):
                s = effective_fixture(speed)
                before = s.omega.copy()
                np.testing.assert_allclose(backarc.loading_speed(s), [speed], rtol=0., atol=1e-14)
                self.assertIsNone(backarc.update(s, 2.))
                row = s.backarc_basins[0]
                self.assertEqual(row['phase'], 'loading')
                self.assertAlmostEqual(row['loading_rate_km_myr'], speed)
                expected = speed*backarc.LOADING_MEMORY_MYR*(1-math.exp(-2./backarc.LOADING_MEMORY_MYR))
                self.assertAlmostEqual(row['loading_km'], expected)
                self.assertEqual(row['opening_km'], 0.)
                self.assertEqual(row['extension_rate_km_myr'], 0.)
                np.testing.assert_array_equal(s.omega, before)
                np.testing.assert_array_equal(s.trench_retreat_speed, 0.)
                self.assertIn('declared effective reservoir', s.recorded[0][0][1])
                self.assertEqual(s.recorded[0][1]['details']['mechanism'], backarc._MECHANISM[2])
                report = s.backarc_driver_diagnostics
                self.assertEqual(report['force_bearing_edges'], 1)
                self.assertEqual(report['force_bearing_length_km'], 1000.)
                self.assertEqual(report['loading_length_km'], 1000.)
                self.assertAlmostEqual(report['mean_speed_km_myr'], speed)

    def test_toward_parallel_and_stationary_upper_plates_do_not_load(self):
        for name, omega in (('toward', [0., 0., -3./backarc.RADIUS_KM]),
                            ('parallel', [0., -3./backarc.RADIUS_KM, 0.]),
                            ('stationary', [0., 0., 0.])):
            with self.subTest(name=name):
                s = effective_fixture()
                s.omega[1] = omega
                np.testing.assert_array_equal(backarc.loading_speed(s), 0.)
                self.assertIsNone(backarc.update(s, 2.))
                self.assertEqual(s.backarc_basins, [])

    def test_effective_driver_never_applies_the_legacy_retreat_correction(self):
        s = effective_fixture()
        s.backarc_basins = deepcopy(fixture().backarc_basins)
        s.trench_retreat_speed[:] = 8.
        before = s.omega.copy()
        backarc.apply_motion(s, 2.)
        np.testing.assert_array_equal(s.omega, before)

    def test_reversed_storage_has_identical_loading_and_diagnostics(self):
        original = effective_fixture()
        reversed_state = deepcopy(original)
        reversed_state.ba, reversed_state.bb = original.bb.copy(), original.ba.copy()
        reversed_state.bp, reversed_state.bq = original.bq.copy(), original.bp.copy()
        reversed_state.bn = -original.bn
        np.testing.assert_array_equal(backarc.loading_speed(reversed_state), backarc.loading_speed(original))
        self.assertEqual(reversed_state.backarc_driver_diagnostics, original.backarc_driver_diagnostics)
        backarc.update(original, 2.)
        backarc.update(reversed_state, 2.)
        self.assertEqual(reversed_state.backarc_basins, original.backarc_basins)

    def test_dry_unmatched_lost_and_nonconverging_declarations_do_not_load(self):
        for change in ('dry', 'unmatched', 'shutdown', 'owner_lost', 'opening',
                       'stationary_pair', 'stale_convergence', 'stale_control_owner'):
            with self.subTest(change=change):
                s = effective_fixture()
                if change == 'dry': s.crust[3] = 1
                elif change == 'unmatched': s.trench_id[:] = 99
                elif change == 'shutdown': s.trench_systems[0]['phase'] = 'shutdown'
                elif change == 'owner_lost': s.active[2] = False
                elif change == 'opening':
                    s.omega[2, 2] = 0.
                    s.normal_speed[:] = 3.
                elif change == 'stationary_pair':
                    s.omega[2] = s.omega[1]
                    s.normal_speed[:] = 0.
                elif change == 'stale_convergence':
                    s.omega[2] = s.omega[1]
                    s.normal_speed[:] = -50.
                else: s.plate[3] = 0
                np.testing.assert_array_equal(backarc.loading_speed(s), 0.)
                self.assertIsNone(backarc.update(s, 2.))
                self.assertEqual(s.backarc_basins, [])
                self.assertEqual(s.backarc_driver_diagnostics['force_bearing_edges'], 0)

    def test_no_detailed_inventory_depth_dip_age_or_force_gain_is_read(self):
        s = effective_fixture()
        with patch.object(backarc.slab_memory, 'line_load', side_effect=AssertionError('Detailed inventory read.')), \
             patch.object(backarc, 'overriding_thickness_km', side_effect=AssertionError('Detailed thickness read.')):
            expected = backarc.loading_speed(s)
            expected_report = deepcopy(s.backarc_driver_diagnostics)
            for dip in (1., 89.):
                with self.subTest(dip=dip), patch.object(backarc.slab_memory, 'SUBDUCTION_DIP_DEG', dip):
                    s.slab_km = float('nan')
                    s.age[:] = float('nan')
                    s.trench_systems[0].update(slab_retained_excess_mass_kg=1e30,
                        slab_length_km=1e6, neck_fraction=0.)
                    s.trench_systems[0]['effective_subduction']['force_n_per_m'] = 5e6
                    np.testing.assert_array_equal(backarc.loading_speed(s), expected)
                    self.assertEqual(s.backarc_driver_diagnostics, expected_report)

    def test_existing_admission_memory_and_rupture_gates_are_reused(self):
        s = effective_fixture()
        pending = None
        while pending is None and s.t < 40.:
            pending = backarc.update(s, 2.)
            s.t += 2.
        self.assertIsNotNone(pending)
        row = s.backarc_basins[0]
        self.assertGreaterEqual(row['loading_km'], backarc.RUPTURE_LOADING_KM)
        crossing = -backarc.LOADING_MEMORY_MYR*math.log(
            1-backarc.RUPTURE_LOADING_KM/(3.*backarc.LOADING_MEMORY_MYR))
        self.assertAlmostEqual(s.t-2.-row['started_myr'], 2.*math.ceil(crossing/2.)-2.)
        for change in ('short_sector', 'slow', 'rift_disabled'):
            other = effective_fixture()
            if change == 'short_sector': other.bl[:] = 799.
            elif change == 'slow': other.omega[1, 2] = .39/backarc.RADIUS_KM
            else: other.config['rift_strength'] = 0.
            self.assertIsNone(backarc.update(other, 2.))
            self.assertEqual(other.backarc_basins, [])

    def test_force_loss_pauses_existing_loading_without_claiming_consumption(self):
        s = effective_fixture()
        backarc.update(s, 2.)
        row = s.backarc_basins[0]
        accumulated = row['loading_km']
        s.trench_id[:] = 99
        s.t = 14.
        backarc.update(s, 2.)
        self.assertEqual(row['phase'], 'quiet')
        self.assertEqual(row['loading_rate_km_myr'], 0.)
        self.assertAlmostEqual(row['loading_km'], accumulated*math.exp(-2./backarc.LOADING_MEMORY_MYR))
        self.assertEqual(row['opening_km'], 0.)
        self.assertIn('Effective-reservoir loading', s.recorded[-1][0][1])


class EffectiveBackarcVersionTests(unittest.TestCase):
    def frame(self):
        s = effective_fixture()
        fields = backarc.snapshot_fields(s)
        return dict(fields, effective_subduction_version=1, subduction_response_version=0)

    def test_frame_law_and_diagnostics_are_strict_and_snapshot_is_detached(self):
        frame = self.frame()
        self.assertEqual(frame['backarc_driver_version'], 2)
        backarc.validate_frame(frame)
        for change in ('effective_missing', 'effective_bool', 'response', 'response_bool', 'missing', 'law', 'version',
                       'negative', 'nan', 'bool', 'length', 'mean', 'float_count'):
            with self.subTest(change=change):
                other = deepcopy(frame)
                row = other['backarc_driver_diagnostics']
                if change == 'effective_missing': del other['effective_subduction_version']
                elif change == 'effective_bool': other['effective_subduction_version'] = True
                elif change == 'response': other['subduction_response_version'] = 1
                elif change == 'response_bool': other['subduction_response_version'] = False
                elif change == 'missing': del other['backarc_driver_diagnostics']
                elif change == 'law': row['law'] = backarc.DRIVER_LAW
                elif change == 'version': row['version'] = 1
                elif change == 'negative': row['max_speed_km_myr'] = -1.
                elif change == 'nan': row['mean_speed_km_myr'] = float('nan')
                elif change == 'bool': other['backarc_driver_version'] = True
                elif change == 'length': row['loading_length_km'] = 1001.
                elif change == 'mean': row['mean_speed_km_myr'] = 4.
                else: row['force_bearing_edges'] = 1.
                with self.assertRaises(ValueError): backarc.validate_frame(other)
        s = effective_fixture()
        fields = backarc.snapshot_fields(s)
        fields['backarc_driver_diagnostics']['max_speed_km_myr'] = 99.
        self.assertEqual(s.backarc_driver_diagnostics['max_speed_km_myr'], 3.)

    def test_old_law_snapshot_fields_and_loading_are_exact(self):
        legacy = fixture()
        np.testing.assert_array_equal(backarc.loading_speed(legacy), [8.])
        self.assertEqual(backarc.snapshot_fields(legacy), {})
        sea = sea_anchor_fixture(3.)
        with patch.object(backarc.slab_memory, 'line_load', attached):
            fields = backarc.snapshot_fields(sea)
        expected = dict(backarc_driver_version=1, backarc_driver_diagnostics=dict(version=1,
            anchored_edges=1, anchored_length_km=1000., loading_length_km=1000.,
            max_speed_km_myr=3., mean_speed_km_myr=3., frame=backarc.DRIVER_FRAME,
            law=backarc.DRIVER_LAW))
        self.assertEqual(fields, expected)
        backarc.validate_frame(dict(fields, subduction_response_version=1))

    def test_effective_driver_rejects_missing_effective_or_moving_hinge_state(self):
        for change in ('no_effective', 'moving_hinge', 'bad_version'):
            s = effective_fixture()
            if change == 'no_effective': del s.effective_subduction_version
            elif change == 'moving_hinge': s.subduction_response_version = 1
            else: s.backarc_driver_version = 3
            with self.subTest(change=change), self.assertRaises(ValueError):
                backarc.loading_speed(s)

    def test_saved_reviewed_profile_does_not_reselect_its_driver(self):
        for version in (None, 0, 1):
            s = SimpleNamespace(config={'physics_profile': 'reviewed_v1',
                'effective_subduction': {'enabled': True}}, physics_profile_version=1)
            if version is not None: s.backarc_driver_version = version
            physics_profile.initialize(s)
            self.assertEqual(backarc.driver_version(s), 0 if version is None else version)

    def test_fresh_effective_profile_records_driver_two_and_valid_frame(self):
        s = native_engine.Simulation(dict(width=48, height=24, mesh_level=2,
            coast_geometry_level=2, plate_count=4, mechanics_nodes=128, seed=37,
            physics_profile='reviewed_v1', primordial_subduction={'enabled': True},
            effective_subduction={'enabled': True}))
        self.assertEqual(s.backarc_driver_version, 2)
        self.assertEqual(s.physics_profile_diagnostics['backarc_driver_version'], 2)
        self.assertEqual(s.subduction_response_version, 0)
        np.testing.assert_array_equal(s.trench_retreat_speed, 0.)
        self.assertFalse(any(key.startswith('slab_') for row in s.trench_systems for key in row))
        frame = s.snapshot()
        self.assertEqual(frame['backarc_driver_version'], 2)
        self.assertEqual(frame['backarc_driver_diagnostics']['law'], backarc.EFFECTIVE_DRIVER_LAW)
        backarc.validate_frame(frame)


if __name__ == '__main__':
    unittest.main()
