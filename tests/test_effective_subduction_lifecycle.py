"""Declared trench persistence, ocean-only intake, and native restart agreement."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import checkpoint
import effective_subduction as effective
import native_engine
import primordial_subduction
import native_subduction
import slab_memory
import trench_history as history
from tests.test_trench_history import fixture


def declared_fixture():
    s = fixture()
    history.initialize(s)
    s.config = dict(effective_subduction=dict(enabled=True, force_n_per_m=5e12))
    s.effective_subduction_version = 1
    s.effective_subduction_settings = effective.normalize(s.config['effective_subduction'])
    s.effective_subduction_diagnostics = {}
    s.slab_memory_version = 0
    s.subduction_response_version = 0
    s.subduction_polarity_version = 1
    s.t, s.steps = 1., 1  # synthetic already-declared trace, beyond startup selection
    for row in s.trench_systems:
        row.update(phase='mature', maturity=1., initial_maturity=1.,
                   effective_subduction=dict(version=1, force_n_per_m=5e12))
    effective.prepare(s)
    return s


class EffectiveSubductionLifecycleTests(unittest.TestCase):
    def test_explicit_configuration_and_incompatible_detailed_laws(self):
        self.assertFalse(native_engine.validate_config()['effective_subduction']['enabled'])
        for value in (True, {'enabled': 1}, {'force_n_per_m': 0.}, {'force_n_per_m': -1.},
                      {'force_n_per_m': float('nan')}, {'force_n_per_m': float('inf')},
                      {'force_n_per_m': True}, {'unknown': 2}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                effective.normalize(value)
        base = dict(physics_profile='reviewed_v1', primordial_subduction={'enabled': True},
                    effective_subduction={'enabled': True})
        for override in ({'physics_profile': 'legacy'}, {'primordial_subduction': None},
                         {'subduction_response': 'moving_hinge_v1'},
                         {'continental_lifecycle': {'enabled': True}},
                         {'rift_traction': {'enabled': True}},
                         {'retained_phases': 'thermal_v1'},
                         {'trench_persistence': 'attached_slab_v1'}, {'slab_allocation': 'fed_v1'}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                native_engine.validate_config(dict(base, **override))
        self.assertTrue(native_engine.validate_config(base)['effective_subduction']['enabled'])

    def test_stall_and_opening_preserve_declared_force_without_intake_or_arc_growth(self):
        s = declared_fixture()
        initial_geometry = deepcopy(s.trench_systems[0]['geometry_xyz'])
        for velocity in (0., 10.):
            s.normal_speed[:] = velocity
            s.bcode[:] = 1 if velocity else 3
            for _ in range(8):
                s.t += 10.
                history.update(s, 10.)
                owners, force, trace = effective.line_state(s)
                np.testing.assert_array_equal(owners[trace], 0)
                np.testing.assert_array_equal(force[trace], 5e12)
                np.testing.assert_array_equal(s.trench_maturity[trace], 1.)
                self.assertEqual(s.trench_systems[0]['phase'], 'mature')
                self.assertEqual(s.trench_systems[0]['geometry_xyz'], initial_geometry)
        self.assertEqual(len(s.trench_systems), 1)

    def test_new_convergent_color_cannot_start_force_or_intake(self):
        s = declared_fixture()
        s.trench_systems.clear()
        s.normal_speed[:] = -20.
        s.bcode[:] = 2
        s.down[:] = 1
        s.t += 1.
        history.update(s, 1.)
        owners, force, trace = effective.line_state(s)
        self.assertFalse(s.trench_systems)
        np.testing.assert_array_equal(owners, -1)
        np.testing.assert_array_equal(force, 0.)
        self.assertFalse(trace.any())
        np.testing.assert_array_equal(s.trench_maturity, 0.)
        np.testing.assert_array_equal(s.down, -1)

    def test_local_buoyant_arrival_arrests_drive_without_reversing_incoming_owner(self):
        s = declared_fixture()
        water = np.ones(len(s.ba))
        water[0], water[1] = 0., .25
        with patch('native_subduction.enabled', return_value=True), patch(
                'native_subduction.edge_ocean_fraction', return_value=water):
            history.prepare(s)
            owners, force, trace = effective.line_state(s)
        self.assertTrue(trace.all())
        np.testing.assert_array_equal(owners, 0)
        self.assertEqual(force[0], 0.)
        self.assertEqual(force[1], 1.25e12)
        self.assertEqual(s.down[0], -1)
        self.assertEqual(s.bcode[0], 4)
        np.testing.assert_array_equal(s.down[1:], 0)
        np.testing.assert_array_equal(s.trench_maturity[1:], 1.)
        self.assertEqual(len(s.trench_systems), 1)
        self.assertFalse(any(e.get('reason') == 'polarity_reversal' for e in s.events))

    def test_owner_loss_terminates_source_and_runtime_entry_cannot_be_silently_ignored(self):
        s = declared_fixture()
        s.active[0] = False
        s.t += 1.
        history.update(s, 1.)
        self.assertEqual(s.trench_systems[0]['phase'], 'shutdown')
        self.assertEqual(s.trench_systems[0]['episodes'][-1]['shutdown_reason'], 'plate_owner_lost')
        self.assertFalse(effective.line_state(s)[2].any())
        s.continental_entry_regions = []
        with self.assertRaisesRegex(ValueError, 'continental entry'):
            effective.validate_compatibility(s)

    def test_partial_downgoing_transfer_keeps_declared_source_on_daughter(self):
        s = declared_fixture()
        s.active = np.array([True, True, True])
        s.plate_uid = np.array([11, 22, 33])
        moved = np.array([True, True, False, False, False])
        s.plate[s.ba[moved]] = 2
        mapping = history.transfer_downgoing(s, 0, 2)
        self.assertEqual(len(s.trench_systems), 2)
        child = s.trench_systems[1]
        self.assertEqual(mapping[s.trench_systems[0]['id']], child['id'])
        self.assertEqual(child['downgoing_plate_uid'], 33)
        self.assertEqual(child['effective_subduction']['force_n_per_m'], 5e12)
        s.bp[moved] = 2
        history.prepare(s)
        owners, force, trace = effective.line_state(s)
        np.testing.assert_array_equal(owners[moved], 2)
        np.testing.assert_array_equal(owners[~moved], 0)
        self.assertTrue(trace.all())
        np.testing.assert_array_equal(force, 5e12)


class EffectiveSubductionNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = dict(width=48, height=24, mesh_level=2, coast_geometry_level=2,
            plate_count=4, mechanics_nodes=128, seed=37, physics_profile='reviewed_v1',
            primordial_subduction={'enabled': True}, effective_subduction={'enabled': True})
        cls.seeded = native_engine.Simulation(cls.config)

    def test_finite_initial_declaration_has_no_slab_inventory_or_fabricated_time(self):
        s = self.seeded
        self.assertTrue(effective.enabled(s))
        self.assertFalse(slab_memory.enabled(s))
        self.assertFalse(hasattr(s, 'primordial_subduction_version'))
        _, selected, _ = primordial_subduction.selected_target_edges(s)
        owners, force, trace = effective.line_state(s)
        np.testing.assert_array_equal(trace, selected)
        self.assertTrue(np.all(force[selected] > 0.))
        self.assertTrue(np.all(owners[selected] >= 0))
        for row in s.trench_systems:
            self.assertFalse(any(key.startswith('slab_') for key in row))
            self.assertEqual(row['active_myr'], 0.)
            self.assertEqual(row['shortening_km'], 0.)
        self.assertEqual(s.t, 0.)
        self.assertEqual(s.steps, 0)
        self.assertEqual(s.process_totals['ocean_consumed_km2'], 0.)
        self.assertEqual(s.physics_profile_diagnostics['initial_slab_mass_kg'], 0.)
        self.assertEqual(s.snapshot()['effective_subduction_version'], 1)

    def test_stationary_declared_source_has_zero_finite_ocean_intake(self):
        s = deepcopy(self.seeded)
        s.omega[:] = 0.
        s.normal_speed[:] = 0.
        s.bcode[:] = 3
        history.prepare(s)
        owners, force, trace = effective.line_state(s)
        self.assertTrue(np.all(force[trace] > 0.))
        np.testing.assert_array_equal(s.trench_maturity[trace], 1.)
        self.assertEqual(native_subduction.capture_polygons(s, .01), [])
        removed = native_subduction.removal(s, s.support.copy(), .01)
        np.testing.assert_array_equal(removed, 0.)
        self.assertEqual(s.native_subduction_diagnostics['removed_area_km2'], 0.)

    def test_native_step_skips_detailed_slab_evolution_and_restart_reproduces_motion(self):
        s = deepcopy(self.seeded)
        with patch('slab_memory.advance', side_effect=AssertionError('Detailed slab evolution was selected.')):
            s.step(.005)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.npz'
            checkpoint.write_checkpoint(path, s, {'config': s.config}, {})
            restored, _ = checkpoint.read_checkpoint(path, {}, native_engine.Simulation)
        self.assertEqual(effective.snapshot(restored), effective.snapshot(s))
        self.assertEqual(restored.trench_systems, s.trench_systems)
        self.assertTrue(effective.initialize(restored))
        s.step(.005)
        restored.step(.005)
        for field in ('omega', 'support', 'pos', 'plate', 'trench_id'):
            np.testing.assert_array_equal(getattr(restored, field), getattr(s, field), err_msg=field)
        self.assertEqual(restored.trench_systems, s.trench_systems)
        self.assertTrue(np.all(restored.trench_retreat_speed == 0.))


if __name__ == '__main__':
    unittest.main()
