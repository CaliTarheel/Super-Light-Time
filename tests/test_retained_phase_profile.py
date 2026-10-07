"""Fresh retained-phase selection, moving-hinge compatibility and continuation."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

import checkpoint
import dense_crust
import mesh_history
import native_engine
import phase_evolution
import physics_profile
import plate_balance
import retained_phase_profile


class RetainedPhaseConfigurationTests(unittest.TestCase):
    def test_selection_is_strict_explicit_and_requires_reviewed_physics(self):
        self.assertEqual(native_engine.validate_config()['retained_phases'], 'disabled')
        for value in (None, False, True, 1, {}, 'thermal_v2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                native_engine.validate_config(dict(retained_phases=value))
        with self.assertRaisesRegex(ValueError, 'reviewed_v1'):
            native_engine.validate_config(dict(retained_phases='thermal_v1'))
        self.assertEqual(native_engine.validate_config(dict(physics_profile='reviewed_v1',
            retained_phases='thermal_v1'))['retained_phases'], 'thermal_v1')


class RetainedPhaseProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = dict(width=48, height=24, mesh_level=2, coast_geometry_level=2,
            plate_count=4, mechanics_nodes=128, seed=37, physics_profile='reviewed_v1',
            primordial_subduction={'enabled': True}, subduction_response='moving_hinge_v1')
        cls.disabled = native_engine.Simulation(cls.config)
        cls.active = native_engine.Simulation(dict(cls.config, retained_phases='thermal_v1'))

    def test_activation_preserves_initial_material_and_ordinary_gravity(self):
        before, after = self.disabled, self.active
        for name in ('pos', 'mass', 'kind', 'parcel_plate', 'parcel_patch', 'parcel_craton',
                     'relief', 'support', 'plate', 'age', 'trace_xyz', 'trace_patch'):
            np.testing.assert_array_equal(getattr(before, name), getattr(after, name), err_msg=name)
        for name in ('thickness_km', 'area_factor', 'reference_thickness_km'):
            np.testing.assert_array_equal(before.structure[name], after.structure[name], err_msg=name)
        self.assertEqual(before.rng.bit_generator.state, after.rng.bit_generator.state)
        self.assertEqual(before.trench_systems, after.trench_systems)
        np.testing.assert_array_equal(after.structure[dense_crust.DENSE], 0.)
        np.testing.assert_allclose(after.omega, before.omega, rtol=2.e-12, atol=1.e-15)
        self.assertEqual(after.subduction_response_version, 1)
        self.assertEqual(after.retained_dense_crust_version, 1)
        self.assertEqual(after.retained_phase_floor_version, phase_evolution.FLOOR_VERSION)
        self.assertEqual(after.retained_phase_floor_migration['from_version'], 0)
        self.assertFalse(after.retained_phase_floor_migration['physical_state_changed'])
        self.assertEqual(after.foundering_version, 2)
        self.assertEqual(after.foundering_depth_version, 1)

    def test_initial_temperature_is_the_recorded_unburied_reference_bath(self):
        s = self.active
        # Independent dimensional oracle for the declared 15 C/km geotherm,
        # sampled at 3/4 column depth and capped at 1300 C.
        expected = np.minimum(1300., 11.25*s.structure['thickness_km'])
        np.testing.assert_allclose(dense_crust.temperature(s.structure), expected, rtol=2.e-14)
        order = np.argsort(s.parcel_patch)
        faces = order[np.searchsorted(s.parcel_patch[order], s.trace_patch)]
        np.testing.assert_allclose(dense_crust.temperature(s.trace_structure), expected[faces], rtol=2.e-14)
        diagnostic = s.retained_phase_profile_diagnostics
        self.assertFalse(diagnostic['inherited_thermal_history_reconstructed'])
        self.assertEqual(diagnostic['constitutive_parameters'], phase_evolution.parameters())
        self.assertEqual(diagnostic['initial_temperature_min_c'], float(expected.min()))
        self.assertEqual(diagnostic['initial_temperature_max_c'], float(expected.max()))
        self.assertEqual(diagnostic['column_floor']['version'], phase_evolution.FLOOR_VERSION)
        self.assertEqual(s.physics_profile_diagnostics['retained_phases'], diagnostic)
        self.assertNotIn('disabled', s.physics_profile_diagnostics['retained_dense_crust'])
        frame = s.snapshot()
        self.assertEqual(frame['retained_phase_profile']['profile'], 'thermal_v1')
        self.assertEqual(frame['retained_phase_floor_version'], phase_evolution.FLOOR_VERSION)
        self.assertEqual(frame['retained_phase_floor_migration'], s.retained_phase_floor_migration)
        self.assertEqual(frame['physics_profile']['retained_dense_crust_version'], 1)
        self.assertEqual(frame['physics_profile']['retained_phase_floor_version'], phase_evolution.FLOOR_VERSION)
        mesh_history.arrays(frame)

    def test_existing_saved_profile_does_not_treat_config_edit_as_activation(self):
        old = deepcopy(self.disabled)
        old.config['retained_phases'] = 'thermal_v1'
        before = deepcopy(old.structure)
        physics_profile.initialize(old)
        self.assertFalse(phase_evolution.enabled(old))

        legacy = deepcopy(self.active)
        del legacy.retained_phase_floor_version
        del legacy.retained_phase_floor_migration
        retained_phase_profile.initialize(legacy)
        self.assertFalse(hasattr(legacy, 'retained_phase_floor_version'))
        self.assertEqual(retained_phase_profile.snapshot(old)['retained_phase_profile']['profile'], 'disabled')
        for key in before:
            np.testing.assert_array_equal(old.structure[key], before[key])
        old.t = 1.
        old.steps = 1
        with self.assertRaisesRegex(ValueError, 'resume migration'):
            retained_phase_profile.initialize(old)
        self.assertFalse(phase_evolution.enabled(old))

    def test_native_moving_phase_step_and_checkpoint_replay_close_mass_volume(self):
        s = deepcopy(self.active)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'thermal-moving.npz'
            checkpoint.write_checkpoint(path, s, dict(config=s.config), {})
            restored, _ = checkpoint.read_checkpoint(path, {}, native_engine.Simulation)
        self.assertEqual(restored.retained_phase_profile_diagnostics, s.retained_phase_profile_diagnostics)
        self.assertEqual(restored.retained_phase_floor_version, phase_evolution.FLOOR_VERSION)
        self.assertEqual(restored.retained_phase_floor_migration, s.retained_phase_floor_migration)
        for world in (s, restored):
            world.step(.02)
            dense_crust.validate(world.structure)
            dense_crust.validate(world.trace_structure)
            self.assertEqual(world.subduction_response_version, 1)
            self.assertEqual(world.retained_dense_crust_version, 1)
            self.assertFalse(hasattr(world, 'continental_entry_regions'))
            budget = world.material_column_budget
            self.assertLess(abs(budget['phase_mass_residual_kg'])/budget['after_columns_mass_kg'], 1.e-11)
            self.assertLess(abs(budget['residual_km3'])/budget['after_columns_volume_km3'], 1.e-11)
            force = world.plate_balance_diagnostics
            self.assertLessEqual(force['scaled_force_residual'], plate_balance.FORCE_RELATIVE_TOLERANCE)
            self.assertEqual(force['negative_resisting_work_elements'], 0)
            self.assertLess(abs(force['power_balance_error_w'])/max(force['driver_work_w'], 1.), 1.e-6)
            mesh_history.arrays(world.snapshot())
        for name in ('pos', 'mass', 'omega', 'support', 'age', 'plate'):
            np.testing.assert_array_equal(getattr(s, name), getattr(restored, name), err_msg=name)
        for name in s.structure:
            np.testing.assert_array_equal(s.structure[name], restored.structure[name], err_msg=name)
        for name in s.trace_structure:
            np.testing.assert_array_equal(s.trace_structure[name], restored.trace_structure[name], err_msg=name)
        self.assertEqual(s.trench_systems, restored.trench_systems)
        self.assertEqual(s.rng.bit_generator.state, restored.rng.bit_generator.state)

    def test_moving_hinge_attachment_requires_explicit_neck_histories(self):
        s = deepcopy(self.active)
        before = s.omega.copy()
        with self.assertRaisesRegex(ValueError, 'neck histories'):
            plate_balance.Balance(s, 1., slab_tethers=True)
        np.testing.assert_array_equal(s.omega, before)
        self.assertFalse(any('slab_tether_channels' in row for row in s.trench_systems))


if __name__ == '__main__':
    unittest.main()
