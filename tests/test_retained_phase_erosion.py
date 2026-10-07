"""Ordinary-surface erosion above a retained dense base, with saved opt-in."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

import checkpoint
import crustal_structure as columns
import crust_inventory as inventory
import dense_crust as phase
import native_engine
import physics_profile
import retained_phase_profile
import structure_engine as structure


def mixture(ordinary_km, dense_km, *, area=1., loaded_height=1000., foreland=0., temperature=700.):
    """Explicit valid two-layer columns; no production inversion in the oracle."""
    ordinary, dense, area, height, foreland, temperature = np.broadcast_arrays(
        np.atleast_1d(ordinary_km).astype(float), np.atleast_1d(dense_km).astype(float),
        area, loaded_height, foreland, temperature)
    state = columns.initialize_structure(np.ones(ordinary.shape, int))
    state['thickness_km'] = (ordinary+dense).copy()
    state['reference_thickness_km'] = (ordinary+dense).copy()
    state['area_factor'] = area.copy()
    state['foreland_m'] = foreland.copy()
    # h_air = reference_air - (3450-2800)/3300 * dense thickness in metres.
    reference_air = np.where(height >= 0., height, height*2270./3300.) + 650./3300.*1000.*dense
    state['reference_elevation_m'] = np.where(reference_air >= 0., reference_air, reference_air*3300./2270.)
    inventory.initialize(state)
    phase.initialize(state, temperature)
    state[phase.DENSE] = dense*area
    state[phase.CONVERTED] = state[phase.DENSE].copy()
    # A column may become mostly dense after its ordinary upper crust erodes.
    state[inventory.REMAINING] = (.25*ordinary+dense*3450./2800.)*area
    state[inventory.BASELINE] = state[inventory.REMAINING].copy()
    state[phase.HEAT] = (ordinary+dense*3450./2800.)*area*temperature
    state[phase.HEAT_BASELINE] = state[phase.HEAT].copy()
    phase.validate(state)
    return state


def erode(state, plan):
    return columns.evolve_structure(state, 1., denudation_m=plan['denudation_m'],
                                    foreland_target_m=state['foreland_m'])


class RetainedPhaseErosionTests(unittest.TestCase):
    def test_independent_density_and_water_crossing_with_fixed_foreland(self):
        state = mixture([20., 25., 25.], [5., 5., 5.], area=[.3, 2., 4.],
                        loaded_height=[20., -100., 1000.], foreland=[300., 20., 0.])
        before = deepcopy(state)
        plan = columns.ordinary_denudation_for_net_loss(state, [50., 100., 100.])
        expected = np.array([(20.+30.*2270./3300.)*3300./500., 100.*2270./500., 100.*3300./500.])
        np.testing.assert_allclose(plan['denudation_m'], expected, atol=2e-11)
        result, budget = erode(state, plan)
        np.testing.assert_allclose(columns.elevation(result), [-330., -220., 900.], atol=2e-11)
        np.testing.assert_allclose(budget['net_erosion_loss_m'], [50., 100., 100.], atol=2e-11)
        np.testing.assert_allclose(plan['realized_net_loss_m'], budget['net_erosion_loss_m'], atol=2e-11)
        for field in state:
            np.testing.assert_array_equal(state[field], before[field])

    def test_removal_closes_mass_heat_volume_and_preserves_dense_base_area_temperature(self):
        state = mixture([20., 25.], [8., 4.], area=[3.5, .7], temperature=[650., 850.])
        plan = columns.ordinary_denudation_for_net_loss(state, [100., 250.])
        result, budget = erode(state, plan)
        removed_volume = plan['denudation_m']/1000.*state['area_factor']
        before_mass = (state['thickness_km']*state['area_factor']*2800.
                       + (3450.-2800.)*state[phase.DENSE])*1.e9
        after_mass = (result['thickness_km']*result['area_factor']*2800.
                      + (3450.-2800.)*result[phase.DENSE])*1.e9
        np.testing.assert_allclose(before_mass-after_mass, removed_volume*2800.e9, rtol=2e-14)
        np.testing.assert_allclose(budget['removed_volume_km_per_reference_km2'], removed_volume, rtol=2e-14)
        np.testing.assert_allclose(result[phase.HEAT_ERODED], removed_volume*np.array([650., 850.]), rtol=2e-14)
        np.testing.assert_allclose(phase.temperature(result), [650., 850.], rtol=2e-14)
        for field in (phase.DENSE, phase.CONVERTED, phase.RETURNED, phase.ERODED, 'area_factor'):
            np.testing.assert_array_equal(result[field], state[field])
        phase.validate(result)
        inventory.validate(result)

    def test_pure_ordinary_phase_state_matches_historical_inverse(self):
        state = mixture([35., 42., 25.], 0., loaded_height=[1000., -100., 20.], area=[.1, 3., 5.])
        for loss in (0., [100., 50., 100.]):
            new = columns.ordinary_denudation_for_net_loss(state, loss)
            np.testing.assert_array_equal(new['denudation_m'], columns.denudation_for_net_loss(state, loss))
            self.assertFalse(np.any(new['limited']))

    def test_available_ordinary_layer_and_phase_adjusted_floor_caps_report_unmet_relief(self):
        state = mixture([2., 0., 7.], [18., 12., 2.], area=[3., .7, 2.])
        plan = columns.ordinary_denudation_for_net_loss(state, [1000., 1000., 1000.])
        expected_denudation = np.array([2000., 0., 1464.2857142857142])
        np.testing.assert_allclose(plan['denudation_m'], expected_denudation, atol=2e-12)
        result, budget = erode(state, plan)
        np.testing.assert_allclose(result['thickness_km'], [18., 12., 7.535714285714286], atol=2e-14)
        np.testing.assert_allclose(columns.restored_thickness(result), [22.17857142857143, 14.785714285714285, 8.], atol=2e-14)
        np.testing.assert_array_equal(result[phase.DENSE], state[phase.DENSE])
        np.testing.assert_array_equal(result[phase.ERODED], 0.)
        np.testing.assert_allclose(plan['realized_net_loss_m'], expected_denudation*500./3300., atol=2e-11)
        np.testing.assert_allclose(plan['unmet_net_loss_m'], 1000.-budget['net_erosion_loss_m'], atol=2e-11)
        self.assertTrue(np.all(plan['limited']))
        phase.validate(result)
        # Removing dense basement has the opposite sign; the cap must stop it.
        raw, raw_budget = columns.evolve_structure(mixture(0., 12.), 1., denudation_m=1000.)
        self.assertLess(float(raw_budget['net_erosion_loss_m'][0]), 0.)
        self.assertGreater(float(raw[phase.ERODED][0]), 0.)

    def test_version_two_requires_saved_phase_state_and_old_one_still_rejects(self):
        state = mixture(20., 10.)
        active = SimpleNamespace(config={'erosion':1.}, erosion_relief_version=2, retained_dense_crust_version=1)
        result = structure._erosion_request(active, state, np.array([1]), np.array([1000.]), 2., np.ones(1))
        expected = 1000.*(-np.expm1(-2./180.))*3300./500.
        np.testing.assert_allclose(result, [expected], atol=2e-11)
        active.erosion_relief_version = 1
        with self.assertRaisesRegex(ValueError, 'retained dense crust'):
            structure._erosion_request(active, state, np.array([1]), np.array([1000.]), 2., np.ones(1))
        with self.assertRaisesRegex(ValueError, 'retained dense crust'):
            columns.denudation_for_net_loss(state, [1.])
        active.erosion_relief_version = 2
        for version in (0, None, True):
            if version is None:
                del active.retained_dense_crust_version
            else:
                active.retained_dense_crust_version = version
            with self.assertRaises(ValueError):
                structure._erosion_request(active, state, np.array([1]), np.array([1000.]), 2., np.ones(1))
        active.retained_dense_crust_version = 1
        with self.assertRaisesRegex(ValueError, 'explicit retained phase state'):
            structure._erosion_request(active, columns.initialize_structure([1]), np.array([1]), np.array([1000.]), 2., np.ones(1))
        for loss in (-1., np.inf, np.nan):
            with self.assertRaises(ValueError):
                columns.ordinary_denudation_for_net_loss(state, loss)


class NativeRetainedPhaseErosionTests(unittest.TestCase):
    def test_fresh_thermal_two_steps_and_checkpoint_replay_preserve_closure(self):
        config = dict(width=48, height=24, mesh_level=2, coast_geometry_level=2,
            plate_count=4, mechanics_nodes=128, seed=37, physics_profile='reviewed_v1',
            primordial_subduction={'enabled':True}, subduction_response='moving_hinge_v1',
            retained_phases='thermal_v1')
        s = native_engine.Simulation(config)
        self.assertEqual(s.erosion_relief_version, 2)
        self.assertIn('Version 2', s.physics_profile_diagnostics['erosion'])
        s.step(2.)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'thermal-erosion.npz'
            checkpoint.write_checkpoint(path, s, dict(config=s.config), {})
            restored, _ = checkpoint.read_checkpoint(path, {}, native_engine.Simulation)
        self.assertEqual(restored.erosion_relief_version, 2)
        for candidate in (s, restored):
            candidate.step(2.)
            phase.validate(candidate.structure)
            phase.validate(candidate.trace_structure)
            self.assertEqual(candidate.erosion_relief_diagnostics['version'], 2)
            self.assertFalse(candidate.erosion_relief_diagnostics['dense_basement_erosion'])
            frame = candidate.snapshot()
            self.assertEqual(frame['erosion_relief_version'], 2)
            self.assertEqual(frame['physics_profile']['erosion_relief_version'], 2)
            self.assertEqual(frame['erosion_relief_diagnostics'], candidate.erosion_relief_diagnostics)
            budget = candidate.material_column_budget
            self.assertLess(abs(budget['phase_mass_residual_kg'])/budget['after_columns_mass_kg'], 1.e-11)
            self.assertLess(abs(budget['residual_km3'])/budget['after_columns_volume_km3'], 1.e-11)
        for name in ('pos', 'mass', 'omega', 'age', 'plate'):
            np.testing.assert_array_equal(getattr(s, name), getattr(restored, name), err_msg=name)
        for name in ('structure', 'trace_structure'):
            for field in getattr(s, name):
                np.testing.assert_array_equal(getattr(s, name)[field], getattr(restored, name)[field], err_msg=field)
        self.assertEqual(s.erosion_relief_diagnostics, restored.erosion_relief_diagnostics)
        self.assertEqual(s.rng.bit_generator.state, restored.rng.bit_generator.state)

    def test_old_saved_thermal_version_is_not_silently_migrated(self):
        # Initialization's idempotent saved-profile path must not replace old
        # checkpoint behavior; the existing flag is authoritative on resume.
        saved = SimpleNamespace(config={'retained_phases':'thermal_v1', 'physics_profile':'reviewed_v1'},
            retained_phase_profile_version=1, retained_phase_profile_diagnostics={'profile':'thermal_v1'},
            physics_profile_version=1, erosion_relief_version=1, t=4., steps=2)
        physics_profile.initialize(saved)
        self.assertEqual(saved.erosion_relief_version, 1)
        with self.assertRaisesRegex(ValueError, 'resume migration'):
            retained_phase_profile.initialize(saved)
        self.assertEqual(saved.erosion_relief_version, 1)
        saved.t, saved.steps = 0., 0
        retained_phase_profile.initialize(saved)
        self.assertEqual(saved.erosion_relief_version, 1)


if __name__ == '__main__':
    unittest.main()
