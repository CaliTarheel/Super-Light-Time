"""Historical activation, native dispatch, contact gates and exact rollback."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import checkpoint
import evolution_policy as policy
import native_engine
import native_material_evolution
from deforming_regions import IncompleteContactStepError
from tests.test_same_sheet_nonpenetration import fan_state, deformation_result


def historical_world():
    s = native_engine.Simulation.__new__(native_engine.Simulation)
    s.t, s.steps = 146., 75
    s.config = dict(dt_myr=1.)
    s.rng = np.random.default_rng(37)
    s.mass = np.array([2., 4., 8.])
    s.alias = s.mass
    s.history = []
    return s


def partial_contact_step(s, dt):
    """Controlled raw operator exercises the real native transaction dispatch."""
    s.mass += dt
    s.history.append((dt, float(s.rng.random())))
    s.steps += 1
    s.t += dt
    if policy.complete_contact_version(s) and dt > .25:
        raise IncompleteContactStepError('Contact covers only part of the requested interval.')


class EvolutionPolicyTests(unittest.TestCase):
    def test_absent_fields_preserve_legacy_and_reviewed_fallback(self):
        self.assertEqual(policy.versions(SimpleNamespace()), dict.fromkeys(policy.FIELDS, 0))
        reviewed = SimpleNamespace(physics_profile_version=1)
        self.assertEqual(policy.versions(reviewed), dict.fromkeys(policy.FIELDS, 1))
        reviewed.adaptive_timestep_version = 0
        self.assertEqual(policy.adaptive_timestep_version(reviewed), 0)
        self.assertEqual(policy.complete_contact_version(reviewed), 1)
        reviewed.complete_contact_version = 0
        self.assertEqual(policy.versions(reviewed), dict.fromkeys(policy.FIELDS, 0))

    def test_only_integer_versions_zero_and_one_are_accepted(self):
        for name in (*policy.FIELDS, 'physics_profile_version'):
            for value in (False, True, np.bool_(True), .0, 1., '1', None, -1, 2, np.nan):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    policy.versions(SimpleNamespace(**{name: value}))
        for value in (0, 1, np.int32(0), np.int64(1)):
            s = SimpleNamespace(adaptive_timestep_version=value, complete_contact_version=value)
            self.assertEqual(policy.versions(s), dict.fromkeys(policy.FIELDS, int(value)))

    def test_explicit_upgrade_is_idempotent_without_initialization(self):
        s = historical_world()
        before = deepcopy(vars(s))
        report = policy.upgrade(s)
        self.assertEqual(policy.upgrade(s), report)
        self.assertEqual(report['time_myr'], 146.)
        self.assertEqual(policy.versions(s), dict.fromkeys(policy.FIELDS, 1))
        self.assertFalse(hasattr(s, 'physics_profile_version'))
        self.assertEqual(s.config, before['config'])
        self.assertEqual(s.rng.bit_generator.state, before['rng'].bit_generator.state)
        np.testing.assert_array_equal(s.mass, before['mass'])
        self.assertIs(s.mass, s.alias)
        self.assertEqual((s.t, s.steps, s.history), (146., 75, []))
        self.assertEqual(set(vars(s))-set(before), {*policy.FIELDS, 'evolution_policy_migration'})
        report['to_versions']['complete_contact_version'] = 0
        self.assertEqual(s.evolution_policy_migration['to_versions']['complete_contact_version'], 1)

    def test_invalid_upgrade_or_step_rejects_before_mutation(self):
        s = historical_world()
        s.complete_contact_version = True
        before = set(vars(s))
        with self.assertRaises(ValueError):
            policy.upgrade(s)
        self.assertEqual(set(vars(s)), before)
        with patch.object(native_engine.Simulation, '_step_once') as raw:
            with self.assertRaises(ValueError):
                s.step(1.)
            raw.assert_not_called()

    def test_native_dispatch_retries_whole_contact_interval_without_profile(self):
        s, expected = historical_world(), historical_world()
        policy.upgrade(s)
        policy.upgrade(expected)
        with patch.object(native_engine.Simulation, '_step_once', partial_contact_step):
            s.step(1.)
            for _ in range(4):
                expected.step(.25)
        self.assertEqual(s.t, 147.)
        self.assertEqual(s.steps, 79)
        self.assertEqual(s.history, expected.history)
        self.assertEqual(s.rng.bit_generator.state, expected.rng.bit_generator.state)
        np.testing.assert_array_equal(s.mass, expected.mass)
        self.assertIs(s.mass, s.alias)
        self.assertEqual(s.timestep_diagnostics['rejected_trials'], 3)
        self.assertFalse(hasattr(s, 'physics_profile_version'))

    def test_native_dispatch_retains_legacy_fallback_and_explicit_override(self):
        legacy, reviewed, overridden = (historical_world() for _ in range(3))
        reviewed.physics_profile_version = overridden.physics_profile_version = 1
        overridden.adaptive_timestep_version = overridden.complete_contact_version = 0
        with patch.object(native_engine.Simulation, '_step_once', partial_contact_step):
            for state in (legacy, reviewed, overridden):
                state.step(1.)
        self.assertEqual([row[0] for row in legacy.history], [1.])
        self.assertEqual([row[0] for row in overridden.history], [1.])
        self.assertEqual([row[0] for row in reviewed.history], [.25]*4)
        self.assertFalse(hasattr(legacy, 'timestep_diagnostics'))
        self.assertFalse(hasattr(overridden, 'timestep_diagnostics'))

    def test_failed_later_substep_restores_entire_historical_checkpoint(self):
        s = historical_world()
        policy.upgrade(s)
        before = deepcopy(vars(s))
        def failing(state, dt):
            old = state.t
            partial_contact_step(state, dt)
            if old >= 146.25:
                raise IncompleteContactStepError('Unrepairable later contact.')
        with patch.object(native_engine.Simulation, '_step_once', failing):
            with self.assertRaises(IncompleteContactStepError):
                s.step(.5)
        self.assertEqual((s.t, s.steps, s.history), (146., 75, []))
        self.assertEqual(s.rng.bit_generator.state, before['rng'].bit_generator.state)
        np.testing.assert_array_equal(s.mass, before['mass'])
        self.assertIs(s.mass, s.alias)
        self.assertEqual(s.evolution_policy_migration, before['evolution_policy_migration'])

    def test_complete_contact_override_reaches_native_material_operator(self):
        for profile, explicit, expected in ((None, None, False), (1, None, True),
                                             (None, 1, True), (1, 0, False)):
            with self.subTest(profile=profile, explicit=explicit):
                s = fan_state()
                if profile is not None:
                    s.physics_profile_version = profile
                if explicit is not None:
                    s.complete_contact_version = explicit
                calls = []
                def deform(surface, omega, boundaries, dt, **kwargs):
                    calls.append(kwargs['require_complete_contact'])
                    return deformation_result(surface, surface['vertices'], kwargs.get('vertex_protected'))
                with patch('deforming_regions.deform', side_effect=deform):
                    native_material_evolution.advect(s, .1, prepared_loading=({}, {}))
                self.assertTrue(calls)
                self.assertTrue(all(value is expected for value in calls))

    def test_checkpoint_roundtrip_preserves_policies_and_adaptive_continuation(self):
        s = historical_world()
        policy.upgrade(s)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.npz'
            checkpoint.write_checkpoint(path, s, dict(config=s.config), {})
            restored, _ = checkpoint.read_checkpoint(path, {}, native_engine.Simulation)
        self.assertEqual(policy.snapshot(s), policy.snapshot(restored))
        with patch.object(native_engine.Simulation, '_step_once', partial_contact_step):
            s.step(.5)
            restored.step(.5)
        self.assertEqual(s.history, restored.history)
        self.assertEqual(s.rng.bit_generator.state, restored.rng.bit_generator.state)
        np.testing.assert_array_equal(s.mass, restored.mass)
        self.assertEqual(policy.snapshot(s), policy.snapshot(restored))
        self.assertFalse(hasattr(restored, 'physics_profile_version'))

    def test_source_capture_includes_policy_for_exact_restart(self):
        import server
        self.assertEqual(server.AUXILIARY_SOURCES['evolution_policy.py'],
                         Path(policy.__file__).read_bytes())


if __name__ == '__main__':
    unittest.main()
