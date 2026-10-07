"""Future-only Lite breakup policy, cancelled loads, and supported paid paths.

No world constructors or simulation integration jobs run in these tests. The
paid-growth fixture uses the real complete Lite ledger and daughter solve.
"""
from copy import deepcopy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import balance_force_ledger
import breakup_mode
import checkpoint
import force_rifting as force
import plate_balance as pb
import plate_limit_analysis as limit
from tests.test_effective_subduction_forces import FORCE_N_PER_M, fixture, opposed_fixture
from tests.test_plate_limit_analysis import hemisphere


POLICY = dict(enabled=True, search_policy=dict(version=1, mode='bounded_axes', max_axes=10))


def activation_state():
    state = fixture()
    state.config.update(physics_profile='reviewed_v1', force_limit_rifting=dict(enabled=False))
    state.t, state.steps = 28., 28
    state.rng = np.random.default_rng(37)
    state.rift_systems = [{'id': 1, 'opened_km': 17.}]
    state.topology_last_myr, state.last_rift_myr = 24., 19.
    return state


def paid_state_and_mode():
    state = opposed_fixture()
    for group, sign in ((slice(0, 3), 1.), (slice(3, 6), -1.)):
        points = np.array([[math.cos(sign*.3+d), math.sin(sign*.3+d), z]
                          for d, z in ((-.025, -.04), (0., .04), (.025, 0.))])
        state.xyz[group] = points/np.linalg.norm(points, axis=1)[:, None]
    state.ba = np.array([0, 3])
    state.config.update(physics_profile='reviewed_v1', force_limit_rifting=dict(enabled=False))
    state.t, state.steps = 28., 28
    state.rng = np.random.default_rng(37)
    model = pb.Balance(state, .1)
    model.solve()
    state.omega = model.rotation()
    cells, piece, axis = np.arange(6), np.array([1, 1, 1, 0, 0, 0], bool), np.array([0., 0., 1.])
    cut = dict(edges=np.array([[0, 3]]), mid=np.array([[1., 0., 0.]]),
               length_m=np.array([1e6]), cut=np.array([True]))
    strength = tuple(np.array([.5*FORCE_N_PER_M*v]) for v in (1., 3., 1./math.sqrt(3.)))
    found = dict(axis=axis, cells=cells, piece=piece, ratio=2., cut_edges=np.array([0]),
                 cut_continental_fraction=0., cut_length_km=1000., search_policy='bounded_axes_v1',
                 searched_axis_count=2, search_scope='held prior axis pair',
                 candidate_axis_sources=['previous supported mode']*2,
                 search_interpretation='Finite directions only.')
    found.update(breakup_mode.solve(state, 0, cells, piece, axis, cut, strength, model))
    if not found['opening_mode_supported'] or found['predicted_opening_cm_yr_mean'] <= 0.:
        raise AssertionError('Fixture must physically overload its represented cut.')
    force.upgrade(state, POLICY)
    return state, model, found


class PolicyActivationTests(unittest.TestCase):
    def test_legacy_defaults_and_explicit_search_normalization(self):
        self.assertEqual(force.normalize(), force.DEFAULTS)
        self.assertNotIn('search_policy', force.normalize(dict(enabled=True)))
        selected = force.normalize(dict(enabled=True, search_policy={}))
        self.assertEqual(selected['search_policy'], force.BOUNDED_SEARCH)
        for bad in (dict(version=True), dict(mode='weak_belt_only'), dict(max_axes=True),
                    dict(max_axes=5), dict(max_axes=17), dict(max_axes=10.), dict(extra=1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                force.normalize(dict(enabled=True, search_policy=bad))

    def test_upgrade_changes_only_declared_policy_and_one_config_branch(self):
        state = activation_state()
        before = deepcopy(vars(state))
        policy = force.upgrade(state, POLICY)
        self.assertEqual(set(vars(state))-set(before), force.POLICY_STATE_FIELDS)
        for name, value in before.items():
            if name == 'config':
                self.assertEqual({k: v for k, v in state.config.items() if k != 'force_limit_rifting'},
                                 {k: v for k, v in value.items() if k != 'force_limit_rifting'})
            elif name == 'rng':
                self.assertEqual(state.rng.bit_generator.state, value.bit_generator.state)
            elif isinstance(value, np.ndarray):
                np.testing.assert_array_equal(getattr(state, name), value)
            else:
                self.assertEqual(getattr(state, name), value)
        self.assertEqual(state.force_rifting_state, {})
        self.assertEqual(state.force_rifting_diagnostics, dict(version=1, checks=0, commits=0, refusals=[]))
        self.assertEqual((policy['activation_myr'], policy['epoch_myr']), (28., 28.))
        self.assertEqual(policy, force.bounded_policy(state))
        policy['epoch_myr'] = 0.
        self.assertEqual(state.force_rifting_policy['epoch_myr'], 28.)

    def test_old_policy_still_fresh_only_and_bad_upgrade_publishes_nothing(self):
        for mutate in (lambda s: s.config.update(physics_profile='legacy'),
                       lambda s: setattr(s, 'plate_resistance_version', 0),
                       lambda s: setattr(s, 'effective_subduction_version', 0),
                       lambda s: s.config.update(force_limit_rifting=dict(enabled=True))):
            state = activation_state(); mutate(state)
            before = deepcopy(state.config)
            with self.assertRaises(ValueError):
                force.upgrade(state, POLICY)
            self.assertEqual(state.config, before)
            self.assertFalse(any(hasattr(state, name) for name in force.POLICY_STATE_FIELDS))
        state = activation_state()
        state.config['force_limit_rifting'] = dict(enabled=True)
        with self.assertRaisesRegex(ValueError, 'fresh-world'):
            force.initialize(state)
        self.assertFalse(any(hasattr(state, name) for name in force.POLICY_STATE_FIELDS))

    def test_duplicate_activation_and_policy_clock_tampering_are_rejected(self):
        state = activation_state(); force.upgrade(state, POLICY)
        before = deepcopy(state.force_rifting_policy)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            force.upgrade(state, POLICY)
        self.assertEqual(state.force_rifting_policy, before)
        for value in (True, 2):
            state.force_rifting_policy['version'] = value
            with self.assertRaises(ValueError):
                force.bounded_policy(state)
        state.force_rifting_policy = before
        state.force_rifting_policy['epoch_myr'] = 29.
        with self.assertRaises(ValueError):
            force.bounded_policy(state)

    def test_fresh_bounded_initialization_failure_is_atomic(self):
        state = activation_state()
        state.t = state.steps = 0
        state.config['force_limit_rifting'] = deepcopy(POLICY)
        state.plate_resistance_version = 0
        with self.assertRaises(ValueError):
            force.initialize(state)
        self.assertFalse(any(hasattr(state, name) for name in force.POLICY_STATE_FIELDS))


class BoundedSpatialSearchTests(unittest.TestCase):
    def test_all_spatially_loaded_owners_include_cancelled_and_nonincoming_sources(self):
        state = fixture(plate_count=4)
        state.active[-1] = False
        rows = {p: dict(cell_generalized_force=np.zeros((6, 3))) for p in range(4)}
        # Opposed ridge/contact tractions have zero rigid sum, nonzero tension.
        rows[1]['cell_generalized_force'][:2, 0] = [7., -7.]
        rows[3]['cell_generalized_force'][0, 0] = 1.
        fake = SimpleNamespace(trench_edges=np.array([0]), slab_owner=np.array([0]))
        self.assertEqual(force._loaded_plates(state, fake), [0])
        self.assertEqual(force._loaded_plates(state, fake, ledger={'plates': rows}), [1])

    def test_cancelled_load_axes_are_deterministic_without_a_weak_belt(self):
        state = SimpleNamespace()
        torque = np.array([[0., 0., 5.], [0., 0., -5.], [0., 2., 0.], [0., -2., 0.]])
        self.assertTrue(np.all(torque.sum(axis=0) == 0.))
        setup = dict(args=(None, torque))
        first, kinds = limit.bounded_candidate_axes(state, 0, setup)
        second, _ = limit.bounded_candidate_axes(state, 0, setup)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(any(abs(axis[2]) == 1. for axis in first))
        self.assertEqual(set(kinds), {'principal spatial load'})
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.)

    def test_inherited_strike_uses_axial_moment_and_held_cadence_preserves_axis(self):
        state = SimpleNamespace(parcel_plate=np.array([0, 0, 1]), rift_id=np.array([1, 1, 1]),
            mass=np.ones(3), rift_tangent=np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]]))
        setup = dict(args=(None, np.array([[0., 0., 2.], [0., 0., -2.]])))
        axes, kinds = limit.bounded_candidate_axes(state, 0, setup)
        self.assertIn('inherited material rift', kinds)
        self.assertTrue(any(abs(axis[0]) == 1. for axis in axes))
        prior = np.array([1., 1., 1.])/math.sqrt(3.)
        held, kinds = limit.bounded_candidate_axes(state, 0, setup, seed_axes=[prior], held_only=True)
        np.testing.assert_allclose(held, [prior, -prior])
        self.assertEqual(kinds, ['previous supported mode']*2)

    def test_finite_axes_keep_full_strength_and_resolved_minimum_cut(self):
        points, torque, edges, mid, length, strength, *_ = hemisphere(2, 5e12, 4e12)
        args = (points, torque, edges, mid, length, (strength, 3*strength, strength/math.sqrt(3.)))
        axes, _ = limit.bounded_candidate_axes(SimpleNamespace(), 0, dict(args=args))
        weak = limit.search_prepared(args, np.zeros(len(edges), bool), [], axes=axes)
        strong_args = (*args[:-1], tuple(2*v for v in args[-1]))
        strong = limit.search_prepared(strong_args, np.zeros(len(edges), bool), [], axes=axes)
        self.assertGreater(weak['ratio'], 1.)
        self.assertLess(strong['ratio'], 1.)
        self.assertGreaterEqual(min(weak['piece'].sum(), (~weak['piece']).sum()), limit.MIN_PIECE_CELLS)
        self.assertAlmostEqual(weak['ratio'], 2*strong['ratio'], delta=.002*weak['ratio'])
        self.assertLessEqual(len(axes), 10)

    def test_existing_suture_strength_is_finite_and_craton_blocking_stays_uncuttable(self):
        state = SimpleNamespace(xyz=np.eye(3), kind=np.array([1, 1, 2]), crust=np.array([1, 1, 2]),
            suture=np.array([.8, .2, .8]), rift_id=np.array([1, -1, -1]),
            parcel_cell=np.arange(3), mass=np.ones(3))
        weak = limit.inherited_weakness(state)
        self.assertTrue(np.all(np.isfinite(weak)))
        self.assertTrue(np.all((weak > 0.) & (weak <= 1.)))
        self.assertLess(weak[0], weak[1])
        self.assertEqual(weak[1], 1.)
        strength = limit.cell_strengths(state.crust, np.zeros(3))
        self.assertTrue(all(np.all(np.isfinite(value*weak)) and np.all(value*weak > 0.) for value in strength))
        points, torque, edges, mid, length, capacity, *_ = hemisphere(2, 5e12, 4e12)
        args = (points, torque, edges, mid, length, (capacity, 3*capacity, capacity/math.sqrt(3.)))
        axes, _ = limit.bounded_candidate_axes(SimpleNamespace(), 0, dict(args=args))
        blocked = (points[edges[:, 0], 0] > .6) & (points[edges[:, 1], 0] > .6)
        found = limit.search_prepared(args, blocked, [], axes=axes)
        self.assertFalse(np.any(found['cut'] & blocked))


class SupportedVirtualPathTests(unittest.TestCase):
    def run_update(self, state, model, found, dt=.1):
        state.t += dt
        with patch.object(limit, 'active_balance', return_value=model), \
                patch.object(force, '_loaded_plates', return_value=[0]), \
                patch.object(limit, 'worst_mechanism', return_value=deepcopy(found)):
            force.update(state, dt)

    def test_first_observation_has_no_past_credit_then_actual_cut_work_pays_growth(self):
        state, model, found = paid_state_and_mode()
        self.run_update(state, model, found)
        uid = str(state.plate_uid[0])
        first = deepcopy(state.force_rifting_state[uid])
        self.assertEqual(first['opened_km'], 0.)
        self.assertEqual(first['paid_cut_work_j'], 0.)
        self.assertEqual(first['supported_elapsed_myr'], 0.)
        self.run_update(state, model, found)
        second = state.force_rifting_state[uid]
        self.assertGreater(second['opened_km'], 0.)
        self.assertGreater(second['paid_cut_work_j'], 0.)
        self.assertAlmostEqual(second['supported_elapsed_myr'], .1)
        receipt = second['last_opening_work']
        self.assertLessEqual(receipt['cut_work_j'], receipt['held_mode_available_cut_work_j']*(1+1e-12))
        self.assertLess(receipt['power_closure_relative_to_cut'], 1e-8)
        self.assertEqual(second['paid_cut_work_j'], state.force_rifting_diagnostics['accepted_cut_work_j'])
        self.assertEqual(second['support_identity'], first['support_identity'])

    def test_changed_axis_resets_without_refunding_and_rechecks_intact(self):
        state, model, found = paid_state_and_mode()
        self.run_update(state, model, found); self.run_update(state, model, found)
        uid = str(state.plate_uid[0]); paid = state.force_rifting_state[uid]['paid_cut_work_j']
        changed = deepcopy(found); changed['axis'] = np.array([0., .1, 1.]); changed['axis'] /= np.linalg.norm(changed['axis'])
        state.t += .1
        with patch.object(limit, 'active_balance', return_value=model), \
                patch.object(force, '_loaded_plates', return_value=[0]), \
                patch.object(limit, 'worst_mechanism', side_effect=[changed, deepcopy(changed)]) as search:
            force.update(state, .1)
        self.assertIsNotNone(search.call_args_list[0].kwargs['edge_strength_scale'])
        self.assertIsNone(search.call_args_list[1].kwargs['edge_strength_scale'])
        self.assertEqual(state.force_rifting_state[uid]['opened_km'], 0.)
        self.assertEqual(state.force_rifting_state[uid]['paid_cut_work_j'], 0.)
        self.assertEqual(state.force_rifting_diagnostics['accepted_cut_work_j'], paid)
        self.assertEqual(state.force_rifting_diagnostics['retired_candidate_paths'][0]['paid_cut_work_j'], paid)

    def test_exact_support_includes_oriented_piece_owner_edges_and_material_partition(self):
        state, _, found = paid_state_and_mode()
        first = force._stable_cut(state, 0, found)
        for mutate in (lambda s, f: f.update(piece=~f['piece']),
                       lambda s, f: f.update(cut_edges=np.array([1])),
                       lambda s, f: s.plate_uid.__setitem__(0, 99)):
            changed, cut = deepcopy(state), deepcopy(found); mutate(changed, cut)
            self.assertNotEqual(force._stable_cut(changed, 0, cut), first)
        state.n = len(state.xyz); state.parcel_plate = np.zeros(2, int)
        state.parcel_patch = np.array([71, 72]); state.mass = np.ones(2)
        state.pos = state.xyz[[0, 3]].copy(); state.parcel_craton = np.full(2, -1)
        state.parcel_cell = np.array([0, 3]); state._indices = lambda points: np.argmax(points@state.xyz.T, axis=1)
        material = force._stable_cut(state, 0, found)
        state.parcel_patch[1] = 73
        self.assertNotEqual(force._stable_cut(state, 0, found), material)

    def test_repeated_epoch_and_later_error_publish_no_history_or_policy_clock(self):
        state, model, found = paid_state_and_mode()
        with self.assertRaisesRegex(ValueError, 'matching accepted'):
            force.update(state, .1)
        before = deepcopy((state.force_rifting_state, state.force_rifting_diagnostics, state.force_rifting_policy))
        state.t += .1
        with patch.object(limit, 'active_balance', return_value=model), \
                patch.object(force, '_loaded_plates', return_value=[0, 1]), \
                patch.object(limit, 'worst_mechanism', side_effect=[found, ValueError('later unpriced source')]):
            with self.assertRaisesRegex(ValueError, 'later unpriced'):
                force.update(state, .1)
        self.assertEqual((state.force_rifting_state, state.force_rifting_diagnostics, state.force_rifting_policy), before)

    def test_healing_keeps_spent_work_and_checkpoint_replay_keeps_new_clock(self):
        state, model, found = paid_state_and_mode()
        self.run_update(state, model, found); self.run_update(state, model, found)
        uid = str(state.plate_uid[0]); before = deepcopy(state.force_rifting_state[uid])
        subyield = deepcopy(found); subyield.update(ratio=.5, predicted_opening_cm_yr_mean=0.)
        self.run_update(state, model, subyield)
        self.assertLess(state.force_rifting_state[uid]['opened_km'], before['opened_km'])
        self.assertEqual(state.force_rifting_state[uid]['paid_cut_work_j'], before['paid_cut_work_j'])
        compatibility = dict(engine_sha256='test', auxiliary_sources_sha256='test', numpy_version=np.__version__,
                             flow_backend='test', material_backend='test')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.npz'
            checkpoint.write_checkpoint(path, state, dict(config=state.config), compatibility)
            replay, _ = checkpoint.read_checkpoint(path, compatibility, SimpleNamespace)
        self.assertEqual(replay.force_rifting_policy, state.force_rifting_policy)
        self.assertEqual(replay.force_rifting_state, state.force_rifting_state)
        self.assertEqual(replay.rng.bit_generator.state, state.rng.bit_generator.state)
        replay.t += .1
        with patch.object(limit, 'active_balance', return_value=model), \
                patch.object(force, '_loaded_plates', return_value=[0]), \
                patch.object(limit, 'worst_mechanism', return_value=deepcopy(found)):
            force.update(replay, .1)
        self.assertGreater(replay.force_rifting_state[uid]['opened_km'], state.force_rifting_state[uid]['opened_km'])

    def test_bounded_frame_rejects_retrospective_clocks_versions_and_work_tampering(self):
        force.validate_frame({})
        force.validate_frame(dict(force_rifting_version=1, force_rifting_plates={}))
        state, model, found = paid_state_and_mode()
        first = dict(time_myr=state.t, **force.snapshot(state))
        force.validate_frame(first)
        self.run_update(state, model, found); self.run_update(state, model, found)
        frame = dict(time_myr=state.t, **force.snapshot(state))
        force.validate_frame(frame)
        uid = str(state.plate_uid[0])
        for mutate in (lambda f: f['force_rifting_policy'].update(version=True),
                       lambda f: f['force_rifting_policy'].update(activation_myr=f['time_myr']+1.),
                       lambda f: f['force_rifting_plates'][uid].update(since_myr=0.),
                       lambda f: f['force_rifting_plates'][uid].update(paid_cut_work_j=float('nan')),
                       lambda f: f['force_rifting_diagnostics'].update(accepted_cut_work_j=0.),
                       lambda f: f['force_rifting_plates'][uid].update(support_identity='not a digest')):
            changed = deepcopy(frame); mutate(changed)
            with self.assertRaises(ValueError):
                force.validate_frame(changed)
        first['force_rifting_diagnostics']['accepted_cut_work_j'] = 1.
        with self.assertRaises(ValueError):
            force.validate_frame(first)
        missing = deepcopy(frame); del missing['force_rifting_policy']
        with self.assertRaisesRegex(ValueError, 'lost its declared policy'):
            force.validate_frame(missing)
        marker = deepcopy(frame); marker['force_rifting_policy_version'] = True
        with self.assertRaises(ValueError):
            force.validate_frame(marker)
        rounded = activation_state(); rounded.t = 28.123456789
        force.upgrade(rounded, POLICY)
        force.validate_frame(dict(time_myr=round(rounded.t, 6), **force.snapshot(rounded)))


if __name__ == '__main__':
    unittest.main()
