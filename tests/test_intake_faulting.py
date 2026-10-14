"""Geometric incompatibility and actual conservative native fault commits."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

import constrained_plate_motion
import intake_faulting as faulting
import plate_balance as pb
from tests.test_native_topology import ocean_partition
from tests.test_constrained_plate_motion import Quadratic
from tests.test_persistent_ocean_intake import attached_fixture
import persistent_ocean_intake as intake


def loaded_cut(s, proposal, multiplier=2.):
    cells = np.flatnonzero(s.plate == 0)
    local = np.full(s.n, -1, int); local[cells] = np.arange(len(cells))
    internal = (s.plate[s.edge_a] == 0) & (s.plate[s.edge_b] == 0)
    edges = np.column_stack((local[s.edge_a[internal]], local[s.edge_b[internal]]))
    mid = s.edge_mid[internal]; length = s.edge_length[internal]*1000.
    piece = np.isin(cells, proposal['child_faces'])
    axis = np.array([0., 0., 1.]); strength = tuple(np.full(len(edges), 1e12) for _ in range(3))
    a, b = edges.T; cut = piece[a] != piece[b]
    normal = s.xyz[cells[b]]-s.xyz[cells[a]]
    normal -= mid*np.sum(mid*normal, axis=1)[:, None]
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    velocity = np.cross(axis, mid)
    # Equal tension/compression/shear strengths give an independent absolute
    # normal-plus-tangential cut capacity, irrespective of side labeling.
    capacity = float(np.sum((length*pb.RADIUS_M*1e12*(np.abs(np.sum(velocity*normal, axis=1))
        +np.abs(np.sum(velocity*np.cross(mid, normal), axis=1))))[cut]))
    torque = np.zeros((len(cells), 3))
    torque[piece, 2] = multiplier*capacity/piece.sum()
    torque[~piece, 2] = -multiplier*capacity/(~piece).sum()
    setup = dict(cells=cells, internal=internal, craton=np.zeros(len(edges), bool),
        continental=np.zeros(len(edges), bool),
        args=(s.xyz[cells], torque, edges, mid, length, strength))
    return dict(plate=0, setup=setup, axis=axis, piece=piece, ratio=multiplier)


class IntakeFaultingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world, cls.proposal = ocean_partition()

    def test_incompatible_loop_has_a_nonnegative_self_balanced_stress_certificate(self):
        rows = np.array([[1., -1., 0.], [0., 1., -1.], [-1., 0., 1.]])
        minimum = np.full(3, .1)
        weights = faulting.incompatibility_stress(rows, minimum)
        np.testing.assert_allclose(weights, np.full(3, 1./3.), atol=1e-12)
        np.testing.assert_allclose(rows.T@weights, 0., atol=1e-12)
        self.assertGreater(float(minimum@weights), 0.)
        with self.assertRaises(constrained_plate_motion.IncompatibleIntake):
            faulting.incompatibility_stress(rows[:2], minimum[:2])

    def test_yielded_native_fault_changes_ownership_without_kicking_or_rejuvenating(self):
        s = deepcopy(self.world)
        before = {key: getattr(s, key).copy() for key in ('mass', 'age', 'omega', 'mantle')}
        support = s.support.sum(axis=0)
        count = int(s.active.sum())
        event = faulting.commit(s, loaded_cut(s, self.proposal))
        self.assertIsNotNone(event)
        self.assertAlmostEqual(event['loading_ratio'], 2., places=10)
        self.assertEqual(int(s.active.sum()), count+1)
        self.assertEqual(event['assigned_opening_km'], 0.)
        q = int(np.flatnonzero(s.plate_uid == event['daughter_plate_uid'])[0])
        np.testing.assert_array_equal(s.omega[q], before['omega'][0])
        np.testing.assert_array_equal(s.mantle[q], before['mantle'][0])
        np.testing.assert_array_equal(s.age, before['age'])
        np.testing.assert_array_equal(s.mass, before['mass'])
        np.testing.assert_array_equal(s.support.sum(axis=0), support)
        self.assertIn('yielded fault', s.events[-1]['description'])

    def test_subcritical_cut_is_not_released(self):
        s = deepcopy(self.world); uid = s.next_plate_uid
        self.assertIsNone(faulting.commit(s, loaded_cut(s, self.proposal, .9)))
        self.assertEqual(s.next_plate_uid, uid)
        np.testing.assert_array_equal(s.plate, self.world.plate)

    def test_feasible_intake_advances_without_searching_more_fractures(self):
        s = attached_fixture(); s.t = 1.
        s.trench_systems[0]['effective_subduction'][intake.LATCH] = 1.
        expected = pb.Balance(s, 1.); expected.solve()
        with patch.object(faulting, 'candidates', side_effect=AssertionError('Unnecessary fracture search')):
            omega, report = faulting.solve(s, 1.)
        np.testing.assert_allclose(omega, expected.rotation(), atol=1e-12)
        self.assertTrue(report['intake_fault_release']['obligations_satisfied'])
        self.assertEqual(report['intake_fault_release']['events'], [])

    def test_trial_ranking_selects_one_resolving_cut_and_preserves_original(self):
        original_rows = np.array([[1., 0., 0., -1., 0., 0.], [-1., 0., 0., 1., 0., 0.]])
        before = SimpleNamespace(plates=np.array([0, 1]), size=6,
            intake_constraints=dict(rows=original_rows, minimum=np.full(2, .1)))
        world = SimpleNamespace(active=np.ones(3, bool), plate_uid=np.arange(1, 4), choice=None,
            _rasterize=lambda: None, _boundaries=lambda: None)

        def release(trial, candidate, **_):
            trial.choice = candidate['choice']
            return dict(daughter_plate_uid=3, parent_plate_uid=1, cut_length_km=10.,
                        loading_ratio=100. if trial.choice == 'ineffective' else 2.)

        class Motion(Quadratic):
            def _evaluate(self, x, delta, **kwargs):
                return super()._evaluate(x, delta)

            def __init__(self, trial, dt):
                super().__init__(np.zeros(9)); self.plates = np.arange(3)
                rows = np.pad(original_rows, ((0, 0), (0, 3)))
                if trial.choice == 'resolving':
                    rows[0, 0] = 0.; rows[0, 6] = 1.
                self.intake_constraints = dict(rows=rows, minimum=np.full(2, .1))

            def solve(self):
                self.x, self.intake_reaction, _ = constrained_plate_motion.solve(
                    self, self.intake_constraints['rows'], self.intake_constraints['minimum'])

        options = [dict(plate=0, choice=name) for name in ('ineffective', 'resolving')]
        with patch.object(faulting, 'commit', side_effect=release), patch.object(faulting.pb, 'Balance', Motion):
            trial, event = faulting.rank_trials(world, 1., before, options)
        self.assertEqual(trial.choice, 'resolving')
        self.assertIsNone(world.choice)
        self.assertGreater(event['intake_shortfall_before_cm_yr'], 0.)
        self.assertEqual(event['intake_shortfall_after_cm_yr'], 0.)
        # Removing a conflicting trench would look feasible, but must fail.
        after = Motion(trial, 1.)
        after.intake_constraints = dict(rows=after.intake_constraints['rows'][:1], minimum=np.array([.1]))
        self.assertFalse(faulting.retains_obligations(before, after, 0, 2))

    def test_shortfall_score_is_independent_of_trace_subdivision(self):
        source = dict(rows=np.array([[1., -1.], [-1., 1.]]), minimum=np.full(2, .1),
                      weights_km=np.array([10., 20.]))
        split = dict(rows=np.array([[1., -1.], [1., -1.], [-1., 1.]]), minimum=np.full(3, .1),
                     weights_km=np.array([5., 5., 20.]))
        self.assertAlmostEqual(faulting.intake_shortfall(source), faulting.intake_shortfall(split), places=12)


if __name__ == '__main__':
    unittest.main()
