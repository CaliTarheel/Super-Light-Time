"""Native policy selection and transactional use of the actual tearing forces."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

import force_rifting
import plate_balance as pb
import plate_limit_analysis as limit
from tests.test_moving_hinge_tether import moving_fixture


class ActiveTearingAccountingTests(unittest.TestCase):
    def state(self):
        state = moving_fixture()
        state.trench_shutdown_version = 1
        state.force_rifting_version = force_rifting.VERSION
        state.force_rifting_state = {}
        state.force_rifting_diagnostics = dict(checks=0, commits=0, refusals=[])
        state.config['force_limit_rifting'] = dict(enabled=True)
        state.active = np.ones(2, bool)
        state.t = 0.
        return state

    def test_traction_sum_matches_active_gradient_without_invented_reaction(self):
        state = self.state()
        model = limit.active_balance(state, 1.)
        self.assertTrue(model.slab_tethers)
        # Arbitrary motion makes the expected force NONZERO. An exporter that
        # simply balances its rows to zero would fail this contract.
        x = np.array([3., -2., 4., -1., 6., -5.])
        factor = pb.CM_YR_M_S/pb.RADIUS_M*pb.SECONDS_PER_MYR
        state.omega = x.reshape(2, 3)*factor
        gradient = model._evaluate(x, pb.HUBER_CONTINUATION_KM_MYR[-1]*pb.KM_MYR_CM_YR)[1]
        for plate in model.plates:
            cells, torque, report = limit.plate_loads(state, plate, model)
            expected = -gradient[3*model.slot[plate]:3*model.slot[plate]+3]*pb.RADIUS_M/pb.CM_YR_M_S
            np.testing.assert_allclose(torque[cells].sum(axis=0), expected, rtol=2e-10, atol=1e5)
            self.assertGreater(np.linalg.norm(expected), 0.)
            self.assertFalse(report['artificial_reaction_added'])

    def test_later_rejected_cut_does_not_publish_partial_opening_history(self):
        state = self.state()
        before = deepcopy((state.force_rifting_state, state.force_rifting_diagnostics))
        cells = np.flatnonzero(state.plate == 0)
        accepted = dict(axis=np.array([0., 0., 1.]), cells=cells,
            piece=np.arange(len(cells)) == 0, ratio=.5,
            predicted_opening_cm_yr_mean=0., cut_continental_fraction=0.,
            cut_length_km=100., cut_edges=np.array([0]),
            piece_rotation_rad_myr=[0., 0., 0.], rest_rotation_rad_myr=[0., 0., 0.])
        with patch.object(limit, 'worst_mechanism', side_effect=[accepted, ValueError('unpriced mode')]):
            with self.assertRaisesRegex(ValueError, 'unpriced mode'):
                force_rifting.update(state, .1)
        self.assertEqual((state.force_rifting_state, state.force_rifting_diagnostics), before)

    def row(self, state, plate=0, cell=0):
        cells = np.flatnonzero(state.plate == plate)
        return dict(axis=np.array([0., 0., 1.]), cells=cells,
            piece=np.arange(len(cells)) == cell, ratio=.5,
            predicted_opening_cm_yr_mean=0., cut_continental_fraction=0.,
            cut_length_km=100., cut_edges=np.array([cell]),
            piece_rotation_rad_myr=[0., 0., 0.], rest_rotation_rad_myr=[0., 0., 0.])

    def previous(self, state):
        result = self.row(state)
        return dict(axis=result['axis'].tolist(), piece_cells=[int(result['cells'][0])],
            cut_edges=[0], opened_km=10., paid_cut_work_j=123., since_myr=0.,
            scanned_myr=0., last_opening_work={'cut_work_j': 123.})

    def test_changed_cut_is_researched_intact_and_keeps_retired_paid_path(self):
        state = self.state()
        uid = str(int(state.plate_uid[0]))
        state.force_rifting_state[uid] = self.previous(state)
        state.force_rifting_diagnostics['accepted_cut_work_j'] = 123.
        changed = self.row(state, cell=1)
        with patch.object(force_rifting, '_loaded_plates', return_value=[0]), \
                patch.object(limit, 'worst_mechanism', side_effect=[changed, deepcopy(changed)]) as search:
            force_rifting.update(state, .1)
        self.assertIsNotNone(search.call_args_list[0].kwargs['edge_strength_scale'])
        self.assertIsNone(search.call_args_list[1].kwargs['edge_strength_scale'])
        self.assertEqual(state.force_rifting_state[uid]['opened_km'], 0.)
        self.assertEqual(state.force_rifting_state[uid]['paid_cut_work_j'], 0.)
        self.assertEqual(state.force_rifting_diagnostics['accepted_cut_work_j'], 123.)
        retired = state.force_rifting_diagnostics['retired_candidate_paths']
        self.assertEqual(len(retired), 1)
        self.assertEqual(retired[0]['paid_cut_work_j'], 123.)
        self.assertEqual(len(retired[0]['cut_edges_sha256']), 64)

    def test_healing_and_owner_removal_never_refund_paid_path(self):
        state = self.state()
        uid = str(int(state.plate_uid[0]))
        state.force_rifting_state[uid] = self.previous(state)
        state.force_rifting_diagnostics['accepted_cut_work_j'] = 123.
        with patch.object(force_rifting, '_loaded_plates', return_value=[0]), \
                patch.object(limit, 'worst_mechanism', return_value=self.row(state)):
            force_rifting.update(state, .1)
        self.assertLess(state.force_rifting_state[uid]['opened_km'], 10.)
        self.assertEqual(state.force_rifting_state[uid]['paid_cut_work_j'], 123.)
        self.assertEqual(state.force_rifting_state[uid]['last_opening_work'], {'cut_work_j': 123.})
        with patch.object(force_rifting, '_loaded_plates', return_value=[]):
            force_rifting.update(state, .1)
        self.assertNotIn(uid, state.force_rifting_state)
        self.assertEqual(state.force_rifting_diagnostics['accepted_cut_work_j'], 123.)
        self.assertEqual(state.force_rifting_diagnostics['retired_candidate_paths'][0]['paid_cut_work_j'], 123.)

    def test_small_cut_power_cancellation_rejects_unresolved_work(self):
        state = self.state()
        found = dict(predicted_opening_cm_yr_mean=1., opening_mode_supported=True,
            current_capacity_n=1e12, mode_work=dict(cut_power_w=1.,
                driving_power_w=1e20, resisting_power_w=1e20))
        with self.assertRaisesRegex(ValueError, 'signed remaining=0.0'):
            force_rifting._paid_opening(state, 0, found, 0., force_rifting.normalize(), .1)


if __name__ == '__main__':
    unittest.main()
