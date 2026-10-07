"""Controlled opposed-slab opening, yield and work regressions."""
from copy import deepcopy
import math
import pickle
import unittest
from unittest import mock

import numpy as np

import breakup_mode as opening
import plate_balance as pb
import slab_tether_forces
from tests.test_slab_force_ledger import opposed_fixture


def fixture():
    """Two resolved daughter supports, pulled by opposite slab-edge ports."""
    state = opposed_fixture()
    angle = .3
    for group, sign in ((slice(0, 3), 1.), (slice(3, 6), -1.)):
        points = np.array([[math.cos(sign*angle+offset), math.sin(sign*angle+offset), height]
                           for offset, height in ((-.025, -.04), (0., .04), (.025, 0.))])
        state.xyz[group] = points/np.linalg.norm(points, axis=1)[:, None]
    state.ba = np.array([0, 3])
    model = pb.Balance(state, 1., slab_tethers=True)
    model.solve()
    state.omega = model.rotation()
    cells = np.arange(6)
    piece = np.array([True, True, True, False, False, False])
    axis = np.array([0., 0., 1.])
    cut = dict(edges=np.array([[0, 3]]), mid=np.array([[1., 0., 0.]]),
               length_m=np.array([1e6]), cut=np.array([True]))
    # Each port is uniformly on its own daughter. This independent edge force
    # oracle gives the differential virtual load before any rate solve.
    local = slab_tether_forces.edge_force_ledger(model.tether_assembly, model.x)
    edge_forces = local['net_generalized_forces'][:, :3]
    differential_work = float(edge_forces[0]@(.5*axis)-edge_forces[1]@(.5*axis))
    assert differential_work > 0.
    yield_strength = differential_work/(pb.CM_YR_M_S*1e6)
    return state, model, cells, piece, axis, cut, yield_strength


def run_case(data, factor, **options):
    state, model, cells, piece, axis, cut, reference = data
    strength = tuple(np.array([factor*reference*v]) for v in (1., 3., 1./math.sqrt(3.)))
    return opening.solve(state, 0, cells, piece, axis, cut, strength, model, **options)


class BreakupModeTests(unittest.TestCase):
    def test_certified_strong_zero_matches_forced_full_model_and_saved_increment(self):
        data = fixture()
        fast = run_case(data, 2.)
        full = run_case(data, 2., allow_zero_fast_path=False)
        self.assertTrue(fast['zero_rate_fast_path'])
        self.assertFalse(full['zero_rate_fast_path'])
        self.assertEqual(fast['mode_solver']['method'], 'certified_zero_rate_kkt')
        self.assertGreater(fast['zero_rate_certificate']['certified_positive_margin_w'], 0.)
        self.assertGreater(fast['zero_rate_certificate']['uncertainty_bound_w'], 0.)
        self.assertEqual(fast['relative_rotation_rad_myr'], full['relative_rotation_rad_myr'])
        for name in ('piece_rotation_rad_myr', 'rest_rotation_rad_myr'):
            np.testing.assert_allclose(fast[name], full[name], rtol=0., atol=1e-14)
            np.testing.assert_allclose(fast[name], 0., rtol=0., atol=1e-14)
        self.assertLess(fast['mode_work']['closure_relative'], 1e-8)

    def test_nonequilibrium_common_motion_refuses_zero_shortcut_and_resolves_full(self):
        data = fixture()
        data[1].x[2] += 1.
        result = run_case(data, 2.)
        self.assertTrue(result['opening_mode_supported'], result.get('opening_mode_reason'))
        self.assertFalse(result['zero_rate_fast_path'])
        self.assertIn('not an accepted intact equilibrium', result['zero_rate_fast_path_reason'])
        self.assertEqual(result['mode_solver']['method'], 'full_projected_convex_solve')
        self.assertEqual(result['relative_rotation_rad_myr'], 0.)

    def test_near_yield_keeps_full_solve_with_a_strict_uncertainty_margin(self):
        data = fixture()
        strong = run_case(data, 2.)
        proof = strong['zero_rate_certificate']
        # Place the positive yield margin *inside* the conservative uncertainty
        # band, without changing the physical load or prescribing a velocity.
        base = data[-1]*1e6*pb.CM_YR_M_S
        factor = (proof['differential_drive_w']+.25*proof['uncertainty_bound_w'])/base
        near = run_case(data, factor)
        self.assertTrue(near['opening_mode_supported'], near.get('opening_mode_reason'))
        self.assertFalse(near['zero_rate_fast_path'])
        self.assertIn('uncertainty', near['zero_rate_fast_path_reason'])
        self.assertEqual(near['mode_solver']['method'], 'full_projected_convex_solve')

    def test_motion_provenance_mismatch_refuses_shortcut_even_with_valid_operators(self):
        import balance_force_ledger
        data = fixture()
        ledger = balance_force_ledger.export(data[1], data[1].x)
        ledger['motion_coordinates'][0] += 1.
        result = run_case(data, 2., ledger=ledger)
        self.assertFalse(result['zero_rate_fast_path'])
        self.assertIn('does not exactly match', result['zero_rate_fast_path_reason'])
        self.assertEqual(result['mode_solver']['method'], 'full_projected_convex_solve')

    def test_falsified_common_matrix_refuses_zero_and_rejects_full_restriction(self):
        import balance_force_ledger
        data = fixture()
        ledger = balance_force_ledger.export(data[1], data[1].x)
        term = next(term for term in ledger['local_terms']
                    if term.shape == 'quadratic' and term.kind == 'attached_slab')
        term.matrix = 2.*term.matrix
        result = run_case(data, 2., ledger=ledger)
        self.assertFalse(result['opening_mode_supported'])
        self.assertIn('Virtual common restriction differs', result['opening_mode_reason'])
        certificate, reason = opening._zero_certificate(data[0], data[1], ledger,
            0, data[2], data[3], data[4], 2.*data[-1]*1e6*pb.CM_YR_M_S)
        self.assertIsNone(certificate)
        self.assertIn('common quadratic differs', reason)

    def test_near_singular_metric_is_rejected_and_unresolved_metric_uses_full_response(self):
        # Equilibration cannot hide a nearly free correlated mode: this matrix
        # has unit diagonal and an eigenvalue far below the numerical threshold.
        nearly_singular = np.array([[1., 1.-1e-12], [1.-1e-12, 1.]])
        with self.assertRaisesRegex(ValueError, 'ill-conditioned or unresolved'):
            opening._metric_inverse(nearly_singular)
        data = fixture()
        full = run_case(data, 2., allow_zero_fast_path=False)
        # Tightening the *numerical qualification* threshold must route this
        # unchanged physical fixture through its full response, never a zero
        # result inferred from an unqualified inverse or extra regularization.
        with mock.patch.object(opening, '_MAX_METRIC_CONDITION', 1.):
            guarded = run_case(data, 2.)
        self.assertTrue(guarded['opening_mode_supported'], guarded.get('opening_mode_reason'))
        self.assertFalse(guarded['zero_rate_fast_path'])
        self.assertIn('ill-conditioned or unresolved', guarded['zero_rate_fast_path_reason'])
        self.assertEqual(guarded['relative_rotation_rad_myr'], full['relative_rotation_rad_myr'])
        np.testing.assert_allclose(guarded['piece_rotation_rad_myr'],
                                   full['piece_rotation_rad_myr'], rtol=0., atol=1e-14)

    def test_inverse_metric_bound_is_inflated_and_uniform_port_guard_is_strict(self):
        import balance_force_ledger
        matrix = np.array([[3., .8], [.8, 2.]])
        bound, _, diagnostic = opening._metric_inverse(matrix)
        self.assertGreater(diagnostic['inverse_metric_inflation'], 1.)
        self.assertGreater(diagnostic['inverse_residual_numerical_allowance'], 0.)
        self.assertGreaterEqual(np.linalg.eigvalsh(bound-np.linalg.inv(matrix))[0], -1e-15)
        data = fixture()
        ledger = balance_force_ledger.export(data[1], data[1].x)
        port = ledger['local_terms'][0].ports[0]
        port.matrices = port.matrices*(1.+1e-8)
        certificate, reason = opening._zero_certificate(data[0], data[1], ledger,
            0, data[2], data[3], data[4], 2.*data[-1]*1e6*pb.CM_YR_M_S)
        self.assertIsNone(certificate)
        self.assertIn('does not preserve uniform motion', reason)

    def test_opposed_slab_strong_cut_stays_exactly_stopped_weak_cut_opens(self):
        data = fixture()
        state, model = data[:2]
        np.testing.assert_array_equal(model.tether_assembly['down_drive'][:3], 0.)
        self.assertLess(np.linalg.norm(model.x[:3]), 1e-10)
        strong, weak = run_case(data, 2.), run_case(data, .5)
        self.assertTrue(strong['opening_mode_supported'], strong.get('opening_mode_reason'))
        self.assertTrue(weak['opening_mode_supported'], weak.get('opening_mode_reason'))
        self.assertEqual(strong['relative_rotation_rad_myr'], 0.)
        self.assertEqual(strong['predicted_opening_cm_yr_mean'], 0.)
        self.assertGreaterEqual(strong['mode_solver']['opening_bound_gradient_w'], 0.)
        self.assertGreater(weak['relative_rotation_rad_myr'], 0.)
        self.assertFalse(weak['zero_rate_fast_path'])
        self.assertGreater(weak['predicted_opening_cm_yr_mean'], 0.)
        self.assertLess(weak['mode_solver']['force_relative_residual'], pb.FORCE_RELATIVE_TOLERANCE)
        # Slab/anchor/contact response is retained while opening. This is not
        # the basal-only excess-force estimate from the historical rift rate.
        self.assertLess(weak['mode_work']['closure_relative'], 1e-8)
        self.assertGreater(weak['mode_work']['cut_power_w'], 0.)
        self.assertLess(weak['common_restriction_relative_error'], 1e-9)
        self.assertLess(weak['differential_virtual_work_relative_error'], 1e-9)

    def test_weakening_monotonically_increases_solved_rate_and_closes_power(self):
        data = fixture()
        rates = []
        for factor in (.8, .5, .2, 0.):
            result = run_case(data, factor)
            self.assertTrue(result['opening_mode_supported'], result.get('opening_mode_reason'))
            rates.append(result['predicted_opening_cm_yr_mean'])
            work = result['mode_work']
            self.assertAlmostEqual((work['driving_power_w']-work['resisting_power_w'])/
                max(work['driving_power_w'], 1.), work['cut_power_w']/max(work['driving_power_w'], 1.), places=8)
            self.assertGreaterEqual(result['mode_passivity']['minimum_local_resisting_work_w'], -1e-4)
        self.assertTrue(all(a < b for a, b in zip(rates, rates[1:])))

    def test_finite_rate_retains_slab_and_boundary_drag_instead_of_basal_only_excess(self):
        data = fixture()
        state, _, cells, piece, axis, cut, reference = data
        result = run_case(data, .5)
        strengths = tuple(np.array([.5*reference*v]) for v in (1., 3., 1./math.sqrt(3.)))
        capacity = opening._cut(state, cells, piece, axis, cut, strengths)['yield_w']
        # Independent basal work of the balanced +/-axis/2 daughter field.
        # Holding the accepted local dead loads and omitting slab/interface
        # response would predict this vastly larger unsupported rate.
        metric = np.linalg.norm(np.cross(axis, state.xyz[cells]), axis=1)**2
        basal = float(np.sum(pb.ASTHENOSPHERE_DRAG_PA_S_M*state.cell_area[cells]*1e6
                             *metric*.25*pb.CM_YR_M_S**2))
        dead_load_rate = (result['differential_drive_at_rigid_w']-capacity)/basal
        self.assertGreater(dead_load_rate, 1000.*result['predicted_opening_cm_yr_mean'])
        self.assertGreater(result['predicted_opening_cm_yr_mean'], 0.)

    def test_source_and_saved_motion_are_unchanged_and_daughter_increments_are_consistent(self):
        data = fixture()
        state, model = data[:2]
        before = pickle.dumps(state, protocol=5)
        old_motion = model.x.copy()
        result = run_case(data, .5)
        self.assertTrue(result['opening_mode_supported'], result.get('opening_mode_reason'))
        self.assertEqual(before, pickle.dumps(state, protocol=5))
        np.testing.assert_array_equal(model.x, old_motion)
        piece = np.asarray(result['piece_rotation_rad_myr'])
        rest = np.asarray(result['rest_rotation_rad_myr'])
        np.testing.assert_allclose(piece-rest, data[4]*result['relative_rotation_rad_myr'], rtol=1e-12, atol=1e-15)
        # Additions are relative to the saved source parent omega; the two
        # resolved daughters separate rather than receiving an imposed speed.
        self.assertGreater(np.dot(piece-rest, data[4]), 0.)

    def test_detached_inventory_supplies_no_ghost_daughter_drive(self):
        data = list(fixture())
        state = data[0]
        state.trench_id[:] = 0
        model = pb.Balance(state, 1., slab_tethers=True)
        model.solve()
        state.omega = model.rotation()
        data[1] = model
        result = run_case(data, .5)
        self.assertTrue(result['opening_mode_supported'], result.get('opening_mode_reason'))
        self.assertEqual(result['differential_drive_at_rigid_w'], 0.)
        self.assertEqual(result['relative_rotation_rad_myr'], 0.)

    def test_scalar_capacity_counts_tension_compression_and_shear_once(self):
        state, model, cells, piece, axis, cut, strength = fixture()
        mixed_axis = np.array([0., 1., 1.])/math.sqrt(2.)
        capacities = tuple(np.array([v]) for v in (10., 30., 4.))
        mode = opening._cut(state, cells, piece, mixed_axis, cut, capacities)
        speed = mode['mean_speed']
        expected = 1e6*(10.*max(mode['normal'][0], 0.)
                        +30.*max(-mode['normal'][0], 0.)+4.*abs(mode['shear'][0]))
        self.assertAlmostEqual(mode['capacity_n']*speed, expected)
        self.assertAlmostEqual(mode['yield_w'], expected*pb.CM_YR_M_S)
        # Reversing the axis changes opening into compression, preserving the
        # declared mixed cut strengths rather than replacing them by tension.
        reverse = opening._cut(state, cells, piece, -mixed_axis, cut, capacities)
        self.assertGreater(reverse['capacity_n'], mode['capacity_n'])

    def test_unsupported_historical_balance_has_no_basal_fallback(self):
        data = list(fixture())
        historical = pb.Balance(data[0], 1., slab_tethers=False)
        historical.solve()
        data[1] = historical
        result = run_case(data, .5)
        self.assertFalse(result['opening_mode_supported'])
        self.assertIn('no basal-only fallback', result['opening_mode_reason'])
        self.assertEqual(result['relative_rotation_rad_myr'], 0.)

    def test_invalid_partition_is_rejected_without_source_mutation(self):
        state, model, cells, piece, axis, cut, reference = fixture()
        before = pickle.dumps(state, protocol=5)
        strength = tuple(np.ones(1) for _ in range(3))
        for invalid in (np.ones(6, bool), np.zeros(6, bool)):
            with self.assertRaises(ValueError):
                opening.solve(state, 0, cells, invalid, axis, cut, strength, model)
        bad_cut = deepcopy(cut)
        bad_cut['edges'] = np.array([[0, 1]])
        with self.assertRaises(ValueError):
            opening.solve(state, 0, cells, piece, axis, bad_cut, strength, model)
        self.assertEqual(before, pickle.dumps(state, protocol=5))


if __name__ == '__main__':
    unittest.main()
