"""Actual finite laws, independently assembled dense parity and sparse solves."""
from copy import deepcopy
import unittest

import numpy as np

from benchmarks.collision_architecture import joint_resistance as dense_laws
from benchmarks.collision_architecture import local_joint_resistance as local
from benchmarks.collision_architecture import shared_contact as dense
from benchmarks.collision_architecture import sparse_shared_contact as sparse
from benchmarks.collision_architecture import sparse_joint_solver as solver
from tests.test_joint_resistance import fixture, unit


def arc_parameters(**updates):
    result = dict(first_face=0, second_face=1, basis=[1, 0, 0], tangent=[0, 0, 1],
        outward_normal=[0, -1, 0], start=-.06, end=.06, capacity_n_per_m=1e13, measure_weight=.5)
    result.update(updates)
    return result


def patch_parameters(**updates):
    result = dict(first_face=0, second_face=1,
        polygon=unit([[1, .02, -.02], [1, .08, -.02], [1, .05, .03]]),
        viscosity_pa_s=1e20, thickness_m=13200.)
    result.update(updates)
    return result


def resistances(system):
    geometry = local.Geometry(system)
    compact = local.Resistance(system, welds=[geometry.weld_arc(**arc_parameters())],
        interfaces=[geometry.interface_patch(**patch_parameters())], smoothing_speed_m_s=2e-9)
    reference = dense_laws.Resistance(system, welds=[dense_laws.weld_arc(system, **arc_parameters())],
        interfaces=[dense_laws.interface_patch(system, **patch_parameters())], smoothing_speed_m_s=2e-9)
    return compact, reference


def sparse_fixture(copies=1):
    original = fixture()
    p = np.tile(original['points'], (copies, 1))
    f = np.concatenate([original['faces']+6*j for j in range(copies)])
    return sparse.assemble(p, f, np.tile(original['vertex_plate'], copies), radius_m=6371e3,
        basal_drag_pa_s_per_m=1e15, viscosity_pa_s=2e21, sheet_thickness_m=4e4,
        basal_reference_velocity_m_s=np.zeros_like(p), other_plate_rotational_drag_n_m_s=np.eye(6)*1e39,
        external_torques_n_m=np.array([[0, 0, 1e25], [0, 0, -1e25]]),
        external_nodal_forces_n=np.zeros_like(p), contacts=[])


class LocalJointResistance(unittest.TestCase):
    def test_actual_laws_dense_parity_including_hessian_actions(self):
        for same in (False, True):
            system = fixture(same_owner=same); compact, reference = resistances(system)
            rng = np.random.default_rng(520)
            for _ in range(5):
                y = rng.normal(size=len(system['load_n']))*1e-9
                a, b = compact.evaluate(y), reference.evaluate(y)
                for key in ('potential_w', 'resisting_power_w'):
                    self.assertAlmostEqual(a[key]/b[key], 1., places=11)
                np.testing.assert_allclose(a['gradient_n'], b['gradient_n'], rtol=2e-10, atol=500.)
                direction = rng.normal(size=len(y))*1e-9
                np.testing.assert_allclose(a['hessian_n_s_m']@direction,
                    b['hessian_n_s_m']@direction, rtol=2e-10, atol=500.)
                np.testing.assert_allclose(a['diagonal_n_s_m'], np.diag(b['hessian_n_s_m']), rtol=2e-12)
                self.assertTrue(np.all(abs(a['gradient_n']) <= a['gradient_absolute_scale_n']*(1+1e-14)))
                action = a['hessian_n_s_m']@direction
                bound = a['hessian_n_s_m'].absolute_matvec(direction)
                self.assertTrue(np.all(abs(action) <= bound*(1+1e-14)))

    def test_opening_cannot_cancel_closing_and_partition_is_invariant(self):
        system = fixture(); geometry = local.Geometry(system)
        a = local.Resistance(system, welds=[geometry.weld_arc(**arc_parameters())], smoothing_speed_m_s=1e-10)
        b = local.Resistance(system, welds=[geometry.weld_arc(**arc_parameters(end=.01)),
            geometry.weld_arc(**arc_parameters(start=.01))], smoothing_speed_m_s=1e-10)
        y = np.zeros(len(system['load_n'])); y[0] = 1e-8
        whole, parts = a.evaluate(y), b.evaluate(y)
        self.assertGreater(whole['resisting_power_w'], 0.)
        self.assertAlmostEqual(whole['resisting_power_w']/parts['resisting_power_w'], 1., places=12)
        np.testing.assert_allclose(whole['gradient_n'], parts['gradient_n'], rtol=2e-12, atol=10.)

    def test_same_owner_residual_shear_and_common_spin_null_work(self):
        system = fixture(same_owner=True); compact, _ = resistances(system)
        y = np.zeros(len(system['load_n'])); y[3:] = np.arange(len(y)-3)*1e-10
        a = compact.evaluate(y)
        self.assertGreater(a['resisting_power_w'], 0.)
        np.testing.assert_array_equal(a['gradient_n'][:3], 0.)
        y[:] = 0.; y[:3] = [1e-9, 2e-9, -3e-9]
        self.assertEqual(compact.evaluate(y)['resisting_power_w'], 0.)

    def test_real_sparse_nonlinear_solution_matches_dense_reference(self):
        system = sparse_fixture(); compact, _ = resistances(system)
        reference_system = fixture(); _, reference = resistances(reference_system)
        expected = dense.solve(reference_system, resistance=reference)
        result = solver.solve(system, resistance=compact)
        np.testing.assert_allclose(result['generalized_velocity_m_s'], expected['generalized_velocity_m_s'],
                                   rtol=2e-7, atol=1e-17)
        d = result['diagnostics']
        self.assertLessEqual(np.max(d['componentwise_stationarity_relative']), 1e-10)
        self.assertLessEqual(abs(d['physical_power_residual_w']), d['physical_power_acceptance_allowance_w'])
        self.assertGreater(d['resistance_dissipation_w'], 0.)

    def test_local_storage_independent_of_global_dof_count(self):
        system = sparse_fixture(copies=80); geometry = local.Geometry(system)
        a = geometry.weld_arc(**arc_parameters()); b = geometry.interface_patch(**patch_parameters())
        self.assertGreater(len(system['load_n']), 900)
        self.assertEqual(a.operator.shape, (2, 18)); self.assertEqual(b.factor.shape, (18, 18))
        compact = local.Resistance(system, welds=[a], interfaces=[b], smoothing_speed_m_s=1e-9)
        y = np.random.default_rng(84).normal(size=len(system['load_n']))*1e-9
        result = compact.evaluate(y)
        support = np.union1d(a.columns, b.columns)
        outside = np.setdiff1d(np.arange(len(y)), support)
        np.testing.assert_array_equal(result['gradient_n'][outside], 0.)
        np.testing.assert_array_equal((result['hessian_n_s_m']@y)[outside], 0.)

    def test_full_mesh_internal_edge_cannot_be_mistaken_for_free_edge(self):
        system = fixture(); system['faces'] = np.vstack([system['faces'], [0, 2, 1]])
        with self.assertRaisesRegex(ValueError, 'internal material'):
            local.Geometry(system).weld_arc(**arc_parameters())

    def test_geometry_binding_and_detached_inputs(self):
        system = fixture(); geometry = local.Geometry(system)
        row = geometry.weld_arc(**arc_parameters())
        resistance = local.Resistance(system, welds=[row], smoothing_speed_m_s=1e-10)
        y = np.ones(len(system['load_n']))*1e-9
        expected = resistance.evaluate(y)['gradient_n'].copy()
        row.operator.flags.writeable = True; row.operator[:] = 0.
        np.testing.assert_array_equal(resistance.evaluate(y)['gradient_n'], expected)
        system['points'][0] = unit([1, 0, -.09])
        with self.assertRaisesRegex(ValueError, 'different'):
            resistance.validate_system(system)
        with self.assertRaisesRegex(ValueError, 'different'):
            local.Resistance(system, welds=[row], smoothing_speed_m_s=1e-10)


if __name__ == '__main__':
    unittest.main()
