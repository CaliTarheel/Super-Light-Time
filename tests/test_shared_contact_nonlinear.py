"""Independent manufactured nonlinear joint variational problems."""
import unittest

import numpy as np

from benchmarks.collision_architecture import shared_contact as joint
from tests.test_shared_contact import fixture, relative, unit


class QuarticPotential:
    """Passive power law on one physical relative-velocity trace, in SI."""
    def __init__(self, row, coefficient):
        self.row = row.copy(); self.coefficient = coefficient

    def evaluate(self, y):
        rate = float(self.row@y); coefficient = self.coefficient
        gradient = coefficient*rate**3*self.row
        factor = (np.sqrt(3*coefficient)*abs(rate)*self.row)[None]
        return dict(potential_w=.25*coefficient*rate**4,
            gradient_n=gradient, hessian_n_s_m=factor.T@factor,
            hessian_factor_sqrt_n_s_m=factor, resisting_power_w=coefficient*rate**4,
            diagnostics=dict(law='manufactured quartic', rate_m_s=rate))


class TestNonlinearSharedContact(unittest.TestCase):
    def test_original_kkt_refines_symmetry_zero_components_without_tolerance_floor(self):
        cfg = fixture(contacts=False)
        cfg['points'] = unit([[1, 0, -.1], [1, .2, 0], [1, 0, .1],
            [1, -.2, -.2], [1, .3, -.15], [1, 0, .3]])
        cfg['other_plate_rotational_drag_n_m_s'] = np.eye(6)*1e39
        cfg['viscosity_pa_s'][:] = 2e21
        cfg['sheet_thickness_m'][:] = 4e4
        result = joint.solve(joint.assemble(**cfg))
        d = result['diagnostics']
        self.assertGreater(d['original_kkt_refinement_steps'], 0)
        self.assertLess(np.max(d['componentwise_stationarity_relative']), 1e-10)
        # No absolute force floor: the symmetry-zero rows satisfy the same
        # componentwise backward criterion as the strongly driven rows.
        self.assertLess(d['joint_relative_residual'], 1e-14)

    def test_scalar_reduction_has_independent_cubic_solution(self):
        cfg = fixture(contacts=False)
        cfg['viscosity_pa_s'][:] = 0.
        system = joint.assemble(**cfg)
        # Choose a physical paired trace, then solve its full constrained linear
        # compliance independently. Cubic stationarity is p + c*J*p^3 = p0.
        contact_system = joint.assemble(**fixture())
        row = contact_system['contact_matrix'][0]
        baseline = joint.solve(system)['generalized_velocity_m_s']
        k = system['hessian_n_s_m']; g = system['gauge_matrix']
        block = np.block([[k, g.T], [g, np.zeros((len(g), len(g)))]])
        response = np.linalg.solve(block, np.r_[row, np.zeros(len(g))])[:len(k)]
        compliance = float(row@response); p0 = float(row@baseline)
        coefficient = 4/(compliance*p0*p0)
        potential = QuarticPotential(row, coefficient)
        result = joint.solve(system, resistance=potential)
        lo, hi = sorted((0., p0))
        for _ in range(100):
            p = .5*(lo+hi)
            if p+coefficient*compliance*p**3-p0 > 0: hi = p
            else: lo = p
        expected_rate = .5*(lo+hi)
        expected = baseline-coefficient*expected_rate**3*response
        self.assertLess(relative(result['generalized_velocity_m_s'], expected), 2e-10)
        d = result['diagnostics']; self.assertGreater(d['nonlinear_iterations'], 0)
        self.assertAlmostEqual(d['resistance_dissipation_w']/d['resistance_potential_w'], 4., places=12)
        self.assertLess(abs(d['physical_power_residual_w'])/abs(d['external_power_w']), 2e-11)
        self.assertLess(relative(result['resistance_generalized_reaction_n'],
            -potential.evaluate(result['generalized_velocity_m_s'])['gradient_n']), 1e-15)

    def test_nonlinear_potential_gradient_matches_finite_difference(self):
        cfg = fixture(contacts=False); system = joint.assemble(**cfg)
        row = joint.assemble(**fixture())['contact_matrix'][0]
        law = QuarticPotential(row, 1e44)
        rng = np.random.default_rng(731); y = rng.normal(size=len(row))*1e-9
        d = rng.normal(size=len(row))*1e-9; h = 1e-5
        value, gradient = joint.evaluate(system, y, resistance=law)
        self.assertTrue(np.isfinite(value))
        numerical = (joint.evaluate(system, y+h*d, resistance=law)[0]-
            joint.evaluate(system, y-h*d, resistance=law)[0])/(2*h)
        self.assertLess(abs(numerical-gradient@d)/abs(gradient@d), 2e-8)
        derivative = (joint.evaluate(system, y+h*d, resistance=law)[1]-
            joint.evaluate(system, y-h*d, resistance=law)[1])/(2*h)
        self.assertLess(relative(derivative, (system['hessian_n_s_m']+law.evaluate(y)['hessian_n_s_m'])@d), 2e-8)

    def test_nonlinear_solution_preserves_existing_contact_rows(self):
        system = joint.assemble(**fixture())
        # Different tangent component than the prescribed contact normal.
        trace = system['contact_velocity_maps'][0, 0]-system['contact_velocity_maps'][0, 1]
        direction = np.cross(system['contact_point'][0], system['contact_normal'][0])
        law = QuarticPotential(direction@trace, 1e45)
        result = joint.solve(system, resistance=law)
        self.assertGreater(result['diagnostics']['resistance_dissipation_w'], 0.)
        self.assertLess(np.max(np.abs(result['diagnostics']['contact_slip_m_s'])), 1e-22)
        self.assertLess(result['diagnostics']['joint_relative_residual'], 1e-10)

    def test_wrong_power_negative_curvature_and_inconsistent_factor_rejected(self):
        system = joint.assemble(**fixture(contacts=False)); size = len(system['load_n'])
        row = np.ones(size); good = QuarticPotential(row, 1e40)
        class Broken:
            def __init__(self, kind): self.kind = kind
            def evaluate(self, y):
                result = good.evaluate(y)
                if self.kind == 'power': result['resisting_power_w'] *= .5
                elif self.kind == 'factor': result['hessian_factor_sqrt_n_s_m'] *= 2
                else:
                    result.pop('hessian_factor_sqrt_n_s_m')
                    result['hessian_n_s_m'] = np.diag([1e60, -1e40]+[1e60]*(size-2))
                return result
        for kind in ('power', 'factor', 'negative'):
            with self.assertRaises(ValueError, msg=kind):
                joint.solve(system, resistance=Broken(kind))

    def test_iteration_exhaustion_cannot_return_unbalanced_velocity(self):
        system = joint.assemble(**fixture(contacts=False))
        row = joint.assemble(**fixture())['contact_matrix'][0]
        with self.assertRaisesRegex(RuntimeError, 'iteration budget'):
            joint.solve(system, resistance=QuarticPotential(row, 1e48), maximum_newton_iterations=1)

    def test_callback_cannot_mutate_trial_velocity(self):
        system = joint.assemble(**fixture(contacts=False))
        class Mutator:
            def evaluate(self, y):
                y[0] = 0.
        with self.assertRaises(ValueError):
            joint.solve(system, resistance=Mutator())

    def test_callback_system_binding_is_checked_for_same_size_geometry(self):
        system = joint.assemble(**fixture(contacts=False))
        class Bound:
            def validate_system(self, supplied):
                if supplied is not system:
                    raise ValueError('Callback bound to a different system.')
            def evaluate(self, y):
                return QuarticPotential(np.ones(len(y)), 1e40).evaluate(y)
        different = joint.assemble(**fixture(contacts=False))
        with self.assertRaisesRegex(ValueError, 'different'):
            joint.solve(different, resistance=Bound())
        with self.assertRaisesRegex(ValueError, 'different'):
            joint.evaluate(different, np.zeros(len(different['load_n'])), resistance=Bound())


if __name__ == '__main__':
    unittest.main()
