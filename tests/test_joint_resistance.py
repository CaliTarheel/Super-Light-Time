"""Finite total-velocity weld/shear laws: independent power and geometry checks."""
from copy import deepcopy
import unittest

import numpy as np

import collision_interface
import weld_geometry
from benchmarks.collision_architecture import shared_contact as joint
from benchmarks.collision_architecture import joint_resistance as laws


def unit(p):
    p = np.asarray(p, float)
    return p/np.linalg.norm(p, axis=-1, keepdims=True)


def fixture(*, same_owner=False):
    p = unit([[1, 0, -.1], [1, .2, 0], [1, 0, .1],
              [1, -.2, -.2], [1, .3, -.15], [1, 0, .3]])
    owner = np.zeros(6, int) if same_owner else np.array([0, 0, 0, 1, 1, 1])
    count = int(owner.max())+1
    return joint.assemble(p, np.array([[0, 1, 2], [3, 4, 5]]), owner,
        radius_m=6371e3, basal_drag_pa_s_per_m=1e15, viscosity_pa_s=2e21,
        sheet_thickness_m=4e4, basal_reference_velocity_m_s=np.zeros_like(p),
        other_plate_rotational_drag_n_m_s=np.eye(3*count)*1e39,
        external_torques_n_m=np.array([[0, 0, 1e25], [0, 0, -1e25]])[:count],
        external_nodal_forces_n=np.zeros_like(p), contacts=[])


def arc(system, **changes):
    args = dict(first_face=0, second_face=1, basis=[1, 0, 0], tangent=[0, 0, 1],
        outward_normal=[0, -1, 0], start=-.06, end=.06,
        capacity_n_per_m=1e13, measure_weight=.5)
    args.update(changes)
    return laws.weld_arc(system, **args)


def patch(system, **changes):
    args = dict(first_face=0, second_face=1,
        polygon=unit([[1, .02, -.02], [1, .08, -.02], [1, .05, .03]]),
        viscosity_pa_s=1e20, thickness_m=13200.)
    args.update(changes)
    return laws.interface_patch(system, **args)


class JointResistanceTests(unittest.TestCase):
    def test_weld_matches_actual_rigid_law_in_absolute_si_units(self):
        system = fixture(); row = arc(system)
        resistance = laws.Resistance(system, welds=[row], smoothing_speed_m_s=1e-10)
        y = np.r_[np.array([.3, -.2, .8, -.1, .2, -.3])*1e-9, np.zeros(12)]
        reference, gradient, hessian, _ = weld_geometry.integrate(row['operator'][:, :6], y[:6], -.06, .06, 1e-10)
        expected_coefficient = 1e13*.5*6371e3
        result = resistance.evaluate(y)
        self.assertAlmostEqual(result['potential_w']/(expected_coefficient*reference), 1., places=13)
        np.testing.assert_allclose(result['gradient_n'][:6], expected_coefficient*gradient, rtol=1e-13, atol=1.)
        np.testing.assert_allclose(result['hessian_n_s_m'][:6, :6], expected_coefficient*hessian, rtol=1e-13, atol=1.)
        self.assertAlmostEqual(result['resisting_power_w']/float(result['gradient_n']@y), 1., places=13)

    def test_weld_retains_local_peeling_when_signed_mean_closure_is_zero(self):
        system = fixture(); row = arc(system)
        y = np.zeros(len(system['load_n'])); y[0] = 1e-8
        # Rotation about x gives antisymmetric +/- sin(theta) normal rates.
        self.assertLess(abs(weld_geometry.moments(-.06, .06)[0]@(row['operator']@y)), 1e-25)
        result = laws.Resistance(system, welds=[row], smoothing_speed_m_s=1e-10).evaluate(y)
        self.assertGreater(result['resisting_power_w'], 0.)
        self.assertGreater(result['potential_w'], 0.)

    def test_free_edge_closure_has_zero_weld_traction_and_opening_has_exact_power(self):
        system = fixture(); resistance = laws.Resistance(system, welds=[arc(system)], smoothing_speed_m_s=1e-10)
        y = np.zeros(len(system['load_n'])); speed = 1e-9; y[2] = speed
        opening = resistance.evaluate(y)
        # q=-speed*cos(theta): the whole arc is beyond the Huber yield width.
        coefficient = 1e13*.5*6371e3
        expected_power = coefficient*2*speed*np.sin(.06)
        expected_potential = coefficient*(2*speed*np.sin(.06)-1e-10*.06)
        self.assertAlmostEqual(opening['resisting_power_w']/expected_power, 1., places=12)
        self.assertAlmostEqual(opening['potential_w']/expected_potential, 1., places=12)
        closing = resistance.evaluate(-y)
        self.assertEqual(closing['potential_w'], 0.)
        np.testing.assert_array_equal(closing['gradient_n'], 0.)

    def test_weld_factor_and_derivatives_include_residual_velocity(self):
        system = fixture(); resistance = laws.Resistance(system, welds=[arc(system)], smoothing_speed_m_s=2e-9)
        rng = np.random.default_rng(972); y = rng.normal(size=len(system['load_n']))*1e-9
        direction = rng.normal(size=len(y))*1e-9
        result = resistance.evaluate(y); step = 1e-5
        high, low = resistance.evaluate(y+step*direction), resistance.evaluate(y-step*direction)
        difference = (high['potential_w']-low['potential_w'])/(2*step)
        self.assertLess(abs(difference-result['gradient_n']@direction)/max(abs(difference), 1.), 2e-9)
        factor = result['hessian_factor_sqrt_n_s_m']
        reference = np.abs(factor).T@np.abs(factor)
        self.assertTrue(np.all(np.abs(factor.T@factor-result['hessian_n_s_m']) <= 2e-13*reference+1e-30))

    def test_opening_and_closing_are_not_cancelled_across_disconnected_arc_pieces(self):
        system = fixture(); whole = arc(system)
        split = [arc(system, start=-.06, end=.01), arc(system, start=.01, end=.06)]
        y = np.random.default_rng(234).normal(size=len(system['load_n']))*1e-9
        a = laws.Resistance(system, welds=[whole], smoothing_speed_m_s=2e-10).evaluate(y)
        b = laws.Resistance(system, welds=split, smoothing_speed_m_s=2e-10).evaluate(y)
        self.assertAlmostEqual(a['potential_w']/b['potential_w'], 1., places=12)
        np.testing.assert_allclose(a['gradient_n'], b['gradient_n'], rtol=3e-12, atol=10.)

    def test_interface_rigid_power_matches_analytic_polygon_metric(self):
        system = fixture(); row = patch(system)
        resistance = laws.Resistance(system, interfaces=[row], smoothing_speed_m_s=1e-10)
        a = np.array([.3, -.2, .8])*1e-9
        y = np.r_[a, np.zeros(len(system['load_n'])-3)]
        polygon = unit([[1, .02, -.02], [1, .08, -.02], [1, .05, .03]])
        expected = (1e20/13200.)*float(a@collision_interface.rotation_metric(polygon, 6371.)@a)
        self.assertAlmostEqual(resistance.evaluate(y)['resisting_power_w']/expected, 1., places=10)
        self.assertLess(row['quadrature'][-1]['rigid_metric_relative_error'], 1e-10)

    def test_same_owner_sheets_can_still_have_positive_shear_work(self):
        system = fixture(same_owner=True); row = patch(system)
        y = np.zeros(len(system['load_n'])); y[3:9] = np.arange(6)*1e-10
        result = laws.Resistance(system, interfaces=[row], smoothing_speed_m_s=1e-10).evaluate(y)
        self.assertGreater(result['resisting_power_w'], 0.)
        np.testing.assert_array_equal(result['gradient_n'][:3], 0.)

    def test_interface_reactions_have_zero_pair_torque_and_common_spin_work(self):
        system = fixture(); resistance = laws.Resistance(system, interfaces=[patch(system)], smoothing_speed_m_s=1e-10)
        y = np.random.default_rng(722).normal(size=len(system['load_n']))*1e-9
        result = resistance.evaluate(y)
        np.testing.assert_allclose(result['gradient_n'][:3]+result['gradient_n'][3:6], 0., atol=3e3)
        common = np.r_[np.array([1., 2., -.5])*1e-9, np.array([1., 2., -.5])*1e-9, np.zeros(12)]
        self.assertLess(resistance.evaluate(common)['resisting_power_w'], 1e-20*result['resisting_power_w'])

    def test_interface_polygon_subdivision_preserves_nonrigid_work_and_gradient(self):
        system = fixture(); polygon = unit([[1, .02, -.02], [1, .08, -.02], [1, .05, .03]])
        center = unit(polygon.sum(axis=0))
        whole = laws.Resistance(system, interfaces=[patch(system, polygon=polygon)], smoothing_speed_m_s=1e-10)
        pieces = [patch(system, polygon=np.array([a, b, center])) for a, b in zip(polygon, np.roll(polygon, -1, axis=0))]
        divided = laws.Resistance(system, interfaces=pieces, smoothing_speed_m_s=1e-10)
        y = np.random.default_rng(833).normal(size=len(system['load_n']))*1e-9
        a, b = whole.evaluate(y), divided.evaluate(y)
        self.assertAlmostEqual(a['resisting_power_w']/b['resisting_power_w'], 1., places=10)
        np.testing.assert_allclose(a['gradient_n'], b['gradient_n'], rtol=1e-10, atol=2e3)

    def test_operator_binding_and_passivity_fail_closed(self):
        system = fixture(); row = arc(system)
        changed = deepcopy(system); changed['points'][0] = unit([1, 0, -.09])
        with self.assertRaisesRegex(ValueError, 'different'):
            laws.Resistance(changed, welds=[row], smoothing_speed_m_s=1e-10)
        row['coefficient_n'] = -1
        with self.assertRaises(ValueError):
            laws.Resistance(system, welds=[row], smoothing_speed_m_s=1e-10)
        row = patch(system); row['hessian_n_s_m'][0, 0] *= 2
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            laws.Resistance(system, interfaces=[row], smoothing_speed_m_s=1e-10)

    def test_invalid_finite_geometry_rejected(self):
        system = fixture()
        with self.assertRaises(ValueError):
            arc(system, end=.3)
        with self.assertRaises(ValueError):
            arc(system, outward_normal=[0, 1, 0])
        with self.assertRaises(ValueError):
            patch(system, polygon=unit([[1, .02, -.02], [1, .05, .03], [1, .08, -.02]]))

    def test_actual_joint_solve_uses_finite_weld_and_interface_reactions(self):
        system = fixture()
        resistance = laws.Resistance(system, welds=[arc(system)], interfaces=[patch(system)], smoothing_speed_m_s=1e-10)
        baseline = joint.solve(system)
        answer = joint.solve(system, resistance=resistance)
        self.assertGreater(np.linalg.norm(answer['plate_omega_rad_s']-baseline['plate_omega_rad_s']),
                           .01*np.linalg.norm(baseline['plate_omega_rad_s']))
        reaction = answer['resistance_generalized_reaction_n']
        np.testing.assert_allclose(reaction[:3]+reaction[3:6], 0., atol=3e3)
        diagnostic = answer['diagnostics']
        self.assertGreater(diagnostic['resistance_dissipation_w'], 0.)
        self.assertLess(abs(diagnostic['physical_power_residual_w']),
                        diagnostic['physical_power_acceptance_allowance_w'])


if __name__ == '__main__':
    unittest.main()
