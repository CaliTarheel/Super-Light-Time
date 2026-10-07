"""Independent SI manufactured checks for the frozen joint mechanics primitive.

These use the analytic second moment of a regular icosahedron, rather than
constructing reference loads from the assembled Hessian. Coefficients below
are explicit test inputs, not a calibration or a production model policy.
"""
import unittest

import numpy as np

import mesh_geometry
from benchmarks.collision_architecture import shared_contact as joint


RADIUS_M = 6_371_000.
BETA = 8.e14
ETA = 1.e23
THICKNESS_M = 40_000.


def sphere_system(*, radius=RADIUS_M, eta=ETA, beta=BETA,
                  mantle_omega=None, torque=None, other_drag=None):
    mesh = mesh_geometry.icosphere(0)
    points = mesh['vertices']
    mantle_omega = np.zeros(3) if mantle_omega is None else np.asarray(mantle_omega)
    return joint.assemble(
        points, mesh['faces'], np.zeros(len(points), dtype=int), radius_m=radius,
        basal_drag_pa_s_per_m=beta, viscosity_pa_s=eta,
        sheet_thickness_m=THICKNESS_M,
        basal_reference_velocity_m_s=radius*np.cross(mantle_omega, points),
        other_plate_rotational_drag_n_m_s=(np.zeros((3, 3)) if other_drag is None else other_drag),
        external_torques_n_m=np.asarray(np.zeros(3) if torque is None else torque)[None],
        external_nodal_forces_n=np.zeros_like(points), contacts=[])


def analytic_basal_rotation_drag(radius=RADIUS_M, beta=BETA):
    # Equal nodal weights and icosahedral symmetry give sum(A*r*r.T)=A/3*I.
    # Torque resisting omega is beta*R^2*sum A*(I-r*r.T)*omega.
    return (8.*np.pi/3.)*beta*radius**4*np.eye(3)


class AnalyticJointMechanicsTests(unittest.TestCase):
    def test_uniform_spherical_drag_has_analytic_si_euler_velocity_and_power(self):
        omega = np.array([1.3, -2.1, .7])*1.e-16
        remaining = np.diag([.8, 1.3, .4])*1.e42
        basal = analytic_basal_rotation_drag()
        torque = (basal+remaining)@omega
        system = sphere_system(torque=torque, other_drag=remaining)
        answer = joint.solve(system)
        np.testing.assert_allclose(answer['plate_omega_rad_s'][0], omega, rtol=2e-12, atol=1e-29)
        np.testing.assert_allclose(answer['residual_velocity_m_s'], 0., atol=2e-21)
        np.testing.assert_allclose(answer['total_velocity_m_s'],
                                   RADIUS_M*np.cross(omega, system['points']), rtol=3e-12, atol=2e-21)
        diagnostic = answer['diagnostics']
        expected_basal = float(omega@basal@omega)
        self.assertAlmostEqual(diagnostic['basal_dissipation_w']/expected_basal, 1., places=11)
        self.assertAlmostEqual(diagnostic['other_drag_dissipation_w']/float(omega@remaining@omega), 1., places=11)
        self.assertLess(abs(diagnostic['viscous_dissipation_w']), expected_basal*1e-12)
        self.assertLess(abs(diagnostic['physical_power_residual_w']), abs(torque@omega)*1e-11)

    def test_moving_mantle_uses_actuator_work_in_physical_power_identity(self):
        mantle = np.array([1., -.5, .8])*1.e-16
        relative = np.array([.2, .6, -.3])*1.e-16
        omega = mantle+relative
        basal = analytic_basal_rotation_drag()
        remaining = np.diag([.3, .5, .9])*1.e42
        torque = basal@relative+remaining@omega
        system = sphere_system(mantle_omega=mantle, torque=torque, other_drag=remaining)
        answer = joint.solve(system)
        np.testing.assert_allclose(answer['plate_omega_rad_s'][0], omega, rtol=3e-12, atol=1e-29)
        np.testing.assert_allclose(answer['residual_velocity_m_s'], 0., atol=2e-21)
        diagnostic = answer['diagnostics']
        expected_drag = float(relative@basal@relative)
        expected_mantle_work = float(-relative@basal@mantle)
        self.assertAlmostEqual(diagnostic['basal_dissipation_w']/expected_drag, 1., places=11)
        self.assertAlmostEqual(diagnostic['basal_reference_input_power_w']/expected_mantle_work, 1., places=11)
        expected_external = float(torque@omega)
        self.assertAlmostEqual(diagnostic['external_power_w']/expected_external, 1., places=11)
        self.assertLess(abs(diagnostic['physical_power_residual_w']),
                        (abs(expected_external)+abs(expected_mantle_work))*1e-11)
        # The effective linear RHS includes beta*A*M and is not physical
        # external work when the basal reference is moving.
        effective_power = float(system['load_n']@answer['generalized_velocity_m_s'])
        self.assertGreater(abs(effective_power-expected_external-expected_mantle_work), expected_drag*.1)

    def test_exact_common_mantle_motion_does_not_fail_a_zero_work_balance(self):
        mantle = np.array([1., -.5, .8])*1e-16
        system = sphere_system(mantle_omega=mantle, eta=0.)
        answer = joint.solve(system)
        expected_velocity = RADIUS_M*np.cross(mantle, system['points'])
        np.testing.assert_allclose(answer['total_velocity_m_s'], expected_velocity, rtol=3e-12, atol=2e-21)
        characteristic_power = float(mantle@analytic_basal_rotation_drag()@mantle)
        diagnostic = answer['diagnostics']
        self.assertLess(diagnostic['basal_dissipation_w'], characteristic_power*1e-24)
        self.assertLess(abs(diagnostic['physical_power_residual_w']), characteristic_power*1e-12)

    def test_radius_scaling_separates_basal_area_from_sheet_gradient(self):
        small = sphere_system(radius=RADIUS_M)
        large = sphere_system(radius=2*RADIUS_M)
        # Fix the same SI nodal velocity coefficients on the same angular mesh.
        velocity = np.random.default_rng(4201).normal(size=len(small['load_n']))*1e-9
        velocity[:3] = 0.
        for name, expected_ratio in [('basal_hessian_n_s_m', 4.), ('viscous_hessian_n_s_m', 1.)]:
            first = float(velocity@small[name]@velocity)
            second = float(velocity@large[name]@velocity)
            self.assertGreater(first, 0.)
            self.assertAlmostEqual(second/first, expected_ratio, places=12)

    def test_equilateral_affine_dilation_has_analytic_viscous_si_power(self):
        # All vertices share a latitude and chord plane. This tangent nodal
        # field has planar D=diag(k,k,0), div=2*k, hence D:D+div^2=6*k^2.
        theta = .2
        azimuth = np.arange(3)*2*np.pi/3
        points = np.column_stack((np.sin(theta)*np.cos(azimuth),
                                  np.sin(theta)*np.sin(azimuth), np.full(3, np.cos(theta))))
        system = joint.assemble(points, np.array([[0, 1, 2]]), np.zeros(3, dtype=int),
            radius_m=RADIUS_M, basal_drag_pa_s_per_m=BETA, viscosity_pa_s=ETA,
            sheet_thickness_m=THICKNESS_M, basal_reference_velocity_m_s=np.zeros((3, 3)),
            other_plate_rotational_drag_n_m_s=np.zeros((3, 3)), external_torques_n_m=np.zeros((1, 3)),
            external_nodal_forces_n=np.zeros((3, 3)), contacts=[])
        rate = 2.e-15
        velocity = rate*RADIUS_M*points.copy()
        velocity[:, 2] = -rate*RADIUS_M*np.sin(theta)**2/np.cos(theta)
        y = np.r_[np.zeros(3), np.einsum('nia,ni->na', system['tangent_basis'], velocity).ravel()]
        cosine_side = np.cos(theta)**2-.5*np.sin(theta)**2
        spherical_excess = 3*np.arccos(cosine_side/(1+cosine_side))-np.pi
        expected = 12*ETA*THICKNESS_M*(RADIUS_M**2*spherical_excess)*rate**2
        actual = float(np.linalg.norm(system['strain_design']@y)**2)
        self.assertAlmostEqual(actual/expected, 1., places=12)

    def test_material_gauge_symmetry_does_not_erase_other_domain_plate_work(self):
        material = sphere_system()
        y = np.random.default_rng(431).normal(size=len(material['load_n']))*1e-9
        delta_a = np.array([.8, -.3, .5])*1e-9
        delta_v = np.cross(delta_a, material['points'])
        shift = np.r_[delta_a, -np.einsum('nia,ni->na', material['tangent_basis'], delta_v).ravel()]
        original, gradient = joint.evaluate(material, y)
        moved, _ = joint.evaluate(material, y+shift)
        self.assertAlmostEqual(moved/original, 1., places=12)
        self.assertLess(abs(gradient@shift), abs(original)*1e-12)
        # Other-domain motion really changes when the plate rotation changes,
        # even if an opposite material residual keeps all represented V fixed.
        other = np.diag([1., 2., 3.])*1e42
        torque = np.array([1., -.7, .5])*1e26
        whole = sphere_system(other_drag=other, torque=torque)
        before, _ = joint.evaluate(whole, y)
        after, _ = joint.evaluate(whole, y+shift)
        old_omega, new_omega = y[:3]/RADIUS_M, (y[:3]+delta_a)/RADIUS_M
        expected_delta = .5*(new_omega@other@new_omega-old_omega@other@old_omega)-torque@(delta_a/RADIUS_M)
        self.assertAlmostEqual((after-before)/expected_delta, 1., places=11)

    def test_negative_remaining_drag_cannot_hide_behind_an_unrelated_large_mode(self):
        # Overall energy can still be SPD due to the positive material drag;
        # each explicitly passive term must nevertheless be passive itself.
        with self.assertRaisesRegex(ValueError, 'positive semidefinite'):
            sphere_system(other_drag=np.diag([1e60, -1e40, 1e60]))

    def test_asymmetric_small_drag_entries_cannot_hide_behind_a_large_diagonal(self):
        remaining = np.eye(3)*1e60
        remaining[0, 1] = 1e40
        remaining[1, 0] = 2e40
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            sphere_system(other_drag=remaining)


if __name__ == '__main__':
    unittest.main()
