"""SI frozen-traction operator: work, symmetry, constitutive limits, no live use."""
from copy import deepcopy
import unittest

import numpy as np
import ocean_traction as traction
import viscous_sheet


def fixture(*, viscosity=1e21, rotation=None):
    lon, lat = np.meshgrid(np.linspace(-.04, .04, 5), np.linspace(-.02, .02, 3))
    lon, lat = lon.ravel(), lat.ravel()
    points = np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
    faces = []
    for row in range(2):
        for col in range(4):
            a = row*5+col
            faces.extend(((a, a+1, a+6), (a, a+6, a+5)))
    faces = np.asarray(faces)
    triangles = points[faces]
    area = 2.*np.arctan2(np.abs(np.einsum('ij,ij->i', triangles[:, 0],
        np.cross(triangles[:, 1], triangles[:, 2]))),
        1.+np.einsum('ij,ij->i', triangles[:, 0], triangles[:, 1])
        +np.einsum('ij,ij->i', triangles[:, 1], triangles[:, 2])
        +np.einsum('ij,ij->i', triangles[:, 2], triangles[:, 0]))*6371e3**2
    force = np.zeros_like(points)
    for col, sign in ((0, -1.), (4, 1.)):
        take = np.arange(col, 15, 5)
        east = np.column_stack((-np.sin(lon[take]), np.cos(lon[take]), np.zeros(len(take))))
        force[take] = sign*1e16*east
    if rotation is not None:
        points, force = points@rotation.T, force@rotation.T
    context = traction.prepare(points, faces, area, radius_m=6371e3, viscosity_pa_s=viscosity,
                               thickness_m=80e3, basal_drag_pa_s_m=1e15)
    return context, force


class OceanTractionTests(unittest.TestCase):
    def test_opposed_zero_torque_loads_have_positive_internal_extension_and_work(self):
        context, force = fixture()
        torque = np.sum(np.cross(context['sheet']['points'], force), axis=0)
        self.assertLess(np.linalg.norm(torque)/np.sum(np.linalg.norm(force, axis=1)), 1e-14)
        result = traction.solve(context, force)
        d = result['diagnostics']
        self.assertGreater(np.linalg.eigvalsh(result['strain_rate_per_s']).max(), 0.)
        self.assertGreater(d['internal_source_work_w'], 0.)
        self.assertGreater(d['membrane_dissipation_w'], 0.)
        self.assertGreaterEqual(d['basal_dissipation_w'], 0.)
        self.assertLess(abs(d['internal_power_residual_w'])/d['internal_source_work_w'], 2e-8)
        self.assertLessEqual(abs(d['internal_power_residual_w']), d['power_acceptance_bound_w'])
        weighted = result['internal_velocity_m_s']*context['projection'].root_area[:, None]
        self.assertLess(np.linalg.norm(context['projection'].fit(weighted)), 1e-18)
        self.assertFalse(d['native_transport_coupled'])

    def test_zero_force_gives_zero_motion_and_damage(self):
        context, force = fixture()
        result = traction.solve(context, force*0.)
        np.testing.assert_array_equal(result['internal_velocity_m_s'], 0.)
        history = traction.initialize_failure(context, tensile_strength_pa=1e6,
            fracture_energy_j_m2=1e6, regularization_length_m=1e4, damage_efficiency=.5)
        advanced = traction.advance_failure(context, result, history, dt_s=1e13)
        np.testing.assert_array_equal(advanced['damage'], 0.)

    def test_force_scaling_and_uniform_resistance_scaling_use_physical_units(self):
        context, force = fixture()
        first = traction.solve(context, force)
        doubled = traction.solve(context, 2.*force)
        np.testing.assert_allclose(doubled['internal_velocity_m_s'], 2.*first['internal_velocity_m_s'], rtol=1e-8, atol=1e-19)
        self.assertAlmostEqual(doubled['diagnostics']['internal_source_work_w']/first['diagnostics']['internal_source_work_w'], 4., places=8)
        other = traction.prepare(context['sheet']['points'], context['sheet']['faces'], context['face_area_m2'],
            radius_m=6371e3, viscosity_pa_s=2e21, thickness_m=80e3, basal_drag_pa_s_m=2e15)
        slower = traction.solve(other, force)
        np.testing.assert_allclose(slower['internal_velocity_m_s'], first['internal_velocity_m_s']/2., rtol=1e-8, atol=1e-19)

    def test_stronger_viscous_interior_reduces_internal_deformation(self):
        weak, force = fixture(viscosity=1e20)
        strong, _ = fixture(viscosity=1e23)
        a, b = traction.solve(weak, force), traction.solve(strong, force)
        self.assertGreater(np.linalg.norm(a['strain_rate_per_s']), 10.*np.linalg.norm(b['strain_rate_per_s']))

    def test_rigid_load_and_motion_are_not_reapplied_as_internal_work(self):
        context, force = fixture()
        points = context['sheet']['points']; area = context['sheet']['data']
        rigid = np.cross(np.array([0., 0., 1e-9]), points)
        rigid_force = 1e15*area[:, None]*rigid
        result = traction.solve(context, rigid_force, rigid_velocity_m_s=rigid)
        self.assertLess(np.linalg.norm(result['internal_velocity_m_s']), 1e-20)
        np.testing.assert_allclose(result['diagnostics']['total_source_work_w'],
                                   result['diagnostics']['rigid_source_work_w'], rtol=1e-12)
        loaded = traction.solve(context, force+rigid_force, rigid_velocity_m_s=rigid)
        baseline = traction.solve(context, force)
        np.testing.assert_allclose(loaded['internal_velocity_m_s'], baseline['internal_velocity_m_s'], rtol=1e-8, atol=1e-18)
        self.assertLess(abs(loaded['diagnostics']['source_partition_residual_w']), 1e-7)

    def test_globe_rotation_preserves_velocity_stress_and_work(self):
        angle = .83
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
        first, force = fixture(); rotated, rotated_force = fixture(rotation=rotation)
        a, b = traction.solve(first, force), traction.solve(rotated, rotated_force)
        np.testing.assert_allclose(b['internal_velocity_m_s'], a['internal_velocity_m_s']@rotation.T, rtol=3e-7, atol=1e-18)
        np.testing.assert_allclose(b['stress_pa'], rotation@a['stress_pa']@rotation.T, rtol=2e-6, atol=1e-3)
        self.assertAlmostEqual(b['diagnostics']['internal_source_work_w']/a['diagnostics']['internal_source_work_w'], 1., places=7)

    def test_fracture_and_heat_share_one_finite_work_budget(self):
        context, force = fixture()
        response = traction.solve(context, force)
        history = traction.initialize_failure(context, tensile_strength_pa=1.,
            fracture_energy_j_m2=1e6, regularization_length_m=1e4, damage_efficiency=.5)
        original = deepcopy(history)
        advanced = traction.advance_failure(context, response, history, dt_s=1e14)
        self.assertTrue(np.any(advanced['locally_exhausted']))
        self.assertTrue(np.all(advanced['damage'] <= 1))
        self.assertTrue(np.all(advanced['membrane_heat_j'] >= 0))
        np.testing.assert_allclose(advanced['fracture_increment_j']+advanced['membrane_heat_j'],
                                  response['membrane_power_by_face_w']*1e14, rtol=1e-14)
        again = traction.advance_failure(context, response, advanced['history'], dt_s=1e14)
        np.testing.assert_array_equal(again['fracture_increment_j'][advanced['locally_exhausted']], 0.)
        for key in history:
            np.testing.assert_array_equal(history[key], original[key])

    def test_strength_threshold_and_compression_do_not_create_failure(self):
        context, force = fixture()
        history = traction.initialize_failure(context, tensile_strength_pa=1e30,
            fracture_energy_j_m2=1e6, regularization_length_m=1e4, damage_efficiency=1.)
        response = traction.solve(context, force)
        result = traction.advance_failure(context, response, history, dt_s=1e14)
        np.testing.assert_array_equal(result['damage'], 0.)
        # Solve an actual inward velocity field; do not fabricate saved strains
        # inconsistent with the solved force or energy budget.
        points = context['sheet']['points']
        velocity = (np.array([1., 0., 0.])-points*points[:, :1])*1e-9
        root_area = context['projection'].root_area[:, None]
        velocity = context['projection'].apply(velocity*root_area)/root_area
        compressive = traction.solve(context, 1e15*viscous_sheet.apply(context['sheet'], velocity))
        history = traction.initialize_failure(context, tensile_strength_pa=0.,
            fracture_energy_j_m2=1e6, regularization_length_m=1e4, damage_efficiency=1.)
        result = traction.advance_failure(context, compressive, history, dt_s=1e14)
        np.testing.assert_array_equal(result['damage'], 0.)
        np.testing.assert_array_equal(result['currently_opening'], False)

    def test_disconnected_owners_have_independent_internal_motion(self):
        context, force = fixture()
        points, faces = context['sheet']['points'], context['sheet']['faces']
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        pair = traction.prepare(np.concatenate((points, points@rotation.T)),
            np.concatenate((faces, faces+len(points))), np.tile(context['face_area_m2'], 2),
            radius_m=6371e3, viscosity_pa_s=1e21, thickness_m=80e3, basal_drag_pa_s_m=1e15,
            owner_labels=np.repeat([3, 7], len(points)))
        response = traction.solve(pair, np.concatenate((force, np.zeros_like(force))))
        baseline = traction.solve(context, force)
        self.assertEqual(response['diagnostics']['components'], 2)
        np.testing.assert_allclose(response['internal_velocity_m_s'][:len(points)],
                                   baseline['internal_velocity_m_s'], rtol=1e-8, atol=1e-20)
        np.testing.assert_array_equal(response['internal_velocity_m_s'][len(points):], 0.)

    def test_radial_constraint_balances_excluded_force_without_internal_work(self):
        context, force = fixture()
        radial = context['sheet']['points']*1e16
        response = traction.solve(context, force+radial)
        baseline = traction.solve(context, force)
        np.testing.assert_allclose(response['internal_velocity_m_s'],
                                   baseline['internal_velocity_m_s'], rtol=1e-8, atol=1e-19)
        np.testing.assert_allclose(response['radial_reaction_n'], -radial, rtol=1e-12, atol=10.)

    def test_large_rigid_load_does_not_hide_resolvable_internal_force(self):
        context, force = fixture()
        points = context['sheet']['points']; area = context['sheet']['data']
        rigid = np.cross(np.array([0., 0., 1e-9]), points)
        rigid_force = 1e15*area[:, None]*rigid
        baseline = traction.solve(context, force*1e-12)
        combined = traction.solve(context, rigid_force+force*1e-12)
        self.assertGreater(combined['diagnostics']['iterations'], 0)
        self.assertGreater(np.linalg.norm(combined['internal_velocity_m_s']), 0.)
        # Addition/projection of the much larger rigid force limits attainable
        # accuracy; require agreement to its reported machine-roundoff floor.
        error = np.linalg.norm(combined['internal_velocity_m_s']-baseline['internal_velocity_m_s'])
        self.assertLess(error/np.linalg.norm(baseline['internal_velocity_m_s']), .05)
        self.assertTrue(combined['diagnostics']['roundoff_limited'])

    def test_mismatched_context_or_mutated_response_cannot_spend_fracture_energy(self):
        context, force = fixture()
        response = traction.solve(context, force)
        other = traction.prepare(context['sheet']['points'], context['sheet']['faces'], context['face_area_m2'],
            radius_m=6371e3, viscosity_pa_s=1e21, thickness_m=160e3, basal_drag_pa_s_m=1e15)
        history = traction.initialize_failure(other, tensile_strength_pa=0.,
            fracture_energy_j_m2=1e6, regularization_length_m=1e4, damage_efficiency=1.)
        with self.assertRaisesRegex(ValueError, 'different geometry or constitutive context'):
            traction.advance_failure(other, response, history, dt_s=1.)
        aligned = traction.initialize_failure(context, tensile_strength_pa=0.,
            fracture_energy_j_m2=1e6, regularization_length_m=1e4, damage_efficiency=1.)
        for field in ('strain_rate_per_s', 'stress_pa', 'membrane_power_by_face_w', 'internal_velocity_m_s'):
            corrupted = deepcopy(response)
            corrupted[field] *= 2.
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'response has changed'):
                traction.advance_failure(context, corrupted, aligned, dt_s=1.)
        context['thickness_m'] *= 2.
        with self.assertRaisesRegex(ValueError, 'prepared SI sheet has changed'):
            traction.solve(context, force)
        np.testing.assert_array_equal(history['spent_energy_j'], 0.)
        np.testing.assert_array_equal(aligned['spent_energy_j'], 0.)

    def test_invalid_inputs_and_unconverged_response_cannot_mutate_history(self):
        context, force = fixture()
        history = traction.initialize_failure(context, tensile_strength_pa=1e6,
            fracture_energy_j_m2=1e6, regularization_length_m=1e4, damage_efficiency=.5)
        response = traction.solve(context, force)
        response['diagnostics']['converged'] = False
        with self.assertRaises(ValueError):
            traction.advance_failure(context, response, history, dt_s=1e13)
        with self.assertRaises(ValueError):
            traction.solve(context, force, iterations=1, tolerance=1e-14)
        with self.assertRaises(ValueError):
            traction.solve(context, force, rigid_velocity_m_s=force*1e-25)
        with self.assertRaises(ValueError):
            traction.prepare(context['sheet']['points'], context['sheet']['faces'], context['face_area_m2'],
                radius_m=6371e3, viscosity_pa_s=1e21, thickness_m=80e3, basal_drag_pa_s_m=1e15,
                owner_labels=np.arange(len(force)))
        np.testing.assert_array_equal(history['spent_energy_j'], 0.)


if __name__ == '__main__':
    unittest.main()
