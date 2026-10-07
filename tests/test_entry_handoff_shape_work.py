"""Independent work checks for the read-only full pair handoff correction."""

import unittest

import numpy as np

import finite_entry_arc
import material_surface
import mesh_coverage
from experiments.entry_channel_pair_oracle import pair_handoff
from experiments.entry_handoff_work import (
    correction_and_field_gradient, paired_rigid_torques)
from ridge_geometry import rotate
from tests.test_affine_overlap_derivatives import fields
from tests.test_entry_stack_work_oracle import FACES, NORMAL
from tests.test_entry_stack_work_oracle import geometry as paired_geometry


class EntryHandoffShapeWorkTests(unittest.TestCase):
    def setUp(self):
        self.points = paired_geometry()
        area = material_surface.spherical_face_areas(self.points, FACES)
        self.volumes = area * 35.
        self.upper_mass = self.volumes[1] * 1e9 * 2800.

    def differentiated(self, points, values, lower_mass, *, base):
        return correction_and_field_gradient(
            points[FACES[0]], points[FACES[1]], values, 50.,
            self.volumes[0], lower_mass, self.volumes[1], self.upper_mass,
            base, 1000., 3300. * 9.81,
            exclusive_upper_coverage=True)

    def independent(self, points, normal, midpoint, lower_mass, *,
                    finite, base):
        return pair_handoff(
            points, FACES[0], FACES[1], normal, 50.,
            self.volumes[0], lower_mass, self.volumes[1], self.upper_mass,
            base, 1000., 3300. * 9.81,
            exclusive_upper_coverage=True,
            finite_midpoint=midpoint if finite else None,
            finite_half_length_km=50. if finite else None,
            quadrature_tolerance=1e-8)

    def test_affine_reference_entry_integral_derivative(self):
        for finite in (False, True):
            with self.subTest(finite=finite):
                values = fields(self.points, finite=finite)
                for integrated_field in (0, 5):
                    mean, gradient, polygon = (
                        finite_entry_arc.integrate_affine_field(
                            values, field_index=integrated_field))
                    self.assertGreater(mean, 0.)
                    self.assertGreaterEqual(len(polygon), 3)
                    for field in range(6):
                        for corner in range(3):
                            step = 1. if field == 0 else 1e-6
                            plus, minus = values.copy(), values.copy()
                            plus[field, corner] += step
                            minus[field, corner] -= step
                            observed = (
                                finite_entry_arc.integrate_affine_field(
                                    plus, integrated_field)[0]
                                - finite_entry_arc.integrate_affine_field(
                                    minus, integrated_field)[0]
                            ) / (2. * step)
                            np.testing.assert_allclose(
                                gradient[field, corner], observed,
                                rtol=2e-6, atol=1e-4)

    def test_complete_handoff_matches_independent_energy_and_fields(self):
        for finite in (False, True):
            for lower_density, base in ((2800., 35000.),
                                        (3450., 100000.)):
                with self.subTest(finite=finite, density=lower_density):
                    lower_mass = self.volumes[0] * 1e9 * lower_density
                    values = fields(self.points, finite=finite)
                    result = self.differentiated(
                        self.points, values, lower_mass, base=base)
                    independent = self.independent(
                        self.points, NORMAL[0], np.array([1., 0., 0.]),
                        lower_mass, finite=finite, base=base)
                    for key in ('correction_j', 'channel_energy_j',
                                'removed_entry_energy_j',
                                'removed_stack_energy_j'):
                        np.testing.assert_allclose(
                            result[key], independent[key], rtol=1e-8,
                            atol=1e9)
                    for field, corner in ((0, 1), (1, 0), (2, 2), (5, 1)):
                        step = 1. if field == 0 else 1e-6
                        plus, minus = values.copy(), values.copy()
                        plus[field, corner] += step
                        minus[field, corner] -= step
                        observed = (
                            self.differentiated(
                                self.points, plus, lower_mass,
                                base=base)['correction_j']
                            - self.differentiated(
                                self.points, minus, lower_mass,
                                base=base)['correction_j']
                        ) / (2. * step)
                        np.testing.assert_allclose(
                            result['field_gradient_j'][field, corner],
                            observed, rtol=3e-6, atol=2e16)

    def test_complete_handoff_matches_independent_two_plate_work(self):
        points = self.points
        lower, upper = points[FACES[0]], points[FACES[1]]
        normal = NORMAL[0]
        midpoint = np.array([1., 0., 0.])
        for finite in (False, True):
            with self.subTest(finite=finite):
                lower_mass = self.volumes[0] * 1e9 * 2800.
                values = fields(points, finite=finite)
                result = self.differentiated(
                    points, values, lower_mass, base=35000.)
                gradient = result['field_gradient_j']
                planes = [normal]
                if finite:
                    planes.extend(finite_entry_arc.endpoint_planes(
                        normal, midpoint, 50., 6371.)[:2])
                else:
                    planes.extend([np.zeros(3), np.zeros(3)])
                planes.extend(mesh_coverage._triangle_planes(upper))
                rotation = np.stack([
                    (6371e3 / np.sqrt(1. - (lower @ normal)**2))[:, None]
                    * np.cross(lower, normal) if index == 0
                    else np.cross(lower, plane)
                    for index, plane in enumerate(planes)])
                torque = np.einsum('ki,kij->j', gradient, rotation)
                for axis in np.eye(3):
                    for owner, predicted in ((0, torque), (1, -torque)):
                        def shifted(sign):
                            moved = points.copy()
                            select = slice(3 * owner, 3 * owner + 3)
                            moved[select] = rotate(
                                moved[select], axis * sign * 1e-6)
                            moved_normal = (
                                rotate(normal, axis * sign * 1e-6)
                                if owner == 1 else normal)
                            moved_midpoint = (
                                rotate(midpoint, axis * sign * 1e-6)
                                if owner == 1 else midpoint)
                            return self.independent(
                                moved, moved_normal, moved_midpoint,
                                lower_mass, finite=finite,
                                base=35000.)['correction_j']
                        observed = (shifted(1.) - shifted(-1.)) / (2e-6)
                        np.testing.assert_allclose(
                            predicted @ axis, observed,
                            rtol=3e-6, atol=2e18)

    def test_direct_paired_force_torques_match_independent_work(self):
        normal = NORMAL[0]
        midpoint = np.array([1., 0., 0.])
        for finite in (False, True):
            for density, base in ((2800., 35000.), (3450., 100000.)):
                with self.subTest(finite=finite, density=density):
                    lower_mass = self.volumes[0] * 1e9 * density
                    result = paired_rigid_torques(
                        self.points, FACES[0], FACES[1], normal, 50.,
                        self.volumes[0], lower_mass, self.volumes[1],
                        self.upper_mass, base, 1000., 3300. * 9.81,
                        exclusive_upper_coverage=True,
                        finite_midpoint=midpoint if finite else None,
                        finite_half_length_km=50. if finite else None)
                    np.testing.assert_array_equal(
                        result['lower_torque_n_m']
                        + result['upper_torque_n_m'], 0.)
                    for axis in np.eye(3):
                        for owner, key in ((0, 'lower_torque_n_m'),
                                           (1, 'upper_torque_n_m')):
                            def shifted(sign):
                                moved = self.points.copy()
                                select = slice(3 * owner, 3 * owner + 3)
                                moved[select] = rotate(
                                    moved[select], axis * sign * 1e-6)
                                moved_normal = (
                                    rotate(normal, axis * sign * 1e-6)
                                    if owner == 1 else normal)
                                moved_midpoint = (
                                    rotate(midpoint, axis * sign * 1e-6)
                                    if owner == 1 else midpoint)
                                return self.independent(
                                    moved, moved_normal, moved_midpoint,
                                    lower_mass, finite=finite,
                                    base=base)['correction_j']
                            observed = -(shifted(1.) - shifted(-1.)) / (2e-6)
                            np.testing.assert_allclose(
                                result[key] @ axis, observed,
                                rtol=4e-6, atol=3e18)

    def test_rejects_nonexclusive_coverage(self):
        values = fields(self.points, finite=False)
        with self.assertRaisesRegex(ValueError, 'exclusive upper coverage'):
            correction_and_field_gradient(
                self.points[FACES[0]], self.points[FACES[1]], values,
                50., self.volumes[0], self.volumes[0] * 1e9 * 2800.,
                self.volumes[1], self.upper_mass, 35000., 1000.,
                3300. * 9.81, exclusive_upper_coverage=False)


if __name__ == '__main__':
    unittest.main()
