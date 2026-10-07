"""Independent field and two-plate work checks for the candidate channel."""

import unittest

import numpy as np

import entry_contact_channel
import finite_entry_arc
import material_surface
import mesh_coverage
from experiments.entry_channel_pair_oracle import pair_energy
from experiments.entry_channel_work import candidate_energy_and_field_gradient
from ridge_geometry import rotate
from tests.test_affine_overlap_derivatives import fields
from tests.test_entry_stack_work_oracle import FACES, NORMAL
from tests.test_entry_stack_work_oracle import geometry as paired_geometry


class EntryChannelShapeWorkTests(unittest.TestCase):
    def setUp(self):
        self.points = paired_geometry()
        lower = self.points[FACES[0]]
        self.area_km2 = float(material_surface.spherical_face_areas(
            lower, np.array([[0, 1, 2]]))[0])
        self.volume = self.area_km2 * 35.
        self.mass = self.volume * 1e9 * 2800.
        self.buoyancy = float(entry_contact_channel.buoyancy_surface_load(
            self.volume, self.mass, self.area_km2))
        self.base = 35000.
        self.lower_stiffness = 1000.
        self.upper_stiffness = 3300. * 9.81

    def differentiated(self, points, *, finite):
        return candidate_energy_and_field_gradient(
            points[FACES[0]], fields(points, finite=finite), 50., self.base,
            self.buoyancy, self.lower_stiffness, self.upper_stiffness,
            tolerance=1e-7)

    def independent(self, points, normal, midpoint, *, finite):
        return pair_energy(
            points, FACES[0], FACES[1], normal, 50., self.volume,
            self.mass, self.base, self.lower_stiffness,
            self.upper_stiffness,
            finite_midpoint=midpoint if finite else None,
            finite_half_length_km=50. if finite else None,
            quadrature_tolerance=1e-8)['energy_j']

    def test_candidate_energy_matches_independent_pair_oracle(self):
        for finite in (False, True):
            with self.subTest(finite=finite):
                energy, gradient = self.differentiated(
                    self.points, finite=finite)
                self.assertEqual(gradient.shape, (6, 3))
                self.assertGreater(energy, 0.)
                expected = self.independent(
                    self.points, NORMAL[0], np.array([1., 0., 0.]),
                    finite=finite)
                np.testing.assert_allclose(energy, expected, rtol=1e-9)

    def test_every_field_derivative_matches_finite_difference(self):
        lower = self.points[FACES[0]]
        for finite in (False, True):
            with self.subTest(finite=finite):
                values = fields(self.points, finite=finite)
                _, gradient = self.differentiated(self.points, finite=finite)
                for field in range(6):
                    for corner in range(3):
                        step = 1. if field == 0 else 1e-6
                        plus, minus = values.copy(), values.copy()
                        plus[field, corner] += step
                        minus[field, corner] -= step
                        forward = candidate_energy_and_field_gradient(
                            lower, plus, 50., self.base, self.buoyancy,
                            self.lower_stiffness, self.upper_stiffness,
                            tolerance=1e-7)[0]
                        backward = candidate_energy_and_field_gradient(
                            lower, minus, 50., self.base, self.buoyancy,
                            self.lower_stiffness, self.upper_stiffness,
                            tolerance=1e-7)[0]
                        observed = (forward - backward) / (2. * step)
                        np.testing.assert_allclose(
                            gradient[field, corner], observed,
                            rtol=1e-6, atol=2e16)

    def test_mixed_contact_free_transition_and_signed_buoyancy(self):
        lower = self.points[FACES[0]]
        values = fields(self.points, finite=True)
        base = 100000.
        dense_mass = self.volume * 1e9 * 3450.
        buoyancy = float(entry_contact_channel.buoyancy_surface_load(
            self.volume, dense_mass, self.area_km2))
        self.assertLess(buoyancy, 0.)
        def candidate(field_values):
            return candidate_energy_and_field_gradient(
                lower, field_values, 50., base, buoyancy,
                self.lower_stiffness, self.upper_stiffness,
                tolerance=1e-7)
        energy, gradient = candidate(values)
        independent = pair_energy(
            self.points, FACES[0], FACES[1], NORMAL[0], 50.,
            self.volume, dense_mass, base, self.lower_stiffness,
            self.upper_stiffness, finite_midpoint=np.array([1., 0., 0.]),
            finite_half_length_km=50., quadrature_tolerance=1e-8)
        np.testing.assert_allclose(energy, independent['energy_j'], rtol=1e-8)
        for field, corner in ((0, 1), (1, 0), (2, 2), (5, 1)):
            step = 1. if field == 0 else 1e-6
            plus, minus = values.copy(), values.copy()
            plus[field, corner] += step
            minus[field, corner] -= step
            observed = (candidate(plus)[0] - candidate(minus)[0]) / (2. * step)
            np.testing.assert_allclose(
                gradient[field, corner], observed, rtol=2e-6, atol=2e16)
        normal = NORMAL[0]
        midpoint = np.array([1., 0., 0.])
        planes = [normal]
        planes.extend(finite_entry_arc.endpoint_planes(
            normal, midpoint, 50., 6371.)[:2])
        planes.extend(mesh_coverage._triangle_planes(self.points[FACES[1]]))
        rotation = np.stack([
            (6371e3 / np.sqrt(1. - (lower @ normal)**2))[:, None]
            * np.cross(lower, normal) if index == 0
            else np.cross(lower, plane)
            for index, plane in enumerate(planes)])
        lower_torque = np.einsum('ki,kij->j', gradient, rotation)
        for axis in np.eye(3):
            for owner, predicted in ((0, lower_torque), (1, -lower_torque)):
                def shifted(sign):
                    moved = self.points.copy()
                    select = slice(3 * owner, 3 * owner + 3)
                    moved[select] = rotate(moved[select], axis * sign * 1e-6)
                    moved_normal = (rotate(normal, axis * sign * 1e-6)
                                    if owner == 1 else normal)
                    moved_midpoint = (rotate(midpoint, axis * sign * 1e-6)
                                      if owner == 1 else midpoint)
                    return pair_energy(
                        moved, FACES[0], FACES[1], moved_normal, 50.,
                        self.volume, dense_mass, base,
                        self.lower_stiffness, self.upper_stiffness,
                        finite_midpoint=moved_midpoint,
                        finite_half_length_km=50.,
                        quadrature_tolerance=1e-8)['energy_j']
                observed = (shifted(1.) - shifted(-1.)) / (2e-6)
                np.testing.assert_allclose(
                    predicted @ axis, observed, rtol=3e-6, atol=3e18)

    def test_fully_free_dense_column_has_no_upper_uplift_work(self):
        lower = self.points[FACES[0]]
        values = fields(self.points, finite=True)
        dense_mass = self.volume * 1e9 * 3450.
        buoyancy = float(entry_contact_channel.buoyancy_surface_load(
            self.volume, dense_mass, self.area_km2))
        energy, gradient = candidate_energy_and_field_gradient(
            lower, values, 50., 1., buoyancy, self.lower_stiffness,
            self.upper_stiffness)
        independent = pair_energy(
            self.points, FACES[0], FACES[1], NORMAL[0], 50.,
            self.volume, dense_mass, 1., self.lower_stiffness,
            self.upper_stiffness, finite_midpoint=np.array([1., 0., 0.]),
            finite_half_length_km=50., quadrature_tolerance=1e-8)
        np.testing.assert_allclose(energy, independent['energy_j'], rtol=1e-9)
        self.assertEqual(independent['area_mean_upper_uplift_m'], 0.)
        self.assertTrue(np.isfinite(gradient).all())

    def test_two_plate_rotation_work_matches_independent_oracle(self):
        points = self.points
        lower, upper = points[FACES[0]], points[FACES[1]]
        normal = NORMAL[0]
        midpoint = np.array([1., 0., 0.])
        step = 1e-6
        for finite in (False, True):
            with self.subTest(finite=finite):
                _, gradient = self.differentiated(points, finite=finite)
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
                lower_torque = np.einsum('ki,kij->j', gradient, rotation)
                upper_torque = -lower_torque
                for axis in np.eye(3):
                    for owner, predicted in ((0, lower_torque), (1, upper_torque)):
                        def shifted(sign):
                            moved = points.copy()
                            select = slice(3 * owner, 3 * owner + 3)
                            moved[select] = rotate(moved[select], axis * sign * step)
                            moved_normal = (rotate(normal, axis * sign * step)
                                            if owner == 1 else normal)
                            moved_midpoint = (rotate(midpoint, axis * sign * step)
                                              if owner == 1 else midpoint)
                            return self.independent(
                                moved, moved_normal, moved_midpoint,
                                finite=finite)
                        observed = (shifted(1.) - shifted(-1.)) / (2. * step)
                        np.testing.assert_allclose(
                            predicted @ axis, observed, rtol=1e-6, atol=2e18)

    def test_empty_intersection_has_zero_energy_and_work(self):
        values = -np.ones((6, 3))
        energy, gradient = candidate_energy_and_field_gradient(
            self.points[FACES[0]], values, 50., self.base,
            self.buoyancy, self.lower_stiffness, self.upper_stiffness)
        self.assertEqual(energy, 0.)
        np.testing.assert_array_equal(gradient, 0.)


if __name__ == '__main__':
    unittest.main()
