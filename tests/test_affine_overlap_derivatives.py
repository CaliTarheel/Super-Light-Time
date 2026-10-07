"""Independent geometry and virtual-work checks for moving overlap clips."""

import unittest

import numpy as np

import continental_entry
import entry_overlap_geometry
import finite_entry_arc
import material_surface
import mesh_coverage
from tests.test_entry_stack_work_oracle import FACES, NORMAL, SHEETS
from tests.test_entry_stack_work_oracle import geometry as paired_geometry
from ridge_geometry import rotate


def fields(points, *, finite):
    lower, upper = points[FACES[0]], points[FACES[1]]
    q = 6371e3 * np.arcsin(lower @ NORMAL[0])
    if finite:
        left, right, _, _ = finite_entry_arc.endpoint_planes(
            NORMAL[0], np.array([1., 0., 0.]), 50., 6371.)
        endpoints = np.array([lower @ left, lower @ right])
    else:
        endpoints = np.ones((2, 3))
    upper_planes = mesh_coverage._triangle_planes(upper)
    return np.vstack((q, endpoints, (lower @ upper_planes.T).T))


def physical_polygon(points, polygon):
    lower = points[FACES[0]]
    vertices = np.array([weights @ lower for weights, _ in polygon])
    return vertices / np.linalg.norm(vertices, axis=1)[:, None]


def ledger_area(points, normal, midpoint, *, finite):
    finite_midpoints = midpoint[None] if finite else None
    half_lengths = np.array([50.]) if finite else None
    fraction = continental_entry.reference_entry_fractions(
        points[FACES[:1]], normal[None], radius_km=6371.,
        finite_midpoints=finite_midpoints,
        finite_half_lengths_km=half_lengths)
    return sum(row['area_km2'] for row in
        entry_overlap_geometry.clipped_entry_stack_intersections(
            points, FACES, SHEETS, np.array([0]), normal[None], fraction,
            radius_km=6371., finite_midpoints=finite_midpoints,
            finite_half_lengths_km=half_lengths))


class AffineOverlapDerivativeTests(unittest.TestCase):
    def test_spherical_area_field_gradient_matches_independent_area(self):
        points = paired_geometry()
        lower = points[FACES[0]]
        for finite in (False, True):
            with self.subTest(finite=finite):
                values = fields(points, finite=finite)
                area, gradient, polygon = (
                    entry_overlap_geometry.spherical_area_and_field_gradient(
                        lower, values))
                self.assertGreater(area, 0.)
                self.assertEqual(gradient.shape, (6, 3))
                self.assertGreaterEqual(len(polygon), 3)
                np.testing.assert_allclose(
                    area, ledger_area(points, NORMAL[0], np.array([1., 0., 0.]),
                                      finite=finite), rtol=3e-13)
                for field in range(6):
                    for corner in range(3):
                        step = 1e-3 if field == 0 else 1e-7
                        plus, minus = values.copy(), values.copy()
                        plus[field, corner] += step
                        minus[field, corner] -= step
                        above = entry_overlap_geometry.spherical_area_and_field_gradient(
                            lower, plus)[0]
                        below = entry_overlap_geometry.spherical_area_and_field_gradient(
                            lower, minus)[0]
                        observed = (above - below) / (2. * step)
                        np.testing.assert_allclose(
                            gradient[field, corner], observed,
                            rtol=3e-6, atol=2e-5)

    def test_spherical_area_obeys_reciprocal_two_plate_rotation_work(self):
        points = paired_geometry()
        lower, upper = points[FACES[0]], points[FACES[1]]
        normal = NORMAL[0]
        midpoint = np.array([1., 0., 0.])
        step = 2e-7
        for finite in (False, True):
            with self.subTest(finite=finite):
                values = fields(points, finite=finite)
                _, gradient, _ = (
                    entry_overlap_geometry.spherical_area_and_field_gradient(
                        lower, values))
                planes = [normal]
                if finite:
                    planes.extend(finite_entry_arc.endpoint_planes(
                        normal, midpoint, 50., 6371.)[:2])
                else:
                    planes.extend([np.zeros(3), np.zeros(3)])
                planes.extend(mesh_coverage._triangle_planes(upper))
                field_rotation = np.stack([
                    (6371e3 / np.sqrt(1. - (lower @ normal)**2))[:, None]
                    * np.cross(lower, normal) if index == 0
                    else np.cross(lower, plane)
                    for index, plane in enumerate(planes)])
                lower_torque = np.einsum('ki,kij->j', gradient, field_rotation)
                upper_torque = -lower_torque
                np.testing.assert_allclose(lower_torque + upper_torque, 0., atol=1e-9)
                for axis in np.eye(3):
                    for owner, torque in ((0, lower_torque), (1, upper_torque)):
                        def shifted(sign):
                            moved = points.copy()
                            select = slice(3 * owner, 3 * owner + 3)
                            moved[select] = rotate(moved[select], axis * sign * step)
                            moved_normal = (rotate(normal, axis * sign * step)
                                            if owner == 1 else normal)
                            moved_midpoint = (rotate(midpoint, axis * sign * step)
                                              if owner == 1 else midpoint)
                            return ledger_area(moved, moved_normal, moved_midpoint,
                                               finite=finite)
                        observed = (shifted(1.) - shifted(-1.)) / (2. * step)
                        np.testing.assert_allclose(torque @ axis, observed,
                                                   rtol=4e-6, atol=2e-4)

    def test_spherical_area_rejects_reversed_face_and_empty_overlap(self):
        lower = paired_geometry()[FACES[0]]
        with self.assertRaisesRegex(ValueError, 'outward nondegenerate'):
            entry_overlap_geometry.spherical_area_and_field_gradient(
                lower[::-1], np.ones((1, 3)))
        area, gradient, polygon = (
            entry_overlap_geometry.spherical_area_and_field_gradient(
                lower, -np.ones((1, 3))))
        self.assertEqual(area, 0.)
        np.testing.assert_array_equal(gradient, 0.)
        self.assertEqual(polygon, [])

    def test_six_halfspaces_match_independent_spherical_clipping(self):
        points = paired_geometry()
        lower_area = float(material_surface.spherical_face_areas(
            points, FACES[:1])[0])
        for finite in (False, True):
            with self.subTest(finite=finite):
                values = fields(points, finite=finite)
                fraction, gradient, polygon = finite_entry_arc.clipped_reference_area(
                    values)
                self.assertEqual(gradient.shape, (6, 3))
                self.assertGreater(fraction, 0.)
                self.assertLess(fraction, 1.)
                spherical_area = mesh_coverage._polygon_area(
                    physical_polygon(points, polygon), 6371.)
                midpoint = np.array([[1., 0., 0.]]) if finite else None
                half = np.array([50.]) if finite else None
                entry_fraction = continental_entry.reference_entry_fractions(
                    points[FACES[:1]], NORMAL, radius_km=6371.,
                    finite_midpoints=midpoint, finite_half_lengths_km=half)
                ledger = entry_overlap_geometry.clipped_entry_stack_intersections(
                    points, FACES, SHEETS, np.array([0]), NORMAL,
                    entry_fraction, radius_km=6371.,
                    finite_midpoints=midpoint, finite_half_lengths_km=half,
                    include_polygon=True)
                np.testing.assert_allclose(
                    spherical_area, sum(row['area_km2'] for row in ledger),
                    rtol=3e-13)
                self.assertLess(fraction * lower_area, lower_area)

    def test_field_derivatives_track_moving_vertices_and_area(self):
        values = fields(paired_geometry(), finite=True)
        fraction, gradient, polygon = finite_entry_arc.clipped_reference_area(
            values)
        self.assertEqual(len(polygon), 5)
        self.assertGreater(fraction, 0.)
        for field in range(len(values)):
            for corner in range(3):
                step = 1e-3 if field == 0 else 1e-7
                plus, minus = values.copy(), values.copy()
                plus[field, corner] += step
                minus[field, corner] -= step
                above, _, moved_plus = finite_entry_arc.clipped_reference_area(
                    plus)
                below, _, moved_minus = finite_entry_arc.clipped_reference_area(
                    minus)
                self.assertEqual(len(moved_plus), len(polygon))
                self.assertEqual(len(moved_minus), len(polygon))
                observed = (above - below) / (2. * step)
                np.testing.assert_allclose(
                    observed, gradient[field, corner],
                    rtol=2e-6, atol=2e-9)
                for original, shifted_plus, shifted_minus in zip(
                        polygon, moved_plus, moved_minus):
                    vertex_difference = (
                        shifted_plus[0] - shifted_minus[0]) / (2. * step)
                    np.testing.assert_allclose(
                        vertex_difference,
                        original[1][:, 3 * field + corner],
                        rtol=2e-6, atol=2e-9)

    def test_inactive_field_scaling_preserves_area_and_zero_work(self):
        values = fields(paired_geometry(), finite=False)
        fraction, gradient, _ = finite_entry_arc.clipped_reference_area(values)
        self.assertTrue(np.array_equal(gradient[1:3], np.zeros((2, 3))))
        scaled = values.copy()
        scaled[1] *= 7.
        scaled[2] *= .25
        updated, derivative, _ = finite_entry_arc.clipped_reference_area(
            scaled)
        self.assertAlmostEqual(updated, fraction, places=14)
        np.testing.assert_allclose(derivative, gradient, atol=2e-15)

    def test_rejects_invalid_and_empty_field_sets(self):
        for invalid in (np.empty((0, 3)), np.ones((2, 2)),
                        np.array([[1., np.nan, 2.]])):
            with self.subTest(invalid=invalid.shape):
                with self.assertRaisesRegex(ValueError, 'nodal halfspaces'):
                    finite_entry_arc.clip_fields(invalid)
        empty, gradient, polygon = finite_entry_arc.clipped_reference_area(
            np.full((2, 3), -1.))
        self.assertEqual(empty, 0.)
        np.testing.assert_array_equal(gradient, 0.)
        self.assertEqual(polygon, [])


if __name__ == '__main__':
    unittest.main()
