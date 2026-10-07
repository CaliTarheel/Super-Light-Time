"""Independent checks of material-reference to spherical area conversion."""

import unittest

import numpy as np

import continental_entry
import entry_overlap_geometry as geometry
import mesh_coverage
from ridge_geometry import rotate
from tests.test_entry_stack_work_oracle import FACES, NORMAL, SHEETS
from tests.test_entry_stack_work_oracle import geometry as stacked_geometry


def clipped_reference_integral(polygon, lower, radius_km=6371.):
    """Integrate the area ratio on a clipped barycentric polygon by Gauss rule."""
    vertices = np.linalg.solve(lower.T, np.asarray(polygon).T).T
    vertices /= vertices.sum(axis=1)[:, None]
    nodes, weights = np.polynomial.legendre.leggauss(20)
    nodes = (nodes + 1.) / 2.
    weights = weights / 2.
    u, v = np.meshgrid(nodes, nodes, indexing='ij')
    duffy = np.stack([1. - u - (1. - u) * v, u, (1. - u) * v], axis=-1)
    face_area = mesh_coverage._polygon_area(lower, radius_km)
    total = 0.
    for second in range(1, len(vertices) - 1):
        a, b, c = vertices[[0, second, second + 1]]
        fraction = ((b[1] - a[1]) * (c[2] - a[2])
                    - (b[2] - a[2]) * (c[1] - a[1]))
        sample = duffy @ np.array([a, b, c])
        ratio = geometry.projective_area_ratio(
            lower, sample, radius_km=radius_km)
        total += 2. * face_area * fraction * np.sum(
            ratio * weights[:, None] * weights[None, :] * (1. - u))
    return total


class ProjectiveAreaRatioTests(unittest.TestCase):
    def test_point_jacobian_matches_finite_difference_of_spherical_map(self):
        lower = stacked_geometry()[FACES[0]]
        bary = np.array([.2, .3, .5])
        step = 1e-6

        def projected(u, v):
            point = (1. - u - v) * lower[0] + u * lower[1] + v * lower[2]
            return point / np.linalg.norm(point)

        u, v = bary[1:]
        derivative_u = (projected(u + step, v)
                        - projected(u - step, v)) / (2. * step)
        derivative_v = (projected(u, v + step)
                        - projected(u, v - step)) / (2. * step)
        physical = 6371.**2 * np.linalg.norm(
            np.cross(derivative_u, derivative_v))
        reference = 2. * mesh_coverage._polygon_area(lower, 6371.)
        measured = physical / reference
        np.testing.assert_allclose(
            geometry.projective_area_ratio(lower, bary), measured,
            rtol=2e-10)

    def test_whole_face_reference_mean_is_one_and_rotation_invariant(self):
        lower = stacked_geometry()[FACES[0]]
        nodes, weights = np.polynomial.legendre.leggauss(24)
        nodes = (nodes + 1.) / 2.
        weights = weights / 2.
        u, v = np.meshgrid(nodes, nodes, indexing='ij')
        bary = np.stack([1. - u - (1. - u) * v, u, (1. - u) * v],
                        axis=-1)
        ratio = geometry.projective_area_ratio(lower, bary)
        reference_mean = 2. * np.sum(
            ratio * weights[:, None] * weights[None, :] * (1. - u))
        np.testing.assert_allclose(reference_mean, 1., rtol=2e-14)
        turned = geometry.projective_area_ratio(
            rotate(lower, [.2, -.1, .3]), bary)
        np.testing.assert_allclose(turned, ratio, rtol=3e-13)
        self.assertGreater(float(ratio.max() - ratio.min()), 1e-4)

    def test_partial_clipped_reference_integral_recovers_spherical_area(self):
        points = stacked_geometry()
        fraction = continental_entry.reference_entry_fractions(
            points[FACES[:1]], NORMAL, radius_km=6371.)
        pieces = geometry.clipped_entry_stack_intersections(
            points, FACES, SHEETS, np.array([0]), NORMAL, fraction,
            radius_km=6371., include_polygon=True)
        observed = 0.
        expected = 0.
        for piece in pieces:
            observed += clipped_reference_integral(
                piece['polygon'], points[FACES[0]])
            expected += piece['area_km2']
        self.assertGreater(expected, 0.)
        np.testing.assert_allclose(observed, expected, rtol=3e-12)

    def test_rejects_invalid_reference_geometry(self):
        lower = stacked_geometry()[FACES[0]]
        with self.assertRaisesRegex(ValueError, 'material barycentric'):
            geometry.projective_area_ratio(lower, [.2, .3, .6])
        with self.assertRaisesRegex(ValueError, 'outward'):
            geometry.projective_area_ratio(lower[[0, 2, 1]], [.2, .3, .5])
        with self.assertRaisesRegex(ValueError, 'material barycentric'):
            geometry.projective_area_ratio(lower * 2., [.2, .3, .5])


if __name__ == '__main__':
    unittest.main()
