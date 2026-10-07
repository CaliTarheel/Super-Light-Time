"""Physical area and joint virtual work against independent scalar clipping."""
import unittest

import numpy as np

from tests.test_gpe_endpoint_derivative import symmetric_area_gradient
from tests.test_gravitational_relaxation import unit
from tests.test_small_spherical_faces import decimal_overlap


def pair(depths):
    return (unit([[-.05, depths[0], 1.], [.05, depths[1], 1.], [0., .03, 1.]]),
            unit([[-.2, 0., 1.], [.2, 0., 1.], [0., .2, 1.]]))


class PhysicalIntersectionWorkTests(unittest.TestCase):
    def test_zero_area_birth_selects_midpoint_of_joint_directional_work(self):
        first = unit([[-.1, 0., 1.], [0., -.2, 1.], [.1, 0., 1.]])
        second = unit([[-.1, 0., 1.], [.1, 0., 1.], [0., .2, 1.]])
        area, gradient = symmetric_area_gradient(first, second)
        self.assertEqual(area, 0.)
        self.assertGreater(np.linalg.norm(gradient), 0.)
        points = np.concatenate((first, second))
        rng = np.random.default_rng(9228)
        for index in range(16):
            direction = rng.normal(size=points.shape)
            direction -= points*np.sum(points*direction, axis=1)[:, None]
            step = 1e-7
            plus, minus = unit(points+step*direction), unit(points-step*direction)
            left = -decimal_overlap(minus[:3], minus[3:])/step
            right = decimal_overlap(plus[:3], plus[3:])/step
            predicted = float(np.sum(gradient*direction))
            allowance = 1e-4*max(abs(left), abs(right), 1.)
            with self.subTest(direction=index):
                self.assertLess(abs(predicted-.5*(left+right)), allowance)

    def test_scalar_area_uses_actual_halfspaces_across_old_tolerance(self):
        for depths in ((-4e-14, -6e-14), (-6e-14, -4e-14),
                       (-4.5e-14, -5.5e-14), (4e-14, 6e-14), (0., 0.)):
            first, second = pair(depths)
            actual, _ = symmetric_area_gradient(first, second)
            expected = decimal_overlap(first, second)
            with self.subTest(depths=depths):
                # The former geometric band adds about 8e-8 km2 here. This
                # oracle measures the physical area, not the engine's ledger.
                self.assertLess(abs(actual-expected), 2e-10)

    def test_one_selected_vector_obeys_joint_nodal_work_limits(self):
        rng = np.random.default_rng(92028)
        for depths in ((0., 0.), (-4e-14, -6e-14), (4e-14, 6e-14),
                       (-4e-14, 6e-14), (6e-14, -4e-14)):
            first, second = pair(depths)
            _, gradient = symmetric_area_gradient(first, second)
            points = np.concatenate((first, second))
            baseline = decimal_overlap(first, second)
            for index in range(16):
                direction = rng.normal(size=points.shape)
                direction -= points*np.sum(points*direction, axis=1)[:, None]
                step = 1e-7
                plus, minus = unit(points+step*direction), unit(points-step*direction)
                left = (baseline-decimal_overlap(minus[:3], minus[3:]))/step
                right = (decimal_overlap(plus[:3], plus[3:])-baseline)/step
                predicted = float(np.sum(gradient*direction))
                allowance = 1e-4*max(abs(left), abs(right), 1.)
                with self.subTest(depths=depths, direction=index):
                    self.assertGreaterEqual(predicted, min(left, right)-allowance)
                    self.assertLessEqual(predicted, max(left, right)+allowance)

    def test_sliding_endpoints_on_coincident_plane_matches_scalar_work(self):
        first, second = pair((0., 0.))
        _, gradient = symmetric_area_gradient(first, second)
        direction = np.array([[.7, 0., 0.], [-.4, 0., 0.], [0., 0., 0.]])
        direction -= first*np.sum(first*direction, axis=1)[:, None]
        predicted = float(np.sum(gradient[:3]*direction))
        measured = []
        for step in (1e-5, 1e-6, 1e-7):
            measured.append((decimal_overlap(unit(first+step*direction), second)
                             - decimal_overlap(unit(first-step*direction), second))/(2*step))
        np.testing.assert_allclose(measured[-1], predicted, rtol=2e-8, atol=1e-4)

    def test_cyclic_indexing_keeps_selected_gradient_and_global_torque(self):
        for depths in ((0., 0.), (-4e-14, -6e-14), (-4e-14, 6e-14)):
            first, second = pair(depths)
            value, gradient = symmetric_area_gradient(first, second)
            points = np.concatenate((first, second))
            residual = np.linalg.norm(np.cross(points, gradient).sum(axis=0))
            self.assertLess(residual, 1e-12*np.linalg.norm(gradient))
            for a in range(3):
                for b in range(3):
                    observed, changed = symmetric_area_gradient(np.roll(first, a, axis=0),
                                                               np.roll(second, b, axis=0))
                    changed = np.concatenate((np.roll(changed[:3], -a, axis=0),
                                              np.roll(changed[3:], -b, axis=0)))
                    np.testing.assert_allclose(observed, value, rtol=1e-13)
                    np.testing.assert_allclose(changed, gradient, rtol=1e-12, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
