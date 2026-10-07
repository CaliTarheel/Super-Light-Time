"""Endpoint virtual work against independent scalar spherical clipping.

An isolated vertex on another edge has a continuous first area derivative;
coincident finite edges can instead have a genuine kink. Test both contracts.
"""
import unittest

import numpy as np

import gravitational_relaxation as gravity
from ridge_geometry import rotate
from tests.test_collision_architecture import overlapping_fixture, orientation
from tests.test_gravitational_relaxation import (
    independent_energy, planar_clip, rotation, sphere_polygon_area, unit)
from tests.test_small_spherical_faces import decimal_overlap


def endpoint_pair(offset=0.):
    # Only the clipper's last corner touches the subject's bottom edge. There
    # is no coincident finite edge and therefore no finite jump in first work.
    first = unit([[-.06, 0., 1.], [.002, 0., 1.], [.002, .06, 1.]])
    second = unit([[.06, .06, 1.], [-.002, .06, 1.], [-.002, offset, 1.]])
    return first, second


def scalar_overlap(first, second):
    polygon = planar_clip(first[:, :2]/first[:, 2, None],
                          second[:, :2]/second[:, 2, None])
    if len(polygon) < 3:
        return 0.
    return sphere_polygon_area(unit(np.column_stack((polygon, np.ones(len(polygon))))))


def symmetric_area_gradient(first, second):
    value, forward, _ = gravity._paired_area_derivative(first[None], second[None], 6371.)
    _, reverse, _ = gravity._paired_area_derivative(second[None], first[None], 6371.)
    reverse = reverse.reshape(6, 3)[[3, 4, 5, 0, 1, 2]]
    return value[0], .5*(forward.reshape(6, 3)+reverse)


class GPEEndpointDerivativeTests(unittest.TestCase):
    def test_exact_and_ulp_endpoint_work_matches_independent_fixed_volume_energy(self):
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        height, reference, sheets = np.array([35., 28.]), np.array([30., 28.]), np.array([1, 2])
        for offset in (0., -np.spacing(.06), np.spacing(.06)):
            points = np.concatenate(endpoint_pair(offset))
            volumes = np.array([sphere_polygon_area(triangle) for triangle in points[faces]])*height
            value, gradient, _ = gravity.energy_gradient(points, faces, height, reference, sheets)
            np.testing.assert_allclose(value, independent_energy(points, faces, volumes, reference, sheets), rtol=2e-14)
            for seed in range(4):
                direction = np.random.default_rng(seed).normal(size=points.shape)
                direction -= points*np.sum(points*direction, axis=1)[:, None]
                expected = float(np.sum(gradient*direction))
                errors = []
                for step in (1e-6, 1e-7, 1e-8):
                    plus = independent_energy(unit(points+step*direction), faces, volumes, reference, sheets)
                    minus = independent_energy(unit(points-step*direction), faces, volumes, reference, sheets)
                    errors.append(abs((plus-minus)/(2*step)-expected)/max(abs(expected), 1.))
                with self.subTest(offset=offset, seed=seed):
                    self.assertLess(errors[-1], 5e-7)
                    self.assertLess(errors[-1], errors[0]*.05)

    def test_reconstructed_rotation_has_no_spurious_transverse_gpe_torque(self):
        # This benchmark originally generated a reduced transverse torque of
        # 8.59e7 where both one-sided energy slopes converge to zero.
        q = orientation()
        for s, basis in ((overlapping_fixture(), np.eye(3)),
                         (overlapping_fixture(rotation=q), q)):
            mesh = s.material_surface
            points, faces, owner = mesh['vertices'], mesh['faces'], mesh['vertex_owner']
            height, reference = s.structure['thickness_km'], s.structure['reference_thickness_km']
            volumes = mesh['area_km2']*height
            _, gradient, _ = gravity.energy_gradient(points, faces, height, reference, s.parcel_collision_sheet)
            torque = np.cross(points[owner == 0], gradient[owner == 0]).sum(axis=0)@basis
            self.assertLess(abs(torque[1]), 1e-10*np.linalg.norm(torque))
            # Independently differentiate the scalar conserved-volume energy.
            energy = gravity.reference_energy(points, faces, volumes, reference, s.parcel_collision_sheet)
            one_sided = []
            for step in (1e-6, 1e-7):
                axis = np.array([0., step, 0.])@basis.T
                plus, minus = points.copy(), points.copy()
                plus[owner == 0] = rotate(plus[owner == 0], axis)
                minus[owner == 0] = rotate(minus[owner == 0], -axis)
                above = gravity.reference_energy(plus, faces, volumes, reference, s.parcel_collision_sheet)
                below = gravity.reference_energy(minus, faces, volumes, reference, s.parcel_collision_sheet)
                one_sided.append(max(abs((above-energy)/step), abs((energy-below)/step)))
                self.assertLess(abs((above-below)/(2*step)-torque[1]), 1e-8*np.linalg.norm(torque))
            self.assertLess(one_sided[1], one_sided[0]*.11)

    def test_endpoint_gradient_rotations_exchange_and_action_reaction(self):
        first, second = endpoint_pair()
        value, gradient = symmetric_area_gradient(first, second)
        points = np.concatenate((first, second))
        self.assertLess(np.linalg.norm(np.cross(points, gradient).sum(axis=0)), 2e-12*np.linalg.norm(gradient))
        for q in (orientation(), rotation(.23, (.7, .1, -.2)), rotation(1.4, (-.2, .8, .4))):
            for offset in (0., -np.spacing(.06), np.spacing(.06)):
                a, b = endpoint_pair(offset)
                observed, other = symmetric_area_gradient(a@q.T, b@q.T)
                np.testing.assert_allclose(observed, value, rtol=2e-13)
                np.testing.assert_allclose(other, gradient@q.T, rtol=2e-10, atol=2e-5)
                swapped, reverse = symmetric_area_gradient(b@q.T, a@q.T)
                np.testing.assert_allclose(observed, swapped, rtol=2e-13)
                np.testing.assert_allclose(reverse[[3, 4, 5, 0, 1, 2]], other, rtol=0., atol=0.)

    def test_thin_overlap_keeps_area_and_work_without_vertex_merging(self):
        first = unit([[-.06, 0., 1.], [.06, 0., 1.], [0., .06, 1.]])
        for width in (1e-3, 1e-5, 1e-7):
            second = unit([[-.03, -.02, 1.], [.03, -.02, 1.], [0., width, 1.]])
            value, gradient = symmetric_area_gradient(first, second)
            expected = decimal_overlap(first, second)
            self.assertGreater(value, 0.)
            np.testing.assert_allclose(value, expected, rtol=2e-8, atol=1e-13)
            direction = np.tile([0., 1., 0.], (3, 1))
            direction -= second*np.sum(second*direction, axis=1)[:, None]
            step = width*1e-3
            measured = (decimal_overlap(first, unit(second+step*direction))
                        - decimal_overlap(first, unit(second-step*direction)))/(2*step)
            np.testing.assert_allclose(float(np.sum(gradient[3:]*direction)), measured, rtol=2e-6, atol=1e-6)

    def test_exact_coincident_edge_selected_work_is_in_directional_limits(self):
        first = unit([[-.05, 0., 1.], [.05, 0., 1.], [0., .03, 1.]])
        second = unit([[-.2, 0., 1.], [.2, 0., 1.], [0., .2, 1.]])
        _, gradient = symmetric_area_gradient(first, second)
        torque = np.cross(first, gradient[:3]).sum(axis=0)
        baseline = scalar_overlap(first, second)
        directions = np.concatenate((np.eye(3), unit(np.random.default_rng(628).normal(size=(24, 3)))))
        for axis in directions:
            step = 1e-7
            left = (baseline-scalar_overlap(rotate(first, -axis*step), second))/step
            right = (scalar_overlap(rotate(first, axis*step), second)-baseline)/step
            allowance = 1e-4*max(abs(left), abs(right), 1.)
            self.assertGreaterEqual(torque@axis, min(left, right)-allowance)
            self.assertLessEqual(torque@axis, max(left, right)+allowance)

    def test_near_collinear_edge_work_respects_physical_geometry_under_rotation(self):
        # Both bottom corners are physically outside. The former inside band
        # kept one of them and extrapolated a crossing outside its segment.
        second = unit([[-.2, 0., 1.], [.2, 0., 1.], [0., .2, 1.]])
        for depth in (0., 1.):
            first = unit([[-.05, -4e-14*depth, 1.], [.05, -6e-14*depth, 1.], [0., .03, 1.]])
            value, gradient = symmetric_area_gradient(first, second)
            torque = np.cross(first, gradient[:3]).sum(axis=0)
            # At this finite coincident edge, a single derivative need not
            # equal a central difference. It must lie between one-sided limits.
            axis = np.array([1., 0., 0.])
            step = 1e-7
            baseline = scalar_overlap(first, second)
            left = (baseline-scalar_overlap(rotate(first, -axis*step), second))/step
            right = (scalar_overlap(rotate(first, axis*step), second)-baseline)/step
            allowance = 1e-4*max(abs(left), abs(right), 1.)
            self.assertGreater(abs(left-right), 1e6)
            self.assertGreaterEqual(torque@axis, min(left, right)-allowance)
            self.assertLessEqual(torque@axis, max(left, right)+allowance)
            np.testing.assert_allclose(value, baseline, rtol=2e-11)
            for q in (orientation(), rotation(.4, (.1, .3, -.7))):
                _, other = symmetric_area_gradient(first@q.T, second@q.T)
                moved_torque = np.cross(first@q.T, other[:3]).sum(axis=0)
                if depth:
                    np.testing.assert_allclose(moved_torque, torque@q.T, rtol=2e-11, atol=1e-5)
                else:
                    # Rounding a finite coincident edge through a rotation can
                    # select either side of its real kink in stored geometry.
                    # Require admissible work, not a unique smooth gradient.
                    a, b = first@q.T, second@q.T
                    baseline = decimal_overlap(a, b)
                    for axis in np.eye(3):
                        left = (baseline-decimal_overlap(rotate(a, -axis*step), b))/step
                        right = (decimal_overlap(rotate(a, axis*step), b)-baseline)/step
                        allowance = 1e-4*max(abs(left), abs(right), 1.)
                        self.assertGreaterEqual(moved_torque@axis, min(left, right)-allowance)
                        self.assertLessEqual(moved_torque@axis, max(left, right)+allowance)

    def test_tolerance_band_subgradient_obeys_every_joint_direction(self):
        # The former inside band retained one genuinely outside vertex and
        # rejected the other. Its resulting vector passed Cartesian checks but
        # violated the independent work interval in this oblique direction.
        first = unit([[-.05, -4e-14, 1.], [.05, -6e-14, 1.], [0., .03, 1.]])
        second = unit([[-.2, 0., 1.], [.2, 0., 1.], [0., .2, 1.]])
        _, gradient = symmetric_area_gradient(first, second)
        torque = np.cross(first, gradient[:3]).sum(axis=0)
        axis = np.array([-.023695226048634502, .2820198531694643, .9591159151430957])
        baseline, step = scalar_overlap(first, second), 1e-7
        left = (baseline-scalar_overlap(rotate(first, -axis*step), second))/step
        right = (scalar_overlap(rotate(first, axis*step), second)-baseline)/step
        allowance = 1e-4*max(abs(left), abs(right), 1.)
        self.assertGreaterEqual(torque@axis, min(left, right)-allowance)
        self.assertLessEqual(torque@axis, max(left, right)+allowance)


if __name__ == '__main__':
    unittest.main()
