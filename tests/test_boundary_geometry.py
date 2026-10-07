import unittest

import numpy as np

from boundary_geometry import refine_boundary_geometry


def fixture(width=256, direction=(.45, -.62, .64), threshold=0., soft=True, third=False, height=None):
    height, radius = width//2 if height is None else height, 6371.
    lon, lat = np.meshgrid((np.arange(width)+.5)*2*np.pi/width-np.pi,
                          np.pi/2-(np.arange(height)+.5)*np.pi/height)
    xyz = np.column_stack((np.cos(lat).ravel()*np.cos(lon).ravel(),
                           np.cos(lat).ravel()*np.sin(lon).ravel(), np.sin(lat).ravel()))
    direction = np.asarray(direction, float)
    direction /= np.linalg.norm(direction)
    distance = xyz @ direction-threshold
    q = .5+.5*np.tanh(distance/(np.pi/height*.65)) if soft else (distance > 0).astype(float)
    support = np.stack((1-q, q, np.zeros(len(q))))
    if third:
        cap = xyz[:, 2] > .5
        support[:, cap] = 0
        support[2, cap] = 1
    plate = np.argmax(support, axis=0).astype(np.int16)
    ids = np.arange(width*height).reshape(height, width)
    a = np.concatenate((ids.ravel(), ids[:-1].ravel()))
    b = np.concatenate((np.roll(ids, -1, axis=1).ravel(), ids[1:].ravel()))
    selected = plate[a] != plate[b]
    a, b = a[selected], b[selected]
    mid = xyz[a]+xyz[b]
    mid /= np.linalg.norm(mid, axis=1)[:, None]
    normal = xyz[b]-xyz[a]
    normal -= mid*np.sum(normal*mid, axis=1)[:, None]
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    lat_edges = np.pi/2-np.arange(height+1)*np.pi/height
    lengths = np.concatenate((np.full(width*height, np.pi*radius/height),
                              np.repeat(2*np.pi*radius/width*np.cos(lat_edges[1:-1]), width)))[selected]
    return (plate, support, xyz, width, height, a, b, mid, normal, lengths), direction


class BoundaryGeometryTests(unittest.TestCase):
    def test_oblique_straight_contact_has_coherent_normals_and_physical_length(self):
        args, direction = fixture()
        result = refine_boundary_geometry(*args)
        signs = np.where(args[0][args[6]] == 1, 1., -1.)
        expected = direction-result['midpoints']*(result['midpoints'] @ direction)[:, None]
        expected /= np.linalg.norm(expected, axis=1)[:, None]
        expected *= signs[:, None]
        dots = np.sum(result['normals']*expected, axis=1).clip(-1, 1)
        error = np.degrees(np.arccos(dots))
        self.assertGreater(float(result['refined'].mean()), .98)
        self.assertLess(float(np.quantile(error, .95)), 5.)
        self.assertLess(float(error.mean()), 2.)
        circumference = 2*np.pi*6371.
        self.assertGreater(args[-1].sum()/circumference, 1.15)
        self.assertLess(abs(result['lengths'].sum()/circumference-1), .015)
        # One closing velocity along the true normal should not alternate into
        # transform classifications as staircase face normals change direction.
        closing = -12*expected
        normal_speed = np.sum(closing*result['normals'], axis=1)
        tangential = np.linalg.norm(closing-normal_speed[:, None]*result['normals'], axis=1)
        self.assertTrue(np.all(normal_speed < -np.maximum(2., tangential*.35)))

    def test_hard_continental_anchor_edges_improve_without_random_curvature(self):
        args, direction = fixture(512, soft=False)
        result = refine_boundary_geometry(*args)
        signs = np.where(args[0][args[6]] == 1, 1., -1.)
        expected = direction-result['midpoints']*(result['midpoints'] @ direction)[:, None]
        expected /= np.linalg.norm(expected, axis=1)[:, None]
        expected *= signs[:, None]
        old_error = np.degrees(np.arccos(np.sum(args[-2]*expected, axis=1).clip(-1, 1)))
        error = np.degrees(np.arccos(np.sum(result['normals']*expected, axis=1).clip(-1, 1)))
        self.assertLess(float(error.mean()), float(old_error.mean())*.3)
        self.assertLess(float(np.quantile(error, .95)), 15.)
        self.assertLess(abs(result['lengths'].sum()/(2*np.pi*6371.)-1), .025)

    def test_grid_aligned_straight_meridians_and_equator_stay_straight(self):
        for direction in ((0, 0, 1), (1, 0, 0)):
            with self.subTest(direction=direction):
                args, _ = fixture(direction=direction)
                result = refine_boundary_geometry(*args)
                np.testing.assert_allclose(result['normals'], args[-2], atol=.005, rtol=0)
                self.assertLess(abs(result['lengths'].sum()/args[-1].sum()-1), .0001)
                self.assertTrue(np.all(result['refined']))

    def test_seam_polar_crossings_and_curved_cap_use_spherical_geometry(self):
        cases = (((.06, 1., .05), 0.), ((1., 0., 0.), 0.), ((.015, -.02, 1.), np.cos(np.radians(8))))
        for direction, threshold in cases:
            with self.subTest(direction=direction, threshold=threshold):
                args, plane = fixture(512, direction, threshold=threshold)
                result = refine_boundary_geometry(*args)
                self.assertTrue(np.isfinite(result['normals']).all())
                np.testing.assert_allclose(np.linalg.norm(result['midpoints'], axis=1), 1., atol=3e-15)
                np.testing.assert_allclose(np.linalg.norm(result['normals'], axis=1), 1., atol=3e-15)
                np.testing.assert_allclose(np.sum(result['midpoints']*result['normals'], axis=1), 0., atol=3e-15)
                expected_length = 2*np.pi*6371.*np.sqrt(1-threshold**2)
                self.assertLess(abs(result['lengths'].sum()/expected_length-1), .06)
                self.assertTrue(np.all(result['lengths'] > 0))
                self.assertTrue(np.all(result['lengths'] <= args[-1]*(1+1e-15)))

    def test_junctions_are_flagged_and_do_not_blend_an_unrelated_owner(self):
        args, _ = fixture(256, third=True)
        result = refine_boundary_geometry(*args)
        self.assertGreater(int(result['junction'].sum()), 0)
        self.assertTrue(np.all(~result['refined'][result['junction']]))
        np.testing.assert_array_equal(result['normals'][result['junction']], args[-2][result['junction']])
        np.testing.assert_array_equal(result['midpoints'][result['junction']], args[-3][result['junction']])
        np.testing.assert_array_equal(result['lengths'][result['junction']], args[-1][result['junction']])
        self.assertGreater(float(result['refined'].mean()), .9)

    def test_batches_are_deterministic_and_inputs_are_unchanged(self):
        args, _ = fixture()
        frozen = [value.copy() if isinstance(value, np.ndarray) else value for value in args]
        first = refine_boundary_geometry(*args, chunk_size=31)
        second = refine_boundary_geometry(*args, chunk_size=2048)
        for key in ('midpoints', 'normals', 'lengths'):
            np.testing.assert_allclose(first[key], second[key], atol=2e-13, rtol=0)
        np.testing.assert_array_equal(first['refined'], second['refined'])
        for original, copy in zip(args, frozen):
            np.testing.assert_array_equal(original, copy)

    def test_independent_dimensions_and_odd_width_use_true_halfturn_at_poles(self):
        for width, height in ((95, 64), (128, 96), (97, 48)):
            with self.subTest(width=width, height=height):
                args, direction = fixture(width, direction=(.06, 1., .05), height=height)
                result = refine_boundary_geometry(*args)
                self.assertGreater(float(result['refined'].mean()), .95)
                self.assertTrue(np.isfinite(result['normals']).all())
                np.testing.assert_allclose(np.linalg.norm(result['normals'], axis=1), 1., atol=5e-15)
                np.testing.assert_allclose(np.sum(result['normals']*result['midpoints'], axis=1), 0., atol=5e-15)
                self.assertLess(abs(result['lengths'].sum()/(2*np.pi*6371.)-1), .025)
                signs = np.where(args[0][args[6]] == 1, 1., -1.)
                expected = direction-result['midpoints']*(result['midpoints'] @ direction)[:, None]
                expected /= np.linalg.norm(expected, axis=1)[:, None]
                expected *= signs[:, None]
                self.assertLess(float(np.degrees(np.arccos(np.sum(result['normals']*expected, axis=1).clip(-1, 1))).mean()), 3.)

    def test_subcell_midpoints_follow_fractional_support_zero_contour(self):
        args, direction = fixture(256)
        values = list(args)
        contrast = np.clip((args[2] @ direction)/(6*np.pi/args[4]), -1, 1)
        values[1] = np.stack(((1-contrast)/2, (1+contrast)/2, np.zeros(len(contrast))))
        result = refine_boundary_geometry(*values)
        self.assertGreater(float(np.abs(args[-3] @ direction).max()), .001)
        self.assertTrue(np.all(result['refined']))
        np.testing.assert_allclose(result['midpoints'] @ direction, 0., atol=2e-15)


if __name__ == '__main__':
    unittest.main()
