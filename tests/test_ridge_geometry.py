"""Independent finite-rotation identities and exhaustive spherical cap checks."""
import unittest
import numpy as np

from ridge_geometry import RADIUS_KM, candidate_cells, finite_half_stage, rotate


def sphere_grid(width, height):
    lon, lat = np.meshgrid((np.arange(width)+.5)*2*np.pi/width-np.pi,
                          np.pi/2-(np.arange(height)+.5)*np.pi/height)
    return np.column_stack((np.cos(lat).ravel()*np.cos(lon).ravel(),
                           np.cos(lat).ravel()*np.sin(lon).ravel(), np.sin(lat).ravel()))


class RidgeGeometryTests(unittest.TestCase):
    def test_rodrigues_known_rotation_zero_tiny_and_broadcast(self):
        np.testing.assert_allclose(rotate([1., 0., 0.], [0., 0., np.pi/2]), [0., 1., 0.], atol=1e-15)
        points = np.array([[1., 2., 3.], [-2., 1., 7.]])
        np.testing.assert_array_equal(rotate(points, np.zeros(3)), points)
        small = rotate([1., 0., 0.], [0., 0., 1e-12])
        np.testing.assert_allclose(small, [1., 1e-12, 0.], atol=1e-25)
        rotations = np.array([[.2, -.4, .8], [0., 0., 0.]])
        moved = rotate(points, rotations)
        np.testing.assert_allclose(np.linalg.norm(moved, axis=1), np.linalg.norm(points, axis=1), atol=2e-15)
        np.testing.assert_allclose(rotate(moved, -rotations), points, atol=2e-15)

    def test_half_stage_exact_relative_rotation_halves_for_nonparallel_axes(self):
        a, b = np.array([.6, -.2, .3]), np.array([-.1, .8, .5])
        mid = finite_half_stage(a, b, 1.3)
        A, B, M = [rotate(np.eye(3), r).T for r in (a*1.3, b*1.3, mid)]
        # Both chronological half increments are the same proper rotation.
        np.testing.assert_allclose(M@A.T, B@M.T, atol=2e-15)
        self.assertAlmostEqual(np.linalg.det(M), 1., places=14)
        self.assertGreater(np.linalg.norm(mid-(a+b)*1.3/2), 1e-3)
        np.testing.assert_allclose(mid, finite_half_stage(b, a, 1.3), atol=1e-15)

    def test_stationary_symmetric_axis_and_common_motion(self):
        omega = np.array([.001, -.003, .005])
        np.testing.assert_array_equal(finite_half_stage(-omega, omega, 2.), np.zeros(3))
        np.testing.assert_allclose(finite_half_stage(omega, omega, 2.), omega*2, atol=1e-17)
        np.testing.assert_allclose(finite_half_stage(np.zeros(3), omega, 2.), omega, atol=1e-17)
        np.testing.assert_array_equal(finite_half_stage(omega, -omega, 0.), np.zeros(3))

    def test_shortest_midpoint_crosses_180_without_collapsing_to_identity(self):
        stage = np.deg2rad(170.)
        mid = finite_half_stage([stage, 0., 0.], [-stage, 0., 0.], 1.)
        self.assertAlmostEqual(np.linalg.norm(mid), np.pi, places=14)
        np.testing.assert_allclose(rotate([0., 1., 0.], mid), [0., -1., 0.], atol=1e-15)

    def test_batch_rotation_and_coordinate_frame_equivariance(self):
        rng = np.random.default_rng(910)
        a, b = rng.normal(size=(12, 3))*.2, rng.normal(size=(12, 3))*.2
        mid = finite_half_stage(a, b, 2.)
        np.testing.assert_allclose(mid, np.stack([finite_half_stage(x, y, 2.) for x, y in zip(a, b)]), atol=1e-15)
        common = np.array([.3, -.8, 1.2])
        changed = finite_half_stage(rotate(a, common), rotate(b, common), 2.)
        np.testing.assert_allclose(changed, rotate(mid, common), atol=6e-16)
        self.assertEqual(finite_half_stage(a, b[0], 2.).shape, (12, 3))

    def assert_caps_match_brute_force(self, centers, radii, width, height):
        centers = np.atleast_2d(centers)
        centers = centers/np.linalg.norm(centers, axis=1)[:, None]
        radii = np.broadcast_to(radii, (len(centers),))
        segments, cells = candidate_cells(centers, radii, width, height)
        xyz = sphere_grid(width, height)
        self.assertEqual(segments.dtype, np.dtype(np.int64))
        self.assertEqual(cells.dtype, np.dtype(np.int64))
        for index, (center, radius) in enumerate(zip(centers, radii)):
            expected = np.flatnonzero(xyz@center >= np.cos(min(radius/RADIUS_KM, np.pi))-32*np.finfo(float).eps)
            found = cells[segments == index]
            np.testing.assert_array_equal(np.sort(found), expected, err_msg=f'cap {index}')
            self.assertEqual(len(found), len(np.unique(found)))

    def test_candidates_cross_seam_both_poles_and_cover_whole_sphere(self):
        lon = np.deg2rad([179.7, -179.7, 12., -53.])
        lat = np.deg2rad([0., 18., 89.8, -89.8])
        centers = np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
        centers = np.vstack((centers, [0., 0., 1.], [0., 0., -1.], [1., 0., 0.]))
        self.assert_caps_match_brute_force(centers, [300., 500., 500., 500., 350., 350., np.pi*RADIUS_KM], 192, 96)

    def test_random_and_exact_tangent_caps_match_exhaustive_search(self):
        rng = np.random.default_rng(1001)
        centers = rng.normal(size=(35, 3))
        self.assert_caps_match_brute_force(centers, rng.uniform(20., 8000., 35), 97, 49)
        xyz = sphere_grid(128, 64)
        a, b = xyz[128*9+11], xyz[128*12+13]
        radius = np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))*RADIUS_KM
        self.assert_caps_match_brute_force([a, b], [radius, 0.], 128, 64)

    def test_many_local_segments_remain_sparse_and_deterministic(self):
        rng = np.random.default_rng(42)
        centers = rng.normal(size=(2000, 3))
        radius = 1.7*np.pi/96*RADIUS_KM
        segments, cells = candidate_cells(centers, radius, 192, 96)
        self.assertGreater(len(cells), 10000)
        self.assertLess(len(cells), 300000)
        self.assertEqual(len(np.unique(segments)), 2000)
        again = candidate_cells(centers, radius, 192, 96)
        np.testing.assert_array_equal(segments, again[0])
        np.testing.assert_array_equal(cells, again[1])

    def test_empty_inputs_and_invalid_contracts(self):
        a, b = candidate_cells(np.empty((0, 3)), 100., 128, 64)
        self.assertEqual(len(a)+len(b), 0)
        self.assertEqual(rotate(np.empty((0, 3)), np.zeros(3)).shape, (0, 3))
        for centers, radius in (([0., 0., 0.], 1.), ([1., 0., 0.], -1.), ([1., 0., 0.], np.inf)):
            with self.assertRaises(ValueError):
                candidate_cells(centers, radius, 128, 64)
        with self.assertRaises(ValueError):
            finite_half_stage([1., 2., 3.], [0., 0., np.nan], 2.)
        with self.assertRaises(ValueError):
            rotate([1., 2.], [0., 0., 0.])


if __name__ == '__main__':
    unittest.main()
