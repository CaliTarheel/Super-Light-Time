"""Independent spherical and mechanical invariants of the local rift solver."""
import unittest

import numpy as np

from ridge_geometry import rotate
from rift_mechanics import solve_loading, RADIUS_KM


def chain(n=17, span=.3):
    angle = np.linspace(-span/2, span/2, n)
    xyz = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(n)))
    east = np.column_stack((-np.sin(angle), np.cos(angle), np.zeros(n)))
    edges = np.column_stack((np.arange(n-1), np.arange(1, n)))
    loads = np.zeros((n, 3))
    loads[0], loads[-1] = -20.*east[0], 20.*east[-1]
    weights = np.zeros(n)
    weights[[0, -1]] = 1.
    return xyz, edges, loads, weights


class RiftMechanicsTests(unittest.TestCase):
    def test_diverging_ends_extend_and_converging_ends_shorten(self):
        xyz, edges, loads, weights = chain()
        extension = solve_loading(xyz, edges, np.ones(len(xyz)), loads, load_weights=weights)
        compression = solve_loading(xyz, edges, np.ones(len(xyz)), -loads, load_weights=weights)
        self.assertTrue(extension['converged'])
        self.assertTrue(np.all(extension['edge_extension'] > 0))
        self.assertTrue(np.all(compression['edge_extension'] < 0))
        np.testing.assert_allclose(compression['edge_extension'], -extension['edge_extension'], atol=1e-12)
        np.testing.assert_allclose(extension['edge_shear'], 0., atol=1e-12)
        self.assertLessEqual(extension['iterations'], 80)
        np.testing.assert_allclose(np.sum(extension['velocity']*xyz, axis=1), 0., atol=1e-12)

    def test_weak_corridor_localizes_the_same_external_loading(self):
        xyz, edges, loads, weights = chain()
        strength = np.ones(len(xyz))
        homogeneous = solve_loading(xyz, edges, strength, loads, load_weights=weights)
        strength[7:10] = .01
        weakened = solve_loading(xyz, edges, strength, loads, load_weights=weights)
        def share(result):
            value = result['edge_extension']
            return np.sum(value[6:10])/np.sum(value)
        self.assertGreater(share(weakened), .55)
        self.assertGreater(share(weakened), 2.*share(homogeneous))
        self.assertTrue(weakened['converged'])
        self.assertGreater(np.sum(weakened['edge_extension']), 0.)

    def test_common_rigid_rotation_cannot_damage_a_plate(self):
        rng = np.random.default_rng(310)
        xyz = rng.normal(size=(80, 3))
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
        edges = np.column_stack((np.arange(79), np.arange(1, 80)))
        omega = np.array([.007, -.009, .003])
        loads = np.cross(omega, xyz)*RADIUS_KM
        result = solve_loading(xyz, edges, np.geomspace(.04, 9., len(xyz)), loads)
        np.testing.assert_array_equal(result['velocity'], 0.)
        np.testing.assert_array_equal(result['edge_extension'], 0.)
        self.assertEqual(result['iterations'], 0)
        self.assertTrue(result['converged'])
        np.testing.assert_allclose(result['removed_omega_radians_myr'][0], omega, atol=1e-14)

    def test_adding_common_rotation_does_not_change_deformation(self):
        xyz, edges, loads, weights = chain()
        # Four constrained sites distinguish internal deformation from a fit
        # to just two freely rotating endpoints.
        weights[[4, 12]] = 1.
        ordinary = solve_loading(xyz, edges, np.ones(len(xyz)), loads, load_weights=weights)
        rigid = np.cross(np.array([.004, -.006, .001]), xyz)*RADIUS_KM
        spinning = solve_loading(xyz, edges, np.ones(len(xyz)), loads+rigid, load_weights=weights)
        np.testing.assert_allclose(spinning['velocity'], ordinary['velocity'], atol=2e-9)
        np.testing.assert_allclose(spinning['edge_extension'], ordinary['edge_extension'], atol=2e-9)

    def test_globe_rotation_preserves_loading_even_across_a_pole(self):
        xyz, edges, loads, weights = chain(23)
        strength = np.linspace(.1, 2., len(xyz))
        rotation = np.array([.07, -1.55, .93])
        first = solve_loading(xyz, edges, strength, loads, load_weights=weights)
        moved = solve_loading(rotate(xyz, rotation), edges, strength,
                              rotate(loads, rotation), load_weights=weights)
        np.testing.assert_allclose(moved['velocity'], rotate(first['velocity'], rotation), atol=2e-8)
        np.testing.assert_allclose(moved['edge_extension'], first['edge_extension'], atol=2e-8)
        np.testing.assert_allclose(moved['edge_shear'], first['edge_shear'], atol=2e-8)
        np.testing.assert_allclose(moved['edge_length_km'], first['edge_length_km'], atol=2e-9)

    def test_independent_components_do_not_share_forcing_or_euler_fit(self):
        xyz, edges, loads, weights = chain()
        other_xyz = rotate(xyz, [0., .8, 0.])
        # A distant disconnected piece has rigid rotation, but no deformation.
        other_loads = np.cross([14., 5., -8.], other_xyz)
        combined = solve_loading(np.vstack((xyz, other_xyz)),
                                 np.vstack((edges, edges+len(xyz))), np.ones(2*len(xyz)),
                                 np.vstack((loads, other_loads)),
                                 load_weights=np.r_[weights, np.ones(len(xyz))])
        separate = solve_loading(xyz, edges, np.ones(len(xyz)), loads, load_weights=weights)
        self.assertEqual(combined['component_count'], 2)
        np.testing.assert_allclose(combined['velocity'][:len(xyz)], separate['velocity'], atol=2e-8)
        np.testing.assert_allclose(combined['velocity'][len(xyz):], 0., atol=2e-8)

    def test_zero_drive_and_unloaded_nodes_create_no_motion(self):
        xyz, edges, loads, weights = chain()
        for supplied, constraint in ((np.zeros_like(loads), weights), (loads, np.zeros(len(xyz)))):
            result = solve_loading(xyz, edges, np.geomspace(1e-4, 1e4, len(xyz)),
                                   supplied, load_weights=constraint)
            np.testing.assert_array_equal(result['velocity'], 0.)
            np.testing.assert_array_equal(result['edge_extension'], 0.)
            self.assertTrue(result['converged'])

    def test_iteration_budget_returns_honest_finite_partial_solution(self):
        xyz, edges, loads, weights = chain(81)
        result = solve_loading(xyz, edges, np.linspace(.1, 5., len(xyz)), loads,
                               load_weights=weights, iterations=1, tolerance=1e-12)
        self.assertEqual(result['iterations'], 1)
        self.assertFalse(result['converged'])
        self.assertTrue(np.isfinite(result['velocity']).all())
        self.assertGreater(result['relative_residual'], 1e-12)
        self.assertGreater(result['residual_norm'], 0.)

    def test_edge_order_direction_and_node_relabeling_do_not_change_solution(self):
        xyz, edges, loads, weights = chain()
        strength = np.linspace(.2, 3., len(xyz))
        first = solve_loading(xyz, edges, strength, loads, load_weights=weights)
        rng = np.random.default_rng(901)
        permutation = rng.permutation(len(xyz))
        reverse = np.argsort(permutation)
        moved = solve_loading(xyz[permutation], reverse[edges[::-1, ::-1]],
                              strength[permutation], loads[permutation], load_weights=weights[permutation])
        np.testing.assert_allclose(moved['velocity'][reverse], first['velocity'], atol=2e-9)
        np.testing.assert_allclose(moved['edge_extension'][::-1], first['edge_extension'], atol=2e-9)

    def test_isolated_nodes_and_empty_graph_have_no_internal_deformation(self):
        result = solve_loading(np.array([[1., 0., 0.], [0., 1., 0.]]),
                               np.empty((0, 2), dtype=int), np.ones(2), np.ones((2, 3)))
        np.testing.assert_array_equal(result['velocity'], 0.)
        self.assertEqual(result['component_count'], 2)
        empty = solve_loading(np.empty((0, 3)), np.empty((0, 2), dtype=int),
                              np.empty(0), np.empty((0, 3)))
        self.assertEqual(empty['node_count'], 0)
        self.assertEqual(empty['velocity'].shape, (0, 3))

    def test_inputs_are_not_mutated(self):
        xyz, edges, loads, weights = chain()
        strength = np.ones(len(xyz))
        original = [x.copy() for x in (xyz, edges, strength, loads, weights)]
        solve_loading(xyz, edges, strength, loads, load_weights=weights)
        for before, after in zip(original, (xyz, edges, strength, loads, weights)):
            np.testing.assert_array_equal(before, after)

    def test_invalid_geometry_or_coefficients_fail_without_fabricated_response(self):
        xyz, edges, loads, weights = chain()
        for keyword, value in (('iterations', 0), ('iterations', 2.5), ('regularization', 0.),
                               ('coupling_km', np.nan), ('tolerance', -1.)):
            with self.subTest(keyword=keyword), self.assertRaises(ValueError):
                solve_loading(xyz, edges, np.ones(len(xyz)), loads, **{keyword: value})
        with self.assertRaises(ValueError):
            solve_loading(xyz, edges, np.zeros(len(xyz)), loads)
        with self.assertRaises(ValueError):
            solve_loading(xyz*2, edges, np.ones(len(xyz)), loads)
        with self.assertRaises(ValueError):
            solve_loading(xyz, np.array([[0, 0]]), np.ones(len(xyz)), loads)
        with self.assertRaises(ValueError):
            solve_loading(xyz, np.array([[0., 1.]]), np.ones(len(xyz)), loads)
        with self.assertRaises(ValueError):
            solve_loading(np.array([[1., 0., 0.], [-1., 0., 0.]]),
                          np.array([[0, 1]]), np.ones(2), np.zeros((2, 3)))


if __name__ == '__main__':
    unittest.main()
