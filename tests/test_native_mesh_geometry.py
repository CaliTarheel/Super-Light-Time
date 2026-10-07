"""Native mesh invariants: topology, spherical coverage and material sampling."""
import math
import unittest

import numpy as np

from mesh_geometry import (RADIUS_KM, build_locator, connected_components,
                           geometry, icosphere, locate_points, sample_vertices,
                           spherical_area)


def rotation():
    q, _ = np.linalg.qr(np.random.default_rng(917).normal(size=(3,3)))
    q[:,0] *= np.linalg.det(q)
    return q


def normalized(points):
    return points / np.linalg.norm(points, axis=-1, keepdims=True)


class NativeMeshGeometryTests(unittest.TestCase):
    def test_closed_manifold_and_exact_global_area(self):
        for level in range(4):
            mesh = icosphere(level)
            vertices, faces, edges = mesh['vertices'], mesh['faces'], mesh['edge_vertices']
            self.assertEqual(len(faces), 20*4**level)
            self.assertEqual(len(vertices)-len(edges)+len(faces), 2)
            self.assertTrue(np.all(mesh['edge_faces'] >= 0))
            self.assertTrue(np.all(mesh['face_area'] > 0))
            self.assertAlmostEqual(mesh['face_area'].sum()/(4*math.pi*RADIUS_KM**2), 1., places=13)
            self.assertTrue(np.all(mesh['face_neighbors'] >= 0))
            for index, neighbors in enumerate(mesh['face_neighbors']):
                self.assertTrue(all(index in mesh['face_neighbors'][neighbor] for neighbor in neighbors))

    def test_exact_area_of_spherical_octant(self):
        vertices = np.eye(3)
        area = spherical_area(vertices, np.array([[0,1,2]]), radius_km=1.)
        self.assertAlmostEqual(area[0], math.pi/2., places=14)

    def test_edges_are_geometric_and_point_toward_second_face(self):
        mesh = icosphere(2)
        normal, middle = mesh['edge_normal'], mesh['edge_mid']
        direction = mesh['xyz'][mesh['edge_faces'][:,1]] - mesh['xyz'][mesh['edge_faces'][:,0]]
        a, b = mesh['vertices'][mesh['edge_vertices']].transpose(1,0,2)
        np.testing.assert_allclose(np.einsum('ij,ij->i', normal, middle), 0., atol=3e-16)
        np.testing.assert_allclose(np.einsum('ij,ij->i', normal, b-a), 0., atol=1e-16)
        self.assertTrue(np.all(np.einsum('ij,ij->i', normal, direction) > 0))
        np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1., atol=3e-16)

    def test_geometry_rotation_invariance(self):
        original = icosphere(3)
        q = rotation()
        moved = geometry(original['vertices']@q, original['faces'])
        np.testing.assert_allclose(original['face_area'], moved['face_area'], rtol=4e-14)
        np.testing.assert_allclose(original['edge_length'], moved['edge_length'], rtol=1e-14)
        np.testing.assert_allclose(original['edge_mid']@q, moved['edge_mid'], atol=5e-16)
        np.testing.assert_allclose(original['edge_normal']@q, moved['edge_normal'], atol=5e-15)
        np.testing.assert_array_equal(original['edge_faces'], moved['edge_faces'])

    def test_random_global_coverage_and_barycentric_reconstruction(self):
        mesh = icosphere(3)
        locator = build_locator(mesh['vertices'], mesh['faces'])
        points = normalized(np.random.default_rng(1).normal(size=(7000,3)))
        face, weights = locate_points(points, locator)
        self.assertTrue(np.all(face >= 0))
        self.assertTrue(np.all(weights >= 0))
        np.testing.assert_allclose(weights.sum(axis=1), 1., atol=3e-16)
        reconstructed = normalized(np.einsum('ni,nij->nj', weights, mesh['vertices'][mesh['faces'][face]]))
        np.testing.assert_allclose(reconstructed, points, atol=8e-16)

    def test_source_triangle_and_barycentric_reference(self):
        mesh = icosphere(2)
        rng = np.random.default_rng(41)
        chosen = rng.integers(0, len(mesh['faces']), 1800)
        weights = rng.uniform(.1, 1., (len(chosen),3))
        weights /= weights.sum(axis=1)[:,None]
        points = normalized(np.einsum('ni,nij->nj', weights, mesh['vertices'][mesh['faces'][chosen]]))
        found, actual = locate_points(points, build_locator(mesh['vertices'], mesh['faces']))
        np.testing.assert_array_equal(found, chosen)
        np.testing.assert_allclose(actual, weights, atol=2e-15)

    def test_rotated_mesh_poles_seam_and_edge_ties(self):
        mesh = icosphere(2)
        q = rotation()
        points = np.r_[mesh['vertices'], mesh['edge_mid'], mesh['xyz'],
                       [[0,0,1], [0,0,-1], [-1,0,0], [-1,1e-14,0], [-1,-1e-14,0]]]
        locator = build_locator(mesh['vertices'], mesh['faces'])
        moved = build_locator(mesh['vertices']@q, mesh['faces'])
        index, weights = locate_points(points, locator)
        moved_index, moved_weights = locate_points(points@q, moved)
        np.testing.assert_array_equal(index, moved_index)
        np.testing.assert_allclose(weights, moved_weights, atol=3e-14)
        self.assertTrue(np.all(index >= 0))

    def test_all_hits_preserve_incident_faces_and_overlaps(self):
        mesh = icosphere(1)
        locator = build_locator(mesh['vertices'], mesh['faces'])
        vertex = 0
        point, face, weight = locate_points(mesh['vertices'][[vertex]], locator, all_hits=True)
        expected = np.flatnonzero(np.any(mesh['faces'] == vertex, axis=1))
        np.testing.assert_array_equal(face, expected)
        np.testing.assert_array_equal(point, np.zeros(len(expected), dtype=int))
        np.testing.assert_allclose(weight.sum(axis=1), 1., atol=2e-16)
        duplicate = build_locator(mesh['vertices'], np.r_[mesh['faces'][[0]], mesh['faces'][[0]]])
        point, face, weight = locate_points(mesh['xyz'][[0]], duplicate, all_hits=True)
        np.testing.assert_array_equal(face, [0,1])
        np.testing.assert_array_equal(point, [0,0])
        np.testing.assert_array_equal(locate_points(mesh['xyz'][[0]], duplicate)[0], [0])

    def test_open_surface_never_fills_missing_material(self):
        mesh = icosphere(1)
        locator = build_locator(mesh['vertices'], mesh['faces'][[0]])
        face, weights = locate_points(np.r_[mesh['xyz'][[0]], -mesh['xyz'][[0]]], locator)
        np.testing.assert_array_equal(face, [0,-1])
        np.testing.assert_array_equal(weights[1], [0,0,0])
        open_mesh = geometry(mesh['vertices'], mesh['faces'][[0]], require_closed=False)
        np.testing.assert_array_equal(open_mesh['face_neighbors'], [[-1,-1,-1]])

    def test_empty_material_surface(self):
        locator = build_locator(np.empty((0,3)), np.empty((0,3), dtype=int))
        points = np.eye(3)
        face, weights = locate_points(points, locator)
        np.testing.assert_array_equal(face, [-1,-1,-1])
        np.testing.assert_array_equal(weights, np.zeros((3,3)))
        point, face, weights = locate_points(points, locator, all_hits=True)
        self.assertEqual(len(point), 0)
        self.assertEqual(weights.shape, (0,3))

    def test_vertex_interpolation_and_missing_fill(self):
        mesh = icosphere(2)
        locator = build_locator(mesh['vertices'], mesh['faces'])
        values = np.arange(len(mesh['vertices']), dtype=float)
        np.testing.assert_allclose(sample_vertices(values, mesh['vertices'], locator), values, atol=2e-10)
        ones = sample_vertices(np.ones((len(values),2)), mesh['xyz'], locator)
        np.testing.assert_allclose(ones, 1., atol=3e-16)
        open_locator = build_locator(mesh['vertices'], mesh['faces'][[0]])
        self.assertTrue(np.isnan(sample_vertices(values, -mesh['xyz'][[0]], open_locator)[0]))

    def test_locator_bin_resolution_is_only_acceleration(self):
        mesh = icosphere(2)
        points = normalized(np.random.default_rng(3).normal(size=(600,3)))
        coarse = locate_points(points, build_locator(mesh['vertices'], mesh['faces'], bin_resolution=1))
        fine = locate_points(points, build_locator(mesh['vertices'], mesh['faces'], bin_resolution=43))
        np.testing.assert_array_equal(coarse[0], fine[0])
        np.testing.assert_allclose(coarse[1], fine[1], atol=0.)

    def test_components_keep_ocean_zero_and_respect_disconnected_masks(self):
        values = np.array([0,0,1,1,0,0])
        edges = np.array([[0,1],[1,2],[2,3],[3,4],[4,5]])
        labels, count = connected_components(values, edges)
        np.testing.assert_array_equal(labels, [0,0,1,1,2,2])
        self.assertEqual(count, 3)
        labels, count = connected_components(np.array([True,False,True,True,False,True]), edges)
        np.testing.assert_array_equal(labels, [0,-1,1,1,-1,2])
        self.assertEqual(count, 3)
        mesh = icosphere(2)
        labels, count = connected_components(np.zeros(len(mesh['faces']), int), mesh['edge_faces'])
        self.assertEqual(count, 1)

    def test_invalid_geometry_is_rejected(self):
        mesh = icosphere(0)
        with self.assertRaises(ValueError):
            geometry(mesh['vertices'], mesh['faces'][1:])
        with self.assertRaises(ValueError):
            geometry(mesh['vertices'], mesh['faces'][:, ::-1])
        with self.assertRaises(ValueError):
            build_locator(mesh['vertices'], np.array([[0,0,1]]))
        with self.assertRaises(ValueError):
            build_locator(mesh['vertices'], np.array([[0,1,999]]))
        with self.assertRaises(ValueError):
            locate_points([[0,0,0]], build_locator(mesh['vertices'], mesh['faces']))


if __name__ == '__main__':
    unittest.main()
