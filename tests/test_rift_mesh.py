"""Material graphs remain spherical, local and resistant to accidental shards."""
import unittest

import numpy as np

from rift_mesh import assign_cells, build_graph, coherent_cut, fibonacci_sites


def rotation():
    axis = np.array([.3, -.8, .4])
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = 1.8
    return np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis, axis)+np.sin(angle)*cross


def grid(rows, columns, spacing=.015):
    x, y = np.meshgrid(np.arange(columns)*spacing, np.arange(rows)*spacing)
    points = np.column_stack((x.ravel(), y.ravel(), np.ones(rows*columns)))
    points /= np.linalg.norm(points, axis=1)[:, None]
    edges = []
    for row in range(rows):
        for col in range(columns):
            node = row*columns+col
            if col+1 < columns:
                edges.append((node, node+1))
            if row+1 < rows:
                edges.append((node, node+columns))
    return points, np.asarray(edges, np.int32)


class FibonacciAssignmentTests(unittest.TestCase):
    def test_nearest_matches_full_search_at_multiple_budgets(self):
        rng = np.random.default_rng(102)
        points = rng.normal(size=(350, 3))
        points /= np.linalg.norm(points, axis=1)[:, None]
        points = np.vstack((points, [[0, 0, 1], [0, 0, -1], [-1, 0, 0]]))
        for budget in (4, 32, 513, 8192):
            sites = fibonacci_sites(budget)
            expected = np.argmin(np.sum((points[:, None]-sites[None])**2, axis=2), axis=1)
            np.testing.assert_array_equal(assign_cells(points, budget), expected)

    def test_reference_sites_assign_to_themselves_and_are_not_mutated(self):
        sites = fibonacci_sites(128)
        sites[0] = [0, 0, 1]
        reference = fibonacci_sites(128)
        self.assertFalse(np.array_equal(sites, reference))
        np.testing.assert_array_equal(assign_cells(reference, 128), np.arange(128))

    def test_multiple_work_batches_and_scaled_input_are_exact(self):
        rng = np.random.default_rng(32)
        points = rng.normal(size=(4500, 3))
        norm = points / np.linalg.norm(points, axis=1)[:, None]
        sites = fibonacci_sites(512)
        expected = np.argmax(norm @ sites.T, axis=1)
        np.testing.assert_array_equal(assign_cells(points, 512), expected)

    def test_invalid_vectors_and_empty_input(self):
        self.assertEqual(assign_cells(np.empty((0, 3)), 512).shape, (0,))
        for values in ([[0, 0, 0]], [[0, np.nan, 1]], [[1, 2]]):
            with self.assertRaises(ValueError):
                assign_cells(values, 512)
        for budget in (3, 4.5, True):
            with self.assertRaises(ValueError):
                fibonacci_sites(budget)


class SphericalGraphTests(unittest.TestCase):
    def test_rotation_does_not_change_material_links(self):
        rng = np.random.default_rng(17)
        points, _ = grid(12, 10)
        points += rng.normal(scale=.001, size=points.shape)
        area = np.full(len(points), 12_000.)
        owners = np.zeros(len(points), int)
        craton = np.linspace(0, 1, len(points))
        reference = build_graph(points, owners, area, craton)
        rotated = build_graph(points @ rotation().T, owners, area, craton)
        np.testing.assert_array_equal(reference, rotated)
        self.assertGreater(len(reference), len(points))

    def test_wide_empty_ocean_does_not_link_same_owner_island(self):
        points, _ = grid(4, 5)
        island = np.array([[-.6, 0, .8]])
        points = np.vstack((points, island))
        graph = build_graph(points, np.zeros(len(points), int), np.full(len(points), 12_000),
                            np.zeros(len(points)))
        self.assertFalse(np.any(graph == len(points)-1))
        self.assertGreater(len(graph), 0)

    def test_exactly_tied_neighbours_remain_stable_after_rotation(self):
        angle = np.arange(24)*2*np.pi/24
        points = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(24)))
        owners, area, craton = np.zeros(24, int), np.full(24, 3e6), np.zeros(24)
        first = build_graph(points, owners, area, craton, neighbors=1)
        second = build_graph(points @ rotation().T, owners, area, craton, neighbors=1)
        np.testing.assert_array_equal(first, second)

    def test_different_owners_and_zero_area_never_link(self):
        points, _ = grid(5, 6)
        owners = np.arange(len(points)) % 2
        owners[0] = -1
        area = np.full(len(points), 12_000.)
        area[1] = 0
        graph = build_graph(points, owners, area, np.zeros(len(points)))
        self.assertTrue(np.all(owners[graph[:, 0]] == owners[graph[:, 1]]))
        self.assertFalse(np.any(np.isin(graph, [0, 1])))

    def test_coincident_and_antipodal_nodes_do_not_make_invalid_springs(self):
        points = np.array([[0, 0, 1], [0, 0, 1], [0, 0, -1], [.01, 0, 1]])
        owners, area, craton = np.zeros(4, int), np.full(4, 1e9), np.zeros(4)
        for support in (None, [[0, 1], [0, 2], [0, 3]]):
            graph = build_graph(points, owners, area, craton, support_edges=support)
            self.assertFalse(np.any(np.all(graph == [0, 1], axis=1)))
            self.assertFalse(np.any(np.all(graph == [0, 2], axis=1)))
            self.assertTrue(np.any(np.all(graph == [0, 3], axis=1)))

    def test_explicit_footprint_contact_prevents_narrow_strait_shortcut(self):
        points, _ = grid(3, 6)
        contact = np.array([[0, 1], [1, 2], [3, 4], [4, 5], [0, 6], [6, 12]])
        owners = np.zeros(len(points), int)
        area = np.full(len(points), 12_000.)
        actual = build_graph(points, owners, area, np.zeros(len(points)), support_edges=contact)
        np.testing.assert_array_equal(actual, contact[np.lexsort((contact[:, 1], contact[:, 0]))])
        self.assertFalse(np.any(np.all(actual == [2, 3], axis=1)))

    def test_seam_and_polar_neighbours_are_local(self):
        angle = .008
        points = np.array([[-np.cos(angle), np.sin(angle), 0], [-np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), 0, np.cos(angle)], [-np.sin(angle), 0, np.cos(angle)]])
        graph = build_graph(points, [0, 0, 1, 1], np.full(4, 12_000), np.zeros(4))
        np.testing.assert_array_equal(graph, [[0, 1], [2, 3]])

    def test_graph_build_does_not_modify_inputs(self):
        points, _ = grid(4, 4)
        owners, area, craton = np.zeros(16, int), np.full(16, 12_000.), np.zeros(16)
        inputs = [points, owners, area, craton]
        copies = [value.copy() for value in inputs]
        build_graph(*inputs)
        for before, after in zip(copies, inputs):
            np.testing.assert_array_equal(before, after)


class CoherentCutTests(unittest.TestCase):
    def setUp(self):
        self.points, self.edges = grid(6, 8)
        self.owners = np.zeros(len(self.points), int)
        self.belt = (self.edges[:, 0] % 8 == 3) & (self.edges[:, 1] == self.edges[:, 0]+1)

    def test_partial_rift_keeps_intact_material_connected(self):
        failed = self.belt.copy()
        failed[np.flatnonzero(failed)[0]] = False
        self.assertEqual(coherent_cut(48, self.edges, failed, self.owners), [])

    def test_complete_failed_belt_returns_two_substantial_components(self):
        actual = coherent_cut(48, self.edges, self.belt, self.owners)
        self.assertEqual(len(actual), 1)
        self.assertEqual([len(c) for c in actual[0]['components']], [24, 24])
        np.testing.assert_array_equal(actual[0]['failed_edges'], self.edges[self.belt])
        np.testing.assert_array_equal(np.sort(np.concatenate(actual[0]['components'])), np.arange(48))

    def test_edge_mask_indices_and_endpoint_pairs_have_identical_results(self):
        alternatives = [self.belt, np.flatnonzero(self.belt), self.edges[self.belt][:, ::-1]]
        outputs = [coherent_cut(48, self.edges, failed, self.owners)[0] for failed in alternatives]
        for actual in outputs[1:]:
            for field in ('parent_nodes', 'failed_edges'):
                np.testing.assert_array_equal(actual[field], outputs[0][field])

    def test_existing_island_is_not_a_new_fracture(self):
        intact = self.edges[~self.belt]
        self.assertEqual(coherent_cut(48, intact, [], self.owners), [])
        # A failed edge in one component still has an alternate intact route.
        self.assertEqual(coherent_cut(48, intact, [0], self.owners), [])

    def test_protected_craton_prevents_whole_proposal(self):
        groups = np.full(48, -1)
        groups[[2, 3, 4, 5]] = 101
        self.assertEqual(coherent_cut(48, self.edges, self.belt, self.owners, groups), [])
        groups[:] = -1
        groups[[1, 2, 3]] = 101
        self.assertEqual(len(coherent_cut(48, self.edges, self.belt, self.owners, groups)), 1)

    def test_single_cell_shards_do_not_become_plates(self):
        failed = np.any(self.edges == 0, axis=1)
        self.assertEqual(coherent_cut(48, self.edges, failed, self.owners), [])

    def test_area_fraction_rejects_numerous_nearly_empty_edge_nodes(self):
        area = np.where(np.arange(48) % 8 < 4, 1., .01)
        self.assertEqual(coherent_cut(48, self.edges, self.belt, self.owners, area=area), [])

    def test_cross_owner_link_cannot_supply_parent_material_connectivity(self):
        self.owners[np.arange(48) % 8 >= 4] = 1
        self.assertEqual(coherent_cut(48, self.edges, self.belt, self.owners), [])

    def test_no_broken_link_and_invalid_broken_indices(self):
        self.assertEqual(coherent_cut(48, self.edges, [], self.owners), [])
        with self.assertRaises(ValueError):
            coherent_cut(48, self.edges, [len(self.edges)], self.owners)


if __name__ == '__main__':
    unittest.main()
