"""Physical interface direction must not inherit triangle-edge directions."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import mesh_geometry
from gospl_export import rotation
import native_boundary_geometry as boundary


def fixture(level=4, axis=(.37, .72, -.48), hard=False):
    mesh = mesh_geometry.icosphere(level)
    normal = np.asarray(axis, float); normal /= np.linalg.norm(normal)
    phi = mesh['xyz']@normal
    owner = (phi > 0).astype(np.int32)
    support = np.eye(2)[owner].T if hard else np.array([(1-phi)/2, (1+phi)/2])
    edge = np.flatnonzero(owner[mesh['edge_faces'][:, 0]] != owner[mesh['edge_faces'][:, 1]])
    return mesh, owner, support, edge, normal


def classify(normal_speed, shear):
    return np.where(normal_speed > np.maximum(2., .35*shear), 1,
                    np.where(normal_speed < -np.maximum(2., .35*shear), -1, 0))


class NativeBoundaryGeometryTests(unittest.TestCase):
    def test_continuous_owner_fill_switches_on_the_reconstructed_contour(self):
        mesh, owner, support, edges, _ = fixture(level=3, hard=True)
        geometry = boundary.reconstruct(mesh, owner, support, edges)
        prepared = boundary.prepare_owner_sampling(mesh, owner, support)
        middle = geometry['segments_start']+geometry['segments_end']
        middle /= np.linalg.norm(middle, axis=1)[:, None]
        normal = geometry['segment_normals']
        offset = 1e-5
        before = middle*np.cos(offset)-normal*np.sin(offset)
        after = middle*np.cos(offset)+normal*np.sin(offset)
        contacts = mesh['edge_faces'][edges[geometry['contact_index']]]
        np.testing.assert_array_equal(boundary.sample_owners(before, prepared, chunk_size=7), owner[contacts[:, 0]])
        np.testing.assert_array_equal(boundary.sample_owners(after, prepared, chunk_size=11), owner[contacts[:, 1]])
        np.testing.assert_array_equal(boundary.sample_owners(middle, prepared), np.zeros(len(middle), np.int32))

    def test_continuous_owner_sampling_is_rotation_and_batch_invariant(self):
        mesh, owner, support, _, _ = fixture(level=3)
        points = mesh_geometry.icosphere(4)['vertices']
        prepared = boundary.prepare_owner_sampling(mesh, owner, support)
        expected = boundary.sample_owners(points, prepared, chunk_size=17)
        np.testing.assert_array_equal(boundary.sample_owners(points, prepared, chunk_size=100000), expected)
        turn = rotation([.31, -.8, .49], 1.7)
        moved = mesh_geometry.geometry(mesh['vertices']@turn, mesh['faces'])
        moved['area_km2'] = mesh['area_km2'].copy()
        context = boundary.prepare_owner_sampling(moved, owner, support)
        np.testing.assert_array_equal(boundary.sample_owners(points@turn, context), expected)

    def test_onehot_uniform_owner_and_zero_support_fallback_are_explicit(self):
        mesh = mesh_geometry.icosphere(2)
        owner = np.full(len(mesh['faces']), 5, np.int32)
        support = np.zeros((12, len(owner))); support[5] = 1.
        before = support.copy()
        points = np.array([[0., 0., 1.], [0., 0., -1.], [-1., 0., 0.]])
        prepared = boundary.prepare_owner_sampling(mesh, owner, support)
        np.testing.assert_array_equal(prepared['owner_slots'], [5])
        self.assertEqual(prepared['scores'].shape, (1, len(mesh['vertices'])))
        np.testing.assert_array_equal(boundary.sample_owners(points, prepared), 5)
        np.testing.assert_array_equal(support, before)
        np.testing.assert_array_equal(owner, 5)
        zero = boundary.prepare_owner_sampling(mesh, owner, np.zeros_like(support))
        np.testing.assert_array_equal(boundary.sample_owners(points, zero), 5)
        self.assertEqual(boundary.sample_owners(np.empty((0, 3)), prepared).shape, (0,))
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            boundary.sample_owners(points, prepared, chunk_size=0)

    def test_compact_owner_slots_keep_actual_ids_and_lowest_slot_ties(self):
        mesh, binary, fraction, _, _ = fixture(level=2, axis=(1., 0., 0.))
        slots = np.array([4, 9], np.int32)
        owner = slots[binary]
        support = np.zeros((20, len(owner))); support[slots] = fraction
        prepared = boundary.prepare_owner_sampling(mesh, owner, support)
        np.testing.assert_array_equal(prepared['owner_slots'], slots)
        self.assertEqual(prepared['scores'].shape, (2, len(mesh['vertices'])))
        points = np.array([[-1., 0., 0.], [1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_array_equal(boundary.sample_owners(points, prepared), [4, 9, 4])

    def test_owner_domain_cells_match_selected_owner_and_remain_local(self):
        mesh, owner, support, edges, _ = fixture(level=3, hard=True)
        geometry = boundary.reconstruct(mesh, owner, support, edges)
        middle = geometry['segments_start']+geometry['segments_end']
        middle /= np.linalg.norm(middle, axis=1)[:, None]
        points = np.concatenate((middle+geometry['segment_normals']*1e-5,
                                 middle-geometry['segment_normals']*1e-5))
        points /= np.linalg.norm(points, axis=1)[:, None]
        context = boundary.prepare_owner_sampling(mesh, owner, support)
        expected = boundary.sample_owners(points, context)
        result, cells = boundary.sample_owners(points, context, return_cells=True, chunk_size=13)
        np.testing.assert_array_equal(result, expected)
        self.assertTrue(np.all(cells >= 0))
        np.testing.assert_array_equal(owner[cells], result)
        containing, _ = mesh_geometry.locate_points(points, context['locator'])
        self.assertGreater(np.count_nonzero(cells != containing), 0)
        for query, source in zip(containing, cells):
            nearby = np.unique(context['vertex_faces'][mesh['faces'][query]][context['vertex_face_valid'][mesh['faces'][query]]])
            for _ in range(2): nearby = np.unique(np.r_[nearby, mesh['face_neighbors'][nearby].ravel()])
            self.assertIn(source, nearby)
        turn = rotation([.31, -.8, .49], 1.7)
        rotated = mesh_geometry.geometry(mesh['vertices']@turn, mesh['faces'])
        rotated['area_km2'] = mesh['area_km2'].copy()
        moved = boundary.prepare_owner_sampling(rotated, owner, support)
        rotated_owner, rotated_cells = boundary.sample_owners(points@turn, moved, return_cells=True)
        np.testing.assert_array_equal(rotated_owner, result)
        np.testing.assert_array_equal(rotated_cells, cells)

    def test_missing_local_owner_returns_unknown_cell_without_using_remote_parent(self):
        mesh = mesh_geometry.icosphere(3)
        labels = np.zeros(len(mesh['faces']), np.int32)
        query = 500; near = int(mesh['face_neighbors'][query, 0]); remote = 0
        labels[[near, remote]] = 1
        scores = np.zeros((2, len(mesh['vertices']))); scores[1] = 1.
        context = boundary.owner_sampling_context(mesh, np.array([0, 1], np.int32), scores, labels)
        distance = np.minimum(np.sum((mesh['xyz']-mesh['xyz'][near])**2, axis=1),
                              np.sum((mesh['xyz']-mesh['xyz'][remote])**2, axis=1))
        unsupported = int(distance.argmax())
        points = mesh['xyz'][[query, unsupported]]
        result = boundary.sample_owner_cells(points, context, chunk_size=1)
        np.testing.assert_array_equal(result['owners'], [1, 1])
        np.testing.assert_array_equal(result['owners'], boundary.sample_owners(points, context))
        np.testing.assert_array_equal(result['cells'], [near, -1])
        self.assertEqual(result['diagnostics']['unresolved_queries'], 1)
        self.assertEqual(result['diagnostics']['queries'], 2)

    def test_pure_transform_rotation_does_not_alternate_false_ridges_and_trenches(self):
        for hard in (False, True):
            mesh, owner, support, edges, axis = fixture(hard=hard)
            result = boundary.reconstruct(mesh, owner, support, edges)
            a, b = mesh['edge_faces'][edges].T
            relative = np.where(owner[b] == 1, 1., -1.)[:, None]*axis*.01
            wrong = []
            for middle, normal in ((mesh['edge_mid'][edges], mesh['edge_normal'][edges]),
                                   (result['midpoints'], result['normals'])):
                dv = np.cross(relative, middle)*6371.
                speed = np.sum(dv*normal, axis=1)
                shear = np.linalg.norm(dv-speed[:, None]*normal, axis=1)
                wrong.append(int(np.count_nonzero(classify(speed, shear))))
            self.assertGreater(wrong[0], .75*len(edges))
            self.assertEqual(wrong[1], 0)
            self.assertTrue(result['refined'].all())
            self.assertLess(abs(result['lengths'].sum()/(2*np.pi*6371.)-1), .015)

    def test_resolved_opening_and_compression_keep_their_physical_sign(self):
        mesh, owner, support, edges, axis = fixture()
        result = boundary.reconstruct(mesh, owner, support, edges)
        a, b = mesh['edge_faces'][edges].T
        # A different Euler axis produces opening on one semicircle and
        # closing on the other. Away from the true transform junctions both
        # signs must agree with the analytic interface, not be suppressed.
        euler = np.cross(axis, [0., 0., 1.])*.01
        signed = np.where(owner[b] == 1, 1., -1.)
        dv = np.cross(signed[:, None]*euler, result['midpoints'])*6371.
        expected = np.sum(dv*(signed[:, None]*axis), axis=1)
        actual = np.sum(dv*result['normals'], axis=1)
        resolved = np.abs(expected) > 20.
        self.assertGreater(np.count_nonzero(expected > 20.), 20)
        self.assertGreater(np.count_nonzero(expected < -20.), 20)
        np.testing.assert_array_equal(np.sign(actual[resolved]), np.sign(expected[resolved]))

    def test_contour_closes_at_native_vertices_and_poles_without_duplicate_edges(self):
        mesh, owner, support, edges, _ = fixture(level=3, axis=(1., 0., 0.))
        result = boundary.reconstruct(mesh, owner, support, edges)
        self.assertEqual(result['diagnostics']['fallback_edges'], 0)
        ends = np.concatenate((result['segments_start'], result['segments_end']))
        self.assertTrue(np.isfinite(ends).all())
        np.testing.assert_allclose(np.linalg.norm(ends, axis=1), 1., atol=3e-15)
        self.assertLess(np.max(np.abs(ends[:, 0])), 1e-12)
        _, count = np.unique(np.round(ends, 10), axis=0, return_counts=True)
        np.testing.assert_array_equal(count, 2)
        lengths = np.arctan2(np.linalg.norm(np.cross(result['segments_start'], result['segments_end']), axis=1),
                             np.sum(result['segments_start']*result['segments_end'], axis=1))*6371.
        self.assertAlmostEqual(lengths.sum(), 2*np.pi*6371., places=7)
        self.assertAlmostEqual(result['lengths'].sum(), lengths.sum(), places=7)

    def test_triple_junction_contours_stop_where_third_owner_dominates(self):
        mesh = mesh_geometry.icosphere(3)
        angles = np.arange(3)*2*np.pi/3+.13
        directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(3)))
        directions = directions@rotation([.3, .7, .1], .8)
        support = (1+.4*directions@mesh['xyz'].T)/3
        owner = support.argmax(axis=0).astype(np.int32)
        result = boundary.reconstruct(mesh, owner, support)
        self.assertGreater(result['diagnostics']['third_owner_clipped_segments'], 0)
        # A discrete contact immediately beside a triple point can disappear
        # from the interpolated scores. It remains an explicit raw fallback.
        self.assertLess(result['diagnostics']['fallback_edges'], .05*len(result['refined']))
        edge = mesh['edge_faces'][owner[mesh['edge_faces'][:, 0]] != owner[mesh['edge_faces'][:, 1]]]
        use = result['refined'][result['contact_index']]
        pair = owner[edge[result['contact_index'][use]]]
        middle = result['segments_start'][use]+result['segments_end'][use]
        middle /= np.linalg.norm(middle, axis=1)[:, None]
        cells, bary = mesh_geometry.locate_points(middle, mesh_geometry.build_locator(mesh['vertices'], mesh['faces']))
        scores = boundary._vertex_support(mesh, support)
        values = np.sum(scores[:, mesh['faces'][cells]]*bary[None, :, :], axis=2).T
        p, q = pair.T; rows = np.arange(len(pair))
        np.testing.assert_allclose(values[rows, p], values[rows, q], atol=1e-12)
        self.assertTrue(np.all(values[rows, p] >= values.max(axis=1)-1e-12))
        self.assertTrue(np.all(values[rows, q] >= values.max(axis=1)-1e-12))

    def test_geometry_and_contact_keys_rotate_equivariantly(self):
        mesh, owner, support, edges, _ = fixture(level=3)
        expected = boundary.reconstruct(mesh, owner, support, edges)
        turn = rotation([.31, -.8, .49], 1.7)
        moved = mesh_geometry.geometry(mesh['vertices']@turn, mesh['faces'])
        # Scores represent the same material values under a change of basis.
        moved['area_km2'] = mesh['area_km2'].copy()
        result = boundary.reconstruct(moved, owner, support, edges)
        np.testing.assert_array_equal(result['contact_index'], expected['contact_index'])
        np.testing.assert_array_equal(result['refined'], expected['refined'])
        for field in ('midpoints', 'normals', 'segments_start', 'segments_end', 'segment_normals'):
            np.testing.assert_allclose(result[field], expected[field]@turn, atol=2e-12, err_msg=field)
        np.testing.assert_allclose(result['lengths'], expected['lengths'], atol=1e-8)

    def test_existing_a_to_b_orientation_is_preserved_even_for_reordered_subset(self):
        mesh, owner, support, edges, axis = fixture(level=3)
        selected = edges[::-1]
        result = boundary.reconstruct(mesh, owner, support, selected)
        a, b = mesh['edge_faces'][selected].T
        signed = np.where(owner[b] == 1, 1., -1.)
        alignment = np.sum(result['normals']*axis, axis=1)*signed
        self.assertTrue(np.all(alignment > .99))
        pair = owner[mesh['edge_faces'][selected[result['contact_index']]]]
        sign = np.where(pair[:, 1] == 1, 1., -1.)
        self.assertTrue(np.all(np.sum(result['segment_normals']*axis, axis=1)*sign > .99))

    def test_unresolved_single_cell_plate_keeps_explicit_raw_fallback(self):
        mesh = mesh_geometry.icosphere(3)
        owner = np.zeros(len(mesh['faces']), np.int32); owner[200] = 1
        support = np.eye(2)[owner].T
        edge = np.flatnonzero(owner[mesh['edge_faces'][:, 0]] != owner[mesh['edge_faces'][:, 1]])
        before = deepcopy(support)
        result = boundary.reconstruct(mesh, owner, support)
        np.testing.assert_array_equal(result['midpoints'], mesh['edge_mid'][edge])
        np.testing.assert_array_equal(result['normals'], mesh['edge_normal'][edge])
        np.testing.assert_array_equal(result['lengths'], mesh['edge_length'][edge])
        self.assertEqual(result['diagnostics']['fallback_edges'], len(edge))
        self.assertEqual(len(result['segments_start']), len(edge))
        np.testing.assert_array_equal(support, before)

    def test_no_contacts_and_invalid_fields_have_explicit_behavior(self):
        mesh = mesh_geometry.icosphere(1)
        owner = np.zeros(len(mesh['faces']), np.int32)
        support = np.ones((1, len(owner)))
        result = boundary.reconstruct(mesh, owner, support)
        self.assertEqual(result['midpoints'].shape, (0, 3))
        self.assertEqual(result['segments_start'].shape, (0, 3))
        with self.assertRaisesRegex(ValueError, 'two-owner'):
            boundary.reconstruct(mesh, owner, support, [0])
        support[0, 2] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite native'):
            boundary.reconstruct(mesh, owner, support)


class CapturedAseinBoundaryGeometryTests(unittest.TestCase):
    """Exact accepted 38 Myr P1 scores; no simulation constructor or step."""

    @staticmethod
    def captured(turn=None):
        path = Path(__file__).parent/'fixtures'/'asein_lite_boundary38.npz'
        with np.load(path, allow_pickle=False) as archive:
            data = {name: archive[name].copy() for name in archive.files}
        vertices = data['mesh_vertices'] if turn is None else data['mesh_vertices']@turn
        mesh = mesh_geometry.geometry(vertices, data['mesh_faces'])
        mesh['vertices'] = vertices.copy()
        mesh['area_km2'] = data['mesh_area_km2'].copy()
        lookup = {tuple(sorted(edge)): i for i, edge in enumerate(mesh['edge_vertices'])}
        edges = np.array([lookup[tuple(sorted(edge))] for edge in data['native_boundary_edges']], np.int64)
        scores = np.zeros((int(data['mesh_owner_slots'].max())+1, len(vertices)))
        scores[data['mesh_owner_slots']] = data['mesh_vertex_support']
        # Only the saved exact vertex reconstruction is mocked. All native
        # contour extraction, clipping, allocation and fallback proofs run.
        support = np.zeros((len(scores), len(mesh['faces'])))
        return data, mesh, edges, scores, support

    def test_two_redundant_raw_edges_are_removed_without_moving_a_p1_piece(self):
        data, mesh, edges, scores, support = self.captured()
        before = deepcopy((mesh, data['mesh_plate'], scores, support, edges))
        with patch.object(boundary, '_vertex_support', return_value=scores.copy()):
            result = boundary.reconstruct(mesh, data['mesh_plate'], support, edges)
        zero = data['redundant_contacts']
        expected_length = data['original_lengths'].copy(); expected_length[zero] = 0.
        np.testing.assert_array_equal(result['lengths'], expected_length)
        for field in ('midpoints', 'normals', 'refined'):
            np.testing.assert_array_equal(result[field], data['original_'+field])
        keep = ~np.isin(data['original_contact_index'], zero)
        for field in ('segments_start', 'segments_end', 'segment_normals', 'contact_index'):
            np.testing.assert_array_equal(result[field], data['original_'+field][keep])
        self.assertEqual(result['diagnostics']['suppressed_redundant_contacts'], [52, 163])
        self.assertEqual(result['diagnostics']['zero_geometry_edges'], 2)
        self.assertEqual(result['diagnostics']['fallback_edges'], 0)
        self.assertEqual(result['diagnostics']['contour_segments'], 615)
        self.assertEqual(result['diagnostics']['unmapped_segments'], 0)
        self.assertAlmostEqual(result['diagnostics']['suppressed_raw_length_km'],
                               float(data['original_lengths'][zero].sum()), places=9)
        physical_length = 6371.*np.arctan2(
            np.linalg.norm(np.cross(result['segments_start'], result['segments_end']), axis=1),
            np.sum(result['segments_start']*result['segments_end'], axis=1))
        self.assertAlmostEqual(float(physical_length.sum()), float(result['lengths'].sum()), places=8)
        for original, current in zip(before, (mesh, data['mesh_plate'], scores, support, edges)):
            if isinstance(original, dict):
                for key in original:
                    np.testing.assert_array_equal(original[key], current[key])
            else:
                np.testing.assert_array_equal(original, current)

    def test_captured_redundancy_proof_rotates_and_preserves_reordered_contact_ids(self):
        turn = rotation([.31, -.8, .49], 1.7)
        data, mesh, edges, scores, support = self.captured(turn)
        with patch.object(boundary, '_vertex_support', return_value=scores.copy()):
            result = boundary.reconstruct(mesh, data['mesh_plate'], support, edges[::-1])
        zero = len(edges)-1-data['redundant_contacts']
        self.assertEqual(result['diagnostics']['suppressed_redundant_contacts'], sorted(zero.tolist()))
        expected_length = data['original_lengths'][::-1].copy(); expected_length[zero] = 0.
        np.testing.assert_allclose(result['lengths'], expected_length, rtol=2e-13, atol=2e-9)
        np.testing.assert_allclose(result['normals'], data['original_normals'][::-1]@turn, atol=2e-12)
        self.assertEqual(result['diagnostics']['fallback_edges'], 0)
        self.assertEqual(result['diagnostics']['contour_segments'], 615)
        self.assertTrue(np.all(~np.isin(result['contact_index'], zero)))

    def test_nearby_same_pair_contour_cannot_replace_a_missing_incident_piece(self):
        data, mesh, edges, scores, support = self.captured()
        original = boundary._pair_segments
        faces = mesh['edge_faces'][edges[52]]
        def missing_incident(mesh, scores, p, q):
            result = original(mesh, scores, p, q)
            if (p, q) == (0, 2):
                keep = result['face'] != faces[0]
                result = {key: value[keep] if isinstance(value, np.ndarray) else value
                          for key, value in result.items()}
            return result
        with patch.object(boundary, '_vertex_support', return_value=scores.copy()), \
                patch.object(boundary, '_pair_segments', side_effect=missing_incident):
            result = boundary.reconstruct(mesh, data['mesh_plate'], support, edges)
        self.assertEqual(result['lengths'][52], data['original_lengths'][52])
        self.assertNotIn(52, result['diagnostics']['suppressed_redundant_contacts'])
        self.assertTrue(np.any(result['contact_index'] == 52))

    def test_strict_crossing_proof_rejects_triple_ties_endpoints_and_disconnected_pieces(self):
        _, mesh, edges, scores, _ = self.captured()
        edge = edges[52]; faces = mesh['edge_faces'][edge]
        pieces = boundary._pair_segments(mesh, scores, 0, 2)
        self.assertIsNotNone(boundary._represented_crossing(mesh, scores, 0, 2, edge, faces, pieces))
        ambiguous = scores.copy()
        vertices = np.unique(mesh['faces'][faces])
        ambiguous[1, vertices] = .5*(ambiguous[0, vertices]+ambiguous[2, vertices])
        self.assertIsNone(boundary._represented_crossing(mesh, ambiguous, 0, 2, edge, faces, pieces))
        endpoint = scores.copy(); vertex = mesh['edge_vertices'][edge, 0]
        endpoint[0, vertex] = endpoint[2, vertex]
        self.assertIsNone(boundary._represented_crossing(mesh, endpoint, 0, 2, edge, faces, pieces))
        remote = deepcopy(pieces)
        row = np.flatnonzero(remote['face'] == faces[0])[0]
        remote['start'][row] = mesh['vertices'][10]
        remote['end'][row] = mesh['vertices'][11]
        self.assertIsNone(boundary._represented_crossing(mesh, scores, 0, 2, edge, faces, remote))

    def test_failed_proof_keeps_the_original_explicit_fallback(self):
        data, mesh, edges, scores, support = self.captured()
        with patch.object(boundary, '_vertex_support', return_value=scores.copy()), \
                patch.object(boundary, '_represented_crossing', return_value=None):
            result = boundary.reconstruct(mesh, data['mesh_plate'], support, edges)
        for field in ('midpoints', 'normals', 'lengths', 'refined', 'segments_start',
                      'segments_end', 'segment_normals', 'contact_index'):
            np.testing.assert_array_equal(result[field], data['original_'+field])
        self.assertEqual(result['diagnostics']['fallback_edges'], 2)
        self.assertEqual(result['diagnostics']['zero_geometry_edges'], 0)


if __name__ == '__main__':
    unittest.main()
