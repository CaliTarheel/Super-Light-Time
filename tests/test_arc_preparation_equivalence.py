"""Independent connectivity oracles for the arc preparation fast paths."""
from collections import Counter
import unittest
import numpy as np
import arc_surface
import material_reconstruction as reconstruction


class PreparationEquivalenceTests(unittest.TestCase):
    def test_stencil_matches_lexical_vertex_owner_nodes(self):
        rng = np.random.default_rng(947)
        vertices = rng.normal(size=(50, 3))
        for dtype in (np.int32, np.int64):
            for count in (0, 1, 24, 400):
                faces = rng.integers(0, 50, (count, 3), dtype=dtype)
                for owner in (np.full(count, -7, dtype), rng.integers(0, 4, count, dtype=dtype)):
                    area = rng.uniform(.1, 10., count)
                    actual = reconstruction.prepare(vertices, faces, owner, area_km2=area)
                    keys = sorted({(int(v), int(o)) for face, o in zip(faces, owner) for v in face})
                    lookup = {key: i for i, key in enumerate(keys)}
                    expected = np.array([[lookup[int(v), int(o)] for v in face]
                                         for face, o in zip(faces, owner)], dtype=int).reshape(-1, 3)
                    np.testing.assert_array_equal(actual['face_vertices'], expected)
                    np.testing.assert_array_equal(actual['vertex_index'], [key[0] for key in keys])
                    np.testing.assert_array_equal(actual['vertex_owner'], [key[1] for key in keys])
                    totals = np.bincount(expected.ravel(), weights=np.repeat(area, 3), minlength=len(keys))
                    np.testing.assert_array_equal(actual['corner_weight'], np.repeat(area, 3)/totals[expected.ravel()])

    def test_open_edges_match_integer_pair_counter(self):
        rng = np.random.default_rng(834)
        for dtype in (np.dtype('i4'), np.dtype('i8'), np.dtype('>i8'), np.dtype('u8')):
            for count in (0, 1, 24, 400):
                faces = rng.integers(0, 40, (count, 3))
                # Include reversed/shared/nonmanifold edges and sparse large IDs.
                identities = np.arange(40, dtype=dtype)
                identities = identities + np.array(2**62 if dtype.itemsize == 8 else 1000, dtype=dtype)
                ids = rng.integers(0, 4, count)
                stencil = dict(face_vertices=faces, vertex_index=identities,
                               vertex_count=40)
                counts = Counter(tuple(sorted((int(identities[a]), int(identities[b]))))
                                 for face in faces for a, b in zip(face, np.roll(face, -1)))
                boundary = np.zeros(40, bool)
                edges = 0
                for face, arc in zip(faces, ids):
                    for a, b in zip(face, np.roll(face, -1)):
                        if arc > 0 and counts[tuple(sorted((int(identities[a]), int(identities[b]))))] == 1:
                            boundary[[a, b]] = True
                            edges += 1
                actual = arc_surface.prepare(stencil, ids)
                self.assertEqual(actual['open_edge_count'], edges)
                np.testing.assert_array_equal(actual['boundary_nodes'], boundary)
                np.testing.assert_array_equal(actual['face_boundary_mask'], boundary[faces])
                np.testing.assert_array_equal(actual['affected_faces'], np.any(boundary[faces], axis=1))


if __name__ == '__main__':
    unittest.main()
