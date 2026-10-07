"""Subtracting a material union prunes pairs it can prove cannot meet.

`_remaining` is the simulation's hot spot: on run SEP21T at 156 Myr a single
arc admission spent ~47 s and made ~68,000 face-against-piece subtractions,
because the candidate faces are chosen once for the whole polygon and were
then each subtracted from every current piece of it. Two provable exclusions
remove most of that work, and the answer must not move at all: these tests
compare against the loop that was replaced, on geometry that actually splits.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tests'), str(ROOT)]
import arc_emplacement_geometry as geometry
import convex_partition
import mesh_geometry
import native_spreading as clipping


def reference_remaining(polygon, contexts):
    """The replaced loop: every candidate face against every current piece."""
    parts = [polygon]
    center, chord_radius = geometry._cap(polygon)
    for context in contexts:
        if np.linalg.norm(center-context['cap_center']) > chord_radius+context['cap_chord_radius']+1e-11:
            continue
        candidates = clipping._candidates(polygon, context)
        if 'active' in context:
            candidates = candidates[context['active'][candidates]]
        if 'exclude' in context:
            candidates = candidates[~np.isin(candidates, context['exclude'])]
        for face in candidates:
            following = []
            for part in parts:
                following.extend(clipping._subtract_polygons(part, [context['triangles'][face]], context))
            parts = following
            if not parts: return []
    return parts


def material(level=3, keep=.6, seed=3):
    """A holed sheet of real spherical triangles, so water actually survives."""
    mesh = mesh_geometry.icosphere(level)
    faces = mesh['faces']
    chosen = np.random.default_rng(seed).permutation(len(faces))[:int(keep*len(faces))]
    context = geometry._context(mesh['vertices'], faces[np.sort(chosen)], 6371.)
    context['active'] = np.ones(len(context['faces']), bool)
    return context


def probes(count=24, span=.28, seed=7):
    """Triangles wide enough to cross several faces and be cut into pieces."""
    random = np.random.default_rng(seed)
    result = []
    for _ in range(count):
        centre = clipping._unit(random.normal(size=3))
        east = clipping._unit(np.cross(centre, [0., 0., 1.] if abs(centre[2]) < .9 else [0., 1., 0.]))
        north = np.cross(centre, east)
        angles = np.sort(random.uniform(0., 2*np.pi, 3))
        offsets = span*(np.cos(angles)[:, None]*east+np.sin(angles)[:, None]*north)
        triangle = clipping._unit(centre+offsets)
        if np.dot(triangle[0], np.cross(triangle[1], triangle[2])) <= 0.:
            triangle = triangle[::-1]
        result.append(triangle)
    return result


def counted(function):
    """Run `function` while counting exact convex partitions."""
    calls = []
    real = convex_partition.partition

    def spy(polygon, edges):
        calls.append(1)
        return real(polygon, edges)
    convex_partition.partition = spy
    try:
        return function(), len(calls)
    finally:
        convex_partition.partition = real


class EmplacementPruningTests(unittest.TestCase):
    def test_pruned_subtraction_returns_the_very_same_pieces(self):
        context = material()
        cut = 0
        for polygon in probes():
            fresh = geometry._context(context['vertices'], context['faces'], 6371.)
            kept = geometry._remaining(polygon, [fresh])
            expected = reference_remaining(polygon, [context])
            self.assertEqual(len(kept), len(expected))
            for a, b in zip(kept, expected):
                # Bit-identical, not merely close: this is a pruning change.
                np.testing.assert_array_equal(a, b)
            cut += len(expected)
        # The fixture has to exercise real splitting, or it proves nothing.
        self.assertGreater(cut, 0)

    def test_a_partly_open_sheet_still_leaves_the_same_water(self):
        context = material()
        context['active'][::3] = False
        total = 0.
        for polygon in probes(count=12):
            kept = geometry._remaining(polygon, [context])
            expected = reference_remaining(polygon, [context])
            self.assertEqual([len(p) for p in kept], [len(p) for p in expected])
            for a, b in zip(kept, expected):
                np.testing.assert_array_equal(a, b)
            total += sum(clipping._area(p, context) for p in kept)
        self.assertGreater(total, 0.)

    def test_most_face_and_piece_pairs_are_refused_without_a_partition(self):
        context, reference = material(), material()
        _, pruned = counted(lambda: [geometry._remaining(p, [context]) for p in probes()])
        _, whole = counted(lambda: [reference_remaining(p, [reference]) for p in probes()])
        self.assertLess(pruned, whole*.25, f'{pruned} partitions against {whole}')

    def test_a_refused_face_really_is_clear_of_the_piece(self):
        # The exclusions are proofs, so an independent exact partition of every
        # refused pair must find no area at all in it.
        context = material()
        centre, chord = geometry._face_caps(context)
        for polygon in probes(count=8):
            base, radius = geometry._cap(polygon)
            near = clipping._candidates(polygon, context)
            offset = centre[near]-base
            far = near[np.einsum('ij,ij->i', offset, offset) > (radius+chord[near]+1e-11)**2]
            for face in far[:40]:
                intersection, _ = convex_partition.partition(
                    polygon, convex_partition.prepare(context['triangles'][face]))
                self.assertLessEqual(clipping._area(intersection, context), clipping.AREA_TOLERANCE_KM2)
            for face in near[:40]:
                planes = context['intersection_planes'][face].T
                if not (polygon@planes < 0.).all(axis=0).any(): continue
                intersection, _ = convex_partition.partition(
                    polygon, convex_partition.prepare(context['triangles'][face]))
                self.assertLessEqual(clipping._area(intersection, context), clipping.AREA_TOLERANCE_KM2)

    def test_face_caps_cover_their_own_triangles(self):
        context = material()
        centre, chord = geometry._face_caps(context)
        for face, triangle in enumerate(context['triangles'][::37]):
            at = face*37
            expected_centre, expected_chord = geometry._cap(triangle)
            np.testing.assert_allclose(centre[at], expected_centre, rtol=0., atol=1e-12)
            self.assertAlmostEqual(chord[at], expected_chord, places=12)
            reach = np.linalg.norm(triangle-centre[at], axis=1).max()
            self.assertLessEqual(reach, chord[at]+1e-12)


if __name__ == '__main__':
    unittest.main()
