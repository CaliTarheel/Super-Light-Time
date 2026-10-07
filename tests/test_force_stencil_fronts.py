"""Complete force terms inherit welds only through finite clipped arcs."""
from copy import deepcopy
import unittest

import numpy as np

import collision_fronts
import mesh_coverage
from tests.test_collision_architecture import overlapping_fixture, orientation
from tests.test_collision_local_fronts import fixture, unit


def groups(s):
    surface, sheets = s.material_surface, s.parcel_collision_sheet
    full = mesh_coverage.material_overlaps(surface['vertices'], surface['faces'], sheets,
                                         include_touching=True)
    rows = s.collision_contacts[0]['local_fronts']
    selected = collision_fronts.force_stencil_groups(surface, sheets, s._collision_overlap,
        rows, full, np.arange(len(full['first'])))
    return full, selected


class ForceStencilFrontTests(unittest.TestCase):
    def test_rotated_overlap_adds_missing_internal_work_without_mutating_fronts(self):
        s = overlapping_fixture(rotation=orientation())
        before = deepcopy(s.collision_contacts)
        full, selected = groups(s)
        self.assertEqual(len(selected), 1)
        pairs = set(zip(full['first'][selected[0]], full['second'][selected[0]]))
        self.assertIn((6, 14), pairs)
        original = set(zip(s._collision_overlap['first'], s._collision_overlap['second']))
        self.assertTrue(original < pairs)
        self.assertEqual(s.collision_contacts, before)

    def test_disconnected_patches_stay_separate_even_with_shared_coarse_face(self):
        for coarse, refine in ((False, False), (True, False), (True, True)):
            s = fixture(coarse=coarse, refine_top=refine)
            full, selected = groups(s)
            self.assertEqual(len(selected), 2)
            self.assertFalse(set(selected[0]) & set(selected[1]))
            # Every force patch stays on one side of the actual 1200 km gap.
            signs = []
            for take in selected:
                centers = s.pos[full['first'][take]]+s.pos[full['second'][take]]
                self.assertTrue(np.all(centers[:, 1] < 0.) or np.all(centers[:, 1] > 0.))
                signs.append(np.sign(centers[0, 1]))
            self.assertEqual(sorted(signs), [-1., 1.])

    def test_zero_area_shared_arc_joins_but_point_touch_does_not(self):
        # Three pair polygons, with permissive candidate adjacency. Only a
        # finite common arc can join the first two; the third shares one point.
        first = unit([[1., 0., 0.], [1., .1, 0.], [1., 0., .1]])
        line = first[:2]
        point_neighbor = unit([[1., .1, 0.], [1., .2, 0.], [1., .1, -.1]])
        geometry = dict(neighbors=np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [3, -1, -1]]),
            polygon_offsets=np.array([0, 3, 5, 8]),
            polygon_vertices=np.concatenate((first, line, point_neighbor)))
        full = dict(first=np.array([0, 1, 2]), second=np.array([3, 3, 3]),
                    area_km2=np.array([1., 0., 1.]))
        old = dict(first=np.array([0, 2]), second=np.array([3, 3]), area_km2=np.ones(2))
        rows = [dict(overlap_indices=[0]), dict(overlap_indices=[1])]
        actual = collision_fronts.force_stencil_groups(dict(radius_km=6371.), np.array([1, 1, 1, 2]),
            old, rows, full, np.arange(3), geometry=geometry)
        np.testing.assert_array_equal(actual[0], [0, 1])
        np.testing.assert_array_equal(actual[1], [2])

    def test_real_thin_bridge_between_two_seeded_fronts_is_rejected(self):
        # The central material triangle shares an edge with each large one,
        # which otherwise meet only at one indexed vertex. Clipping its lower
        # hundredth leaves a real positive overlap below the ledger floor.
        xy = np.array([[0., 0.], [100., 0.], [100., 100.],
                       [100.+1e-6, 100.], [200., 0.],
                       [150., -400.], [400., 1.], [-100., 1.]])
        points = unit(np.column_stack((np.ones(8), xy/6371.)))
        faces = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3], [5, 6, 7]])
        sheets = np.array([1, 1, 1, 2])
        surface = dict(vertices=points, faces=faces, radius_km=6371.)
        full = mesh_coverage.material_overlaps(points, faces, sheets, include_touching=True)
        old = mesh_coverage.material_overlaps(points, faces, sheets)
        self.assertEqual(set(zip(old['first'], old['second'])), {(0, 3), (2, 3)})
        middle = np.flatnonzero(full['first'] == 1)
        self.assertEqual(len(middle), 1)
        self.assertGreater(full['area_km2'][middle[0]], 0.)
        self.assertLess(full['area_km2'][middle[0]], 1e-8)
        rows = [dict(overlap_indices=[0]), dict(overlap_indices=[1])]
        with self.assertRaisesRegex(ValueError, 'ambiguously joins existing weld fronts'):
            collision_fronts.force_stencil_groups(surface, sheets, old, rows, full,
                                                  np.arange(len(full['first'])))

    def test_interleaved_face_indices_preserve_physical_membership(self):
        s = overlapping_fixture(rotation=orientation())
        full, baseline = groups(s)
        expected = set(zip(full['first'][baseline[0]], full['second'][baseline[0]]))
        surface = dict(s.material_surface)
        n = len(surface['faces'])//2
        permutation = np.column_stack((np.arange(n), np.arange(n, 2*n))).ravel()
        surface['faces'] = surface['faces'][permutation]
        sheets = s.parcel_collision_sheet[permutation]
        old = mesh_coverage.material_overlaps(surface['vertices'], surface['faces'], sheets)
        complete = mesh_coverage.material_overlaps(surface['vertices'], surface['faces'], sheets,
                                                 include_touching=True)
        selected = collision_fronts.force_stencil_groups(surface, sheets, old,
            [dict(overlap_indices=list(range(len(old['first']))))], complete,
            np.arange(len(complete['first'])))
        actual = {tuple(sorted((int(permutation[a]), int(permutation[b]))))
                  for a, b in zip(complete['first'][selected[0]], complete['second'][selected[0]])}
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
