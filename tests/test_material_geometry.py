"""Fractures preserve the area patches and buried cratons they operate on."""
import unittest
import json
from pathlib import Path

import numpy as np

from material_geometry import coherent_partition, patch_centres, splits_protected_groups
from fracture import Fracture
from raster_engine import _rotate, _xyz


class MaterialGeometryTests(unittest.TestCase):
    def test_fracture_does_not_send_one_cell_quadrant_on_a_separate_motion(self):
        # The bent crack clips one quadrature point from a cell. The old
        # per-parcel decision detached 1/4 cell; a material patch is indivisible.
        lon = np.radians(np.array([-.75, -.25, -.75, -.25]))
        lat = np.radians(np.array([.75, .75, .25, .25]))
        position = _xyz(lon, lat)
        mass = np.ones(4)
        crack = lambda p: p[:, 1] + p[:, 2] - np.radians(.3)
        old_side = crack(position) > 0
        self.assertEqual(int(old_side.sum()), 1)
        new_side = coherent_partition(position, mass, np.full(4, 41), crack)
        self.assertTrue(np.all(new_side == new_side[0]))
        original_lengths = np.linalg.norm(position[:, None] - position[None, :], axis=2)
        # Carry the result over the pole with two separating plate rotations.
        directions = np.array([[0., -np.pi/2, 0.], [.35, -np.pi/2, 0.]])
        old_position = _rotate(position, directions[old_side.astype(int)])
        new_position = _rotate(position, directions[new_side.astype(int)])
        old_lengths = np.linalg.norm(old_position[:, None] - old_position[None, :], axis=2)
        new_lengths = np.linalg.norm(new_position[:, None] - new_position[None, :], axis=2)
        self.assertGreater(float(np.max(np.abs(old_lengths-original_lengths))), .1)
        np.testing.assert_allclose(new_lengths, original_lengths, atol=2e-15)

    def test_centres_cross_the_longitude_seam_and_both_poles(self):
        for pole in (-1, 1):
            with self.subTest(pole=pole):
                longitude = np.radians([179., -179., 179., -179.])
                latitude = pole*np.radians([82., 82., 84., 84.])
                positions = _xyz(longitude, latitude)
                ids, centre, mapping = patch_centres(positions, np.ones(4), np.full(4, 91))
                self.assertEqual(ids.tolist(), [91])
                np.testing.assert_array_equal(mapping, np.zeros(4, int))
                self.assertLess(centre[0, 0], 0)
                self.assertAlmostEqual(float(centre[0, 1]), 0, places=14)
                self.assertGreater(float(centre[0, 2])*pole, .99)
                rotation = np.array([0., pole*np.radians(10), 0.])
                _, after, _ = patch_centres(_rotate(positions, rotation), np.ones(4), np.full(4, 91))
                np.testing.assert_allclose(after, _rotate(centre, rotation), atol=2e-15)

    def test_centres_use_exact_area_and_sparse_identity(self):
        positions = _xyz(np.radians([5., 15., -5., -15.]), np.zeros(4))
        mass = np.array([1., 3., 1., 3.])
        ids, centre, inverse = patch_centres(positions, mass, np.array([8, 8, 9002, 9002]))
        self.assertEqual(ids.tolist(), [8, 9002])
        self.assertEqual(inverse.tolist(), [0, 0, 1, 1])
        expected = positions[:2].T @ mass[:2]
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(centre[0], expected)
        self.assertGreater(float(np.degrees(np.arctan2(centre[0, 1], centre[0, 0]))), 12)

    def test_hidden_craton_material_rejects_a_cut_even_with_no_visible_pixels(self):
        # The real 742 Myr failure detached 8 of 276 craton samples. Crust
        # overlapped at its edge, so the displayed craton mask missed the cut.
        groups = np.r_[np.zeros(276, int), np.full(100, -1)]
        side = np.zeros(376, bool)
        side[:8] = True
        side[300:] = True
        self.assertTrue(splits_protected_groups(groups, side))
        side[:276] = True
        self.assertFalse(splits_protected_groups(groups, side))

    def test_separate_cratons_may_move_apart_without_being_broken(self):
        groups = np.array([4, 4, 4, 19, 19, -1, -1])
        side = np.array([False, False, False, True, True, False, True])
        self.assertFalse(splits_protected_groups(groups, side))
        side[1] = True
        self.assertTrue(splits_protected_groups(groups, side))

    def test_real_742_myr_fracture_is_rejected_by_enduring_craton_identity(self):
        # Captured by replaying the archived current run at its actual failing
        # step. That accepted cut later separates eight original craton samples
        # by thousands of kilometres. Whole-cell decisions alone still divide
        # three of its 69 patches; enduring craton identity must reject it too.
        fixture = Path(__file__).parent/'fixtures/craton_cut_742_myr.json'
        source = json.loads(fixture.read_text())
        crack = Fracture(**{name:np.asarray(source[name]) for name in
                            ('center', 'normal', 'tangent', 'knots', 'offsets')})
        centers = np.asarray(source['patch_centres'])
        sides = coherent_partition(centers, np.ones(len(centers)),
                                   np.asarray(source['patch_ids']), crack.signed_distance)
        self.assertEqual(len(sides), 69)
        self.assertEqual(int(sides.sum()), 66)
        self.assertTrue(splits_protected_groups(np.zeros(len(sides), int), sides))

    def test_empty_material_is_a_valid_partition(self):
        sides = coherent_partition(np.empty((0,3)), np.empty(0), np.empty(0,int), lambda p:p[:,0])
        self.assertEqual(sides.shape, (0,))
        self.assertFalse(splits_protected_groups(np.empty(0,int), sides))


if __name__ == '__main__':
    unittest.main()
