"""A roundoff-sized overlap without a recorded layer order must not stop a run.

Captured from run SEP21T (20260921-011257-b2f85b) at 85 Myr: juvenile arcs 38
and 93 on block 02 (sheets 49 and 104) touched with 1.83e-8 km2 of measured
overlap after arc emplacement. Arc admission allows that much by its own
clipping; mesh_coverage counts anything above 1e-8 km2. No ContactAdmission
ever ran for the pair, so it had no order and reference_energy raised.
"""
import unittest
from pathlib import Path

import numpy as np

import column_density
import gravitational_relaxation
import mesh_coverage

FIXTURE = Path(__file__).resolve().parent/'fixtures'/'unordered_arc_overlap_sep21t_85myr.npz'


def captured():
    with np.load(FIXTURE) as data:
        return {name: data[name] for name in data.files}


class UnorderedOverlapFloorTests(unittest.TestCase):
    def setUp(self):
        self.data = captured()
        self.profile = dict(dense_fraction=self.data['dense_fraction'], sheet_order={})

    def test_captured_pair_is_a_roundoff_overlap(self):
        overlap = mesh_coverage.material_overlaps(self.data['vertices'], self.data['faces'], self.data['sheets'])
        self.assertEqual(len(overlap['first']), 1)
        self.assertGreater(overlap['area_km2'][0], 1e-8)
        self.assertLess(overlap['area_km2'][0], 1e-6)

    def test_reference_energy_and_gradient_accept_the_captured_pair(self):
        d = self.data
        volumes = np.array([10., 12.])*gravitational_relaxation.spherical_face_areas(d['vertices'], d['faces'])
        reference = np.array([9., 11.])
        energy = gravitational_relaxation.reference_energy(d['vertices'], d['faces'], volumes, reference,
                                                           d['sheets'], density_profile=self.profile)
        self.assertTrue(np.isfinite(energy))
        area = gravitational_relaxation.spherical_face_areas(d['vertices'], d['faces'])
        value, gradient, _ = gravitational_relaxation.energy_gradient(
            d['vertices'], d['faces'], volumes/area, reference, d['sheets'], density_profile=self.profile)
        self.assertTrue(np.isfinite(value) and np.isfinite(gradient).all())
        self.assertAlmostEqual(value, energy, delta=1e-9*max(1., abs(energy)))

    def test_sub_floor_unordered_pair_takes_the_mean_of_both_orders(self):
        fraction = np.array([.1, .4])
        _, pair = column_density.weights(fraction, np.array([1, 2]), [0], [1], {}, area_km2=[2e-8])
        _, over = column_density.weights(fraction, np.array([1, 2]), [0], [1], {1: {2}})
        _, under = column_density.weights(fraction, np.array([1, 2]), [0], [1], {2: {1}})
        self.assertAlmostEqual(pair[0], .5*(over[0]+under[0]), places=15)

    def test_real_unordered_overlap_still_raises(self):
        fraction = np.array([.1, .4])
        for area in (None, [column_density.UNORDERED_OVERLAP_FLOOR_KM2*1.001], [50.]):
            with self.assertRaisesRegex(ValueError, 'unique persistent layer order'):
                column_density.weights(fraction, np.array([1, 2]), [0], [1], {}, area_km2=area)

    def test_contradictory_order_still_raises_below_the_floor(self):
        with self.assertRaisesRegex(ValueError, 'unique persistent layer order'):
            column_density.weights(np.array([.1, .4]), np.array([1, 2]), [0], [1], {1: {2}, 2: {1}}, area_km2=[1e-8])

    def test_ordered_pairs_are_bitwise_unchanged(self):
        fraction = np.array([.1, .4, .7])
        args = (fraction, np.array([1, 2, 3]), [0, 1], [1, 2], {1: {2, 3}, 2: {3}})
        self.assertTrue(np.array_equal(column_density.weights(*args)[1],
                                       column_density.weights(*args, area_km2=[3., 1e-9])[1]))


if __name__ == '__main__':
    unittest.main()
