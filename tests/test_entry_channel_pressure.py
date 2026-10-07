"""Layered source-load checks for underthrust continental crust."""

import unittest

import numpy as np

from experiments.entry_channel_pressure import (
    layered_top_pressure, ordered_top_pressure)


class EntryChannelPressureTests(unittest.TestCase):
    def test_mantle_only_reduces_to_existing_entry_pressure(self):
        depth = np.array([0., 35000., 80000.])
        pressure = layered_top_pressure(depth, 0., 0.)
        np.testing.assert_allclose(
            pressure['pressure_pa'], 9.81 * 3300. * depth,
            rtol=2e-15)
        np.testing.assert_array_equal(
            pressure['mantle_gap_m'], depth)

    def test_upper_column_replaces_mantle_instead_of_adding_twice(self):
        rock_mass = 2800. * 35000.
        result = layered_top_pressure(80000., 35000., rock_mass)
        self.assertEqual(float(result['mantle_gap_m']), 45000.)
        expected = 9.81 * (rock_mass + 3300. * 45000.)
        self.assertAlmostEqual(float(result['pressure_pa']), expected)
        mantle_only = 9.81 * 3300. * 80000.
        naive_stack_plus_entry = 9.81 * (rock_mass + 3300. * 80000.)
        self.assertAlmostEqual(mantle_only - expected,
                               9.81 * (3300. - 2800.) * 35000.)
        self.assertAlmostEqual(naive_stack_plus_entry - expected,
                               9.81 * 3300. * 35000.)

    def test_contact_and_water_load_have_distinct_sources(self):
        rock_mass = 2800. * 35000.
        contact = layered_top_pressure(25000., 25000., rock_mass,
                                       water_depth_m=1000.)
        self.assertEqual(float(contact['mantle_gap_mass_kg_m2']), 0.)
        self.assertEqual(float(contact['water_mass_kg_m2']), 1030. * 1000.)
        self.assertAlmostEqual(
            float(contact['pressure_pa']),
            9.81 * (rock_mass + 1030. * 1000.))
        with self.assertRaisesRegex(ValueError, 'nonpenetrating'):
            layered_top_pressure(24999., 25000., rock_mass)

    def test_ordered_column_matches_single_layer_and_counts_each_sheet_once(self):
        single = ordered_top_pressure(80000., [
            dict(kind='rock', top_depth_m=0., base_depth_m=35000.,
                 sheet_id=2, mass_kg_m2=2800. * 35000.),
            dict(kind='mantle', top_depth_m=35000., base_depth_m=80000.)],
            (2,))
        self.assertEqual(single['pressure_pa'],
                         layered_top_pressure(80000., 35000.,
                                              2800. * 35000.)['pressure_pa'])
        ordered = ordered_top_pressure(80000., [
            dict(kind='water', top_depth_m=0., base_depth_m=1000.),
            dict(kind='rock', top_depth_m=1000., base_depth_m=21000.,
                 sheet_id=3, mass_kg_m2=2800. * 20000.),
            dict(kind='rock', top_depth_m=21000., base_depth_m=46000.,
                 sheet_id=2, mass_kg_m2=2900. * 25000.),
            dict(kind='mantle', top_depth_m=46000., base_depth_m=80000.)],
            (3, 2))
        self.assertEqual(ordered['rock_sheet_ids'], (3, 2))
        self.assertEqual(ordered['pressure_pa'], 9.81 * (
            1030. * 1000. + 2800. * 20000. + 2900. * 25000.
            + 3300. * 34000.))

    def test_ordered_column_rejects_missing_overlapping_or_reordered_layers(self):
        rock = dict(kind='rock', top_depth_m=0., base_depth_m=20000.,
                    sheet_id=3, mass_kg_m2=2800. * 20000.)
        mantle = dict(kind='mantle', top_depth_m=20000., base_depth_m=80000.)
        with self.assertRaisesRegex(ValueError, 'tile'):
            ordered_top_pressure(80000., [rock, dict(mantle, top_depth_m=21000.)],
                                 (3,))
        with self.assertRaisesRegex(ValueError, 'tile'):
            ordered_top_pressure(80000., [rock, dict(mantle, top_depth_m=19000.)],
                                 (3,))
        with self.assertRaisesRegex(ValueError, 'geometric vertical order'):
            ordered_top_pressure(80000., [rock, mantle], (2, 3))
        with self.assertRaisesRegex(ValueError, 'geometric vertical order'):
            ordered_top_pressure(80000., [rock, mantle], ())


if __name__ == '__main__':
    unittest.main()
