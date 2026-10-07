"""Explicit vertical paths with rock mass taken only from native columns."""

import unittest

import numpy as np

from channel_native_pressure import pressure_from_measured_rock
from entry_channel_pressure import ordered_top_pressure


def zone():
    return dict(zone_id='contact', rock_layers_top_to_bottom=[dict(
        face_id=200, sheet_id=2, region_id='cool',
        physical_thickness_km=35., rock_mass_kg_m2=2800. * 35000.)])


class NativeOrderedPressureTests(unittest.TestCase):
    def test_exposed_upper_rock_contributes_full_measured_mass(self):
        path = [dict(kind='rock', top_depth_m=-1000., base_depth_m=34000.,
                     sheet_id=2),
                dict(kind='mantle', top_depth_m=34000., base_depth_m=80000.)]
        result = pressure_from_measured_rock(zone(), 80000., path)
        self.assertEqual(result['zone_id'], 'contact')
        self.assertEqual(result['rock_sheet_ids'], (2,))
        np.testing.assert_allclose(result['pressure_pa'],
                                   9.81 * (2800. * 35000. + 3300. * 46000.))
        self.assertEqual(path[0].keys(),
                         {'kind', 'top_depth_m', 'base_depth_m', 'sheet_id'})

    def test_submerged_upper_rock_keeps_water_and_mantle_separate(self):
        path = [dict(kind='water', top_depth_m=0., base_depth_m=1000.),
                dict(kind='rock', top_depth_m=1000., base_depth_m=36000.,
                     sheet_id=2),
                dict(kind='mantle', top_depth_m=36000., base_depth_m=80000.)]
        result = pressure_from_measured_rock(zone(), 80000., path)
        np.testing.assert_allclose(result['overburden_mass_kg_m2'],
                                   1030. * 1000. + 2800. * 35000.
                                   + 3300. * 44000.)

    def test_two_rock_layers_follow_sheet_order_and_physical_thickness(self):
        upper = dict(zone_id='stack', rock_layers_top_to_bottom=[
            dict(sheet_id=3, physical_thickness_km=20.,
                 rock_mass_kg_m2=3450. * 20000.),
            dict(sheet_id=2, physical_thickness_km=30.,
                 rock_mass_kg_m2=2800. * 30000.)])
        path = [dict(kind='rock', sheet_id=3,
                     top_depth_m=-500., base_depth_m=19500.),
                dict(kind='rock', sheet_id=2,
                     top_depth_m=19500., base_depth_m=49500.),
                dict(kind='mantle', top_depth_m=49500., base_depth_m=80000.)]
        result = pressure_from_measured_rock(upper, 80000., path)
        self.assertEqual(result['rock_sheet_ids'], (3, 2))
        np.testing.assert_allclose(result['overburden_mass_kg_m2'],
            3450. * 20000. + 2800. * 30000. + 3300. * 30500.)

    def test_missing_mass_geometry_and_negative_water_fail_closed(self):
        rock = dict(kind='rock', sheet_id=2,
                    top_depth_m=-1000., base_depth_m=34000.)
        mantle = dict(kind='mantle', top_depth_m=34000., base_depth_m=80000.)
        with self.assertRaisesRegex(ValueError, 'measured physical thickness'):
            pressure_from_measured_rock(zone(), 80000.,
                [dict(rock, base_depth_m=33000.),
                 dict(mantle, top_depth_m=33000.)])
        with self.assertRaisesRegex(ValueError, 'sheet order'):
            pressure_from_measured_rock(zone(), 80000.,
                [dict(rock, sheet_id=3), mantle])
        with self.assertRaisesRegex(ValueError, 'sheet order'):
            pressure_from_measured_rock(zone(), 80000.,
                [dict(rock, mass_kg_m2=1.), mantle])
        with self.assertRaisesRegex(ValueError, 'nonoverlapping vertical'):
            pressure_from_measured_rock(zone(), 80000.,
                [rock, dict(mantle, top_depth_m=35000.)])
        with self.assertRaisesRegex(ValueError, 'nonoverlapping vertical'):
            ordered_top_pressure(1000., [dict(kind='water',
                top_depth_m=-100., base_depth_m=1000.)], ())


if __name__ == '__main__':
    unittest.main()
