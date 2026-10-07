"""Current native upper-column elevation determines the single-sheet path."""

import unittest

import numpy as np

import channel_entry_zones
from channel_native_pressure import pressure_from_measured_rock
from channel_native_source_preflight import prepare_ordered_entry_source
from channel_native_vertical_path import current_single_upper_path
from tests.test_channel_entry_zones import native_contact_world
from tests.test_channel_native_source_preflight import other_face_loads


def zone(elevation_m):
    return dict(zone_id='contact', rock_layers_top_to_bottom=[dict(
        sheet_id=2, current_surface_elevation_m=elevation_m,
        physical_thickness_km=35., rock_mass_kg_m2=2800. * 35000.)])


class NativeVerticalPathTests(unittest.TestCase):
    def test_exposed_and_submerged_native_upper_columns_tile_the_path(self):
        exposed = current_single_upper_path(zone(1000.), 80000.)
        self.assertEqual(exposed['segments'], [
            dict(kind='rock', sheet_id=2, top_depth_m=-1000., base_depth_m=34000.),
            dict(kind='mantle', top_depth_m=34000., base_depth_m=80000.)])
        submerged = current_single_upper_path(zone(-1000.), 80000.)
        self.assertEqual(submerged['segments'], [
            dict(kind='water', top_depth_m=0., base_depth_m=1000.),
            dict(kind='rock', sheet_id=2, top_depth_m=1000., base_depth_m=36000.),
            dict(kind='mantle', top_depth_m=36000., base_depth_m=80000.)])
        for measured, path in ((zone(1000.), exposed), (zone(-1000.), submerged)):
            result = pressure_from_measured_rock(
                measured, path['lower_top_depth_m'], path['segments'])
            self.assertGreater(result['pressure_pa'], 0.)

    def test_free_path_and_unresolved_geometry_fail_closed(self):
        free = dict(zone_id='free', rock_layers_top_to_bottom=[])
        self.assertEqual(current_single_upper_path(free, 80000.)['segments'],
                         [dict(kind='mantle', top_depth_m=0., base_depth_m=80000.)])
        self.assertEqual(current_single_upper_path(free, 0.)['segments'], [])
        with self.assertRaisesRegex(ValueError, 'contact solve'):
            current_single_upper_path(zone(1000.), 30000.)
        with self.assertRaisesRegex(ValueError, 'one upper sheet'):
            current_single_upper_path(dict(zone(1000.),
                rock_layers_top_to_bottom=zone(1000.)['rock_layers_top_to_bottom'] * 2),
                80000.)
        with self.assertRaisesRegex(ValueError, 'finite lower depth'):
            current_single_upper_path(free, np.nan)

    def test_native_regional_elevation_feeds_complete_source_preflight(self):
        s, lower_id, upper_id = native_contact_world()
        stage = channel_entry_zones.prepare_native_entry_zone_remap(s)
        paths = {}
        for zone_rock in stage['upper_rock']['by_lower_face'][lower_id]:
            path = current_single_upper_path(zone_rock, 80000.)
            paths[zone_rock['zone_id']] = dict(
                lower_top_depth_m=path['lower_top_depth_m'],
                segments=path['segments'], effective_thermal_depth_km=80.)
        result = prepare_ordered_entry_source(
            s, {lower_id: paths}, other_face_loads(stage, upper_id), 1.)
        self.assertEqual(result['source_stage']['store']['epoch_myr'], s.t + 1.)
        self.assertEqual(len(s.channel_region_store['records'][0]['history']['regions']), 1)
        self.assertTrue(any(zone['rock_layers_top_to_bottom'] for zone in
                            stage['upper_rock']['by_lower_face'][lower_id]))


if __name__ == '__main__':
    unittest.main()
