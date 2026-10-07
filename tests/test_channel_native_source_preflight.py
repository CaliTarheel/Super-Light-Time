"""A complete native contact can stage its ordered sources without mutation."""

from copy import deepcopy
import unittest

import numpy as np

import channel_entry_zones
from channel_native_source_preflight import prepare_ordered_entry_source
from entry_channel_pressure import ordered_top_pressure
from tests.test_channel_entry_zones import native_contact_world


def explicit_paths(stage, lower_id):
    """Supply geometry independently of the source compositor under test."""
    paths = {}
    for zone in stage['upper_rock']['by_lower_face'][lower_id]:
        depth = 0.
        segments = []
        for layer in zone['rock_layers_top_to_bottom']:
            base = depth + 1000. * layer['physical_thickness_km']
            segments.append(dict(kind='rock', sheet_id=layer['sheet_id'],
                                 top_depth_m=depth, base_depth_m=base))
            depth = base
        lower_depth = max(80000., depth + 10000.)
        segments.append(dict(kind='mantle', top_depth_m=depth,
                             base_depth_m=lower_depth))
        paths[zone['zone_id']] = dict(lower_top_depth_m=lower_depth,
                                      segments=segments,
                                      effective_thermal_depth_km=80.)
    return {lower_id: paths}


def other_face_loads(stage, upper_id):
    record = next(row for row in stage['store']['records']
                  if row['face_id'] == upper_id)
    return {upper_id: [dict(region_id=row['region_id'], lower_top_depth_m=0.,
                            segments=[], covering_sheets_top_to_bottom=(),
                            effective_thermal_depth_km=0.)
                       for row in record['history']['regions']]}


class NativeOrderedSourcePreflightTests(unittest.TestCase):
    def setup_contact(self):
        s, lower_id, upper_id = native_contact_world()
        stage = channel_entry_zones.prepare_native_entry_zone_remap(s)
        return (s, lower_id, upper_id, explicit_paths(stage, lower_id),
                other_face_loads(stage, upper_id))

    def test_real_contact_stages_complete_ordered_source_without_commit(self):
        s, lower_id, upper_id, paths, other = self.setup_contact()
        before_store = deepcopy(s.channel_region_store)
        before_structure = {key: value.copy() for key, value in s.structure.items()}
        result = prepare_ordered_entry_source(s, paths, other, 1.)
        self.assertEqual(result['source_stage']['store']['epoch_myr'], s.t + 1.)
        self.assertEqual(set(result['pressure_by_face_zone'][lower_id]),
                         set(paths[lower_id]))
        self.assertEqual({row['face_id'] for row in result['source_stage']['store']['records']},
                         {lower_id, upper_id})
        contact_pressures = {layer['region_id']: result['pressure_by_face_zone'][
            lower_id][zone['zone_id']]['pressure_pa']
            for zone in result['zone_stage']['upper_rock']['by_lower_face'][lower_id]
            for layer in zone['rock_layers_top_to_bottom']
            if layer['region_id'] in {'cool', 'warm'}}
        self.assertEqual(set(contact_pressures), {'cool', 'warm'})
        self.assertNotAlmostEqual(contact_pressures['cool'], contact_pressures['warm'])
        for zone_id, pressure in result['pressure_by_face_zone'][lower_id].items():
            self.assertGreaterEqual(pressure['pressure_pa'], 0.)
            zone = next(row for row in
                        result['zone_stage']['upper_rock']['by_lower_face'][lower_id]
                        if row['zone_id'] == zone_id)
            rocks = [segment for segment in pressure['validated_segments']
                     if segment['kind'] == 'rock']
            self.assertEqual([segment['mass_kg_m2'] for segment in rocks],
                             [layer['rock_mass_kg_m2'] for layer in
                              zone['rock_layers_top_to_bottom']])
            self.assertAlmostEqual(pressure['pressure_pa'],
                ordered_top_pressure(paths[lower_id][zone_id]['lower_top_depth_m'],
                                     pressure['validated_segments'],
                                     pressure['rock_sheet_ids'])['pressure_pa'])
        self.assertEqual(len(s.channel_region_store['records'][0]['history']['regions']), 1)
        lower_next = next(row for row in result['source_stage']['store']['records']
                          if row['face_id'] == lower_id)
        self.assertGreater(len(lower_next['history']['regions']), 1)
        self.assertEqual(s.channel_region_store['epoch_myr'], before_store['epoch_myr'])
        for name in s.structure:
            np.testing.assert_array_equal(s.structure[name], before_structure[name])

    def test_incomplete_or_invented_contact_loads_fail_closed(self):
        s, lower_id, upper_id, paths, other = self.setup_contact()
        zone_id = next(zone_id for zone_id, path in paths[lower_id].items()
                       if any(segment['kind'] == 'rock' for segment in path['segments']))
        missing = deepcopy(paths)
        del missing[lower_id][zone_id]
        with self.assertRaisesRegex(ValueError, 'one path per exact contact zone'):
            prepare_ordered_entry_source(s, missing, other, 1.)
        with self.assertRaisesRegex(ValueError, 'other tracked face'):
            prepare_ordered_entry_source(s, paths, {}, 1.)
        invented = deepcopy(paths)
        segment = next(segment for segment in invented[lower_id][zone_id]['segments']
                       if segment['kind'] == 'rock')
        segment['mass_kg_m2'] = 1.
        with self.assertRaisesRegex(ValueError, 'sheet order'):
            prepare_ordered_entry_source(s, invented, other, 1.)
        wrong_thickness = deepcopy(paths)
        segment = next(segment for segment in
                       wrong_thickness[lower_id][zone_id]['segments']
                       if segment['kind'] == 'rock')
        segment['base_depth_m'] -= 1000.
        with self.assertRaisesRegex(ValueError, 'measured physical thickness'):
            prepare_ordered_entry_source(s, wrong_thickness, other, 1.)
        s.collision_contacts[0]['last_seen_myr'] -= 1.
        with self.assertRaisesRegex(ValueError, 'not current|stale'):
            prepare_ordered_entry_source(s, paths, other, 1.)
        self.assertEqual(len(s.channel_region_store['records'][0]['history']['regions']), 1)


if __name__ == '__main__':
    unittest.main()
