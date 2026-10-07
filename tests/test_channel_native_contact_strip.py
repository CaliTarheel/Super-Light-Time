"""Current contact identity for a read-only trench-normal upper strip."""

import unittest

import numpy as np

import channel_entry_zones
from channel_native_contact_strip import bind_contact_zone_upper_strip
from tests.test_channel_entry_zones import native_contact_world


def contact_fixture():
    s, lower_id, upper_id = native_contact_world()
    # The synthetic active-contact fixture predates pre-contact snapshots.
    # Attach a fixed test datum; its initialization rules are tested elsewhere.
    upper = next(record for record in s.channel_region_store['records']
                 if int(record['face_id']) == upper_id)
    for region in upper['history']['regions']:
        region['upper_basal_reference'] = dict(depth_m=35000., epoch_myr=s.t)
    spec = channel_entry_zones.entry_regions.specification(s)
    records = {int(record['face_id']): record
               for record in s.channel_region_store['records']}
    geometry = channel_entry_zones.partition_native_entry_zones(
        s.material_surface, s.parcel_collision_sheet, s.parcel_plate,
        s.plate_uid, s.collision_contacts, spec, list(records),
        epoch_myr=s.t, upper_records_by_face_id=records)
    zone = next(zone for zone in geometry['entry_faces'][lower_id]['zones']
                if zone['kind'] == 'entered-contact'
                and zone['adjacent_upper_face_id'] == upper_id
                and len(zone['covering_sheets_top_to_bottom']) == 1)
    point = zone['polygon'].sum(axis=0)
    point /= np.linalg.norm(point)
    row = int(np.flatnonzero(spec['face_ids'] == lower_id)[0])
    contact_offset = 6371000. * np.arcsin(point @ spec['hinge_normals'][row])
    offsets = contact_offset + np.linspace(-100., 100., 5)
    return s, lower_id, upper_id, zone['zone_id'], point, offsets


class NativeContactStripTests(unittest.TestCase):
    def test_current_contact_zone_locates_actual_upper_material(self):
        s, lower_id, upper_id, zone_id, point, offsets = contact_fixture()
        before = s.channel_region_store
        report = bind_contact_zone_upper_strip(
            s, lower_id, zone_id, point, offsets)
        self.assertIs(s.channel_region_store, before)
        self.assertEqual(report['lower_face_id'], lower_id)
        self.assertEqual(report['adjacent_upper_face_id'], upper_id)
        np.testing.assert_array_equal(report['face_id'], upper_id)
        np.testing.assert_array_equal(report['unloaded_upper_base_depth_m'], 35000.)
        self.assertAlmostEqual(report['contact_offset_m'], offsets[2], places=5)

    def test_wrong_zone_missing_datum_and_stale_contact_fail_closed(self):
        s, lower_id, _, zone_id, point, offsets = contact_fixture()
        with self.assertRaisesRegex(ValueError, 'entered-contact zone'):
            bind_contact_zone_upper_strip(s, lower_id, 'absent', point, offsets)
        with self.assertRaisesRegex(ValueError, 'outside its current contact zone'):
            bind_contact_zone_upper_strip(s, lower_id, zone_id, -point, offsets)
        upper = s.channel_region_store['records'][1]
        for region in upper['history']['regions']:
            del region['upper_basal_reference']
        with self.assertRaisesRegex(ValueError, 'unloaded base'):
            bind_contact_zone_upper_strip(s, lower_id, zone_id, point, offsets)
        for region in upper['history']['regions']:
            region['upper_basal_reference'] = dict(depth_m=35000., epoch_myr=s.t)
        s.collision_contacts[0]['last_seen_myr'] -= 1.
        with self.assertRaisesRegex(ValueError, 'not current'):
            bind_contact_zone_upper_strip(s, lower_id, zone_id, point, offsets)


if __name__ == '__main__':
    unittest.main()
