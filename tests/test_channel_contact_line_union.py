"""Disjoint contact zones share one nonlocal strip at each hinge position."""

import unittest

import numpy as np

import channel_entry_zones
import mesh_coverage
from channel_contact_line_quadrature import spherical_contact_line_quadrature
from channel_contact_line_union import spherical_contact_line_union
from ridge_geometry import rotate
from tests.test_channel_contact_line_quadrature import local_polygon
from tests.test_channel_native_contact_strip import contact_fixture


def split_zones():
    triangle = local_polygon()
    a, b, c = triangle
    middle = a + b
    middle /= np.linalg.norm(middle)
    return triangle, [dict(zone_id='left', polygon=np.array((a, middle, c))),
                      dict(zone_id='right', polygon=np.array((middle, b, c)))]


class ContactLineUnionTests(unittest.TestCase):
    def test_splitting_zone_does_not_duplicate_full_strip_work(self):
        polygon, zones = split_zones()
        normal = np.array([0., 1., 0.])
        full = spherical_contact_line_quadrature(
            polygon, normal, area_tolerance=1e-5)
        union = spherical_contact_line_union(
            zones, normal, area_tolerance=1e-5)
        separately = [spherical_contact_line_quadrature(
            zone['polygon'], normal, area_tolerance=1e-5) for zone in zones]
        np.testing.assert_allclose(union['polygon_area_m2'],
                                   full['polygon_area_m2'], rtol=2e-12)
        np.testing.assert_allclose(union['line_weight_m'].sum(),
                                   full['line_weight_m'].sum(), rtol=1e-12)
        self.assertGreater(sum(part['line_weight_m'].sum() for part in separately),
                           union['line_weight_m'].sum())
        self.assertTrue(any(len(intervals) == 2
                            for intervals in union['contact_intervals']))
        self.assertLess(abs(union['relative_area_residual']), 1e-5)
        self.assertTrue(all(set(interval['zone_id'] for interval in intervals)
                            <= {'left', 'right'}
                            for intervals in union['contact_intervals']))

    def test_order_rotation_and_current_native_zones_preserve_area(self):
        s, lower_id, upper_id, _, _, _ = contact_fixture()
        spec = channel_entry_zones.entry_regions.specification(s)
        records = {int(record['face_id']): record
                   for record in s.channel_region_store['records']}
        geometry = channel_entry_zones.partition_native_entry_zones(
            s.material_surface, s.parcel_collision_sheet, s.parcel_plate,
            s.plate_uid, s.collision_contacts, spec, list(records),
            epoch_myr=s.t, upper_records_by_face_id=records)
        rows = [zone for zone in geometry['entry_faces'][lower_id]['zones']
                if zone['kind'] == 'entered-contact'
                and zone['adjacent_upper_face_id'] == upper_id]
        self.assertEqual(len(rows), 2)
        zones = [dict(zone_id=zone['zone_id'], polygon=zone['polygon'])
                 for zone in rows]
        row = int(np.flatnonzero(spec['face_ids'] == lower_id)[0])
        normal = spec['hinge_normals'][row]
        original = spherical_contact_line_union(zones, normal)
        reordered = spherical_contact_line_union(zones[::-1], normal)
        np.testing.assert_allclose(original['polygon_area_m2'],
            sum(mesh_coverage._polygon_area(zone['polygon'], 6371.) * 1e6
                for zone in zones), rtol=2e-12)
        np.testing.assert_allclose(reordered['line_weight_m'],
                                   original['line_weight_m'], rtol=1e-12)
        rotation = np.array([.13, -.08, .17])
        rotated = spherical_contact_line_union([
            dict(zone_id=zone['zone_id'],
                 polygon=rotate(zone['polygon'], rotation)) for zone in zones],
            rotate(normal[None], rotation)[0])
        np.testing.assert_allclose(rotated['hinge_xyz'],
                                   rotate(original['hinge_xyz'], rotation), atol=1e-12)
        np.testing.assert_allclose(rotated['line_weight_m'],
                                   original['line_weight_m'], rtol=1e-12)

    def test_duplicate_area_and_ids_fail_closed(self):
        _, zones = split_zones()
        normal = np.array([0., 1., 0.])
        with self.assertRaisesRegex(ValueError, 'distinct zone IDs'):
            spherical_contact_line_union([zones[0], zones[0]], normal)
        duplicate = dict(zone_id='duplicate', polygon=zones[0]['polygon'])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            spherical_contact_line_union([zones[0], duplicate], normal)
        with self.assertRaisesRegex(ValueError, 'panel budget'):
            spherical_contact_line_union(zones, normal,
                                         area_tolerance=1e-12, max_panels=4)


if __name__ == '__main__':
    unittest.main()
