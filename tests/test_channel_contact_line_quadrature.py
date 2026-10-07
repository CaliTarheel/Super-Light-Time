"""Dimensionally correct hinge-length measure and spherical coarea audit."""

import unittest

import numpy as np

import channel_entry_zones
import mesh_coverage
from channel_contact_line_quadrature import spherical_contact_line_quadrature
from ridge_geometry import rotate
from tests.test_channel_native_contact_strip import contact_fixture


def unit_rows(points):
    return points / np.linalg.norm(points, axis=1)[:, None]


def local_polygon():
    return unit_rows(np.array([[1., .01, -.03],
                               [1., .07, -.02],
                               [1., .05, .04]]))


class ContactLineQuadratureTests(unittest.TestCase):
    def test_local_zone_coarea_closes_spherical_area_and_line_units(self):
        polygon = local_polygon()
        normal = np.array([0., 1., 0.])
        result = spherical_contact_line_quadrature(
            polygon, normal, area_tolerance=1e-5)
        area = mesh_coverage._polygon_area(polygon, 6371.) * 1e6
        self.assertLess(abs(result['relative_area_residual']), 1e-5)
        np.testing.assert_allclose(result['polygon_area_m2'], area)
        np.testing.assert_allclose(result['represented_area_m2'].sum(), area,
                                   rtol=1e-5)
        self.assertTrue(np.all(result['line_weight_m'] > 0.))
        self.assertTrue(np.all(result['represented_area_m2'] > 0.))
        self.assertTrue(np.all(result['contact_xyz'] @ normal > 0.))
        midpoint = polygon[0] + polygon[1]
        midpoint /= np.linalg.norm(midpoint)
        divided = np.vstack((polygon[0], midpoint, polygon[1:]))
        same_zone = spherical_contact_line_quadrature(
            divided, normal, area_tolerance=1e-5)
        self.assertEqual(same_zone['panels'], result['panels'])
        np.testing.assert_allclose(same_zone['contact_xyz'],
                                   result['contact_xyz'], atol=1e-12)
        np.testing.assert_allclose(same_zone['line_weight_m'],
                                   result['line_weight_m'], rtol=1e-12)
        # Flexure energy is J/m of trench; only hinge-length weights yield J.
        energy_j_per_m = 2.5e10
        total_j = energy_j_per_m * result['line_weight_m'].sum()
        self.assertAlmostEqual(total_j / result['line_weight_m'].sum(),
                               energy_j_per_m)

    def test_current_native_zone_and_rotation_preserve_line_measure(self):
        s, lower_id, _, zone_id, _, _ = contact_fixture()
        spec = channel_entry_zones.entry_regions.specification(s)
        records = {int(record['face_id']): record
                   for record in s.channel_region_store['records']}
        geometry = channel_entry_zones.partition_native_entry_zones(
            s.material_surface, s.parcel_collision_sheet, s.parcel_plate,
            s.plate_uid, s.collision_contacts, spec, list(records),
            epoch_myr=s.t, upper_records_by_face_id=records)
        zone = next(zone for zone in geometry['entry_faces'][lower_id]['zones']
                    if zone['zone_id'] == zone_id)
        row = int(np.flatnonzero(spec['face_ids'] == lower_id)[0])
        normal = spec['hinge_normals'][row]
        original = spherical_contact_line_quadrature(zone['polygon'], normal)
        self.assertLess(abs(original['relative_area_residual']), 1e-4)
        self.assertTrue(np.all(original['contact_xyz'] @
                               mesh_coverage._triangle_planes(zone['polygon']).T
                               >= -1e-10))
        rotation = np.array([.12, -.08, .17])
        turned = spherical_contact_line_quadrature(
            rotate(zone['polygon'], rotation),
            rotate(normal[None], rotation)[0])
        self.assertEqual(turned['panels'], original['panels'])
        np.testing.assert_allclose(turned['contact_xyz'],
                                   rotate(original['contact_xyz'], rotation), atol=1e-12)
        np.testing.assert_allclose(turned['line_weight_m'],
                                   original['line_weight_m'], rtol=1e-12)

    def test_wrong_side_and_insufficient_panel_budget_fail_closed(self):
        polygon = local_polygon()
        normal = np.array([0., 1., 0.])
        with self.assertRaisesRegex(ValueError, 'positive entry hemisphere'):
            spherical_contact_line_quadrature(polygon, -normal)
        with self.assertRaisesRegex(ValueError, 'unit hinge'):
            spherical_contact_line_quadrature(polygon, normal * 2.)
        with self.assertRaisesRegex(ValueError, 'did not close'):
            spherical_contact_line_quadrature(
                polygon, normal, area_tolerance=1e-12, max_panels=4)


if __name__ == '__main__':
    unittest.main()
