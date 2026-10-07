"""Area conservation and convergence of spherical contact quadrature."""

import unittest

import numpy as np

import mesh_coverage
from channel_contact_quadrature import spherical_contact_quadrature
from channel_native_contact_strip import bind_contact_zone_upper_strip
from ridge_geometry import rotate
from tests.test_channel_native_contact_strip import contact_fixture


class ContactQuadratureTests(unittest.TestCase):
    def test_current_native_zone_weights_match_exact_spherical_area(self):
        s, lower_id, _, zone_id, _, _ = contact_fixture()
        import channel_entry_zones
        spec = channel_entry_zones.entry_regions.specification(s)
        records = {int(record['face_id']): record
                   for record in s.channel_region_store['records']}
        geometry = channel_entry_zones.partition_native_entry_zones(
            s.material_surface, s.parcel_collision_sheet, s.parcel_plate,
            s.plate_uid, s.collision_contacts, spec, list(records),
            epoch_myr=s.t, upper_records_by_face_id=records)
        zone = next(zone for zone in geometry['entry_faces'][lower_id]['zones']
                    if zone['zone_id'] == zone_id)
        polygon = zone['polygon']
        expected_m2 = 1e6 * mesh_coverage._polygon_area(polygon, 6371.)
        for level in range(4):
            result = spherical_contact_quadrature(
                polygon, refinement_level=level)
            self.assertEqual(len(result['xyz']), (len(polygon) - 2) * 4**level)
            np.testing.assert_allclose(result['area_m2'].sum(), expected_m2,
                                       rtol=2e-10)
            np.testing.assert_allclose(result['polygon_area_m2'], expected_m2)
            self.assertTrue(np.all(result['area_m2'] > 0.))
            self.assertTrue(np.all(np.linalg.norm(result['xyz'], axis=1) > 1. - 1e-12))
            planes = mesh_coverage._triangle_planes(polygon)
            self.assertTrue(np.all(result['xyz'] @ planes.T >= -1e-11))
            if level == 0:
                contact = result['xyz'][0]
                row = int(np.flatnonzero(spec['face_ids'] == lower_id)[0])
                position = 6371000. * np.arcsin(
                    contact @ spec['hinge_normals'][row])
                bound = bind_contact_zone_upper_strip(
                    s, lower_id, zone_id, contact,
                    position + np.linspace(-20., 20., 5))
                self.assertEqual(bound['zone_id'], zone_id)
                self.assertEqual(len(bound['face_id']), 5)

    def test_rotation_preserves_sites_and_weights(self):
        triangle = np.array([[1., 0., 0.], [1., .12, 0.], [1., .05, .1]])
        triangle /= np.linalg.norm(triangle, axis=1)[:, None]
        original = spherical_contact_quadrature(triangle, refinement_level=2)
        rotation = np.array([.13, -.2, .4])
        turned = spherical_contact_quadrature(
            rotate(triangle, rotation), refinement_level=2)
        np.testing.assert_allclose(turned['xyz'],
                                   rotate(original['xyz'], rotation), atol=1e-14)
        np.testing.assert_allclose(turned['area_m2'], original['area_m2'],
                                   rtol=2e-12)

    def test_refinement_improves_smooth_field_integral(self):
        triangle = np.array([[1., 0., 0.], [1., .15, 0.], [1., .06, .12]])
        triangle /= np.linalg.norm(triangle, axis=1)[:, None]
        def integrate(level):
            result = spherical_contact_quadrature(triangle, refinement_level=level)
            return float(result['area_m2'] @ (result['xyz'][:, 1]**2))
        reference = integrate(5)
        errors = [abs(integrate(level) - reference) for level in (0, 1, 2, 3)]
        self.assertTrue(all(after < before for before, after in zip(errors, errors[1:])))

    def test_invalid_geometry_and_budget_fail_closed(self):
        triangle = np.array([[1., 0., 0.], [1., .1, 0.], [1., .05, .1]])
        triangle /= np.linalg.norm(triangle, axis=1)[:, None]
        with self.assertRaisesRegex(ValueError, 'node budget'):
            spherical_contact_quadrature(
                triangle, refinement_level=5, max_nodes=100)
        with self.assertRaisesRegex(ValueError, 'bounded refinement'):
            spherical_contact_quadrature(triangle, refinement_level=-1)
        with self.assertRaisesRegex(ValueError, 'convex polygon'):
            spherical_contact_quadrature(triangle[::-1])


if __name__ == '__main__':
    unittest.main()
