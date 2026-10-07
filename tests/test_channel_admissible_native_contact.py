"""Physically admissible native identities through read-only contact loads."""

import unittest

import numpy as np

import channel_entry_zones
import entry_regions
import mesh_coverage
import slab_memory
from channel_contact_line_union import spherical_contact_line_union
from channel_contact_union_loads import lump_projected_contact_loads
from channel_native_contact_strip import bind_contact_zone_upper_strip
from channel_native_regional_contact_loads import bind_regional_projected_loads
from entry_projected_contact import projected_pair_areas
from tests.admissible_native_contact import admissible_native_contact_world


class AdmissibleNativeContactTests(unittest.TestCase):
    def test_native_entry_upper_reference_and_projected_lower_force(self):
        s, lower_id, upper_id = admissible_native_contact_world()
        before = s.channel_region_store
        spec = entry_regions.specification(s)
        self.assertEqual(list(spec['face_ids']), [lower_id])
        normal = spec['hinge_normals'][0]
        dip = float(spec['dip_degrees'][0])
        lower = s.material_surface['vertices'][s.material_surface['faces'][0]]
        upper = s.material_surface['vertices'][s.material_surface['faces'][1]]
        q = 6371000. * np.arcsin(lower @ normal)
        self.assertLess(float(q.max()),
                        slab_memory.upper_mantle_length_km(dip) * 1000.)
        records = {int(record['face_id']): record
                   for record in s.channel_region_store['records']}
        current = channel_entry_zones.partition_native_entry_zones(
            s.material_surface, s.parcel_collision_sheet, s.parcel_plate,
            s.plate_uid, s.collision_contacts, spec, list(records),
            epoch_myr=s.t, upper_records_by_face_id=records)
        zones = [zone for zone in current['entry_faces'][lower_id]['zones']
                 if zone['kind'] == 'entered-contact'
                 and zone['adjacent_upper_face_id'] == upper_id]
        self.assertTrue(zones)
        zone = max(zones, key=lambda row: mesh_coverage._polygon_area(
            row['polygon'], 6371.))
        point = zone['polygon'].sum(axis=0)
        point /= np.linalg.norm(point)
        contact_offset = 6371000. * np.arcsin(point @ normal)
        strip = bind_contact_zone_upper_strip(
            s, lower_id, zone['zone_id'], point,
            contact_offset + np.linspace(-100., 100., 5))
        np.testing.assert_array_equal(strip['unloaded_upper_base_depth_m'], 35000.)
        self.assertEqual(strip['adjacent_upper_face_id'], upper_id)
        projected = projected_pair_areas(
            lower, upper, normal, dip, refinement=1, include_cells=True,
            coordinate_discretization='affine_nodal')
        self.assertLess(projected['physical_contact_area_km2'],
                        current['entry_faces'][lower_id]['contact_area_km2'])
        loads = bind_regional_projected_loads(s, lower_id, projected)
        union = spherical_contact_line_union([
            dict(zone_id=str(index), polygon=cell['physical_polygon'])
            for index, cell in enumerate(projected['contact_cells'])],
            normal, area_tolerance=1e-4)
        nodal = lump_projected_contact_loads(
            union, np.linspace(0., 160000., 65),
            {str(cell['cell_index']): cell for cell in loads['cell_loads']})
        np.testing.assert_allclose(nodal['total_represented_force_n'],
                                   loads['contact_buoyancy_force_n'], rtol=2e-12)
        self.assertIs(s.channel_region_store, before)

    def test_stale_hinge_epoch_is_rejected(self):
        s, _, _ = admissible_native_contact_world()
        s.continental_entry_regions['epoch_myr'] -= 1.
        with self.assertRaisesRegex(ValueError, 'epochs disagree'):
            entry_regions.specification(s)


if __name__ == '__main__':
    unittest.main()
