"""Current native lower material mass and volume for projected contact."""

import unittest
from types import SimpleNamespace

import numpy as np

import dense_crust
import mesh_coverage
from channel_contact_line_union import spherical_contact_line_union
from channel_contact_union_loads import lump_projected_contact_loads
from channel_native_lower_inventory import bind_uniform_lower_inventory
from entry_projected_contact import projected_pair_areas
from entry_projected_loads import uniform_cell_buoyancy
from tests.test_channel_entry_zones import paired
from tests.test_channel_region_native import uniform_world, installed_world


class NativeLowerInventoryTests(unittest.TestCase):
    def test_native_format_lower_column_feeds_projected_hinge_loads(self):
        surface, _, _, _, _, spec = paired()
        lower = surface['vertices'][surface['faces'][0]]
        upper = surface['vertices'][surface['faces'][1]]
        normal = spec['hinge_normals'][0]
        column = uniform_world().channel_region_store['records'][0][
            'history']['regions'][0]['column']
        area = mesh_coverage._polygon_area(lower, 6371.)
        native_surface = dict(vertices=lower.copy(), faces=np.array([[0, 1, 2]]),
                              face_id=np.array([11]), area_km2=np.array([area]),
                              reference_area_km2=np.array([area]),
                              geometry_revision=0, radius_km=6371.)
        record = dict(face_id=11, face_triangle=lower.copy(),
                      polygons=[lower.copy()], history=dict(regions=[
                          dict(region_id='uniform', fraction=1., column=column)]))
        s = SimpleNamespace(t=0., material_surface=native_surface,
                            mass=np.array([area]),
                            structure={key: value.copy() for key, value in column.items()},
                            channel_region_store=dict(version=1, epoch_myr=0.,
                                geometry_revision=0, records=[record]))
        inventory = bind_uniform_lower_inventory(s, 11)
        geometry = projected_pair_areas(
            inventory['lower_triangle'], upper, normal, 50., refinement=1,
            include_cells=True, coordinate_discretization='affine_nodal')
        loads = uniform_cell_buoyancy(
            inventory['physical_volume_km3'], inventory['crust_mass_kg'],
            inventory['physical_area_km2'], geometry['contact_cells'])
        union = spherical_contact_line_union([
            dict(zone_id=str(index), polygon=cell['physical_polygon'])
            for index, cell in enumerate(geometry['contact_cells'])],
            normal, area_tolerance=1e-4)
        result = lump_projected_contact_loads(
            union, np.linspace(0., 160000., 65),
            {str(cell['cell_index']): cell for cell in loads['cells']})
        np.testing.assert_allclose(result['total_represented_force_n'],
                                   loads['contact_buoyancy_force_n'], rtol=2e-12)

    def test_uniform_native_column_supplies_conserved_projected_force(self):
        s = uniform_world()
        face_id = int(s.channel_region_store['records'][0]['face_id'])
        before = s.channel_region_store
        result = bind_uniform_lower_inventory(s, face_id)
        self.assertIs(s.channel_region_store, before)
        self.assertEqual(result['face_id'], face_id)
        self.assertEqual(result['region_id'], 'uniform')
        self.assertEqual(result['dense_volume_km3'], 0.)
        self.assertAlmostEqual(result['mean_density_kg_m3'], 2800.)
        area = result['physical_area_km2']
        cell = dict(material_reference_area_km2=area / 2.,
                    physical_area_km2=area / 4.)
        load = uniform_cell_buoyancy(result['physical_volume_km3'],
                                     result['crust_mass_kg'], area, [cell])
        self.assertAlmostEqual(load['cells'][0]['material_volume_km3']
                               / result['physical_volume_km3'], .5)
        self.assertAlmostEqual(load['cells'][0]['material_mass_kg']
                               / result['crust_mass_kg'], .5)
        self.assertAlmostEqual(load['cells'][0]['mean_buoyancy_pa']
                               / load['source_reference_buoyancy_pa'], 2.)

    def test_retained_dense_phase_and_area_strain_preserve_native_mass_basis(self):
        s = uniform_world()
        face_id = int(s.channel_region_store['records'][0]['face_id'])
        column = s.channel_region_store['records'][0]['history']['regions'][0]['column']
        column[dense_crust.DENSE][:] = 1.
        column[dense_crust.CONVERTED][:] = 1.
        column['area_factor'][:] = .8
        column['thickness_km'][:] /= .8
        for key in s.structure:
            s.structure[key][0] = column[key][0]
        physical = float(s.material_surface['area_km2'][0])
        s.material_surface['reference_area_km2'][0] = physical / .8
        s.mass[0] = physical / .8
        dense_crust.validate(column)
        result = bind_uniform_lower_inventory(s, face_id)
        self.assertAlmostEqual(result['material_reference_area_km2']
                               / result['physical_area_km2'], 1.25)
        self.assertAlmostEqual(result['dense_volume_km3']
                               / result['physical_area_km2'], 1.25)
        self.assertAlmostEqual(result['ordinary_volume_km3']
                               + result['dense_volume_km3'],
                               result['physical_volume_km3'])
        self.assertGreater(result['mean_density_kg_m3'], 2800.)
        self.assertLess(result['mean_density_kg_m3'], 3450.)
        np.testing.assert_allclose(result['crust_mass_kg'],
                                   1e9 * (2800. * result['ordinary_volume_km3']
                                          + 3450. * result['dense_volume_km3']),
                                   rtol=2e-13)

    def test_mixed_stale_and_misaligned_native_regions_fail_closed(self):
        mixed = installed_world()
        face_id = int(mixed.channel_region_store['records'][0]['face_id'])
        with self.assertRaisesRegex(ValueError, 'one uniform regional'):
            bind_uniform_lower_inventory(mixed, face_id)
        s = uniform_world()
        with self.assertRaisesRegex(ValueError, 'current native regional geometry'):
            s.material_surface['geometry_revision'] += 1
            bind_uniform_lower_inventory(s, face_id)
        s.material_surface['geometry_revision'] -= 1
        s.structure['thickness_km'][0] += 1.
        with self.assertRaisesRegex(ValueError, 'current native column'):
            bind_uniform_lower_inventory(s, face_id)
        s.structure['thickness_km'][0] -= 1.
        s.mass[0] *= 2.
        with self.assertRaisesRegex(ValueError, 'area, strain'):
            bind_uniform_lower_inventory(s, face_id)


if __name__ == '__main__':
    unittest.main()
