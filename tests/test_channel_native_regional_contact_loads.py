"""Keep separate native phase histories through projected contact force."""

from types import SimpleNamespace
import unittest

import numpy as np

import channel_region_native
import dense_crust
import mesh_coverage
from channel_contact_line_union import spherical_contact_line_union
from channel_contact_union_loads import lump_projected_contact_loads
from channel_native_regional_contact_loads import bind_regional_projected_loads
from entry_projected_contact import projected_pair_areas
from tests.test_channel_entry_zones import paired
from tests.test_channel_region_native import uniform_world


def regional_pair():
    surface, _, _, _, _, spec = paired()
    lower = surface['vertices'][surface['faces'][0]]
    upper = surface['vertices'][surface['faces'][1]]
    normal = spec['hinge_normals'][0]
    a, b, c = lower
    middle = (a + b) / np.linalg.norm(a + b)
    polygons = [np.array([a, middle, c]), np.array([middle, b, c])]
    face_area = mesh_coverage._polygon_area(lower, 6371.)
    fractions = [mesh_coverage._polygon_area(polygon, 6371.) / face_area
                 for polygon in polygons]
    source = uniform_world().channel_region_store['records'][0][
        'history']['regions'][0]['column']
    columns = [{key: value.copy() for key, value in source.items()}
               for _ in polygons]
    columns[1][dense_crust.DENSE][:] = 1.5
    columns[1][dense_crust.CONVERTED][:] = 1.5
    regions = [dict(region_id=str(index), fraction=float(fraction), column=column)
               for index, (fraction, column) in enumerate(zip(fractions, columns))]
    record = dict(face_id=11, face_triangle=lower.copy(), polygons=polygons,
                  history=dict(face_id=11, face_index=0, regions=regions))
    native_surface = dict(vertices=lower.copy(), faces=np.array([[0, 1, 2]]),
                          face_id=np.array([11]), area_km2=np.array([face_area]),
                          reference_area_km2=np.array([face_area]),
                          geometry_revision=0, radius_km=6371.)
    s = SimpleNamespace(t=0., material_surface=native_surface,
                        mass=np.array([face_area]),
                        structure=channel_region_native._project(record, set(source)),
                        channel_region_store=dict(version=1, epoch_myr=0.,
                                                  geometry_revision=0,
                                                  records=[record]))
    geometry = projected_pair_areas(
        lower, upper, normal, 50., refinement=1, include_cells=True,
        coordinate_discretization='affine_nodal')
    return s, geometry, normal


class NativeRegionalContactLoadTests(unittest.TestCase):
    def test_distinct_phase_regions_feed_conserved_hinge_force(self):
        s, geometry, normal = regional_pair()
        before = s.channel_region_store
        result = bind_regional_projected_loads(s, 11, geometry)
        self.assertIs(s.channel_region_store, before)
        self.assertEqual(len(result['cell_loads']), len(geometry['contact_cells']))
        self.assertTrue(any(len(cell['regional_contributions']) == 2
                            for cell in result['cell_loads']))
        self.assertGreater(result['max_affine_spherical_area_mismatch'], 0.)
        self.assertLess(result['max_affine_spherical_area_mismatch'], 1e-3)
        self.assertGreater(result['max_outside_source_area_fraction'], 0.)
        self.assertLess(result['max_outside_source_area_fraction'], 1e-4)
        for cell in result['cell_loads']:
            np.testing.assert_allclose(
                cell['buoyancy_force_n'],
                sum(row['buoyancy_force_n']
                    for row in cell['regional_contributions']), rtol=2e-13)
            for row in cell['regional_contributions']:
                if row['region_id'] == '0':
                    self.assertEqual(row['dense_volume_km3'], 0.)
                else:
                    self.assertGreater(row['dense_volume_km3'], 0.)
        union = spherical_contact_line_union([
            dict(zone_id=str(index), polygon=cell['physical_polygon'])
            for index, cell in enumerate(geometry['contact_cells'])],
            normal, area_tolerance=1e-4)
        loads = lump_projected_contact_loads(
            union, np.linspace(0., 160000., 65),
            {str(cell['cell_index']): cell for cell in result['cell_loads']})
        np.testing.assert_allclose(loads['total_represented_force_n'],
                                   result['contact_buoyancy_force_n'], rtol=2e-12)

    def test_phase_and_source_area_mismatch_fail_closed(self):
        s, geometry, _ = regional_pair()
        with self.assertRaisesRegex(ValueError, 'spherical source areas disagree'):
            bind_regional_projected_loads(s, 11, geometry,
                                          area_mismatch_tolerance=1e-5)
        forged = dict(geometry, contact_cells=[dict(cell)
                      for cell in geometry['contact_cells']])
        forged['contact_cells'][0]['material_reference_area_km2'] *= 1.01
        with self.assertRaisesRegex(ValueError, 'spherical source areas disagree'):
            bind_regional_projected_loads(s, 11, forged)
        s.channel_region_store['records'][0]['history']['regions'][1][
            'column'][dense_crust.DENSE][:] = 40.
        with self.assertRaises(ValueError):
            bind_regional_projected_loads(s, 11, geometry)


if __name__ == '__main__':
    unittest.main()
