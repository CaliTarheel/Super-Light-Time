"""Reciprocal force assembly from conserved physical contact cells."""

import math
import unittest

import numpy as np

from channel_contact_line_union import spherical_contact_line_union
from channel_contact_union_loads import lump_projected_contact_loads
from channel_contact_union_work import integrate_two_plate_contact_union
from entry_projected_contact import projected_pair_areas
from entry_projected_loads import uniform_cell_buoyancy
from tests.test_channel_entry_zones import paired


class ContactUnionLoadTests(unittest.TestCase):
    def test_lumped_load_has_reciprocal_work_in_free_hinge_contact(self):
        radius = 6371000.
        line_weight = 1200.
        upper = 80000.
        area = radius * line_weight * math.sin(upper / radius)
        line = dict(radius_m=radius, line_weight_m=np.array([line_weight]),
                    contact_intervals=[[dict(zone_id='one', lower_offset_m=0.,
                                             upper_offset_m=upper,
                                             represented_area_m2=area)]])
        offsets = np.linspace(0., 160000., 65)
        source = dict(one=dict(physical_area_km2=area / 1e6,
                               mean_buoyancy_pa=1e6,
                               buoyancy_force_n=area * 1e6))
        load = lump_projected_contact_loads(line, offsets, source)
        preferred = (offsets * np.tan(np.deg2rad(50.)))[None, :]
        base = np.full_like(preferred, 35000.)
        def solve(scale):
            return integrate_two_plate_contact_union(
                line, offsets, preferred, base, load['buoyancy_pa'] * scale,
                1e23, 1e23, 20000., free_upper_hinge=True)
        result = solve(1.)
        delta = .001
        derivative = (solve(1. + delta)['energy_j']
                      - solve(1. - delta)['energy_j']) / (2. * delta)
        expected = line_weight * float(load['nodal_force_n_per_m'][0]
                                       @ result['depth_m'][0])
        np.testing.assert_allclose(derivative, expected, rtol=2e-8)
        self.assertGreater(result['hinge_line_reaction_n_per_m'][0], 0.)
        self.assertGreaterEqual(result['minimum_gap_m'], -1e-6)

    def test_one_interval_preserves_signed_force_and_affine_virtual_work(self):
        radius = 6371000.
        line_weight = 1200.
        upper = 12500.
        area = radius * line_weight * math.sin(upper / radius)
        line = dict(radius_m=radius,
                    line_weight_m=np.array([line_weight]),
                    contact_intervals=[[dict(
                        zone_id='one', lower_offset_m=0., upper_offset_m=upper,
                        represented_area_m2=area)]])
        offsets = np.linspace(0., 20000., 9)
        for pressure in (900., -900.):
            load = dict(one=dict(physical_area_km2=area / 1e6,
                                 mean_buoyancy_pa=pressure,
                                 buoyancy_force_n=pressure * area))
            result = lump_projected_contact_loads(line, offsets, load)
            np.testing.assert_allclose(result['total_represented_force_n'],
                                       pressure * area, rtol=2e-14)
            np.testing.assert_allclose(result['nodal_force_n_per_m'].sum() * line_weight,
                                       pressure * area, rtol=2e-14)
            displacement = 400. + .02 * offsets
            actual = line_weight * float(result['nodal_force_n_per_m'][0] @ displacement)
            sites, weights = np.polynomial.legendre.leggauss(64)
            x = .5 * upper * (sites + 1.)
            expected = (pressure * line_weight * .5 * upper
                        * float(weights @ (np.cos(x / radius) * (400. + .02 * x))))
            np.testing.assert_allclose(actual, expected, rtol=3e-14)

    def test_native_pair_cell_forces_close_on_shared_hinge_rays(self):
        surface, _, _, _, _, spec = paired()
        lower = surface['vertices'][surface['faces'][0]]
        upper = surface['vertices'][surface['faces'][1]]
        normal = spec['hinge_normals'][0]
        geometry = projected_pair_areas(
            lower, upper, normal, 50., refinement=1, include_cells=True,
            coordinate_discretization='affine_nodal')
        cells = geometry['contact_cells']
        union = spherical_contact_line_union([
            dict(zone_id=str(index), polygon=cell['physical_polygon'])
            for index, cell in enumerate(cells)], normal, area_tolerance=1e-4)
        face_area = geometry['face_reference_area_km2']
        volume = 35. * face_area
        offsets = np.linspace(0., 160000., 65)
        for density in (2800., 3450.):
            material = uniform_cell_buoyancy(
                volume, volume * density * 1e9, face_area, cells)
            loads = {str(cell['cell_index']): cell for cell in material['cells']}
            assembled = lump_projected_contact_loads(union, offsets, loads)
            np.testing.assert_allclose(
                assembled['total_represented_force_n'],
                assembled['source_cell_force_n'], rtol=2e-12)
            self.assertEqual(np.sign(assembled['total_represented_force_n']),
                             np.sign(material['contact_buoyancy_force_n']))
            np.testing.assert_allclose(
                assembled['total_represented_force_n'],
                float(union['line_weight_m'] @
                      assembled['nodal_force_n_per_m'].sum(axis=1)),
                rtol=2e-13)
            for index, cell in enumerate(cells):
                self.assertLess(abs(assembled['represented_area_m2_by_zone'][str(index)]
                                    / (cell['physical_area_km2'] * 1e6) - 1.), 1e-3)
                np.testing.assert_allclose(
                    assembled['represented_force_n_by_zone'][str(index)],
                    loads[str(index)]['buoyancy_force_n'], rtol=2e-10)

    def test_unresolved_cell_and_mismatched_area_fail_closed(self):
        radius = 6371000.
        area = radius * 1000. * math.sin(10000. / radius)
        line = dict(radius_m=radius, line_weight_m=np.array([1000.]),
                    contact_intervals=[[dict(zone_id='one', lower_offset_m=0.,
                                             upper_offset_m=10000.,
                                             represented_area_m2=area)]])
        offsets = np.linspace(0., 20000., 9)
        valid = dict(one=dict(physical_area_km2=area / 1e6,
                              mean_buoyancy_pa=100., buoyancy_force_n=100. * area))
        with self.assertRaisesRegex(ValueError, 'physical area'):
            wrong = dict(one=dict(valid['one'], physical_area_km2=area / 1e6 * 2.,
                                  buoyancy_force_n=200. * area))
            lump_projected_contact_loads(line, offsets, wrong)
        with self.assertRaisesRegex(ValueError, 'no matching material cell'):
            lump_projected_contact_loads(line, offsets, dict(other=valid['one']))
        with self.assertRaisesRegex(ValueError, 'conserved force'):
            wrong = dict(one=dict(valid['one'], buoyancy_force_n=0.))
            lump_projected_contact_loads(line, offsets, wrong)
        overlap = dict(line, contact_intervals=[[
            line['contact_intervals'][0][0],
            dict(zone_id='one', lower_offset_m=5000., upper_offset_m=10000.,
                 represented_area_m2=radius * 1000.
                 * (math.sin(10000. / radius) - math.sin(5000. / radius)))]])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            lump_projected_contact_loads(overlap, offsets, valid)


if __name__ == '__main__':
    unittest.main()
