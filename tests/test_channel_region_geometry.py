"""Spherical material-area checks for moving source-region histories."""

import unittest

import numpy as np

import crust_inventory
import dense_crust
import mesh_coverage
from experiments.channel_region_geometry import (
    spherical_partition_intersections, transport_material_polygon)
from experiments.entry_channel_source_oracle import (
    initialize_region_history, remap_history_from_polygons,
    remap_history_after_face_motion, remap_history_with_local_strain,
    advance_region_history, expand_zone_loads)
from tests.test_entry_phase_depth import world


def unit(rows):
    points = np.asarray(rows, float)
    return points / np.linalg.norm(points, axis=1)[:, None]


class ChannelRegionGeometryTests(unittest.TestCase):
    def setUp(self):
        self.face = unit([[1., 0., 0.], [1., .08, 0.], [1., 0., .08]])
        old_plane = np.array([0., 1., -1.])
        new_plane = np.array([0., 1., -.4])
        self.old = [mesh_coverage._clip_planes(self.face, [plane])
                    for plane in (old_plane, -old_plane)]
        self.new = [mesh_coverage._clip_planes(self.face, [plane])
                    for plane in (new_plane, -new_plane)]

    def test_area_matrix_closes_both_spherical_partitions(self):
        geometry = spherical_partition_intersections(
            self.face, self.old, self.new)
        matrix = geometry['overlap_fractions']
        self.assertEqual(matrix.shape, (2, 2))
        np.testing.assert_allclose(matrix.sum(axis=1),
                                   geometry['old_fractions'], rtol=0., atol=2e-10)
        np.testing.assert_allclose(matrix.sum(axis=0),
                                   geometry['new_fractions'], rtol=0., atol=2e-10)
        self.assertAlmostEqual(float(matrix.sum()), 1., places=9)
        self.assertEqual(np.count_nonzero(matrix > 1e-12), 3)
        history = initialize_region_history(world().structure, 0, [
            dict(region_id='old-a', fraction=float(geometry['old_fractions'][0])),
            dict(region_id='old-b', fraction=float(geometry['old_fractions'][1]))])
        report = remap_history_from_polygons(
            history, self.face, self.old,
            [dict(zone_id='new-a', polygon=self.new[0]),
             dict(zone_id='new-b', polygon=self.new[1])])
        self.assertEqual(len(report['history']['regions']), 3)
        np.testing.assert_allclose(
            sorted(region['fraction'] for region in report['history']['regions']),
            sorted(matrix[matrix > 0.]), rtol=0., atol=2e-10)

    def test_rigid_rotation_preserves_intersection_fractions(self):
        axis = np.array([.3, .6, .7])
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        angle = .9
        rotation = (np.eye(3) + np.sin(angle) * skew
                    + (1. - np.cos(angle)) * skew @ skew)
        original = spherical_partition_intersections(
            self.face, self.old, self.new)
        rotated = spherical_partition_intersections(
            self.face @ rotation.T,
            [polygon @ rotation.T for polygon in self.old],
            [polygon @ rotation.T for polygon in self.new])
        np.testing.assert_allclose(
            rotated['overlap_fractions'], original['overlap_fractions'],
            rtol=2e-10, atol=2e-12)

    def test_material_frame_transport_accepts_rigid_motion_and_guards_strain(self):
        geometry = spherical_partition_intersections(
            self.face, self.old, self.new)
        history = initialize_region_history(world().structure, 0, [
            dict(region_id='a', fraction=float(geometry['old_fractions'][0])),
            dict(region_id='b', fraction=float(geometry['old_fractions'][1]))])
        angle = .4
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.],
                             [0., 0., 1.]])
        moved_face = self.face @ rotation.T
        report = remap_history_after_face_motion(
            history, self.face, moved_face, self.old,
            [dict(zone_id='open', polygon=self.new[0] @ rotation.T),
             dict(zone_id='contact', polygon=self.new[1] @ rotation.T)])
        np.testing.assert_allclose(
            report['geometry']['overlap_fractions'],
            geometry['overlap_fractions'], rtol=2e-10, atol=2e-12)
        for old, transported in zip(self.old, report['transported_old_polygons']):
            np.testing.assert_allclose(
                transport_material_polygon(moved_face, self.face, transported),
                old, rtol=0., atol=2e-12)

        strained_face = unit([[1., 0., 0.], [1., .12, 0.], [1., 0., .08]])
        with self.assertRaisesRegex(ValueError, 'saved material fractions'):
            remap_history_after_face_motion(
                history, self.face, strained_face, self.old,
                [dict(zone_id='all', polygon=strained_face)])

    def test_local_strain_preserves_child_mass_and_heat_before_source(self):
        model = world()
        geometry = spherical_partition_intersections(
            self.face, self.old, self.new)
        history = initialize_region_history(model.structure, 0, [
            dict(region_id='a', fraction=float(geometry['old_fractions'][0])),
            dict(region_id='b', fraction=float(geometry['old_fractions'][1]))])
        strained_face = unit([[1., 0., 0.], [1., .12, 0.], [1., 0., .08]])
        zones = [dict(zone_id='all', polygon=strained_face)]
        remapped = remap_history_with_local_strain(
            history, self.face, strained_face, self.old, zones)
        self.assertEqual(len(remapped['history']['regions']), 2)
        np.testing.assert_allclose(
            remapped['new_crust_volume_km3'],
            remapped['old_crust_volume_km3'], rtol=2e-10)
        np.testing.assert_allclose(
            remapped['new_stored_heat_km3_c'],
            remapped['old_stored_heat_km3_c'], rtol=2e-10)
        self.assertGreater(
            abs(remapped['history']['regions'][0]['column']['area_factor'][0]
                - remapped['history']['regions'][1]['column']['area_factor'][0]),
            .001)
        for old, child in zip(history['regions'], remapped['history']['regions']):
            self.assertAlmostEqual(
                old['column']['thickness_km'][0]
                * old['column']['area_factor'][0],
                child['column']['thickness_km'][0]
                * child['column']['area_factor'][0], places=11)
        loads = expand_zone_loads(remapped['history'], [
            dict(zone_id='all', lower_top_depth_m=80000.,
                 current_upper_base_depth_m=35000.,
                 upper_column_mass_kg_m2=2800. * 35000.,
                 effective_thermal_depth_km=80.)])
        evolved = advance_region_history(
            remapped['history'], loads, .25,
            constitutive_parameters=model.retained_phase_parameters)
        after = sum(
            cell['new_area_km2'] * (
                dense_crust.mass_volume(region['column'])[0]
                + region['column'][crust_inventory.RETURNED][0])
            / region['column']['area_factor'][0]
            for cell, region in zip(remapped['geometry']['cells'],
                                    evolved['history']['regions']))
        before = sum(
            area * (dense_crust.mass_volume(region['column'])[0]
                    + region['column'][crust_inventory.RETURNED][0])
            / region['column']['area_factor'][0]
            for area, region in zip(
                geometry['old_fractions'] * geometry['face_area_km2'],
                history['regions']))
        np.testing.assert_allclose(after, before, rtol=2e-10)
        area = remapped['geometry']['new_geometry']['face_area_km2']
        np.testing.assert_allclose(
            dense_crust.mass_volume(evolved['column'])[0]
            / evolved['column']['area_factor'][0],
            sum(cell['new_area_km2'] / area
                * dense_crust.mass_volume(region['column'])[0]
                / region['column']['area_factor'][0]
                for cell, region in zip(remapped['geometry']['cells'],
                                        evolved['history']['regions'])),
            rtol=2e-10)
        np.testing.assert_allclose(
            evolved['column'][dense_crust.HEAT][0]
            / evolved['column']['area_factor'][0],
            sum(cell['new_area_km2'] / area
                * region['column'][dense_crust.HEAT][0]
                / region['column']['area_factor'][0]
                for cell, region in zip(remapped['geometry']['cells'],
                                        evolved['history']['regions'])),
            rtol=2e-10)

    def test_local_strain_rejects_crust_outside_mechanical_bounds(self):
        fractions = spherical_partition_intersections(
            self.face, self.old, self.new)['old_fractions']
        history = initialize_region_history(world().structure, 0, [
            dict(region_id='a', fraction=float(fractions[0])),
            dict(region_id='b', fraction=float(fractions[1]))])
        collapsed_face = unit([[1., 0., 0.], [1., .008, 0.], [1., 0., .008]])
        with self.assertRaisesRegex(ValueError, 'thickness_km'):
            remap_history_with_local_strain(
                history, self.face, collapsed_face, self.old,
                [dict(zone_id='all', polygon=collapsed_face)])

    def test_gap_overlap_outside_and_winding_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap each other'):
            spherical_partition_intersections(
                self.face, [self.face, self.face], self.new)
        with self.assertRaisesRegex(ValueError, 'do not cover'):
            spherical_partition_intersections(
                self.face, [self.old[0]], self.new)
        with self.assertRaisesRegex(ValueError, 'extends beyond'):
            spherical_partition_intersections(
                self.face, [self.face, unit([[1., -.01, 0.],
                                             [1., -.01, -.01], [1., 0., -.01]])],
                self.new)
        with self.assertRaisesRegex(ValueError, 'outward convex'):
            spherical_partition_intersections(
                self.face, [self.old[0][::-1], self.old[1]], self.new)


if __name__ == '__main__':
    unittest.main()
