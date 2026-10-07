"""Read-only material refinement of separate channel phase histories."""

from copy import deepcopy
import unittest

import numpy as np

import adaptive_material
import dense_crust
import mesh_coverage
from experiments.channel_marker_binding import bind_marker_regions
from experiments.channel_region_history import (
    advance_region_history, expand_zone_loads)
from experiments.channel_region_refinement import split_history_to_children
from tests.test_channel_marker_binding import fixture, unit


def children_of(triangle):
    a, b, c = triangle
    ab, bc, ca = unit(a+b), unit(b+c), unit(c+a)
    triangles = [np.array([a, ab, ca]), np.array([ab, b, bc]),
                 np.array([ca, bc, c]), np.array([ab, bc, ca])]
    return [dict(face_id=500+i, face_index=i, triangle=points)
            for i, points in enumerate(triangles)]


def absolute_inventory(records, field):
    total = 0.
    for record in records:
        for region, polygon in zip(record['history']['regions'], record['polygons']):
            column = region['column']
            total += (mesh_coverage._polygon_area(polygon, 6371.)
                      * field(column)[0] / column['area_factor'][0])
    return total


class ChannelRegionRefinementTests(unittest.TestCase):
    def test_native_refinement_map_supplies_a_conservative_child_partition(self):
        model, record, _ = fixture()
        surface = model.material_surface
        target = np.full(len(surface['faces']), np.inf)
        target[record['history']['face_index']] = 1.
        mapping = adaptive_material.refine(
            surface['vertices'], surface['faces'], surface['face_owner'],
            surface['face_kind'], desired_edge_km=target,
            face_ids=surface['face_id'])
        indices = np.flatnonzero(mapping['source_face'] == record['history']['face_index'])
        self.assertGreater(len(indices), 1)
        children = [dict(face_id=100000+int(index), face_index=int(index),
                         triangle=mapping['vertices'][mapping['faces'][index]])
                    for index in indices]
        result = split_history_to_children(record, children)
        np.testing.assert_allclose(result['new_crust_volume_km3'],
                                   result['old_crust_volume_km3'], rtol=2e-10)
        np.testing.assert_allclose(result['new_stored_heat_km3_c'],
                                   result['old_stored_heat_km3_c'], rtol=2e-10)

    def test_subdivision_preserves_distinct_heat_and_marker_selection(self):
        model, record, _ = fixture()
        right = record['history']['regions'][1]['column']
        right[dense_crust.HEAT] += 10.
        right[dense_crust.HEAT_ADDED] += 10.
        original = deepcopy(record)
        children = children_of(record['face_triangle'])
        result = split_history_to_children(record, children)
        self.assertEqual(len(result['records']), 4)
        self.assertGreater(sum(len(row['history']['regions'])
                               for row in result['records']), 4)
        self.assertAlmostEqual(result['old_crust_volume_km3'] /
                               result['new_crust_volume_km3'], 1., places=10)
        self.assertAlmostEqual(result['old_stored_heat_km3_c'] /
                               result['new_stored_heat_km3_c'], 1., places=10)
        self.assertEqual(original['history']['regions'][1]['column'][dense_crust.HEAT][0],
                         record['history']['regions'][1]['column'][dense_crust.HEAT][0])
        surface = dict(vertices=np.concatenate([child['triangle'] for child in children]),
                       faces=np.arange(12).reshape(4, 3),
                       face_id=np.array([child['face_id'] for child in children]))
        records = result['records']
        markers = [(region['region_id'], row['face_id'], unit(polygon.sum(axis=0)))
                   for row in records
                   for region, polygon in zip(row['history']['regions'], row['polygons'])]
        bound = bind_marker_regions(surface, records,
                                    np.arange(len(markers)),
                                    np.array([marker[1] for marker in markers]),
                                    np.array([marker[2] for marker in markers]))
        self.assertEqual([row['region_id'] for row in bound],
                         [marker[0] for marker in markers])

    def test_source_step_commutes_with_exact_refinement(self):
        model, record, _ = fixture()
        children = children_of(record['face_triangle'])
        split = split_history_to_children(record, children)['records']
        zone_loads = [dict(zone_id='left', lower_top_depth_m=0.,
                           current_upper_base_depth_m=0.,
                           upper_column_mass_kg_m2=0.,
                           effective_thermal_depth_km=0.),
                      dict(zone_id='right', lower_top_depth_m=80000.,
                           current_upper_base_depth_m=0.,
                           upper_column_mass_kg_m2=0.,
                           effective_thermal_depth_km=80.)]
        old_loads = expand_zone_loads(record['history'], zone_loads)
        old = advance_region_history(record['history'], old_loads, .25,
                                     constitutive_parameters=model.retained_phase_parameters)
        record['history'] = old['history']
        advanced = []
        for child in split:
            active = {region['zone_id'] for region in child['history']['regions']}
            loads = expand_zone_loads(
                child['history'], [load for load in zone_loads if load['zone_id'] in active])
            updated = advance_region_history(
                child['history'], loads, .25,
                constitutive_parameters=model.retained_phase_parameters)
            child['history'] = updated['history']
            advanced.append(child)
        for field in (dense_crust.mass_volume,
                      lambda column: column[dense_crust.HEAT]):
            np.testing.assert_allclose(absolute_inventory([record], field),
                                       absolute_inventory(advanced, field),
                                       rtol=2e-10, atol=1e-6)

    def test_incomplete_or_ambiguous_child_partition_is_rejected(self):
        _, record, _ = fixture()
        children = children_of(record['face_triangle'])
        with self.assertRaisesRegex(ValueError, 'cover'):
            split_history_to_children(record, children[:-1])
        repeated = deepcopy(children)
        repeated[1]['face_id'] = repeated[0]['face_id']
        with self.assertRaisesRegex(ValueError, 'unique'):
            split_history_to_children(record, repeated)
        stale = deepcopy(record)
        stale['history']['face_id'] = -1
        with self.assertRaisesRegex(ValueError, 'persistent old-face'):
            split_history_to_children(stale, children)


if __name__ == '__main__':
    unittest.main()
