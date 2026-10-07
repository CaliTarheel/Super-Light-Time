"""Persistent material face and trace selection for channel phase histories."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

import checkpoint
import dense_crust
import mesh_coverage
from experiments.channel_marker_binding import bind_marker_regions
from experiments.channel_region_history import (
    initialize_region_history, advance_region_history, remap_region_history)
from tests.test_entry_phase_depth import world


def unit(point):
    return point / np.linalg.norm(point)


def fixture():
    model = world()
    surface = model.material_surface
    face = 0
    triangle = surface['vertices'][surface['faces'][face]]
    a, b, c = triangle
    middle = unit(a + b)
    polygons = [np.array([a, middle, c]), np.array([middle, b, c])]
    radius = float(surface.get('radius_km', 6371.))
    area = np.array([mesh_coverage._polygon_area(polygon, radius)
                     for polygon in polygons])
    face_id = int(surface['face_id'][face])
    history = initialize_region_history(model.structure, face, [
        dict(region_id='left', fraction=float(area[0] / area.sum())),
        dict(region_id='right', fraction=float(area[1] / area.sum()))], face_id=face_id)
    record = dict(face_id=face_id, face_triangle=triangle.copy(),
                  history=history, polygons=polygons)
    points = np.array([unit(polygon.sum(axis=0)) for polygon in polygons])
    return model, record, points


class ChannelMarkerBindingTests(unittest.TestCase):
    def test_trace_markers_select_distinct_regional_histories_by_stable_face_id(self):
        model, record, points = fixture()
        right = record['history']['regions'][1]['column']
        right[dense_crust.HEAT] += 10.
        right[dense_crust.HEAT_ADDED] += 10.
        result = bind_marker_regions(model.material_surface, [record],
                                     np.array([8, 3]),
                                     np.array([record['face_id']] * 2), points)
        self.assertEqual([row['region_id'] for row in result], ['left', 'right'])
        self.assertEqual([row['trace_id'] for row in result], [8, 3])
        self.assertEqual(result[1]['column'][dense_crust.HEAT][0], right[dense_crust.HEAT][0])
        result[1]['column'][dense_crust.HEAT][0] = 0.
        self.assertNotEqual(right[dense_crust.HEAT][0], 0.)
        reordered = dict(model.material_surface,
                         face_id=model.material_surface['face_id'][::-1].copy(),
                         faces=model.material_surface['faces'][::-1].copy())
        moved = bind_marker_regions(reordered, [record], np.array([8, 3]),
                                    np.array([record['face_id']] * 2), points)
        self.assertEqual([row['region_id'] for row in moved], ['left', 'right'])

    def test_unresolved_phase_boundary_and_stale_face_are_rejected(self):
        model, record, points = fixture()
        boundary = unit(record['polygons'][0][1] + record['polygons'][0][2])
        call = lambda source, point: bind_marker_regions(
            model.material_surface, [source], np.array([5]),
            np.array([source['face_id']]), point[None])
        self.assertEqual(call(record, boundary)[0]['region_id'], 'left')
        right = record['history']['regions'][1]['column']
        right[dense_crust.HEAT] += 10.
        right[dense_crust.HEAT_ADDED] += 10.
        with self.assertRaisesRegex(ValueError, 'unresolved boundary'):
            call(record, boundary)
        stale = deepcopy(record)
        stale['face_triangle'][0] = unit(stale['face_triangle'][0] + 1e-4 * points[0])
        with self.assertRaisesRegex(ValueError, 'stale'):
            call(stale, points[0])
        missing = deepcopy(record)
        missing['face_id'] = -1
        with self.assertRaisesRegex(ValueError, 'existing material face'):
            call(missing, points[0])

    def test_plain_checkpoint_replays_marker_binding_without_index_guessing(self):
        model, record, points = fixture()
        state = SimpleNamespace(t=1., config={}, rng=np.random.default_rng(1),
                                material_surface=model.material_surface,
                                channel_records=[record], trace_id=np.array([9, 4]),
                                trace_patch=np.array([record['face_id']] * 2),
                                trace_xyz=points)
        before = bind_marker_regions(state.material_surface, state.channel_records,
                                     state.trace_id, state.trace_patch, state.trace_xyz)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'channel-markers.npz'
            checkpoint.write_checkpoint(path, state, dict(config=state.config), {})
            restored, _ = checkpoint.read_checkpoint(path, {}, SimpleNamespace)
        after = bind_marker_regions(restored.material_surface, restored.channel_records,
                                    restored.trace_id, restored.trace_patch,
                                    restored.trace_xyz)
        self.assertEqual([(row['trace_id'], row['face_id'], row['region_id'])
                          for row in before],
                         [(row['trace_id'], row['face_id'], row['region_id'])
                          for row in after])
        for first, second in zip(before, after):
            for key in first['column']:
                np.testing.assert_array_equal(first['column'][key], second['column'][key])

    def test_boundary_rejects_different_unloaded_bases_with_equal_rock(self):
        model, record, points = fixture()
        regions = record['history']['regions']
        regions[0]['upper_basal_reference'] = dict(depth_m=34000., epoch_myr=1.)
        regions[1]['upper_basal_reference'] = dict(depth_m=35000., epoch_myr=1.)
        boundary = unit(record['polygons'][0][1] + record['polygons'][0][2])
        def bind(point):
            return bind_marker_regions(
                model.material_surface, [record], np.array([5]),
                np.array([record['face_id']]), point[None])[0]
        with self.assertRaisesRegex(ValueError, 'unresolved boundary'):
            bind(boundary)
        self.assertEqual(bind(points[0])['upper_basal_reference'],
                         regions[0]['upper_basal_reference'])
        chosen = bind(points[0])
        chosen['upper_basal_reference']['depth_m'] = 1.
        self.assertEqual(regions[0]['upper_basal_reference']['depth_m'], 34000.)
        regions[1]['upper_basal_reference']['depth_m'] = np.nan
        with self.assertRaisesRegex(ValueError, 'metadata is invalid'):
            bind(points[0])

    def test_face_identity_survives_source_step_and_exact_zone_split(self):
        model, record, points = fixture()
        history = record['history']
        loads = [dict(region_id=region['region_id'], lower_top_depth_m=0.,
                      current_upper_base_depth_m=0., upper_column_mass_kg_m2=0.,
                      effective_thermal_depth_km=0.)
                 for region in history['regions']]
        advanced = advance_region_history(
            history, loads, .1,
            constitutive_parameters=model.retained_phase_parameters)['history']
        self.assertEqual(advanced['face_id'], record['face_id'])
        zones = [dict(zone_id=region['region_id'], fraction=region['fraction'])
                 for region in advanced['regions']]
        matrix = np.diag([region['fraction'] for region in advanced['regions']])
        remapped = remap_region_history(advanced, zones, matrix)
        self.assertEqual(remapped['face_id'], record['face_id'])
        record['history'] = remapped
        selected = bind_marker_regions(model.material_surface, [record],
                                       np.array([1, 2]),
                                       np.array([record['face_id']] * 2), points)
        self.assertEqual([row['region_id'] for row in selected],
                         ['["left","left"]', '["right","right"]'])


if __name__ == '__main__':
    unittest.main()
