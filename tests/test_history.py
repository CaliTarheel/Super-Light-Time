"""History inspection contracts exercised with small, deliberately moving fixtures."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from server import SimulationManager, write_json


class HistoryTests(unittest.TestCase):
    """Synthetic trajectories make material/location confusion easy to detect."""

    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / 'tmp'
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.manager = SimulationManager(self.temporary.name)
        self.width, self.height = 8, 4
        self.marker_cells = [9, 10, 19, 20]
        self.arc_cells = {2: 26, 3: 27}
        self.events = [
            dict(id=1, time_myr=0, type='initial', text='Initial plate',
                 plate_ids=[0], plate_uids=[7], lon=-112.5, lat=22.5),
            dict(id=2, time_myr=2, type='rift', text='Plate separation',
                 plate_ids=[0, 1], plate_uids=[7, 8], lon=-67.5, lat=22.5),
            dict(id=3, time_myr=4, type='rift', text='Unrelated later plate reusing a numeric slot',
                 plate_ids=[0], plate_uids=[30], lon=112.5, lat=67.5),
            dict(id=4, time_myr=6, type='collision', text='Later collision',
                 plate_ids=[1, 2], plate_uids=[8, 9], lon=22.5, lat=-22.5),
        ]
        self.write_run('tracked', tracked=True)

    def coordinates(self, cell):
        return (-180 + (cell % self.width + .5) * 360 / self.width,
                90 - (cell // self.width + .5) * 180 / self.height)

    def xyz(self, cell):
        lon, lat = np.deg2rad(self.coordinates(cell))
        return [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]

    def snapshot(self, index, tracked):
        size = self.width * self.height
        marker_cell = self.marker_cells[index]
        marker_plate = 0 if index < 2 else 1
        marker_uid = 7 if index < 2 else 8
        frame = dict(width=self.width, height=self.height, time_myr=index * 2,
                     elevation=(100 * index + np.arange(size)).astype(np.float32),
                     plate=np.zeros(size, dtype=np.int32),
                     crust=np.zeros(size, dtype=np.uint8),
                     age=np.full(size, 20 + index * 2, dtype=np.float32),
                     boundary=np.zeros(size, dtype=np.uint8), stats={},
                     events=[event for event in self.events if event['time_myr'] <= index * 2],
                     plates=[dict(id=0, uid=7, generation=0, name='First'),
                             dict(id=1, uid=8, generation=0, name='Second'),
                             dict(id=2, uid=9, generation=0, name='Arc')])
        frame['plate'][marker_cell] = marker_plate
        frame['crust'][marker_cell] = 2
        if index == 2:
            frame['boundary'][marker_cell] = 4
        if index in self.arc_cells:
            frame['plate'][self.arc_cells[index]] = 2
            frame['crust'][self.arc_cells[index]] = 3
        if tracked:
            has_arc = index >= 2
            frame.update(history_version=1,
                         trace_id=np.array([101, 202] if has_arc else [101], dtype=np.int64),
                         trace_xyz=np.asarray([self.xyz(marker_cell)] +
                                              ([self.xyz(self.arc_cells[index])] if has_arc else [])),
                         trace_plate=np.array([marker_plate, 2] if has_arc else [marker_plate], dtype=np.int32),
                         trace_plate_uid=np.array([marker_uid, 9] if has_arc else [marker_uid], dtype=np.int64),
                         trace_kind=np.array([2, 3] if has_arc else [2], dtype=np.uint8),
                         trace_origin_kind=np.array([2, 3] if has_arc else [2], dtype=np.uint8),
                         trace_birth_myr=np.array([0., 4.] if has_arc else [0.]),
                         trace_relief_m=np.array([200. + index * 10, 800.] if has_arc else [200. + index * 10]),
                         trace_uplift_m=np.array([index * 30., 5.] if has_arc else [index * 30.]),
                         trace_extension_m=np.array([index * 4., 1.] if has_arc else [index * 4.]),
                         trace_erosion_m=np.array([index * 2., -3.] if has_arc else [index * 2.]),
                         trace_adjustment_m=np.array([0., 2.] if has_arc else [0.]),
                         trace_suture=np.array([index * .1, 0.] if has_arc else [index * .1]))
        return frame

    def write_run(self, run_id, tracked):
        run_path = self.manager.path(run_id)
        run_path.mkdir(exist_ok=True)
        config = dict(width=self.width, height=self.height, duration_myr=6,
                      dt_myr=2, snapshot_myr=2)
        self.manager.current = dict(state='complete', run_id=run_id, config=config,
                                    frame_count=4, time_myr=6, duration_myr=6,
                                    frames=[dict(index=i, time_myr=i * 2, stats={}) for i in range(4)])
        for index in range(4):
            self.manager.save_frame(index, self.snapshot(index, tracked))
        write_json(run_path / 'initial.json', dict(width=self.width, height=self.height,
                                                  crust=self.snapshot(0, tracked)['crust']))
        write_json(run_path / 'config.json', config)
        self.manager.persist()

    def test_material_history_moves_with_marker_and_preserves_plate_transfer(self):
        result = self.manager.history(3, self.marker_cells[3], 'tracked')
        self.assertEqual(result['mode'], 'material')
        self.assertEqual(result['trace_id'], 101)
        self.assertEqual(result['selection']['frame'], 3)
        self.assertEqual(result['selection']['marker_distance_km'], 0)
        points = result['points']
        self.assertEqual([p['time_myr'] for p in points], [0, 2, 4, 6])
        np.testing.assert_allclose([[p['lon'], p['lat']] for p in points],
                                   [self.coordinates(cell) for cell in self.marker_cells])
        # Geographic elevation follows each moving cell, not the selected final cell.
        self.assertEqual([p['elevation_m'] for p in points], [9, 110, 219, 320])
        self.assertEqual([p['plate_id'] for p in points], [0, 0, 1, 1])
        self.assertEqual([p['plate_uid'] for p in points], [7, 7, 8, 8])
        self.assertEqual(points[2]['boundary'], 4)
        self.assertEqual(points[3]['uplift_m'], 90)
        self.assertEqual(points[3]['extension_m'], 12)
        self.assertEqual(points[3]['erosion_m'], 6)
        self.assertEqual(result['events_scope'], 'host_plate')
        self.assertEqual([e['id'] for e in result['events']], [1, 2, 4])

    def test_later_arc_has_no_invented_prebirth_history(self):
        result = self.manager.history(-1, self.arc_cells[3])
        self.assertEqual(result['mode'], 'material')
        self.assertEqual(result['trace_id'], 202)
        self.assertEqual([p['time_myr'] for p in result['points']], [4, 6])
        self.assertEqual([p['birth_myr'] for p in result['points']], [4, 4])
        self.assertTrue(all(p['origin_kind'] == 3 for p in result['points']))
        # Negative relief relaxation and numerical adjustment remain distinguishable.
        self.assertEqual(result['points'][-1]['erosion_m'], -3)
        self.assertEqual(result['points'][-1]['adjustment_m'], 2)

    def test_full_record_includes_events_later_than_selected_frame(self):
        self.manager.frame(0)
        result = self.manager.record('tracked')
        self.assertEqual(result['history_version'], 1)
        self.assertEqual(result['frame_count'], 4)
        self.assertEqual([f['time_myr'] for f in result['frames']], [0, 2, 4, 6])
        self.assertEqual([e['id'] for e in result['events']], [1, 2, 3, 4])
        self.assertEqual(result['events'][-1]['time_myr'], 6)
        earlier_selection = self.manager.history(0, self.marker_cells[0])
        self.assertEqual([p['time_myr'] for p in earlier_selection['points']], [0, 2, 4, 6])
        self.assertEqual(earlier_selection['events'][-1]['time_myr'], 6)

    def test_ocean_location_is_not_assigned_a_continental_material_history(self):
        result = self.manager.history(3, 0)
        self.assertEqual(result['mode'], 'location')
        self.assertIsNone(result['trace_id'])
        self.assertEqual([p['elevation_m'] for p in result['points']], [0, 100, 200, 300])
        self.assertTrue(all((p['lon'], p['lat']) == self.coordinates(0) for p in result['points']))
        self.assertTrue(all('uplift_m' not in p for p in result['points']))

    def test_legacy_history_is_fixed_location_even_on_continental_crust(self):
        self.write_run('legacy', tracked=False)
        result = self.manager.history(3, self.marker_cells[3], 'legacy')
        self.assertEqual(result['mode'], 'location')
        self.assertIsNone(result['trace_id'])
        self.assertIn('Fixed geographic location', result['description'])
        self.assertEqual([p['elevation_m'] for p in result['points']], [20, 120, 220, 320])
        self.assertTrue(all((p['lon'], p['lat']) == self.coordinates(20) for p in result['points']))
        self.assertEqual(self.manager.record()['history_version'], 0)

    def test_invalid_frame_cell_and_stale_run_are_rejected(self):
        for frame, cell in ((4, 0), (-5, 0), (0, -1), (0, 32)):
            with self.subTest(frame=frame, cell=cell), self.assertRaises(ValueError):
                self.manager.history(frame, cell)
        for method in (lambda: self.manager.record('missing'),
                       lambda: self.manager.history(0, 9, 'missing')):
            with self.assertRaisesRegex(ValueError, 'experiment changed'):
                method()
        empty = SimulationManager(Path(self.temporary.name) / 'empty')
        with self.assertRaises(ValueError):
            empty.record()
        with self.assertRaises(ValueError):
            empty.history(0, 0)

    def test_saved_export_contains_traces_and_complete_catalogue(self):
        reloaded = SimulationManager(self.temporary.name)
        self.assertEqual(reloaded.history(3, 20)['trace_id'], 101)
        # The ordinary raster endpoint stays small; markers live in NPZ and inspection.
        self.assertFalse(any(key.startswith('trace_') for key in reloaded.frame(3)))
        with reloaded.export_file(0) as exported, zipfile.ZipFile(exported) as archive:
            self.assertIsNone(archive.testzip())
            catalogue = json.loads(archive.read('events.json'))
            self.assertEqual(catalogue['events'][-1]['time_myr'], 6)
            schema = json.loads(archive.read('history_schema.json'))
            self.assertEqual(schema['version'], 1)
            self.assertIn('trace_adjustment_m', schema['trace_fields'])
            with np.load(io.BytesIO(archive.read('history/frame_0003.npz')), allow_pickle=False) as data:
                for key, value in self.snapshot(3, tracked=True).items():
                    if key.startswith('trace_'):
                        np.testing.assert_array_equal(data[key], value)
            metadata = json.loads(archive.read('history/frame_0003.json'))
            self.assertEqual(metadata['history_version'], 1)
            self.assertFalse(any(key.startswith('trace_') for key in metadata))

    def test_persistence_rejects_broken_marker_identity_and_geometry(self):
        broken = self.snapshot(3, tracked=True)
        broken['trace_id'] = np.array([101, 101])
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.manager.save_frame(3, broken)
        broken = self.snapshot(3, tracked=True)
        broken['trace_xyz'][0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'invalid material'):
            self.manager.save_frame(3, broken)
        broken = self.snapshot(3, tracked=True)
        broken['trace_xyz'] = np.ones((2, 2))
        with self.assertRaisesRegex(ValueError, 'invalid material'):
            self.manager.save_frame(3, broken)

    def test_optional_ridge_counters_preserve_subset_and_current_offset(self):
        # A mixed record must not manufacture zero counters in earlier frames.
        for index, thermal in ((2, 140.), (3, 70.)):
            frame = self.snapshot(index, tracked=True)
            frame['trace_ridge_uplift_m'] = np.array([12. + index, 0.])
            frame['trace_ridge_thermal_m'] = np.array([thermal, 0.])
            frame['ridge_episodes'] = [dict(id=3, center=self.xyz(self.marker_cells[index]),
                                          started_myr=2., peak_myr=6., end_myr=34.,
                                          radius_km=600., overriding_plate_uid=8,
                                          incoming_plate_uids=[7, 9])]
            self.manager.save_frame(index, frame)
        points = self.manager.history(3, self.marker_cells[3])['points']
        self.assertNotIn('ridge_uplift_m', points[0])
        self.assertNotIn('ridge_thermal_m', points[1])
        self.assertEqual([p['ridge_uplift_m'] for p in points[2:]], [14., 15.])
        self.assertEqual([p['ridge_thermal_m'] for p in points[2:]], [140., 70.])
        self.assertEqual(points[-1]['uplift_m'], 90.)
        self.assertEqual(points[-1]['relief_m'], 230.)
        self.assertEqual(self.manager.frame(3)['ridge_episodes'][0]['id'], 3)
        with self.manager.export_file(3) as exported, zipfile.ZipFile(exported) as archive:
            schema = json.loads(archive.read('history_schema.json'))
            self.assertIn('subset', schema['trace_ridge_uplift_m'])
            self.assertIn('current', schema['trace_ridge_thermal_m'])
            self.assertIn('trace_ridge_thermal_m', schema['optional_trace_fields'])
            with np.load(io.BytesIO(archive.read('history/frame_0003.npz')), allow_pickle=False) as data:
                np.testing.assert_array_equal(data['trace_ridge_thermal_m'], [70., 0.])

    def test_optional_rift_history_records_origin_stretching_then_inversion(self):
        for index in range(1, 4):
            frame = self.snapshot(index, tracked=True)
            tail = [-1] if index >= 2 else []
            frame['trace_rift_id'] = np.array([17] + tail, dtype=np.int64)
            frame['trace_rift_birth_myr'] = np.array([0.] + tail)
            frame['trace_rift_extension_m'] = np.array([0. if index == 1 else 6.] + ([0.] if tail else []))
            frame['trace_inversion_uplift_m'] = np.array([15. if index == 3 else 0.] + ([0.] if tail else []))
            frame['rift_records'] = [dict(id=17, started_myr=0., origin='initial scar',
                                           center=self.xyz(self.marker_cells[0]), parent_plate_uids=[7])]
            self.manager.save_frame(index, frame)
        points = self.manager.history(3, self.marker_cells[3])['points']
        self.assertNotIn('rift_id', points[0])
        self.assertNotIn('rift_extension_m', points[0])
        self.assertIsInstance(points[1]['rift_id'], int)
        self.assertEqual(points[1]['rift_birth_myr'], 0.)
        self.assertEqual([p['rift_extension_m'] for p in points[1:]], [0., 6., 6.])
        self.assertEqual([p['inversion_uplift_m'] for p in points[1:]], [0., 0., 15.])
        self.assertEqual(points[-1]['uplift_m'], 90.)
        self.assertEqual(points[-1]['extension_m'], 12.)
        self.assertEqual(self.manager.frame(3)['rift_records'][0]['id'], 17)
        arc = self.manager.history(3, self.arc_cells[3])['points']
        self.assertTrue(all(p['rift_id'] == -1 and p['rift_birth_myr'] == -1 for p in arc))
        with self.manager.export_file(3) as exported, zipfile.ZipFile(exported) as archive:
            schema = json.loads(archive.read('history_schema.json'))
            self.assertIn('subset', schema['trace_inversion_uplift_m'])
            self.assertIn('does not imply', schema['trace_rift_birth_myr'])
            with np.load(io.BytesIO(archive.read('history/frame_0003.npz')), allow_pickle=False) as data:
                self.assertEqual(data['trace_rift_id'].dtype, np.int64)
                np.testing.assert_array_equal(data['trace_inversion_uplift_m'], [15., 0.])


if __name__ == '__main__':
    unittest.main()
