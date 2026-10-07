"""Bounded HTTP contracts for oriented review, frozen jobs, and live timing."""
from copy import deepcopy
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import zipfile

import numpy as np
from PIL import Image

import gospl_jobs
import terrain_jobs
from orientation import normalize_orientation, orient_metadata
from progress_timing import ProgressTiming
from server import Handler, png_bytes
from tests import test_history as fixture


class Clock:
    now = 0.

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class GatedBuilder:
    """Only test request/publication semantics; create a tiny eight-by-four output."""
    def __init__(self, kind, clock):
        self.kind, self.clock = kind, clock
        self.entered, self.release = threading.Event(), threading.Event()
        self.options = None
        self.source = None

    def __call__(self, first, path, *args, **options):
        for value in (.1, .2, .3):
            self.clock.advance(2)
            if self.kind == 'terrain': options['progress'](value)
            else: options['progress'](value, 'Preparing test output')
        self.entered.set()
        if not self.release.wait(10): raise RuntimeError('Test builder gate timed out')
        self.options = deepcopy({key: value for key, value in options.items() if key not in ('progress', 'cancel')})
        path = Path(path)
        if self.kind == 'terrain':
            self.source = dict(run_id=first['run_id'], elevation=np.asarray(first['elevation']).copy())
            values = np.arange(32, dtype=np.float32).reshape(4, 8)
            np.save(path/'elevation_m.npy', values)
            Image.fromarray((values+12000).astype(np.uint16)).save(path/'heightmap_16bit.png')
            Image.fromarray(np.zeros((4, 8, 3), dtype=np.uint8)).save(path/'preview.png')
            (path/'terrain_metadata.json').write_text(json.dumps({'time_myr': first['time_myr']}))
            options['progress'](1.)
        else:
            self.source = dict(run_id=args[0]['run_id'], path=str(first))
            with zipfile.ZipFile(path/'gospl-history.zip', 'w') as archive:
                archive.writestr('request.json', json.dumps(self.options))
            options['progress'](1., 'Test output ready')
            return dict(intervals=[dict(relaxation_supported_fraction=1.)])


class OrientationRouteTests(unittest.TestCase):
    coordinates = fixture.HistoryTests.coordinates
    xyz = fixture.HistoryTests.xyz
    snapshot = fixture.HistoryTests.snapshot
    write_run = fixture.HistoryTests.write_run

    def setUp(self):
        fixture.HistoryTests.setUp(self)
        self.clock = Clock()
        self.builders = []
        self.terrain = terrain_jobs.TerrainManager(Path(self.temporary.name)/'terrain')
        self.gospl = gospl_jobs.GosplManager(Path(self.temporary.name)/'gospl')
        simulation, terrain, gospl = self.manager, self.terrain, self.gospl

        class LocalHandler(Handler):
            manager, terrain_manager, gospl_manager = simulation, terrain, gospl
            def log_message(self, *args): pass

        self.http = ThreadingHTTPServer(('127.0.0.1', 0), LocalHandler)
        self.thread = threading.Thread(target=lambda: self.http.serve_forever(poll_interval=.01), daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.http.server_port}'
        self.addCleanup(self.close)

    def close(self):
        for builder in self.builders: builder.release.set()
        for manager in (self.terrain, self.gospl):
            if manager.worker: manager.worker.join(10)
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(5)

    def request(self, route, body=None, query=None, raw=False):
        if query: route += '?'+urlencode(query)
        request = Request(self.base+route, data=None if body is None else json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'}, method='GET' if body is None else 'POST')
        with urlopen(request, timeout=10) as response:
            return (response.read(), response.headers) if raw else json.load(response)

    def test_http_frame_preview_heightmap_and_material_history_use_same_rotation(self):
        angles = {'yaw': 90}
        query = dict(frame=3, orientation=json.dumps(angles))
        original = self.request('/api/frame', query=dict(frame=3))
        result = self.request('/api/frame', query=query)
        expected = np.roll(np.asarray(original['elevation']).reshape(4, 8), 2, axis=1)
        np.testing.assert_array_equal(np.asarray(result['elevation']).reshape(4, 8), expected)
        self.assertEqual(result['orientation'], normalize_orientation(angles))
        for route, colored in [('/api/preview', True), ('/api/heightmap', False)]:
            payload, headers = self.request(route, query=query, raw=True)
            self.assertEqual(headers['Content-Type'], 'image/png')
            actual = np.asarray(Image.open(io.BytesIO(payload)))
            wanted = np.asarray(Image.open(io.BytesIO(png_bytes(result, colored))))
            np.testing.assert_array_equal(actual, wanted)
        destination = self.marker_cells[3]+2
        history = self.request('/api/history', query={**query, 'run_id': 'tracked', 'cell': destination})
        self.assertEqual(history['trace_id'], 101)
        self.assertEqual(history['selection']['cell'], destination)
        self.assertEqual(history['selection']['source_cell'], self.marker_cells[3])
        self.assertEqual(history['points'][-1]['elevation_m'], result['elevation'][destination])
        baseline = self.request('/api/record')
        rotated = self.request('/api/record', query=query)
        self.assertEqual(rotated, orient_metadata(baseline, angles))
        self.assertEqual(self.request('/api/frame', query=dict(frame=3)), original)

    def test_arbitrary_rotation_selection_coordinates_match_clicked_destination_cell(self):
        destination = 20
        result = self.request('/api/history', query=dict(frame=3, cell=destination,
                                                       orientation=json.dumps(dict(yaw=37, pitch=63, roll=-21))))
        self.assertEqual(result['selection']['cell'], destination)
        np.testing.assert_allclose([result['selection']['lon'], result['selection']['lat']],
                                   self.coordinates(destination), atol=1e-10)

    def test_malformed_orientation_is_rejected_across_review_and_initial_routes(self):
        malformed = ('not-json', '[]', 'true', '3', '{"pitch":"90"}', '{"yaw":true}',
                     '{"roll":NaN}', '{"yaw":1e309}', '{"heading":20}')
        for value in malformed:
            for route in ('/api/frame', '/api/preview', '/api/history', '/api/record', '/api/export'):
                with self.subTest(route=route, value=value), self.assertRaises(HTTPError) as caught:
                    self.request(route, query=dict(frame=3, cell=20, orientation=value))
                self.assertEqual(caught.exception.code, 400)
                self.assertTrue(json.load(caught.exception)['error'])
        for initial in (None, [], {}, {'width': 48, 'height': 24, 'crust': [0]}):
            with self.subTest(initial=initial), self.assertRaises(HTTPError) as caught:
                self.request('/api/orient-initial', dict(initial=initial, orientation={'yaw': 90}))
            self.assertEqual(caught.exception.code, 400)

    def test_initial_post_bakes_rotation_without_mutating_the_saved_world(self):
        saved = self.request('/api/initial')
        crust = np.zeros((24, 48), dtype=np.uint8)
        crust[5:18, 7:20] = 1
        crust[9:13, 11:15] = 2
        initial = dict(width=48, height=24, crust=crust.ravel().tolist())
        rotated = self.request('/api/orient-initial', dict(initial=initial, orientation={'yaw': 450}))
        np.testing.assert_array_equal(np.asarray(rotated['crust']).reshape(24, 48), np.roll(crust, 12, axis=1))
        identity = self.request('/api/orient-initial', dict(initial=initial, orientation=None))
        self.assertEqual(identity, initial)
        self.assertEqual(self.request('/api/initial'), saved)
        with self.assertRaises(HTTPError) as caught:
            self.request('/api/orient-initial', dict(initial=initial, orientation={'yaw': False}))
        self.assertEqual(caught.exception.code, 400)

    def test_jobs_freeze_orientation_and_status_elapsed_advances_without_progress_callbacks(self):
        original_frame = self.snapshot(3, True)
        for kind, module, manager, route, extra in (
                ('terrain', terrain_jobs, self.terrain, '/api/terrain', dict(frame=3, width=2048)),
                ('gospl', gospl_jobs, self.gospl, '/api/gospl/start', dict(start_index=0, end_index=3, subdivisions=6))):
            with self.subTest(kind=kind):
                self.manager.load('tracked')
                angles = dict(yaw=450, pitch=31, roll=-20)
                expected = normalize_orientation(angles)
                builder = GatedBuilder(kind, self.clock)
                self.builders.append(builder)
                manager.builder = builder
                with patch.object(module, 'ProgressTiming', side_effect=lambda: ProgressTiming(clock=self.clock)):
                    job = self.request(route, dict(run_id='tracked', orientation=angles, **extra))
                    self.assertTrue(builder.entered.wait(5))
                    query = {'job_id': job['job_id']}
                    status_route = f'/api/{kind}/status'
                    first = self.request(status_route, query=query)
                    self.assertEqual(first['orientation'], expected)
                    self.assertEqual(first['elapsed_seconds'], 6.)
                    self.assertIsNotNone(first['eta_seconds'])
                    self.clock.advance(3)
                    later = self.request(status_route, query=query)
                    self.assertEqual(later['elapsed_seconds'], 9.)
                    self.assertGreater(later['eta_seconds'], first['eta_seconds'])
                    listed = self.request(f'/api/{kind}/list')['jobs']
                    self.assertEqual(next(item for item in listed if item['job_id'] == job['job_id'])['elapsed_seconds'], 9.)
                    angles['yaw'] = 0
                    self.request('/api/frame', query=dict(frame=3, orientation=json.dumps({'pitch': 90})))
                    self.write_run('different', tracked=True)
                    self.manager.load('different')
                    builder.release.set()
                    manager.worker.join(10)
                final = self.request(status_route, query=query)
                self.assertEqual(final['state'], 'complete', final)
                self.assertEqual(final['orientation'], expected)
                self.assertEqual(builder.options['orientation'], expected)
                self.assertEqual(builder.source['run_id'], 'tracked')
                self.assertEqual(final['elapsed_seconds'], 9.)
                self.assertEqual(final['eta_seconds'], 0.)
                self.clock.advance(50)
                self.assertEqual(self.request(status_route, query=query)['elapsed_seconds'], 9.)
                if kind == 'terrain':
                    np.testing.assert_array_equal(builder.source['elevation'], original_frame['elevation'])
                    with zipfile.ZipFile(manager.download(job['job_id'], 'zip')) as archive:
                        self.assertEqual(json.loads(archive.read('terrain_metadata.json'))['orientation'], expected)
                        self.assertIn('orientation.py', archive.namelist())

    def test_recovered_interrupted_jobs_clear_the_old_eta(self):
        for kind, manager, cls in (('terrain', self.terrain, terrain_jobs.TerrainManager),
                                   ('gospl', self.gospl, gospl_jobs.GosplManager)):
            with self.subTest(kind=kind):
                path = manager.path('interrupted')
                path.mkdir()
                (path/'job.json').write_text(json.dumps(dict(job_id='interrupted', state='running',
                    created='2099-01-01T00:00:00Z', progress=.5, elapsed_seconds=7., eta_seconds=42.)))
                restored = cls(manager.root)
                status = restored.status()
                self.assertIn(status['state'], ('cancelled', 'error'))
                self.assertIsNone(status['eta_seconds'])
                self.assertEqual(status['elapsed_seconds'], 7.)


if __name__ == '__main__':
    unittest.main()
