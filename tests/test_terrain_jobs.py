"""Terrain jobs keep a fixed source epoch and expose actual native pixels."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import zipfile
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from terrain import TerrainCancelled, color_tile
from terrain_jobs import TerrainManager
from server import Handler


class SourceHistory:
    def __init__(self, root):
        self.root = root
        self.active = 'world-a'
        for run_id, time_myr in [('world-a', 1000), ('world-b', 20)]:
            path = root / run_id
            path.mkdir()
            frame = {'width': 8, 'height': 4, 'time_myr': time_myr}
            (path / 'frame_0000.json').write_text(json.dumps(frame))
            np.savez(path / 'frame_0000.npz', elevation=np.arange(32, dtype=np.float32))
            (path / 'config.json').write_text('{"seed":12}')
            (path / 'engine.py').write_text('# preserved source')

    def _history_context(self, run_id=None):
        run_id = run_id or self.active
        return self.root / run_id, {'run_id': run_id, 'frame_count': 1,
                                   'config': {'seed': 12}, 'engine_sha256': 'source-hash'}


def fake_builder(frame, path, width, seed, detail, progress, cancel):
    """Distinctive rows/columns reveal tile-coordinate and sampling mistakes."""
    if cancel():
        raise TerrainCancelled()
    values = np.broadcast_to(np.arange(width, dtype=np.float32), (width // 2, width)).copy()
    values += np.arange(width // 2, dtype=np.float32)[:, None] * 3
    np.save(path / 'elevation_m.npy', values)
    Image.fromarray((values + 12000).astype(np.uint16)).save(path / 'heightmap_16bit.png')
    color_tile(values, 0, 0, 128).save(path / 'preview.png')
    (path / 'terrain_metadata.json').write_text(json.dumps({'time_myr': frame['time_myr'],
                                                           'source_run_id': frame['run_id']}))
    progress(1.)


class TerrainJobTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / 'tmp'
        scratch.mkdir(exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.scratch.name)
        self.source = SourceHistory(self.root)
        self.jobs = TerrainManager(self.root / 'terrain', builder=fake_builder)

    def tearDown(self):
        self.jobs.cancel()
        if self.jobs.worker:
            self.jobs.worker.join(10)
        self.scratch.cleanup()

    def finish(self):
        self.jobs.worker.join(20)
        self.assertFalse(self.jobs.worker.is_alive())
        state = self.jobs.status()
        self.assertEqual(state['state'], 'complete', state)
        return state

    def test_fixed_epoch_native_tiles_downloads_and_restart(self):
        entered, release = threading.Event(), threading.Event()
        captured = []

        def held_builder(frame, path, **options):
            captured.append((frame['run_id'], frame['time_myr']))
            entered.set()
            if not release.wait(10):
                raise RuntimeError('Test source switch timed out')
            return fake_builder(frame, path, **options)

        self.jobs.builder = held_builder
        job = self.jobs.start(self.source, -1, width=2048)
        try:
            self.assertTrue(entered.wait(10))
            self.source.active = 'world-b'
            with self.assertRaises(ValueError):
                self.jobs.start(self.source, 0, width=2048)
        finally:
            release.set()
        state = self.finish()
        self.assertEqual(captured, [('world-a', 1000)])
        self.assertEqual(state['source_width'], 8)
        sample = self.jobs.sample(job['job_id'], 700, 600)
        self.assertEqual(sample['elevation_m'], 2500)
        self.assertEqual(sample['run_id'], 'world-a')
        self.assertEqual(sample['time_myr'], 1000)
        self.assertAlmostEqual(sample['lon'], -180 + 700.5 * 360 / 2048)
        self.assertAlmostEqual(sample['lat'], 90 - 600.5 * 180 / 1024)
        values = np.load(self.jobs.download(job['job_id'], 'npy'), mmap_mode='r')
        expected = np.asarray(color_tile(values, 512, 512, 512))
        actual = np.asarray(Image.open(io.BytesIO(self.jobs.tile(job['job_id'], 1, 1))))
        np.testing.assert_array_equal(actual, expected)
        del values
        with zipfile.ZipFile(self.jobs.download(job['job_id'], 'zip')) as archive:
            self.assertIsNone(archive.testzip())
            metadata = json.loads(archive.read('terrain_metadata.json'))
            self.assertEqual(metadata['source_run_id'], 'world-a')
            self.assertEqual(json.loads(archive.read('source/frame.json'))['time_myr'], 1000)
            self.assertEqual(hashlib.sha256(archive.read('terrain_builder.py')).hexdigest(),
                             metadata['terrain_builder_sha256'])
        self.assertEqual(TerrainManager(self.jobs.root).status()['job_id'], job['job_id'])
        self.jobs.builder = fake_builder
        newer = self.jobs.start(self.source, 0, width=2048)
        self.finish()
        self.assertNotEqual(newer['job_id'], job['job_id'])
        self.assertEqual(self.jobs.status(job['job_id'])['run_id'], 'world-a')
        self.assertEqual([item['job_id'] for item in self.jobs.list_jobs()], [newer['job_id'], job['job_id']])
        self.assertEqual(self.jobs.sample(job['job_id'], 700, 600), sample)
        for x, y in [(-1, 0), (2048, 0), (0, 1024), (.5, 0)]:
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                self.jobs.sample(job['job_id'], x, y)
        for x, y in [(-1, 0), (4, 0), (0, 2), (.5, 0)]:
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                self.jobs.tile(job['job_id'], x, y)

    def test_cancel_and_interrupted_restart_preserve_source(self):
        entered = threading.Event()

        def cancelled_builder(frame, path, **options):
            (path / 'elevation_m.npy').write_bytes(b'partial')
            entered.set()
            self.jobs.stop.wait(10)
            if options['cancel']():
                raise TerrainCancelled()
            raise RuntimeError('Test cancellation timed out')

        self.jobs.builder = cancelled_builder
        job = self.jobs.start(self.source, 0, width=2048)
        self.assertTrue(entered.wait(10))
        with self.assertRaises(ValueError):
            self.jobs.cancel('unrelated-job')
        self.jobs.cancel(job['job_id'])
        self.jobs.worker.join(10)
        self.assertEqual(self.jobs.status()['state'], 'cancelled')
        self.assertFalse((self.jobs.path(job['job_id']) / 'elevation_m.npy').exists())
        self.assertTrue((self.source.root / 'world-a' / 'frame_0000.npz').exists())
        with self.assertRaises(ValueError):
            self.jobs.download(job['job_id'], 'npy')
        persisted = self.jobs.path(job['job_id']) / 'job.json'
        value = json.loads(persisted.read_text())
        value['state'] = 'running'
        persisted.write_text(json.dumps(value))
        (persisted.parent / 'elevation_m.npy').write_bytes(b'interrupted')
        self.assertEqual(TerrainManager(self.jobs.root).status()['state'], 'cancelled')
        self.assertFalse((persisted.parent / 'elevation_m.npy').exists())

    def test_cancel_during_packaging_removes_large_partial_products(self):
        package = self.jobs.package

        def cancel_package(path, run_path, job, progress, cancel):
            def on_progress(value):
                progress(value)
                self.jobs.stop.set()
            return package(path, run_path, job, progress=on_progress, cancel=cancel)

        self.jobs.package = cancel_package
        job = self.jobs.start(self.source, 0, width=2048)
        self.jobs.worker.join(20)
        self.assertFalse(self.jobs.worker.is_alive())
        self.assertEqual(self.jobs.status()['state'], 'cancelled')
        for name in ('elevation_m.npy', 'heightmap_16bit.png', 'terrain.zip'):
            self.assertFalse((self.jobs.path(job['job_id']) / name).exists())
        self.assertTrue((self.jobs.path(job['job_id']) / 'job.json').exists())

    def test_invalid_requests_do_not_create_jobs(self):
        for width in [8193, True, 2048.5, float('nan')]:
            with self.subTest(width=width), self.assertRaises(ValueError):
                self.jobs.start(self.source, 0, width=width)
        for detail in [-1, 3, True, float('inf')]:
            with self.subTest(detail=detail), self.assertRaises(ValueError):
                self.jobs.start(self.source, 0, width=2048, detail=detail)
        for index in [1, -2, True, .5]:
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.jobs.start(self.source, index, width=2048)
        self.assertEqual(list(self.jobs.root.iterdir()), [])
        for job_id in ['../world-a', 'a/b', '', None]:
            with self.subTest(job_id=job_id), self.assertRaises(ValueError):
                self.jobs.path(job_id)

    def test_http_generation_history_picker_tiles_and_downloads(self):
        class TestHandler(Handler):
            manager = self.source
            terrain_manager = self.jobs

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), TestHandler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            request = Request(base + '/api/terrain', method='POST',
                              data=json.dumps({'frame': 0, 'run_id': 'world-a', 'width': 2048}).encode(),
                              headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=10) as response:
                job = json.load(response)
            self.finish()
            with urlopen(base + '/api/terrain/list', timeout=10) as response:
                self.assertEqual(json.load(response)['jobs'][0]['job_id'], job['job_id'])
            query = '?job_id=' + job['job_id']
            with urlopen(base + '/api/terrain/status' + query, timeout=10) as response:
                self.assertEqual(json.load(response)['state'], 'complete')
            with urlopen(base + '/api/terrain/sample' + query + '&x=700&y=600', timeout=10) as response:
                self.assertEqual(json.load(response)['elevation_m'], 2500)
            with urlopen(base + '/api/terrain/tile' + query + '&x=1&y=1', timeout=10) as response:
                self.assertEqual(response.headers['Content-Type'], 'image/png')
                self.assertEqual(Image.open(io.BytesIO(response.read())).size, (512, 512))
            with urlopen(base + '/api/terrain/download' + query + '&format=png', timeout=10) as response:
                image = Image.open(io.BytesIO(response.read()))
                self.assertEqual(image.size, (2048, 1024))
                self.assertEqual(image.getpixel((700, 600)) - 12000, 2500)
                self.assertIn('attachment', response.headers['Content-Disposition'])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(10)


if __name__ == '__main__':
    unittest.main()
