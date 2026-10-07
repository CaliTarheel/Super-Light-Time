"""Background goSPL lifecycle and real HTTP routes, with tiny gated builders."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from gospl_jobs import GosplManager
from server import Handler, SimulationManager


class TinyBuilder:
    """Exercise manager publication without allocating a production mesh."""
    def __init__(self, block_before=False, block_after=False):
        self.entered = threading.Event()
        self.published = threading.Event()
        self.release_before = threading.Event()
        self.release_after = threading.Event()
        if not block_before: self.release_before.set()
        if not block_after: self.release_after.set()
        self.calls = []

    def release(self):
        self.release_before.set()
        self.release_after.set()

    def __call__(self, run_path, output, manifest, **options):
        self.entered.set()
        if not self.release_before.wait(5): raise RuntimeError('Test builder was not released')
        # Inspect only after the test has changed selected simulation state, so
        # this observes the actual frozen request, not an earlier copied result.
        details = dict(source_path=str(run_path), manifest=deepcopy(manifest),
                       options={key: value for key, value in options.items()
                                if key not in ('cancel', 'progress')})
        self.calls.append(details)
        options['progress'](.5, 'Small test package')
        output = Path(output)
        inputs = output/'input'
        inputs.mkdir(exist_ok=True)
        np.savez_compressed(inputs/'mesh.npz', fixture=np.arange(3))
        np.savez_compressed(inputs/'forcing_0000.npz', fixture=np.arange(2))
        with zipfile.ZipFile(output/'gospl-history.zip', 'w') as archive:
            archive.writestr('request.json', json.dumps(details))
        self.published.set()
        if not self.release_after.wait(5): raise RuntimeError('Test publishing gate was not released')
        # Deliberately return despite cancellation. The job manager must itself
        # prevent a late builder return from publishing cancelled work.
        return dict(intervals=[dict(relaxation_supported_fraction=.75)])


class GosplJobTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='test-gospl-jobs-', dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.simulation = SimulationManager(self.root/'runs')
        self.write_run('world-a', (0., 2., 4.))
        self.write_run('world-b', (20., 22.))
        self.simulation.load('world-a')
        self.jobs_root = self.root/'jobs'
        self.builders = []
        self.managers = []
        self.addCleanup(self.release_workers)

    def write_run(self, run_id, times):
        path = self.simulation.path(run_id)
        path.mkdir(exist_ok=True)
        manifest = dict(run_id=run_id, state='complete', created='2026-09-05T00:00:00Z',
                        config=dict(dt_myr=2., duration_myr=times[-1], erosion=1.),
                        frames=[dict(index=i, time_myr=time) for i, time in enumerate(times)],
                        frame_count=len(times), time_myr=times[-1], duration_myr=times[-1])
        (path/'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
        for i, time in enumerate(times):
            np.savez_compressed(path/f'frame_{i:04d}.npz', trace_erosion_m=np.array([time]))
        return path

    def manager(self, **builder_options):
        builder = TinyBuilder(**builder_options)
        jobs = GosplManager(self.jobs_root, builder=builder)
        self.builders.append(builder)
        self.managers.append(jobs)
        return jobs, builder

    def release_workers(self):
        for builder in self.builders: builder.release()
        for manager in self.managers:
            if manager.worker: manager.worker.join(5)
        if self.simulation.worker: self.simulation.worker.join(5)

    def finished(self, jobs, expected='complete'):
        self.assertIsNotNone(jobs.worker)
        jobs.worker.join(5)
        self.assertFalse(jobs.worker.is_alive(), 'goSPL worker did not terminate')
        status = jobs.status()
        self.assertEqual(status['state'], expected, status)
        return status

    def test_request_keeps_source_and_frame_bounds_across_load_and_new_run(self):
        jobs, builder = self.manager(block_before=True)
        job = jobs.start(self.simulation, run_id='world-a', start_index=1, subdivisions=6)
        self.assertTrue(builder.entered.wait(2))
        # Existing integration may append frames after the export was accepted.
        with self.simulation.lock:
            self.simulation.current['frames'].append(dict(index=3, time_myr=6.))
            self.simulation.current['frame_count'] = 4
            self.simulation.current['config']['erosion'] = 99.
        self.simulation.load('world-b')
        # Exercise the real new-run state transition while omitting its physics.
        with patch.object(self.simulation, '_run', return_value=None):
            started = self.simulation.start(dict(width=48, height=24, duration_myr=2., dt_myr=2.), None)
            self.simulation.worker.join(2)
        self.assertNotEqual(started['run_id'], 'world-a')
        builder.release_before.set()
        completed = self.finished(jobs)
        request = builder.calls[0]
        self.assertEqual(request['source_path'], str(self.simulation.path('world-a')))
        self.assertEqual(request['manifest']['run_id'], 'world-a')
        self.assertEqual(request['manifest']['frame_count'], 3)
        self.assertEqual(len(request['manifest']['frames']), 3)
        self.assertEqual(request['manifest']['config']['erosion'], 1.)
        self.assertEqual(request['options']['start_index'], 1)
        self.assertEqual(request['options']['end_index'], 2)
        self.assertEqual(completed['start_myr'], 2.)
        self.assertEqual(completed['end_myr'], 4.)
        self.assertEqual(completed['run_id'], 'world-a')
        self.assertEqual(completed['job_id'], job['job_id'])
        self.assertEqual(self.simulation.status()['run_id'], started['run_id'])

    def test_completed_jobs_reopen_and_download_independently_of_selected_world(self):
        jobs, _ = self.manager()
        job = jobs.start(self.simulation, run_id='world-a', subdivisions=6)
        completed = self.finished(jobs)
        original = jobs.download(job['job_id']).read_bytes()
        self.simulation.load('world-b')
        reopened = GosplManager(self.jobs_root)
        self.assertEqual(reopened.status()['job_id'], job['job_id'])
        self.assertEqual(reopened.status()['state'], 'complete')
        self.assertEqual(reopened.download(job['job_id']).read_bytes(), original)
        self.assertEqual(reopened.list_jobs()[0]['bytes'], completed['bytes'])
        self.assertEqual(reopened.status()['correction_min_supported_fraction'], .75)
        with zipfile.ZipFile(io.BytesIO(original)) as archive:
            self.assertEqual(json.loads(archive.read('request.json'))['manifest']['run_id'], 'world-a')
        # Callers receive detached snapshots and cannot edit durable job status.
        detached = reopened.status()
        detached['state'] = 'running'
        self.assertEqual(reopened.status()['state'], 'complete')

    def test_interrupted_running_job_recovers_as_error_but_completed_job_survives(self):
        jobs, _ = self.manager()
        done = jobs.start(self.simulation, subdivisions=6)
        self.finished(jobs)
        lost = jobs.path('lost-worker')
        lost.mkdir()
        (lost/'job.json').write_text(json.dumps(dict(job_id='lost-worker', state='running',
                                                   progress=.4, created='2099-01-01T00:00:00Z')))
        reopened = GosplManager(self.jobs_root)
        status = reopened.status('lost-worker')
        self.assertEqual(status['state'], 'error')
        self.assertIn('closed before', status['error'])
        self.assertEqual(json.loads((lost/'job.json').read_text())['state'], 'error')
        self.assertTrue(reopened.download(done['job_id']).is_file())
        with self.assertRaisesRegex(ValueError, 'not complete'):
            reopened.download('lost-worker')

    def test_cancel_at_publication_wins_and_only_removes_its_generated_products(self):
        jobs, builder = self.manager(block_after=True)
        outside = self.jobs_root/'keep-me.npz'
        outside.write_bytes(b'unrelated data')
        job = jobs.start(self.simulation, subdivisions=6)
        self.assertTrue(builder.published.wait(2))
        job_path = jobs.path(job['job_id'])
        self.assertTrue((job_path/'gospl-history.zip').exists())
        jobs.cancel(job['job_id'])
        builder.release_after.set()
        self.finished(jobs, 'cancelled')
        for name in ('gospl-history.zip', 'gospl-history.zip.tmp', 'input/mesh.npz', 'input/forcing_0000.npz'):
            self.assertFalse((job_path/name).exists(), name)
        self.assertEqual(outside.read_bytes(), b'unrelated data')
        self.assertEqual(GosplManager(self.jobs_root).status()['state'], 'cancelled')
        with self.assertRaisesRegex(ValueError, 'not complete'):
            jobs.download(job['job_id'])

    def test_stale_requests_and_invalid_paths_cannot_cancel_or_download_another_job(self):
        jobs, builder = self.manager(block_before=True)
        with self.assertRaisesRegex(ValueError, 'selected experiment changed'):
            jobs.start(self.simulation, run_id='world-b', subdivisions=6)
        self.assertEqual(jobs.list_jobs(), [])
        job = jobs.start(self.simulation, run_id='world-a', subdivisions=6)
        with self.assertRaisesRegex(ValueError, 'already being built'):
            jobs.start(self.simulation, subdivisions=6)
        for bad in ('../outside', '..\\outside', 'C:\\outside', '/absolute', '', 17):
            with self.subTest(identifier=bad):
                with self.assertRaises(ValueError): jobs.status(bad)
                with self.assertRaises(ValueError): jobs.download(bad)
                with self.assertRaises(ValueError): jobs.cancel(bad)
        with self.assertRaisesRegex(ValueError, 'not found'): jobs.status('unknown-job')
        with self.assertRaisesRegex(ValueError, 'not the current'): jobs.cancel('old-job')
        self.assertFalse(jobs.stop.is_set())
        self.assertEqual(jobs.status()['job_id'], job['job_id'])
        builder.release_before.set()
        self.finished(jobs)

    def test_http_start_status_list_download_and_cancel_use_isolated_managers(self):
        jobs, builder = self.manager(block_before=True)
        simulation = self.simulation
        class LocalHandler(Handler):
            manager = simulation
            gospl_manager = jobs
            def log_message(self, *args): pass
        http = ThreadingHTTPServer(('127.0.0.1', 0), LocalHandler)
        thread = threading.Thread(target=lambda: http.serve_forever(poll_interval=.02), daemon=True)
        thread.start()
        self.addCleanup(http.server_close)
        self.addCleanup(http.shutdown)
        base = f'http://127.0.0.1:{http.server_port}'
        def request(route, body=None, raw=False):
            req = Request(base+route, data=None if body is None else json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'},
                          method='GET' if body is None else 'POST')
            with urlopen(req, timeout=3) as response:
                return (response.read(), response.headers) if raw else json.load(response)
        with self.assertRaises(HTTPError) as stale:
            request('/api/gospl/start', dict(run_id='world-b', subdivisions=6))
        self.assertEqual(stale.exception.code, 400)
        job = request('/api/gospl/start', dict(run_id='world-a', start_index=0, end_index=2,
                                              subdivisions=6, dt_years=50_000, rainfall_m_yr=1.25))
        query = '?'+urlencode({'job_id': job['job_id']})
        self.assertEqual(request('/api/gospl/status'+query)['state'], 'running')
        self.assertEqual(request('/api/gospl/list')['jobs'][0]['job_id'], job['job_id'])
        with self.assertRaises(HTTPError) as incomplete:
            request('/api/gospl/download'+query)
        self.assertEqual(incomplete.exception.code, 400)
        builder.release_before.set()
        self.finished(jobs)
        self.simulation.load('world-b')
        payload, headers = request('/api/gospl/download'+query, raw=True)
        self.assertEqual(headers['Content-Type'], 'application/zip')
        self.assertIn('gospl-history.zip', headers['Content-Disposition'])
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertIsNone(archive.testzip())
            saved = json.loads(archive.read('request.json'))
            self.assertEqual(saved['manifest']['run_id'], 'world-a')
            self.assertEqual(saved['options']['rainfall_m_yr'], 1.25)
            self.assertEqual(saved['options']['dt_years'], 50_000)
        self.assertEqual(request('/api/gospl/status'+query)['state'], 'complete')
        # A second isolated job exercises HTTP cancellation after publication.
        second_builder = TinyBuilder(block_after=True)
        self.builders.append(second_builder)
        jobs.builder = second_builder
        second = request('/api/gospl/start', dict(run_id='world-b', subdivisions=6))
        self.assertTrue(second_builder.published.wait(2))
        with self.assertRaises(HTTPError) as stale_cancel:
            request('/api/gospl/cancel', dict(job_id=job['job_id']))
        self.assertEqual(stale_cancel.exception.code, 400)
        request('/api/gospl/cancel', dict(job_id=second['job_id']))
        second_builder.release_after.set()
        self.finished(jobs, 'cancelled')
        status = request('/api/gospl/status?'+urlencode({'job_id': second['job_id']}))
        self.assertEqual(status['state'], 'cancelled')
        self.assertEqual(len(request('/api/gospl/list')['jobs']), 2)
        # Historical completed jobs remain downloadable after another job ends.
        self.assertEqual(request('/api/gospl/download'+query, raw=True)[0], payload)
        for route in ('status', 'download'):
            with self.subTest(route=route), self.assertRaises(HTTPError) as unsafe:
                request('/api/gospl/'+route+'?'+urlencode({'job_id': '../outside'}))
            self.assertEqual(unsafe.exception.code, 400)


if __name__ == '__main__':
    unittest.main()
