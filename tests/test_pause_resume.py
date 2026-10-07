"""Exact continuation, durable pause commits, and explicit legacy behavior."""
import json
from pathlib import Path
import tempfile
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import unittest
from unittest.mock import patch

import numpy as np

import checkpoint
import server
from checkpoint import checkpoint_header, read_checkpoint, write_checkpoint
from server import SimulationManager, write_json
from tectonics import Simulation, validate_config


class PauseResumeTests(unittest.TestCase):
    def setUp(self):
        scratch=Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.manager=SimulationManager(self.temporary.name)
        self.addCleanup(self.stop_worker)
        self.config=dict(width=48,height=24,mesh_level=2,coast_geometry_level=2,mechanics_nodes=128,
                         adaptive_refinement=0,
                         seed=12,duration_myr=24,dt_myr=3,snapshot_myr=10,plate_count=6)

    def stop_worker(self):
        worker = self.manager.worker
        if worker and worker.is_alive():
            self.manager.stop.set()
            worker.join(timeout=120)

    def finished_worker(self, manager=None):
        manager=manager or self.manager
        worker=manager.worker
        if worker:
            worker.join(timeout=120)
            self.assertFalse(worker.is_alive(),'simulation worker did not finish')
        self.assertNotEqual(manager.status()['state'],'error',manager.status().get('error'))

    def pause_at(self, epoch=6):
        real_step=Simulation.step
        triggered=[]
        def step(sim,dt=None):
            real_step(sim,dt)
            if sim.t >= epoch and not triggered:
                triggered.append(True)
                self.manager.pause()
        with patch.object(Simulation,'step',step):
            result=self.manager.start(self.config,None)
            self.finished_worker()
        self.assertEqual(self.manager.status()['state'],'paused')
        self.assertTrue(self.manager.status()['can_resume'])
        return result['run_id']

    def assert_snapshot(self, actual, expected):
        for key,value in expected.items():
            if isinstance(value,np.ndarray):
                np.testing.assert_array_equal(actual[key],np.asarray(value,dtype=actual[key].dtype),err_msg=key)
            elif key not in ('index','pause_frame'):
                self.assertEqual(actual[key],value,key)

    def assert_state(self, actual, expected, path='state'):
        """Compare arbitrarily nested checkpoint state without losing arrays."""
        if isinstance(expected, np.ndarray):
            self.assertIsInstance(actual, np.ndarray, path)
            self.assertEqual(actual.dtype, expected.dtype, path)
            np.testing.assert_array_equal(actual, expected, err_msg=path)
        elif isinstance(expected, dict):
            self.assertIsInstance(actual, dict, path)
            self.assertEqual(set(actual), set(expected), path)
            for key, value in expected.items():
                self.assert_state(actual[key], value, f'{path}[{key!r}]')
        elif isinstance(expected, (list, tuple)):
            self.assertIsInstance(actual, type(expected), path)
            self.assertEqual(len(actual), len(expected), path)
            for index, value in enumerate(expected):
                self.assert_state(actual[index], value, f'{path}[{index}]')
        else:
            self.assertEqual(actual, expected, path)

    def test_exact_resume_after_restart_keeps_original_cadence_and_saved_frames(self):
        baseline=Simulation(self.config,server.native(server.make_initial(self.config)))
        expected=list(baseline.snapshots())
        run_id=self.pause_at(6)
        before={p.name:p.read_bytes() for p in self.manager.path().glob('frame_*')}
        paused=self.manager.status()
        self.assertEqual([f['time_myr'] for f in paused['frames']],[0,6])
        self.assertTrue(paused['frames'][-1]['pause_frame'])
        self.assertIsNone(self.manager.worker)  # no retained worker/simulation RAM
        restored=SimulationManager(self.temporary.name)
        self.assertEqual(restored.status()['run_id'],run_id)
        self.assertTrue(restored.status()['can_resume'])
        restored.resume(run_id)
        self.finished_worker(restored)
        status=restored.status()
        self.assertEqual(status['state'],'complete')
        self.assertEqual(status['run_id'],run_id)
        times=[f['time_myr'] for f in status['frames']]
        self.assertEqual(times,[0,6,10,20,24])
        self.assertEqual(len(times),len(set(times)))
        for name,payload in before.items():self.assertEqual((restored.path()/name).read_bytes(),payload)
        for snapshot in expected:
            index=times.index(snapshot['time_myr'])
            with np.load(restored.path()/f'frame_{index:04d}.npz',allow_pickle=False) as arrays:
                actual=dict(arrays)
            actual.update(json.loads((restored.path()/f'frame_{index:04d}.json').read_text()))
            self.assert_snapshot(actual,snapshot)

    def test_checkpoint_restores_all_state_rng_and_tracker_without_initialization(self):
        self.pause_at(6)
        simulation,manifest=read_checkpoint(self.manager.path()/'checkpoint.npz',self.manager.compatibility(),Simulation)
        baseline=Simulation(self.config,server.native(server.make_initial(self.config)))
        baseline.step(3);baseline.step(3)
        self.assertEqual(set(vars(simulation)),set(vars(baseline)))
        for name,value in vars(baseline).items():
            actual=getattr(simulation,name)
            if name=='rng':self.assert_state(actual.bit_generator.state,value.bit_generator.state,name)
            elif name=='domain_tracker' and value is not None:
                self.assert_state(vars(actual),vars(value),name)
            else:self.assert_state(actual,value,name)
        np.testing.assert_array_equal(simulation.rng.random(100),baseline.rng.random(100))
        self.assertEqual(manifest['time_myr'],6)

    def test_fractional_cadence_restores_exact_pending_output_not_recomputed_rounding(self):
        self.config.update(duration_myr=10,dt_myr=.5,snapshot_myr=1.3)
        expected=list(Simulation(self.config,server.native(server.make_initial(self.config))).snapshots())
        self.pause_at(6.5)
        restored=SimulationManager(self.temporary.name)
        restored.resume();self.finished_worker(restored)
        times=[f['time_myr'] for f in restored.status()['frames']]
        for snapshot in expected:
            index=times.index(snapshot['time_myr'])
            with np.load(restored.path()/f'frame_{index:04d}.npz',allow_pickle=False) as arrays:
                actual=dict(arrays)
            actual.update(json.loads((restored.path()/f'frame_{index:04d}.json').read_text()))
            self.assert_snapshot(actual,snapshot)

    def test_pause_during_initialization_is_durable_and_does_not_duplicate_zero_frame(self):
        entered,release=threading.Event(),threading.Event()
        real_init=Simulation.__init__
        def initialize(sim,*args,**kwargs):
            entered.set();release.wait(timeout=20)
            real_init(sim,*args,**kwargs)
        with patch.object(Simulation,'__init__',initialize):
            self.manager.start(self.config,None)
            self.assertTrue(entered.wait(10))
            self.assertEqual(self.manager.pause()['state'],'pausing')
            self.assertEqual(self.manager.pause()['state'],'pausing')
            release.set()
            self.finished_worker()
        self.assertEqual(self.manager.status()['state'],'paused')
        self.assertEqual(self.manager.status()['time_myr'],0)
        self.manager.resume()
        self.finished_worker()
        self.assertEqual([f['time_myr'] for f in self.manager.status()['frames']],[0,10,20,24])

    def test_checkpoint_commit_recovers_if_manifest_was_not_yet_replaced(self):
        self.pause_at(6)
        run_id=self.manager.status()['run_id']
        stale=self.manager.status()
        stale.update(state='pausing',frame_count=1,time_myr=0,frames=stale['frames'][:1],can_resume=False)
        write_json(self.manager.path()/'manifest.json',stale)
        recovered=SimulationManager(self.temporary.name)
        self.assertEqual(recovered.status()['run_id'],run_id)
        self.assertEqual(recovered.status()['state'],'paused')
        self.assertEqual(recovered.status()['time_myr'],6)
        self.assertEqual(recovered.status()['frame_count'],2)
        self.assertTrue(recovered.status()['can_resume'])

    def test_new_experiment_and_second_pause_preserve_the_original_checkpoint(self):
        first_run=self.pause_at(6)
        checkpoint_path=self.manager.path()/'checkpoint.npz'
        first_checkpoint=checkpoint_path.read_bytes()
        other=dict(self.config,duration_myr=3,snapshot_myr=3,seed=99)
        self.manager.start(other,None);self.finished_worker()
        self.assertEqual(checkpoint_path.read_bytes(),first_checkpoint)
        self.manager.load(first_run)
        real_step=Simulation.step
        def pause_again(sim,dt=None):
            real_step(sim,dt)
            if sim.t >= 13:self.manager.pause()
        with patch.object(Simulation,'step',pause_again):
            self.manager.resume(first_run);self.finished_worker()
        self.assertEqual(self.manager.status()['time_myr'],13)
        self.assertEqual([f['time_myr'] for f in self.manager.status()['frames']],[0,6,10,13])
        self.manager.resume(first_run);self.finished_worker()
        self.assertEqual([f['time_myr'] for f in self.manager.status()['frames']],[0,6,10,13,20,24])
        self.assertEqual(self.manager.status()['run_id'],first_run)

    def test_incompatible_sources_numpy_missing_checkpoint_and_legacy_fail_clearly(self):
        self.pause_at(6)
        for attribute,replacement,reason in (('ENGINE_SOURCE',b'# different engine','engine source'),
                                             ('AUXILIARY_SOURCES',{},'helper sources')):
            with patch.object(server,attribute,replacement):
                loaded=SimulationManager(self.temporary.name)
                self.assertFalse(loaded.status()['can_resume'])
                self.assertIn(reason,loaded.status()['resume_reason'])
                with self.assertRaisesRegex(ValueError,reason):loaded.resume()
        with patch.object(np,'__version__','incompatible'):
            loaded=SimulationManager(self.temporary.name)
            self.assertFalse(loaded.status()['can_resume'])
            with self.assertRaisesRegex(ValueError,'NumPy version'):loaded.resume()
        (self.manager.path()/'checkpoint.npz').unlink()
        # Periodic recovery now retains a second durable generation.
        checkpoint.previous_checkpoint(self.manager.path()/'checkpoint.npz').unlink(missing_ok=True)
        loaded=SimulationManager(self.temporary.name)
        self.assertFalse(loaded.status()['can_resume'])
        with self.assertRaisesRegex(ValueError,'checkpoint'):loaded.resume()
        loaded.current['state']='cancelled';loaded.persist()
        legacy=SimulationManager(self.temporary.name)
        with self.assertRaisesRegex(ValueError,'view-only'):legacy.resume()

    def test_cancel_is_terminal_even_when_requested_during_checkpoint_write(self):
        entered,release=threading.Event(),threading.Event()
        real_write=server.write_checkpoint
        def blocked(*args,**kwargs):
            real_write(*args,**kwargs)
            entered.set();release.wait(timeout=20)
        with patch.object(server,'write_checkpoint',blocked):
            self.manager.start(self.config,None)
            self.manager.pause()
            self.assertTrue(entered.wait(10))
            self.manager.cancel()
            # Simulate restarting after cancellation was acknowledged but before
            # the paused worker had committed its final manifest.
            self.assertEqual(json.loads((self.manager.path()/'manifest.json').read_text())['state'],'cancelled')
            recovered=SimulationManager(self.temporary.name)
            self.assertEqual(recovered.status()['state'],'cancelled')
            self.assertFalse(recovered.status()['can_resume'])
            release.set();self.finished_worker()
        self.assertEqual(self.manager.status()['state'],'cancelled')
        self.assertFalse(self.manager.status()['can_resume'])
        self.assertEqual(SimulationManager(self.temporary.name).status()['state'],'cancelled')
        with self.assertRaisesRegex(ValueError,'view-only'):self.manager.resume()

    def test_shutdown_requests_pause_and_completion_wins_at_terminal_time(self):
        real_step=Simulation.step
        def shutdown(sim,dt=None):
            real_step(sim,dt)
            self.manager.prepare_shutdown()
        with patch.object(Simulation,'step',shutdown):
            self.manager.start(self.config,None);self.finished_worker()
        self.assertEqual(self.manager.status()['state'],'paused')
        self.assertTrue(self.manager.status()['shutting_down'])
        with self.assertRaisesRegex(ValueError,'shutting down'):self.manager.resume()
        restored=SimulationManager(self.temporary.name)
        self.assertFalse(restored.status()['shutting_down'])
        self.assertTrue(restored.status()['can_resume'])
        restored.cancel()
        self.manager=SimulationManager(Path(self.temporary.name)/'terminal')
        self.config.update(duration_myr=3,snapshot_myr=3)
        with patch.object(Simulation,'step',shutdown):
            self.manager.start(self.config,None);self.finished_worker()
        self.assertEqual(self.manager.status()['state'],'complete')
        self.assertFalse(self.manager.status()['can_resume'])

    def test_atomic_checkpoint_failure_preserves_previous_commit_and_rejects_unknown_types(self):
        self.pause_at(6)
        path=self.manager.path()/'checkpoint.npz'
        before=path.read_bytes()
        simulation,manifest=read_checkpoint(path,self.manager.compatibility(),Simulation)
        with patch.object(np,'savez_compressed',side_effect=OSError('disk write failed')):
            with self.assertRaisesRegex(OSError,'disk write failed'):
                write_checkpoint(path,simulation,manifest,self.manager.compatibility())
        self.assertEqual(path.read_bytes(),before)
        self.assertFalse(list(path.parent.glob('checkpoint.npz.*.tmp')))
        with np.load(path,allow_pickle=False) as data:
            arrays={name:data[name] for name in data.files}
        metadata=json.loads(arrays[checkpoint.METADATA_KEY].tobytes())
        metadata['state']={'t':'arbitrary_python_object','v':'never execute this'}
        arrays[checkpoint.METADATA_KEY]=np.frombuffer(json.dumps(metadata).encode(),np.uint8)
        bad=path.parent/'invalid-checkpoint.npz';np.savez_compressed(bad,**arrays)
        with self.assertRaisesRegex(ValueError,'Unsupported checkpoint type'):
            read_checkpoint(bad,self.manager.compatibility(),Simulation)

    def test_http_pause_resume_and_initial_preset_contract(self):
        class LocalHandler(server.Handler):
            manager=self.manager
            def log_message(self,*args):pass
        http=ThreadingHTTPServer(('127.0.0.1',0),LocalHandler)
        thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
        self.addCleanup(http.server_close)
        self.addCleanup(http.shutdown)
        base=f'http://127.0.0.1:{http.server_port}'
        def post(route,body):
            request=Request(base+route,data=json.dumps(body).encode(),
                            headers={'Content-Type':'application/json'},method='POST')
            with urlopen(request,timeout=10) as response:return json.load(response)
        initial=post('/api/initial',{'config':self.config,'preset':'land65'})
        expected=server.native(server.make_initial(validate_config(self.config),preset='land65'))
        self.assertEqual(initial,expected)
        entered,release=threading.Event(),threading.Event()
        real_init=Simulation.__init__
        def initialize(sim,*args,**kwargs):
            entered.set();release.wait(timeout=20);real_init(sim,*args,**kwargs)
        try:
            with patch.object(Simulation,'__init__',initialize):
                run=post('/api/run',{'config':self.config})
                self.assertTrue(entered.wait(10))
                with self.assertRaises(HTTPError) as error:
                    post('/api/pause',{'run_id':'wrong-experiment'})
                self.assertEqual(error.exception.code,400)
                self.assertEqual(post('/api/pause',run)['state'],'pausing')
                release.set();self.finished_worker()
        finally:
            release.set()
        with urlopen(base+'/api/status',timeout=10) as response:state=json.load(response)
        self.assertEqual(state['state'],'paused');self.assertTrue(state['can_resume'])
        state=post('/api/resume',run)
        self.assertIn(state['state'],('resuming','running'))
        self.finished_worker()
        self.assertEqual(self.manager.status()['state'],'complete')
        self.assertEqual(post('/api/cancel',run),{'ok':True})


if __name__=='__main__':unittest.main()
