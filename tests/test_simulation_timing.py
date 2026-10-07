"""Simulation timing follows actual work boundaries without changing its clock."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import server
from progress_timing import ProgressTiming
from server import SimulationManager
from tectonics import Simulation


class Clock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class SimulationTimingTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.scratch.cleanup)
        self.clock = Clock()
        timing = patch.object(server, 'ProgressTiming',
                              side_effect=lambda **kwargs: ProgressTiming(clock=self.clock, **kwargs))
        timing.start()
        self.addCleanup(timing.stop)
        self.manager = SimulationManager(self.scratch.name)
        self.config = dict(width=48, height=24, mesh_level=2, mechanics_nodes=128,
                           seed=12, duration_myr=12,
                           dt_myr=2, snapshot_myr=12, plate_count=6)

    def join(self, manager=None):
        manager = manager or self.manager
        worker = manager.worker
        if worker:
            worker.join(40)
            self.assertFalse(worker.is_alive())
        self.assertNotEqual(manager.status()['state'], 'error', manager.status().get('error'))

    def test_initialization_and_final_files_count_as_live_processing(self):
        self.config.update(duration_myr=8, snapshot_myr=8)
        initializing, release_init = threading.Event(), threading.Event()
        saving, release_save = threading.Event(), threading.Event()
        actual_init, actual_step = Simulation.__init__, Simulation.step

        def initialize(simulation, *args, **kwargs):
            initializing.set()
            if not release_init.wait(20): raise RuntimeError('Initialization test barrier timed out')
            actual_init(simulation, *args, **kwargs)

        def step(simulation, dt):
            actual_step(simulation, dt)
            self.clock.advance(2)

        def save_final(frame):
            saving.set()
            if not release_save.wait(20): raise RuntimeError('Final output test barrier timed out')

        try:
            with patch.object(Simulation, '__init__', initialize), patch.object(Simulation, 'step', step), \
                    patch.object(self.manager, 'save_final', save_final):
                self.manager.start(self.config, None)
                self.assertTrue(initializing.wait(10))
                self.clock.advance(5)
                self.assertEqual(self.manager.status()['elapsed_seconds'], 5.)
                self.assertIsNone(self.manager.status()['eta_seconds'])
                release_init.set()
                self.assertTrue(saving.wait(20))
                status = self.manager.status()
                self.assertEqual(status['state'], 'running')
                self.assertEqual(status['progress'], 1.)
                self.assertEqual(status['elapsed_seconds'], 13.)
                self.assertIsNone(status['eta_seconds'])
                self.clock.advance(3)
                self.assertEqual(self.manager.status()['elapsed_seconds'], 16.)
                release_save.set()
                self.join()
        finally:
            release_init.set()
            release_save.set()
            self.join()
        status = self.manager.status()
        self.assertEqual(status['state'], 'complete')
        self.assertEqual(status['eta_seconds'], 0.)
        self.clock.advance(100)
        self.assertEqual(self.manager.status()['elapsed_seconds'], 16.)
        self.assertEqual(json.loads((self.manager.path()/'manifest.json').read_text())['elapsed_seconds'], 16.)

    def test_pause_serialization_and_resume_after_restart_keep_only_active_seconds(self):
        checkpointing, release_checkpoint = threading.Event(), threading.Event()
        actual_step, actual_write = Simulation.step, server.write_checkpoint

        def step(simulation, dt):
            actual_step(simulation, dt)
            self.clock.advance(2)
            if simulation.t >= 6: self.manager.pause()

        def checkpoint(*args, **kwargs):
            actual_write(*args, **kwargs)
            checkpointing.set()
            if not release_checkpoint.wait(20): raise RuntimeError('Checkpoint test barrier timed out')

        try:
            with patch.object(Simulation, 'step', step), patch.object(server, 'write_checkpoint', checkpoint):
                run_id = self.manager.start(self.config, None)['run_id']
                self.assertTrue(checkpointing.wait(20))
                self.assertEqual(self.manager.status()['state'], 'pausing')
                self.clock.advance(4)
                self.assertEqual(self.manager.status()['elapsed_seconds'], 10.)
                release_checkpoint.set()
                self.join()
        finally:
            release_checkpoint.set()
            self.join()
        self.assertEqual(self.manager.status()['state'], 'paused')
        self.assertIsNone(self.manager.status()['eta_seconds'])
        self.clock.advance(3600)
        restored = SimulationManager(self.scratch.name)
        self.assertEqual(restored.status()['elapsed_seconds'], 10.)
        self.assertIsNone(restored.status()['eta_seconds'])
        resumed, release_step = threading.Event(), threading.Event()

        def resume_step(simulation, dt):
            actual_step(simulation, dt)
            self.clock.advance(2)
            if simulation.t == 8:
                resumed.set()
                if not release_step.wait(20): raise RuntimeError('Resume test barrier timed out')

        try:
            with patch.object(Simulation, 'step', resume_step):
                restored.resume(run_id)
                self.assertTrue(resumed.wait(20))
                self.assertEqual(restored.status()['elapsed_seconds'], 12.)
                self.assertIsNone(restored.status()['eta_seconds'])
                release_step.set()
                self.join(restored)
        finally:
            release_step.set()
            self.join(restored)
        self.assertEqual(restored.status()['elapsed_seconds'], 16.)
        self.assertEqual(restored.status()['eta_seconds'], 0.)
        self.assertEqual([frame['time_myr'] for frame in restored.status()['frames']], [0, 6, 12])

    def test_eta_uses_integration_steps_even_when_no_new_frame_is_saved(self):
        working, release = threading.Event(), threading.Event()
        actual_step = Simulation.step

        def step(simulation, dt):
            if simulation.t == 6:
                working.set()
                if not release.wait(20): raise RuntimeError('Integration test barrier timed out')
            actual_step(simulation, dt)
            self.clock.advance(2)

        try:
            with patch.object(Simulation, 'step', step):
                self.manager.start(self.config, None)
                self.assertTrue(working.wait(20))
                status = self.manager.status()
                self.assertEqual(status['time_myr'], 0)
                self.assertEqual(status['integration_time_myr'], 6)
                self.assertEqual(status['elapsed_seconds'], 6.)
                self.assertEqual(status['eta_seconds'], 6.)
                self.clock.advance(3)
                self.assertEqual(self.manager.status()['elapsed_seconds'], 9.)
                self.assertEqual(self.manager.status()['eta_seconds'], 9.)
                self.manager.pause()
                release.set()
                self.join()
        finally:
            release.set()
            self.join()
        self.assertEqual(self.manager.status()['state'], 'paused')
        self.assertEqual(self.manager.status()['elapsed_seconds'], 11.)
        self.assertIsNone(self.manager.status()['eta_seconds'])
        self.clock.advance(500)
        self.assertEqual(self.manager.status()['elapsed_seconds'], 11.)


if __name__ == '__main__':
    unittest.main()
