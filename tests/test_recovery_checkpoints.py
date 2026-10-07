"""Recovery rolls back an entire generation without losing abandoned evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import checkpoint
import server
from checkpoint import checkpoint_header, previous_checkpoint, read_checkpoint, write_checkpoint
from server import SimulationManager
from tectonics import Simulation


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.manager = SimulationManager(self.temporary.name, recovery_interval_myr=4)
        self.config = dict(width=48, height=24, mesh_level=2, coast_geometry_level=2,
                           mechanics_nodes=128, seed=12, duration_myr=12,
                           dt_myr=1, snapshot_myr=2, plate_count=4)

    def finish(self, manager=None):
        manager = manager or self.manager
        worker = manager.worker
        if worker:
            # Recovery fixtures can integrate twelve Myr on a shared host.
            # Keep the bound finite without treating two minutes as a defect.
            worker.join(timeout=900)
            self.assertFalse(worker.is_alive())
        return manager.status()

    def interrupt_at(self, epoch):
        step = Simulation.step
        def fail(simulation, dt=None):
            step(simulation, dt)
            if simulation.t >= epoch:
                raise RuntimeError('injected interrupted integration')
        with patch.object(Simulation, 'step', fail):
            self.manager.start(self.config, None)
            self.assertEqual(self.finish()['state'], 'error')

    def assert_history(self, manager, expected):
        times = [frame['time_myr'] for frame in manager.status()['frames']]
        self.assertEqual(times, [frame['time_myr'] for frame in expected])
        for index, snapshot in enumerate(expected):
            with np.load(manager.path()/f'frame_{index:04d}.npz', allow_pickle=False) as arrays:
                for name in arrays.files:
                    # Compare the public storage contract: review age/columns
                    # serialize to float32, native mechanical fields to float64.
                    np.testing.assert_array_equal(arrays[name], np.asarray(snapshot[name], dtype=arrays[name].dtype),
                                                  err_msg=f'frame {index} {name}')

    def test_error_recovery_is_exact_and_archives_later_output_before_reusing_indexes(self):
        expected = list(Simulation(self.config, server.native(server.make_initial(self.config))).snapshots())
        self.interrupt_at(7)
        root = self.manager.path()
        abandoned = {p.name: p.read_bytes() for p in root.glob('frame_0003.*')}
        with patch.object(Simulation, '__init__', side_effect=AssertionError('must restore state')):
            restored = SimulationManager(self.temporary.name)
            self.assertEqual(restored.status()['state'], 'interrupted')
            self.assertEqual(restored.status()['time_myr'], 4)
            self.assertEqual(restored.status()['recovery']['reverted_frame_count'], 1)
            self.assertTrue(restored.status()['can_resume'])
            restored.resume()
            self.assertEqual(self.finish(restored)['state'], 'complete')
        for name, payload in abandoned.items():
            copies = list((root/'recovery-abandoned').glob('*/'+name))
            self.assertEqual(len(copies), 1)
            self.assertEqual(copies[0].read_bytes(), payload)
        self.assert_history(restored, expected)

    def test_corrupt_latest_array_falls_back_even_when_its_header_is_readable(self):
        self.interrupt_at(11)
        root = self.manager.path()
        primary = root/'checkpoint.npz'
        self.assertEqual(checkpoint_header(primary)['time_myr'], 8)
        damaged = root/'damaged.npz'
        with zipfile.ZipFile(primary) as source, zipfile.ZipFile(damaged, 'w') as target:
            for index, name in enumerate(source.namelist()):
                target.writestr(name, b'damaged array' if index == 0 else source.read(name))
        damaged.replace(primary)
        self.assertEqual(checkpoint_header(primary)['time_myr'], 8)
        restored = SimulationManager(self.temporary.name)
        self.assertEqual(restored.status()['time_myr'], 4)
        self.assertEqual(restored.status()['recovery']['checkpoint_file'], 'checkpoint.previous.npz')
        restored.resume()
        self.assertEqual(self.finish(restored)['state'], 'complete')

    def test_recovery_between_outputs_preserves_fractional_scheduler_and_adds_no_frames(self):
        self.config.update(duration_myr=9.1, snapshot_myr=3.7, dt_myr=.6)
        self.manager.recovery_interval_myr = 2
        expected = list(Simulation(self.config, server.native(server.make_initial(self.config))).snapshots())
        self.interrupt_at(6.8)
        restored = SimulationManager(self.temporary.name)
        state = restored.status()
        self.assertGreater(state['time_myr'], state['frames'][-1]['time_myr'])
        restored.resume()
        self.assertEqual(self.finish(restored)['state'], 'complete')
        self.assert_history(restored, expected)

    def test_failed_atomic_replacement_keeps_latest_and_previous_complete(self):
        self.interrupt_at(7)
        primary = self.manager.path()/'checkpoint.npz'
        before = primary.read_bytes()
        simulation, manifest = read_checkpoint(primary, self.manager.compatibility(), Simulation)
        real_replace = Path.replace
        def fail_latest(source, destination):
            if destination == primary:
                raise OSError('injected checkpoint commit failure')
            return real_replace(source, destination)
        with patch.object(Path, 'replace', autospec=True, side_effect=fail_latest):
            with self.assertRaisesRegex(OSError, 'commit failure'):
                write_checkpoint(primary, simulation, manifest, self.manager.compatibility())
        self.assertEqual(primary.read_bytes(), before)
        self.assertEqual(previous_checkpoint(primary).read_bytes(), before)
        self.assertFalse(list(primary.parent.glob('checkpoint*.tmp')))
        self.assertEqual(read_checkpoint(primary, self.manager.compatibility(), Simulation)[0].t, 4)

    def test_committed_checkpoint_wins_when_manifest_write_was_interrupted(self):
        persist = self.manager.persist
        failed = []
        def interrupted_manifest():
            if self.manager.current.get('checkpoint_time_myr') == 4 and not failed:
                failed.append(True)
                raise OSError('injected manifest interruption')
            return persist()
        with patch.object(self.manager, 'persist', interrupted_manifest):
            self.manager.start(self.config, None)
            self.finish()
        restored = SimulationManager(self.temporary.name)
        self.assertEqual(restored.status()['time_myr'], 4)
        self.assertTrue(restored.status()['can_resume'])
        restored.resume()
        self.assertEqual(self.finish(restored)['state'], 'complete')

    def test_cancelled_run_never_recovers_a_periodic_generation(self):
        self.interrupt_at(7)
        self.manager.load(self.manager.current['run_id'])
        self.manager.cancel()
        restored = SimulationManager(self.temporary.name)
        self.assertEqual(restored.status()['state'], 'cancelled')
        self.assertFalse(restored.status()['can_resume'])


if __name__ == '__main__':
    unittest.main()
