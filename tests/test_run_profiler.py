import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_profiler import RunProfiler, observe


class ProfilerTests(unittest.TestCase):
    def test_samples_target_and_stops_without_changing_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'physical failure'):
                with RunProfiler(directory, .01) as profiler:
                    self.assertTrue(profiler.ready.wait(2))
                    raise RuntimeError('physical failure')
            self.assertFalse(profiler.thread.is_alive())
            record = json.loads(next(Path(directory).glob('*.json')).read_text())
            self.assertTrue(any('test_samples_target' in item for item in record['stack']))
            self.assertEqual(record['metric'], 'simulation-thread wall-stack samples')
            self.assertIsNone(profiler.error)

    def test_io_failure_does_not_stop_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory)/'file'
            invalid.write_text('not a directory')
            with RunProfiler(invalid, .01) as profiler:
                self.assertTrue(profiler.ready.wait(2))
                result = 42
            self.assertEqual(result, 42)
            self.assertIn('FileExistsError', profiler.error)

    def test_retry_observation_only_copies_allowlisted_scalars(self):
        namespace = dict(observe=observe, sys=sys)
        exec(compile('def attempt():\n'
                     '    interval = .03125\n'
                     '    depth = 6\n'
                     '    old_time = 204.71875\n'
                     '    trials = 17\n'
                     '    checkpoint = object()\n'
                     '    diagnostics = dict(accepted_substeps=5, rejected_trials=11, last_rejection="bound failed")\n'
                     '    return observe(sys._getframe())\n', 'adaptive_timestepping.py', 'exec'), namespace)
        record = namespace['attempt']()
        row = record['adaptive_trials'][0]
        self.assertEqual(row['old_time'], 204.71875)
        self.assertEqual(row['rejected_trials'], 11)
        self.assertEqual(row['last_rejection'], 'bound failed')
        self.assertNotIn('checkpoint', row)

    def test_resume_preserves_previous_session(self):
        with tempfile.TemporaryDirectory() as directory:
            for _ in range(2):
                with RunProfiler(directory, .01) as profiler:
                    self.assertTrue(profiler.ready.wait(2))
            self.assertEqual(len(list(Path(directory).glob('*.json'))), 2)

    def test_server_profiles_resume_outside_engine_checkpoint_closure(self):
        import server
        from contextlib import nullcontext
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            manager = SimpleNamespace(path=lambda: Path(directory),
                parallel=SimpleNamespace(activate=nullcontext),
                _run_owned=lambda config, initial, resuming: (config, initial, resuming))
            with patch.object(server, 'RunProfiler', wraps=RunProfiler) as sampler:
                self.assertEqual(server.SimulationManager._run(manager, {'seed': 37}, None, True),
                                 ({'seed': 37}, None, True))
                sampler.assert_called_once_with(Path(directory)/'profiling')
        self.assertNotIn('run_profiler.py', server.AUXILIARY_SOURCES)


if __name__ == '__main__':
    unittest.main()
