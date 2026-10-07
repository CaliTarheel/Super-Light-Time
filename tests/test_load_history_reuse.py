"""One load request validates each checkpoint history once; resume revalidates.

``SimulationManager.load`` used to decode the selected checkpoint and read every
frame JSON twice: once while choosing the recovery generation and again while
computing ``can_resume``.  The first validation is now passed through that
single call.  These tests count real file reads, compare every outcome with the
fresh path, and prove the result never outlives its load call.

The "fresh" baseline (``fresh_path``) is this tree's ``_checkpoint_for_current``,
whose identity checks were moved into ``_check_current_matches``; it is not the
pristine code.  Equivalence with the pristine server itself was checked outside
the repository (25 scenarios, identical ``status()`` and selected checkpoint);
here the explicit ``can_resume``/``resume_reason``/generation assertions guard
against a regression shared by both paths.
"""
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import server
from checkpoint import previous_checkpoint, read_checkpoint, write_checkpoint
from server import SimulationManager, write_json
from tectonics import Simulation

CONFIG = dict(width=48, height=24, mesh_level=2, coast_geometry_level=2,
              mechanics_nodes=128, seed=12, duration_myr=12,
              dt_myr=1, snapshot_myr=2, plate_count=4)
FRAME_JSON = re.compile(r'frame_\d{4}\.json$')
REAL_SET_RESUME_STATUS = SimulationManager._set_resume_status


class Reads:
    """Counts frame JSON reads and checkpoint decodes made through server."""
    def __init__(self):
        self.frames, self.decoded, self.headers = [], [], []


@contextmanager
def counting():
    reads = Reads()
    read_text, decode, header = Path.read_text, server.read_checkpoint, server.checkpoint_header

    def counted_read_text(path, *args, **kwargs):
        if FRAME_JSON.search(str(path)):
            reads.frames.append(Path(path).name)
        return read_text(path, *args, **kwargs)

    def counted_decode(path, *args, **kwargs):
        reads.decoded.append(Path(path).name)
        return decode(path, *args, **kwargs)

    def counted_header(path, *args, **kwargs):
        reads.headers.append(Path(path).name)
        return header(path, *args, **kwargs)
    with patch.object(Path, 'read_text', counted_read_text), \
            patch.object(server, 'read_checkpoint', counted_decode), \
            patch.object(server, 'checkpoint_header', counted_header):
        yield reads


@contextmanager
def fresh_path():
    """The pre-change behaviour: every can_resume check rereads from disk."""
    def fresh(self, validated=None):
        return REAL_SET_RESUME_STATUS(self)
    with patch.object(SimulationManager, '_set_resume_status', fresh):
        yield


def comparable(manager):
    status = manager.status()
    status.pop('workers')
    return status, manager._resume_checkpoint_name


def damage(path):
    damaged = path.with_name('damaged.npz')
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(damaged, 'w') as target:
        for index, name in enumerate(source.namelist()):
            target.writestr(name, b'damaged array' if index == 0 else source.read(name))
    damaged.replace(path)


def rewrite_checkpoint(path, **changes):
    simulation, manifest = read_checkpoint(path, SimulationManager.compatibility(), Simulation)
    replacement = path.with_name('rewritten.npz')
    write_checkpoint(replacement, simulation, dict(manifest, **changes), SimulationManager.compatibility())
    os.replace(replacement, path)


def edit_json(path, **changes):
    value = json.loads(path.read_text(encoding='utf-8'))
    value.update(changes)
    write_json(path, value)


class LoadHistoryReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        manager = SimulationManager(cls.fixture.name, recovery_interval_myr=4)
        step = Simulation.step
        fired = []

        def pause(simulation, dt=None):
            step(simulation, dt)
            if simulation.t >= 6 and not fired:
                fired.append(True)
                manager.pause()
        with patch.object(Simulation, 'step', pause):
            cls.run_id = manager.start(CONFIG, None)['run_id']
            manager.worker.join(timeout=300)
        assert manager.status()['state'] == 'paused', manager.status()
        cls.frame_count = manager.status()['frame_count']
        cls.previous_frames = read_checkpoint(previous_checkpoint(manager.path()/'checkpoint.npz'),
                                              manager.compatibility(), Simulation)[1]['frame_count']
        assert (cls.frame_count, cls.previous_frames) == (4, 3), (cls.frame_count, cls.previous_frames)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def copy(self, run_id=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        shutil.copytree(Path(self.fixture.name)/self.run_id, root/(run_id or self.run_id))
        return root, root/(run_id or self.run_id)

    def open(self, root, fresh=False):
        with fresh_path() if fresh else nullcontext():
            manager = SimulationManager(root)
        self.addCleanup(self.stop, manager)
        return manager

    def stop(self, manager):
        if manager.worker and manager.worker.is_alive():
            manager.stop.set()
            manager.worker.join(timeout=120)

    def assert_same_as_fresh(self, root):
        candidate = self.open(root)
        baseline = self.open(root, fresh=True)
        self.assertEqual(comparable(candidate), comparable(baseline))
        return candidate.status()

    # Outcome equivalence and read counts ---------------------------------

    def test_valid_paused_load_reads_each_generation_once(self):
        root, _ = self.copy()
        with counting() as fresh:
            self.open(root, fresh=True)
        with counting() as reused:
            manager = self.open(root)
        both = self.frame_count + self.previous_frames
        self.assertEqual(len(fresh.frames), both + self.frame_count)
        self.assertEqual(fresh.decoded, ['checkpoint.npz', 'checkpoint.previous.npz', 'checkpoint.npz'])
        self.assertEqual(fresh.headers, ['checkpoint.npz'])
        self.assertEqual(len(reused.frames), both)
        self.assertEqual(reused.decoded, ['checkpoint.npz', 'checkpoint.previous.npz'])
        self.assertEqual(reused.headers, [])
        status = self.assert_same_as_fresh(root)
        self.assertEqual((status['state'], status['can_resume'], status['time_myr']), ('paused', True, 6))
        self.assertNotIn('resume_reason', status)
        self.assertNotIn('recovery', status)
        self.assertEqual(manager._resume_checkpoint_name, 'checkpoint.npz')

    def test_corrupt_primary_recovers_previous_generation(self):
        root, run = self.copy()
        damage(run/'checkpoint.npz')
        with counting() as reads:
            self.open(root)
        self.assertEqual(len(reads.frames), self.previous_frames)
        self.assertEqual(reads.headers, [])
        status = self.assert_same_as_fresh(root)
        self.assertEqual(status['recovery']['checkpoint_file'], 'checkpoint.previous.npz')
        self.assertEqual((status['state'], status['can_resume'], status['time_myr']), ('interrupted', True, 4))

    def test_both_generations_invalid(self):
        root, run = self.copy()
        damage(run/'checkpoint.npz')
        damage(previous_checkpoint(run/'checkpoint.npz'))
        status = self.assert_same_as_fresh(root)
        self.assertFalse(status['can_resume'])
        self.assertTrue(status['resume_reason'])

    def test_missing_frame_json_or_npz(self):
        for name, fallback in (('frame_0003.json', True), ('frame_0003.npz', True),
                               ('frame_0001.json', False), ('frame_0001.npz', False)):
            with self.subTest(name=name):
                root, run = self.copy()
                (run/name).unlink()
                status = self.assert_same_as_fresh(root)
                self.assertEqual(status['can_resume'], fallback)
                if fallback:
                    self.assertEqual(status['recovery']['checkpoint_file'], 'checkpoint.previous.npz')
                else:
                    self.assertIn('missing', status['resume_reason'])

    def test_wrong_frame_index_or_time(self):
        for name, change, fallback in (('frame_0003.json', dict(index=2), True),
                                       ('frame_0003.json', dict(time_myr=5.5), True),
                                       ('frame_0001.json', dict(index=7), False),
                                       ('frame_0001.json', dict(time_myr=1.), False)):
            with self.subTest(name=name, change=change):
                root, run = self.copy()
                edit_json(run/name, **change)
                status = self.assert_same_as_fresh(root)
                self.assertEqual(status['can_resume'], fallback)
                if not fallback:
                    self.assertIn('does not match its history index', status['resume_reason'])

    def test_changed_run_identity_or_configuration(self):
        # The checkpoints still name the original run.
        root, run = self.copy('20990101-000000-renamed')
        edit_json(run/'manifest.json', run_id=run.name)
        status = self.assert_same_as_fresh(root)
        self.assertFalse(status['can_resume'])
        root, run = self.copy()
        edit_json(run/'manifest.json', config=dict(CONFIG, seed=13))
        status = self.assert_same_as_fresh(root)
        self.assertFalse(status['can_resume'])

    def test_incompatible_sources_and_numpy(self):
        root, _ = self.copy()
        for target, attribute, replacement, reason in (
                (server, 'ENGINE_SOURCE', b'# different engine', 'engine source'),
                (server, 'AUXILIARY_SOURCES', {}, 'helper sources'),
                (np, '__version__', 'incompatible', 'NumPy version')):
            with self.subTest(attribute=attribute), patch.object(target, attribute, replacement):
                status = self.assert_same_as_fresh(root)
                self.assertFalse(status['can_resume'])
                self.assertIn(reason, status['resume_reason'])

    def test_invalid_next_output_scheduler(self):
        root, run = self.copy()
        rewrite_checkpoint(run/'checkpoint.npz', next_output_myr=6.)
        status = self.assert_same_as_fresh(root)
        self.assertEqual(status['recovery']['checkpoint_file'], 'checkpoint.previous.npz')
        rewrite_checkpoint(previous_checkpoint(run/'checkpoint.npz'), next_output_myr=None)
        status = self.assert_same_as_fresh(root)
        self.assertFalse(status['can_resume'])
        self.assertIn('scheduler', status['resume_reason'])

    def test_in_memory_identity_checks_still_run_on_reused_validation(self):
        # History validation accepts these generations; only the comparison
        # of the recovered manifest with the checkpoint can reject them.
        for changes, reason in ((dict(time_myr=5.5), 'does not match the latest saved history'),
                                (dict(engine_sha256='0'*64), 'engine source')):
            with self.subTest(changes=changes):
                root, run = self.copy()
                rewrite_checkpoint(run/'checkpoint.npz', **changes)
                with counting() as reads:
                    self.open(root)
                self.assertEqual(reads.headers, [])  # the reuse path was taken
                status = self.assert_same_as_fresh(root)
                self.assertFalse(status['can_resume'])
                self.assertIn(reason, status['resume_reason'])

    def test_interrupted_stale_and_terminal_manifests(self):
        for changes in (dict(state='running'), dict(state='resuming'), dict(state='error', error='x'),
                        dict(state='pausing', frame_count=1, time_myr=0),
                        dict(state='cancelled'), dict(state='complete')):
            with self.subTest(changes=changes):
                root, run = self.copy()
                edit_json(run/'manifest.json', **changes)
                status = self.assert_same_as_fresh(root)
                # Recovery turns error/running manifests into interrupted.
                self.assertEqual(status['can_resume'], changes['state'] not in ('cancelled', 'complete'))

    # Lifetime and identity of the passed result -------------------------

    def test_resume_revalidates_dependencies_replaced_after_load(self):
        run_frames = self.frame_count

        def replace_primary(run):
            shutil.copyfile(previous_checkpoint(run/'checkpoint.npz'), run/'checkpoint.npz')
        mutations = dict(
            frame_time=lambda run: edit_json(run/'frame_0002.json', time_myr=3.),
            frame_npz=lambda run: (run/'frame_0000.npz').unlink(),
            frame_json=lambda run: (run/'frame_0003.json').unlink(),
            damaged_checkpoint=lambda run: damage(run/'checkpoint.npz'),
            other_generation=replace_primary,
            scheduler=lambda run: rewrite_checkpoint(run/'checkpoint.npz', next_output_myr=None))
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                root, run = self.copy()
                manager = self.open(root)
                self.assertTrue(manager.status()['can_resume'])
                mutate(run)
                with patch.object(manager, '_run', side_effect=AssertionError('must not start')):
                    with self.assertRaises(ValueError):
                        manager.resume(self.run_id)
                self.assertIsNone(manager.worker)
                self.assertEqual(manager.status()['state'], 'paused')
        root, _ = self.copy()
        manager = self.open(root)
        with patch.object(manager, '_run', lambda *args: None), counting() as reads:
            manager.resume(self.run_id)
        manager.worker.join(10)
        self.assertEqual(reads.headers, ['checkpoint.npz'])
        self.assertEqual(reads.decoded, ['checkpoint.npz'])
        self.assertEqual(len(reads.frames), run_frames)

    def test_validation_is_never_retained_by_the_manager(self):
        root, run = self.copy()
        manager = self.open(root)

        def contains_validation(value):
            if isinstance(value, server._LoadValidation):
                return True
            if isinstance(value, dict):
                return any(map(contains_validation, value.values()))
            if isinstance(value, (list, tuple, set)):
                return any(map(contains_validation, value))
            return False
        self.assertFalse(contains_validation(vars(manager)))
        edit_json(run/'frame_0002.json', time_myr=3.)
        with counting() as reads:
            manager._set_resume_status()
        self.assertEqual(reads.headers, ['checkpoint.npz'])
        self.assertFalse(manager.status()['can_resume'])

    def test_validation_from_another_run_or_changed_file_is_not_applied(self):
        root, run = self.copy()
        other = root/'20990101-000000-other'
        shutil.copytree(run, other)
        edit_json(other/'manifest.json', run_id=other.name)
        manager = self.open(root)
        captured = []
        real = manager._recover_manifest_validated

        def capture(*args):
            result = real(*args)
            captured.append(result[1])
            return result
        with patch.object(manager, '_recover_manifest_validated', capture):
            manager.load(self.run_id)
        validated = captured[-1]
        self.assertIsInstance(validated, server._LoadValidation)
        # Selected run changed: A's record must not answer for B.
        manager.load(other.name)
        self.assertFalse(manager.status()['can_resume'])  # B's checkpoint names run A
        for checkpoint in ('checkpoint.npz', 'checkpoint.previous.npz'):
            rewrite_checkpoint(other/checkpoint, run_id=other.name)
        manager.load(other.name)
        self.assertTrue(manager.status()['can_resume'])
        edit_json(other/'frame_0003.json', time_myr=5.)
        with counting() as reads:
            manager._set_resume_status(validated)
        self.assertEqual(reads.headers, ['checkpoint.npz'])
        self.assertFalse(manager.status()['can_resume'])
        # Same run and file name, but the file was replaced after validation.
        manager.load(self.run_id)
        shutil.copyfile(run/'checkpoint.npz', run/'copy.npz')
        os.replace(run/'copy.npz', run/'checkpoint.npz')
        edit_json(run/'frame_0003.json', time_myr=5.)
        with counting() as reads:
            manager._set_resume_status(validated)
        self.assertEqual(reads.headers, ['checkpoint.npz'])
        self.assertFalse(manager.status()['can_resume'])

    def test_signature_is_taken_before_the_decode(self):
        """A replacement landing after the decode must force the fresh path."""
        root, run = self.copy()
        manager = self.open(root)
        decode, replaced = server.read_checkpoint, []

        def decode_then_replace(path, *args, **kwargs):
            result = decode(path, *args, **kwargs)
            if Path(path).name == 'checkpoint.npz' and not replaced:
                replaced.append(True)
                shutil.copyfile(run/'checkpoint.npz', run/'copy.npz')  # same bytes, new file
                os.replace(run/'copy.npz', run/'checkpoint.npz')
            return result
        with patch.object(server, 'read_checkpoint', decode_then_replace), counting() as reads:
            manager.load(self.run_id)
        self.assertEqual(replaced, [True])
        self.assertEqual(reads.headers, ['checkpoint.npz'])  # record rejected, files reread
        self.assertEqual(reads.decoded, ['checkpoint.npz', 'checkpoint.previous.npz', 'checkpoint.npz'])
        self.assertTrue(manager.status()['can_resume'])
        self.assertEqual(comparable(manager), comparable(self.open(root, fresh=True)))

    def test_failed_load_leaves_no_result_for_the_next_load(self):
        root, run = self.copy()
        invalid = root/'20990101-000000-invalid'
        shutil.copytree(run, invalid)
        edit_json(invalid/'manifest.json', run_id=invalid.name)
        broken = root/'20990101-000000-broken'
        broken.mkdir()
        (broken/'manifest.json').write_text('{not json', encoding='utf-8')
        manager = self.open(root)
        manager.load(self.run_id)
        self.assertTrue(manager.status()['can_resume'])
        with self.assertRaises(ValueError):
            manager.load(broken.name)
        with patch.object(manager, '_validate_checkpoint_history', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                manager.load(self.run_id)
        with counting() as reads:
            manager.load(invalid.name)
        self.assertFalse(manager.status()['can_resume'])
        self.assertEqual(reads.headers, ['checkpoint.npz'])  # fresh check, nothing reused
        baseline = self.open(root, fresh=True)
        with fresh_path():
            baseline.load(invalid.name)
        self.assertEqual(comparable(manager), comparable(baseline))

    def test_no_stale_result_after_generation_change_save_or_selection(self):
        root, run = self.copy()
        manager = self.open(root)
        self.assertEqual(manager._resume_checkpoint_name, 'checkpoint.npz')
        damage(run/'checkpoint.npz')
        manager.load(self.run_id)
        self.assertEqual(manager._resume_checkpoint_name, 'checkpoint.previous.npz')
        self.assertEqual(manager.status()['time_myr'], 4)
        self.assertEqual(comparable(manager), comparable(self.open(root, fresh=True)))
        # Save a new generation by resuming and pausing again at 8 Myr.
        root, run = self.copy()
        manager = self.open(root)
        step = Simulation.step

        def pause(simulation, dt=None):
            step(simulation, dt)
            if simulation.t >= 8:
                manager.pause()
        with patch.object(Simulation, 'step', pause):
            manager.resume(self.run_id)
            manager.worker.join(timeout=120)
        self.assertEqual((manager.status()['state'], manager.status()['time_myr']), ('paused', 8))
        with counting() as reads:
            manager.load(self.run_id)
        self.assertEqual(len(reads.frames), 5 + 4)  # new primary and rotated previous
        self.assertTrue(manager.status()['can_resume'])
        edit_json(run/'frame_0004.json', time_myr=7.)
        manager.load(self.run_id)
        self.assertEqual(manager.status()['time_myr'], 6)
        self.assertEqual(comparable(manager), comparable(self.open(root, fresh=True)))

    def test_resumed_fixture_continues_identically(self):
        expected = list(Simulation(CONFIG, server.native(server.make_initial(CONFIG))).snapshots())
        root, run = self.copy()
        manager = self.open(root)
        self.assertTrue(manager.status()['can_resume'])
        manager.resume(self.run_id)
        manager.worker.join(timeout=300)
        self.assertEqual(manager.status()['state'], 'complete', manager.status().get('error'))
        times = [frame['time_myr'] for frame in manager.status()['frames']]
        self.assertEqual(times, [snapshot['time_myr'] for snapshot in expected])
        for index, snapshot in enumerate(expected):
            with np.load(run/f'frame_{index:04d}.npz', allow_pickle=False) as arrays:
                for name in arrays.files:
                    np.testing.assert_array_equal(arrays[name], np.asarray(snapshot[name], dtype=arrays[name].dtype),
                                                  err_msg=f'frame {index} {name}')


if __name__ == '__main__':
    unittest.main()
