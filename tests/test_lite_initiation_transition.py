"""Physical-policy source boundaries preserve state/history and fail closed.

The portable fixture is a typed checkpoint with native-shaped arrays and a tiny
saved history. It exercises transition/storage contracts without evolving a
second tectonic world; initiation physics is tested in its producer suite.
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import checkpoint
import lite_initiation_transition as transition
import native_engine


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


class PolicyFixture:
    @staticmethod
    def normalize(raw=None):
        return dict(dict(enabled=True, compression_myr=10., shortening_km=100., slip_km=100.), **(raw or {}))

    @staticmethod
    def upgrade(state, parameters):
        config = deepcopy(state.config)
        config['effective_subduction']['initiation'] = deepcopy(parameters)
        state.config = config
        state.effective_subduction_initiation = dict(version=1, policy='future-only test fixture',
            parameters=deepcopy(parameters), activation_myr=state.t, epoch_myr=state.t,
            candidates=[], next_candidate_id=1, births=[], events=[])
        return dict(physical_time_advanced_myr=0., candidates=0, births=0)


class LiteInitiationTransitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = self.root / 'parent-lite'
        self.parent.mkdir()
        self.engine = b'import effective_subduction\nimport native_engine\n'
        self.old_effective = b'import unchanged\nVALUE = 1\n'
        self.new_effective = b'import unchanged\nimport effective_subduction_initiation\nVALUE = 2\n'
        self.unchanged = b'VALUE = 3\n'
        self.old_native, self.new_native = b'NATIVE_VERSION = 1\n', b'NATIVE_VERSION = 2\n'
        self.added = b'VERSION = 1\n'
        self.prior = dict(engine_sha256=sha(self.engine),
            auxiliary_sources_sha256={'effective_subduction.py': sha(self.old_effective), 'unchanged.py': sha(self.unchanged),
                'native_engine.py': sha(self.old_native)},
            numpy_version=np.__version__, flow_backend={'name': 'fixture', 'binary_sha256': 'flow1'},
            material_backend={'name': 'fixture', 'binary_sha256': 'material1'})
        self.helpers = {'effective_subduction.py': self.new_effective,
                        'unchanged.py': self.unchanged, 'effective_subduction_initiation.py': self.added,
                        'native_engine.py': self.new_native}
        self.current = dict(self.prior, auxiliary_sources_sha256={name: sha(value) for name, value in self.helpers.items()})
        for name, value in {'engine.py': self.engine, 'effective_subduction.py': self.old_effective,
                            'unchanged.py': self.unchanged, 'native_engine.py': self.old_native}.items():
            (self.parent / name).write_bytes(value)
        config = dict(seed=37, duration_myr=1000., snapshot_myr=2., dt_myr=1.,
            effective_subduction=dict(enabled=True, force_n_per_m=3e13))
        self.original = SimpleNamespace(t=25., steps=25, config=config, rng=np.random.default_rng(37),
            effective_subduction_version=1, effective_subduction_settings=deepcopy(config['effective_subduction']),
            mass=np.array([7., 11.]), omega=np.arange(6, dtype=float).reshape(2, 3),
            material_surface=dict(vertices=np.eye(3), faces=np.array([[0, 1, 2]], np.int32),
                reference_area_km2=np.array([18.]), area_km2=np.array([17.])),
            structure=dict(thickness_km=np.array([36.]), inventories=np.arange(6.).reshape(2, 3)),
            trench_systems=[dict(id=4, downgoing_plate_uid=2, overriding_plate_uid=7,
                effective_subduction={'version': 1, 'force_n_per_m': 3e13}, geometry_xyz=[[1., 0., 0.]])],
            events=[dict(type='old-event', time_myr=24.)], nested={4: (np.array([1, 2], np.int64), 'retained')})
        self.manifest = dict(run_id=self.parent.name, title='Asein Lite', state='paused', can_resume=True,
            config=deepcopy(config), frame_count=2, frames=[dict(index=0, time_myr=0.),
                dict(index=1, time_myr=25., pause_frame=True)], time_myr=25.,
            checkpoint_time_myr=25., integration_time_myr=25., next_output_myr=26.,
            duration_myr=1000., **deepcopy(self.prior))
        for row in self.manifest['frames']:
            stem = self.parent / f"frame_{row['index']:04d}"
            np.savez_compressed(stem.with_suffix('.npz'), material=np.arange(9).reshape(3, 3),
                                native_owner=np.array([2, 7]))
            stem.with_suffix('.json').write_text(json.dumps(row))
        (self.parent / 'initial.json').write_bytes(b'{"artwork":"unchanged-original"}\n')
        (self.parent / 'config.json').write_text(json.dumps(config))
        self.publish_parent()
        self.seal = self.root / 'parent-preservation.json'
        self.write_seal()
        self.patches = [patch.object(transition, 'REVIEW_READY', True),
            patch.object(transition, 'APPROVED_PARENT_RUN_ID', self.parent.name),
            patch.object(transition, 'APPROVED_PARENT_CHECKPOINT_SHA256', transition._file_sha(self.parent / 'checkpoint.npz')),
            patch.object(transition, 'PARENT_PRESERVATION_RECEIPT', self.seal),
            patch.object(transition, 'APPROVED_PARENT_PRESERVATION_SHA256', transition._file_sha(self.seal)),
            patch.object(transition, 'APPROVED_CHANGED_HELPERS', {'effective_subduction.py': (sha(self.old_effective), sha(self.new_effective)),
                'native_engine.py': (sha(self.old_native), sha(self.new_native))}),
            patch.object(transition, 'APPROVED_ADDED_HELPERS', {'effective_subduction_initiation.py': sha(self.added)}),
            patch.object(transition, '_current_context', side_effect=lambda: (deepcopy(self.current), self.engine, dict(self.helpers))),
            patch.object(transition, '_policy_api', return_value=PolicyFixture)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def publish_parent(self):
        checkpoint.write_checkpoint(self.parent / 'checkpoint.npz', self.original, self.manifest, self.prior)
        (self.parent / 'manifest.json').write_text(json.dumps(self.manifest))

    def approve_checkpoint(self):
        transition.APPROVED_PARENT_CHECKPOINT_SHA256 = transition._file_sha(self.parent / 'checkpoint.npz')
        self.write_seal()
        transition.APPROVED_PARENT_PRESERVATION_SHA256 = transition._file_sha(self.seal)

    def write_seal(self):
        self.seal.write_text(json.dumps(dict(run_id=self.parent.name,
            files=[dict(path=name, sha256=value) for name, value in transition._files(self.parent).items()])))

    def plan(self):
        return transition.plan_transition(self.parent, controls={'enabled': True})

    def test_exact_state_history_source_config_scheduler_and_permanent_boundary(self):
        before = transition._files(self.parent)
        with patch.object(native_engine.Simulation, '__init__', side_effect=AssertionError('constructor called')):
            plan = self.plan()
            child, receipt = transition.create_transition(plan, run_id='child')
            restored, saved = checkpoint.read_checkpoint(child / 'checkpoint.npz', self.current, native_engine.Simulation)
        self.assertEqual(transition._fingerprint(transition._physical_state(restored)),
                         transition._fingerprint(transition._physical_state(self.original)))
        self.assertEqual(transition._files(self.parent), before)
        self.assertEqual(transition._files(child / 'inherited-parent'), before)
        self.assertEqual(restored.effective_subduction_settings, self.original.effective_subduction_settings)
        self.assertEqual(restored.trench_systems, self.original.trench_systems)
        self.assertEqual(restored.rng.bit_generator.state, self.original.rng.bit_generator.state)
        self.assertEqual(saved['next_output_myr'], 26.)
        self.assertEqual(saved['frames'], self.manifest['frames'])
        self.assertEqual(saved['state'], 'paused')
        self.assertEqual(receipt['preservation']['preexisting_array_count'], 9)
        self.assertEqual((child / 'checkpoint.policy-boundary.npz').read_bytes(), (child / 'checkpoint.npz').read_bytes())
        for index in range(2):
            for suffix in ('.npz', '.json'):
                name = f'frame_{index:04d}{suffix}'
                self.assertEqual((child / name).read_bytes(), (self.parent / name).read_bytes())
        for frame in saved['frame_sources']:
            self.assertEqual(frame['auxiliary_sources_sha256'], self.prior['auxiliary_sources_sha256'])
            self.assertNotEqual(frame['auxiliary_sources_sha256'], saved['auxiliary_sources_sha256'])
        self.assertEqual((child / 'initial.json').read_bytes(), (self.parent / 'initial.json').read_bytes())
        self.assertEqual(restored.effective_subduction_initiation['epoch_myr'], 25.)
        self.assertEqual(restored.effective_subduction_initiation['candidates'], [])
        self.assertEqual(transition._file_sha(child / 'checkpoint.npz'), receipt['child_checkpoint_sha256'])
        self.assertEqual(json.loads((child / 'manifest.json').read_text()), saved)

    def test_readiness_blocks_all_planning_before_read_or_write(self):
        with patch.object(transition, 'REVIEW_READY', False), self.assertRaisesRegex(ValueError, 'not sealed'):
            self.plan()
        self.assertEqual(set(self.root.iterdir()), {self.parent, self.seal})

    def test_running_parent_and_mismatched_pause_manifest_are_rejected(self):
        original = (self.parent / 'manifest.json').read_bytes()
        for alteration in ({'state': 'running'}, {'time_myr': 24.}, {'frame_count': 1}):
            with self.subTest(alteration=alteration):
                (self.parent / 'manifest.json').write_text(json.dumps(dict(self.manifest, **alteration)))
                with self.assertRaises(ValueError):
                    self.plan()
                (self.parent / 'manifest.json').write_bytes(original)
        self.assertEqual(set(self.root.iterdir()), {self.parent, self.seal})

    def test_unapproved_changed_added_removed_sources_fail_closed(self):
        original_helpers, original_current = deepcopy(self.helpers), deepcopy(self.current)
        for mode in ('changed', 'added', 'removed'):
            self.helpers, self.current = deepcopy(original_helpers), deepcopy(original_current)
            if mode == 'changed':
                self.helpers['unchanged.py'] = b'VALUE = 4\n'
            elif mode == 'added':
                self.helpers['arbitrary.py'] = b'VALUE = 5\n'
            else:
                self.helpers.pop('unchanged.py')
            self.current['auxiliary_sources_sha256'] = {name: sha(value) for name, value in self.helpers.items()}
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'source delta'):
                self.plan()
        self.assertEqual(set(self.root.iterdir()), {self.parent, self.seal})

    def test_backend_and_numpy_changes_are_independently_rejected(self):
        baseline = deepcopy(self.current)
        for field in ('flow_backend', 'material_backend', 'numpy_version'):
            self.current = deepcopy(baseline)
            self.current[field] = 'different' if field == 'numpy_version' else {'binary_sha256': 'other'}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                self.plan()
        self.assertEqual(set(self.root.iterdir()), {self.parent, self.seal})

    def test_parent_source_damage_checkpoint_mutation_and_frame_damage_are_rejected(self):
        original = (self.parent / 'unchanged.py').read_bytes()
        (self.parent / 'unchanged.py').write_bytes(b'# altered\n')
        with self.assertRaisesRegex(ValueError, 'sealed parent files'):
            self.plan()
        (self.parent / 'unchanged.py').write_bytes(original)
        plan = self.plan()
        (self.parent / 'frame_0001.npz').write_bytes(b'# changed after plan')
        with self.assertRaisesRegex(ValueError, 'parent changed after planning'):
            transition.create_transition(plan, run_id='child')
        self.assertFalse((self.root / 'child').exists())
        self.original.mass[0] += 1.
        self.publish_parent()
        with self.assertRaisesRegex(ValueError, 'exact reviewed accepted-step boundary'):
            self.plan()

    def test_configuration_state_rng_trench_and_clock_mutations_cannot_hide_in_activation(self):
        base = PolicyFixture.upgrade
        changes = ('config', 'array', 'rng', 'trench', 'clock', 'birth', 'force_settings')
        for change in changes:
            def bad(state, controls):
                result = base(state, controls)
                if change == 'config': state.config['seed'] += 1
                if change == 'array': state.material_surface['vertices'][0, 0] += 1
                if change == 'rng': state.rng.random()
                if change == 'trench': state.trench_systems[0]['id'] += 1
                if change == 'clock': state.t += 1
                if change == 'birth': state.effective_subduction_initiation['births'].append({'id': 1})
                if change == 'force_settings': state.effective_subduction_settings['force_n_per_m'] *= 2
                return result
            with self.subTest(change=change), patch.object(PolicyFixture, 'upgrade', side_effect=bad), self.assertRaises(ValueError):
                self.plan()
        self.assertEqual(set(self.root.iterdir()), {self.parent, self.seal})

    def test_exact_scheduler_and_history_required(self):
        original = deepcopy(self.manifest)
        for pending in (None, 25., float('inf')):
            self.manifest = deepcopy(original)
            self.manifest['next_output_myr'] = pending
            if pending == float('inf'):
                # The writer correctly refuses nonfinite state; use a finite invalid endpoint.
                self.manifest['next_output_myr'] = 1001.
            self.publish_parent()
            self.approve_checkpoint()
            with self.subTest(pending=pending), self.assertRaisesRegex(ValueError, 'scheduler'):
                self.plan()
        self.manifest = original
        self.publish_parent()
        self.approve_checkpoint()
        frame = self.parent / 'frame_0001.json'
        frame.write_text(json.dumps(dict(index=1, time_myr=24.)))
        self.write_seal()
        transition.APPROVED_PARENT_PRESERVATION_SHA256 = transition._file_sha(self.seal)
        with self.assertRaisesRegex(ValueError, 'history index'):
            self.plan()

    def test_preplan_frame_corruption_and_preservation_receipt_tampering_are_rejected(self):
        frame = self.parent / 'frame_0001.npz'
        original = frame.read_bytes()
        frame.write_bytes(b'# substituted before planning')
        with self.assertRaisesRegex(ValueError, 'sealed parent files'):
            self.plan()
        frame.write_bytes(original)
        self.seal.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'preservation receipt'):
            self.plan()
        self.assertEqual(set(self.root.iterdir()), {self.parent, self.seal})

    def test_mutable_plan_cannot_forge_another_physical_state_or_policy(self):
        for target in ('array', 'births'):
            plan = self.plan()
            if target == 'array':
                plan.original.mass[0] += 1.
                plan.derived.mass[0] += 1.
            else:
                plan.derived.effective_subduction_initiation['births'].append({'id': 99})
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'modified after validation'):
                transition.create_transition(plan, run_id='child')
            self.assertFalse((self.root / 'child').exists())

    def test_readback_failure_never_publishes_child_manifest_and_parent_stays_exact(self):
        plan = self.plan()
        parent_files = transition._files(self.parent)
        real_read = checkpoint.read_checkpoint
        def fail_child(path, *args):
            if Path(path).parent.name == 'child':
                raise ValueError('injected readback failure')
            return real_read(path, *args)
        with patch.object(checkpoint, 'read_checkpoint', side_effect=fail_child), self.assertRaisesRegex(ValueError, 'injected'):
            transition.create_transition(plan, run_id='child')
        self.assertFalse((self.root / 'child/manifest.json').exists())
        self.assertEqual(transition._files(self.parent), parent_files)

    def test_source_change_after_plan_is_rejected_before_destination_write(self):
        plan = self.plan()
        self.helpers['unchanged.py'] = b'VALUE = 9\n'
        self.current['auxiliary_sources_sha256']['unchanged.py'] = sha(self.helpers['unchanged.py'])
        with self.assertRaisesRegex(ValueError, 'sources changed after planning'):
            transition.create_transition(plan, run_id='child')
        self.assertFalse((self.root / 'child').exists())

    def test_unsealed_source_membership_and_disabled_policy_are_rejected(self):
        with patch.object(transition, 'APPROVED_CHANGED_HELPERS', {}), self.assertRaisesRegex(ValueError, 'source delta'):
            self.plan()
        with self.assertRaisesRegex(ValueError, 'explicitly enabled'):
            transition.plan_transition(self.parent, controls={'enabled': False})
        self.assertEqual(set(self.root.iterdir()), {self.parent, self.seal})

    def test_actual_producer_zero_history_api_preserves_every_preexisting_typed_field(self):
        import effective_subduction_initiation as producer
        with patch.object(transition, '_policy_api', return_value=producer):
            plan = transition.plan_transition(self.parent)
            child, receipt = transition.create_transition(plan, run_id='real-policy-child')
        self.assertEqual(receipt['activated_controls'], producer.normalize({'enabled': True}))
        self.assertEqual(plan.derived.effective_subduction_initiation['retrospective_time_myr'], 0.)
        self.assertEqual(plan.derived.effective_subduction_initiation['retrospective_shortening_km'], 0.)
        self.assertFalse(plan.derived.effective_subduction_initiation['physical_state_changed'])
        restored, _ = checkpoint.read_checkpoint(child / 'checkpoint.npz', self.current, native_engine.Simulation)
        self.assertTrue(producer.enabled(restored))
        self.assertEqual(transition._fingerprint(transition._physical_state(restored)),
            transition._fingerprint(transition._physical_state(self.original)))

    def test_nested_tuple_sets_survive_deepcopy_and_checkpoint_without_history_change(self):
        self.original.event_keys = {(i, 'collision', i * 31) for i in range(26)}
        self.original.rift_systems = [{'id': 3, '_bonds': {(i, i + 1) for i in range(45)}}]
        self.publish_parent()
        self.approve_checkpoint()
        plan = self.plan()
        child, _ = transition.create_transition(plan, run_id='set-history-child')
        restored, _ = checkpoint.read_checkpoint(child / 'checkpoint.npz', self.current, native_engine.Simulation)
        self.assertEqual(restored.event_keys, self.original.event_keys)
        self.assertEqual(restored.rift_systems, self.original.rift_systems)
        self.assertEqual(transition._fingerprint(transition._physical_state(restored)),
                         transition._fingerprint(transition._physical_state(self.original)))
        base = PolicyFixture.upgrade
        for field in ('event_keys', 'rift_systems'):
            def altered(state, controls):
                report = base(state, controls)
                target = state.event_keys if field == 'event_keys' else state.rift_systems[0]['_bonds']
                target.remove(next(iter(target)))
                return report
            with self.subTest(field=field), patch.object(PolicyFixture, 'upgrade', side_effect=altered), self.assertRaisesRegex(ValueError, 'inherited physical'):
                self.plan()

    def test_semantic_set_sorting_keeps_order_types_and_nested_array_associations(self):
        from plate_domains import PersistentDomainTracker
        def tracker(value):
            result = PersistentDomainTracker.__new__(PersistentDomainTracker)
            result.values = np.array(value, dtype=np.int64)
            return result
        left, right = tracker([1, 2]), tracker([3, 4])
        baseline = {'members': {left, right}, 'ordered': [(1, 2), (3, 4)], 'mapping': {1: 'a', 2: 'b'}}
        reference = transition._fingerprint(baseline)
        self.assertEqual(reference, transition._fingerprint(deepcopy(baseline)))
        # Exercise reversed encoded set order explicitly; content is inlined
        # before sorting, so renamed array references cannot change semantics.
        arrays = {}
        encoded = checkpoint._encode(baseline, arrays)
        reversed_tree = deepcopy(encoded)
        reversed_tree['v'][0][1]['v'].reverse()
        self.assertEqual(transition._semantic_tree(encoded, arrays),
                         transition._semantic_tree(reversed_tree, arrays))
        for mode in ('member_bytes', 'member_dtype', 'member_shape', 'tuple_order', 'dict_order', 'container_type'):
            changed = deepcopy(baseline)
            if mode.startswith('member_'):
                first = next(item for item in changed['members'] if item.values[0] == 1)
                if mode == 'member_bytes': first.values[0] = 9
                if mode == 'member_dtype': first.values = first.values.astype(np.uint64)
                if mode == 'member_shape': first.values = first.values.reshape(1, 2)
            elif mode == 'tuple_order': changed['ordered'][0] = (2, 1)
            elif mode == 'dict_order': changed['mapping'] = {2: 'b', 1: 'a'}
            else: changed['ordered'][0] = [1, 2]
            with self.subTest(mode=mode):
                self.assertNotEqual(reference, transition._fingerprint(changed))

    def test_public_cli_default_is_plan_only_and_stdout_is_compact(self):
        receipt = self.root / 'planned.json'
        before = transition._files(self.parent)
        stdout = io.StringIO()
        with patch('sys.argv', ['lite_initiation_transition', '--parent', str(self.parent), '--receipt', str(receipt)]), \
                patch('sys.stdout', stdout), patch.object(transition, 'create_transition', side_effect=AssertionError('unexpected publication')):
            transition.main()
        summary = json.loads(stdout.getvalue())
        self.assertLess(len(stdout.getvalue()), 1500)
        self.assertEqual(summary['state'], 'read_only_plan_validated')
        self.assertNotIn('parent_header', summary)
        self.assertNotIn('parent_files_sha256', summary)
        self.assertEqual(summary['receipt_sha256'], transition._file_sha(receipt))
        self.assertEqual(json.loads(receipt.read_text())['parent_header']['compatibility'], self.prior)
        self.assertEqual(transition._files(self.parent), before)


if __name__ == '__main__':
    unittest.main()
