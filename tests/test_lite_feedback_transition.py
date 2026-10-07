"""Accepted-boundary policy import preserves predecessor bytes and typed state.

The fixture is a constructorless native checkpoint, not another simulation.
It includes inherited initiation, ordinary rift/back-arc and ancestor frame
lineage to exercise this second-generation transition's storage contracts.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
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
import lite_feedback_transition as transition
import native_engine


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


class FractureFixture:
    @staticmethod
    def normalize(raw=None):
        result = dict(dict(enabled=True, rift_width_km=150., breakup_stretch=3.,
            search_policy=dict(version=1, mode='bounded_axes', max_axes=10)), **(raw or {}))
        for name in ('rift_width_km', 'breakup_stretch'):
            result[name] = float(result[name])
        return result

    @staticmethod
    def upgrade(state, controls):
        state.config = deepcopy(state.config)
        state.config['force_limit_rifting'] = deepcopy(controls)
        state.force_rifting_version = 1
        state.force_rifting_state = {}
        state.force_rifting_diagnostics = dict(version=1, checks=0, commits=0, refusals=[])
        state.force_rifting_policy = dict(version=1, mode='bounded_axes', parameters=deepcopy(controls),
            activation_myr=float(state.t), epoch_myr=float(state.t),
            history='Future matched cut observations only; virtual development, not realized material extension.')
        return deepcopy(state.force_rifting_policy)


class CarrierFixture:
    POLICY = 'Fixture declared mantle-connected carrier traction; omitted reservoir.'

    @staticmethod
    def normalize(raw=None):
        result = dict(dict(version=1, enabled=True, alpha=.1), **(raw or {}))
        result['alpha'] = float(result['alpha'])
        return result

    @staticmethod
    def upgrade(state, controls):
        state.config = deepcopy(state.config)
        state.config['effective_subduction']['carrier_traction'] = deepcopy(controls)
        state.effective_subduction_carrier_traction = dict(version=1, enabled=True,
            activation_myr=float(state.t), parameters=deepcopy(controls), retrospective_work_j=0., law=CarrierFixture.POLICY)
        return deepcopy(state.effective_subduction_carrier_traction)


class LiteFeedbackTransitionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parent = self.root / 'current-lite-parent'
        self.parent.mkdir()
        self.changed_names = sorted(transition.REQUIRED_CHANGED_HELPERS)
        self.engine = ('\n'.join('import ' + name[:-3] for name in self.changed_names) + '\n').encode()
        self.old = {name: b'VALUE = 1\n' for name in self.changed_names}
        self.new = {name: b'VALUE = 2\n' for name in self.changed_names}
        self.new['effective_subduction.py'] += b'import effective_subduction_carrier_traction\n'
        self.new['effective_subduction_carrier_traction.py'] = b'VERSION = 1\n'
        self.old['unchanged.py'] = b'VALUE = 3\n'
        self.new['unchanged.py'] = self.old['unchanged.py']
        self.engine += b'import unchanged\n'
        self.prior = dict(engine_sha256=sha(self.engine),
            auxiliary_sources_sha256={name: sha(value) for name, value in self.old.items()},
            numpy_version=np.__version__, flow_backend={'name': 'fixture', 'binary_sha256': 'flow1'},
            material_backend={'name': 'fixture', 'binary_sha256': 'material1'})
        self.current = dict(self.prior, auxiliary_sources_sha256={name: sha(value) for name, value in self.new.items()})
        (self.parent / 'engine.py').write_bytes(self.engine)
        for name, value in self.old.items():
            (self.parent / name).write_bytes(value)
        config = dict(seed=37, duration_myr=1000., snapshot_myr=2., dt_myr=1.,
            enhanced_rifting=False, physics_profile='reviewed_v1',
            force_limit_rifting={'enabled': False},
            effective_subduction=dict(enabled=True, force_n_per_m=3e13, initiation={'enabled': True}))
        self.original = SimpleNamespace(t=38., steps=38, config=config, rng=np.random.default_rng(37),
            plate_resistance_version=1,
            effective_subduction_version=1, effective_subduction_settings=deepcopy(config['effective_subduction']),
            effective_subduction_initiation=dict(version=1, activation_myr=25., epoch_myr=38.,
                next_candidate_id=3, candidates=[dict(id=2, compression_myr=4., shortening_km=37.)],
                births=[], events=[dict(time_myr=37., kind='candidate-observed')]),
            omega=np.arange(6, dtype=np.float64).reshape(2, 3),
            material_surface=dict(vertices=np.eye(3), faces=np.array([[0, 1, 2]], np.int32),
                reference_area_km2=np.array([18.]), area_km2=np.array([17.])),
            structure=dict(thickness_km=np.array([36.]), inventories=np.arange(6.).reshape(2, 3)),
            trench_systems=[dict(id=4, downgoing_plate_uid=2, overriding_plate_uid=7,
                effective_subduction={'version': 1, 'force_n_per_m': 3e13}, geometry_xyz=[[1., 0., 0.]])],
            backarc_systems=[dict(id=1, loaded_km=55., last_attempt_myr=35.)],
            rift_systems=[dict(id=3, _bonds={(1, 2), (2, 3)}, counted_extension_km=42.)],
            events=[dict(type='old-event', time_myr=37.)], event_keys={(i, 'old-event') for i in range(12)})
        self.manifest = dict(run_id=self.parent.name, title='Asein Lite', state='paused', can_resume=True,
            config=deepcopy(config), frame_count=3, frames=[dict(index=0, time_myr=0.),
                dict(index=1, time_myr=26.), dict(index=2, time_myr=38.)], time_myr=38.,
            checkpoint_time_myr=38., integration_time_myr=38., next_output_myr=40., duration_myr=1000.,
            source_transition={'kind': 'lite_effective_subduction_initiation_source_transition',
                'epoch_myr': 25., 'parent_run_id': 'earlier-lite-parent', 'inherited_frame_count': 1},
            frame_sources=[dict(index=0, time_myr=0., run_id='earlier-lite-parent',
                engine_sha256='ancestor_engine', auxiliary_sources_sha256={'old.py': 'ancestor_helper'},
                inherited=True, parent_frame_source=None)], **deepcopy(self.prior))
        for row in self.manifest['frames']:
            stem = self.parent / f"frame_{row['index']:04d}"
            np.savez_compressed(stem.with_suffix('.npz'), material=np.arange(9).reshape(3, 3),
                native_owner=np.array([2, 7]))
            stem.with_suffix('.json').write_text(json.dumps(row))
        (self.parent / 'inherited-parent').mkdir()
        (self.parent / 'inherited-parent/manifest.json').write_text('{"run_id":"earlier-lite-parent"}')
        (self.parent / 'initial.json').write_bytes(b'{"artwork":"unchanged-original"}\n')
        (self.parent / 'config.json').write_text(json.dumps(config))
        self.publish_parent()
        self.preservation = self.root / 'parent-preservation.json'
        self.seal = self.root / 'runtime-seal.json'
        self.review = self.root / 'source-review.json'
        self.controls = dict(force_limit_rifting=FractureFixture.normalize(), carrier_traction=CarrierFixture.normalize())
        self.approve()
        self.patched_runtime_context = transition._current_context
        patches = [patch.object(transition, 'REVIEWED_APPROVAL', self.approval),
            patch.object(transition, '_current_context', side_effect=lambda approved: (deepcopy(self.current), self.engine, dict(self.new))),
            patch.object(transition, '_policy_apis', return_value=(FractureFixture, CarrierFixture))]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def publish_parent(self):
        checkpoint.write_checkpoint(self.parent / 'checkpoint.npz', self.original, self.manifest, self.prior)
        (self.parent / 'manifest.json').write_text(json.dumps(self.manifest))

    def approve(self, *, review_changes=None):
        transition._write_json(self.preservation, dict(run_id=self.parent.name,
            files=[dict(path=name, sha256=value) for name, value in transition._files(self.parent).items()]))
        transition._write_json(self.seal, self.current['auxiliary_sources_sha256'])
        self.review_data = dict(kind=transition.SOURCE_REVIEW_KIND, version=1, ready=True,
            boundary_kind=getattr(self, 'boundary_kind', 'paused'),
            numerical_repair_review_sha256=transition._file_sha(self.numerical_review)
                if getattr(self, 'numerical_review', None) is not None else None,
            parent_run_id=self.parent.name, parent_checkpoint_sha256=transition._file_sha(self.parent / 'checkpoint.npz'),
            parent_compatibility_sha256=sha(transition._json_bytes(self.prior)),
            child_compatibility_sha256=sha(transition._json_bytes(self.current)), policy=deepcopy(self.controls),
            changed_helpers={name: dict(parent_sha256=sha(self.old[name]), child_sha256=sha(self.new[name]))
                for name in self.old.keys() & self.new.keys() if self.old[name] != self.new[name]},
            added_helpers={name: sha(self.new[name]) for name in self.new.keys() - self.old.keys()}, removed_helpers=[])
        self.review_data.update(review_changes or {})
        transition._write_json(self.review, self.review_data)
        self.approval = transition.TransitionApproval(self.parent.name, transition._file_sha(self.parent / 'checkpoint.npz'),
            self.preservation, transition._file_sha(self.preservation), self.seal, transition._file_sha(self.seal),
            self.review, transition._file_sha(self.review), transition._json_bytes(self.controls).decode(), review_ready=True,
            boundary_kind=getattr(self, 'boundary_kind', 'paused'),
            stop_evidence_receipt=getattr(self, 'stop_evidence', None),
            stop_evidence_sha256=transition._file_sha(self.stop_evidence) if getattr(self, 'stop_evidence', None) else None,
            numerical_repair_receipt=getattr(self, 'numerical_review', None),
            numerical_repair_sha256=transition._file_sha(self.numerical_review) if getattr(self, 'numerical_review', None) else None)
        transition.REVIEWED_APPROVAL = self.approval

    def plan(self):
        return transition.plan_transition(self.parent)

    def stopped_boundary(self):
        self.boundary_kind = 'durable_stopped'
        self.engine += b'import bounded_gravity\n'
        self.old['bounded_gravity.py'], self.new['bounded_gravity.py'] = b'VALUE = 1\n', b'VALUE = 2\n'
        self.prior.update(engine_sha256=sha(self.engine),
            auxiliary_sources_sha256={name: sha(value) for name, value in self.old.items()})
        self.current.update(engine_sha256=sha(self.engine),
            auxiliary_sources_sha256={name: sha(value) for name, value in self.new.items()})
        (self.parent / 'engine.py').write_bytes(self.engine)
        (self.parent / 'bounded_gravity.py').write_bytes(self.old['bounded_gravity.py'])
        self.original.t, self.original.steps = 43., 44
        self.manifest.update(state='interrupted', error=None, time_myr=43., integration_time_myr=43.,
            checkpoint_time_myr=43., next_output_myr=44., **deepcopy(self.prior))
        self.manifest['frames'][-1]['time_myr'] = 42.
        (self.parent / 'frame_0002.json').write_text(json.dumps(self.manifest['frames'][-1]))
        self.publish_parent()
        self.external = dict(deepcopy(self.manifest), state='error', time_myr=42., next_output_myr=26.,
            error='IncompleteContactStepError: original numerical SQP budget failed; entire trial rolled back.')
        (self.parent / 'manifest.json').write_text(json.dumps(self.external))
        self.stop_evidence = self.root / 'status-error.json'
        self.stop_data = {key: deepcopy(self.external[key]) for key in
            ('run_id', 'state', 'error', 'time_myr', 'integration_time_myr', 'checkpoint_time_myr', 'frame_count')}
        transition._write_json(self.stop_evidence, self.stop_data)
        validations = []
        for role in ('original-policy', 'feedback-policy'):
            path = self.root / (role + '-validation.json')
            transition._write_json(path, dict(kind='portable fixture interval proof', policy=role,
                accepted_from_myr=43., accepted_to_myr=44., physical_guards_unchanged=True))
            validations.append(dict(path=str(path), sha256=transition._file_sha(path)))
        self.numerical_review = self.root / 'numerical-review.json'
        self.numerical_data = dict(kind=transition.NUMERICAL_REVIEW_KIND, version=1, ready=True,
            parent_run_id=self.parent.name, parent_checkpoint_sha256=transition._file_sha(self.parent / 'checkpoint.npz'),
            helper='bounded_gravity.py', parent_sha256=sha(self.old['bounded_gravity.py']),
            child_sha256=sha(self.new['bounded_gravity.py']), physical_guards_unchanged=True,
            original_policy_interval_passed=True, feedback_policy_interval_passed=True,
            validation_receipts=validations)
        transition._write_json(self.numerical_review, self.numerical_data)
        self.approve()

    def test_exact_arrays_rng_existing_clocks_and_nested_source_lineage_are_retained(self):
        before = transition._files(self.parent)
        with patch.object(native_engine.Simulation, '__init__', side_effect=AssertionError('constructor called')):
            plan = self.plan()
            child, receipt = transition.create_transition(plan, run_id='child')
            restored, saved = checkpoint.read_checkpoint(child / 'checkpoint.npz', self.current, native_engine.Simulation)
        self.assertEqual(transition._fingerprint(transition._physical_state(restored)),
            transition._fingerprint(transition._physical_state(self.original)))
        self.assertEqual(transition._files(self.parent), before)
        self.assertEqual(transition._files(child / 'inherited-parent'), before)
        self.assertEqual(saved['frames'], self.manifest['frames'])
        self.assertEqual(saved['next_output_myr'], 40.)
        self.assertEqual(restored.effective_subduction_initiation, self.original.effective_subduction_initiation)
        self.assertEqual(restored.backarc_systems, self.original.backarc_systems)
        self.assertEqual(restored.rift_systems, self.original.rift_systems)
        self.assertEqual(restored.rng.bit_generator.state, self.original.rng.bit_generator.state)
        self.assertEqual(restored.force_rifting_state, {})
        self.assertEqual(restored.force_rifting_policy['epoch_myr'], 38.)
        self.assertEqual(restored.effective_subduction_carrier_traction['retrospective_work_j'], 0.)
        self.assertEqual(saved['frame_sources'][0]['run_id'], self.parent.name)
        self.assertEqual(saved['frame_sources'][0]['parent_frame_source'], self.manifest['frame_sources'][0])
        self.assertIsNone(saved['frame_sources'][1]['parent_frame_source'])
        for index in range(3):
            for suffix in ('.npz', '.json'):
                name = f'frame_{index:04d}{suffix}'
                self.assertEqual((child / name).read_bytes(), (self.parent / name).read_bytes())
        self.assertEqual((child / 'checkpoint.policy-boundary.npz').read_bytes(), (child / 'checkpoint.npz').read_bytes())
        self.assertEqual(transition._file_sha(child / 'checkpoint.npz'), receipt['child_checkpoint_sha256'])
        self.assertEqual(receipt['preservation']['preexisting_array_count'], 7)
        self.assertEqual(json.loads((child / 'manifest.json').read_text()), saved)
        self.assertEqual((child / 'initial.json').read_bytes(), (self.parent / 'initial.json').read_bytes())

    def test_absent_unready_or_replaced_approval_blocks_planning_and_frozen_fields_cannot_mutate(self):
        with self.assertRaises(FrozenInstanceError):
            self.approval.review_ready = False
        for approval in (None, replace(self.approval, review_ready=False)):
            with self.subTest(approval=approval), patch.object(transition, 'REVIEWED_APPROVAL', approval), \
                    self.assertRaisesRegex(ValueError, 'not sealed'):
                self.plan()
        with self.assertRaisesRegex(ValueError, 'not sealed'):
            transition.plan_transition(self.parent, approval=replace(self.approval))

    def test_running_boundary_bad_scheduler_and_source_lineage_mismatch_rejected(self):
        initial = deepcopy(self.manifest)
        for alteration in ({'state': 'running'}, {'next_output_myr': 38.}, {'next_output_myr': 1001.}):
            self.manifest = dict(deepcopy(initial), **alteration)
            self.publish_parent()
            self.approve()
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                self.plan()
        self.manifest = initial
        self.publish_parent()
        self.approve()
        external = deepcopy(self.manifest)
        external['frame_sources'][0]['run_id'] = 'forged-ancestor'
        (self.parent / 'manifest.json').write_text(json.dumps(external))
        self.approve()
        with self.assertRaisesRegex(ValueError, 'source lineage'):
            self.plan()

    def test_checkpoint_and_preservation_tampering_rejected_before_plan(self):
        self.original.omega[0, 0] += 1.
        self.publish_parent()
        with self.assertRaisesRegex(ValueError, 'exact reviewed accepted-step'):
            self.plan()
        self.approve()
        self.preservation.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'preservation receipt'):
            self.plan()

    def test_incomplete_unready_or_wrong_source_delta_review_rejected(self):
        for alteration in ({'ready': False}, {'changed_helpers': {}}, {'parent_run_id': 'other-run'},
                {'policy': {}}, {'parent_compatibility_sha256': '0' * 64}):
            self.approve(review_changes=alteration)
            with self.subTest(alteration=alteration), self.assertRaisesRegex(ValueError, 'source delta'):
                self.plan()
        self.approve()
        self.review.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'source delta review receipt'):
            self.plan()

    def test_backend_engine_and_unreviewed_runtime_helper_changes_rejected(self):
        baseline = deepcopy(self.current)
        for field in ('flow_backend', 'material_backend', 'numpy_version'):
            self.current = dict(deepcopy(baseline), **{field: 'different'})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                self.plan()
        self.current = baseline
        self.new['unchanged.py'] = b'VALUE = 4\n'
        self.current['auxiliary_sources_sha256']['unchanged.py'] = sha(self.new['unchanged.py'])
        with self.assertRaisesRegex(ValueError, 'source delta review'):
            self.plan()
        self.approve()
        with self.assertRaisesRegex(ValueError, 'narrow Lite feedback scope'):
            self.plan()

    def test_mutated_inherited_material_state_and_new_unknown_policy_fields_rejected(self):
        upgrade = FractureFixture.upgrade
        for mode in ('material', 'rng', 'initiation_clock', 'backarc_clock', 'new_policy_field', 'paid_work'):
            def bad(state, controls):
                report = upgrade(state, controls)
                if mode == 'material': state.material_surface['faces'] = state.material_surface['faces'].astype(np.int64)
                if mode == 'rng': state.rng.random()
                if mode == 'initiation_clock': state.effective_subduction_initiation['epoch_myr'] += 1.
                if mode == 'backarc_clock': state.backarc_systems[0]['last_attempt_myr'] += 1.
                if mode == 'new_policy_field': state.force_rifting_policy['arbitrary_clock'] = 0.
                if mode == 'paid_work': state.force_rifting_state['2'] = {'paid_cut_work_j': 1.}
                return report
            with self.subTest(mode=mode), patch.object(FractureFixture, 'upgrade', side_effect=bad), self.assertRaises(ValueError):
                self.plan()
        carrier_upgrade = CarrierFixture.upgrade
        def extra(state, controls):
            report = carrier_upgrade(state, controls)
            state.effective_subduction_carrier_traction['unknown_work_j'] = 0.
            return report
        with patch.object(CarrierFixture, 'upgrade', side_effect=extra), self.assertRaisesRegex(ValueError, 'exactly declared'):
            self.plan()

    def test_unknown_disabled_and_noncanonical_controls_rejected(self):
        for change in ('unknown', 'disabled', 'noncanonical', 'unnormalized_numeric_type'):
            controls = deepcopy(self.controls)
            if change == 'unknown': controls['unknown'] = {}
            if change == 'disabled': controls['carrier_traction']['enabled'] = False
            if change == 'unnormalized_numeric_type': controls['force_limit_rifting']['rift_width_km'] = 150
            text = json.dumps(controls) if change == 'noncanonical' else transition._json_bytes(controls).decode()
            approval = replace(self.approval, controls_json=text)
            with self.subTest(change=change), patch.object(transition, 'REVIEWED_APPROVAL', approval), self.assertRaises(ValueError):
                self.plan()

    def test_actual_policy_apis_initialize_only_declared_future_state(self):
        import force_rifting
        import effective_subduction_carrier_traction as carrier
        self.controls = dict(force_limit_rifting=force_rifting.normalize(dict(enabled=True,
            search_policy=deepcopy(force_rifting.BOUNDED_SEARCH))),
            carrier_traction=carrier.normalize(dict(enabled=True)))
        self.approve()
        with patch.object(transition, '_policy_apis', return_value=(force_rifting, carrier)):
            plan = self.plan()
            child, _ = transition.create_transition(plan, run_id='actual-producers-child')
        restored, _ = checkpoint.read_checkpoint(child / 'checkpoint.npz', self.current, native_engine.Simulation)
        self.assertTrue(carrier.enabled(restored))
        self.assertEqual(force_rifting.bounded_policy(restored)['epoch_myr'], 38.)
        self.assertEqual(transition._fingerprint(transition._physical_state(restored)),
            transition._fingerprint(transition._physical_state(self.original)))

    def test_mutable_plan_parent_source_change_and_readback_failure_never_publish_manifest(self):
        plan = self.plan()
        plan.derived.effective_subduction_initiation['epoch_myr'] += 1.
        with self.assertRaisesRegex(ValueError, 'modified after validation'):
            transition.create_transition(plan, run_id='child')
        self.assertFalse((self.root / 'child').exists())
        plan = self.plan()
        real_read = checkpoint.read_checkpoint
        def fail_child(path, *args):
            if Path(path).parent.name == 'child': raise ValueError('injected readback failure')
            return real_read(path, *args)
        before = transition._files(self.parent)
        with patch.object(checkpoint, 'read_checkpoint', side_effect=fail_child), self.assertRaisesRegex(ValueError, 'injected'):
            transition.create_transition(plan, run_id='child')
        self.assertFalse((self.root / 'child/manifest.json').exists())
        self.assertEqual(transition._files(self.parent), before)

    def test_changed_parent_after_plan_and_runtime_after_plan_reject_before_writes(self):
        plan = self.plan()
        self.new['unchanged.py'] = b'VALUE = 9\n'
        self.current['auxiliary_sources_sha256']['unchanged.py'] = sha(self.new['unchanged.py'])
        with self.assertRaisesRegex(ValueError, 'sources changed after planning'):
            transition.create_transition(plan, run_id='child')
        self.assertFalse((self.root / 'child').exists())
        self.new['unchanged.py'] = self.old['unchanged.py']
        self.current['auxiliary_sources_sha256']['unchanged.py'] = sha(self.old['unchanged.py'])
        (self.parent / 'frame_0002.npz').write_bytes(b'changed after planning')
        with self.assertRaisesRegex(ValueError, 'parent changed after planning'):
            transition.create_transition(plan, run_id='child')
        self.assertFalse((self.root / 'child').exists())

    def test_child_cannot_be_created_inside_its_own_parent_history(self):
        before = transition._files(self.parent)
        with self.assertRaisesRegex(ValueError, 'must be disjoint'):
            transition.create_transition(self.plan(), output_root=self.parent, run_id='nested-child')
        self.assertFalse((self.parent / 'nested-child').exists())
        self.assertEqual(transition._files(self.parent), before)

    def test_actual_runtime_capture_must_match_frozen_seal_and_current_disk_files(self):
        runtime_root = self.root / 'runtime'
        runtime_root.mkdir()
        (runtime_root / 'tectonics.py').write_bytes(self.engine)
        for name, value in self.new.items():
            (runtime_root / name).write_bytes(value)
        # Reach the real context checker under this fixture's independent
        # compatibility/capture, instead of the general storage-test override.
        real = self.patched_runtime_context
        with patch.object(transition, 'ROOT', runtime_root), \
                patch.object(transition.server.SimulationManager, 'compatibility', return_value=self.current), \
                patch.object(transition.server, 'ENGINE_SOURCE', self.engine), \
                patch.object(transition.server, 'AUXILIARY_SOURCES', self.new):
            self.assertEqual(real(self.approval), (self.current, self.engine, self.new))
            (runtime_root / 'unchanged.py').write_bytes(b'altered after capture')
            with self.assertRaisesRegex(ValueError, 'files changed after local source capture'):
                real(self.approval)
            (runtime_root / 'unchanged.py').write_bytes(self.new['unchanged.py'])
            self.seal.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'runtime source seal'):
                real(self.approval)

    def test_stopped_durable_generation_is_imported_without_forging_old_manifest_or_stale_scheduler(self):
        self.stopped_boundary()
        before = transition._files(self.parent)
        external_bytes = (self.parent / 'manifest.json').read_bytes()
        plan = self.plan()
        child, receipt = transition.create_transition(plan, run_id='recovered-feedback-child')
        restored, saved = checkpoint.read_checkpoint(child / 'checkpoint.npz', self.current, native_engine.Simulation)
        self.assertEqual(restored.t, 43.)
        self.assertEqual(restored.steps, 44)
        self.assertEqual(saved['state'], 'paused')
        self.assertEqual(saved['next_output_myr'], 44.)
        self.assertEqual(saved['frames'][-1]['time_myr'], 42.)
        self.assertNotIn('error', saved)
        self.assertEqual(receipt['boundary_kind'], 'durable_stopped')
        self.assertEqual(receipt['durable_recovery']['external_stale_next_output_myr'], 26.)
        self.assertFalse(receipt['durable_recovery']['historical_manifests_rewritten'])
        self.assertEqual((child / 'inherited-parent/manifest.json').read_bytes(), external_bytes)
        self.assertEqual((self.parent / 'manifest.json').read_bytes(), external_bytes)
        self.assertEqual(transition._files(self.parent), before)
        self.assertEqual(transition._fingerprint(transition._physical_state(restored)),
            transition._fingerprint(transition._physical_state(self.original)))
        self.assertEqual((child / 'stop-evidence.json').read_bytes(), self.stop_evidence.read_bytes())
        self.assertEqual((child / 'numerical-repair-review.json').read_bytes(), self.numerical_review.read_bytes())

    def test_stopped_future_public_checkpoint_error_or_generation_mismatch_rejected(self):
        self.stopped_boundary()
        original = deepcopy(self.external)
        for alteration in ({'time_myr': 44.}, {'integration_time_myr': 44.}, {'checkpoint_time_myr': 44.},
                {'error': ''}, {'state': 'paused'}, {'frame_count': 2}, {'config': {}}):
            self.external = dict(deepcopy(original), **alteration)
            (self.parent / 'manifest.json').write_text(json.dumps(self.external))
            self.stop_data.update({key: self.external.get(key) for key in self.stop_data})
            transition._write_json(self.stop_evidence, self.stop_data)
            self.approve()
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                self.plan()
        self.assertFalse((self.root / 'recovered-feedback-child').exists())

    def test_stopped_missing_failure_and_unvalidated_numerical_repair_are_rejected(self):
        self.stopped_boundary()
        approval = replace(self.approval, stop_evidence_receipt=None, stop_evidence_sha256=None)
        with patch.object(transition, 'REVIEWED_APPROVAL', approval), self.assertRaisesRegex(ValueError, 'requires sealed failure'):
            self.plan()
        for alteration in ({'physical_guards_unchanged': False}, {'original_policy_interval_passed': False},
                {'feedback_policy_interval_passed': False}, {'validation_receipts': []},
                {'validation_receipts': [self.numerical_data['validation_receipts'][0]] * 2},
                {'child_sha256': '0' * 64}):
            modified = dict(deepcopy(self.numerical_data), **alteration)
            transition._write_json(self.numerical_review, modified)
            self.approve()
            with self.subTest(alteration=alteration), self.assertRaisesRegex(ValueError, 'numerical'):
                self.plan()
        transition._write_json(self.numerical_review, self.numerical_data)
        self.approve()
        Path(self.numerical_data['validation_receipts'][0]['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'validation evidence'):
            self.plan()

    def test_stopped_evidence_hash_mismatch_and_unsupported_boundary_fail_before_writes(self):
        self.stopped_boundary()
        self.stop_evidence.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'stopped status evidence'):
            self.plan()
        approval = replace(self.approval, boundary_kind='pretend-paused')
        with patch.object(transition, 'REVIEWED_APPROVAL', approval), self.assertRaisesRegex(ValueError, 'boundary kind'):
            self.plan()

    def test_relative_numerical_proofs_remain_exact_self_contained_after_external_review_changes(self):
        self.stopped_boundary()
        for row in self.numerical_data['validation_receipts']:
            row['path'] = Path(row['path']).name
        transition._write_json(self.numerical_review, self.numerical_data)
        self.approve()
        before = transition._files(self.parent)
        expected_review = self.numerical_review.read_bytes()
        child, receipt = transition.create_transition(self.plan(), run_id='self-contained-child')
        self.assertEqual((child / 'numerical-repair-review.json').read_bytes(), expected_review)
        self.assertEqual(len(receipt['numerical_validation_evidence']), 2)
        for item in receipt['numerical_validation_evidence']:
            source = Path(item['source_path'])
            self.assertEqual(item['storage'], 'child_relative_exact_copy')
            self.assertEqual(item['child_file'], item['reference_path'])
            self.assertEqual((child / item['child_file']).read_bytes(), source.read_bytes())
            self.assertEqual(transition._file_sha(child / item['child_file']), item['sha256'])
            source.write_text('{"later_external_review": true}')
        self.numerical_review.write_text('{"later_external_review": true}')
        # The retained root-sealed review still resolves both unchanged proof
        # references locally; later external receipt replacements cannot alter it.
        retained_review = json.loads((child / 'numerical-repair-review.json').read_text())
        for row in retained_review['validation_receipts']:
            self.assertEqual(transition._file_sha(child / row['path']), row['sha256'])
        self.assertEqual(transition._files(self.parent), before)
        self.assertEqual(json.loads((child / 'source-transition.json').read_text())['numerical_validation_evidence'],
            receipt['numerical_validation_evidence'])

    def test_external_numerical_proof_references_are_truthful_and_relative_metadata_collisions_fail(self):
        self.stopped_boundary()
        child, receipt = transition.create_transition(self.plan(), run_id='external-proof-child')
        for item in receipt['numerical_validation_evidence']:
            self.assertEqual(item['storage'], 'external_reference')
            self.assertIsNone(item['child_file'])
            self.assertTrue(Path(item['reference_path']).is_absolute())
            self.assertEqual(item['source_path'], item['reference_path'])
            self.assertFalse((child / Path(item['reference_path']).name).exists())
        for role, name in enumerate(('manifest.json', 'initial.json', 'source-transition.json.tmp', 'frame_9999.json',
                'MANIFEST.JSON', 'FRAME_9999.JSON')):
            path = self.root / name
            transition._write_json(path, dict(kind='proof with reserved basename', role=role))
            self.numerical_data['validation_receipts'][0] = dict(path=name, sha256=transition._file_sha(path))
            transition._write_json(self.numerical_review, self.numerical_data)
            self.approve()
            run_id = 'collision-child-' + str(role)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'basename collides'):
                transition.create_transition(self.plan(), run_id=run_id)
            self.assertFalse((self.root / run_id / 'manifest.json').exists())

    def test_cli_plans_only_and_prints_compact_result(self):
        receipt = self.root / 'planned.json'
        output = io.StringIO()
        with patch('sys.argv', ['lite_feedback_transition', '--parent', str(self.parent), '--receipt', str(receipt)]), \
                patch('sys.stdout', output), patch.object(transition, 'create_transition', side_effect=AssertionError('unexpected publication')):
            transition.main()
        summary = json.loads(output.getvalue())
        self.assertEqual(summary['state'], 'read_only_plan_validated')
        self.assertNotIn('parent_files_sha256', summary)
        self.assertLess(len(output.getvalue()), 1800)
        self.assertEqual(summary['receipt_sha256'], transition._file_sha(receipt))


if __name__ == '__main__':
    unittest.main()
