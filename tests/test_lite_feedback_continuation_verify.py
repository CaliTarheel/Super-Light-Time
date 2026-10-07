"""Small evidence/scheduler tests; no simulation constructors or integration."""
from copy import deepcopy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1]/'lite_feedback_verification.py'
SPEC = importlib.util.spec_from_file_location('feedback_continuation_verify', SCRIPT)
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


def history(folder):
    folder = Path(folder)
    binary, metadata = folder/'frame_0000.npz', folder/'frame_0000.json'
    binary.write_bytes(b'accepted inherited frame bytes')
    metadata.write_text(json.dumps(dict(index=0, time_myr=42.)))
    (folder/'frame_0001.json').write_text(json.dumps(dict(index=1, time_myr=44.)))
    sealed = {path.name: verify.sha(path) for path in (binary, metadata)}
    parent = dict(frame_count=1, frames=[dict(index=0, time_myr=42.)],
        frame_sources=[dict(run_id='older', inherited=True, parent_frame_source=dict(run_id='oldest'))])
    compatibility = dict(engine_sha256='e'*64, auxiliary_sources_sha256={'helper.py': 'h'*64})
    receipt = dict(parent_run_id=verify.PARENT, parent_manifest_sha256='m'*64,
        parent_compatibility=compatibility, inherited_frame_count=1)
    source = dict(index=0, time_myr=42., run_id=verify.PARENT, source_manifest='inherited-parent/manifest.json',
        source_manifest_sha256=receipt['parent_manifest_sha256'], engine_sha256=compatibility['engine_sha256'],
        auxiliary_sources_sha256=compatibility['auxiliary_sources_sha256'], files_sha256={
            suffix: sealed[f'frame_0000{suffix}'] for suffix in ('.npz', '.json')},
        inherited=True, parent_frame_source=deepcopy(parent['frame_sources'][0]))
    manifest = dict(frames=parent['frames']+[dict(index=1, time_myr=44.)], frame_sources=[source],
        source_transition=dict(kind=verify.TRANSITION_KIND, version=1, epoch_myr=43., parent_run_id=verify.PARENT,
            inherited_frame_count=1, inherited_frame_files_sha256=sealed, inherited_source_compatibility=compatibility))
    return parent, manifest, receipt, sealed


def review_evidence(folder):
    root = Path(folder)
    run = root/'output/runs/child'; run.mkdir(parents=True)
    reviews = root/'reviews/asein-integration'; reviews.mkdir(parents=True)
    validation = []
    for name in ('original-interval.json', 'feedback-interval.json'):
        path = reviews/name
        path.write_text(json.dumps(dict(kind='accepted_interval', name=name)))
        (run/name).write_bytes(path.read_bytes())
        validation.append(dict(path=name, sha256=verify.sha(path)))
    prior = dict(auxiliary_sources_sha256={'bounded_gravity.py': 'a'*64})
    current = dict(auxiliary_sources_sha256={'bounded_gravity.py': 'b'*64})
    controls = dict(force_limit_rifting={'enabled': True}, carrier_traction={'alpha': .1})
    delta = dict(changed_helpers={'bounded_gravity.py': dict(parent_sha256='a'*64, child_sha256='b'*64)},
        added_helpers={'effective_subduction_carrier_traction.py': 'c'*64}, removed_helpers=[])
    numerical = dict(kind='lite_feedback_numerical_contact_repair_review', version=1, ready=True,
        parent_run_id=verify.PARENT, parent_checkpoint_sha256='d'*64,
        helper='bounded_gravity.py', parent_sha256='a'*64, child_sha256='b'*64,
        physical_guards_unchanged=True, original_policy_interval_passed=True,
        feedback_policy_interval_passed=True, validation_receipts=validation)
    numerical_path = reviews/'lite-feedback-numerical-contact-repair-review.json'
    numerical_path.write_text(json.dumps(numerical))
    (run/'numerical-repair-review.json').write_bytes(numerical_path.read_bytes())
    source_review = dict(kind='lite_feedback_runtime_source_delta_review', version=1, ready=True,
        policy=controls, parent_run_id=verify.PARENT, parent_checkpoint_sha256='d'*64,
        boundary_kind='durable_stopped', parent_compatibility_sha256=verify.canonical_sha(prior),
        child_compatibility_sha256=verify.canonical_sha(current), **delta,
        numerical_repair_review_sha256=verify.sha(numerical_path))
    source_path = run/'source-delta-review.json'; source_path.write_text(json.dumps(source_review))
    receipt = dict(parent_run_id=verify.PARENT, parent_checkpoint_sha256='d'*64,
        parent_compatibility=prior, child_compatibility=current, source_delta=delta,
        source_delta_review_sha256=verify.sha(source_path),
        numerical_repair_review_sha256=verify.sha(numerical_path))
    return root, run, receipt, source_review, numerical, controls


class ReadOnlyContinuationEvidenceTests(unittest.TestCase):
    def test_parser_default_boundary_and_invalid_epoch_never_load_runtime(self):
        args = verify.parser().parse_args(['--run-id', 'child', '--preservation', 'preserved'])
        self.assertEqual((args.minimum_myr, args.parent_run_id, args.port), (44., verify.PARENT, 8769))
        for epoch in (43., float('nan'), float('inf')):
            args.minimum_myr = epoch
            with patch.object(verify, 'load_runtime') as runtime, self.assertRaises(ValueError):
                verify.audit(args)
            runtime.assert_not_called()

    def test_invalid_child_path_is_rejected_before_runtime_loading(self):
        for identity in ('../parent', verify.PARENT):
            args = verify.parser().parse_args(['--run-id', identity, '--preservation', 'preserved'])
            with patch.object(verify, 'load_runtime') as runtime, self.assertRaises(ValueError):
                verify.audit(args)
            runtime.assert_not_called()

    def test_sealed_reviews_bind_exact_policy_and_numerical_helper_delta(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence = review_evidence(folder)
            self.assertTrue(all(verify.review_checks(*evidence).values()))
            root, run, receipt, source, numerical, controls = evidence
            changed = deepcopy(source)
            changed['activated_controls'] = changed.pop('policy')
            self.assertFalse(verify.review_checks(root, run, receipt, changed, numerical, controls)
                ['source_delta_and_numerical_review_sealed'])
            changed = deepcopy(numerical)
            changed['parent_sha256'] = changed['child_sha256']
            self.assertFalse(verify.review_checks(root, run, receipt, source, changed, controls)
                ['exact_reviewed_numerical_parent_and_child_helper'])
            changed = deepcopy(receipt)
            changed['child_compatibility']['auxiliary_sources_sha256']['bounded_gravity.py'] = 'f'*64
            self.assertFalse(verify.review_checks(root, run, changed, source, numerical, controls)
                ['exact_reviewed_numerical_parent_and_child_helper'])

    def test_review_requires_distinct_unchanged_validation_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root, run, receipt, source, numerical, controls = review_evidence(folder)
            changed = deepcopy(numerical)
            changed['validation_receipts'][1] = deepcopy(changed['validation_receipts'][0])
            self.assertFalse(verify.review_checks(root, run, receipt, source, changed, controls)
                ['retained_original_and_feedback_validation_receipts_exact'])
            (root/'reviews/asein-integration/feedback-interval.json').write_text('{}')
            self.assertFalse(verify.review_checks(root, run, receipt, source, numerical, controls)
                ['retained_original_and_feedback_validation_receipts_exact'])

    def test_unrelated_root_review_cannot_shadow_child_retained_proofs(self):
        with tempfile.TemporaryDirectory() as folder:
            root, run, receipt, source, numerical, controls = review_evidence(folder)
            root_review = root/'reviews/asein-integration/lite-feedback-numerical-contact-repair-review.json'
            root_review.write_text(json.dumps(dict(kind='a_later_different_recovery')))
            (root/'reviews/asein-integration/original-interval.json').write_text('{}')
            (root/'reviews/asein-integration/feedback-interval.json').write_text('{}')
            self.assertTrue(verify.review_checks(root, run, receipt, source, numerical, controls)
                ['retained_original_and_feedback_validation_receipts_exact'])
            (run/'feedback-interval.json').write_text('{}')
            self.assertFalse(verify.review_checks(root, run, receipt, source, numerical, controls)
                ['retained_original_and_feedback_validation_receipts_exact'])

    def test_runtime_loader_blocks_accidental_constructor_and_step(self):
        class FakeSimulation:
            def __init__(self):
                raise RuntimeError('Original constructor must never run.')
            def step(self):
                raise RuntimeError('Original step must never run.')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            base = SimpleNamespace(ROOT=root, native_engine=SimpleNamespace(Simulation=FakeSimulation))
            def module(name):
                return base if name == 'lite_checkpoint_verification' else SimpleNamespace()
            with patch.object(verify.importlib, 'import_module', side_effect=module), \
                    patch.object(verify.sys, 'path', []), patch.dict(verify.os.environ):
                actual, _ = verify.load_runtime(root)
                self.assertIs(actual, base)
                with self.assertRaisesRegex(AssertionError, 'forbids constructors'):
                    FakeSimulation()
                subject = FakeSimulation.__new__(FakeSimulation)
                with self.assertRaisesRegex(AssertionError, 'forbids constructors'):
                    subject.step()

    def test_sorted_json_controls_normalize_before_strict_zero_policy_proof(self):
        # Constructorless metadata fixtures reproduce the producer's strict
        # typed-dict ordering contract without loading or advancing a world.
        import lite_feedback_transition as transition
        from tests.test_lite_feedback_transition import FractureFixture, CarrierFixture
        ordered = dict(force_limit_rifting=FractureFixture.normalize(),
                       carrier_traction=CarrierFixture.normalize())
        raw = json.loads(json.dumps(ordered, sort_keys=True))
        self.assertEqual(raw, ordered)
        self.assertNotEqual(list(raw['carrier_traction']), list(ordered['carrier_traction']))
        normalized = verify.normalize_controls(raw, FractureFixture, CarrierFixture)
        self.assertEqual(list(normalized['carrier_traction']), list(ordered['carrier_traction']))
        original = SimpleNamespace(t=43., steps=43,
            config=dict(force_limit_rifting=dict(enabled=False), effective_subduction=dict(enabled=True)),
            effective_subduction_initiation=dict(activation_myr=25., epoch_myr=43., births=[]))
        with patch.object(transition, '_policy_apis', return_value=(FractureFixture, CarrierFixture)):
            child, _ = transition._activate(original, normalized)
            proof = transition._check_activation(original, child, normalized)
        self.assertTrue(proof['inherited_state_exact'])
        self.assertEqual(child.force_rifting_state, {})
        self.assertEqual(child.force_rifting_diagnostics, dict(version=1, checks=0, commits=0, refusals=[]))
        self.assertEqual(child.effective_subduction_carrier_traction['retrospective_work_j'], 0.)
        self.assertEqual(child.force_rifting_policy['activation_myr'], child.force_rifting_policy['epoch_myr'])

    def test_inherited_bytes_and_nested_source_provenance_are_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, manifest, receipt, sealed = history(folder)
            checks, report = verify.history_checks(Path(folder), parent, manifest, receipt, sealed)
            self.assertTrue(all(checks.values()), checks)
            self.assertEqual(report['maximum_recursive_source_depth'], 3)
            changed = deepcopy(manifest)
            changed['frame_sources'][0]['parent_frame_source']['parent_frame_source']['run_id'] = 'wrong'
            checks, _ = verify.history_checks(Path(folder), parent, changed, receipt, sealed)
            self.assertFalse(checks['recursive_frame_provenance'])
            (Path(folder)/'frame_0000.npz').write_bytes(b'changed inherited data')
            checks, _ = verify.history_checks(Path(folder), parent, manifest, receipt, sealed)
            self.assertFalse(checks['inherited_frame_bytes'])

    def test_future_frame_rejects_old_epoch_and_policy_validation_error(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, manifest, receipt, sealed = history(folder)
            def reject(frame):
                raise ValueError('tampered policy')
            checks, _ = verify.history_checks(Path(folder), parent, manifest, receipt, sealed, frame_validators=(reject,))
            self.assertFalse(checks['future_frame_policy_receipts_valid'])
            changed = deepcopy(manifest); changed['frames'][1]['time_myr'] = 43.
            (Path(folder)/'frame_0001.json').write_text(json.dumps(dict(index=1, time_myr=43.)))
            checks, _ = verify.history_checks(Path(folder), parent, changed, receipt, sealed)
            self.assertFalse(checks['future_frames_after_policy_boundary'])

    def test_scheduler_uses_embedded_44_not_stale_external_26(self):
        state = SimpleNamespace(t=43., config=dict(duration_myr=1000., snapshot_myr=2.))
        receipt = dict(next_output_myr=44., durable_recovery=dict(embedded_next_output_myr=44.,
            external_stale_next_output_myr=26., historical_manifests_rewritten=False))
        self.assertTrue(all(verify.scheduler_checks(state, dict(next_output_myr=44.), receipt).values()))
        state.t = 44.
        self.assertTrue(all(verify.scheduler_checks(state, dict(next_output_myr=46.), receipt).values()))
        bad = verify.scheduler_checks(state, dict(next_output_myr=26.), receipt)
        self.assertFalse(bad['valid_durable_output_scheduler'])
        self.assertFalse(bad['original_repeated_addition_scheduler'])
        state.t = 1000.
        self.assertTrue(all(verify.scheduler_checks(state, dict(next_output_myr=1000.), receipt).values()))

    def test_file_map_reports_missing_and_changed_files(self):
        report = verify.file_differences({'a': 'old', 'b': 'same'}, {'a': 'new', 'b': 'same', 'c': 'extra'})
        self.assertFalse(report['equal'])
        self.assertEqual(report['first_mismatches'], ['a', 'c'])
        self.assertEqual(report['mismatch_count'], 2)

    def test_checkpoint_race_writes_truthful_retry_receipt_without_model_actions(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'receipt.json'
            args = ['verify', '--run-id', 'child', '--preservation', folder, '--output', str(output)]
            with patch.object(verify.sys, 'argv', args), patch.object(verify, 'audit',
                    side_effect=RuntimeError('Checkpoint changed during the bounded read')), \
                    redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
                verify.main()
            self.assertEqual(error.exception.code, 2)
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt['state'], 'retry_later')
            self.assertFalse(receipt['constructor_called'])
            self.assertFalse(receipt['simulation_stepped'])
            self.assertFalse(receipt['server_mutated'])


if __name__ == '__main__':
    unittest.main()
