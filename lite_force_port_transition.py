"""Explicit durable-stopped source transition for a fractional force-port fix.

No model constructor, step, upgrade or boundary rebuild runs here. Every typed
state field remains exact, including the accepted geometry cache. The first
successor force solve uses that inherited source-epoch cache. The reviewed
ledger helper is used only when successor evolution requests its next export.
Old frames and solver reports remain predecessor evidence. Root seals new
evidence before publication.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import uuid

import numpy as np

import checkpoint
from lite_initiation_transition import _fingerprint
import native_engine
import server

ROOT = Path(__file__).resolve().parent
KIND = 'lite_fractional_force_port_source_transition'
SOURCE_REVIEW_KIND = 'lite_fractional_force_port_runtime_source_delta_review'
REPAIR_REVIEW_KIND = 'lite_fractional_force_port_numerical_repair_review'
AUTHORITY_KIND = 'explicit_user_asein_lite_implementation_repair_authorization'
VERSION = 1
EXPECTED_HELPERS = 173
HELPER = 'balance_force_ledger.py'
PARENT_SOURCE_COMMIT = '09aeb192589eb85c74269fc828b5060e57cab30c'
PARENT_HELPER_SHA256 = '79764838eb55779d628223a63dc4eab875b74cfa76b86fedb70bc9084e5804a6'
CHILD_HELPER_SHA256 = '33d1b830935190a9331928c8f7b14d0db1fdea6bffc8e917284bbf9008c8750e'


@dataclass(frozen=True)
class TransitionApproval:
    parent_run_id: str
    parent_checkpoint_sha256: str
    parent_preservation_receipt: Path
    parent_preservation_sha256: str
    runtime_source_seal: Path
    runtime_source_seal_sha256: str
    source_delta_receipt: Path
    source_delta_receipt_sha256: str
    repair_review_receipt: Path
    repair_review_sha256: str
    authority_receipt: Path
    authority_sha256: str
    stop_evidence_receipt: Path
    stop_evidence_sha256: str
    review_ready: bool = False


# Earlier healthy-boundary/physical-policy approvals cannot authorize this repair.
# Root installs one immutable, ready approval only after the captured regression,
# guarded 64-to-65 interval and exact source/error/preservation review are sealed.
REVIEWED_APPROVAL: TransitionApproval | None = TransitionApproval(
    parent_run_id='20261002-174700-59dde9',
    parent_checkpoint_sha256='f16b11de04ad723792cb158c22fa46b36662d7703d3ed81ebe838a03bccfd46f',
    parent_preservation_receipt=ROOT.parent / 'Deep-Time-asein-lite-20261002/reviews/asein-integration/preserved-stop-20261002-174700-59dde9-64myr/preservation.json',
    parent_preservation_sha256='27cb4c8358df4f9a1e76c6f9bb72da725d2cece2e8bd0bbe905f4e6522848a1a',
    runtime_source_seal=ROOT / 'reviews/asein-integration/lite-fractional-force-port-runtime-source-seal.json',
    runtime_source_seal_sha256='5484943a780d95684ff9d79aeb460381d99eac19464e6347460771243e7d09ca',
    source_delta_receipt=ROOT / 'reviews/asein-integration/lite-fractional-force-port-runtime-source-delta-review.json',
    source_delta_receipt_sha256='217ad045da2f61fda8800cd73bf2525dce0ac05d17cc4529c07e08a8c439b0e1',
    repair_review_receipt=ROOT / 'reviews/asein-integration/lite-fractional-force-port-numerical-repair-review.json',
    repair_review_sha256='f643bae26ee2d0c692e249d6cd753c08c4c2cb8a998ee403c8c360be430b6004',
    authority_receipt=ROOT / 'reviews/asein-integration/lite-fractional-force-port-human-authorization.json',
    authority_sha256='5a49430471c70c6fd579e6740595c41583404d9062879cfae7e81775e936384f',
    stop_evidence_receipt=ROOT / 'reviews/asein-integration/lite-fractional-force-port-stopped-status.json',
    stop_evidence_sha256='19e1d4370f36d3bc47ddc934bb0399f059b79446d1186284e6d2f8bc8fa62392',
    review_ready=True)


def _require(condition, message):
    if not condition:
        raise ValueError('Lite fractional force-port source transition: ' + message)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path):
    return _sha(Path(path).read_bytes())


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _write_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + '.tmp')
    temporary.write_bytes(_json_bytes(value) + b'\n')
    temporary.replace(path)


def _files(folder):
    result = {}
    for path in sorted(Path(folder).rglob('*')):
        _require(not path.is_symlink(), 'symlinks cannot form a sealed source history.')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = _file_sha(path)
    return result


def _read_evidence(path, sha256, description):
    path = Path(path)
    _require(path.is_file() and _file_sha(path) == sha256, description + ' changed or is missing.')
    return json.loads(path.read_text(encoding='utf-8'))


def _approved(approval=None):
    selected = REVIEWED_APPROVAL if approval is None else approval
    _require(isinstance(selected, TransitionApproval) and selected is REVIEWED_APPROVAL and
        selected.review_ready is True, 'new exact stopped boundary, guarded repair and existing user authority are not sealed yet.')
    _require(isinstance(selected.parent_run_id, str) and
        re.fullmatch(r'[A-Za-z0-9_-]{1,80}', selected.parent_run_id), 'approved parent identity is invalid.')
    for name in ('parent_checkpoint_sha256', 'parent_preservation_sha256', 'runtime_source_seal_sha256',
            'source_delta_receipt_sha256', 'repair_review_sha256', 'authority_sha256', 'stop_evidence_sha256'):
        value = getattr(selected, name)
        _require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value), 'approved evidence hash is invalid.')
    _require(isinstance(CHILD_HELPER_SHA256, str) and re.fullmatch(r'[0-9a-f]{64}', CHILD_HELPER_SHA256),
        'final independently reviewed helper bytes are not sealed yet.')
    return selected


def _current_context(approved):
    expected = server.SimulationManager.compatibility()
    seal = _read_evidence(approved.runtime_source_seal, approved.runtime_source_seal_sha256,
        'independently frozen runtime source seal')
    _require(seal == expected['auxiliary_sources_sha256'] and len(seal) == EXPECTED_HELPERS,
        'current runtime differs from the exact 173-helper source seal.')
    _require(_file_sha(ROOT / 'tectonics.py') == expected['engine_sha256'] and
        all(_file_sha(ROOT / name) == value for name, value in seal.items()),
        'runtime files changed after local source capture; reload only after source freeze.')
    return expected, server.ENGINE_SOURCE, dict(server.AUXILIARY_SOURCES)


def _check_reviews(approved):
    authority = _read_evidence(approved.authority_receipt, approved.authority_sha256,
        'existing human implementation-repair authority')
    _require(authority.get('kind') == AUTHORITY_KIND and authority.get('version') == VERSION and
        authority.get('parent_run_id') == approved.parent_run_id and
        authority.get('source_commit') == PARENT_SOURCE_COMMIT and authority.get('authorized') is True and
        authority.get('parent_checkpoint_sha256') == approved.parent_checkpoint_sha256 and
        authority.get('new_physical_policy_authorized') is False and
        isinstance(authority.get('authorization_source'), str) and bool(authority['authorization_source'].strip()) and
        authority.get('instruction') == 'If unexpectedly stopped, preserve manifest/source closure/checkpoint/frames/error; '
            'diagnose implementation bugs within authorized laws, add meaningful regression coverage and relevant checks, '
            'and resume only through truthful supported explicit source transitions or preserved fresh audited histories. '
            'Never weaken guards, forge compatibility or repeatedly retry an unfixed failure.' and
        authority.get('scope') == 'Repair only the numerical finite-force ledger representation of existing fractional native '
            'support; preserve forces, coefficients, resistance, physical laws, material/ownership, guards, all inherited typed '
            'state, history and scheduling. Validate before truthful explicit source-only stopped-checkpoint continuation.',
        'existing user authority does not cover this exact numerical stopped-source repair.')
    repair = _read_evidence(approved.repair_review_receipt, approved.repair_review_sha256,
        'completed new fractional force-port numerical repair review')
    _require(repair.get('kind') == REPAIR_REVIEW_KIND and repair.get('version') == VERSION and
        repair.get('ready') is True and repair.get('parent_run_id') == approved.parent_run_id and
        repair.get('parent_checkpoint_sha256') == approved.parent_checkpoint_sha256 and
        repair.get('helper') == HELPER and repair.get('parent_sha256') == PARENT_HELPER_SHA256 and
        repair.get('child_sha256') == CHILD_HELPER_SHA256 and repair.get('physical_guards_unchanged') is True and
        repair.get('captured_regression_passed') is True and repair.get('future_interval_passed') is True and
        repair.get('physical_laws_unchanged') is True and repair.get('virtual_work_preserved') is True and
        repair.get('stop_evidence_sha256') == approved.stop_evidence_sha256,
        'new repair review must prove regression and future interval under unchanged laws/guards and exact sources.')
    evidence = repair.get('validation_receipts')
    _require(isinstance(evidence, list) and len(evidence) >= 2, 'repair review needs separate regression and interval receipts.')
    seen = set()
    for row in evidence:
        _require(type(row) is dict and set(row) == {'path', 'sha256'} and
            isinstance(row['path'], str) and isinstance(row['sha256'], str) and
            re.fullmatch(r'[0-9a-f]{64}', row['sha256']), 'repair evidence needs exact paths and hashes.')
        path = Path(row['path'])
        path = (path if path.is_absolute() else Path(approved.repair_review_receipt).parent / path).resolve()
        _require(path not in seen, 'repair evidence must name distinct receipts.')
        seen.add(path)
        _read_evidence(path, row['sha256'], 'new force-port validation evidence')
    return authority, repair


def _check_source(parent, recorded, expected, engine, helpers, approved):
    _require(_file_sha(parent / 'engine.py') == recorded['engine_sha256'] == expected['engine_sha256'] == _sha(engine),
        'engine entrypoint changes are outside this source transition.')
    for key in ('numpy_version', 'flow_backend', 'material_backend'):
        _require(recorded.get(key) == expected.get(key), key + ' changed outside this repair.')
    prior = recorded['auxiliary_sources_sha256']
    _require(len(prior) == EXPECTED_HELPERS and all(Path(name).name == name and name.endswith('.py') for name in prior),
        'parent source closure is not the exact 173-helper local closure.')
    _require(all(_file_sha(parent / name) == value for name, value in prior.items()), 'parent helper source is damaged.')
    captured = server.capture_auxiliary_sources((parent / 'engine.py').read_bytes(), root=parent)
    _require({name: _sha(payload) for name, payload in captured.items()} == prior,
        'parent recorded closure is not its complete saved import closure.')
    current = {name: _sha(payload) for name, payload in helpers.items()}
    _require(current == expected['auxiliary_sources_sha256'], 'current capture and compatibility differ.')
    changed = {name: dict(parent_sha256=prior[name], child_sha256=current[name])
        for name in prior.keys() & current.keys() if prior[name] != current[name]}
    required = {HELPER: dict(parent_sha256=PARENT_HELPER_SHA256, child_sha256=CHILD_HELPER_SHA256)}
    _require(set(current) == set(prior) and changed == required,
        'only the exact reviewed force-port ledger helper delta is supported; no same-source or additional helper changes.')
    review = _read_evidence(approved.source_delta_receipt, approved.source_delta_receipt_sha256,
        'new runtime source delta review')
    _require(review.get('kind') == SOURCE_REVIEW_KIND and review.get('version') == VERSION and
        review.get('ready') is True and review.get('parent_run_id') == approved.parent_run_id and
        review.get('parent_checkpoint_sha256') == approved.parent_checkpoint_sha256 and
        review.get('parent_compatibility_sha256') == _sha(_json_bytes(recorded)) and
        review.get('child_compatibility_sha256') == _sha(_json_bytes(expected)) and
        review.get('changed_helpers') == required and review.get('added_helpers') == {} and
        review.get('removed_helpers') == [] and review.get('helper_count') == EXPECTED_HELPERS and
        review.get('repair_review_sha256') == approved.repair_review_sha256 and
        review.get('boundary_kind') == 'durable_stopped' and review.get('stop_evidence_sha256') == approved.stop_evidence_sha256 and
        review.get('authority_sha256') == approved.authority_sha256 and review.get('configuration_unchanged') is True and
        review.get('state_import') == 'entire typed state exact; cached geometry retained',
        'new source delta review does not bind this exact stopped boundary, repair, authority and full-state import.')
    return dict(changed_helpers=changed, added_helpers={}, removed_helpers=[])


def _check_stop(parent, original, saved, external, approved, header, stop):
    _require(saved.get('run_id') == external.get('run_id') == approved.parent_run_id,
        'parent identity differs from the reviewed recovery.')
    epoch = float(original.t)
    # The complete accepted generation remains interrupted. The outer manifest
    # records the failed trial; neither predecessor generation is rewritten.
    _require(saved.get('state') == 'interrupted' and saved.get('error') is None and
        header['time_myr'] == saved.get('time_myr') == saved.get('integration_time_myr') ==
        saved.get('checkpoint_time_myr') == epoch,
        'stopped recovery needs the exact valid durable embedded checkpoint generation.')
    _require(external.get('state') == 'error' and isinstance(external.get('error'), str) and bool(external['error'].strip()) and
        external.get('integration_time_myr') == external.get('checkpoint_time_myr') == epoch,
        'external stopped manifest must retain its failure and exact durable integration epoch.')
    _require(isinstance(stop, dict) and stop.get('state') == 'error' and stop.get('run_id') == approved.parent_run_id and
        stop.get('error') == external['error'] and stop.get('integration_time_myr') == stop.get('checkpoint_time_myr') == epoch and
        stop.get('time_myr') == external.get('time_myr') and stop.get('frame_count') == external.get('frame_count') and
        stop.get('next_output_myr') == external.get('next_output_myr'),
        'sealed stop evidence differs from the failed external generation.')
    _require(_fingerprint(saved.get('config')) == _fingerprint(external.get('config')) == _fingerprint(original.config),
        'parent configurations differ; this source-only transition cannot change controls.')
    _require(json.loads((parent / 'config.json').read_text()) == original.config, 'saved config file differs from typed state.')
    for manifest in (saved, external):
        checkpoint.verify_compatibility(manifest, header['compatibility'])
    count, frames = saved.get('frame_count'), saved.get('frames', [])
    _require(type(count) is int and count > 0 and len(frames) == count and
        external.get('frames') == frames and external.get('frame_count') == count, 'parent accepted frame history differs.')
    _require(external.get('time_myr') == frames[-1].get('time_myr') and external['time_myr'] <= epoch,
        'stopped public epoch must name the last saved accepted frame, not a future trial.')
    _require(saved.get('source_transition') == external.get('source_transition') and
        saved.get('frame_sources') == external.get('frame_sources'), 'parent recursive source lineage differs.')
    pending = saved.get('next_output_myr')
    _require(type(pending) in (float, int) and np.isfinite(pending) and epoch < pending <= original.config['duration_myr'],
        'exact embedded durable output scheduler is missing or invalid.')
    # External next_output_myr may still name the prior source-boundary value.
    # Preserve it as error evidence; only embedded accepted scheduling is used.
    for index, row in enumerate(frames):
        stem = parent / f'frame_{index:04d}'
        _require(stem.with_suffix('.npz').is_file() and stem.with_suffix('.json').is_file(), 'an inherited frame file is missing.')
        metadata = json.loads(stem.with_suffix('.json').read_text())
        _require(row.get('index') == metadata.get('index') == index and row.get('time_myr') == metadata.get('time_myr') and
            row['time_myr'] <= epoch, 'an inherited frame disagrees with its exact history.')


def _derive(original):
    return deepcopy(original)


@dataclass
class TransitionPlan:
    parent: Path
    original: object
    derived: object
    manifest: dict
    receipt: dict
    parent_files: dict
    compatibility: dict
    engine_source: bytes
    helper_sources: dict
    approval: TransitionApproval


def plan_transition(parent, *, approval=None):
    """Validate an exact preserved durable stop; no writes and no state rebuild."""
    approved = _approved(approval)
    parent = Path(parent).resolve()
    _require(parent.is_dir(), 'parent history directory is missing.')
    before = _files(parent)
    _require(before.get('checkpoint.npz') == approved.parent_checkpoint_sha256, 'parent checkpoint differs from reviewed stopped boundary.')
    preserved = _read_evidence(approved.parent_preservation_receipt, approved.parent_preservation_sha256,
        'new immutable parent preservation receipt')
    _require(preserved.get('kind') == 'asein_lite_complete_failed_history_preservation' and
        preserved.get('run_id') == approved.parent_run_id and preserved.get('all_copied_files_exact') is True and
        preserved.get('checkpoint_sha256') == approved.parent_checkpoint_sha256 and preserved.get('file_count') == len(before) and
        len(preserved['files']) == len(before) and
        {row['path']: row['sha256'] for row in preserved['files']} == before, 'parent files differ from immutable failed-history preservation.')
    authority, repair = _check_reviews(approved)
    header = checkpoint.checkpoint_header(parent / 'checkpoint.npz')
    expected, engine, helpers = _current_context(approved)
    delta = _check_source(parent, header['compatibility'], expected, engine, helpers, approved)
    original, saved = checkpoint.read_checkpoint(parent / 'checkpoint.npz', header['compatibility'], native_engine.Simulation)
    external = json.loads((parent / 'manifest.json').read_text())
    stop = _read_evidence(approved.stop_evidence_receipt, approved.stop_evidence_sha256, 'sealed stopped status evidence')
    _check_stop(parent, original, saved, external, approved, header, stop)
    _require(preserved.get('checkpoint_time_myr') == float(original.t) and
        preserved.get('public_frame_myr') == external['time_myr'] and preserved.get('error') == external['error'],
        'failed-history preservation does not bind the accepted epoch and external error.')
    derived = _derive(original)
    fingerprint = _fingerprint(vars(original))
    _require(_fingerprint(vars(derived)) == fingerprint, 'source import changed a typed array, config, RNG, cache, clock or history field.')
    _require(_files(parent) == before, 'parent changed while planning; wait for the exact stable stopped boundary.')
    receipt = dict(kind=KIND, version=VERSION, parent_run_id=approved.parent_run_id, epoch_myr=float(original.t),
        integration_steps=int(original.steps), parent_checkpoint_sha256=before['checkpoint.npz'],
        parent_manifest_sha256=before['manifest.json'], parent_files_sha256=before, parent_file_count=len(before),
        parent_preservation_receipt_sha256=approved.parent_preservation_sha256,
        runtime_source_seal_sha256=approved.runtime_source_seal_sha256,
        source_delta_review_sha256=approved.source_delta_receipt_sha256,
        repair_review_sha256=approved.repair_review_sha256, authority_sha256=approved.authority_sha256,
        boundary_kind='durable_stopped', stop_evidence_sha256=approved.stop_evidence_sha256,
        predecessor_embedded_state=saved['state'], predecessor_external_state=external['state'],
        predecessor_error=external['error'], predecessor_public_time_myr=external['time_myr'],
        predecessor_external_next_output_myr=external.get('next_output_myr'), scheduler_authority='embedded durable predecessor checkpoint',
        parent_compatibility=header['compatibility'], child_compatibility=expected, source_delta=delta,
        full_state_fingerprint=fingerprint['sha256'], preexisting_array_count=len(fingerprint['arrays']),
        entire_typed_state_exact=True, configuration_unchanged=True, rng_unchanged=True,
        all_existing_policy_clocks_and_history_unchanged=True, initial_file_sha256=before['initial.json'],
        next_output_myr=saved['next_output_myr'], inherited_frame_count=saved['frame_count'],
        cached_boundary_geometry_unchanged=True, geometry_rebuilt_at_import=False,
        first_source_solve_geometry='Accepted predecessor cache retained exactly; reviewed ledger used only on successor export.',
        physical_time_advanced_myr=0., model_constructed=False, model_stepped=False,
        historical_frames_rewritten=False, physical_policies_added=False,
        interpretation='Separate stopped-source recovery boundary for numerical fractional force-port representation repair. '
            'Inherited solver reports and frames remain predecessor evidence until future accepted successor evolution.')
    return TransitionPlan(parent, original, derived, deepcopy(saved), receipt, before, expected, engine, helpers, approved)


def _retain_proofs(approved, child):
    _, repair = _check_reviews(approved)
    reserved = {'manifest.json', 'source-transition.json', 'checkpoint.npz', 'checkpoint.source-boundary.npz',
        'config.json', 'initial.json', 'engine.py', 'inherited-parent', 'parent-preservation.json',
        'source-delta-review.json', 'runtime-source-seal.json', 'repair-review.json', 'application-authority.json', 'stop-evidence.json'}
    reserved |= {name + '.tmp' for name in reserved}
    mappings = []
    for row in repair['validation_receipts']:
        reference = Path(row['path'])
        source = (reference if reference.is_absolute() else Path(approved.repair_review_receipt).parent / reference).resolve()
        item = dict(reference_path=row['path'], source_path=str(source), sha256=row['sha256'], storage='external_reference', child_file=None)
        if not reference.is_absolute() and reference.name == row['path'] and reference.name not in ('.', '..'):
            namespace = reference.name.casefold()
            managed = re.fullmatch(r'(?:frame_\d+\.(?:json|npz)|checkpoint(?:\.[a-z0-9_-]+)*\.npz)(?:\.tmp)?', namespace)
            target = child / reference.name
            _require(namespace not in reserved and managed is None and not target.exists(), 'validation proof basename collides with child history.')
            shutil.copy2(source, target)
            _require(_file_sha(target) == row['sha256'], 'retained validation proof differs from sealed evidence.')
            item.update(storage='child_relative_exact_copy', child_file=reference.name)
        mappings.append(item)
    return mappings


def _verify_child_artifacts(plan, child, manifest, receipt):
    """Re-read copied bytes and both checkpoints before the public manifest."""
    approved = plan.approval
    expected = {'engine.py': plan.compatibility['engine_sha256'],
        **plan.compatibility['auxiliary_sources_sha256'],
        'initial.json': plan.parent_files['initial.json'], 'config.json': plan.parent_files['config.json'],
        'parent-preservation.json': approved.parent_preservation_sha256,
        'runtime-source-seal.json': approved.runtime_source_seal_sha256,
        'source-delta-review.json': approved.source_delta_receipt_sha256,
        'repair-review.json': approved.repair_review_sha256,
        'application-authority.json': approved.authority_sha256,
        'stop-evidence.json': approved.stop_evidence_sha256,
        'checkpoint.npz': receipt['child_checkpoint_sha256'],
        'checkpoint.source-boundary.npz': receipt['child_checkpoint_sha256']}
    expected.update({name: value for name, value in plan.parent_files.items()
        if re.fullmatch(r'frame_\d+\.(npz|json)', name)})
    expected.update({row['child_file']: row['sha256'] for row in receipt['validation_evidence']
        if row['child_file'] is not None})
    for name, value in expected.items():
        path = child / name
        _require(path.is_file() and _file_sha(path) == value,
            'child artifact differs from sealed bytes before publication: ' + name)
    _require(_files(child / 'inherited-parent') == plan.parent_files,
        'copied complete parent history changed before publication.')
    _require(receipt['permanent_source_boundary_sha256'] == receipt['child_checkpoint_sha256'],
        'permanent source boundary is not an exact copy of the validated checkpoint.')
    for name in ('checkpoint.npz', 'checkpoint.source-boundary.npz'):
        restored, saved = checkpoint.read_checkpoint(child / name, plan.compatibility, native_engine.Simulation)
        _require(_fingerprint(vars(restored)) == _fingerprint(vars(plan.derived)) and saved == manifest and
            saved['next_output_myr'] == plan.manifest['next_output_myr'],
            'child checkpoint readback changed full state or exact scheduler/history: ' + name)


def create_transition(plan, *, output_root=None, run_id=None):
    """Publish a separately compatible paused child; manifest commits last."""
    _require(isinstance(plan, TransitionPlan), 'an independently reviewed transition plan is required.')
    approved = _approved(plan.approval)
    fresh = plan_transition(plan.parent, approval=approved)
    _require(_fingerprint(vars(plan.original)) == _fingerprint(vars(fresh.original)) and
        _fingerprint(vars(plan.derived)) == _fingerprint(vars(fresh.derived)) and plan.manifest == fresh.manifest and
        plan.receipt == fresh.receipt and plan.parent_files == fresh.parent_files and
        plan.compatibility == fresh.compatibility and plan.engine_source == fresh.engine_source and
        plan.helper_sources == fresh.helper_sources, 'mutable plan state/source/provenance changed after validation.')
    output_root = Path(output_root or plan.parent.parent).resolve()
    run_id = run_id or datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    _require(isinstance(run_id, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id), 'invalid child identity.')
    child = output_root / run_id
    _require(not child.exists() and not child.is_relative_to(plan.parent) and not plan.parent.is_relative_to(child),
        'child already exists or is not disjoint from parent history.')
    manifest = deepcopy(plan.manifest)
    manifest.update(run_id=run_id, created=datetime.now(timezone.utc).isoformat(), state='paused', can_resume=True,
        branch_origin=dict(run_id=approved.parent_run_id, time_myr=plan.receipt['epoch_myr'],
            checkpoint_file='inherited-parent/checkpoint.npz', checkpoint_sha256=plan.receipt['parent_checkpoint_sha256'], source_transition_kind=KIND),
        source_transition=dict(kind=KIND, version=VERSION, receipt='source-transition.json', epoch_myr=plan.receipt['epoch_myr'],
            parent_run_id=approved.parent_run_id, inherited_frame_count=plan.manifest['frame_count'],
            inherited_frame_files_sha256={name: value for name, value in plan.parent_files.items() if re.fullmatch(r'frame_\d+\.(npz|json)', name)},
            inherited_source_compatibility=deepcopy(plan.receipt['parent_compatibility']),
            boundary_kind='durable_stopped',
            scope='One numerical force-port helper delta; entire typed state, policies and geometry cache remain exact.'),
        durable_source_recovery=dict(kind='explicit_valid_durable_stopped_source_transition',
            parent_run_id=approved.parent_run_id, epoch_myr=plan.receipt['epoch_myr'],
            error_evidence_file='stop-evidence.json', error_evidence_sha256=approved.stop_evidence_sha256,
            numerical_repair_review_file='repair-review.json', numerical_repair_review_sha256=approved.repair_review_sha256,
            scheduler_authority='embedded durable predecessor checkpoint'),
        frame_sources=[dict(index=row['index'], time_myr=row['time_myr'], run_id=approved.parent_run_id,
            source_manifest='inherited-parent/manifest.json', source_manifest_sha256=plan.receipt['parent_manifest_sha256'],
            engine_sha256=plan.receipt['parent_compatibility']['engine_sha256'],
            auxiliary_sources_sha256=deepcopy(plan.receipt['parent_compatibility']['auxiliary_sources_sha256']),
            files_sha256={suffix: plan.parent_files[f"frame_{row['index']:04d}{suffix}"] for suffix in ('.npz', '.json')},
            inherited=True, parent_frame_source=deepcopy(plan.manifest.get('frame_sources', [])[index])
                if index < len(plan.manifest.get('frame_sources', [])) else None)
            for index, row in enumerate(plan.manifest['frames'])], **deepcopy(plan.compatibility))
    for name in ('error', 'resume_reason', 'recovery'):
        manifest.pop(name, None)
    child.mkdir(parents=False)
    shutil.copytree(plan.parent, child / 'inherited-parent')
    _require(_files(child / 'inherited-parent') == plan.parent_files, 'copied complete parent history differs.')
    for source, name in ((approved.parent_preservation_receipt, 'parent-preservation.json'),
            (approved.runtime_source_seal, 'runtime-source-seal.json'), (approved.source_delta_receipt, 'source-delta-review.json'),
            (approved.repair_review_receipt, 'repair-review.json'), (approved.authority_receipt, 'application-authority.json'),
            (approved.stop_evidence_receipt, 'stop-evidence.json')):
        shutil.copy2(source, child / name)
    (child / 'engine.py').write_bytes(plan.engine_source)
    for name, payload in plan.helper_sources.items():
        (child / name).write_bytes(payload)
    for name in ('initial.json', 'config.json'):
        shutil.copy2(plan.parent / name, child / name)
    for index in range(plan.manifest['frame_count']):
        for suffix in ('.npz', '.json'):
            name = f'frame_{index:04d}{suffix}'
            shutil.copy2(plan.parent / name, child / name)
            _require(_file_sha(child / name) == plan.parent_files[name], 'inherited frame bytes changed during copying.')
    checkpoint.write_checkpoint(child / 'checkpoint.npz', plan.derived, manifest, plan.compatibility)
    restored, saved = checkpoint.read_checkpoint(child / 'checkpoint.npz', plan.compatibility, native_engine.Simulation)
    _require(_fingerprint(vars(restored)) == _fingerprint(vars(plan.derived)) and saved == manifest and
        saved['next_output_myr'] == plan.manifest['next_output_myr'], 'child readback changed full state or exact scheduler/history.')
    child_checkpoint_sha256 = _file_sha(child / 'checkpoint.npz')
    permanent = child / 'checkpoint.source-boundary.npz'
    shutil.copy2(child / 'checkpoint.npz', permanent)
    receipt = deepcopy(plan.receipt)
    receipt.update(child_run_id=run_id, child_checkpoint_sha256=child_checkpoint_sha256,
        permanent_source_boundary_file=permanent.name, permanent_source_boundary_sha256=_file_sha(permanent),
        inherited_parent_copy_exact=True, child_decoded_state_exact=True, published_state='paused',
        validation_evidence=_retain_proofs(approved, child))
    _require(_files(plan.parent) == plan.parent_files, 'parent changed before publication.')
    _require(_current_context(approved) == (plan.compatibility, plan.engine_source, plan.helper_sources), 'runtime changed before publication.')
    _check_reviews(approved)
    _read_evidence(approved.parent_preservation_receipt, approved.parent_preservation_sha256, 'parent preservation receipt')
    _read_evidence(approved.source_delta_receipt, approved.source_delta_receipt_sha256, 'source delta review')
    _read_evidence(approved.stop_evidence_receipt, approved.stop_evidence_sha256, 'sealed stopped status evidence')
    _write_json(child / 'source-transition.json', receipt)
    _require(_file_sha(child / 'source-transition.json') == _sha(_json_bytes(receipt) + b'\n') and
        json.loads((child / 'source-transition.json').read_text(encoding='utf-8')) == receipt,
        'child source-transition receipt readback differs before publication.')
    _verify_child_artifacts(plan, child, manifest, receipt)
    _write_json(child / 'manifest.json', manifest)
    return child, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--create', action='store_true')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    plan = plan_transition(args.parent)
    receipt, child = plan.receipt, None
    if args.create:
        child, receipt = create_transition(plan, output_root=args.output_root, run_id=args.run_id)
    if args.receipt:
        _write_json(args.receipt, receipt)
    print(json.dumps(dict(kind=KIND, state='paused_child_created' if child else 'read_only_plan_validated',
        parent_run_id=receipt['parent_run_id'], child_run_id=receipt.get('child_run_id'), epoch_myr=receipt['epoch_myr'],
        next_output_myr=receipt['next_output_myr'], preexisting_array_count=receipt['preexisting_array_count'],
        entire_typed_state_exact=True, cached_boundary_geometry_unchanged=True, physical_time_advanced_myr=0.,
        child_path=str(child) if child else None), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
