"""Explicit accepted-step source-only continuation for exact geometry acceleration.

The disabled scaffold requires a new immutable review binding the future pause,
old/new source hashes, measured equivalence/performance evidence and human
computational-acceleration authority. It never constructs, advances, upgrades or
rebuilds a model. Entire inherited typed state, RNG, clocks, cache, configuration
and output scheduling remain exact; predecessor frames retain recursive sources.
A private staging directory publishes only after all evidence/readbacks pass.
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
import sys
import uuid

import numpy as np

import checkpoint
import lite_initiation_transition
from lite_initiation_transition import _fingerprint
import native_engine
import server

ROOT = Path(__file__).resolve().parent
KIND = 'lite_geometry_acceleration_source_transition'
SOURCE_REVIEW_KIND = 'lite_geometry_acceleration_runtime_source_delta_review'
ACCELERATION_REVIEW_KIND = 'lite_geometry_acceleration_equivalence_performance_review'
AUTHORITY_KIND = 'explicit_user_asein_lite_geometry_acceleration_authorization'
VERSION = 1
EXPECTED_HELPERS = 173
_WINDOWS_PATHS = sys.platform == 'win32'
HELPERS = ('arc_emplacement_geometry.py', 'arc_material_exclusion.py')
AUTHORITY_REQUEST = "Let's see if we can speed things up"
AUTHORITY_SCOPE = ('Computational geometry acceleration only; preserve physical laws, force coefficients, resolution, '
    'guards, ownership/material, entire typed state, RNG, policy clocks, geometry cache, recursive history and scheduling. '
    'Validate exact outputs and measured performance before a supported accepted-step pause and truthful source-only continuation.')


@dataclass(frozen=True)
class TransitionApproval:
    parent_run_id: str
    parent_source_commit: str
    parent_epoch_myr: float
    parent_checkpoint_sha256: str
    parent_preservation_receipt: Path
    parent_preservation_sha256: str
    runtime_source_seal: Path
    runtime_source_seal_sha256: str
    source_delta_receipt: Path
    source_delta_receipt_sha256: str
    acceleration_review_receipt: Path
    acceleration_review_sha256: str
    authority_receipt: Path
    authority_sha256: str
    helper_delta: tuple[tuple[str, str, str], ...]
    review_ready: bool = False


# Root may install only a fresh exact approval after runtime/evidence freeze and
# the controlled accepted-step pause. No earlier repair approval is inherited.
REVIEWED_APPROVAL: TransitionApproval | None = None


def _require(condition, message):
    if not condition:
        raise ValueError('Lite geometry acceleration source transition: ' + message)


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


def _check_publication_paths(parent, staging, published, parent_files, child_names):
    """Fail before copying if ordinary Windows paths cannot name the full tree.

    Keep canonical paths and relative history names unchanged. A short sibling
    stage avoids adding a long run identity to every recursively inherited path;
    the final public name must also fit, including arbitrary approved run IDs.
    """
    if not _WINDOWS_PATHS:
        return
    paths = [parent / name for name in parent_files]
    for root in (staging, published):
        paths.extend(root / 'inherited-parent' / name for name in parent_files)
        paths.extend(root / name for name in child_names)
    _require(all(len(str(path)) < 260 for path in paths),
        'a source, staged or final public file path exceeds the supported Windows path budget; '
        'preserve history and use a shorter publication root or child identity.')
    directories = {folder for path in paths for folder in path.parents}
    _require(all(len(str(folder)) < 248 for folder in directories),
        'a source, staged or final public directory exceeds the supported Windows path budget; '
        'preserve history and use a shorter publication root or child identity.')


def _read_evidence(path, sha256, description):
    path = Path(path)
    _require(path.is_file() and _file_sha(path) == sha256, description + ' changed or is missing.')
    return json.loads(path.read_text(encoding='utf-8'))


def _approved(approval=None):
    selected = REVIEWED_APPROVAL if approval is None else approval
    _require(isinstance(selected, TransitionApproval) and selected is REVIEWED_APPROVAL and
        selected.review_ready is True, 'new exact pause, frozen acceleration evidence and human authority are not sealed yet.')
    _require(isinstance(selected.parent_run_id, str) and
        re.fullmatch(r'[A-Za-z0-9_-]{1,80}', selected.parent_run_id), 'approved parent identity is invalid.')
    _require(type(selected.parent_epoch_myr) in (int, float) and np.isfinite(selected.parent_epoch_myr) and
        selected.parent_epoch_myr >= 0., 'approved accepted boundary epoch is invalid.')
    _require(isinstance(selected.parent_source_commit, str) and
        re.fullmatch(r'[0-9a-f]{40}', selected.parent_source_commit), 'approved predecessor commit is invalid.')
    for name in ('parent_checkpoint_sha256', 'parent_preservation_sha256', 'runtime_source_seal_sha256',
            'source_delta_receipt_sha256', 'acceleration_review_sha256', 'authority_sha256'):
        _require(isinstance(getattr(selected, name), str) and re.fullmatch(r'[0-9a-f]{64}', getattr(selected, name)),
            'approved evidence hash is invalid.')
    _require(type(selected.helper_delta) is tuple and len(selected.helper_delta) == len(HELPERS) and
        all(type(row) is tuple and len(row) == 3 for row in selected.helper_delta) and
        tuple(row[0] for row in selected.helper_delta) == HELPERS,
        'approval must bind only the two ordered geometry helper deltas.')
    for _, old, new in selected.helper_delta:
        _require(isinstance(old, str) and isinstance(new, str) and re.fullmatch(r'[0-9a-f]{64}', old) and
            re.fullmatch(r'[0-9a-f]{64}', new) and old != new, 'exact distinct old/new helper hashes are not sealed.')
    return selected


def _required_delta(approved):
    return {name: dict(parent_sha256=old, child_sha256=new) for name, old, new in approved.helper_delta}


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
    authority = _read_evidence(approved.authority_receipt, approved.authority_sha256, 'new human acceleration authority')
    _require(authority.get('kind') == AUTHORITY_KIND and authority.get('version') == VERSION and
        authority.get('parent_run_id') == approved.parent_run_id and
        authority.get('source_commit') == approved.parent_source_commit and authority.get('authorized') is True and
        authority.get('parent_checkpoint_sha256') == approved.parent_checkpoint_sha256 and
        authority.get('epoch_myr') == approved.parent_epoch_myr and authority.get('request') == AUTHORITY_REQUEST and
        isinstance(authority.get('authorization_source'), str) and bool(authority['authorization_source'].strip()) and
        authority.get('physical_policy_change_authorized') is False and authority.get('scope') == AUTHORITY_SCOPE,
        'authority does not cover this exact parent/pause and computational acceleration scope.')
    review = _read_evidence(approved.acceleration_review_receipt, approved.acceleration_review_sha256,
        'completed exact geometry equivalence/performance review')
    _require(review.get('kind') == ACCELERATION_REVIEW_KIND and review.get('version') == VERSION and
        review.get('ready') is True and review.get('parent_run_id') == approved.parent_run_id and
        review.get('parent_checkpoint_sha256') == approved.parent_checkpoint_sha256 and
        review.get('epoch_myr') == approved.parent_epoch_myr and review.get('changed_helpers') == _required_delta(approved) and
        review.get('runtime_source_seal_sha256') == approved.runtime_source_seal_sha256 and
        review.get('captured_regression_passed') is True and review.get('captured_equivalence_passed') is True and
        review.get('geometry_outputs_exact') is True and review.get('performance_measured') is True and
        review.get('performance_improved') is True and review.get('physical_laws_unchanged') is True and
        review.get('physical_guards_unchanged') is True and review.get('resolution_unchanged') is True,
        'acceleration review must bind passed exact-output regression and measured speed under unchanged laws/guards/resolution.')
    evidence = review.get('validation_receipts')
    _require(isinstance(evidence, list) and len(evidence) >= 2,
        'acceleration review needs distinct regression and measured-equivalence receipts.')
    seen = set()
    for row in evidence:
        _require(type(row) is dict and set(row) == {'path', 'sha256'} and isinstance(row['path'], str) and
            isinstance(row['sha256'], str) and re.fullmatch(r'[0-9a-f]{64}', row['sha256']),
            'validation evidence needs exact paths and hashes.')
        path = Path(row['path'])
        path = (path if path.is_absolute() else Path(approved.acceleration_review_receipt).parent / path).resolve()
        _require(path not in seen, 'validation evidence must name distinct receipts.')
        seen.add(path)
        _read_evidence(path, row['sha256'], 'new acceleration validation evidence')
    return authority, review


def _check_source(parent, recorded, expected, engine, helpers, approved):
    _require(_file_sha(parent / 'engine.py') == recorded['engine_sha256'] == expected['engine_sha256'] == _sha(engine),
        'engine entrypoint changes are outside this source transition.')
    for key in ('numpy_version', 'flow_backend', 'material_backend'):
        _require(recorded.get(key) == expected.get(key), key + ' changed outside this acceleration.')
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
    required = _required_delta(approved)
    _require(set(current) == set(prior) and changed == required,
        'only the exact reviewed two geometry helper deltas are supported; no same-source or additional changes.')
    review = _read_evidence(approved.source_delta_receipt, approved.source_delta_receipt_sha256,
        'new runtime source delta review')
    _require(review.get('kind') == SOURCE_REVIEW_KIND and review.get('version') == VERSION and
        review.get('ready') is True and review.get('parent_run_id') == approved.parent_run_id and
        review.get('parent_checkpoint_sha256') == approved.parent_checkpoint_sha256 and
        review.get('epoch_myr') == approved.parent_epoch_myr and
        review.get('boundary_kind') == 'healthy_accepted_step_pause' and
        review.get('parent_preservation_sha256') == approved.parent_preservation_sha256 and
        review.get('runtime_source_seal_sha256') == approved.runtime_source_seal_sha256 and
        review.get('parent_compatibility_sha256') == _sha(_json_bytes(recorded)) and
        review.get('child_compatibility_sha256') == _sha(_json_bytes(expected)) and
        review.get('changed_helpers') == required and review.get('added_helpers') == {} and
        review.get('removed_helpers') == [] and review.get('helper_count') == EXPECTED_HELPERS and
        review.get('acceleration_review_sha256') == approved.acceleration_review_sha256 and
        review.get('authority_sha256') == approved.authority_sha256 and review.get('configuration_unchanged') is True and
        review.get('state_import') == 'entire typed state exact; cached geometry retained',
        'new source delta review does not bind this exact pause, acceleration, authority and full-state import.')
    return dict(changed_helpers=changed, added_helpers={}, removed_helpers=[])


def _check_pause(parent, original, saved, external, approved, header):
    _require(saved.get('run_id') == external.get('run_id') == approved.parent_run_id,
        'parent identity differs from the reviewed application.')
    _require(saved.get('state') == external.get('state') == 'paused' and not saved.get('error') and not external.get('error'),
        'parent must finish its healthy accepted-step pause before import.')
    epoch = float(original.t)
    _require(epoch == approved.parent_epoch_myr, 'accepted boundary epoch differs from the sealed dynamic pause.')
    _require(saved.get('title') == external.get('title') == 'Asein Lite' and
        saved.get('can_resume') is True and external.get('can_resume') is True, 'parent is not a resumable Asein Lite history.')
    for manifest in (saved, external):
        checkpoint.verify_compatibility(manifest, header['compatibility'])
    _require(header['time_myr'] == saved.get('time_myr') == external.get('time_myr') ==
        saved.get('integration_time_myr') == external.get('integration_time_myr') ==
        saved.get('checkpoint_time_myr') == external.get('checkpoint_time_myr') == epoch,
        'pause generation is stale or does not name its exact durable epoch.')
    _require(_fingerprint(saved.get('config')) == _fingerprint(external.get('config')) == _fingerprint(original.config),
        'parent configurations differ; this source-only transition cannot change controls.')
    _require(json.loads((parent / 'config.json').read_text()) == original.config, 'saved config file differs from typed state.')
    count, frames = saved.get('frame_count'), saved.get('frames', [])
    _require(type(count) is int and count > 0 and len(frames) == count and
        external.get('frames') == frames and external.get('frame_count') == count, 'parent accepted frame history differs.')
    _require(saved.get('source_transition') == external.get('source_transition') and
        saved.get('frame_sources') == external.get('frame_sources'), 'parent recursive source lineage differs.')
    pending = saved.get('next_output_myr')
    _require(type(pending) in (float, int) and np.isfinite(pending) and epoch < pending <= original.config['duration_myr'] and
        external.get('next_output_myr') == pending, 'exact paused output scheduler is missing or differs.')
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
    """Validate an exact preserved pause; no writes and no state rebuild."""
    approved = _approved(approval)
    parent = Path(parent).resolve()
    _require(parent.is_dir(), 'parent history directory is missing.')
    before = _files(parent)
    _require(before.get('checkpoint.npz') == approved.parent_checkpoint_sha256, 'parent checkpoint differs from reviewed pause.')
    preserved = _read_evidence(approved.parent_preservation_receipt, approved.parent_preservation_sha256,
        'new immutable parent preservation receipt')
    _require(preserved.get('run_id') == approved.parent_run_id and
        preserved.get('all_copied_files_exact') is True and preserved.get('file_count') == len(before) and
        preserved.get('checkpoint_sha256') == approved.parent_checkpoint_sha256 and
        {row['path']: row['sha256'] for row in preserved['files']} == before, 'parent files differ from immutable preservation.')
    authority, acceleration = _check_reviews(approved)
    header = checkpoint.checkpoint_header(parent / 'checkpoint.npz')
    expected, engine, helpers = _current_context(approved)
    delta = _check_source(parent, header['compatibility'], expected, engine, helpers, approved)
    original, saved = checkpoint.read_checkpoint(parent / 'checkpoint.npz', header['compatibility'], native_engine.Simulation)
    external = json.loads((parent / 'manifest.json').read_text())
    _check_pause(parent, original, saved, external, approved, header)
    derived = _derive(original)
    fingerprint = _fingerprint(vars(original))
    _require(_fingerprint(vars(derived)) == fingerprint, 'source import changed a typed array, config, RNG, cache, clock or history field.')
    _require(_files(parent) == before, 'parent changed while planning; wait for the exact stable pause.')
    receipt = dict(kind=KIND, version=VERSION, parent_run_id=approved.parent_run_id,
        parent_source_commit=approved.parent_source_commit, epoch_myr=float(original.t),
        producer_sha256=_file_sha(Path(__file__)),
        fingerprint_utility_sha256=_file_sha(Path(lite_initiation_transition.__file__)),
        integration_steps=int(original.steps), parent_checkpoint_sha256=before['checkpoint.npz'],
        parent_manifest_sha256=before['manifest.json'], parent_files_sha256=before, parent_file_count=len(before),
        parent_preservation_receipt_sha256=approved.parent_preservation_sha256,
        runtime_source_seal_sha256=approved.runtime_source_seal_sha256,
        source_delta_review_sha256=approved.source_delta_receipt_sha256,
        acceleration_review_sha256=approved.acceleration_review_sha256, authority_sha256=approved.authority_sha256,
        parent_compatibility=header['compatibility'], child_compatibility=expected, source_delta=delta,
        full_state_fingerprint=fingerprint['sha256'], preexisting_array_count=len(fingerprint['arrays']),
        entire_typed_state_exact=True, configuration_unchanged=True, rng_unchanged=True,
        all_existing_policy_clocks_and_history_unchanged=True, initial_file_sha256=before['initial.json'],
        next_output_myr=saved['next_output_myr'], inherited_frame_count=saved['frame_count'],
        cached_boundary_geometry_unchanged=True, geometry_rebuilt_at_import=False,
        first_source_solve_geometry='Accepted predecessor cache; reviewed acceleration is used only during normal future geometry evaluation.',
        physical_time_advanced_myr=0., model_constructed=False, model_stepped=False,
        historical_frames_rewritten=False, physical_policies_added=False,
        interpretation='Separate source boundary for computationally accelerated exact geometry, with no physical-policy change. '
            'Inherited solver reports and frames remain predecessor evidence until future accepted successor evolution.')
    return TransitionPlan(parent, original, derived, deepcopy(saved), receipt, before, expected, engine, helpers, approved)


def _retain_proofs(approved, child):
    _, acceleration = _check_reviews(approved)
    reserved = {'manifest.json', 'source-transition.json', 'checkpoint.npz', 'checkpoint.source-boundary.npz',
        'config.json', 'initial.json', 'engine.py', 'inherited-parent', 'parent-preservation.json',
        'source-delta-review.json', 'runtime-source-seal.json', 'acceleration-review.json', 'application-authority.json'}
    reserved |= {name + '.tmp' for name in reserved}
    mappings = []
    for row in acceleration['validation_receipts']:
        reference = Path(row['path'])
        source = (reference if reference.is_absolute() else Path(approved.acceleration_review_receipt).parent / reference).resolve()
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
        'acceleration-review.json': approved.acceleration_review_sha256,
        'application-authority.json': approved.authority_sha256,
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
            scope='Two exact geometry acceleration helper deltas; entire typed state, policies and geometry cache remain exact.'),
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
    _require(output_root.is_dir(), 'publication root must already exist.')
    published_child = child
    child = output_root / ('.src-' + uuid.uuid4().hex[:12])
    child_names = {'engine.py', *plan.helper_sources, 'initial.json', 'config.json',
        'checkpoint.npz', 'checkpoint.source-boundary.npz', 'manifest.json', 'manifest.json.tmp',
        'source-transition.json', 'source-transition.json.tmp', 'parent-preservation.json',
        'runtime-source-seal.json', 'source-delta-review.json', 'acceleration-review.json', 'application-authority.json'}
    # checkpoint.write_checkpoint uses this UUID-suffixed atomic temporary file.
    child_names.add('checkpoint.npz.' + '0' * 32 + '.tmp')
    child_names.update(name for name in plan.parent_files if re.fullmatch(r'frame_\d+\.(npz|json)', name))
    _, acceleration = _check_reviews(approved)
    child_names.update(Path(row['path']).name for row in acceleration['validation_receipts']
        if not Path(row['path']).is_absolute() and Path(row['path']).name == row['path'])
    _check_publication_paths(plan.parent, child, published_child, plan.parent_files, child_names)
    child.mkdir(parents=False)
    shutil.copytree(plan.parent, child / 'inherited-parent')
    _require(_files(child / 'inherited-parent') == plan.parent_files, 'copied complete parent history differs.')
    for source, name in ((approved.parent_preservation_receipt, 'parent-preservation.json'),
            (approved.runtime_source_seal, 'runtime-source-seal.json'), (approved.source_delta_receipt, 'source-delta-review.json'),
            (approved.acceleration_review_receipt, 'acceleration-review.json'), (approved.authority_receipt, 'application-authority.json')):
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
    _write_json(child / 'source-transition.json', receipt)
    _require(_file_sha(child / 'source-transition.json') == _sha(_json_bytes(receipt) + b'\n') and
        json.loads((child / 'source-transition.json').read_text(encoding='utf-8')) == receipt,
        'child source-transition receipt readback differs before publication.')
    _verify_child_artifacts(plan, child, manifest, receipt)
    _write_json(child / 'manifest.json', manifest)
    _require(json.loads((child / 'manifest.json').read_text()) == manifest, 'staged manifest readback differs.')
    _require(not published_child.exists(), 'child appeared during source publication.')
    child.replace(published_child)
    return published_child, receipt


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
