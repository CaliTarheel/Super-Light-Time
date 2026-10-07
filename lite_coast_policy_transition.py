"""Explicit, same-source Lite coast-v2 surface-policy boundary.

Only collision_coast_version and its deterministic migration record change.
All preexisting typed arrays, configuration, RNG, clocks, accepted geometry cache
and recursively inherited frame bytes remain exact. The canonical neutral datum
already exists in structure.reference_elevation_m; no datum is reconstructed.
This producer cannot construct/evolve/rebuild a model or control a live server.
Publication is separately approved, paused and committed from private staging.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import uuid

import numpy as np

import checkpoint
import collision_coast
import collision_surface
import gravitational_relaxation
import lite_gravity_iteration_transition as history_tools
import lite_initiation_transition
from lite_initiation_transition import _fingerprint
import native_engine
import server
import surface_erosion
import viscous_sheet

ROOT = Path(__file__).resolve().parent
LIVE = Path(r'E:\Deep-Time\Deep-Time-asein-lite-20261002')
KIND = 'lite_coast_surface_policy_transition'
VERSION = 1
EXPECTED_HELPERS = 173
MINIMUM_FRAMES = 53
PARENT_RUN_ID = '20261003-074721-7fbe9c'
PARENT_SOURCE_COMMIT = '3e190ee7a070b5607c559e28d2573effb16d2276'
AUTHORITY_KIND = 'asein_lite_collision_coast_policy_human_authorization'
REVIEW_KIND = 'asein_lite_collision_coast_surface_policy_review'
PRESERVATION_KIND = 'asein_lite_complete_accepted_boundary_history_preservation'
BOUNDARY_KIND = 'asein_lite_supported_accepted_step_pause'
DRY_KIND = 'asein_lite_coast_policy_constructorless_dry_validation'
AUTHORITY_SHA256 = '5e2c6b0ac752246e94feac95c2e36eb175fc434aec7588244c81f5e8d43f72b4'
POLICY_REVIEW_SHA256 = '896e8a0af46fd15e4913ce3b4be76cd9a7ceed7aa576d7815cbd051b52160f48'
POLICY_STATE_FIELD_NAMES = ('collision_coast_version', 'collision_coast_migration')
POLICY_STATE_FIELDS = frozenset(POLICY_STATE_FIELD_NAMES)
POLICY_DELTA = {'collision_coast_version': {'from': 0, 'to': 2}}
REFERENCE_PATH = 'structure.reference_elevation_m'


@dataclass(frozen=True)
class TransitionApproval:
    parent_run_id: str
    parent_source_commit: str
    parent_epoch_myr: float
    parent_checkpoint_sha256: str
    expected_frame_count: int
    expected_next_output_myr: float
    parent_preservation_receipt: Path
    parent_preservation_sha256: str
    runtime_source_seal: Path
    runtime_source_seal_sha256: str
    policy_review_receipt: Path
    policy_review_sha256: str
    authority_receipt: Path
    authority_sha256: str
    boundary_receipt: Path
    boundary_sha256: str
    expected_producer_sha256: str
    dry_validation_receipt: Path | None = None
    dry_validation_sha256: str | None = None
    review_ready: bool = False


REVIEWED_APPROVAL: TransitionApproval | None = None


def _require(condition, message):
    if not condition:
        raise ValueError('Lite coast surface-policy transition: ' + message)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path):
    return _sha(Path(path).read_bytes())


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _scope(path, *, existing=False):
    raw = Path(path)
    raw = raw if raw.is_absolute() else ROOT / raw
    _require(ROOT.resolve() == LIVE.resolve(), 'only the authorized isolated LIVE checkout is allowed.')
    _require(not raw.is_symlink() and all(not p.is_symlink() for p in raw.parents), 'symlink paths are forbidden.')
    result = raw.resolve()
    _require(result.is_relative_to(ROOT.resolve()), 'a path escapes the isolated LIVE checkout.')
    if existing:
        _require(result.exists(), 'a required sealed path is missing: ' + str(result))
    return result


def _write_fresh_json(path, value):
    path = _scope(path)
    _require(path.parent.is_dir() and not path.exists(), 'new receipt path must be absent in an existing directory.')
    with path.open('xb') as stream:
        stream.write(_json_bytes(value) + b'\n')
    _require(json.loads(path.read_text(encoding='utf-8')) == value, 'new receipt readback differs.')


def _files(folder):
    return history_tools._files(_scope(folder, existing=True))


def _read_evidence(path, sha256, description):
    path = _scope(path, existing=True)
    _require(path.is_file() and _file_sha(path) == sha256, description + ' changed or is missing.')
    return json.loads(path.read_text(encoding='utf-8'))


def approval_from_json(path, expected_sha256):
    raw = _read_evidence(path, expected_sha256, 'externally pinned approval')
    _require(type(raw) is dict and set(raw) <= {f.name for f in fields(TransitionApproval)}, 'unknown approval fields.')
    converted = dict(raw)
    for name in ('parent_preservation_receipt', 'runtime_source_seal', 'policy_review_receipt',
            'authority_receipt', 'boundary_receipt', 'dry_validation_receipt'):
        if converted.get(name) is not None:
            _require(isinstance(converted[name], str), 'approval evidence paths must be strings.')
            converted[name] = _scope(converted[name], existing=True)
    return _approved(TransitionApproval(**converted))


def _approved(approval=None):
    selected = REVIEWED_APPROVAL if approval is None else approval
    _require(type(selected) is TransitionApproval and selected.review_ready is True, 'no sealed coast-policy approval is active.')
    _require(selected.parent_run_id == PARENT_RUN_ID and selected.parent_source_commit == PARENT_SOURCE_COMMIT,
        'approval must bind the current gravity continuation, with its actual unchanged source commit.')
    _require(type(selected.parent_epoch_myr) in (int, float) and np.isfinite(selected.parent_epoch_myr) and
        selected.parent_epoch_myr >= 103. and type(selected.expected_frame_count) is int and
        selected.expected_frame_count >= MINIMUM_FRAMES, 'accepted boundary epoch/frame count is invalid.')
    _require(type(selected.expected_next_output_myr) in (int, float) and np.isfinite(selected.expected_next_output_myr) and
        selected.expected_next_output_myr > selected.parent_epoch_myr, 'embedded future public-output schedule is invalid.')
    for name in ('parent_checkpoint_sha256', 'parent_preservation_sha256', 'runtime_source_seal_sha256',
            'policy_review_sha256', 'authority_sha256', 'boundary_sha256', 'expected_producer_sha256'):
        _require(isinstance(getattr(selected, name), str) and re.fullmatch(r'[0-9a-f]{64}', getattr(selected, name)),
            'an approval hash is invalid: ' + name)
    _require(selected.authority_sha256 == AUTHORITY_SHA256 and selected.policy_review_sha256 == POLICY_REVIEW_SHA256,
        'the exact human authority and existing completed policy review are required.')
    _require(_file_sha(Path(__file__)) == selected.expected_producer_sha256, 'producer source differs from its approval pin.')
    for name in ('parent_preservation_receipt', 'runtime_source_seal', 'policy_review_receipt', 'authority_receipt', 'boundary_receipt'):
        _scope(getattr(selected, name), existing=True)
    _require((selected.dry_validation_receipt is None) == (selected.dry_validation_sha256 is None),
        'dry validation path and hash must be supplied together.')
    if selected.dry_validation_sha256 is not None:
        _require(isinstance(selected.dry_validation_sha256, str) and re.fullmatch(r'[0-9a-f]{64}', selected.dry_validation_sha256),
            'dry validation hash is invalid.')
        _scope(selected.dry_validation_receipt, existing=True)
    return selected


@contextmanager
def _no_model_work():
    """Private-process barriers; checkpoint __new__ import and array copies only."""
    saved = []
    def forbidden(*args, **kwargs):
        raise RuntimeError('Constructors, evolution, solvers and geometry rebuilds are forbidden during coast transition.')
    targets = [(native_engine.Simulation, name) for name in ('__init__', 'step', 'snapshot',
        '_step_once', '_advance_step', '_advect', '_boundaries', '_rasterize', '_forces',
        '_topology', '_deform_and_accrete', '_capture_initial_snapshot')]
    targets += [(viscous_sheet, 'solve'), (gravitational_relaxation, 'relax')]
    try:
        for target, name in targets:
            if hasattr(target, name):
                saved.append((target, name, name in vars(target), getattr(target, name)))
                setattr(target, name, forbidden)
        yield
    finally:
        for target, name, owned, value in reversed(saved):
            if owned:
                setattr(target, name, value)
            else:
                delattr(target, name)


def _current_context(approved):
    expected = server.SimulationManager.compatibility()
    seal = _read_evidence(approved.runtime_source_seal, approved.runtime_source_seal_sha256, 'runtime source seal')
    _require(type(seal) is dict and len(seal) == EXPECTED_HELPERS and seal == expected['auxiliary_sources_sha256'],
        'runtime source must equal the exact unchanged 173-helper closure.')
    _require(all(Path(name).name == name and name.endswith('.py') for name in seal), 'runtime helper names are invalid.')
    _require(_file_sha(ROOT / 'tectonics.py') == expected['engine_sha256'] and
        all(_file_sha(ROOT / name) == value for name, value in seal.items()), 'runtime files changed since their source capture.')
    head = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    _require(head == approved.parent_source_commit, 'same-source policy boundary must retain the exact current source commit.')
    return expected, server.ENGINE_SOURCE, dict(server.AUXILIARY_SOURCES)


def _check_reviews(approved):
    authority = _read_evidence(approved.authority_receipt, approved.authority_sha256, 'human coast authority')
    _require(authority.get('kind') == AUTHORITY_KIND and authority.get('version') == VERSION and authority.get('approved') is True and
        authority.get('parent_run_id') == approved.parent_run_id and authority.get('source_commit') == approved.parent_source_commit and
        authority.get('workspace') == str(LIVE) and authority.get('policy_delta') == POLICY_DELTA and
        authority.get('surface_erosion_version') == 0 and
        ' '.join(authority.get('human_instruction', '').split()) == 'Coast approved. Run it the next time a turn completes',
        'human authority does not cover the exact future coast-v2 policy without erosion.')
    review = _read_evidence(approved.policy_review_receipt, approved.policy_review_sha256, 'completed coast policy review')
    expected = dict(kind=REVIEW_KIND, version=VERSION, parent_run_id=approved.parent_run_id,
        source_commit=approved.parent_source_commit, authority_sha256=approved.authority_sha256,
        collision_coast_version_before=0, collision_coast_version_after=2, surface_erosion_version_before=0,
        surface_erosion_version_after=0, transition_kind='surface_policy', runtime_helper_changes=[],
        physical_laws_or_guards_changed=False, future_accepted_frames_only=True, historical_frames_rewritten=False,
        entire_state_byte_exact_claim=False, expected_exact_nonpolicy_typed_state=True,
        uses_current_canonical_neutral_reference=True, repeat_private_coupled_trial_required=False)
    _require(all(review.get(name) == value for name, value in expected.items()), 'policy review scope or authority differs.')
    docs = review.get('documentation')
    _require(type(docs) is dict and set(docs) == {'COASTAL-DEFORMATION.md', 'COLLISION-COAST-RECONSTRUCTION.md'} and
        all(_file_sha(_scope(name, existing=True)) == value for name, value in docs.items()), 'reviewed policy documentation differs.')
    evidence = review.get('existing_validation_evidence')
    _require(type(evidence) is list and len(evidence) == 3 and
        {row.get('role') for row in evidence} == {'read_only96_diagnosis', 'read_only96_v2_counterfactual', 'prior_investigation_completion'},
        'completed diagnostic and surface reconstruction evidence is missing.')
    for row in evidence:
        _read_evidence(row['path'], row['sha256'], 'existing completed coast evidence')
    return authority, review


def _check_source(parent, recorded, expected, engine, helpers):
    _require(recorded == expected, 'coast policy must retain complete true source/backend compatibility exactly.')
    _require(_file_sha(parent / 'engine.py') == recorded['engine_sha256'] == _sha(engine), 'saved engine source differs.')
    prior = recorded.get('auxiliary_sources_sha256')
    _require(type(prior) is dict and len(prior) == EXPECTED_HELPERS and
        all(Path(name).name == name and name.endswith('.py') and _file_sha(parent / name) == value for name, value in prior.items()),
        'parent source closure is not the exact unchanged 173-helper closure.')
    closure = server.capture_auxiliary_sources((parent / 'engine.py').read_bytes(), root=parent)
    _require({name: _sha(value) for name, value in closure.items()} == prior ==
        {name: _sha(value) for name, value in helpers.items()}, 'saved or current runtime closure is incomplete or changed.')
    _require(Path(__file__).name not in helpers, 'the producer must not silently enter the runtime helper closure.')


def _migration(epoch):
    return dict(version=1, from_version=0, to_version=2, time_myr=float(epoch), surface_only=True,
        surface_erosion_version=0, canonical_reference_path=REFERENCE_PATH, physical_state_changed=False,
        geometry_rebuilt=False, initial_conditions_rerun=False, historical_frames_rewritten=False)


def _nonpolicy(state):
    return {name: value for name, value in state.items() if name not in POLICY_STATE_FIELDS}


def _reference(simulation):
    structure = getattr(simulation, 'structure', None)
    _require(type(structure) is dict and 'reference_elevation_m' in structure, 'canonical current neutral reference is absent.')
    reference = structure['reference_elevation_m']
    _require(isinstance(reference, np.ndarray) and reference.dtype == np.dtype('float64') and
        reference.shape == (len(simulation.mass),) and np.isfinite(reference).all(), 'canonical neutral reference is not finite aligned float64 state.')
    return reference


def _derive(original, epoch):
    frozen_original = _fingerprint(vars(original))
    _require(type(getattr(original, 'collision_coast_version', 0)) is int and getattr(original, 'collision_coast_version', 0) == 0,
        'only the absent/zero predecessor coast policy can transition to v2.')
    _require(not hasattr(original, 'collision_coast_migration'), 'an existing coast migration cannot be relabeled.')
    _require(surface_erosion.version(original) == 0 and getattr(original, 'collision_surface_version', 0) == 1,
        'coast-v2 needs existing collision support and must retain erosion version zero.')
    _reference(original)
    derived = deepcopy(original)
    derived.collision_coast_version = 2
    derived.collision_coast_migration = _migration(epoch)
    _assert_policy_only(original, derived, epoch)
    _require(_fingerprint(vars(original)) == frozen_original, 'policy derivation mutated the imported predecessor.')
    return derived


def _assert_policy_only(original, derived, epoch):
    before, after = _fingerprint(vars(original)), _fingerprint(vars(derived))
    _require(before['arrays'] == after['arrays'], 'a preexisting typed array changed during coast policy import.')
    _require(_fingerprint(_nonpolicy(vars(original))) == _fingerprint(_nonpolicy(vars(derived))),
        'nonpolicy state, RNG, configuration, physical clock or accepted cache changed.')
    _require(before['sha256'] != after['sha256'] and derived.collision_coast_version == 2 and
        derived.collision_coast_migration == _migration(epoch) and surface_erosion.version(derived) == 0,
        'the explicit two-field policy delta is missing, different or includes erosion.')
    _require(set(vars(derived)) - set(vars(original)) <= POLICY_STATE_FIELDS and
        not (set(vars(original)) - set(vars(derived))), 'an undeclared state field was added or removed.')
    _require(np.array_equal(_reference(original), _reference(derived)), 'canonical neutral datum changed.')
    return before, after, _fingerprint(_nonpolicy(vars(original)))


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


def _check_boundary(parent, original, saved, external, header, boundary, approved):
    epoch = float(original.t)
    _require(epoch == approved.parent_epoch_myr == header['time_myr'], 'accepted boundary checkpoint time differs.')
    for manifest in (saved, external):
        _require(manifest.get('run_id') == approved.parent_run_id and manifest.get('title') == 'Asein Lite' and
            manifest.get('state') == 'paused' and manifest.get('error') is None and manifest.get('can_resume') is True and
            manifest.get('integration_time_myr') == manifest.get('checkpoint_time_myr') == epoch,
            'only the stable supported accepted-step paused parent can transition.')
        checkpoint.verify_compatibility(manifest, header['compatibility'])
        _require(_json_bytes(manifest.get('config')) == _json_bytes(original.config), 'configuration differs from accepted state.')
    _require(saved.get('time_myr') == epoch and json.loads((parent / 'config.json').read_text()) == original.config,
        'embedded accepted time or saved configuration differs.')
    for field in ('frames', 'frame_count', 'frame_sources', 'source_transition'):
        _require(saved.get(field) == external.get(field), 'external and embedded accepted history differ: ' + field)
    count, frames_saved = saved.get('frame_count'), saved.get('frames')
    _require(count == approved.expected_frame_count and type(count) is int and type(frames_saved) is list and
        len(frames_saved) == count and count >= MINIMUM_FRAMES, 'accepted inherited frame count differs.')
    _require(saved.get('next_output_myr') == approved.expected_next_output_myr and
        epoch < saved['next_output_myr'] <= original.config['duration_myr'], 'embedded accepted scheduler differs.')
    _require(boundary.get('kind') == BOUNDARY_KIND and boundary.get('state') == 'paused' and boundary.get('port') == 8769 and
        boundary.get('run_id') == approved.parent_run_id and boundary.get('source_commit') == approved.parent_source_commit and
        boundary.get('checkpoint_sha256') == approved.parent_checkpoint_sha256 and boundary.get('error') is None and
        boundary.get('time_myr') == boundary.get('integration_time_myr') == boundary.get('checkpoint_time_myr') == epoch and
        boundary.get('frame_count') == count and boundary.get('next_output_myr') == saved['next_output_myr'],
        'sealed supported pause does not bind the actual accepted checkpoint and embedded scheduler.')
    previous = -np.inf
    for index, row in enumerate(frames_saved):
        stem = parent / f'frame_{index:04d}'
        metadata = json.loads(stem.with_suffix('.json').read_text())
        time = row.get('time_myr')
        _require(row.get('index') == metadata.get('index') == index and type(time) in (int, float) and
            np.isfinite(time) and previous <= time <= epoch and metadata.get('time_myr') == time and
            stem.with_suffix('.npz').is_file(), 'an accepted frame index, epoch or file differs.')
        previous = time
    _require(previous == epoch and external.get('time_myr') == epoch, 'supported pause must retain its accepted final snapshot.')
    history_tools._verify_recursive_history(parent, saved)


def plan_transition(parent, *, approval=None):
    """Read-only deterministic import plan; no writes, constructors or evolution."""
    approved = _approved(approval)
    parent = _scope(parent, existing=True)
    _require(parent.is_dir(), 'preserved parent directory is absent.')
    before = _files(parent)
    _require(before.get('checkpoint.npz') == approved.parent_checkpoint_sha256, 'parent checkpoint differs from the sealed accepted pause.')
    preserved = _read_evidence(approved.parent_preservation_receipt, approved.parent_preservation_sha256, 'accepted parent preservation')
    records = preserved.get('files')
    _require(preserved.get('kind') == PRESERVATION_KIND and preserved.get('run_id') == approved.parent_run_id and
        preserved.get('source_commit') == approved.parent_source_commit and preserved.get('state') == 'paused' and
        preserved.get('checkpoint_sha256') == approved.parent_checkpoint_sha256 and
        preserved.get('checkpoint_time_myr') == approved.parent_epoch_myr and preserved.get('all_copied_files_exact') is True and
        preserved.get('source_closure_verified') is True and preserved.get('helper_count') == EXPECTED_HELPERS and
        preserved.get('file_count') == len(before) and type(records) is list and len(records) == len(before) and
        {row['path']: row['sha256'] for row in records} == before, 'complete preserved parent history differs.')
    _check_reviews(approved)
    header = checkpoint.checkpoint_header(parent / 'checkpoint.npz')
    expected, engine, helpers = _current_context(approved)
    _check_source(parent, header['compatibility'], expected, engine, helpers)
    with _no_model_work():
        original, saved = checkpoint.read_checkpoint(parent / 'checkpoint.npz', header['compatibility'], native_engine.Simulation)
        external = json.loads((parent / 'manifest.json').read_text())
        boundary = _read_evidence(approved.boundary_receipt, approved.boundary_sha256, 'accepted boundary evidence')
        _check_boundary(parent, original, saved, external, header, boundary, approved)
        frozen_original = _fingerprint(vars(original))
        derived = _derive(original, original.t)
        before_state, after_state, nonpolicy = _assert_policy_only(original, derived, original.t)
        _require(before_state == frozen_original, 'imported predecessor changed during policy validation.')
    _require(_files(parent) == before, 'parent changed while reading the accepted boundary.')
    reference = _reference(original)
    receipt = dict(kind=KIND, version=VERSION, transition_kind='surface_policy', boundary_kind='supported_accepted_step_pause',
        parent_run_id=approved.parent_run_id, parent_source_commit=approved.parent_source_commit, child_source_commit=approved.parent_source_commit,
        epoch_myr=float(original.t), integration_steps=int(original.steps),
        transition_producer_sha256=approved.expected_producer_sha256, producer_sha256=approved.expected_producer_sha256,
        fingerprint_utility_sha256=_file_sha(Path(lite_initiation_transition.__file__)),
        history_validation_utility_sha256=_file_sha(Path(history_tools.__file__)), checkpoint_utility_sha256=_file_sha(Path(checkpoint.__file__)),
        parent_checkpoint_sha256=before['checkpoint.npz'], parent_manifest_sha256=before['manifest.json'],
        parent_files_sha256=before, parent_file_count=len(before), parent_preservation_sha256=approved.parent_preservation_sha256,
        runtime_source_seal_sha256=approved.runtime_source_seal_sha256, policy_review_sha256=approved.policy_review_sha256,
        authority_sha256=approved.authority_sha256, boundary_sha256=approved.boundary_sha256,
        predecessor_embedded_state=saved['state'], predecessor_external_state=external['state'], predecessor_error=external.get('error'),
        predecessor_public_time_myr=external['time_myr'], predecessor_external_next_output_myr=external.get('next_output_myr'),
        scheduler_authority='embedded durable predecessor checkpoint', next_output_myr=saved['next_output_myr'],
        inherited_frame_count=saved['frame_count'], parent_compatibility=header['compatibility'], child_compatibility=expected,
        source_delta=dict(changed_helpers={}, added_helpers={}, removed_helpers=[]), policy_delta=deepcopy(POLICY_DELTA),
        policy_state_fields=list(POLICY_STATE_FIELD_NAMES), collision_coast_migration=_migration(original.t), surface_erosion_version=0,
        parent_full_state_fingerprint=before_state['sha256'], child_full_state_fingerprint=after_state['sha256'],
        nonpolicy_state_fingerprint=nonpolicy['sha256'], preexisting_array_count=len(before_state['arrays']),
        preexisting_array_inventory=before_state['arrays'], all_preexisting_arrays_exact=True, entire_typed_state_exact=False,
        configuration_unchanged=True, rng_unchanged=True, cache_unchanged=True, all_existing_policy_clocks_and_history_unchanged=True,
        initial_file_sha256=before['initial.json'], cached_boundary_geometry_unchanged=True, geometry_rebuilt_at_import=False,
        canonical_reference_path=REFERENCE_PATH, neutral_reference_sha256=_sha(np.ascontiguousarray(reference).tobytes()),
        neutral_reference_dtype=reference.dtype.str, neutral_reference_shape=list(reference.shape),
        future_frame_reference_field=collision_coast.REFERENCE_FIELD, neutral_reference_interpretation='Current accepted canonical neutral datum; lossless existing snapshot copy, never a guessed initial or target height.',
        physical_time_advanced_myr=0., model_constructed=False, model_stepped=False, historical_frames_rewritten=False,
        physical_laws_or_guards_changed=False, future_accepted_frames_only=True,
        interpretation='Same-source surface reconstruction policy 0 to 2 and explicit migration metadata only. '
            'All nonpolicy typed state, arrays, geometry cache, RNG, clocks and inherited frame bytes remain exact. '
            'Future frames retain signed H-R under the existing ordered reconstruction; no forced emergence, gap filling, welding or erosion upgrade.')
    return TransitionPlan(parent, original, derived, deepcopy(saved), receipt, before, expected, engine, helpers, approved)


def _fresh_plan(plan):
    _require(type(plan) is TransitionPlan, 'a reviewed constructorless transition plan is required.')
    fresh = plan_transition(plan.parent, approval=plan.approval)
    _require(_fingerprint(vars(plan.original)) == _fingerprint(vars(fresh.original)) and
        _fingerprint(vars(plan.derived)) == _fingerprint(vars(fresh.derived)) and plan.manifest == fresh.manifest and
        plan.receipt == fresh.receipt and plan.parent_files == fresh.parent_files and plan.compatibility == fresh.compatibility and
        plan.engine_source == fresh.engine_source and plan.helper_sources == fresh.helper_sources,
        'mutable plan state, sources, scheduler or evidence changed after import validation.')
    return fresh


def validate_dry_run(plan, *, validation_root):
    """Fresh private checkpoint round-trip plus pure existing snapshot field copy.

    No interval is attempted. This validates a serialization/policy migration,
    not future collision mechanics or long-term map behavior.
    """
    _fresh_plan(plan)
    destination = _scope(validation_root)
    _require(destination.is_relative_to((ROOT / 'tmp').resolve()) and not destination.exists() and destination.parent.is_dir(),
        'dry validation requires a fresh directory below LIVE/tmp.')
    _require(not destination.is_relative_to(plan.parent) and not plan.parent.is_relative_to(destination),
        'dry candidate directory must be disjoint from preserved predecessor history.')
    destination.mkdir()
    candidate = destination / 'candidate.npz'
    dry_manifest = deepcopy(plan.manifest)
    dry_manifest.update(diagnostic_only=True, diagnostic_kind=DRY_KIND, can_resume=False)
    with _no_model_work():
        before = _fingerprint(vars(plan.derived))
        checkpoint.write_checkpoint(candidate, plan.derived, dry_manifest, plan.compatibility)
        restored, saved = checkpoint.read_checkpoint(candidate, plan.compatibility, native_engine.Simulation)
        _require(saved == dry_manifest and _fingerprint(vars(restored)) == before, 'private candidate checkpoint round-trip differs.')
        _assert_policy_only(plan.original, restored, plan.original.t)
        snapshot = collision_surface.snapshot_fields(restored)
        _require(snapshot.get('collision_coast_version') == 2 and snapshot.get('collision_surface_version') == 1,
            'existing pure snapshot helper does not retain the approved versions.')
        reference = collision_coast.reference_field(snapshot, len(restored.mass))
        _require(reference is not None and reference.dtype == _reference(restored).dtype and
            _sha(np.ascontiguousarray(reference).tobytes()) == plan.receipt['neutral_reference_sha256'],
            'existing future-frame reference copy is not lossless canonical state.')
        _require(_fingerprint(vars(restored)) == before and _fingerprint(vars(plan.derived)) == before,
            'pure snapshot field validation mutated imported state.')
    _fresh_plan(plan)
    receipt = dict(kind=DRY_KIND, version=VERSION, passed=True, transition_kind=KIND,
        parent_run_id=plan.approval.parent_run_id, parent_source_commit=plan.approval.parent_source_commit,
        parent_checkpoint_sha256=plan.approval.parent_checkpoint_sha256, epoch_myr=plan.receipt['epoch_myr'],
        transition_producer_sha256=plan.approval.expected_producer_sha256, runtime_source_seal_sha256=plan.approval.runtime_source_seal_sha256,
        parent_preservation_sha256=plan.approval.parent_preservation_sha256, policy_review_sha256=plan.approval.policy_review_sha256,
        authority_sha256=plan.approval.authority_sha256, boundary_sha256=plan.approval.boundary_sha256,
        parent_full_state_fingerprint=plan.receipt['parent_full_state_fingerprint'],
        child_full_state_fingerprint=plan.receipt['child_full_state_fingerprint'], nonpolicy_state_fingerprint=plan.receipt['nonpolicy_state_fingerprint'],
        all_preexisting_arrays_exact=True, preexisting_array_count=plan.receipt['preexisting_array_count'],
        inherited_frame_count=plan.receipt['inherited_frame_count'], next_output_myr=plan.receipt['next_output_myr'],
        neutral_reference_sha256=plan.receipt['neutral_reference_sha256'], future_reference_copy_exact=True,
        policy_delta=deepcopy(POLICY_DELTA), collision_coast_migration=deepcopy(plan.receipt['collision_coast_migration']),
        surface_erosion_version=0, parent_and_sources_unchanged=True, private_state_unchanged_after_snapshot=True,
        private_checkpoint_file='candidate.npz', private_checkpoint_sha256=_file_sha(candidate),
        model_constructed=False, model_stepped=False, solve_called=False, geometry_rebuilt=False,
        physical_time_advanced_myr=0., public_child_created=False, coupled_world_trial=False,
        limitation='Checkpoint and pure snapshot-field validation only; no future model interval or map is simulated.')
    path = destination / 'coast-dry-validation.json'
    _write_fresh_json(path, receipt)
    return path, receipt


def _check_dry_validation(plan):
    approved = plan.approval
    _require(approved.dry_validation_receipt is not None and approved.dry_validation_sha256 is not None,
        'publication/startup requires the new pinned successful constructorless dry validation.')
    dry = _read_evidence(approved.dry_validation_receipt, approved.dry_validation_sha256, 'new coast dry validation')
    required = dict(kind=DRY_KIND, version=VERSION, passed=True, transition_kind=KIND,
        parent_run_id=approved.parent_run_id, parent_source_commit=approved.parent_source_commit,
        parent_checkpoint_sha256=approved.parent_checkpoint_sha256, epoch_myr=plan.receipt['epoch_myr'],
        transition_producer_sha256=approved.expected_producer_sha256, runtime_source_seal_sha256=approved.runtime_source_seal_sha256,
        parent_preservation_sha256=approved.parent_preservation_sha256, policy_review_sha256=approved.policy_review_sha256,
        authority_sha256=approved.authority_sha256, boundary_sha256=approved.boundary_sha256,
        parent_full_state_fingerprint=plan.receipt['parent_full_state_fingerprint'], child_full_state_fingerprint=plan.receipt['child_full_state_fingerprint'],
        nonpolicy_state_fingerprint=plan.receipt['nonpolicy_state_fingerprint'], all_preexisting_arrays_exact=True,
        preexisting_array_count=plan.receipt['preexisting_array_count'], inherited_frame_count=plan.receipt['inherited_frame_count'],
        next_output_myr=plan.receipt['next_output_myr'], neutral_reference_sha256=plan.receipt['neutral_reference_sha256'],
        future_reference_copy_exact=True, policy_delta=POLICY_DELTA, collision_coast_migration=plan.receipt['collision_coast_migration'],
        surface_erosion_version=0, parent_and_sources_unchanged=True, private_state_unchanged_after_snapshot=True,
        model_constructed=False, model_stepped=False, solve_called=False, geometry_rebuilt=False,
        physical_time_advanced_myr=0., public_child_created=False, coupled_world_trial=False)
    _require(all(dry.get(name) == value for name, value in required.items()), 'dry receipt does not bind this exact same-source policy migration.')
    candidate_name = dry.get('private_checkpoint_file')
    _require(candidate_name == 'candidate.npz', 'dry candidate path is not the declared local fresh artifact.')
    candidate = _scope(Path(approved.dry_validation_receipt).parent / candidate_name, existing=True)
    _require(_file_sha(candidate) == dry.get('private_checkpoint_sha256'), 'sealed dry candidate bytes changed.')
    with _no_model_work():
        restored, saved = checkpoint.read_checkpoint(candidate, plan.compatibility, native_engine.Simulation)
        _require(_fingerprint(vars(restored))['sha256'] == plan.receipt['child_full_state_fingerprint'] and
            saved.get('next_output_myr') == plan.receipt['next_output_myr'] and saved.get('diagnostic_only') is True,
            'sealed dry candidate state or schedule differs.')
        _assert_policy_only(plan.original, restored, plan.receipt['epoch_myr'])
    return dry


def _evidence_files(approved):
    return {'parent-preservation.json': (approved.parent_preservation_receipt, approved.parent_preservation_sha256),
        'runtime-source-seal.json': (approved.runtime_source_seal, approved.runtime_source_seal_sha256),
        'coast-policy-review.json': (approved.policy_review_receipt, approved.policy_review_sha256),
        'application-authority.json': (approved.authority_receipt, approved.authority_sha256),
        'accepted-boundary.json': (approved.boundary_receipt, approved.boundary_sha256),
        'coast-dry-validation.json': (approved.dry_validation_receipt, approved.dry_validation_sha256)}


def _child_manifest(plan, run_id):
    manifest = deepcopy(plan.manifest)
    manifest.update(run_id=run_id, created=datetime.now(timezone.utc).isoformat(), state='paused', can_resume=True,
        branch_origin=dict(run_id=plan.approval.parent_run_id, time_myr=plan.receipt['epoch_myr'],
            checkpoint_file='inherited-parent/checkpoint.npz', checkpoint_sha256=plan.receipt['parent_checkpoint_sha256'], source_transition_kind=KIND),
        source_transition=dict(kind=KIND, version=VERSION, receipt='source-transition.json', epoch_myr=plan.receipt['epoch_myr'],
            parent_run_id=plan.approval.parent_run_id, inherited_frame_count=plan.manifest['frame_count'],
            inherited_frame_files_sha256={name: value for name, value in plan.parent_files.items() if re.fullmatch(r'frame_\d+\.(npz|json)', name)},
            inherited_source_compatibility=deepcopy(plan.receipt['parent_compatibility']), boundary_kind='supported_accepted_step_pause',
            policy_delta=deepcopy(POLICY_DELTA), policy_state_fields=list(POLICY_STATE_FIELD_NAMES), surface_erosion_version=0,
            scope='Same-source future coast-v2 surface reconstruction and explicit migration metadata; nonpolicy state and historical frame bytes exact.'),
        surface_policy_transition=dict(kind=KIND, parent_run_id=plan.approval.parent_run_id, epoch_myr=plan.receipt['epoch_myr'],
            policy_delta=deepcopy(POLICY_DELTA), migration=deepcopy(plan.receipt['collision_coast_migration']),
            authority_file='application-authority.json', authority_sha256=plan.approval.authority_sha256,
            policy_review_file='coast-policy-review.json', policy_review_sha256=plan.approval.policy_review_sha256,
            boundary_file='accepted-boundary.json', boundary_sha256=plan.approval.boundary_sha256,
            scheduler_authority='embedded durable predecessor checkpoint'),
        frame_sources=[dict(index=row['index'], time_myr=row['time_myr'], run_id=plan.approval.parent_run_id,
            source_manifest='inherited-parent/manifest.json', source_manifest_sha256=plan.receipt['parent_manifest_sha256'],
            engine_sha256=plan.compatibility['engine_sha256'], auxiliary_sources_sha256=deepcopy(plan.compatibility['auxiliary_sources_sha256']),
            files_sha256={suffix: plan.parent_files[f"frame_{row['index']:04d}{suffix}"] for suffix in ('.npz', '.json')},
            inherited=True, parent_frame_source=deepcopy(plan.manifest.get('frame_sources', [])[index])
                if index < len(plan.manifest.get('frame_sources', [])) else None)
            for index, row in enumerate(plan.manifest['frames'])], **deepcopy(plan.compatibility))
    for name in ('error', 'resume_reason', 'recovery', 'durable_source_recovery'):
        manifest.pop(name, None)
    return manifest


def _verify_child_artifacts(plan, child, manifest=None, receipt=None):
    """Read-only verification of a paused published/staged child; no publication."""
    _approved(plan.approval)
    _check_reviews(plan.approval)
    _check_dry_validation(plan)
    child = _scope(child, existing=True)
    manifest = json.loads((child / 'manifest.json').read_text()) if manifest is None else manifest
    actual_receipt = json.loads((child / 'source-transition.json').read_text())
    receipt = actual_receipt if receipt is None else receipt
    _require(receipt == actual_receipt and all(receipt.get(name) == value for name, value in plan.receipt.items()),
        'child transition receipt differs from its exact fresh plan.')
    _require(manifest.get('run_id') == receipt.get('child_run_id') and manifest.get('state') == 'paused' and
        manifest.get('can_resume') is True and manifest.get('error') is None and
        manifest.get('next_output_myr') == plan.manifest['next_output_myr'] and manifest.get('frames') == plan.manifest['frames'] and
        manifest.get('frame_count') == plan.manifest['frame_count'] and manifest.get('config') == plan.manifest['config'] and
        manifest.get('source_transition', {}).get('kind') == KIND and
        manifest.get('source_transition', {}).get('policy_delta') == POLICY_DELTA and
        manifest.get('surface_policy_transition', {}).get('migration') == plan.receipt['collision_coast_migration'],
        'child manifest identity, pause, policy or accepted schedule differs.')
    _require(isinstance(manifest.get('created'), str) and isinstance(manifest.get('run_id'), str) and
        re.fullmatch(r'[A-Za-z0-9_-]{1,80}', manifest['run_id']) and manifest['run_id'] != plan.approval.parent_run_id,
        'child manifest publication identity or timestamp is invalid.')
    expected_manifest = _child_manifest(plan, manifest['run_id'])
    expected_manifest['created'] = manifest['created']
    _require(expected_manifest == manifest, 'child manifest contains an undeclared provenance or scheduling change.')
    checkpoint.verify_compatibility(manifest, plan.compatibility)
    _require(receipt.get('published_state') == 'paused' and receipt.get('inherited_parent_copy_exact') is True and
        receipt.get('child_decoded_state_exact') is True and receipt.get('dry_validation_sha256') == plan.approval.dry_validation_sha256 and
        receipt.get('permanent_source_boundary_file') == 'checkpoint.source-boundary.npz' and
        receipt.get('permanent_source_boundary_sha256') == receipt.get('child_checkpoint_sha256'), 'publication evidence is incomplete.')
    expected = {'engine.py': plan.compatibility['engine_sha256'], **plan.compatibility['auxiliary_sources_sha256'],
        'initial.json': plan.parent_files['initial.json'], 'config.json': plan.parent_files['config.json'],
        'checkpoint.npz': receipt['child_checkpoint_sha256'], 'checkpoint.source-boundary.npz': receipt['child_checkpoint_sha256']}
    expected.update({name: value for name, value in plan.parent_files.items() if re.fullmatch(r'frame_\d+\.(npz|json)', name)})
    expected.update({name: sha for name, (_, sha) in _evidence_files(plan.approval).items()})
    for name, sha in expected.items():
        _require(_file_sha(child / name) == sha, 'child artifact differs from sealed bytes: ' + name)
    _require(_files(child / 'inherited-parent') == plan.parent_files, 'copied complete predecessor tree differs.')
    _check_source(child, plan.compatibility, plan.compatibility, plan.engine_source, plan.helper_sources)
    history_tools._verify_recursive_history(child, manifest)
    with _no_model_work():
        for name in ('checkpoint.npz', 'checkpoint.source-boundary.npz'):
            restored, saved = checkpoint.read_checkpoint(child / name, plan.compatibility, native_engine.Simulation)
            _require(saved == manifest and _fingerprint(vars(restored))['sha256'] == plan.receipt['child_full_state_fingerprint'],
                'child decoded checkpoint differs from the exact policy-only state: ' + name)
            _assert_policy_only(plan.original, restored, plan.receipt['epoch_myr'])
    _require(_files(plan.parent) == plan.parent_files and
        _current_context(plan.approval) == (plan.compatibility, plan.engine_source, plan.helper_sources), 'parent or runtime changed during child verification.')
    return True


def create_transition(plan, *, output_root=None, run_id=None):
    """Explicit reviewed publication only; supported model control is elsewhere."""
    _fresh_plan(plan)
    _check_dry_validation(plan)
    _require(output_root is not None, 'publication requires an explicit LIVE/output/runs root.')
    output_root = _scope(output_root, existing=True)
    _require(output_root == (ROOT / 'output/runs').resolve() and output_root.is_dir(), 'publication root must be this isolated LIVE/output/runs.')
    run_id = run_id or datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    _require(isinstance(run_id, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id) and run_id != plan.approval.parent_run_id, 'invalid fresh child identity.')
    public = output_root / run_id
    _require(not public.exists() and not public.is_relative_to(plan.parent) and not plan.parent.is_relative_to(public),
        'child already exists or overlaps the preserved predecessor.')
    stage = output_root / ('.coast-' + uuid.uuid4().hex[:12])
    names = {'engine.py', *plan.helper_sources, 'initial.json', 'config.json', 'checkpoint.npz', 'checkpoint.source-boundary.npz',
        'checkpoint.npz.' + '0' * 32 + '.tmp', 'manifest.json', 'source-transition.json', *_evidence_files(plan.approval)}
    names.update(name for name in plan.parent_files if re.fullmatch(r'frame_\d+\.(npz|json)', name))
    history_tools._check_publication_paths(plan.parent, stage, public, plan.parent_files, names)
    stage.mkdir()
    # If any check fails, retain this unpublished private stage for truthful diagnosis.
    shutil.copytree(plan.parent, stage / 'inherited-parent')
    _require(_files(stage / 'inherited-parent') == plan.parent_files, 'copied complete predecessor tree differs.')
    for name, (source, sha) in _evidence_files(plan.approval).items():
        shutil.copy2(_scope(source, existing=True), stage / name)
        _require(_file_sha(stage / name) == sha, 'copied immutable evidence differs: ' + name)
    (stage / 'engine.py').write_bytes(plan.engine_source)
    for name, value in plan.helper_sources.items():
        (stage / name).write_bytes(value)
    for name in ('initial.json', 'config.json'):
        shutil.copy2(plan.parent / name, stage / name)
    for index in range(plan.manifest['frame_count']):
        for suffix in ('.npz', '.json'):
            name = f'frame_{index:04d}{suffix}'
            shutil.copy2(plan.parent / name, stage / name)
            _require(_file_sha(stage / name) == plan.parent_files[name], 'accepted inherited frame bytes changed.')
    manifest = _child_manifest(plan, run_id)
    with _no_model_work():
        checkpoint.write_checkpoint(stage / 'checkpoint.npz', plan.derived, manifest, plan.compatibility)
        restored, saved = checkpoint.read_checkpoint(stage / 'checkpoint.npz', plan.compatibility, native_engine.Simulation)
        _require(saved == manifest and _fingerprint(vars(restored)) == _fingerprint(vars(plan.derived)), 'staged child checkpoint readback differs.')
        _assert_policy_only(plan.original, restored, plan.receipt['epoch_myr'])
    shutil.copy2(stage / 'checkpoint.npz', stage / 'checkpoint.source-boundary.npz')
    receipt = deepcopy(plan.receipt)
    receipt.update(child_run_id=run_id, child_checkpoint_sha256=_file_sha(stage / 'checkpoint.npz'),
        permanent_source_boundary_file='checkpoint.source-boundary.npz', permanent_source_boundary_sha256=_file_sha(stage / 'checkpoint.source-boundary.npz'),
        dry_validation_sha256=plan.approval.dry_validation_sha256, inherited_parent_copy_exact=True,
        child_decoded_state_exact=True, published_state='paused')
    _fresh_plan(plan)
    _check_dry_validation(plan)
    _write_fresh_json(stage / 'source-transition.json', receipt)
    _verify_child_artifacts(plan, stage, manifest, receipt)
    _write_fresh_json(stage / 'manifest.json', manifest)
    _require(not public.exists(), 'child identity appeared before commit.')
    stage.replace(public)
    return public, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--approval-json', type=Path, required=True)
    parser.add_argument('--approval-sha256', required=True)
    parser.add_argument('--mode', choices=('plan', 'dry-run', 'create'), default='plan')
    parser.add_argument('--validation-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    approved = approval_from_json(args.approval_json, args.approval_sha256)
    plan = plan_transition(args.parent, approval=approved)
    path, receipt = None, plan.receipt
    if args.mode == 'dry-run':
        _require(args.validation_root is not None, 'dry mode needs a fresh LIVE/tmp validation root.')
        path, receipt = validate_dry_run(plan, validation_root=args.validation_root)
    elif args.mode == 'create':
        path, receipt = create_transition(plan, output_root=args.output_root, run_id=args.run_id)
    if args.receipt:
        _write_fresh_json(args.receipt, receipt)
    print(json.dumps(dict(kind=KIND, mode=args.mode, parent_run_id=approved.parent_run_id,
        child_run_id=receipt.get('child_run_id'), epoch_myr=plan.receipt['epoch_myr'],
        inherited_frame_count=plan.receipt['inherited_frame_count'], next_output_myr=plan.receipt['next_output_myr'],
        preexisting_array_count=plan.receipt['preexisting_array_count'], all_preexisting_arrays_exact=True,
        entire_typed_state_exact=False, coast_policy_before=0, coast_policy_after=2, surface_erosion_version=0,
        model_constructed=False, model_stepped=False, physical_time_advanced_myr=0., result_path=str(path) if path else None),
        sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
