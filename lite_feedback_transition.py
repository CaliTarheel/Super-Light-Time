"""Reviewed accepted-boundary transition for Lite's optional mechanical feedback.

This producer never resumes an old checkpoint against a new compatibility seal.
It decodes the independently preserved predecessor under that predecessor's
recorded closure, imports no predecessor source, and initializes only the two
declared future policies. A frozen approval names every reviewed evidence hash.
The producer is deliberately unready until the root review supplies that exact
accepted pause or valid durable stop and the completed source/validation receipts.
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
KIND = 'lite_mechanical_feedback_source_transition'
VERSION = 1
SOURCE_REVIEW_KIND = 'lite_feedback_runtime_source_delta_review'
NUMERICAL_REVIEW_KIND = 'lite_feedback_numerical_contact_repair_review'
# Membership is a ceiling, never an alternative to the exact approved hashes.
ALLOWED_CHANGED_HELPERS = frozenset({'effective_subduction.py', 'plate_balance.py',
    'balance_force_ledger.py', 'force_rifting.py', 'plate_limit_analysis.py', 'backarc.py',
    'mesh_history.py', 'native_frame_sampling.py', 'bounded_gravity.py'})
REQUIRED_CHANGED_HELPERS = frozenset({'effective_subduction.py', 'plate_balance.py',
    'balance_force_ledger.py', 'force_rifting.py', 'plate_limit_analysis.py'})
ALLOWED_ADDED_HELPERS = frozenset({'effective_subduction_carrier_traction.py'})
POLICY_STATE_FIELDS = frozenset({'force_rifting_version', 'force_rifting_state',
    'force_rifting_diagnostics', 'force_rifting_policy', 'effective_subduction_carrier_traction'})


@dataclass(frozen=True)
class TransitionApproval:
    """The root's locked selection of reviewed boundary and policy evidence.

    Paths name local evidence, while hashes prevent their contents changing
    after review. Controls are canonical immutable JSON, not mutable mappings.
    A caller-created replacement is not accepted as the module's root approval.
    """
    parent_run_id: str
    parent_checkpoint_sha256: str
    parent_preservation_receipt: Path
    parent_preservation_sha256: str
    runtime_source_seal: Path
    runtime_source_seal_sha256: str
    source_delta_receipt: Path
    source_delta_receipt_sha256: str
    controls_json: str
    review_ready: bool = False
    boundary_kind: str = 'paused'
    stop_evidence_receipt: Path | None = None
    stop_evidence_sha256: str | None = None
    numerical_repair_receipt: Path | None = None
    numerical_repair_sha256: str | None = None


# Set only after completed tests, independent source review and accepted boundary.
# A healthy running predecessor is never a ready activation boundary.
REVIEWED_APPROVAL: TransitionApproval | None = TransitionApproval(
    parent_run_id='20261002-140444-3e196a',
    parent_checkpoint_sha256='b8bc7145ec519dd93ce4ded4dfbf0501f4bb28bbefae4224d8318e7a07ccdd15',
    parent_preservation_receipt=Path('E:\\Deep-Time\\Deep-Time-asein-lite-20261002\\reviews\\asein-integration\\preserved-stop-20261002-140444-3e196a-43myr\\preservation.json'),
    parent_preservation_sha256='fb8ccbd57ffbd272ff69f15d7234e52bbcdeccc43cd453c9d85bdacbbed78c29',
    runtime_source_seal=ROOT / 'reviews/asein-integration/lite-feedback-runtime-source-seal.json',
    runtime_source_seal_sha256='a21824864fcb7fad17e00fdb1081643de50814bc25811ecb7d094445c22dde6e',
    source_delta_receipt=ROOT / 'reviews/asein-integration/lite-feedback-runtime-source-delta-review.json',
    source_delta_receipt_sha256='75522903d3c4ef995e1f61764ecaa09f5d4ba80d56f9ed22278498cb909a491f',
    controls_json='{"carrier_traction":{"alpha":0.1,"enabled":true,"version":1},"force_limit_rifting":{"breakup_stretch":3.0,"enabled":true,"forbid_cratons":true,"heal_myr":50.0,"inherited_weakness":true,"ocean_breakup_opening_km":30.0,"rescan_interval_myr":10.0,"rift_width_km":150.0,"search_policy":{"max_axes":10,"mode":"bounded_axes","version":1}}}',
    review_ready=True, boundary_kind='durable_stopped',
    stop_evidence_receipt=Path('E:\\Deep-Time\\Deep-Time-asein-lite-20261002\\reviews\\asein-integration\\preserved-stop-20261002-140444-3e196a-43myr\\status-error.json'),
    stop_evidence_sha256='9cbe37ad9bd1a7811cac55368be7df6e631f59abd4c86d30003e2b1b369131a9',
    numerical_repair_receipt=ROOT / 'reviews/asein-integration/lite-feedback-numerical-contact-repair-review.json',
    numerical_repair_sha256='619be28341a6ec43a04cb817b3f6300da7210f32e9b53cecc9ab607324838bf7')


def _require(condition, message):
    if not condition:
        raise ValueError('Lite feedback transition: ' + message)


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


def _approved(approval=None):
    selected = REVIEWED_APPROVAL if approval is None else approval
    _require(isinstance(selected, TransitionApproval) and selected is REVIEWED_APPROVAL and
        selected.review_ready is True, 'exact accepted boundary and completed source review are not sealed yet.')
    _require(isinstance(selected.parent_run_id, str) and
        re.fullmatch(r'[A-Za-z0-9_-]{1,80}', selected.parent_run_id) is not None,
        'approved parent identity is invalid.')
    for value in (selected.parent_checkpoint_sha256, selected.parent_preservation_sha256,
            selected.runtime_source_seal_sha256, selected.source_delta_receipt_sha256):
        _require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
            'approved evidence hash is invalid.')
    _require(selected.boundary_kind in ('paused', 'durable_stopped'), 'approved boundary kind is unsupported.')
    for path, value, purpose in ((selected.stop_evidence_receipt, selected.stop_evidence_sha256, 'stop evidence'),
            (selected.numerical_repair_receipt, selected.numerical_repair_sha256, 'numerical repair')):
        _require((path is None and value is None) or
            (isinstance(path, Path) and isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None),
            'approved ' + purpose + ' must name a path and exact hash together.')
    _require(selected.boundary_kind != 'durable_stopped' or
        selected.stop_evidence_receipt is not None and selected.numerical_repair_receipt is not None,
        'a stopped boundary requires sealed failure evidence and a validated numerical repair.')
    return selected


def _read_evidence(path, sha256, description):
    path = Path(path)
    _require(path.is_file() and _file_sha(path) == sha256, description + ' changed or is missing.')
    return json.loads(path.read_text(encoding='utf-8'))


def _physical_state(value):
    result = deepcopy(vars(value))
    for name in POLICY_STATE_FIELDS:
        result.pop(name, None)
    if 'config' in result:
        # Parent normalization already contains this disabled top-level policy;
        # remove it on both sides rather than manufacturing a legacy setting.
        result['config'].pop('force_limit_rifting', None)
        result['config'].get('effective_subduction', {}).pop('carrier_traction', None)
    return result


def _current_context(approval):
    expected = server.SimulationManager.compatibility()
    seal = _read_evidence(approval.runtime_source_seal, approval.runtime_source_seal_sha256,
        'independently frozen runtime source seal')
    _require(seal == expected['auxiliary_sources_sha256'],
        'current helper closure differs from the independently frozen runtime source seal.')
    _require(_file_sha(ROOT / 'tectonics.py') == expected['engine_sha256'] and
        all(_file_sha(ROOT / name) == value for name, value in expected['auxiliary_sources_sha256'].items()),
        'runtime files changed after local source capture; reload only after source freeze.')
    return expected, server.ENGINE_SOURCE, dict(server.AUXILIARY_SOURCES)


def _policy_apis():
    import force_rifting
    import effective_subduction_carrier_traction
    return force_rifting, effective_subduction_carrier_traction


def _controls(approval):
    raw = json.loads(approval.controls_json)
    _require(type(raw) is dict and set(raw) == {'force_limit_rifting', 'carrier_traction'},
        'reviewed controls must name exactly the two declared feedback policies.')
    fracture, carrier = _policy_apis()
    normalized = dict(force_limit_rifting=fracture.normalize(raw['force_limit_rifting']),
        carrier_traction=carrier.normalize(raw['carrier_traction']))
    _require(_json_bytes(normalized).decode() == approval.controls_json,
        'approved controls must be fully normalized canonical JSON.')
    _require(normalized['force_limit_rifting']['enabled'] is True and
        normalized['carrier_traction']['enabled'] is True, 'both reviewed feedback policies must be explicitly enabled.')
    return normalized


def _activate(original, controls):
    derived = deepcopy(original)
    fracture, carrier = _policy_apis()
    report = dict(force_limit_rifting=fracture.upgrade(derived, deepcopy(controls['force_limit_rifting'])),
        carrier_traction=carrier.upgrade(derived, deepcopy(controls['carrier_traction'])))
    return derived, report


def _check_history(parent, simulation, saved, external, approval, header, stop_evidence):
    _require(saved['run_id'] == external['run_id'] == parent.name == approval.parent_run_id,
        'parent experiment identity differs from the reviewed activation.')
    _require(saved.get('config') == external.get('config') == simulation.config,
        'saved parent configuration and decoded state differ.')
    if approval.boundary_kind == 'paused':
        _require(saved.get('state') == external.get('state') == 'paused',
            'parent must finish its accepted-step pause before a transition is planned.')
        _require(float(saved.get('time_myr', -1)) == float(external.get('time_myr', -2)) == float(simulation.t),
            'pause manifest does not name the exact accepted checkpoint.')
    else:
        # A recovery checkpoint is a complete accepted generation even when
        # the public manifest records a later failed trial. Neither manifest
        # is rewritten or falsely marked paused to make this transition fit.
        epoch = float(simulation.t)
        _require(saved.get('state') == 'interrupted' and saved.get('error') is None and
            header.get('time_myr') == saved.get('time_myr') == saved.get('integration_time_myr') ==
            saved.get('checkpoint_time_myr') == epoch,
            'stopped recovery needs the exact valid durable embedded checkpoint generation.')
        _require(external.get('state') == 'error' and isinstance(external.get('error'), str) and
            bool(external['error'].strip()) and external.get('integration_time_myr') ==
            external.get('checkpoint_time_myr') == epoch,
            'external stopped manifest must retain the actual failure and matching durable integration epoch.')
        _require(isinstance(stop_evidence, dict) and stop_evidence.get('state') == 'error' and
            stop_evidence.get('run_id') == approval.parent_run_id and stop_evidence.get('error') == external['error'] and
            stop_evidence.get('integration_time_myr') == stop_evidence.get('checkpoint_time_myr') == epoch and
            stop_evidence.get('time_myr') == external.get('time_myr') and
            stop_evidence.get('frame_count') == external.get('frame_count'),
            'sealed stop evidence does not match the exact failed external generation and durable epoch.')
    count, frames = saved.get('frame_count'), saved.get('frames', [])
    _require(type(count) is int and count > 0 and len(frames) == count and
        external.get('frames') == frames and external.get('frame_count') == count,
        'parent frame history is incomplete or differs between committed generations.')
    if approval.boundary_kind == 'durable_stopped':
        _require(external.get('time_myr') == frames[-1].get('time_myr') and
            external['time_myr'] <= float(simulation.t),
            'stopped public epoch must be the last accepted saved frame, never a future trial.')
    _require(saved.get('source_transition') == external.get('source_transition') and
        saved.get('frame_sources') == external.get('frame_sources'),
        'predecessor source lineage differs between committed generations.')
    pending = saved.get('next_output_myr')
    _require(type(pending) in (float, int) and np.isfinite(pending) and
        float(simulation.t) < pending <= simulation.config['duration_myr'],
        'exact pending output scheduler is missing or invalid.')
    for index, row in enumerate(frames):
        stem = parent / f'frame_{index:04d}'
        _require(stem.with_suffix('.npz').is_file() and stem.with_suffix('.json').is_file(),
            'an inherited frame file is missing.')
        metadata = json.loads(stem.with_suffix('.json').read_text(encoding='utf-8'))
        _require(row.get('index') == metadata.get('index') == index and
            row.get('time_myr') == metadata.get('time_myr') and row['time_myr'] <= simulation.t,
            'an inherited frame disagrees with its committed history index.')


def _check_numerical_review(approval, changed):
    repair = changed.get('bounded_gravity.py')
    if repair is None:
        _require(approval.numerical_repair_receipt is None and approval.boundary_kind != 'durable_stopped',
            'stopped recovery cannot omit its reviewed numerical source repair.')
        return None
    _require(approval.numerical_repair_receipt is not None, 'bounded gravity changes require a sealed numerical repair review.')
    reviewed = _read_evidence(approval.numerical_repair_receipt, approval.numerical_repair_sha256,
        'completed numerical contact repair review receipt')
    _require(reviewed.get('kind') == NUMERICAL_REVIEW_KIND and reviewed.get('version') == VERSION and
        reviewed.get('ready') is True and reviewed.get('parent_run_id') == approval.parent_run_id and
        reviewed.get('parent_checkpoint_sha256') == approval.parent_checkpoint_sha256 and
        reviewed.get('helper') == 'bounded_gravity.py' and reviewed.get('parent_sha256') == repair['parent_sha256'] and
        reviewed.get('child_sha256') == repair['child_sha256'] and reviewed.get('physical_guards_unchanged') is True and
        reviewed.get('original_policy_interval_passed') is True and reviewed.get('feedback_policy_interval_passed') is True,
        'numerical repair review must prove the original and feedback intervals under the exact unchanged physical guards.')
    validations = reviewed.get('validation_receipts')
    _require(isinstance(validations, list) and len(validations) >= 2,
        'numerical repair requires separately recorded original/feedback validation evidence.')
    seen = set()
    for row in validations:
        _require(type(row) is dict and set(row) == {'path', 'sha256'} and
            isinstance(row['path'], str) and isinstance(row['sha256'], str),
            'numerical validation receipt must name an exact file and hash.')
        path = Path(row['path'])
        if not path.is_absolute():
            path = Path(approval.numerical_repair_receipt).parent / path
        path = path.resolve()
        _require(path not in seen, 'numerical interval validation evidence must name distinct receipts.')
        seen.add(path)
        _read_evidence(path, row['sha256'], 'numerical interval validation evidence')
    return reviewed


def _retain_numerical_validation_evidence(approval, changed, child):
    """Keep simple relative proof references valid beside the unchanged review.

    Absolute and nested references retain their original external meaning;
    this producer does not rewrite a reviewed receipt to invent portability.
    Only an exact simple basename can be copied without changing that meaning.
    """
    reviewed = _check_numerical_review(approval, changed)
    if reviewed is None:
        return []
    retained = []
    reserved = {'manifest.json', 'source-transition.json', 'checkpoint.npz',
        'checkpoint.policy-boundary.npz', 'config.json', 'initial.json', 'engine.py',
        'inherited-parent', 'parent-preservation.json', 'source-delta-review.json',
        'runtime-source-seal.json', 'stop-evidence.json', 'numerical-repair-review.json'}
    reserved |= {name + '.tmp' for name in reserved}
    for row in reviewed['validation_receipts']:
        reference = Path(row['path'])
        source = reference if reference.is_absolute() else Path(approval.numerical_repair_receipt).parent / reference
        source = source.resolve()
        item = dict(reference_path=row['path'], source_path=str(source), sha256=row['sha256'],
            storage='external_reference', child_file=None)
        if not reference.is_absolute() and reference.name == row['path'] and reference.name not in ('.', '..'):
            destination = child / reference.name
            namespace = reference.name.casefold()
            managed_history = re.fullmatch(r'(?:frame_\d+\.(?:json|npz)|checkpoint(?:\.[a-z0-9_-]+)*\.npz)(?:\.tmp)?', namespace)
            _require(namespace not in reserved and managed_history is None and not destination.exists(),
                'numerical validation proof basename collides with child source/history metadata.')
            _read_evidence(source, row['sha256'], 'numerical interval validation evidence before retention')
            shutil.copy2(source, destination)
            _require(_file_sha(destination) == row['sha256'], 'copied numerical interval proof differs from its sealed receipt.')
            item.update(storage='child_relative_exact_copy', child_file=reference.name)
        retained.append(item)
    return retained


def _check_source_delta(parent, recorded, expected, engine, helpers, approval, review, controls):
    _require(_file_sha(parent / 'engine.py') == recorded['engine_sha256'], 'parent engine source is damaged.')
    _require(_sha(engine) == expected['engine_sha256'] == recorded['engine_sha256'],
        'engine entrypoint changes are outside this transition.')
    for key in ('numpy_version', 'flow_backend', 'material_backend'):
        _require(recorded.get(key) == expected.get(key), key + ' changed outside the reviewed policy transition.')
    prior = recorded['auxiliary_sources_sha256']
    _require(all(Path(name).name == name and name.endswith('.py') for name in prior),
        'source closure contains invalid helper paths.')
    _require(all(_file_sha(parent / name) == value for name, value in prior.items()),
        'a parent helper source is damaged.')
    captured = server.capture_auxiliary_sources((parent / 'engine.py').read_bytes(), root=parent)
    _require({name: _sha(payload) for name, payload in captured.items()} == prior,
        'parent recorded closure is not its complete saved local import closure.')
    current = {name: _sha(payload) for name, payload in helpers.items()}
    _require(current == expected['auxiliary_sources_sha256'], 'current frozen source capture and compatibility differ.')
    changed = {name: dict(parent_sha256=prior[name], child_sha256=current[name])
        for name in prior.keys() & current.keys() if prior[name] != current[name]}
    added = {name: current[name] for name in current.keys() - prior.keys()}
    removed = sorted(prior.keys() - current.keys())
    _require(review.get('kind') == SOURCE_REVIEW_KIND and review.get('version') == VERSION and
        review.get('ready') is True and review.get('parent_run_id') == approval.parent_run_id and
        review.get('parent_checkpoint_sha256') == approval.parent_checkpoint_sha256 and
        review.get('boundary_kind') == approval.boundary_kind and
        review.get('parent_compatibility_sha256') == _sha(_json_bytes(recorded)) and
        review.get('child_compatibility_sha256') == _sha(_json_bytes(expected)) and
        review.get('policy') == controls, 'source delta review does not name this exact boundary, closure and policy.')
    _require(not removed and changed == review.get('changed_helpers') and added == review.get('added_helpers') and
        removed == review.get('removed_helpers'), 'runtime source delta differs from the exact reviewed helper hashes.')
    _require(REQUIRED_CHANGED_HELPERS <= set(changed) <= ALLOWED_CHANGED_HELPERS and
        set(added) == ALLOWED_ADDED_HELPERS, 'reviewed source membership is outside the narrow Lite feedback scope.')
    _check_numerical_review(approval, changed)
    _require(review.get('numerical_repair_review_sha256') == approval.numerical_repair_sha256,
        'source delta review and numerical repair review are not bound together.')
    return dict(changed_helpers=changed, added_helpers=added, removed_helpers=removed)


def _check_activation(original, derived, controls):
    before, after = _fingerprint(_physical_state(original)), _fingerprint(_physical_state(derived))
    _require(before == after, 'activation altered an inherited physical/clock/RNG/history field.')
    expected = deepcopy(original.config)
    expected['force_limit_rifting'] = deepcopy(controls['force_limit_rifting'])
    expected['effective_subduction']['carrier_traction'] = deepcopy(controls['carrier_traction'])
    _require(derived.config == expected, 'activation changed configuration outside the two declared policies.')
    added, removed = set(vars(derived)) - set(vars(original)), set(vars(original)) - set(vars(derived))
    _require(not removed and added == POLICY_STATE_FIELDS and not POLICY_STATE_FIELDS & set(vars(original)),
        'activation must add exactly its declared new policy state; preexisting feedback state is unsupported.')
    _require(type(derived.force_rifting_version) is int and derived.force_rifting_version == 1 and
        type(derived.force_rifting_state) is dict and derived.force_rifting_state == {},
        'force-rift policy must start with no inherited candidate paths or opening/work credit.')
    diagnostics = derived.force_rifting_diagnostics
    _require(type(diagnostics) is dict and diagnostics == dict(version=1, checks=0, commits=0, refusals=[]) and
        all(type(diagnostics[name]) is int for name in ('version', 'checks', 'commits')),
        'force-rift diagnostics must start without retrospective checks or commits.')
    # The exact producer metadata schemas below are declared by these APIs;
    # normalized controls and zero clocks/work must not mask extra mutable state.
    force_policy = dict(version=1, mode='bounded_axes', parameters=controls['force_limit_rifting'],
        activation_myr=float(original.t), epoch_myr=float(original.t),
        history='Future matched cut observations only; virtual development, not realized material extension.')
    _require(_fingerprint(derived.force_rifting_policy) == _fingerprint(force_policy),
        'force-rift policy must have exactly declared future-only bounded-search metadata.')
    _, carrier_api = _policy_apis()
    carrier = dict(version=1, enabled=True, activation_myr=float(original.t),
        parameters=controls['carrier_traction'], retrospective_work_j=0., law=carrier_api.POLICY)
    _require(_fingerprint(derived.effective_subduction_carrier_traction) == _fingerprint(carrier),
        'carrier traction must start with exactly declared policy metadata and no historical work.')
    for name in added:
        _require(not _fingerprint(getattr(derived, name))['arrays'], 'new policy metadata must not introduce physical arrays.')
    return dict(inherited_state_exact=True, preexisting_array_count=len(before['arrays']),
        inherited_state_fingerprint=before['sha256'], added_state_fields=sorted(added),
        rng_unchanged=True, clocks_unchanged=True, initiation_history_unchanged=True,
        physical_time_advanced_myr=0., force_solve_performed=False,
        past_fracture_loading_reconstructed=False, past_carrier_work_reconstructed=False)


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
    """Validate a preserved accepted-boundary child plan; write nothing."""
    approved = _approved(approval)
    controls = _controls(approved)
    parent = Path(parent).resolve()
    _require(parent.is_dir(), 'parent history directory does not exist.')
    before = _files(parent)
    cp = parent / 'checkpoint.npz'
    _require(before.get('checkpoint.npz') == approved.parent_checkpoint_sha256,
        'parent checkpoint differs from the exact reviewed accepted-step boundary.')
    preservation = _read_evidence(approved.parent_preservation_receipt, approved.parent_preservation_sha256,
        'independent parent preservation receipt')
    _require(preservation.get('run_id') == parent.name and
        {row['path']: row['sha256'] for row in preservation['files']} == before,
        'sealed parent files differ from the independent immutable preservation.')
    review = _read_evidence(approved.source_delta_receipt, approved.source_delta_receipt_sha256,
        'completed runtime source delta review receipt')
    stop = (_read_evidence(approved.stop_evidence_receipt, approved.stop_evidence_sha256, 'sealed stopped status evidence')
        if approved.stop_evidence_receipt is not None else None)
    header = checkpoint.checkpoint_header(cp)
    expected, engine, helpers = _current_context(approved)
    delta = _check_source_delta(parent, header['compatibility'], expected, engine, helpers, approved, review, controls)
    original, saved = checkpoint.read_checkpoint(cp, header['compatibility'], native_engine.Simulation)
    external = json.loads((parent / 'manifest.json').read_text(encoding='utf-8'))
    _check_history(parent, original, saved, external, approved, header, stop)
    _require(getattr(original, 'effective_subduction_version', None) == 1 and
        original.config.get('effective_subduction', {}).get('enabled') is True,
        'parent must use the declared Lite traction law.')
    _require(original.config.get('force_limit_rifting', {}).get('enabled', False) is False and
        not original.config['effective_subduction'].get('carrier_traction', {}).get('enabled', False),
        'parent feedback policies must be disabled before this explicit activation.')
    derived, activation = _activate(original, controls)
    proof = _check_activation(original, derived, controls)
    _require(_files(parent) == before, 'parent changed during planning; wait for a stable accepted pause.')
    receipt = dict(kind=KIND, version=VERSION, parent_run_id=parent.name,
        boundary_kind=approved.boundary_kind, stop_evidence_sha256=approved.stop_evidence_sha256,
        numerical_repair_review_sha256=approved.numerical_repair_sha256,
        epoch_myr=float(original.t), integration_steps=int(original.steps),
        parent_checkpoint_sha256=before['checkpoint.npz'], parent_manifest_sha256=before['manifest.json'],
        parent_header=header, parent_files_sha256=before, parent_file_count=len(before),
        parent_preservation_receipt_sha256=approved.parent_preservation_sha256,
        runtime_source_seal_sha256=approved.runtime_source_seal_sha256,
        source_delta_review_sha256=approved.source_delta_receipt_sha256,
        parent_compatibility=header['compatibility'], child_compatibility=expected,
        source_delta=delta, activated_controls=controls, activation=activation, preservation=proof,
        next_output_myr=saved['next_output_myr'], inherited_frame_count=saved['frame_count'],
        interpretation='Explicit future-only force-rift and carrier-traction policy boundary. '
            'All stored velocities and solver reports remain predecessor evidence until the next accepted successor interval.')
    if approved.boundary_kind == 'durable_stopped':
        receipt['durable_recovery'] = dict(embedded_state=saved['state'], external_state=external['state'],
            failed_public_epoch_myr=external['time_myr'], durable_epoch_myr=float(original.t),
            embedded_next_output_myr=saved['next_output_myr'],
            external_stale_next_output_myr=external.get('next_output_myr'),
            scheduler_authority='Exact embedded durable checkpoint only; failed external generation remains preserved unchanged.',
            error_evidence=external['error'], historical_manifests_rewritten=False)
    return TransitionPlan(parent, original, derived, deepcopy(saved), receipt, before, expected, engine, helpers, approved)


def create_transition(plan, *, output_root=None, run_id=None, title='Asein Lite'):
    """Publish a reviewed plan as a paused history, committing manifest last."""
    _require(isinstance(plan, TransitionPlan), 'a reviewed TransitionPlan is required.')
    approved = _approved(plan.approval)
    current, engine, helpers = _current_context(approved)
    _require(current == plan.compatibility and engine == plan.engine_source and helpers == plan.helper_sources,
        'runtime sources changed after planning.')
    _require(_files(plan.parent) == plan.parent_files, 'parent changed after planning.')
    # Mutable decoded plans are not authorizations: reproduce the complete
    # proof before trusting their stored objects, receipt or scheduler.
    fresh = plan_transition(plan.parent, approval=approved)
    _require(_fingerprint(vars(plan.original)) == _fingerprint(vars(fresh.original)) and
        _fingerprint(vars(plan.derived)) == _fingerprint(vars(fresh.derived)) and
        plan.manifest == fresh.manifest and plan.receipt == fresh.receipt and
        plan.parent_files == fresh.parent_files, 'plan state/provenance was modified after validation.')
    output_root = Path(output_root or plan.parent.parent).resolve()
    run_id = run_id or datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    _require(isinstance(run_id, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id), 'invalid child run identifier.')
    child = output_root / run_id
    _require(child != plan.parent and not child.exists(), 'child directory already exists or names its parent.')
    _require(not child.is_relative_to(plan.parent) and not plan.parent.is_relative_to(child),
        'child and parent history paths must be disjoint; recursive history nesting is unsupported.')
    origin = dict(run_id=plan.parent.name, time_myr=plan.receipt['epoch_myr'],
        checkpoint_file='inherited-parent/checkpoint.npz', checkpoint_sha256=plan.receipt['parent_checkpoint_sha256'],
        physical_policy=KIND)
    manifest = deepcopy(plan.manifest)
    manifest.update(run_id=run_id, title=title, created=datetime.now(timezone.utc).isoformat(),
        config=deepcopy(plan.derived.config), state='paused', can_resume=True, branch_origin=origin,
        source_transition=dict(kind=KIND, version=VERSION, receipt='source-transition.json',
            boundary_kind=approved.boundary_kind,
            epoch_myr=plan.receipt['epoch_myr'], parent_run_id=plan.parent.name,
            inherited_frame_count=plan.manifest['frame_count'],
            inherited_frame_files_sha256={name: value for name, value in plan.parent_files.items()
                if re.fullmatch(r'frame_\d+\.(npz|json)', name)},
            inherited_source_compatibility=deepcopy(plan.receipt['parent_compatibility']),
            scope='Future-only force-rift and carrier-traction physical policies; no inherited state evolution at import'),
        frame_sources=[dict(index=row['index'], time_myr=row['time_myr'], run_id=plan.parent.name,
            source_manifest='inherited-parent/manifest.json', source_manifest_sha256=plan.receipt['parent_manifest_sha256'],
            engine_sha256=plan.receipt['parent_compatibility']['engine_sha256'],
            auxiliary_sources_sha256=deepcopy(plan.receipt['parent_compatibility']['auxiliary_sources_sha256']),
            files_sha256={suffix: plan.parent_files[f"frame_{row['index']:04d}{suffix}"] for suffix in ('.npz', '.json')},
            inherited=True, parent_frame_source=deepcopy(plan.manifest.get('frame_sources', [])[index])
                if len(plan.manifest.get('frame_sources', [])) > index else None)
            for index, row in enumerate(plan.manifest['frames'])],
        **deepcopy(plan.compatibility))
    for name in ('error', 'recovery', 'resume_reason'):
        manifest.pop(name, None)
    if approved.boundary_kind == 'durable_stopped':
        manifest['durable_source_recovery'] = dict(kind='explicit_valid_durable_stopped_source_transition',
            parent_run_id=plan.parent.name, epoch_myr=plan.receipt['epoch_myr'],
            error_evidence_file='stop-evidence.json', error_evidence_sha256=approved.stop_evidence_sha256,
            numerical_repair_review_file='numerical-repair-review.json',
            numerical_repair_review_sha256=approved.numerical_repair_sha256,
            scheduler_authority='embedded durable predecessor checkpoint')
    child.mkdir(parents=False)
    # An interrupted publish remains an unpublished directory; it cannot modify
    # or clean any predecessor and never appears as a committed saved run.
    shutil.copytree(plan.parent, child / 'inherited-parent')
    _require(_files(child / 'inherited-parent') == plan.parent_files, 'copied parent bytes differ from the sealed history.')
    shutil.copy2(approved.parent_preservation_receipt, child / 'parent-preservation.json')
    shutil.copy2(approved.source_delta_receipt, child / 'source-delta-review.json')
    shutil.copy2(approved.runtime_source_seal, child / 'runtime-source-seal.json')
    if approved.stop_evidence_receipt is not None:
        shutil.copy2(approved.stop_evidence_receipt, child / 'stop-evidence.json')
    if approved.numerical_repair_receipt is not None:
        shutil.copy2(approved.numerical_repair_receipt, child / 'numerical-repair-review.json')
    (child / 'engine.py').write_bytes(plan.engine_source)
    for name, payload in plan.helper_sources.items():
        (child / name).write_bytes(payload)
    for index in range(plan.manifest['frame_count']):
        for suffix in ('.npz', '.json'):
            name = f'frame_{index:04d}{suffix}'
            shutil.copy2(plan.parent / name, child / name)
            _require(_file_sha(child / name) == plan.parent_files[name], 'an inherited frame changed during copying.')
    shutil.copy2(plan.parent / 'initial.json', child / 'initial.json')
    _write_json(child / 'config.json', manifest['config'])
    checkpoint.write_checkpoint(child / 'checkpoint.npz', plan.derived, manifest, plan.compatibility)
    restored, written = checkpoint.read_checkpoint(child / 'checkpoint.npz', plan.compatibility, native_engine.Simulation)
    _require(_fingerprint(vars(restored)) == _fingerprint(vars(plan.derived)), 'written child changed decoded state.')
    _require(written == manifest and written['next_output_myr'] == plan.manifest['next_output_myr'],
        'written child changed its exact scheduler/history.')
    boundary = child / 'checkpoint.policy-boundary.npz'
    shutil.copy2(child / 'checkpoint.npz', boundary)
    receipt = deepcopy(plan.receipt)
    receipt.update(child_run_id=run_id, child_checkpoint_sha256=_file_sha(child / 'checkpoint.npz'),
        permanent_policy_boundary_file=boundary.name, permanent_policy_boundary_sha256=_file_sha(boundary),
        child_decoded_state_exact=True, inherited_parent_copy_exact=True,
        published_state='paused', physical_time_advanced_myr=0., model_constructed=False, model_stepped=False)
    _require(_files(plan.parent) == plan.parent_files, 'parent changed before publication.')
    _require(_current_context(approved) == (plan.compatibility, plan.engine_source, plan.helper_sources),
        'runtime source closure changed before publication.')
    # Re-read all evidence immediately before committing the public manifest.
    _read_evidence(approved.parent_preservation_receipt, approved.parent_preservation_sha256, 'parent preservation receipt')
    _read_evidence(approved.source_delta_receipt, approved.source_delta_receipt_sha256, 'source delta review receipt')
    if approved.stop_evidence_receipt is not None:
        _read_evidence(approved.stop_evidence_receipt, approved.stop_evidence_sha256, 'sealed stopped status evidence')
    receipt['numerical_validation_evidence'] = _retain_numerical_validation_evidence(
        approved, plan.receipt['source_delta']['changed_helpers'], child)
    for item in receipt['numerical_validation_evidence']:
        if item['storage'] == 'child_relative_exact_copy':
            _require(_file_sha(child / item['child_file']) == item['sha256'],
                'retained numerical proof changed before publication.')
    _write_json(child / 'source-transition.json', receipt)
    _write_json(child / 'manifest.json', manifest)
    return child, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--create', action='store_true', help='Explicitly publish a paused child; default plans only.')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    plan = plan_transition(args.parent)
    result, child = plan.receipt, None
    if args.create:
        child, result = create_transition(plan, output_root=args.output_root, run_id=args.run_id)
    if args.receipt:
        _write_json(args.receipt, result)
    summary = dict(kind=KIND, state='paused_child_created' if child else 'read_only_plan_validated',
        parent_run_id=result['parent_run_id'], epoch_myr=result['epoch_myr'],
        integration_steps=result['integration_steps'], inherited_frame_count=result['inherited_frame_count'],
        preexisting_array_count=result['preservation']['preexisting_array_count'],
        next_output_myr=result['next_output_myr'], parent_checkpoint_sha256=result['parent_checkpoint_sha256'],
        changed_helpers=sorted(result['source_delta']['changed_helpers']),
        added_helpers=sorted(result['source_delta']['added_helpers']))
    if child:
        summary.update(child_run_id=result['child_run_id'], child_path=str(child),
            child_checkpoint_sha256=result['child_checkpoint_sha256'])
    if args.receipt:
        summary.update(receipt=str(args.receipt.resolve()), receipt_sha256=_file_sha(args.receipt))
    print(json.dumps(summary, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
