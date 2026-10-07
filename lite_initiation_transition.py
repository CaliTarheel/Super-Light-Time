"""Explicit paused-history transition to Lite's optional initiation policy.

This is a source and physical-policy boundary, not ordinary checkpoint resume.
The old source is verified as data and never executed. Existing physics/state
and historical frame bytes are retained; only declared future policy metadata
is initialized. New runtime sources are published under their actual hashes.
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
import native_engine
import server

ROOT = Path(__file__).resolve().parent
KIND = 'lite_effective_subduction_initiation_source_transition'
VERSION = 1
REVIEW_READY = True
APPROVED_PARENT_RUN_ID = '20261002-122914-072bbc'
APPROVED_PARENT_CHECKPOINT_SHA256 = '46f5c91174d8519321e8539b868536d83994be873a2f43912370f14fcf67d246'
PARENT_PRESERVATION_RECEIPT = ROOT / 'reviews/asein-integration/preserved-lite-20261002-122914-072bbc-before-initiation/preservation.json'
APPROVED_PARENT_PRESERVATION_SHA256 = '333e8b4c6eb45a85dbb20389d9b38eb1c42e57842ebf03e3c7e0c5f508ee35f9'
RUNTIME_SOURCE_SEAL = ROOT / 'reviews/asein-integration/lite-initiation-runtime-source-seal.json'
APPROVED_RUNTIME_SOURCE_SEAL_SHA256 = 'e21ce190061156b10ae51147bfc0623b4c4fb67760ddd11ae1119d5968d950f9'
APPROVED_CHANGED_HELPERS = {
    'effective_subduction.py': ('4e7638ea0b8b293382a3f73fa05e0573e617604e6bdf0af2db14f596cce38f83',
        '374893536d7d0b9862d4264846e028f088ef8d2a2f3c0089ebffbb4217841b99'),
    'native_engine.py': ('90a1805b05c5a7519010601ae703f029a41a60c5e6e2019aaa2bac16ca70034b',
        '144ef0f535242e67a0f13c6b7715e81c0f2f4567b5ef91909a6ee4b2fc669459')}
APPROVED_ADDED_HELPERS = {'effective_subduction_initiation.py':
    'c764ad7635e321fe5ba8d0a245e45920300c72df15f7b4bb7af46734fea10b14'}
POLICY_STATE_FIELDS = frozenset({'effective_subduction_initiation'})


def _require(condition, message):
    if not condition:
        raise ValueError('Lite initiation transition: ' + message)


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


def _semantic_tree(value, arrays):
    """Canonicalize only unordered sets, retaining typed array positions."""
    if isinstance(value, list):
        return [_semantic_tree(item, arrays) for item in value]
    if not isinstance(value, dict):
        return value
    if value['t'] == 'array':
        item = arrays[value['v']]
        return dict(t='array', v=dict(dtype=item.dtype.str, shape=list(item.shape),
            sha256=_sha(np.ascontiguousarray(item).tobytes())))
    payload = _semantic_tree(value['v'], arrays)
    if value['t'] == 'set':
        # Array content must be in each member before sorting, never its
        # incidental checkpoint aNNNNN reference or hash-table iteration.
        payload.sort(key=_json_bytes)
    return dict(t=value['t'], v=payload)


def _fingerprint(value):
    """Exact typed values/RNG, with set membership independent of bucket order."""
    arrays = {}
    tree = _semantic_tree(checkpoint._encode(value, arrays), arrays)
    inventory = {}
    def collect(node):
        if isinstance(node, list):
            for item in node:
                collect(item)
        elif isinstance(node, dict):
            if node['t'] == 'array':
                inventory[f'a{len(inventory):05d}'] = node['v']
            else:
                collect(node['v'])
    collect(tree)
    return dict(tree_sha256=_sha(_json_bytes(tree)), arrays=inventory,
        sha256=_sha(_json_bytes(dict(tree=tree, arrays=inventory))))


def _physical_state(value):
    result = deepcopy(vars(value))
    for name in POLICY_STATE_FIELDS:
        result.pop(name, None)
    if 'config' in result:
        result['config'].get('effective_subduction', {}).pop('initiation', None)
    return result


def _current_context():
    expected = server.SimulationManager.compatibility()
    _require(_file_sha(RUNTIME_SOURCE_SEAL) == APPROVED_RUNTIME_SOURCE_SEAL_SHA256 and
        json.loads(RUNTIME_SOURCE_SEAL.read_text(encoding='utf-8')) == expected['auxiliary_sources_sha256'],
        'current helper closure differs from the independently frozen runtime source seal.')
    _require(_file_sha(ROOT / 'tectonics.py') == expected['engine_sha256'] and
        all(_file_sha(ROOT / name) == value for name, value in expected['auxiliary_sources_sha256'].items()),
        'runtime files changed after the local source capture; reload only after source freeze.')
    return expected, server.ENGINE_SOURCE, dict(server.AUXILIARY_SOURCES)


def _policy_api():
    import effective_subduction_initiation
    return effective_subduction_initiation


def _activate(original, controls):
    """The reviewed producer owns normalization and future-only initialization."""
    derived = deepcopy(original)
    api = _policy_api()
    normalized = api.normalize(dict(enabled=True) if controls is None else controls)
    _require(normalized['enabled'] is True, 'new initiation policy must be explicitly enabled.')
    report = api.upgrade(derived, normalized)
    return derived, normalized, report


def _check_history(parent, simulation, saved, external):
    _require(saved['run_id'] == external['run_id'] == parent.name == APPROVED_PARENT_RUN_ID,
        'parent experiment identity differs from the reviewed activation.')
    _require(saved.get('state') == external.get('state') == 'paused',
        'parent must finish its accepted-step pause before a transition is planned.')
    _require(saved.get('config') == external.get('config') == simulation.config,
        'saved parent configuration and decoded state differ.')
    _require(float(saved.get('time_myr', -1)) == float(external.get('time_myr', -2)) == float(simulation.t),
        'pause manifest does not name the exact accepted checkpoint.')
    count = saved.get('frame_count')
    frames = saved.get('frames', [])
    _require(type(count) is int and count > 0 and len(frames) == count and
        external.get('frames') == frames and external.get('frame_count') == count,
        'parent frame history is incomplete or differs between committed generations.')
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


def _check_source_delta(parent, recorded, expected, engine, helpers):
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
    changed = {name: (prior[name], current[name]) for name in prior.keys() & current.keys()
               if prior[name] != current[name]}
    added = {name: current[name] for name in current.keys() - prior.keys()}
    removed = sorted(prior.keys() - current.keys())
    _require(not removed and changed == APPROVED_CHANGED_HELPERS and added == APPROVED_ADDED_HELPERS,
        'runtime source delta differs from the exact reviewed changed/added helper hashes.')
    _require(set(changed) == {'effective_subduction.py', 'native_engine.py'} and
        set(added) == {'effective_subduction_initiation.py'}, 'reviewed source membership is not the narrow Lite initiation scope.')
    return dict(changed_helpers={name: dict(parent_sha256=pair[0], child_sha256=pair[1])
        for name, pair in changed.items()}, added_helpers=added, removed_helpers=removed)


def _check_activation(original, derived, controls):
    before = _fingerprint(_physical_state(original))
    after = _fingerprint(_physical_state(derived))
    _require(before == after, 'activation altered an inherited physical/clock/RNG/history field.')
    expected_config = deepcopy(original.config)
    expected_config['effective_subduction']['initiation'] = deepcopy(controls)
    _require(derived.config == expected_config, 'activation changed configuration outside nested effective_subduction.initiation.')
    added = set(vars(derived)) - set(vars(original))
    removed = set(vars(original)) - set(vars(derived))
    _require(not removed and added <= POLICY_STATE_FIELDS, 'activation added or removed an unreviewed state field.')
    _require(not any(name in vars(original) for name in POLICY_STATE_FIELDS), 'parent already has an initiation policy state.')
    _require(added == POLICY_STATE_FIELDS, 'activation must add exactly its declared future policy state.')
    policy = derived.effective_subduction_initiation
    _require(type(policy) is dict and policy.get('version') == 1 and
        policy.get('parameters') == controls and policy.get('activation_myr') == original.t and
        policy.get('epoch_myr') == original.t and policy.get('candidates') == [] and
        policy.get('births') == [] and policy.get('events') == [] and policy.get('next_candidate_id') == 1,
        'activation must start with zero future-only history and no retroactive candidates/births/events.')
    for name in added:
        _require(not _fingerprint(getattr(derived, name))['arrays'], 'new policy metadata must not introduce physical arrays.')
    return dict(inherited_state_exact=True, preexisting_array_count=len(before['arrays']),
        inherited_state_fingerprint=before['sha256'], added_state_fields=sorted(added),
        rng_unchanged=True, clocks_unchanged=True, physical_time_advanced_myr=0.,
        force_solve_performed=False, past_initiation_history_reconstructed=False)


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


def plan_transition(parent, *, controls=None):
    """Validate and return a detached paused child plan; write nothing."""
    _require(REVIEW_READY, 'reviewed producer source hashes and parent boundary are not sealed yet.')
    parent = Path(parent).resolve()
    _require(parent.is_dir(), 'parent history directory does not exist.')
    before = _files(parent)
    cp = parent / 'checkpoint.npz'
    _require(before.get('checkpoint.npz') == APPROVED_PARENT_CHECKPOINT_SHA256,
        'parent checkpoint differs from the exact reviewed accepted-step boundary.')
    _require(_file_sha(PARENT_PRESERVATION_RECEIPT) == APPROVED_PARENT_PRESERVATION_SHA256,
        'independent parent preservation receipt changed or is missing.')
    preservation = json.loads(PARENT_PRESERVATION_RECEIPT.read_text(encoding='utf-8'))
    _require(preservation.get('run_id') == parent.name and
        {row['path']: row['sha256'] for row in preservation['files']} == before,
        'sealed parent files differ from the independent immutable preservation.')
    header = checkpoint.checkpoint_header(cp)
    expected, engine, helpers = _current_context()
    source_delta = _check_source_delta(parent, header['compatibility'], expected, engine, helpers)
    original, saved = checkpoint.read_checkpoint(cp, header['compatibility'], native_engine.Simulation)
    external = json.loads((parent / 'manifest.json').read_text(encoding='utf-8'))
    _check_history(parent, original, saved, external)
    _require(getattr(original, 'effective_subduction_version', None) == 1 and
        original.config.get('effective_subduction', {}).get('enabled') is True,
        'parent must use the declared Lite traction law.')
    _require(not original.config['effective_subduction'].get('initiation', {}).get('enabled', False),
        'initiation is already enabled in the parent.')
    derived, normalized, activation = _activate(original, controls)
    proof = _check_activation(original, derived, normalized)
    _require(_files(parent) == before, 'parent changed during planning; wait for a stable accepted pause.')
    receipt = dict(kind=KIND, version=VERSION, parent_run_id=parent.name,
        epoch_myr=float(original.t), integration_steps=int(original.steps),
        parent_checkpoint_sha256=before['checkpoint.npz'], parent_manifest_sha256=before['manifest.json'],
        parent_header=header, parent_files_sha256=before, parent_file_count=len(before),
        parent_preservation_receipt_sha256=APPROVED_PARENT_PRESERVATION_SHA256,
        runtime_source_seal_sha256=APPROVED_RUNTIME_SOURCE_SEAL_SHA256,
        parent_compatibility=header['compatibility'], child_compatibility=expected,
        source_delta=source_delta, activated_controls=normalized, activation=activation, preservation=proof,
        next_output_myr=saved['next_output_myr'], inherited_frame_count=saved['frame_count'],
        interpretation='Explicit optional initiation law starts after this accepted boundary. All stored velocities and solver reports remain predecessor evidence until the next accepted successor interval.')
    return TransitionPlan(parent, original, derived, deepcopy(saved), receipt, before, expected, engine, helpers)


def create_transition(plan, *, output_root=None, run_id=None, title='Asein Lite'):
    """Publish the reviewed plan as a paused derived history, manifest last."""
    _require(isinstance(plan, TransitionPlan), 'a reviewed TransitionPlan is required.')
    _require(REVIEW_READY, 'source review is no longer sealed.')
    current, engine, helpers = _current_context()
    _require(current == plan.compatibility and engine == plan.engine_source and helpers == plan.helper_sources,
        'runtime sources changed after planning.')
    _require(_files(plan.parent) == plan.parent_files, 'parent changed after planning.')
    # A mutable Python plan is not an authorization token: reconstruct its proof
    # from the exact parent before trusting any caller-supplied decoded state.
    fresh = plan_transition(plan.parent, controls=plan.receipt['activated_controls'])
    _require(_fingerprint(vars(plan.original)) == _fingerprint(vars(fresh.original)) and
        _fingerprint(vars(plan.derived)) == _fingerprint(vars(fresh.derived)) and
        plan.manifest == fresh.manifest and plan.receipt == fresh.receipt and
        plan.parent_files == fresh.parent_files, 'plan state/provenance was modified after validation.')
    output_root = Path(output_root or plan.parent.parent).resolve()
    run_id = run_id or datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    _require(isinstance(run_id, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id), 'invalid child run identifier.')
    child = output_root / run_id
    _require(child != plan.parent and not child.exists(), 'child directory already exists or names its parent.')
    origin = dict(run_id=plan.parent.name, time_myr=plan.receipt['epoch_myr'],
        checkpoint_file='inherited-parent/checkpoint.npz', checkpoint_sha256=plan.receipt['parent_checkpoint_sha256'],
        physical_policy=KIND)
    manifest = deepcopy(plan.manifest)
    manifest.update(run_id=run_id, title=title, created=datetime.now(timezone.utc).isoformat(),
        config=deepcopy(plan.derived.config), state='paused', can_resume=True, branch_origin=origin,
        source_transition=dict(kind=KIND, version=VERSION, receipt='source-transition.json',
            epoch_myr=plan.receipt['epoch_myr'], parent_run_id=plan.parent.name,
            inherited_frame_count=plan.manifest['frame_count'],
            inherited_frame_files_sha256={name: value for name, value in plan.parent_files.items()
                if re.fullmatch(r'frame_\d+\.(npz|json)', name)},
            inherited_source_compatibility=deepcopy(plan.receipt['parent_compatibility']),
            scope='Future-only optional initiation physical policy; no inherited state evolution at import'),
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
    child.mkdir(parents=False)
    # Failure leaves an unpublished directory; never damage or clean the parent.
    shutil.copytree(plan.parent, child / 'inherited-parent')
    _require(_files(child / 'inherited-parent') == plan.parent_files, 'copied parent bytes differ from the sealed history.')
    shutil.copy2(PARENT_PRESERVATION_RECEIPT, child / 'parent-preservation.json')
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
    _require(_current_context() == (plan.compatibility, plan.engine_source, plan.helper_sources),
        'runtime source closure changed before publication.')
    _write_json(child / 'source-transition.json', receipt)
    # The saved-run list sees the child only after all provenance/readback checks pass.
    _write_json(child / 'manifest.json', manifest)
    return child, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--controls-json', type=Path)
    parser.add_argument('--create', action='store_true', help='Explicitly publish a paused child; default is read-only planning.')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    controls = json.loads(args.controls_json.read_text()) if args.controls_json else None
    plan = plan_transition(args.parent, controls=controls)
    result = plan.receipt
    child = None
    if args.create:
        child, result = create_transition(plan, output_root=args.output_root, run_id=args.run_id)
        result = dict(result, child_path=str(child))
    if args.receipt:
        _write_json(args.receipt, result)
    # Full immutable evidence belongs in the receipt, never in terminal output.
    summary = dict(kind=KIND, state='paused_child_created' if child is not None else 'read_only_plan_validated',
        parent_run_id=result['parent_run_id'], epoch_myr=result['epoch_myr'],
        integration_steps=result['integration_steps'], inherited_frame_count=result['inherited_frame_count'],
        preexisting_array_count=result['preservation']['preexisting_array_count'],
        next_output_myr=result['next_output_myr'], parent_checkpoint_sha256=result['parent_checkpoint_sha256'],
        changed_helpers=sorted(result['source_delta']['changed_helpers']),
        added_helpers=sorted(result['source_delta']['added_helpers']))
    if child is not None:
        summary.update(child_run_id=result['child_run_id'], child_path=str(child),
            child_checkpoint_sha256=result['child_checkpoint_sha256'])
    if args.receipt:
        summary.update(receipt=str(args.receipt.resolve()), receipt_sha256=_file_sha(args.receipt))
    print(json.dumps(summary, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
