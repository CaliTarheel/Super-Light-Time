"""Atomic, explicitly typed simulation checkpoints; never execute saved code.

The archive contains ordinary NumPy arrays and a JSON tree. Only the engine's
PCG64 generator and PersistentDomainTracker are reconstructed as objects. No
pickle, module lookup from saved strings, or general-purpose object loading is
used. The entire archive is replaced atomically before its manifest is committed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time
import uuid
import zipfile

import numpy as np

from plate_domains import PersistentDomainTracker

VERSION = 1
METADATA_KEY = '__checkpoint_json'


def previous_checkpoint(path):
    path = Path(path)
    return path.with_name(path.stem + '.previous' + path.suffix)


def _replace(temporary, path):
    """Bounded retry for Windows readers holding the destination briefly."""
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(.02 * 2**attempt, .2))


def verify_compatibility(recorded, expected):
    for key, description in (('engine_sha256', 'engine source'),
                             ('auxiliary_sources_sha256', 'engine helper sources'),
                             ('numpy_version', 'NumPy version'),
                             ('flow_backend', 'max-flow backend'),
                             ('material_backend', 'material-loading backend')):
        if recorded.get(key) != expected.get(key):
            raise ValueError(f'This checkpoint uses a different {description}; resume requires the exact saved engine and NumPy version.')


def _encode(value, arrays):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('Object arrays cannot be written to a checkpoint.')
        name = f'a{len(arrays):05d}'
        arrays[name] = value
        return {'t': 'array', 'v': name}
    if isinstance(value, np.generic):
        return _encode(value.item(), arrays)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError('Checkpoint state contains a non-finite number.')
        return value
    if isinstance(value, dict):
        return {'t': 'dict', 'v': [[_encode(k, arrays), _encode(v, arrays)] for k, v in value.items()]}
    if isinstance(value, (list, tuple, set)):
        return {'t': type(value).__name__, 'v': [_encode(v, arrays) for v in value]}
    if isinstance(value, np.random.Generator) and type(value.bit_generator) is np.random.PCG64:
        return {'t': 'pcg64', 'v': _encode(value.bit_generator.state, arrays)}
    if type(value) is PersistentDomainTracker:
        return {'t': 'domain_tracker', 'v': _encode(vars(value), arrays)}
    raise ValueError(f'Checkpoint cannot serialize unsupported state type {type(value).__name__}.')


def _decode(value, archive):
    if not isinstance(value, dict):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise ValueError('Invalid checkpoint JSON value.')
    if set(value) != {'t', 'v'}:
        raise ValueError('Invalid checkpoint type record.')
    kind, payload = value['t'], value['v']
    if kind == 'array':
        if not isinstance(payload, str) or not payload.startswith('a') or payload not in archive.files:
            raise ValueError('Checkpoint references a missing array.')
        result = archive[payload]
        if not isinstance(result, np.ndarray) or result.dtype.hasobject:
            raise ValueError('Checkpoint contains an unsupported object array.')
        return result
    if kind == 'dict':
        return {_decode(k, archive): _decode(v, archive) for k, v in payload}
    if kind in ('list', 'tuple', 'set'):
        values = [_decode(v, archive) for v in payload]
        return values if kind == 'list' else tuple(values) if kind == 'tuple' else set(values)
    if kind == 'pcg64':
        generator = np.random.PCG64()
        generator.state = _decode(payload, archive)
        return np.random.Generator(generator)
    if kind == 'domain_tracker':
        tracker = PersistentDomainTracker.__new__(PersistentDomainTracker)
        tracker.__dict__.update(_decode(payload, archive))
        return tracker
    raise ValueError(f'Unsupported checkpoint type tag: {kind!r}.')


def _metadata(archive, expected=None):
    try:
        payload = archive[METADATA_KEY]
        if not isinstance(payload, np.ndarray) or payload.dtype != np.uint8 or payload.ndim != 1:
            raise ValueError('Invalid checkpoint metadata array.')
        metadata = json.loads(payload.tobytes().decode('utf-8'))
        if not isinstance(metadata, dict) or metadata.get('version') != VERSION:
            raise ValueError('Unsupported checkpoint version.')
        if expected is not None:
            verify_compatibility(metadata['compatibility'], expected)
        return metadata
    except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Checkpoint metadata is missing or damaged.') from exc


def checkpoint_header(path, expected=None):
    """Read the small metadata record without loading the simulation arrays."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = _metadata(archive, expected)
            return {key: metadata[key] for key in ('version', 'compatibility', 'manifest', 'time_myr')}
    except (OSError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError('No readable resume checkpoint exists for this experiment.') from exc


def write_checkpoint(path, simulation, manifest, compatibility):
    arrays = {}
    metadata = dict(version=VERSION, compatibility=compatibility, manifest=manifest,
                    time_myr=float(simulation.t), state=_encode(vars(simulation), arrays))
    encoded = json.dumps(metadata, allow_nan=False, separators=(',', ':')).encode('utf-8')
    arrays[METADATA_KEY] = np.frombuffer(encoded, dtype=np.uint8)
    path = Path(path)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    backup_temporary = previous_checkpoint(path).with_name(path.name+'.previous.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        if path.is_file():
            try:
                # Never replace a usable previous generation with a damaged
                # latest archive. Decode every referenced array, including CRCs.
                previous_simulation, _ = read_checkpoint(path, compatibility, type(simulation))
                del previous_simulation
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                pass
            else:
                with path.open('rb') as source, backup_temporary.open('wb') as target:
                    shutil.copyfileobj(source, target)
                    target.flush()
                    os.fsync(target.fileno())
                _replace(backup_temporary, previous_checkpoint(path))
        # The primary remains readable until this single atomic replacement.
        # A crash before/after either replacement leaves a complete generation.
        _replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
        if backup_temporary.exists():
            backup_temporary.unlink()


def read_checkpoint(path, expected, simulation_type):
    """Restore the explicitly supplied engine class without calling its init."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = _metadata(archive, expected)
            state = _decode(metadata['state'], archive)
    except (OSError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError('Resume checkpoint is missing or damaged.') from exc
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise ValueError('Checkpoint simulation state is invalid.')
    if float(state.get('t', -1)) != metadata['time_myr']:
        raise ValueError('Checkpoint time does not match its simulation state.')
    if state.get('config') != metadata['manifest'].get('config'):
        raise ValueError('Checkpoint configuration does not match its experiment.')
    if not isinstance(state.get('rng'), np.random.Generator):
        raise ValueError('Checkpoint has no valid simulation random generator.')
    simulation = simulation_type.__new__(simulation_type)
    simulation.__dict__.update(state)
    return simulation, metadata['manifest']
