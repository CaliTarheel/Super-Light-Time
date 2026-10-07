"""Read a sparse, committed saved history without constructing a simulation.

The selected frame pairs and captured source bytes are fingerprinted. Manifest
appends are allowed; replacement of the observed prefix or selected files is
not. Review tools never write to the input run.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def committed_prefix(manifest):
    rows = manifest.get('frames')
    count = manifest.get('frame_count')
    if not isinstance(rows, list) or type(count) is not int or not 0 <= count <= len(rows):
        raise ValueError('A committed frame count and frame list are required.')
    prefix = rows[:count]
    previous = -math.inf
    for i, row in enumerate(prefix):
        epoch = row.get('time_myr')
        if (type(row.get('index')) is not int or row['index'] != i
                or isinstance(epoch, bool) or not isinstance(epoch, (int, float))
                or not math.isfinite(epoch) or epoch <= previous):
            raise ValueError('Committed history must have ordered indices and unique increasing epochs.')
        previous = epoch
    return prefix


def select_epochs(prefix, epochs):
    if not prefix:
        raise ValueError('The run has no committed frames.')
    selected = set()
    for value in epochs:
        if value == 'latest':
            selected.add(len(prefix) - 1)
            continue
        try:
            epoch = float(value)
        except (ValueError, TypeError):
            raise ValueError('Select exact saved epochs or latest.') from None
        if not math.isfinite(epoch):
            raise ValueError('Selected epoch must be finite.')
        matches = [r['index'] for r in prefix if float(r['time_myr']) == epoch]
        if len(matches) != 1:
            raise ValueError(f'Epoch {value} is not an exact committed snapshot.')
        selected.add(matches[0])
    if not selected:
        raise ValueError('Select at least one committed epoch.')
    return sorted(selected)


def load_history(run_path, epochs):
    run = Path(run_path).resolve()
    manifest_path = run / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    prefix = committed_prefix(manifest)
    indices = select_epochs(prefix, epochs)
    run_id = manifest.get('run_id')
    if not isinstance(run_id, str) or not run_id:
        raise ValueError('The committed run identity is required.')
    hashes, frames = {}, []
    declared_sources = dict(manifest.get('auxiliary_sources_sha256', {}))
    if manifest.get('engine_sha256'):
        declared_sources['engine.py'] = manifest['engine_sha256']
    for name, expected in declared_sources.items():
        # Captured sources are flat files. A manifest cannot select other paths.
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Invalid captured source filename.')
        if digest(run / name) != expected:
            raise ValueError(f'Captured source changed: {name}')
        hashes[name] = expected
    for index in indices:
        data_path = run / f'frame_{index:04d}.npz'
        meta_path = data_path.with_suffix('.json')
        for path in (meta_path, data_path):
            hashes[path.name] = digest(path)
        frame = json.loads(meta_path.read_bytes())
        if (frame.get('index') != index
                or frame.get('time_myr') != prefix[index]['time_myr']
                or frame.get('run_id', run_id) != run_id):
            raise ValueError(f'Frame {index} metadata disagrees with the committed manifest.')
        with np.load(data_path, allow_pickle=False) as archive:
            for name in archive.files:
                if name in frame:
                    raise ValueError(f'Frame {index} duplicates metadata field {name} in its arrays.')
                frame[name] = archive[name].copy()
        frame.update(run_id=run_id, source_frame_sha256=hashes[data_path.name],
                     source_metadata_sha256=hashes[meta_path.name])
        frames.append(frame)
    provenance = dict(run_path=str(run), run_id=run_id,
                      manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                      committed_frame_count=len(prefix), selected_indices=indices,
                      selected_epochs_myr=[prefix[i]['time_myr'] for i in indices],
                      input_sha256=hashes,
                      scope='Exact saved epochs; no time interpolation or new model integration.')
    loaded = dict(manifest=manifest, prefix=prefix, frames=frames, provenance=provenance)
    verify_history(loaded)
    return loaded


def verify_history(loaded):
    run = Path(loaded['provenance']['run_path'])
    fresh = json.loads((run / 'manifest.json').read_bytes())
    prefix = committed_prefix(fresh)
    old = loaded['prefix']
    if fresh.get('run_id') != loaded['provenance']['run_id'] or prefix[:len(old)] != old:
        raise ValueError('The observed committed history prefix changed during review.')
    for key in ('config', 'engine_sha256', 'auxiliary_sources_sha256'):
        if fresh.get(key) != loaded['manifest'].get(key):
            raise ValueError(f'The committed run declaration changed during review: {key}')
    for name, expected in loaded['provenance']['input_sha256'].items():
        if digest(run / name) != expected:
            raise ValueError(f'Review input changed: {name}')
    return True
