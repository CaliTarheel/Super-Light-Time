"""Explicit frame semantics for a nonzero-time production initialization.

Historical solver reports remain provenance. They are never relabeled as a
completed interval under the successor's laws. Only the exact saved migration
epoch and integration-step counter may emit this unevolved boundary frame.
"""
from copy import deepcopy
import hashlib
import json
import math
import numpy as np


FIELD = 'production_initialization_boundary'
DIAGNOSTICS = ('deformation_diagnostics', 'plate_balance_diagnostics',
               'timestep_diagnostics', 'material_column_budget', 'foundering_diagnostics')
DEFORMATION = dict(model='explicit production restart initialization',
                   initialized=True, physical_time_advanced_myr=0.,
                   mechanical_solve_performed=False)


def _require(condition, message):
    if not condition:
        raise ValueError('Invalid production initialization boundary: ' + message)


def _native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_native(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def record(s):
    """Capture diagnostics as immutable receipt data without changing the state."""
    inherited = {name: deepcopy(getattr(s, name)) for name in DIAGNOSTICS if hasattr(s, name)}
    _require(isinstance(inherited.get('deformation_diagnostics'), dict),
             'historical deformation diagnostics must be retained.')
    return dict(version=1, epoch_myr=float(s.t), integration_steps=int(s.steps),
        physical_time_advanced_myr=0., mechanical_solve_performed=False,
        inherited_diagnostics=_native(inherited), inherited_diagnostics_sha256=_digest(inherited),
        interpretation='Initialization under newly selected laws; inherited reports are historical evidence, not accepted successor evolution.',
        velocity_scope='Stored plate velocities are inherited; the successor force law has not yet been solved.')


def _receipt(frame):
    receipt = frame.get('production_restart_migration')
    _require(isinstance(receipt, dict) and type(receipt.get('version')) is int
             and receipt['version'] == 1 and receipt.get('kind') == 'explicit_repaired_production_restart',
             'an explicit production migration receipt is required.')
    boundary = receipt.get('initialization_boundary')
    _require(isinstance(boundary, dict), 'migration omitted its original boundary record.')
    _require(type(boundary.get('version')) is int and boundary['version'] == 1,
             'unsupported boundary version.')
    epoch = boundary.get('epoch_myr')
    steps = boundary.get('integration_steps')
    _require(type(epoch) in (int, float) and math.isfinite(epoch) and epoch > 0.
             and epoch == receipt.get('epoch_myr'), 'receipt epochs differ.')
    _require(type(steps) is int and steps >= 0 and steps == receipt.get('epoch_steps'),
             'receipt integration-step counters differ.')
    _require(type(boundary.get('physical_time_advanced_myr')) in (int, float)
             and boundary['physical_time_advanced_myr'] == 0.
             and boundary.get('mechanical_solve_performed') is False,
             'initialization cannot claim a physical or mechanical step.')
    inherited = boundary.get('inherited_diagnostics')
    _require(isinstance(inherited, dict) and isinstance(inherited.get('deformation_diagnostics'), dict)
             and set(inherited).issubset(DIAGNOSTICS), 'historical diagnostics are incomplete or unrecognized.')
    _require(boundary.get('inherited_diagnostics_sha256') == _digest(inherited),
             'historical diagnostic provenance does not match its receipt.')
    return boundary


def project_snapshot(s, frame):
    """Change only the new frame's diagnostic view at the exact restart boundary."""
    receipt = getattr(s, 'production_restart_migration', None)
    if receipt is None:
        return frame
    boundary = _receipt(frame)
    if float(s.t) != boundary['epoch_myr']:
        return frame
    _require(int(s.steps) == boundary['integration_steps'],
             'migration time was reused with a different step counter.')
    _require(_digest({name: getattr(s, name) for name in DIAGNOSTICS if hasattr(s, name)})
             == boundary['inherited_diagnostics_sha256'],
             'solver diagnostics changed without advancing the migration boundary.')
    frame[FIELD] = deepcopy(boundary)
    frame['deformation_diagnostics'] = deepcopy(DEFORMATION)
    for name in DIAGNOSTICS:
        if name != 'deformation_diagnostics' and name in frame:
            frame.pop(name)
    validate_frame(frame)
    return frame


def validate_frame(frame):
    """Return False for ordinary frames; strictly validate a boundary otherwise."""
    if FIELD not in frame:
        return False
    boundary = _receipt(frame)
    _require(frame[FIELD] == boundary, 'frame boundary differs from the migration receipt.')
    _require(type(frame.get('time_myr')) in (int, float)
             and frame['time_myr'] == boundary['epoch_myr'], 'frame is not at the migration epoch.')
    stats = frame.get('stats', {})
    _require(type(stats.get('integration_steps')) is int
             and stats['integration_steps'] == boundary['integration_steps'],
             'frame is not at the migration step counter.')
    for name in ('mesh_version', 'material_mechanics_version', 'gravity_constraint_version'):
        _require(type(frame.get(name)) is int and frame[name] == 1,
                 'boundary requires '+name+'=1.')
    _require(frame.get('deformation_diagnostics') == DEFORMATION,
             'initialization cannot carry active deformation/gravity interval claims.')
    _require(all(frame.get(name) is None for name in DIAGNOSTICS if name != 'deformation_diagnostics'),
             'initialization cannot carry active solver/source interval claims.')
    return True
