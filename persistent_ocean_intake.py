"""Opt-in Lite rule: established oceanic trenches retain positive intake.

An actual accepted ocean sink latches the trench; declarations at 0 Myr alone
do not. Latches travel with existing trench history through plate splits. The
constraint applies only to finite currently oceanic incoming intervals. Rates
above the configured small floor remain the instantaneous force solution.

This is a prescribed world-building constraint, not a resolved slab law. Its
extra equal/opposite reactions and supplied power are reported independently of
the user's slab-pull coefficient and must enter any breakup calculation.
"""
from copy import deepcopy
import math
import numpy as np

VERSION = 1
FIELD = 'persistent_ocean_intake_state'
LATCH = 'intake_started_myr'


def normalize(value=None):
    defaults = dict(enabled=False, minimum_intake_cm_yr=.1, breakup_mode='gradual')
    if value is None:
        return defaults
    if not isinstance(value, dict) or set(value)-set(defaults):
        raise ValueError('Persistent ocean intake settings contain unknown fields.')
    result = dict(defaults, **value)
    if type(result['enabled']) is not bool:
        raise ValueError('Persistent ocean intake enabled must be boolean.')
    rate = result['minimum_intake_cm_yr']
    if (isinstance(rate, (bool, np.bool_)) or not isinstance(rate, (int, float, np.number))
            or not math.isfinite(float(rate)) or float(rate) <= 0.):
        raise ValueError('Persistent ocean intake needs a finite positive minimum rate.')
    result['minimum_intake_cm_yr'] = float(rate)
    if result['breakup_mode'] not in ('gradual', 'on_conflict'):
        raise ValueError('Persistent intake breakup_mode must be gradual or on_conflict.')
    return result


def _configured(s):
    return normalize(getattr(s, 'config', {}).get('effective_subduction', {}).get('persistent_intake'))


def enabled(s):
    settings = _configured(s)
    state = getattr(s, FIELD, None)
    if not settings['enabled']:
        if state is not None:
            raise ValueError('Recorded persistent ocean intake cannot be silently disabled.')
        return False
    if (not isinstance(state, dict) or type(state.get('version')) is not int or state.get('version') != VERSION
            or state.get('parameters') != settings or state.get('activation_myr') != 0.):
        raise ValueError('Persistent ocean intake needs its recorded fresh-world policy.')
    if (getattr(s, 'effective_subduction_version', 0) != 1
            or getattr(s, 'plate_resistance_version', 0) != 1):
        raise ValueError('Persistent ocean intake requires Lite and passive plate resistance.')
    if settings['breakup_mode'] == 'on_conflict' and s.config.get('force_limit_rifting', {}).get('enabled', False):
        raise ValueError('Immediate intake faulting and gradual force-limit rifting are alternative breakup policies.')
    import scipy
    if state.get('scipy_version') != scipy.__version__:
        raise ValueError('Persistent intake continuation requires its recorded SciPy solver version.')
    return True


def initialize(s):
    settings = _configured(s)
    if not settings['enabled']:
        return
    if float(s.t) != 0. or hasattr(s, FIELD):
        raise ValueError('Persistent ocean intake must be explicitly initialized in a fresh world.')
    try:
        import scipy
    except ImportError as error:
        raise ValueError('Install requirements-intake.txt to enable persistent ocean intake.') from error
    setattr(s, FIELD, dict(version=VERSION, parameters=settings, activation_myr=0.,
                          scipy_version=scipy.__version__, bootstrap='first accepted ocean intake'))
    enabled(s)


def update(s):
    """Latch accepted intake, never a velocity guess or a colored boundary."""
    if not enabled(s):
        return
    record = getattr(s, 'native_subduction_diagnostics', {})
    if record.get('step_end_myr') != float(s.t) or float(s.t) <= 0.:
        return
    fed = {int(row['trench_id']) for row in record.get('removed_area_by_trench', ())
           if float(row['area_km2']) > 0.}
    for row in s.trench_systems:
        law = row.get('effective_subduction')
        if law and row['id'] in fed:
            law.setdefault(LATCH, float(s.t))


def assemble(balance):
    """Endpoint constraints on exact wet intervals of the native finite trace.

    A great-circle chord's normal Euler numerator interpolates linearly, so
    positive endpoint values keep the complete interval positive. Parent mean
    normals or center-cell crust classes cannot establish that condition.
    """
    s = balance.s
    empty = dict(rows=np.zeros((0, balance.size)), minimum=np.zeros(0), ports=[])
    if not enabled(s):
        return empty
    histories = {int(row['id']): row for row in s.trench_systems
                 if LATCH in row.get('effective_subduction', {})
                 and row['phase'] not in ('shutdown', 'joined')}
    if not histories:
        return empty
    for row in histories.values():
        started = row['effective_subduction'][LATCH]
        if (isinstance(started, (bool, np.bool_)) or not isinstance(started, (int, float, np.number))
                or not math.isfinite(float(started)) or not 0. < float(started) <= float(s.t)):
            raise ValueError('An intake latch must record an accepted positive model epoch.')
    import native_spreading as geometry
    contour = getattr(s, 'native_boundary_geometry', None)
    surface = getattr(s, 'material_surface', None)
    if not isinstance(contour, dict) or not isinstance(surface, dict):
        raise ValueError('Persistent ocean intake requires finite native trace and material geometry.')
    selected = np.zeros(len(s.ba), bool)
    selected[balance.trench_edges] = True
    owners = np.asarray(s.parcel_plate)
    contexts = {}
    rows, ports, weights = [], [], []
    for start, end, normal, parent in zip(contour['segments_start'], contour['segments_end'],
                                         contour['segment_normals'], contour['contact_index']):
        ident = int(s.trench_id[parent])
        if not selected[parent] or ident not in histories:
            continue
        down = int(balance.slab_owner[parent])
        over = int(s.bq[parent] if down == s.bp[parent] else s.bp[parent])
        if down not in contexts:
            contexts[down] = geometry.prepare_material(dict(vertices=surface['vertices'],
                                                       faces=surface['faces'][owners == down]))
        direction = normal*(1. if down == s.bp[parent] else -1.)
        for low, high in geometry.ocean_intervals(start, end, contexts[down]):
            endpoints = np.array([(1.-p)*start+p*end for p in (low, high)])
            endpoints /= np.linalg.norm(endpoints, axis=1)[:, None]
            half_length = 6371.*np.arctan2(np.linalg.norm(np.cross(*endpoints)),
                                         np.dot(*endpoints))*.5
            for parameter in (low, high):
                radial = (1.-parameter)*start+parameter*end
                radial /= np.linalg.norm(radial)
                tangent = direction-radial*np.dot(direction, radial)
                magnitude = np.linalg.norm(tangent)
                if not np.isfinite(magnitude) or magnitude <= 1e-12:
                    raise ValueError('An attached ocean trace has an unresolved normal.')
                lever = np.cross(radial, tangent/magnitude)
                row = np.zeros(balance.size)
                balance._block(down, lever, row)
                balance._block(over, -lever, row)
                rows.append(row)
                weights.append(half_length)
                ports.append(dict(trench_id=ident, down=down, over=over,
                                  parent=int(parent), point=radial.tolist()))
    return dict(rows=np.asarray(rows).reshape(-1, balance.size),
                minimum=np.full(len(rows), _configured(s)['minimum_intake_cm_yr']), ports=ports,
                weights_km=np.asarray(weights))


def snapshot(s):
    if not enabled(s):
        return {}
    return dict(persistent_ocean_intake_version=VERSION,
        persistent_ocean_intake_policy=deepcopy(getattr(s, FIELD)),
        persistent_ocean_intake_trenches=[dict(trench_id=int(row['id']),
            started_myr=float(row['effective_subduction'][LATCH])) for row in s.trench_systems
            if LATCH in row.get('effective_subduction', {})],
        persistent_intake_faults=deepcopy(getattr(s, 'persistent_intake_faults', [])))


def validate_frame(frame):
    fields = {'persistent_ocean_intake_version', 'persistent_ocean_intake_policy',
              'persistent_ocean_intake_trenches', 'persistent_intake_faults'}
    if not fields.intersection(frame):
        return
    if not fields.issubset(frame) or type(frame['persistent_ocean_intake_version']) is not int or frame['persistent_ocean_intake_version'] != VERSION:
        raise ValueError('Persistent intake frame metadata is incomplete or unsupported.')
    policy = frame['persistent_ocean_intake_policy']
    if (not isinstance(policy, dict) or type(policy.get('version')) is not int or policy['version'] != VERSION
            or policy.get('activation_myr') != 0. or not isinstance(policy.get('scipy_version'), str)
            or not policy['scipy_version'] or policy.get('bootstrap') != 'first accepted ocean intake'):
        raise ValueError('Persistent intake frame has no valid recorded policy.')
    settings = normalize(policy.get('parameters'))
    if not settings['enabled'] or settings != policy['parameters']:
        raise ValueError('Persistent intake frame parameters do not match their normalized law.')
    epoch = float(frame.get('time_myr', frame.get('t', 0.)))
    seen = set()
    traces = frame['persistent_ocean_intake_trenches']
    if not isinstance(traces, list):
        raise ValueError('Persistent intake frame needs its latched trench identities.')
    for row in traces:
        ident = row.get('trench_id') if isinstance(row, dict) else None
        started = row.get('started_myr') if isinstance(row, dict) else None
        if (type(ident) is not int or ident <= 0 or ident in seen
                or isinstance(started, bool) or not isinstance(started, (float, int))
                or not math.isfinite(started) or not 0. < started <= epoch):
            raise ValueError('Persistent intake frame has an invalid latch identity or epoch.')
        seen.add(ident)
    faults = frame['persistent_intake_faults']
    if not isinstance(faults, list) or (faults and settings['breakup_mode'] != 'on_conflict'):
        raise ValueError('Intake fault events disagree with the recorded breakup policy.')
    for row in faults:
        if (not isinstance(row, dict) or row.get('cause') != 'persistent_intake_yield'
                or row.get('assigned_opening_km') != 0. or row.get('assigned_velocity_impulse') is not False
                or not 0. <= float(row.get('time_myr', -1.)) <= epoch
                or not math.isfinite(float(row.get('loading_ratio', float('nan'))))
                or float(row.get('loading_ratio', 0.)) <= 1.):
            raise ValueError('Intake fault event lacks a qualified strength-onset record.')
