"""Optional reduced mantle-mediated traction on the real trench carrier.

The prescribed source power is f L (u_down - alpha u_carrier). The remaining
force reaction and the supplied power belong to Lite's omitted slab/mantle
reservoir. This is neither a resolved flow model nor an independent moving
hinge. It adds no passive operator, velocity correction or material source.
"""
from copy import copy, deepcopy
import math
import numpy as np

VERSION = 1
DEFAULTS = dict(version=VERSION, enabled=False, alpha=.1)
FIELD = 'effective_subduction_carrier_traction'
POLICY = 'prescribed finite-port source f L (u_down - alpha u_carrier); unresolved slab/mantle exchange'


def normalize(raw=None):
    if raw is None:
        return deepcopy(DEFAULTS)
    if not isinstance(raw, dict) or set(raw)-set(DEFAULTS):
        raise ValueError('Unknown effective carrier-traction controls.')
    result = dict(DEFAULTS, **raw)
    if type(result['version']) is not int or result['version'] != VERSION:
        raise ValueError('Unsupported effective carrier-traction version.')
    if type(result['enabled']) is not bool:
        raise ValueError('Effective carrier-traction enabled must be boolean.')
    alpha = result['alpha']
    if (isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, (int, float, np.integer, np.floating))
            or not np.isfinite(alpha) or not 0. <= alpha <= 1.):
        raise ValueError('Effective carrier-traction alpha must be finite in [0,1].')
    result['alpha'] = float(alpha)
    return result


def _configured(s):
    configured = s.config.get('effective_subduction', {})
    if not isinstance(configured, dict):
        raise ValueError('Effective carrier traction requires a dictionary base configuration.')
    return normalize(configured.get('carrier_traction'))


def validate(s):
    """Validate checkpoint policy without changing state or physical arrays."""
    controls = _configured(s)
    state = getattr(s, FIELD, None)
    if state is None:
        if controls['enabled']:
            raise ValueError('Enabled carrier traction needs explicit activation metadata.')
        return False
    keys = {'version', 'enabled', 'activation_myr', 'parameters', 'retrospective_work_j', 'law'}
    if (not isinstance(state, dict) or set(state) != keys
            or type(state['version']) is not int or state['version'] != VERSION
            or state['enabled'] is not True or state['law'] != POLICY):
        raise ValueError('Invalid effective carrier-traction activation metadata.')
    parameters = normalize(state['parameters'])
    if not parameters['enabled'] or parameters != controls or state['parameters'] != parameters:
        raise ValueError('Effective carrier-traction configuration and state disagree.')
    import effective_subduction
    if not effective_subduction.enabled(s):
        raise ValueError('Carrier traction requires the existing effective subduction law.')
    epoch = state['activation_myr']
    zero = state['retrospective_work_j']
    if (isinstance(epoch, (bool, np.bool_))
            or not isinstance(epoch, (int, float, np.integer, np.floating))
            or not math.isfinite(float(epoch))
            or isinstance(s.t, (bool, np.bool_))
            or not isinstance(s.t, (int, float, np.integer, np.floating))
            or not math.isfinite(float(s.t))
            or not 0. <= float(epoch) <= float(s.t)
            or isinstance(zero, (bool, np.bool_))
            or not isinstance(zero, (int, float, np.integer, np.floating))
            or not math.isfinite(float(zero)) or float(zero) != 0.):
        raise ValueError('Carrier traction requires a valid future-only activation epoch and zero historical work.')
    return True


def enabled(s):
    return validate(s)


def alpha(s):
    return float(getattr(s, FIELD)['parameters']['alpha']) if validate(s) else 0.


def upgrade(s, parameters=None):
    """Declare a new source law at the accepted epoch, with no past work credit.

    Only the nested configuration and this policy's new plain attribute change.
    Inherited effective settings, trench history, geometry, physical arrays and
    RNG are unchanged; no constructor, boundary rebuild or force solve is used.
    """
    requested = normalize(dict(enabled=True) if parameters is None else parameters)
    if not requested['enabled']:
        raise ValueError('Carrier-traction upgrade must explicitly enable its policy.')
    if (isinstance(s.t, (bool, np.bool_))
            or not isinstance(s.t, (int, float, np.integer, np.floating))
            or not math.isfinite(float(s.t)) or float(s.t) < 0.):
        raise ValueError('Carrier traction requires a finite nonnegative accepted epoch.')
    if getattr(s, FIELD, None) is not None:
        validate(s)
        if getattr(s, FIELD)['parameters'] != requested:
            raise ValueError('Existing carrier traction cannot be silently reparameterized.')
        return deepcopy(getattr(s, FIELD))
    import effective_subduction
    # Validate the inherited law independently of this not-yet-published policy.
    view = copy(s)
    view.config = deepcopy(s.config)
    view.config.get('effective_subduction', {}).pop('carrier_traction', None)
    if not effective_subduction.enabled(view):
        raise ValueError('Carrier traction requires an initialized effective subduction law.')
    effective_subduction.validate_compatibility(view)
    configuration = deepcopy(s.config)
    configuration['effective_subduction']['carrier_traction'] = requested
    policy = dict(version=VERSION, enabled=True, activation_myr=float(s.t),
                  parameters=deepcopy(requested), retrospective_work_j=0., law=POLICY)
    view.config = configuration
    setattr(view, FIELD, policy)
    validate(view)
    s.config = configuration
    setattr(s, FIELD, policy)
    return deepcopy(policy)


def initialize(s):
    controls = _configured(s)
    if not controls['enabled']:
        return False
    upgrade(s, controls)
    return True


def carriers(s, incoming, force, live):
    """Resolve the other actual owner of precisely the admitted incoming ports.

    No matching halo, dry portion, pending declaration or unsupported initiated
    support is recruited: those gates are the effective law's shared line_state.
    An invalid active owner/finite port fails before force assembly.
    """
    fraction = alpha(s)
    count = len(s.ba)
    result = np.full(count, -1, int)
    if not validate(s):
        return result, fraction
    incoming, force, live = np.asarray(incoming), np.asarray(force), np.asarray(live)
    bp, bq = np.asarray(s.bp), np.asarray(s.bq)
    uid, active = np.asarray(s.plate_uid), np.asarray(s.active)
    if (incoming.shape != (count,) or incoming.dtype.kind not in 'iu'
            or force.shape != (count,) or not np.isfinite(force).all() or np.any(force < 0.)
            or live.shape != (count,) or live.dtype.kind != 'b'
            or bp.shape != (count,) or bq.shape != (count,)
            or bp.dtype.kind not in 'iu' or bq.dtype.kind not in 'iu'
            or uid.shape != active.shape or uid.dtype.kind not in 'iu'
            or active.dtype.kind != 'b' or active.ndim != 1
            or np.any(uid[active] <= 0) or len(np.unique(uid[active])) != int(active.sum())):
        raise ValueError('Carrier traction needs aligned original owner integers and unique active UIDs.')
    selected = live & (force > 0.)
    edges = np.flatnonzero(selected)
    if not len(edges):
        return result, fraction
    other = np.where(incoming[edges] == bp[edges], bq[edges], bp[edges])
    owners = np.r_[incoming[edges], other]
    if (np.any((incoming[edges] != bp[edges]) & (incoming[edges] != bq[edges]))
            or np.any(owners < 0) or np.any(owners >= len(active))
            or np.any(incoming[edges] == other) or not np.all(active[owners])):
        raise ValueError('Carrier traction requires two distinct current active owners at every loaded port.')
    mid, normal, length = np.asarray(s.bmid), np.asarray(s.bn), np.asarray(s.bl)
    if (mid.shape != (count, 3) or normal.shape != (count, 3) or length.shape != (count,)
            or not np.isfinite(mid[edges]).all() or not np.isfinite(normal[edges]).all()
            or not np.isfinite(length[edges]).all() or np.any(length[edges] <= 0.)
            or np.any(np.linalg.norm(mid[edges], axis=1) == 0.)
            or np.any(np.linalg.norm(np.cross(mid[edges], normal[edges]), axis=1) == 0.)):
        raise ValueError('Carrier traction requires finite positive-length resolved tangent ports.')
    result[edges] = other
    return result, fraction


def snapshot(s):
    if not validate(s):
        return {}
    diagnostics = deepcopy(getattr(s, FIELD))
    diagnostics.update(observation_myr=float(s.t), independent_within_host_hinge_motion=False,
                       measured_arc_host_motion=_arc_host_motion(s),
                       observation_scope='Accepted Euler motion at actual finite admitted contour support; diagnostic only, no loading clock or source.')
    return dict(effective_subduction_carrier_traction_version=VERSION,
                effective_subduction_carrier_traction_parameters=deepcopy(getattr(s, FIELD)['parameters']),
                effective_subduction_carrier_traction_diagnostics=diagnostics)


def _arc_host_motion(s):
    """Observe solved real-carrier/host motion, without advancing any history."""
    rows = [row for row in getattr(s, 'backarc_basins', ())
            if row.get('arc_plate_uid') is not None and row.get('phase') != 'closed']
    if not rows:
        return []
    import effective_subduction
    incoming, force, live = effective_subduction.line_state(s)
    carrier, _ = carriers(s, incoming, force, live)
    slots = {int(uid): int(p) for p, uid in enumerate(s.plate_uid) if s.active[p]}
    geometry = getattr(s, 'native_boundary_geometry', None)
    if not isinstance(geometry, dict):
        raise ValueError('Arc-host motion needs actual finite native contour geometry.')
    required = ('segments_start', 'segments_end', 'segment_normals', 'contact_index')
    if not set(required).issubset(geometry):
        raise ValueError('Arc-host finite contour metadata is incomplete.')
    start, end, normal = (np.asarray(geometry[k], float) for k in required[:3])
    parents = np.asarray(geometry['contact_index'])
    if (start.shape != end.shape or start.shape != normal.shape
            or start.shape != (len(parents), 3) or parents.dtype.kind not in 'iu'
            or np.any(parents < 0) or np.any(parents >= len(s.ba))
            or not np.isfinite(start).all() or not np.isfinite(end).all()
            or not np.isfinite(normal).all()):
        raise ValueError('Arc-host finite contours require aligned bounded geometry.')
    omega = np.asarray(s.omega, float)
    if omega.shape != (len(s.active), 3) or not np.isfinite(omega).all():
        raise ValueError('Arc-host motion requires finite accepted Euler rotations.')
    report = []
    for row in rows:
        host, arc, down = (slots.get(row.get(k)) for k in
                          ('parent_plate_uid', 'arc_plate_uid', 'downgoing_plate_uid'))
        if host is None or arc is None or down is None or len({host, arc, down}) != 3:
            continue
        support = live & (force > 0.) & (incoming == down) & (carrier == arc)
        if row.get('trench_id') is not None:
            support &= np.asarray(s.trench_id) == row['trench_id']
        ids = np.flatnonzero(support[parents])
        lengths, rates = [], []
        for piece in ids:
            a, b = start[piece], end[piece]
            radial = a+b
            if (np.linalg.norm(radial) == 0. or abs(np.linalg.norm(a)-1.) > 1e-10
                    or abs(np.linalg.norm(b)-1.) > 1e-10):
                raise ValueError('Arc-host observation needs resolved unit finite endpoints.')
            radial = radial/np.linalg.norm(radial)
            tangent_normal = normal[piece]-radial*float(normal[piece]@radial)
            norm = float(np.linalg.norm(tangent_normal))
            if norm == 0.:
                raise ValueError('Arc-host observation needs a resolved tangent normal.')
            inland = tangent_normal/norm*(1. if incoming[parents[piece]] == s.bp[parents[piece]] else -1.)
            length = 6371.*math.atan2(float(np.linalg.norm(np.cross(a, b))), float(a@b))
            if length == 0.:
                continue
            velocity = np.cross(omega[arc]-omega[host], radial)*6371.
            lengths.append(length)
            rates.append(-float(velocity@inland))
        total = math.fsum(lengths)
        report.append(dict(backarc_id=int(row['id']), trench_id=row.get('trench_id'),
            carrier_plate_uid=int(s.plate_uid[arc]), host_plate_uid=int(s.plate_uid[host]),
            finite_piece_count=len(lengths), supported_length_km=total,
            mean_retreat_relative_host_km_myr=(math.fsum(l*v for l, v in zip(lengths, rates))/total if total else 0.),
            maximum_retreat_relative_host_km_myr=max(rates, default=0.),
            positive_retreat_length_km=math.fsum(l for l, v in zip(lengths, rates) if v > 0.),
            sign='positive is real carrier motion toward incoming ocean relative to host',
            source_or_loading_added=False))
    return report


def source_audit(model, x):
    """Instantaneous represented source power; never an accumulated-energy claim."""
    values = np.asarray(x, float)
    down = float(model.slab_down_drive@values)
    carrier = float(model.slab_over_drive@values)
    total = float(model.drivers['slab']@values)
    return dict(version=VERSION, alpha=float(model.effective_carrier_alpha),
                downgoing_power_w=down, carrier_power_w=carrier,
                represented_source_power_w=total, reservoir_exchange_power_w=-total,
                power_partition_residual_w=total-down-carrier,
                unrepresented_force_reaction_fraction=1.-float(model.effective_carrier_alpha),
                law=POLICY, passive_operators_changed=False,
                accumulated_work_reported=False)


def validate_frame(frame):
    """Accept legacy frames; require complete aligned metadata for this policy."""
    fields = {f'{FIELD}_version', f'{FIELD}_parameters', f'{FIELD}_diagnostics'}
    present = fields.intersection(frame)
    if not present:
        if frame.get('config', {}).get('effective_subduction', {}).get('carrier_traction', {}).get('enabled', False):
            raise ValueError('Enabled carrier-traction frame lacks activation metadata.')
        return False
    if present != fields or type(frame[f'{FIELD}_version']) is not int or frame[f'{FIELD}_version'] != VERSION:
        raise ValueError('Carrier-traction frame metadata is incomplete or versioned incorrectly.')
    controls = normalize(frame[f'{FIELD}_parameters'])
    if not controls['enabled'] or controls != frame[f'{FIELD}_parameters']:
        raise ValueError('Carrier-traction frame controls must be enabled and normalized.')
    diagnostics = frame[f'{FIELD}_diagnostics']
    if not isinstance(diagnostics, dict):
        raise ValueError('Carrier-traction frame diagnostics must be plain metadata.')
    policy = {key: diagnostics.get(key) for key in
              ('version', 'enabled', 'activation_myr', 'parameters', 'retrospective_work_j', 'law')}
    from types import SimpleNamespace
    configured = frame.get('config', {}).get('effective_subduction', {}).get('carrier_traction', controls)
    observed = diagnostics.get('observation_myr')
    if (isinstance(observed, (bool, np.bool_))
            or not isinstance(observed, (int, float, np.integer, np.floating))
            or not math.isfinite(float(observed)) or float(observed) < 0.):
        raise ValueError('Carrier-traction frame needs a finite observation epoch.')
    # Native review frames round time to six decimals; retain the exact epoch
    # in this metadata rather than reject a legitimate rounded activation frame.
    if 'time_myr' in frame and round(float(observed), 6) != frame['time_myr']:
        raise ValueError('Carrier-traction observation epoch disagrees with saved frame time.')
    subject = SimpleNamespace(config=dict(effective_subduction=dict(enabled=True, carrier_traction=configured)),
        effective_subduction_version=frame.get('effective_subduction_version', 0),
        t=float(observed))
    setattr(subject, FIELD, policy)
    validate(subject)
    if (diagnostics.get('independent_within_host_hinge_motion') is not False
            or not isinstance(diagnostics.get('measured_arc_host_motion'), list)):
        raise ValueError('Carrier-traction frame must distinguish measured motion from independent hinge dynamics.')
    for row in diagnostics['measured_arc_host_motion']:
        if not isinstance(row, dict) or row.get('source_or_loading_added') is not False:
            raise ValueError('Arc-host frame observations cannot add a loading source.')
        for key in ('supported_length_km', 'mean_retreat_relative_host_km_myr',
                    'maximum_retreat_relative_host_km_myr', 'positive_retreat_length_km'):
            value = row.get(key)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                    or not math.isfinite(float(value))):
                raise ValueError('Arc-host frame observations require finite real scalars.')
        if (row['supported_length_km'] < 0. or not 0. <= row['positive_retreat_length_km'] <= row['supported_length_km']
                or type(row.get('finite_piece_count')) is not int or row['finite_piece_count'] < 0):
            raise ValueError('Arc-host frame observation has invalid finite support.')
    return True
