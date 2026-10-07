"""Candidate native root-local welding qualification.

A mature front cannot transfer an unengaged remote part of a connected body.
Finite contact profiles propagate only through shared material edges inside
the configured process zone (and, under welded-stack policy 1, through the
thrust contact between overlapping sheets of one plate). Whole roots
accumulate exposure while engaged, with exponential loss of memory while
unloaded, and
only a whole coherent component with every root engaged and mature can change
motion owner. No area cap, new fracture, positional edit, or rock sink is used.
This is a parameterized accretion heuristic, not calibrated fault mechanics.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import math
import numpy as np

VERSION = 1
MEMORY_MYR = 35.
THRESHOLD = 18.
# A suture this strong welds a continental margin without waiting for the whole
# component to engage. suture_strength = -expm1(cumulative_convergence_km/-100),
# so 0.95 is ~300 km of sustained convergence -- comfortably past the point where
# an ocean basin has closed and two margins are in contact, and far short of the
# 1,927 km this run has already accumulated on one contact. Requiring several
# engaged roots as well keeps a nominal or stale contact from welding a continent
# on convergence history alone.
SUTURE_WELD_STRENGTH = .95
SUTURE_WELD_MIN_ROOTS = 3
# RETIRED at 392 Myr. The suture path shipped at 338 Myr and never fired, because
# local_accretion read suture_strength off the local-accretion row (which never
# carries it) instead of the persistent collision contact. Fixing that wiring
# exposed the real problem: ownership transfer here is WHOLE-COMPONENT, and at
# 344 Myr the qualifying component was all of Continental block 01 -- 7.73e7 km2,
# 15.1% of the planet -- which would have merged into block 02 as a single plate
# covering 46.4% of the surface (Earth's largest, the Pacific, is ~20%). It would
# have qualified on 2-3 mature roots out of 1,184, which is exactly the "huge body
# borrows a mature front" case tests/test_native_localized_accretion.py forbids.
# No suture-length rule separates the two either: suture/perimeter is 15.4% for
# the live body and 29.7% for the forbidden test body (India ~39%).
# Whole-component transfer is right for a terrane docking and wrong for two
# continents in collision -- India is still its own plate 50 Myr on. Collision
# stays with collision_contacts (thrust stacking) plus gravitational spreading.
# The criterion is still evaluated and reported as suture_would_qualify so the
# retirement stays visible in every frame rather than silently dead.
SUTURE_WELD_ENABLED = False
TABLE = ('source_uid', 'target_uid', 'root_id', 'loading', 'active_weight')
ARRAY_FIELDS = {f'accretion_welding_{name}': ('int64' if name.endswith('uid') or name == 'root_id' else 'float64')
                for name in TABLE}


def initialize(s):
    s.native_accretion_version = VERSION
    if not hasattr(s, 'accretion_welding_state'):
        s.accretion_welding_state = dict(version=VERSION, last_time_myr=float(s.t),
            geometry_signature='', **{name: np.empty(0, dtype=ARRAY_FIELDS['accretion_welding_'+name]) for name in TABLE})
    if not getattr(s, 'native_accretion_diagnostics', None):
        s.native_accretion_diagnostics = _diagnostics(s)


def enabled(s):
    return getattr(s, 'native_accretion_version', 0) == VERSION


def _diagnostics(s, elapsed=0., gap=0.):
    state = s.accretion_welding_state
    return dict(version=VERSION,
        model='root-local engaged exposure and unloaded memory decay within a connected finite process zone',
        process_zone_width_km=float(getattr(s, 'config', {}).get('deformation_width_km', 400.)),
        memory_myr=MEMORY_MYR, maturity_threshold=THRESHOLD, loading_units='process-weighted Myr',
        elapsed_myr=elapsed, unloaded_gap_myr=gap, directed_root_records=len(state['root_id']),
        currently_engaged_roots=int(np.count_nonzero(state['active_weight'] > 0.)),
        mature_engaged_roots=int(np.count_nonzero((state['loading'] >= THRESHOLD) & (state['active_weight'] > 0.))),
        qualification_checks=[], whole_component_required=True,
        transfer_policy='Whole-component ownership change; this module introduces no material cuts or sinks.',
        model_limit='Parameterized contact-zone maturation; no calibrated fault strength or new fracture criterion.')


def geometry_signature(s):
    h = hashlib.sha256()
    for value in (s.material_surface['vertices'], s.material_surface['faces'], s.parcel_plate):
        h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def _roots(s):
    result = np.asarray(getattr(s, 'material_lineage', {}).get('root_id', s.parcel_patch))
    if result.shape != np.asarray(s.parcel_patch).shape or result.dtype.kind not in 'iu':
        raise ValueError('Welding requires one integer material root per face.')
    return result


def _component(s, component):
    material = np.asarray(component['parcel_indices'], dtype=np.int64)
    faces = np.asarray(s.material_surface['faces'])[material]
    points = np.asarray(s.material_surface['vertices'])
    root_ids, root_inverse = np.unique(_roots(s)[material], return_inverse=True)
    edge_vertices = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edge_faces = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((edge_vertices[:, 1], edge_vertices[:, 0]))
    edge_vertices, edge_faces = edge_vertices[order], edge_faces[order]
    neighbours = [set() for _ in faces]
    for i in np.flatnonzero(np.all(edge_vertices[1:] == edge_vertices[:-1], axis=1)):
        a, b = int(edge_faces[i]), int(edge_faces[i+1])
        neighbours[a].add(b); neighbours[b].add(a)
    import local_accretion
    if len(material) and local_accretion.welded_stack_enabled(s):
        # A welded stack is one body: loading crosses the thrust or suture
        # between its overlapping sheets, never an ocean gap between bodies.
        pairs = component.get('welded_pairs')
        if pairs is None:
            import collision_contacts
            pairs = collision_contacts.same_owner_overlap_pairs(s)
        pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
        slot = np.full(len(np.asarray(s.parcel_patch)), -1, np.int64)
        slot[material] = np.arange(len(material))
        local = slot[pairs]
        for a, b in local[np.all(local >= 0, axis=1)]:
            neighbours[int(a)].add(int(b)); neighbours[int(b)].add(int(a))
    return dict(material=material, faces=faces, points=points, roots=root_ids, inverse=root_inverse,
                patches=np.asarray(s.parcel_patch)[material], neighbours=neighbours)


def _distance(points, observation, radius):
    middle = np.asarray(observation['center'], float)
    normal = np.asarray(observation['normal'], float)
    middle = middle/np.linalg.norm(middle)
    normal = normal-middle*np.dot(middle, normal)
    if np.linalg.norm(normal) <= 1e-12:
        raise ValueError('Welding contact normal must have a finite tangent direction.')
    normal /= np.linalg.norm(normal)
    length = float(observation['length'])
    if not math.isfinite(length) or not 0 < length < math.pi*radius:
        raise ValueError('Welding needs a positive finite minor-arc contact length.')
    tangent = np.cross(normal, middle)
    half = length/(2*radius)
    ends = np.cos(half)*middle+np.sin(half)*np.array([tangent, -tangent])
    along = np.arctan2(points@tangent, points@middle)
    side = np.abs(np.arcsin(np.clip(points@normal, -1., 1.)))
    end_distance = [np.arctan2(np.linalg.norm(np.cross(points, end), axis=-1), points@end) for end in ends]
    return np.where(np.abs(along) <= half, side, np.minimum(*end_distance))*radius


def process_weights(s, observation, component, source, *, prepared=None):
    """Conservative whole-face profile, restricted to paths through material.

    All vertices of a face must lie inside the finite zone. Mere geographic
    proximity across an ocean gap or a protected-ID union cannot carry loading.
    Shared-edge paths start only at the observed contacting material faces.
    Under welded-stack policy 1 a path may also step between two overlapping
    faces of one plate's welded stack.
    """
    data = _component(s, component) if prepared is None else prepared
    width = float(getattr(s, 'config', {}).get('deformation_width_km', 400.))
    radius = float(s.material_surface.get('radius_km', 6371.))
    if not math.isfinite(width) or not 0 < width < math.pi*radius:
        raise ValueError('The native welding process-zone width must be finite and positive.')
    points = data['points'][data['faces']]
    observations = [observation] if isinstance(observation, dict) else list(observation)
    distances = np.full(points.shape[:2], np.inf)
    patches = []
    for row in observations:
        distances = np.minimum(distances, _distance(points.reshape(-1, 3), row, radius).reshape(-1, 3))
        patches.extend(row['p_patches' if source == row['p'] else 'q_patches'])
    maximum = np.max(distances, axis=1)
    allowed = maximum < width
    reached = np.zeros(len(allowed), bool)
    queue = deque(map(int, np.flatnonzero(allowed & np.isin(data['patches'], patches))))
    while queue:
        face = queue.popleft()
        if reached[face]: continue
        reached[face] = True
        queue.extend(other for other in data['neighbours'][face] if allowed[other] and not reached[other])
    weight = np.where(reached, np.cos(.5*np.pi*np.minimum(maximum/width, 1.))**2, 0.)
    return weight, data


def update(s, observations, geometry, dt):
    """Accumulate each directed root once per physical interval, using max loading.

    Multiple local fronts cannot double a dose. Partial timesteps use exact
    constant-rate exposure integration or unloaded exponential decay. Leakage
    is not applied simultaneously to engaged roots: that would make a small
    sustained front permanently unable to mature solely because it is short.
    """
    if not enabled(s): return
    if not math.isfinite(float(dt)) or dt <= 0: raise ValueError('Welding timestep must be positive.')
    initialize(s)
    state = s.accretion_welding_state
    now = float(s.t); last = float(state['last_time_myr'])
    if not math.isfinite(now) or now < last-1e-10:
        raise ValueError('Welding physical time cannot move backwards.')
    elapsed = max(0., now-max(last, now-dt))
    gap = max(0., now-dt-last)
    # Per-component face unions preserve local participation under subdivision
    # and allow two fronts to engage different portions of the same root.
    groups = {}
    for observation in observations:
        for source, target, cid in ((observation['p'], observation['q'], observation['cp']),
                                    (observation['q'], observation['p'], observation['cq'])):
            key = (int(source), int(target), int(cid))
            groups.setdefault(key, []).append(observation)
    current, pair_faces = {}, {}
    for (source, target, cid), observations in groups.items():
        # The finite contact union is evaluated at each vertex before taking a
        # whole-face bound. Splitting the same geometric arc cannot change it.
        weight, data = process_weights(s, observations, geometry[source]['components'][cid], source)
        key = (source, target)
        if key not in pair_faces: pair_faces[key] = np.zeros(len(s.parcel_patch))
        pair_faces[key][data['material']] = np.maximum(pair_faces[key][data['material']], weight)
    for (source, target), face_weight in pair_faces.items():
        ids = np.flatnonzero(np.asarray(s.parcel_plate) == source)
        roots, inverse = np.unique(_roots(s)[ids], return_inverse=True)
        root_weight = np.ones(len(roots))
        np.minimum.at(root_weight, inverse, face_weight[ids])
        for root, w in zip(roots, root_weight):
            key = (int(s.plate_uid[source]), int(s.plate_uid[target]), int(root))
            # A root split across disconnected material cannot borrow engagement
            # from its other part. Explicit new motion owners have separate keys.
            current[key] = float(w), float(w)
    old = {tuple(map(int, key)): float(value) for key, value in zip(
        zip(state['source_uid'], state['target_uid'], state['root_id']), state['loading'])}
    keys = sorted(set(old) | set(current))
    gap_decay = math.exp(-gap/MEMORY_MYR)
    unloaded_decay = math.exp(-elapsed/MEMORY_MYR)
    loading = [old.get(key, 0.)*gap_decay*(1. if current.get(key, (0., 0.))[0] > 0. else unloaded_decay)
               +current.get(key, (0., 0.))[1]*elapsed for key in keys]
    for i, name in enumerate(TABLE[:3]): state[name] = np.array([key[i] for key in keys], np.int64)
    state['loading'] = np.array(loading, float)
    state['active_weight'] = np.array([current.get(key, (0., 0.))[0] for key in keys], float)
    state['last_time_myr'] = now
    state['geometry_signature'] = geometry_signature(s)
    s.native_accretion_diagnostics = _diagnostics(s, elapsed, gap)


def qualify(s, component, source, target, suture_strength=0.):
    """Accept a coherent component as a welded terrane.

    NOTE: the sutured-margin path argued for below is RETIRED (SUTURE_WELD_ENABLED
    is False; see the comment at that constant). The reasoning is kept because the
    criterion is still evaluated and reported as suture_would_qualify, and because
    the argument's premise -- that a mature suture should transfer ownership -- is
    what turned out wrong: it would have merged a whole continent.

    The original rule -- every root of the component engaged and mature -- is the
    right test for a TERRANE, which arrives as a unit and should weld only once
    all of it is in contact. It is unsatisfiable for a CONTINENT, because only the
    leading margin ever engages: at 332 Myr this run reported 1,184 component
    roots, 46 engaged and 3 mature, so `eligible` could never become true and
    accreted_km2 stood frozen at 29,475 km2 through 5,000 km of active collision.

    Continental collision completes on a different criterion, and the model has
    been computing it all along without consuming it: collision_contacts tracks
    cumulative_convergence_km per contact and derives
    suture_strength = -expm1(-convergence/100). One live contact carries 1,927 km
    and a strength of 0.99999999 -- more convergence than India-Asia -- while
    nothing reads either number. A margin that has been converging that long IS
    sutured, whatever the far interior of the continent is doing.

    So a mature suture qualifies on its own, provided the margin is genuinely
    engaged. Both paths still require the loading state to be current, and the
    terrane rule is untouched, so small-body accretion behaves exactly as before.
    """
    if not enabled(s): return dict(eligible=True, version=0)
    state = s.accretion_welding_state
    roots = np.unique(_roots(s)[np.asarray(component['parcel_indices'], int)])
    source_uid, target_uid = int(s.plate_uid[source]), int(s.plate_uid[target])
    use = (state['source_uid'] == source_uid) & (state['target_uid'] == target_uid)
    table = {int(root): (float(load), float(w)) for root, load, w in zip(
        state['root_id'][use], state['loading'][use], state['active_weight'][use])}
    values = np.array([table.get(int(root), (0., 0.)) for root in roots], float).reshape(-1, 2)
    engaged = values[:, 1] > 0.
    mature = values[:, 0] >= THRESHOLD
    current = abs(float(state['last_time_myr'])-float(s.t)) <= 1e-10
    terrane = bool(len(roots) and np.all(engaged & mature))
    # A sutured margin needs real engagement as well as accumulated convergence,
    # so a stale or purely nominal contact cannot weld a continent.
    suture_criterion = bool(len(roots) and float(suture_strength) >= SUTURE_WELD_STRENGTH
                            and engaged.sum() >= SUTURE_WELD_MIN_ROOTS and mature.any())
    sutured = SUTURE_WELD_ENABLED and suture_criterion
    return dict(version=VERSION, eligible=bool(current and (terrane or sutured)),
        qualified_by=('terrane' if terrane else 'suture' if sutured else 'none'),
        suture_strength=float(suture_strength), suture_would_qualify=suture_criterion,
        component_roots=len(roots), engaged_roots=int(engaged.sum()), mature_roots=int(mature.sum()),
        minimum_root_loading=float(values[:, 0].min(initial=math.inf)) if len(roots) else 0.,
        minimum_process_weight=float(values[:, 1].min(initial=math.inf)) if len(roots) else 0.,
        source_uid=source_uid, target_uid=target_uid, epoch_myr=float(s.t),
        geometry_signature=state['geometry_signature'])


def validate_plan(s, plan):
    if not enabled(s): return True
    row = plan.get('local_welding_qualification', {})
    if row.get('version') != VERSION or not row.get('eligible'): return False
    if abs(float(row.get('epoch_myr', -1.))-float(s.t)) > 1e-10: return False
    if row.get('geometry_signature') != geometry_signature(s): return False
    # Re-check with the SAME suture strength the plan was qualified under, taken
    # from the plan rather than re-derived: a plan qualified by suture would
    # otherwise be rejected here by a default of 0.
    current = qualify(s, {'parcel_indices': plan['parcel_indices']}, plan['source'], plan['target'],
                      suture_strength=float(row.get('suture_strength', 0.)))
    return current['eligible']


def snapshot_fields(s):
    if not enabled(s): return {}
    state = s.accretion_welding_state
    stack = getattr(s, 'accretion_welded_stack_version', 0)
    return dict(native_accretion_version=VERSION,
        native_accretion_welding_time_myr=float(state['last_time_myr']),
        native_accretion_diagnostics=deepcopy(s.native_accretion_diagnostics),
        **({'accretion_welded_stack_version': int(stack)} if stack else {}),
        **{'accretion_welding_'+name: state[name].copy() for name in TABLE})


def validate_frame(frame):
    version = frame.get('native_accretion_version', 0)
    if type(version) is not int or version not in (0, VERSION): raise ValueError('Unsupported native accretion policy.')
    # local_accretion.WELDED_STACK_VERSION; importing it here would be circular.
    stack = frame.get('accretion_welded_stack_version', 0)
    if type(stack) is not int or stack not in (0, 1):
        raise ValueError('Unsupported welded-stack accretion policy.')
    if stack and version != VERSION:
        raise ValueError('Welded-stack accretion requires root-local welding.')
    if version == 0:
        if any(name in frame for name in ARRAY_FIELDS) or 'native_accretion_welding_time_myr' in frame or 'native_accretion_diagnostics' in frame:
            raise ValueError('Local welding fields require their native policy version.')
        return
    arrays = [np.asarray(frame.get(name)) for name in ARRAY_FIELDS]
    if any(a.ndim != 1 or a.shape != arrays[0].shape or a.dtype.kind not in 'iuf' or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Local welding fields must be finite aligned vectors.')
    if any(a.dtype.kind not in 'iu' or np.any(a < 0) for a in arrays[:3]) or np.any(arrays[0] == arrays[1]):
        raise ValueError('Local welding requires distinct integer source/target and root identity.')
    if np.any(arrays[3] < 0.) or np.any((arrays[4] < 0.) | (arrays[4] > 1.)):
        raise ValueError('Local welding dose and process weights are outside physical bounds.')
    keys = list(zip(*arrays[:3]))
    if len(set(keys)) != len(keys): raise ValueError('Duplicate directed material-root welding history.')
    time = frame.get('native_accretion_welding_time_myr')
    if not isinstance(time, (int, float)) or not math.isfinite(time) or time < 0 or time > frame.get('time_myr', time)+1e-10:
        raise ValueError('Invalid local welding physical time.')
    row = frame.get('native_accretion_diagnostics')
    if (not isinstance(row, dict) or row.get('version') != VERSION
            or row.get('memory_myr') != MEMORY_MYR or row.get('maturity_threshold') != THRESHOLD
            or row.get('loading_units') != 'process-weighted Myr' or row.get('whole_component_required') is not True):
        raise ValueError('Local welding diagnostics must declare the versioned process law.')
    for name in ('process_zone_width_km', 'elapsed_myr', 'unloaded_gap_myr'):
        value = row.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid local welding interval or process-zone diagnostic.')
    if not 0 < row['process_zone_width_km'] < math.pi*6371.:
        raise ValueError('Invalid local welding process-zone width.')
    counts = dict(directed_root_records=len(arrays[0]),currently_engaged_roots=int(np.count_nonzero(arrays[4] > 0.)),
                  mature_engaged_roots=int(np.count_nonzero((arrays[3] >= THRESHOLD) & (arrays[4] > 0.))))
    if any(type(row.get(name)) is not int or row[name] != count for name,count in counts.items()):
        raise ValueError('Local welding diagnostic counts do not match the saved root arrays.')
    if not isinstance(row.get('qualification_checks'), list):
        raise ValueError('Local welding diagnostics require a list of component qualifications.')
