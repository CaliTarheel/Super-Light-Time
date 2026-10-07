"""Versioned, portable authored regions for existing material deformation.

Regions are fixed geographic spherical caps. They change the bounded response
to admitted boundary motion, never mantle forces, plate velocity, crust kind,
reference area or column budgets. This first version does not prescribe rift
breakup, uplift, volcanism, or checkpoint branching.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real
import re

import numpy as np

VERSION = 1
MAX_REGIONS = 32
_FIELDS = {'id', 'kind', 'lon_deg', 'lat_deg', 'radius_deg', 'strength',
           'start_myr', 'end_myr', 'reason'}


def _number(value, label, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or not low <= value <= high):
        raise ValueError(f'World design {label} must be a finite number from {low} to {high}.')
    return float(value)


def normalize(value=None):
    """Validate and deep-copy a portable design; omitted means exact baseline."""
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value)-{'version', 'enabled', 'revision', 'interventions'}:
        raise ValueError('World design must contain version, enabled, revision and interventions only.')
    version = value.get('version', VERSION)
    if isinstance(version, bool) or version != VERSION:
        raise ValueError('Unsupported world design version.')
    enabled = value.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('World design enabled must be true or false.')
    revision = _number(value.get('revision', 0), 'revision', 0, 2147483647)
    if revision != int(revision):
        raise ValueError('World design revision must be a nonnegative integer.')
    rows = value.get('interventions', [])
    if not isinstance(rows, list) or len(rows) > MAX_REGIONS:
        raise ValueError(f'World design permits at most {MAX_REGIONS} geographic regions.')
    result = []
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != _FIELDS:
            raise ValueError('Every authored region needs its ID, kind, geographic cap, strength, interval and reason.')
        uid = row['id']
        if not isinstance(uid, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', uid) or uid in ids:
            raise ValueError('Authored region IDs must be unique portable names of at most 64 characters.')
        ids.add(uid)
        if row['kind'] not in ('protected', 'weak'):
            raise ValueError('Authored region kind must be protected or weak.')
        if not isinstance(row['reason'], str) or not 1 <= len(row['reason'].strip()) <= 500:
            raise ValueError('Give every authored region a reason of 1–500 characters.')
        item = dict(id=uid, kind=row['kind'], reason=row['reason'].strip())
        for name, limits in dict(lon_deg=(-180, 180), lat_deg=(-90, 90), radius_deg=(.25, 45),
                                 strength=(0, 1), start_myr=(0, 1000), end_myr=(0, 1000)).items():
            item[name] = _number(row[name], name, *limits)
        item['lon_deg'] = (item['lon_deg']+180.) % 360.-180.
        if item['end_myr'] <= item['start_myr']:
            raise ValueError('An authored region must end after its starting time.')
        result.append(item)
    return dict(version=VERSION, enabled=enabled, revision=int(revision), interventions=result)


def active_regions(design, time_myr):
    if not design['enabled']:
        return []
    return [row for row in design['interventions']
            if row['strength'] > 0 and row['start_myr'] <= time_myr < row['end_myr']]


def center_xyz(row):
    lon, lat = np.radians([row['lon_deg'], row['lat_deg']])
    return np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])


def cap_weights(points, row):
    """Interior unit weight, cosine taper in outer 20%; spherical, seam-free."""
    points = np.asarray(points, float)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or np.any(np.abs(np.linalg.norm(points, axis=1)-1) > 1e-8)):
        raise ValueError('World design sampling requires unit spherical points.')
    centre = center_xyz(row)
    distance = np.degrees(np.arctan2(np.linalg.norm(np.cross(points, centre), axis=1), points@centre))
    taper = np.clip((distance/row['radius_deg']-.8)/.2, 0., 1.)
    weight = .5+.5*np.cos(np.pi*taper)
    weight[distance >= row['radius_deg']] = 0.
    return weight


def rotate_region(row, matrix):
    """Return a cap in another orientation, preserving authored IDs and time."""
    result = deepcopy(row)
    point = center_xyz(row) @ np.asarray(matrix, float)
    result['lon_deg'] = ((math.degrees(math.atan2(point[1], point[0]))+180.) % 360.-180.
                         if math.hypot(point[0], point[1]) > 1e-14 else 0.)
    result['lat_deg'] = math.degrees(math.asin(float(np.clip(point[2], -1., 1.))))
    return result


def protected_face_mask(mesh, row):
    """Exact cap/convex spherical triangle contact, including sub-face caps."""
    triangles = np.asarray(mesh['vertices'])[np.asarray(mesh['faces'])]
    first, last = triangles, np.roll(triangles, -1, axis=1)
    normal = np.cross(first, last)
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-30)
    centre = center_xyz(row)
    side = normal@centre
    inside = (side >= -1e-13).all(axis=1)
    projected = centre-side[..., None]*normal
    projected /= np.maximum(np.linalg.norm(projected, axis=2, keepdims=True), 1e-30)
    on_arc = ((np.sum(np.cross(first, projected)*normal, axis=2) >= -1e-13)
              & (np.sum(np.cross(projected, last)*normal, axis=2) >= -1e-13))
    vertex_distance = np.arctan2(np.linalg.norm(np.cross(first, centre), axis=2), first@centre)
    edge_distance = np.where(on_arc, np.abs(np.arcsin(np.clip(side, -1., 1.))), np.inf)
    distance = np.minimum(vertex_distance.min(axis=1), edge_distance.min(axis=1))
    return inside | (distance < math.radians(row['radius_deg']))


def initialize(s):
    s.world_design = normalize(s.config.get('world_design'))
    s.world_design_active_ids = []
    s.world_design_diagnostics = dict(active_regions=0, protected_vertices=0,
                                      biased_vertices=0, response_min=1., response_max=1.)
    if s.world_design['interventions']:
        s._record('authored_world_design', 'The author supplied regional deformation controls.', details=dict(
            origin='authored', world_design=deepcopy(s.world_design),
            scope='Fixed geographic regions; response to admitted boundary motion only.'))
    record_transitions(s)


def _design(s):
    # Checkpoints capture these fields. Older fixture callers may only expose config.
    return getattr(s, 'world_design', None) or normalize(s.config.get('world_design'))


def record_transitions(s):
    design = _design(s)
    active = {row['id'] for row in active_regions(design, float(s.t))}
    previous = set(getattr(s, 'world_design_active_ids', []))
    for row in design['interventions']:
        uid = row['id']
        if (uid in active) == (uid in previous):
            continue
        starting = uid in active
        s._record('authored_region_started' if starting else 'authored_region_ended',
            f"Author's {row['kind']} region {uid} {'started' if starting else 'ended'}: {row['reason']}",
            details=dict(origin='authored', design_revision=design['revision'], region=deepcopy(row)))
    s.world_design_active_ids = sorted(active)


def next_transition(s):
    design = _design(s)
    if not design['enabled']:
        return None
    future = [row[key] for row in design['interventions'] if row['strength'] > 0
              for key in ('start_myr', 'end_myr') if row[key] > float(s.t)+1e-10]
    return min(future) if future else None


def validate_branch(previous, proposed, time_myr):
    """Permit future controls without reinterpreting a recorded authored past."""
    previous, proposed = normalize(previous), normalize(proposed)
    time_myr = _number(time_myr, 'branch time', 0, 1000)
    before = {row['id']: row for row in previous['interventions']}
    after = {row['id']: row for row in proposed['interventions']}
    historical = {uid: row for uid, row in before.items()
                  if previous['enabled'] and row['strength'] > 0 and row['start_myr'] < time_myr}
    if historical and not proposed['enabled']:
        raise ValueError('A branch must retain its recorded active design; add future regions instead of disabling its past.')
    if any(after.get(uid) != row for uid, row in historical.items()):
        raise ValueError('A branch cannot remove or edit a region that already affected its preserved history.')
    if proposed['enabled']:
        for uid, row in after.items():
            unchanged = previous['enabled'] and before.get(uid) == row
            if not unchanged and row['strength'] > 0 and row['start_myr'] < time_myr:
                raise ValueError(f'New or changed region {uid} must start at or after the {time_myr:g} Myr branch checkpoint.')
    return proposed


def deformation_options(s):
    design = _design(s)
    rows = active_regions(design, float(s.t)) if design['enabled'] else []
    if not rows:
        s.world_design_diagnostics = dict(active_regions=0, protected_vertices=0,
            biased_vertices=0, response_min=1., response_max=1.)
        return {}
    mesh = s.material_surface
    points, faces = mesh['vertices'], mesh['faces']
    protection = np.zeros(len(points))
    weakness = np.zeros(len(points))
    protected = np.zeros(len(points), bool)
    for row in rows:
        weight = cap_weights(points, row)
        if row['kind'] == 'weak':
            weakness = np.maximum(weakness, weight*row['strength'])
        else:
            protection = np.maximum(protection, weight*row['strength'])
            if row['strength'] == 1.:
                # Protect entire intersected indexed faces: no triangle in the
                # protected footprint may shear around one constrained vertex.
                selected = protected_face_mask(mesh, row)
                protected[np.unique(faces[selected])] = True
    response = (1.+.5*weakness)*(1.-protection)
    response[protected] = 0.
    s.world_design_diagnostics = dict(active_regions=len(rows),
        protected_vertices=int(protected.sum()), biased_vertices=int(np.count_nonzero(response != 1.)),
        response_min=float(response.min(initial=1.)), response_max=float(response.max(initial=1.)),
        active_region_ids=[row['id'] for row in rows],
        model='Bounded response to admitted boundary motion; geographic caps fixed in world coordinates.',
        footprint='Full protection includes all vertices of faces touching the cap; mesh resolution limits edges.')
    return dict(vertex_protected=protected, vertex_response=response)


def snapshot_metadata(s):
    design = _design(s)
    return dict(world_design=deepcopy(design), world_design_diagnostics=dict(
        deepcopy(getattr(s, 'world_design_diagnostics', {})),
        active_at_epoch_ids=[row['id'] for row in active_regions(design, float(s.t))]))
