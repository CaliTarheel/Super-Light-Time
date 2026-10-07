"""Opt-in slab pull where the slab actually is (``slab_allocation: fed_v1``).

Version 2 of ``slab_memory`` carries each trench system's retained slab as one
conserved inventory and spreads it uniformly along the system's current trace.
Where part of a system converges and part is stationary or opening, the
stationary part then pulls with slab it never consumed; the subduction audit
measured 38% of loaded trench in that state on a mixed-plate world.

Here each system also carries **anchors**: points on the sphere holding part of
its retained area and excess mass. Anchors move with the overriding plate, as
the trench trace does. New slab is added at the boundary contact where the
ocean was actually consumed; decay (the existing retention law) acts on every
anchor alike. The force reader assigns each anchor's mass to the system's
current edges within ``SUPPORT_KM`` and applies the upper-mantle window to each
edge's local column. Slab that no current edge is near (its trace has moved
away) stays in the inventory and retires, but does not pull.

The row's existing ledgers remain authoritative and unchanged in meaning;
anchor sums equal the row's retained area and mass at every commit. Rows on the
experimental local-neck path (``slab_tether_local``) keep that path's own
spatial allocation and carry no anchors. This is a localization of the same
inventory, not a slab-shape, depth or mantle-flow solution.
"""
from __future__ import annotations

import math
import numpy as np

VERSION = 1
FIELD = 'slab_anchors'
POLICIES = ('uniform', 'fed_v1')
RADIUS_KM = 6371.
# A slab pulls on trench within this distance of where it went down: about
# one upper-mantle down-dip length at 50 degrees (862 km) halved, i.e. the
# lateral reach of a coherent slab segment. Anchors closer than MERGE_KM are
# combined (mass-weighted position) to bound state size.
SUPPORT_KM = 400.
MERGE_KM = 150.
MAX_ANCHORS = 192


def normalize(value='uniform'):
    if not isinstance(value, str) or value not in POLICIES:
        raise ValueError('Slab allocation must be uniform or fed_v1.')
    return value


def enabled(s):
    version = getattr(s, 'slab_allocation_version', 0)
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer)) or version not in (0, 1):
        raise ValueError('Unsupported slab allocation version.')
    return version == VERSION


def carries(row):
    return FIELD in row


def _unit(v):
    v = np.asarray(v, float)
    return v/np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


def _arrays(row):
    anchors = row.get(FIELD, [])
    if not anchors:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    xyz = np.array([a['xyz'] for a in anchors], float)
    area = np.array([a['area_km2'] for a in anchors], float)
    mass = np.array([a['excess_mass_kg'] for a in anchors], float)
    return xyz, area, mass


def _store(row, xyz, area, mass):
    keep = (area > 0.) | (mass > 0.)
    row[FIELD] = [dict(xyz=_unit(p).tolist(), area_km2=float(a), excess_mass_kg=float(m))
                  for p, a, m in zip(xyz[keep], area[keep], mass[keep])]


def _match_totals(row):
    """Remove floating drift so anchor sums equal the row's retained ledgers."""
    import slab_memory
    xyz, area, mass = _arrays(row)
    target_area = float(row.get('slab_retained_area_km2', 0.))
    target_mass = float(row.get(slab_memory.RETAINED_MASS_FIELD, 0.))
    if len(area) and area.sum() > 0.:
        area *= target_area/area.sum()
    if len(mass) and mass.sum() > 0.:
        mass *= target_mass/mass.sum()
    if (target_area > 0. or target_mass > 0.) and not len(area):
        raise ValueError('A retained slab inventory lost its anchors.')
    _store(row, xyz, area, mass)


def validate(row):
    import slab_memory
    if not carries(row):
        return
    xyz, area, mass = _arrays(row)
    if (not np.isfinite(xyz).all() or np.any(np.abs(np.linalg.norm(xyz, axis=1)-1.) > 1e-9)
            or not np.isfinite(area).all() or not np.isfinite(mass).all()
            or np.any(area < 0.) or np.any(mass < 0.)):
        raise ValueError('Slab anchors require unit positions and finite nonnegative inventories.')
    for total, key in ((area.sum(), 'slab_retained_area_km2'), (mass.sum(), slab_memory.RETAINED_MASS_FIELD)):
        expected = float(row.get(key, 0.))
        if abs(total-expected) > max(1e-6, expected*1e-9):
            raise ValueError('Slab anchors do not sum to the retained inventory.')


def seed(row, points, lengths):
    """Uniform baseline along a known trace (declared inherited slabs only)."""
    import slab_memory
    lengths = np.asarray(lengths, float)
    points = np.asarray(points, float)
    keep = lengths > 0.
    if not keep.any():
        _store(row, np.zeros((0, 3)), np.zeros(0), np.zeros(0))
        return
    share = lengths[keep]/lengths[keep].sum()
    _store(row, points[keep], share*float(row.get('slab_retained_area_km2', 0.)),
           share*float(row.get(slab_memory.RETAINED_MASS_FIELD, 0.)))
    _merge(row)


def ensure(s):
    """Give every conservative row anchors; an existing inventory starts uniform on its trace."""
    import slab_tether_local as local
    for row in s.trench_systems:
        if carries(row) or local.enabled(row) or row['phase'] == 'joined':
            continue
        trace = np.flatnonzero(np.asarray(s.trench_id) == row['id'])
        seed(row, np.asarray(s.bmid)[trace], np.asarray(s.bl)[trace])
        if float(row.get('slab_retained_area_km2', 0.)) > 0. and not row[FIELD]:
            # No current trace: keep the inventory at the recorded centre.
            _store(row, np.asarray([row['center']], float),
                   np.array([row['slab_retained_area_km2']]),
                   np.array([row.get('slab_retained_excess_mass_kg', 0.)]))


def _merge(row):
    xyz, area, mass = _arrays(row)
    if len(xyz) < 2:
        return
    cos_merge = math.cos(MERGE_KM/RADIUS_KM)
    order = np.argsort(-mass)
    used = np.zeros(len(xyz), bool)
    out_xyz, out_area, out_mass = [], [], []
    for i in order:
        if used[i]:
            continue
        group = np.flatnonzero(~used & (xyz@xyz[i] >= cos_merge))
        used[group] = True
        weight = mass[group] if mass[group].sum() > 0. else area[group]
        out_xyz.append(_unit((xyz[group]*weight[:, None]).sum(axis=0)) if weight.sum() > 0. else xyz[i])
        out_area.append(area[group].sum())
        out_mass.append(mass[group].sum())
    xyz, area, mass = np.array(out_xyz), np.array(out_area), np.array(out_mass)
    while len(xyz) > MAX_ANCHORS:
        # Combine the two closest anchors.
        dots = xyz@xyz.T
        np.fill_diagonal(dots, -2.)
        i, j = np.unravel_index(int(np.argmax(dots)), dots.shape)
        weight = np.array([mass[i], mass[j]]) if mass[i]+mass[j] > 0. else np.array([area[i], area[j]])
        xyz[i] = _unit(xyz[i]*weight[0]+xyz[j]*weight[1]) if weight.sum() > 0. else xyz[i]
        area[i] += area[j]; mass[i] += mass[j]
        xyz, area, mass = np.delete(xyz, j, 0), np.delete(area, j), np.delete(mass, j)
    _store(row, xyz, area, mass)


def advance(row, decay, retain_feed, located, gain_area, gain_mass):
    """Retire every anchor alike and add this step's feed where it went down.

    ``located`` lists (xyz, area_km2, mass_weight) per consuming contact. The
    row's own totals were already advanced by slab_memory; anchor sums are
    matched to them exactly afterwards.
    """
    xyz, area, mass = _arrays(row)
    area, mass = area*decay, mass*decay
    if located and (gain_area > 0. or gain_mass > 0.):
        points = np.array([p for p, _, _ in located], float)
        feed_area = np.array([a for _, a, _ in located], float)
        weight = np.array([w for _, _, w in located], float)
        area_share = feed_area/feed_area.sum() if feed_area.sum() > 0. else np.full(len(points), 1./len(points))
        mass_share = weight/weight.sum() if weight.sum() > 0. else area_share
        xyz = np.vstack((xyz, points))
        area = np.r_[area, gain_area*retain_feed*area_share]
        mass = np.r_[mass, gain_mass*retain_feed*mass_share]
    _store(row, xyz, area, mass)
    _merge(row)
    _match_totals(row)


def advect(row, rotation):
    from ridge_geometry import rotate
    if carries(row) and row[FIELD]:
        xyz, area, mass = _arrays(row)
        _store(row, rotate(xyz, rotation), area, mass)


def partition(parent, child, child_points, parent_points):
    """Split anchors by proximity to each side's trace; return area and mass fractions."""
    xyz, area, mass = _arrays(parent)
    child_points, parent_points = np.asarray(child_points, float), np.asarray(parent_points, float)
    if not len(xyz):
        _store(child, np.zeros((0, 3)), np.zeros(0), np.zeros(0))
        return 0., 0.
    if not len(parent_points):
        to_child = np.ones(len(xyz), bool)
    elif not len(child_points):
        to_child = np.zeros(len(xyz), bool)
    else:
        to_child = (xyz@child_points.T).max(axis=1) > (xyz@parent_points.T).max(axis=1)
    _store(child, xyz[to_child], area[to_child], mass[to_child])
    _store(parent, xyz[~to_child], area[~to_child], mass[~to_child])
    total_area, total_mass = area.sum(), mass.sum()
    return (float(area[to_child].sum()/total_area) if total_area > 0. else 0.,
            float(mass[to_child].sum()/total_mass) if total_mass > 0. else 0.)


def join(source, target):
    xs, as_, ms = _arrays(source)
    xt, at, mt = _arrays(target)
    _store(target, np.vstack((xt, xs)), np.r_[at, as_], np.r_[mt, ms])
    _store(source, np.zeros((0, 3)), np.zeros(0), np.zeros(0))
    _merge(target)


def edge_loads(row, edges, midpoints, lengths_km):
    """Per-edge (mass kg, area km2) from anchors within SUPPORT_KM of each edge.

    Each anchor's inventory is shared among the row's current edges within its
    support, in proportion to their length. Returns the arrays and the mass
    attached to no edge (it stays in the inventory but does not pull).
    """
    xyz, area, mass = _arrays(row)
    edge_mass, edge_area = np.zeros(len(edges)), np.zeros(len(edges))
    if not len(xyz) or not len(edges):
        return edge_mass, edge_area, float(mass.sum())
    lengths_km = np.asarray(lengths_km, float)
    near = (xyz@np.asarray(midpoints, float).T) >= math.cos(SUPPORT_KM/RADIUS_KM)
    weight = near*lengths_km[None, :]
    total = weight.sum(axis=1)
    attached = total > 0.
    share = np.divide(weight, total[:, None], out=np.zeros_like(weight), where=attached[:, None])
    edge_mass = share.T@mass
    edge_area = share.T@area
    return edge_mass, edge_area, float(mass[~attached].sum())
