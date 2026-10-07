"""Read-only audit: is the slab pull in a state attached to real subduction?

Answers, per trench system, per plate and globally, from the saved state alone:

* loaded and converging  -- slab pull where ocean is actually being consumed
                             (reported by length and by pull force);
* loaded but stationary  -- pull held by a balanced plate (|v_n| <= floor);
* loaded but opening     -- pull on a trace that is separating (no intake);
* converging but unloaded -- kinematic subduction with no slab force behind it;
* polarity mismatch       -- the live kinematic downgoing side differs from
                             the slab owner on a converging loaded edge;
* trace inflation         -- resolved boundary length over a smooth length
                             (spanning tree through a plate's loaded contact
                             midpoints; per plate, since one trench system may
                             hold only a few long contacts); the
                             staircase excess scales pull, since slab mass is
                             seeded and allocated per resolved metre;
* cancellation per plate  -- 1 - |net slab torque| / sum |edge slab torque|:
                             near 1 means opposing pulls cancel in the rigid
                             balance and only internal stress remains.

Nothing is modified. Floors and flags are review thresholds, not physics.
"""
from __future__ import annotations

import math
import numpy as np

import normal_partition
import plate_balance
import slab_memory

VERSION = 1
SPEED_FLOOR_KM_MYR = .5
CANCELLATION_FLAG = .5


def _spanning_km(points, lengths):
    """Smooth trench length: minimum spanning tree through contact midpoints.

    Approaches the smooth length as midpoints densify and, unlike the sum of
    resolved edge lengths, does not count staircase zigzag. Links longer than
    the two contacts' half-lengths plus 300 km bridge separate trenches and
    are not counted, so several trenches on one plate are not joined up.
    """
    points, lengths = np.asarray(points, float), np.asarray(lengths, float)
    if len(points) < 2:
        return float(lengths.sum())
    angle = lambda a: np.arccos(np.clip(points@a, -1., 1.))*plate_balance.RADIUS_KM
    best = angle(points[0])
    parent = np.zeros(len(points), int)
    inside = np.zeros(len(points), bool)
    inside[0] = True
    total = 0.
    for _ in range(len(points)-1):
        candidate = np.where(inside, np.inf, best)
        k = int(np.argmin(candidate))
        if candidate[k] <= .5*(lengths[k]+lengths[parent[k]])+300.:
            total += float(candidate[k])
        else:
            total += float(lengths[k])      # a separate trench: its own length
        inside[k] = True
        closer = angle(points[k]) < best
        best = np.where(closer, angle(points[k]), best)
        parent[closer] = k
    return total


def audit(s, floor=SPEED_FLOOR_KM_MYR):
    owners, load, _ = slab_memory.line_load(s)
    length = np.asarray(s.bl, float)
    speed = np.asarray(s.normal_speed, float)
    force = plate_balance.GRAVITY_M_S2*load*math.sin(math.radians(slab_memory.SUBDUCTION_DIP_DEG))
    loaded = (owners >= 0) & (force > 0.) & (length > 1e-10)
    converging, opening = speed < -floor, speed > floor
    stationary = ~converging & ~opening
    kinematic = normal_partition.subduction(s) & converging & (np.asarray(s.down) >= 0)
    mismatch = loaded & converging & (np.asarray(s.down) >= 0) & (np.asarray(s.down) != owners)
    ids = np.asarray(s.trench_id)
    names = {int(p): s.names[p] for p in np.flatnonzero(s.active)}

    def km(mask):
        return float(length[mask].sum())

    rows = []
    for row in s.trench_systems:
        if row['phase'] == 'joined':
            continue
        mine = loaded & (ids == row['id'])
        trace = ids == row['id']
        rows.append(dict(id=int(row['id']), phase=row['phase'],
            downgoing_plate_uid=int(row['downgoing_plate_uid']),
            overriding_plate_uid=int(row['overriding_plate_uid']),
            inherited=bool(row.get('initial_subduction')),
            retained_excess_mass_kg=float(row.get(slab_memory.RETAINED_MASS_FIELD, 0.)),
            loaded_length_km=km(mine), trace_length_km=km(trace),
            mean_line_force_n_per_m=float(np.average(force[mine], weights=length[mine])) if mine.any() else 0.,
            loaded_converging_km=km(mine & converging), loaded_stationary_km=km(mine & stationary),
            loaded_opening_km=km(mine & opening),
            mean_normal_speed_km_myr=float(np.average(speed[trace], weights=length[trace])) if km(trace) > 0. else None))

    plates = []
    for p in np.flatnonzero(s.active):
        mine = loaded & (owners == p)
        if not mine.any():
            continue
        edges = np.flatnonzero(mine)
        direction = np.where((owners[edges] == np.asarray(s.bp)[edges])[:, None],
                             np.asarray(s.bn)[edges], -np.asarray(s.bn)[edges])
        torque = np.cross(np.asarray(s.bmid)[edges], direction)*(force[edges]*length[edges]*1e3)[:, None]
        gross = float(np.linalg.norm(torque, axis=1).sum())
        net = float(np.linalg.norm(torque.sum(axis=0)))
        systems = sorted({int(i) for i in ids[edges] if i > 0})
        keep = mine & (length > 1e-6)
        smooth = _spanning_km(np.asarray(s.bmid)[keep], length[keep])
        plates.append(dict(plate_uid=int(s.plate_uid[p]), name=names[int(p)], trench_systems=len(systems),
            loaded_length_km=km(mine), smooth_length_km=smooth,
            trace_inflation=km(mine)/smooth if smooth > 0. else None, gross_slab_force_n=gross*1.,
            cancellation=1.-net/gross if gross > 0. else 0.,
            complex=bool(len(systems) >= 2 and gross > 0. and 1.-net/gross >= CANCELLATION_FLAG),
            loaded_converging_fraction=km(mine & converging)/km(mine)))

    total = km(loaded)
    smooth = sum(p['smooth_length_km'] for p in plates)
    # Length shares count a barely loaded initiating trench like a mature
    # slab; force shares weigh each edge by the pull it actually applies.
    pull = force*length*1e3

    def pull_share(mask):
        denominator = float(pull[loaded].sum())
        return float(pull[mask & loaded].sum())/denominator if denominator > 0. else 0.
    summary = dict(version=VERSION, time_myr=float(s.t), speed_floor_km_myr=floor,
        loaded_length_km=total,
        trace_inflation=total/smooth if smooth > 0. else None,
        loaded_converging_fraction=km(loaded & converging)/total if total else 0.,
        loaded_stationary_fraction=km(loaded & stationary)/total if total else 0.,
        loaded_opening_fraction=km(loaded & opening)/total if total else 0.,
        pull_converging_fraction=pull_share(converging), pull_stationary_fraction=pull_share(stationary),
        pull_opening_fraction=pull_share(opening), total_slab_pull_n=float(pull[loaded].sum()),
        converging_unloaded_km=km(kinematic & ~loaded),
        polarity_mismatch_km=km(mismatch),
        live_systems=sum(r['phase'] not in ('shutdown',) for r in rows),
        shutdown_systems=sum(r['phase'] == 'shutdown' for r in rows),
        complex_plates=[p['name'] for p in plates if p['complex']])
    return dict(summary=summary, plates=plates, trenches=rows)
