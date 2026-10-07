"""Bounded inversion of material-attached inherited continental rift scars.

This is a kinematic worldbuilding closure, not a lithospheric stress solver.
Current compression at an attached boundary is transmitted a finite distance
into its plate. Short contacts, distant contacts and compression along the
rift strike contribute little. Only previously recorded extension supplies a
finite inversion budget; scar age alone never supplies uplift.
"""
from __future__ import annotations

import itertools
import math
import numpy as np
import normal_partition

RADIUS_KM = 6371.
INFLUENCE_KM = 1800.
CONTACT_LENGTH_KM = 300.
MAX_CAPACITY_M = 2000.
CAPACITY_PER_EXTENSION = 3.
MAX_RATE_M_MYR = 30.
FULL_LOAD_KM_MYR = 30.
PAIR_CHUNK = 256
_NEIGHBOR_OFFSETS = tuple(itertools.product((-1, 0, 1), repeat=3))


def _groups(points, cell_size):
    keys = np.floor((points + 1.) / cell_size).astype(np.int16)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    ends = np.cumsum(np.bincount(inverse, minlength=len(unique)))
    return {tuple(int(v) for v in key): group for key, group in
            zip(unique, np.split(order, ends[:-1]))}


def _boundary_load(s):
    """Recompute actual convergence so stale labels cannot create a load."""
    partitioned = normal_partition.enabled(s)
    indices = np.flatnonzero(np.asarray(s.bl) > 1e-10 if partitioned else np.isin(s.bcode, (2, 4)))
    if not len(indices):
        return None
    mid = np.asarray(s.bmid[indices], float)
    mid /= np.maximum(np.linalg.norm(mid, axis=1)[:, None], 1e-30)
    normal = np.asarray(s.bn[indices], float).copy()
    normal -= mid * np.sum(normal * mid, axis=1)[:, None]
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-30)
    p, q = s.bp[indices], s.bq[indices]
    velocity = np.cross(s.omega[q] - s.omega[p], mid) * RADIUS_KM
    signed = np.sum(velocity * normal, axis=1)
    shear = np.linalg.norm(velocity - signed[:, None] * normal, axis=1)
    valid = ((signed < 0.) if partitioned else (-signed > np.maximum(2., .35 * shear))) & (s.bl[indices] > 0)
    down = s.down[indices]
    collision = ((s.crust[s.ba[indices]] > 0) & (s.crust[s.bb[indices]] > 0)
                 if partitioned else s.bcode[indices] == 4)
    valid &= collision | ((down == p) | (down == q))
    indices, mid, normal, speed = indices[valid], mid[valid], normal[valid], -signed[valid]
    if not len(indices):
        return None
    return indices, mid, normal, speed


def inversion_increment(s, xyz, rift_tangent, owner_slots, extension_m, inverted_m, dt):
    """Return additional uplift in metres, without mutating simulation state.

    ``rift_tangent`` is material-attached rift strike, not extension direction.
    ``extension_m`` records inherited extensional depression; ``inverted_m``
    records *realized* cumulative inversion construction. The engine retains
    scar IDs, transports tangents and applies strength/saturation to this gain.

    Continental collision loads both plates; subduction loads only its
    overrider. The versioned normal-motion policy admits compression during
    simultaneous sliding; legacy worlds retain their label/obliquity gates.
    Great-circle parallel transport compares compression to the
    cross-rift direction. A compact taper reaches zero at1800 km. Length-
    weighted averaging and L/(L+300 km) contact coherence prevent cell-count
    amplification. Each sample consumes at most min(2000 m,3*extension_m),
    with an initial rate capped at30 m/Myr and exact constant-load integration.
    """
    if not math.isfinite(float(dt)) or dt < 0:
        raise ValueError('Inversion timestep must be finite and nonnegative.')
    points = np.asarray(xyz, dtype=float)
    tangent = np.asarray(rift_tangent, dtype=float)
    owners = np.asarray(owner_slots)
    extension = np.asarray(extension_m, dtype=float)
    inverted = np.asarray(inverted_m, dtype=float)
    count = len(points)
    if (points.shape != (count, 3) or tangent.shape != points.shape
            or any(field.shape != (count,) for field in (owners, extension, inverted))):
        raise ValueError('Rift samples need matching position, tangent, owner and budget arrays.')
    result = np.zeros(count)
    if not count or dt == 0:
        return result
    capacity = np.minimum(MAX_CAPACITY_M, CAPACITY_PER_EXTENSION * np.maximum(extension, 0.))
    remaining = np.maximum(capacity - np.maximum(inverted, 0.), 0.)
    scarred = (remaining > 0) & (np.linalg.norm(tangent, axis=1) > 1e-12) & (owners >= 0)
    if not np.any(scarred):
        return result
    boundary = _boundary_load(s)
    if boundary is None:
        return result
    edge, mid, normal, speed = boundary
    p, q, down = s.bp[edge], s.bq[edge], s.down[edge]
    collision = ((s.crust[s.ba[edge]] > 0) & (s.crust[s.bb[edge]] > 0)
                 if normal_partition.enabled(s) else s.bcode[edge] == 4)
    over = np.where(down == p, q, p)
    chord_limit = 2 * math.sin(INFLUENCE_KM / (2 * RADIUS_KM))
    cosine_limit = math.cos(INFLUENCE_KM / RADIUS_KM)
    for owner in np.unique(owners[scarred]):
        attached = ((p == owner) | (q == owner)) & (collision | (over == owner))
        if not np.any(attached):
            continue
        local_mid, local_normal = mid[attached], normal[attached]
        local_speed, local_length = speed[attached], np.asarray(s.bl[edge[attached]], float)
        sample_indices = np.flatnonzero(scarred & (owners == owner))
        sample_xyz = points[sample_indices].copy()
        sample_xyz /= np.maximum(np.linalg.norm(sample_xyz, axis=1)[:, None], 1e-30)
        # Projection removes accumulated roundoff without changing stored strike.
        strike = tangent[sample_indices].copy()
        strike -= sample_xyz * np.sum(strike * sample_xyz, axis=1)[:, None]
        strike /= np.maximum(np.linalg.norm(strike, axis=1)[:, None], 1e-30)
        cross_rift = np.cross(sample_xyz, strike)
        bins = _groups(local_mid, chord_limit)
        for key, samples in _groups(sample_xyz, chord_limit).items():
            neighbours = [bins[near] for offset in _NEIGHBOR_OFFSETS
                          if (near := tuple(a + b for a, b in zip(key, offset))) in bins]
            if not neighbours:
                continue
            candidates = np.concatenate(neighbours)
            for start in range(0, len(samples), PAIR_CHUNK):
                selected = samples[start:start + PAIR_CHUNK]
                x, across = sample_xyz[selected], cross_rift[selected]
                numerator, contact = np.zeros(len(selected)), np.zeros(len(selected))
                for begin in range(0, len(candidates), PAIR_CHUNK):
                    edges = candidates[begin:begin + PAIR_CHUNK]
                    m, n = local_mid[edges], local_normal[edges]
                    cosine = np.clip(x @ m.T, -1., 1.)
                    nearby = cosine > cosine_limit
                    if not np.any(nearby):
                        continue
                    distance = RADIUS_KM * np.arccos(cosine)
                    weight = np.maximum(1. - (distance / INFLUENCE_KM) ** 2, 0.) ** 2
                    # Parallel transport n from m to x along their short great
                    # circle: n' = n-(n.x)/(1+m.x)*(m+x). Since across.x=0,
                    # the alignment can be evaluated without an N*M*3 array.
                    alignment = across @ n.T - (x @ n.T) * (across @ m.T) / np.maximum(1. + cosine, 1e-12)
                    alignment2 = np.clip(alignment * alignment, 0., 1.)
                    length_weight = local_length[edges] * weight
                    numerator += np.sum(length_weight * weight * alignment2 * local_speed[edges], axis=1)
                    contact += np.sum(length_weight, axis=1)
                compression = numerator / (CONTACT_LENGTH_KM + contact)
                rate = MAX_RATE_M_MYR * np.clip(compression / FULL_LOAD_KM_MYR, 0., 1.)
                destination = sample_indices[selected]
                # Integrate the remaining fraction analytically. Equal total
                # time under unchanged load yields the same finite construction.
                fraction = -np.expm1(-rate * dt / np.maximum(capacity[destination], 1e-30))
                result[destination] = remaining[destination] * fraction
    return result
