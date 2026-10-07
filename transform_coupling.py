"""Simultaneous dissipative transform resistance, a bounded kinematic closure.

Relative along-front angular motion is penalized in an area-weighted metric.
The angular shear axis of a great-circle segment is its constant plane normal:
integrating its physical length therefore survives collinear subdivision.
This is not a friction/thermal or physical angular-momentum solver.
"""
from copy import deepcopy
import math
import numpy as np

VERSION = 1
COUPLING_WIDTH_KM = 500.
RELAXATION_MYR = 15.
RADIUS_KM = 6371.


def enabled(s):
    return getattr(s, 'transform_coupling_version', 0) == VERSION


def _fronts(s):
    native = getattr(s, 'native_boundary_geometry', None)
    if isinstance(native, dict):
        for a, b, edge in zip(native['segments_start'], native['segments_end'], native['contact_index']):
            edge = int(edge)
            if s.bcode[edge] != 3:
                continue
            axis = np.cross(a, b)
            sine = float(np.linalg.norm(axis))
            if sine <= 1e-14:
                continue
            yield edge, axis/sine, RADIUS_KM*math.atan2(sine, float(np.dot(a, b)))
    else:
        for edge in np.flatnonzero(s.bcode == 3):
            center, normal = s.bmid[edge], s.bn[edge]
            axis = np.cross(center, np.cross(center, normal))
            length = float(np.linalg.norm(axis))
            if length > 1e-14:
                yield int(edge), axis/length, float(s.bl[edge])


def apply(s, dt):
    """One SPD solve for every active transform; no ordered pairwise impulses.

    Minimize .5 sum A_p |omega_p-omega0_p|² plus
    .5 dt W/tau sum L_e [(omega_q-omega_p).axis_e]².
    Equal/opposite area-weighted changes and nonincreasing omega² follow from
    this convex relative-motion penalty. Normal motion receives no penalty.
    """
    if not enabled(s):
        return
    if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Transform resistance needs finite positive elapsed time.')
    before = np.asarray(s.omega, float).copy()
    area = np.bincount(s.plate, weights=s.cell_area, minlength=len(before))
    if not np.isfinite(before).all() or not np.isfinite(area).all() or np.any(area < 0):
        raise ValueError('Transform motion requires finite velocities and nonnegative owner areas.')
    groups = {}
    count = 0
    for edge, axis, length in _fronts(s):
        if not np.isfinite(axis).all() or not math.isfinite(length) or length < 0:
            raise ValueError('Transform geometry requires finite axes and nonnegative lengths.')
        p, q = sorted((int(s.bp[edge]), int(s.bq[edge])))
        if p == q or not (s.active[p] and s.active[q]) or min(area[p], area[q]) <= 0 or length == 0:
            continue
        # The dyad is sign-invariant. This only canonicalizes exact duplicates.
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        groups.setdefault((p, q, *map(float, axis)), []).append(length)
        count += 1
    fronts = sorted(groups)
    owners = sorted({p for row in fronts for p in row[:2]})
    slots = {p: i for i, p in enumerate(owners)}
    root = np.sqrt(area[owners])
    common = area[owners]@before[owners]/area[owners].sum() if owners else np.zeros(3)
    shared = bool(owners) and np.array_equal(before[owners], np.broadcast_to(before[owners[0]], (len(owners), 3)))
    initial = np.zeros(3*len(owners)) if shared else (root[:, None]*(before[owners]-common)).ravel()
    hessian = np.eye(len(initial))
    operator = np.zeros_like(hessian)
    for row in fronts:
        p, q = row[:2]
        axis = np.array(row[2:])
        n = np.zeros((len(owners), 3))
        n[slots[p]] = -axis/np.sqrt(area[p])
        n[slots[q]] = axis/np.sqrt(area[q])
        n = n.ravel()
        operator += (float(dt)*COUPLING_WIDTH_KM/RELAXATION_MYR*math.fsum(groups[row]))*np.outer(n, n)
    hessian += operator
    after = before.copy()
    initial_penalty = float(initial@operator@initial)
    residual = 0.
    if initial_penalty > 0:
        solved = np.linalg.solve(hessian, initial)
        residual = float(np.linalg.norm(hessian@solved-initial))
        if residual > 1e-11*max(float(np.linalg.norm(initial)), 1e-20):
            raise ValueError('Transform coupled solve did not converge.')
        after[owners] = common+solved.reshape(-1, 3)/root[:, None]
    energy0, energy1 = (float(np.sum(area[:, None]*v*v)) for v in (before, after))
    motion_residual = float(np.linalg.norm(area@(after-before)))
    motion_scale = max(float(np.sum(area*np.linalg.norm(before, axis=1))), 1e-20)
    if (energy1 > energy0+1e-12*max(energy0, 1e-30)
            or motion_residual > 1e-12*motion_scale or not np.isfinite(after).all()):
        raise ValueError('Transform resistance violated its dissipative area-weighted motion bounds.')
    diagnostics = dict(version=VERSION, input_segments=count, distinct_axes=len(fronts),
        length_km=math.fsum(math.fsum(groups[row]) for row in fronts),
        participating_plates=len(owners), coupling_width_km=COUPLING_WIDTH_KM,
        relaxation_myr=RELAXATION_MYR, step_duration_myr=float(dt),
        area_basis='current control-face owner area including ocean',
        rotational_energy_proxy_before=energy0, rotational_energy_proxy_after=energy1,
        rotational_energy_proxy_removed=energy0-energy1,
        area_weighted_motion_residual=motion_residual, stationarity_residual=residual,
        normal_penalty=False, model='simultaneous relative along-front quadratic resistance')
    s.omega[:] = after
    s.transform_motion_diagnostics = diagnostics


def snapshot(s):
    if not enabled(s):
        return {}
    return dict(transform_coupling_version=VERSION,
                transform_motion_diagnostics=deepcopy(getattr(s, 'transform_motion_diagnostics',
                    dict(version=VERSION, state='initialized', coupling_width_km=COUPLING_WIDTH_KM,
                         relaxation_myr=RELAXATION_MYR))))
