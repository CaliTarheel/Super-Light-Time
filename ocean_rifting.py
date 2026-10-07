"""Native spherical ocean deformation and connected, uncommitted split plans.

This reduced model resolves incompatible *boundary-velocity* loading with the
existing sparse membrane solver. Its strengths and damage are dimensionless
worldbuilding closures, not stresses in pascals. Thermal age changes strength;
age, plate size and elapsed time never supply a fracture load. Only positive
resolved extension grows damage. Compression heals it and cannot create a ridge.

History arrays belong to ocean material. A fixed control-mesh caller MUST carry
them with its conservative ocean transport, just as it carries thermal age.
They are not permanent damage clocks attached to geographical edge indices.
This module changes no input, plate identity, material ownership or mantle field.
"""
from __future__ import annotations

import math
import numpy as np

import rift_mechanics
from rift_mesh import coherent_cut

RADIUS_KM = 6371.
HISTORY_FIELDS = ('damage', 'tensile_strain', 'weakness', 'tensile_exposure_myr')


def initialize(face_count, *, weakness=None):
    """Zero unfractured material history; optional inherited weakness is [0,1]."""
    if isinstance(face_count, (bool, np.bool_)) or int(face_count) != face_count or face_count < 0:
        raise ValueError('Ocean history needs a nonnegative integer face count.')
    n = int(face_count)
    weak = np.zeros(n) if weakness is None else np.asarray(weakness, float)
    if weak.shape != (n,) or not np.isfinite(weak).all() or np.any((weak < 0) | (weak > 1)):
        raise ValueError('Inherited ocean weakness must be aligned and between zero and one.')
    return dict(damage=np.zeros(n), tensile_strain=np.zeros(n), weakness=weak.copy(),
                tensile_exposure_myr=np.zeros(n))


def cooling_strength(age_myr, weakness=0., damage=0.):
    """Bounded relative stiffness/yield proxy for cooling ocean lithosphere.

    Young lithosphere is easier to deform; an inherited damaged corridor can
    remain weak in an old plate. Chronological birth age is never reset here.
    The constants set a model contrast and are not laboratory yield strengths.
    """
    age, weak, failed = np.broadcast_arrays(np.asarray(age_myr, float),
                                           np.asarray(weakness, float), np.asarray(damage, float))
    if (not all(np.isfinite(x).all() for x in (age, weak, failed)) or np.any(age < 0)
            or np.any((weak < 0) | (weak > 1)) or np.any((failed < 0) | (failed > 1))):
        raise ValueError('Cooling age and normalized ocean damage/weakness must be finite and valid.')
    cold = np.clip(.45+.8*np.sqrt(age/80.), .45, 1.8)
    return np.maximum(.08, cold*np.exp(-1.5*weak)*(1-.75*failed))


def _validate(mesh, material, plates, history):
    xyz = np.asarray(mesh['xyz'], float)
    area = np.asarray(mesh['area'] if 'area' in mesh else mesh['area_km2'], float)
    edges = np.asarray(mesh['edge_faces'])
    owner, ocean, age = np.asarray(material['owner']), np.asarray(material['ocean']), np.asarray(material['age_myr'], float)
    omega, uids = np.asarray(plates['omega'], float), np.asarray(plates['uids'])
    active = np.asarray(plates.get('active', np.ones(len(omega), bool)), bool)
    n = len(xyz)
    if (xyz.shape != (n, 3) or not np.isfinite(xyz).all()
            or np.any(np.abs(np.linalg.norm(xyz, axis=1)-1) > 1e-6)
            or area.shape != (n,) or not np.isfinite(area).all() or np.any(area <= 0)
            or edges.ndim != 2 or edges.shape[1:] != (2,) or edges.dtype.kind not in 'iu'
            or np.any(edges < 0) or np.any(edges >= n) or np.any(edges[:, 0] == edges[:, 1])
            or owner.shape != (n,) or owner.dtype.kind not in 'iu'
            or ocean.shape != (n,) or ocean.dtype.kind != 'b'
            or age.shape != (n,) or not np.isfinite(age).all() or np.any(age < 0)
            or omega.ndim != 2 or omega.shape[1:] != (3,) or not np.isfinite(omega).all()
            or uids.shape != (len(omega),) or uids.dtype.kind not in 'iu'
            or active.shape != (len(omega),) or np.any(owner < 0) or np.any(owner >= len(omega))):
        raise ValueError('Native ocean mesh, material and plate arrays must be finite and aligned.')
    state = {}
    for key in HISTORY_FIELDS:
        value = np.asarray(history[key], float)
        if value.shape != (n,) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError('Ocean material history must contain aligned finite nonnegative arrays.')
        if key in ('damage', 'weakness') and np.any(value > 1):
            raise ValueError('Ocean damage and weakness must lie between zero and one.')
        state[key] = value.copy()
    return xyz, area, edges.astype(np.int64, copy=False), owner, ocean, age, omega, uids, active, state


def boundary_loading(xyz, owner, omega, active, boundaries, *, strength_scale=1.):
    """Length-weighted current boundary loading on actual adjacent native faces.

    No nearest-cell or nearest-node search occurs. Native boundary normals point
    from ba/bp toward bb/bq. The existing relative-motion closure transmits
    extension, 0.6 times compression and 0.25 times tangential slip. The membrane
    solver removes each component's best rigid Euler field before responding.
    """
    n = len(xyz)
    ba, bb, bp, bq = (np.asarray(boundaries[k]) for k in ('ba', 'bb', 'bp', 'bq'))
    mid, normal, length = (np.asarray(boundaries[k], float) for k in ('bmid', 'bn', 'bl'))
    m = len(ba)
    valid = np.asarray(boundaries.get('valid', np.ones(m, bool)), bool).copy()
    if (any(v.shape != (m,) or v.dtype.kind not in 'iu' for v in (ba, bb, bp, bq))
            or mid.shape != (m, 3) or normal.shape != (m, 3) or length.shape != (m,)
            or valid.shape != (m,) or not all(np.isfinite(v).all() for v in (mid, normal, length))
            or np.any(length < 0) or any(np.any(v < 0) or np.any(v >= n) for v in (ba, bb))
            or any(np.any(v < 0) or np.any(v >= len(omega)) for v in (bp, bq))):
        raise ValueError('Native boundary indices, geometry and owners must align.')
    valid &= (bp != bq) & active[bp] & active[bq] & (owner[ba] == bp) & (owner[bb] == bq)
    valid &= length > 0
    loads, weight = np.zeros((n, 3)), np.zeros(n)
    if np.any(valid):
        ba, bb, bp, bq, mid, normal, length = (v[valid] for v in (ba, bb, bp, bq, mid, normal, length))
        if (np.any(np.abs(np.linalg.norm(mid, axis=1)-1) > 1e-6)
                or np.any(np.abs(np.linalg.norm(normal, axis=1)-1) > 1e-6)
                or np.any(np.abs(np.sum(mid*normal, axis=1)) > 1e-6)):
            raise ValueError('Boundary midpoint and tangent normal must be unit spherical vectors.')
        relative = np.cross(omega[bq]-omega[bp], mid)*RADIUS_KM
        opening = np.sum(relative*normal, axis=1)
        shear = relative-opening[:, None]*normal
        drive = normal*(np.maximum(opening, 0.)-.6*np.maximum(-opening, 0.))[:, None]+.25*shear
        for face, sign in ((ba, 1.), (bb, -1.)):
            np.add.at(loads, face, sign*drive*length[:, None])
            np.add.at(weight, face, length)
    loads = np.divide(loads, weight[:, None], out=np.zeros_like(loads), where=weight[:, None] > 0)
    return loads*strength_scale, np.minimum(weight/800., 3.)


def _node_average(values, edges, length, n):
    numerator, denominator = np.zeros(n), np.zeros(n)
    for endpoints in edges.T:
        np.add.at(numerator, endpoints, values*length)
        np.add.at(denominator, endpoints, length)
    return np.divide(numerator, denominator, out=np.zeros(n), where=denominator > 0)


def _daughter_fit(xyz, area, velocity, child):
    """Two least-squares Euler residuals with zero common angular moment."""
    matrices, fits = [], []
    for selected in (~child, child):
        matrix = np.eye(3)*area[selected].sum()-np.einsum('n,ni,nj->ij', area[selected], xyz[selected], xyz[selected])
        moment = np.sum(np.cross(xyz[selected], velocity[selected])*area[selected, None], axis=0)/RADIUS_KM
        matrices.append(matrix)
        fits.append(np.linalg.pinv(matrix, rcond=1e-10)@moment)
    common = np.linalg.pinv(matrices[0]+matrices[1], rcond=1e-10)@(matrices[0]@fits[0]+matrices[1]@fits[1])
    return np.asarray(fits)-common


def update(mesh, material, plates, boundaries, history, dt, *, strength_scale=1.,
           iterations=None, rupture_strain=.35, damage_threshold=.95,
           min_component_faces=3, min_fraction=.04):
    """Advance material damage and propose connected ocean splits, without commits.

    Required dicts: mesh xyz(F,3), area(F), edge_faces(E,2); material owner(F),
    ocean(F, bool), age_myr(F); plates omega(P,3), uids(P), optional active(P);
    boundaries ba/bb/bp/bq/bmid/bn/bl, optional valid. See initialize for history.

    Non-ocean faces transmit loading as strong lithosphere and cannot be cut.
    An ocean belt can partition an ocean-only plate, including the primordial
    plate, or the oceanic part of a mixed plate without cutting its continents.
    Current tensile opening, accumulated strain, developed damage, a connected
    two-component partition and separating daughter fits are all required.
    """
    if (not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0
            or not np.isscalar(strength_scale) or not np.isfinite(strength_scale) or strength_scale < 0
            or not 0 < rupture_strain or not np.isfinite(rupture_strain)
            or not 0 < damage_threshold <= 1 or not 0 <= min_fraction < .5
            or min_component_faces < 1):
        raise ValueError('Ocean-rift time, strengths and failure safeguards must be positive and finite.')
    xyz, area, all_edges, owner, ocean, age, omega, uids, active, state = _validate(mesh, material, plates, history)
    n = len(xyz)
    same = (owner[all_edges[:, 0]] == owner[all_edges[:, 1]]) & active[owner[all_edges[:, 0]]]
    edge_indices = np.flatnonzero(same)
    edges = all_edges[same]
    strength = cooling_strength(age, state['weakness'], state['damage'])
    strength[~ocean] = 6.
    loads, weights = boundary_loading(xyz, owner, omega, active, boundaries, strength_scale=strength_scale)
    cap = max(120, int(math.ceil(10*math.sqrt(n)))) if iterations is None else iterations
    response = rift_mechanics.solve_loading(xyz, edges, strength, loads, load_weights=weights,
                                           iterations=cap, tolerance=1e-5)
    reliable = bool(response['converged'] or response['relative_residual'] < .01)
    length = response['edge_length_km']
    axial = response['edge_extension']/np.maximum(length, 1.)
    positive = np.minimum(.25*np.maximum(axial, 0.), .08)*dt
    negative = np.minimum(.25*np.maximum(-axial, 0.), .08)*dt
    node_tension = _node_average(positive, edges, length, n)
    node_compression = _node_average(negative, edges, length, n)
    node_tension[~ocean | ~active[owner]] = 0.
    if not reliable:
        node_tension[:] = 0.
    old = state['damage']
    state['damage'] = np.clip(old*np.exp(-dt/1200.-.5*node_compression)
                              + node_tension*(1+2*old)/(.55*np.maximum(strength, .1)), 0., 1.)
    state['tensile_strain'] += node_tension
    state['tensile_exposure_myr'] += np.where(node_tension > 1e-12, dt, 0.)
    a, b = edges.T
    failed = (ocean[a] & ocean[b] & (np.minimum(state['damage'][a], state['damage'][b]) >= damage_threshold)
              & (np.minimum(state['tensile_strain'][a], state['tensile_strain'][b]) >= rupture_strain)
              & (response['edge_extension'] > .02) & reliable)
    proposals = []
    cuts = coherent_cut(n, edges, failed, owner, area=area,
                        min_nodes=min_component_faces, min_fraction=min_fraction)
    for cut in cuts:
        if len(cut['components']) != 2:
            continue
        parent = cut['parent_nodes']
        child = min(cut['components'], key=lambda c: (float(area[c].sum()), int(c[0])))
        selected = np.isin(parent, child)
        rotations = _daughter_fit(xyz[parent], area[parent], response['velocity'][parent], selected)
        child_mask = np.zeros(n, bool); child_mask[child] = True
        broken = failed & (child_mask[a] != child_mask[b]) & np.isin(a, parent)
        if not np.any(broken):
            continue
        left, right = a[broken], b[broken]
        mid = xyz[left]+xyz[right]; mid /= np.linalg.norm(mid, axis=1)[:, None]
        normal = xyz[right]-xyz[left]; normal /= np.linalg.norm(normal, axis=1)[:, None]
        delta = rotations[child_mask[right].astype(int)]-rotations[child_mask[left].astype(int)]
        opening = np.sum(np.cross(delta, mid)*normal, axis=1)*RADIUS_KM
        cut_length = (np.asarray(mesh['edge_length'], float)[edge_indices[broken]]
                      if 'edge_length' in mesh else length[broken])
        mean_opening = float(np.average(opening, weights=cut_length))
        if mean_opening <= .01 or np.sum(cut_length[opening > .01]) < .75*cut_length.sum():
            continue
        p = cut['owner']
        proposals.append(dict(parent_slot=int(p), parent_uid=int(uids[p]),
             parent_faces=parent.copy(), child_faces=child.copy(),
             failed_edge_indices=edge_indices[broken].copy(), delta_omega=rotations,
             child_area_fraction=float(area[child].sum()/area[parent].sum()),
             child_area_km2=float(area[child].sum()), mean_opening_km_myr=mean_opening,
             cause='resolved tensile deformation across a developed ocean-material belt'))
    proposals.sort(key=lambda row: (-row['mean_opening_km_myr'], row['parent_uid'], int(row['child_faces'][0])))
    response = {**response, 'edge_indices': edge_indices}
    diagnostics = dict(model='native spherical ocean tensile-damage closure',
        material_faces=n, mechanical_edges=len(edges), ocean_faces=int(ocean.sum()),
        solver_converged=bool(response['converged']), solver_reliable=reliable,
        solver_iterations=int(response['iterations']), iteration_limit=int(cap),
        residual=float(response['relative_residual']),
        max_tensile_strain=float(state['tensile_strain'].max(initial=0.)),
        max_damage=float(state['damage'].max(initial=0.)), failed_links=int(failed.sum()),
        split_proposals=len(proposals), history_requires_material_transport=True,
        inherited_primordial_immunity=False)
    return dict(history=state, response=response, proposals=proposals, diagnostics=diagnostics)
