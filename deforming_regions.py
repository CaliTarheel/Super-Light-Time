"""Connected spherical deformation around rigid continental interiors.

Version zero retains the bounded kinematic screen used by earlier histories.
Version one uses a coupled thin viscous sheet: relative Euler motion supplies
finite contact-belt targets while tensor stress transmits lateral response
through the connected body. Craton-incident vertices remain rigid constraints.

Only shared indexed vertices move. Fault-side copies remain separate, no new
force/owner/material is created, and spherical geometry determines signed
column area strain. Quality and caller-provided column bounds limit actual
motion by backtracking, rather than clipping strain after moving the surface.
"""
from __future__ import annotations

import numpy as np
import heapq

from material_forcing import sample_belts
from material_surface import spherical_face_areas
from ridge_geometry import rotate


def _material_components(points, faces, width, unloaded, craton_faces, radius):
    """Shared-edge bodies and genuine interior anchors, never width-edge islands."""
    all_edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edges, inverse, counts = np.unique(all_edges, axis=0, return_inverse=True, return_counts=True)
    edge_face = np.tile(np.arange(len(faces)), 3)
    order = np.argsort(inverse, kind='stable')
    parent = np.arange(len(faces))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for k in np.flatnonzero(inverse[order[1:]] == inverse[order[:-1]]):
        a, b = root(int(edge_face[order[k]])), root(int(edge_face[order[k+1]]))
        if a != b: parent[max(a, b)] = min(a, b)
    _, face_component = np.unique([root(i) for i in range(len(faces))], return_inverse=True)
    number = int(face_component.max(initial=-1))+1
    low, high = np.full(len(points), number), np.full(len(points), -1)
    np.minimum.at(low, faces.ravel(), np.repeat(face_component, 3))
    np.maximum.at(high, faces.ravel(), np.repeat(face_component, 3))
    vertex_component = np.where(low == high, high, -1)
    # Point-only contacts do not transmit a finite traction. Shared corner
    # vertices stay fixed instead of joining otherwise separate material sheets.
    a, b = edges.T
    length = np.arctan2(np.linalg.norm(np.cross(points[a], points[b]), axis=1), np.sum(points[a]*points[b], axis=1))*radius
    start = np.r_[a, b]; end = np.r_[b, a]; cost = np.r_[length, length]
    ordered = np.argsort(start, kind='stable'); end, cost = end[ordered], cost[ordered]
    offsets = np.r_[0, np.cumsum(np.bincount(start, minlength=len(points)))]
    boundary = np.unique(edges[counts == 1])
    distance = np.full(len(points), np.inf); distance[boundary] = 0.
    queue = [(0., int(i)) for i in boundary]; heapq.heapify(queue)
    while queue:
        value, i = heapq.heappop(queue)
        if value != distance[i] or value >= width: continue
        for at in range(offsets[i], offsets[i+1]):
            j = end[at]; candidate = value+cost[at]
            if candidate < distance[j]:
                distance[j] = candidate; heapq.heappush(queue, (candidate, int(j)))
    anchored = np.zeros(number, bool)
    interior = (vertex_component >= 0) & unloaded & (distance >= width)
    anchored[vertex_component[interior]] = True
    anchored[face_component[craton_faces]] = True
    return vertex_component, face_component, anchored, distance


def _front_intersections(triangles, first, last):
    """Positive-length intersection of a finite great-circle arc and triangles."""
    planes = _unit(np.cross(triangles, np.roll(triangles, -1, axis=1)))
    before, after = planes@first, planes@last
    change = after-before
    crossing = np.divide(-before, change, out=np.zeros_like(before), where=np.abs(change) > 1e-15)
    lower = np.maximum(0., np.max(np.where(change > 1e-15, crossing, -np.inf), axis=1))
    upper = np.minimum(1., np.min(np.where(change < -1e-15, crossing, np.inf), axis=1))
    parallel_outside = np.any((np.abs(change) <= 1e-15) & (before < -5e-14), axis=1)
    return ~parallel_outside & (upper-lower > 1e-10)


def _front_distance(point, middle, normal, length, radius):
    tangent = np.cross(normal, middle); half = length/(2*radius)
    along = np.arctan2(point@tangent, point@middle)
    if abs(along) <= half:
        return abs(np.arcsin(np.clip(point@normal, -1., 1.)))*radius
    ends = np.cos(half)*middle+np.sin(half)*np.array([tangent, -tangent])
    return float(np.min(np.arctan2(np.linalg.norm(np.cross(ends, point), axis=1), ends@point)))*radius


def _bridged_front_faces(points, faces, touching, normal):
    """Faces where a finite front cuts material, rather than following a free edge.

    ``touching`` already has positive-length intersection with the finite arc.
    A triangle crossing its great-circle plane is a bridge. When the front
    coincides with an indexed internal edge, the two incident faces together
    are the bridge. Coordinate coincidence between disconnected edge copies
    cannot supply tensile continuity. This distinction must survive refinement
    and rotating the mesh, including when the front lies on a mesh edge.
    """
    if not len(touching):
        return touching
    triangles = faces[touching]
    side = points[triangles]@normal
    tolerance = 5e-14  # angular roundoff, not a geological length threshold
    bridged = (side.min(axis=1) < -tolerance) & (side.max(axis=1) > tolerance)
    edges = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                   triangles[:, [2, 0]])), axis=1)
    unique, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    on_front = np.all(np.abs(points[unique]@normal) <= tolerance, axis=1)
    paired = np.flatnonzero((counts == 2) & on_front)
    if len(paired):
        order = np.argsort(inverse, kind='stable')
        starts = np.r_[0, np.cumsum(counts)]
        edge_face = np.tile(np.arange(len(touching)), 3)
        a = edge_face[order[starts[paired]]]
        b = edge_face[order[starts[paired]+1]]
        across = ((np.minimum(side[a].min(axis=1), side[b].min(axis=1)) < -tolerance)
                  & (np.maximum(side[a].max(axis=1), side[b].max(axis=1)) > tolerance))
        bridged[a[across]] = True
        bridged[b[across]] = True
    return touching[bridged]


def _contact_loading(surface, omega, boundaries, weight, width, radius):
    """Restrict each body/front pair to anchored margins or real buoyant contact.

    A body is anchored by cratonic material, or by an unloaded interior at
    least one deformation width from its actual free edges. It must also
    reach the particular finite front. Under opening, that front must cut
    continuous material; a separated free edge has no tensile attachment to
    the other plate and cannot be held at the mean plate velocity. An unanchored
    body needs actual different-owner triangle contact under convergence; mere trench proximity
    or a coarse collision label never qualifies. Sparse exact spherical
    geometry supplies admission, not a height/class/area threshold.
    """
    from mesh_geometry import build_locator
    from mesh_coverage import _candidates, clip_triangle
    points, faces = surface['vertices'], surface['faces']
    owners = np.asarray(surface['face_owner'])
    component, face_component, anchored, depth = _material_components(
        points, faces, width, weight <= 1e-12, np.asarray(surface['face_kind']) == 2, radius)
    count = len(boundaries.get('bmid', []))
    mid = _unit(np.asarray(boundaries.get('bmid', np.empty((0, 3)))))
    normal = _unit(np.asarray(boundaries.get('bn', np.empty((0, 3)))))
    length = np.asarray(boundaries.get('bl', np.empty(0)))
    p, q = (np.asarray(boundaries.get(k, np.empty(0, int))) for k in ('bp', 'bq'))
    valid = np.asarray(boundaries.get('valid', np.ones(count, bool)), bool)
    opening = np.sum(np.cross(omega[q]-omega[p], mid)*normal, axis=1)*radius
    active = valid & (p != q) & (length > 0) & (np.linalg.norm(omega[q]-omega[p], axis=1) > 1e-14)
    admitted, anchored_pairs, collision_pairs = set(), set(), set()
    rejected_open_pairs = set()
    candidate_faces = np.zeros(len(faces), bool)
    owner_boundaries = {}
    contact_count = 0; tested_pairs = 0
    if len(faces) and active.any():
        locator = build_locator(points, faces)
        triangles = points[faces]
        centres = _unit(triangles.sum(axis=1))
        chord = np.max(np.linalg.norm(triangles-centres[:, None], axis=2), axis=1)
        planes = _unit(np.cross(triangles, np.roll(triangles, -1, axis=1)))
        for edge in np.flatnonzero(active):
            key = tuple(sorted((int(p[edge]), int(q[edge]))))
            owner_boundaries.setdefault(key, []).append(int(edge))
            half = length[edge]/(2*radius)
            tangent = np.cross(normal[edge], mid[edge])
            first = np.cos(half)*mid[edge]+np.sin(half)*tangent
            last = np.cos(half)*mid[edge]-np.sin(half)*tangent
            cap = min(np.pi, half+width/radius)
            nearby = _candidates(None, mid[edge], 2*np.sin(cap/2), locator)
            nearby = nearby[(owners[nearby] == p[edge]) | (owners[nearby] == q[edge])]
            candidate_faces[nearby] = True
            selected = nearby[anchored[face_component[nearby]]]
            touching = selected[_front_intersections(triangles[selected], first, last)]
            if opening[edge] > 1e-10:
                bridged = _bridged_front_faces(points, faces, touching, normal[edge])
                rejected = np.setdiff1d(np.unique(face_component[touching]),
                                        np.unique(face_component[bridged]))
                rejected_open_pairs.update((int(body), int(edge)) for body in rejected)
                touching = bridged
            for body in np.unique(face_component[touching]):
                anchored_pairs.add((int(body), int(edge)))
        # Candidate bins only prune impossible pairs. Separating great-circle
        # planes and convex clipping below decide actual contact, including
        # thin crossing triangles whose vertices miss one another entirely.
        for i in np.flatnonzero(candidate_faces):
            candidates = _candidates(None, centres[i], chord[i], locator)
            candidates = candidates[(candidates > i) & (owners[candidates] != owners[i])]
            candidates = np.array([j for j in candidates if tuple(sorted((int(owners[i]), int(owners[j])))) in owner_boundaries], int)
            if not len(candidates): continue
            outside_a = np.any(np.max(np.einsum('fei,vi->fev', planes[candidates], triangles[i]), axis=2) < -5e-14, axis=1)
            outside_b = np.any(np.max(np.einsum('fvi,ei->fev', triangles[candidates], planes[i]), axis=2) < -5e-14, axis=1)
            for j in candidates[~(outside_a | outside_b)]:
                tested_pairs += 1
                polygon = clip_triangle(triangles[i], triangles[j])
                if len(polygon) < 2 or np.max(np.linalg.norm(polygon-polygon[0], axis=1), initial=0) <= 1e-9:
                    continue
                contact = _unit(polygon.sum(axis=0))
                key = tuple(sorted((int(owners[i]), int(owners[j]))))
                contact_used = False
                for edge in owner_boundaries[key]:
                    if opening[edge] >= -.02: continue
                    if _front_distance(contact, mid[edge], normal[edge], length[edge], radius) >= width: continue
                    collision_pairs.update(((int(face_component[i]), edge), (int(face_component[j]), edge)))
                    contact_used = True
                contact_count += int(contact_used)
    admitted = anchored_pairs | collision_pairs
    entries = sorted(admitted, key=lambda row: (row[1], row[0]))
    bodies = np.array([row[0] for row in entries], int)
    edges = np.array([row[1] for row in entries], int)
    result = {name: np.asarray(boundaries[name])[edges] for name in ('bmid', 'bn', 'bl', 'bp', 'bq')}
    for name in ('valid', 'admit_normal', 'admit_shear'):
        if name in boundaries: result[name] = np.asarray(boundaries[name])[edges]
    result.update(sampling_component=component, component_a=bodies, component_b=bodies, source_index=edges)
    candidate_bodies = np.unique(component[(component >= 0) & (weight > 1e-12)])
    admitted_bodies = np.unique(bodies)
    rejected = np.setdiff1d(candidate_bodies, admitted_bodies)
    diagnostics = dict(enabled=True, material_components=len(anchored), anchored_components=int(anchored.sum()),
        anchored_front_pairs=len(anchored_pairs), buoyant_contact_pairs=len(collision_pairs),
        rejected_open_free_edge_pairs=len(rejected_open_pairs),
        resolved_triangle_contacts=contact_count, tested_triangle_pairs=tested_pairs,
        rejected_uncontacted_components=len(rejected),
        rejected_detached_components=int(np.count_nonzero(~anchored[rejected])),
        rejected_uncontacted_vertices=int(np.count_nonzero(np.isin(component, rejected) & (weight > 1e-12))),
        anchor_rule='craton or unloaded interior at least one deformation width from actual free material edges',
        contact_rule='specific finite front on anchored body, or exact convergent buoyant triangle contact',
        opening_contact_rule='front must cross continuous indexed material; separated free edges carry no tensile loading')
    return result, diagnostics


def _unit(v):
    return v/np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


def finite_strain_summary(points, moved, faces, face_owner, source_boundary,
                          region_weight, boundaries, radius=6371.):
    """Observed finite material strain along/across the admitted contact normal.

    A planar-triangle metric supplies the two-dimensional Hencky tensor; this
    is a diagnostic only. Physical column volume uses exact spherical areas.
    The front with greatest admitted vertex weight supplies the local axes.
    Per-owner rows expose both sides without using the top/under ordering.
    """
    triangles, new = np.asarray(points)[faces], np.asarray(moved)[faces]
    ab, ac = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    ex = _unit(ab); normal = _unit(np.cross(ab, ac)); ey = np.cross(normal, ex)
    basis = np.stack((ex, ey), axis=2)
    edges = np.stack((ab, ac), axis=2)
    coordinates = np.einsum('fia,fib->fab', basis, edges)
    new_edges = np.stack((new[:, 1]-new[:, 0], new[:, 2]-new[:, 0]), axis=2)
    gradient = new_edges@np.linalg.inv(coordinates)
    metric = np.einsum('fia,fib->fab', gradient, gradient)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    if np.any(eigenvalues <= 0) or not np.isfinite(eigenvalues).all():
        raise ValueError('Finite material strain requires a positive deformed face metric.')
    strain = .5*(eigenvectors*np.log(eigenvalues)[:, None, :])@eigenvectors.transpose(0, 2, 1)
    strongest = np.argmax(region_weight[faces], axis=1)
    source = source_boundary[faces[np.arange(len(faces)), strongest]]
    valid = (source >= 0) & (region_weight[faces].max(axis=1) > 1e-12)
    selected = np.flatnonzero(valid)
    across = np.zeros(len(faces)); transverse = np.zeros(len(faces))
    if len(selected):
        vector = np.asarray(boundaries['bn'])[source[selected]]
        local = np.einsum('fia,fi->fa', basis[selected], vector)
        local /= np.maximum(np.linalg.norm(local, axis=1), 1e-30)[:, None]
        along = np.column_stack((-local[:, 1], local[:, 0]))
        across[selected] = np.einsum('fi,fij,fj->f', local, strain[selected], local)
        transverse[selected] = np.einsum('fi,fij,fj->f', along, strain[selected], along)
    area = spherical_face_areas(points, faces, radius)
    rows = []
    for owner in np.unique(np.asarray(face_owner)[valid]):
        mask = valid & (np.asarray(face_owner) == owner)
        total = float(area[mask].sum())
        shortening = mask & (across < -1e-12)
        loss = float(np.sum(area[shortening]*-across[shortening]))
        escape = float(np.sum(area[shortening]*np.maximum(transverse[shortening], 0.)))
        rows.append(dict(owner=int(owner), loaded_faces=int(mask.sum()), loaded_area_km2=total,
            mean_across_front_log_strain=float(area[mask]@across[mask]/total),
            mean_transverse_log_strain=float(area[mask]@transverse[mask]/total),
            shortening_area_km2=float(area[shortening].sum()),
            shortening_with_transverse_extension_area_km2=float(area[shortening & (transverse > 1e-12)].sum()),
            transverse_extension_to_shortening_ratio=escape/loss if loss > 0 else None))
    return dict(version=1, geometry='observed final planar-face Hencky metric in original admitted front axes',
        stage='prescribed rotation, accepted collision deformation and subsequent gravity if enabled',
        physical_column_strain_uses='exact spherical area ratio instead of this planar diagnostic',
        faces=len(faces), loaded_faces=int(valid.sum()), owner_rows=rows,
        maximum_principal_log_extension=float(np.max(.5*np.log(eigenvalues), initial=0.)),
        minimum_principal_log_extension=float(np.min(.5*np.log(eigenvalues), initial=0.)))


def _tangent(v, x):
    return v-x*np.sum(v*x, axis=1)[:, None]


def _edges(faces):
    return np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1), axis=0)


def _quality(points, faces):
    a, b, c = points[faces].transpose(1, 0, 2)
    signed = np.einsum('ij,ij->i', a, np.cross(b, c))
    ab, bc, ca = b-a, c-b, a-c
    scale = np.sum(ab*ab+bc*bc+ca*ca, axis=1)
    quality = np.divide(2*np.sqrt(3.)*np.linalg.norm(np.cross(ab, -ca), axis=1), scale,
                        out=np.zeros_like(scale), where=scale > 0)
    return signed, quality


def _regions(edges, moving):
    """Connected active vertices; rigid anchors separate independent belts."""
    parent = np.arange(len(moving))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges[moving[edges].all(axis=1)]:
        x, y = root(int(a)), root(int(b))
        if x != y:
            parent[max(x, y)] = min(x, y)
    labels = np.full(len(moving), -1, np.int32)
    active = np.flatnonzero(moving)
    if len(active):
        _, labels[active] = np.unique([root(int(v)) for v in active], return_inverse=True)
    return labels


def _advance_region(points, faces, axis, velocity, radius, dt, old_area, minimum, maximum,
                    old_quality, max_substeps, max_backtracks, numerical_policy=None):
    """Geometric line search for one connected belt, with rigid boundary nodes."""
    accepted = 0.; attempts = 0; backtracks = 0
    limited = np.zeros(len(faces), bool)
    moved = points.copy()
    edges = _edges(faces); a, b = edges.T
    length = np.arctan2(np.linalg.norm(np.cross(points[a], points[b]), axis=1), np.sum(points[a]*points[b], axis=1))*radius
    difference = np.linalg.norm(velocity[a]-velocity[b], axis=1)
    gradient = float(np.max(np.divide(difference, length, out=np.zeros_like(length), where=length > 0), initial=0))
    planned = max(1, min(int(max_substeps), int(np.ceil(gradient*dt/.1))))
    nominal = dt/planned
    quality_floor = np.minimum(.15, old_quality*.5)
    for attempts in range(1, int(max_substeps)+1):
        if accepted >= dt*(1-1e-12):
            attempts -= 1
            break
        step = min(nominal, dt-accepted)
        valid = False
        for _ in range(int(max_backtracks)):
            trial = _unit(rotate(points, axis*(accepted+step)))
            signed, quality = _quality(trial, faces)
            current_area = spherical_face_areas(trial, faces, radius)
            ratio = current_area/old_area
            area_bad = ((ratio < minimum-1e-12) | (ratio > maximum+1e-12) if numerical_policy is None
                else ~numerical_policy.feasible_mask(current_area,old_area*minimum,old_area*maximum))
            bad = ((signed <= 1e-15) | (quality < quality_floor) | area_bad)
            if not np.any(bad):
                valid = True
                break
            limited |= bad
            step *= .5; backtracks += 1
        if not valid or step < dt*1e-10:
            break
        accepted += step
        moved = trial
        if numerical_policy is not None:
            numerical_policy=numerical_policy.advance(current_area,old_area*minimum,old_area*maximum)
    result=(moved, accepted/dt, attempts, backtracks, limited)
    return result if numerical_policy is None else (*result,numerical_policy)



def _contact_return_stage(points, faces, owner, face_owner, omega, dt, radius,
                          rigid, rigid_vertices, region_id, face_ids=None):
    """Immutable transient data for the caller's actual contact return map."""
    if (points.ndim != 2 or points.shape[1:] != (3,) or points.dtype.kind != 'f'
            or faces.ndim != 2 or faces.shape[1:] != (3,) or faces.dtype.kind not in 'iu'
            or owner.shape != (len(points),) or owner.dtype.kind not in 'iu'
            or face_owner.shape != (len(faces),) or face_owner.dtype.kind not in 'iu'
            or rigid.shape != (len(points),) or rigid.dtype.kind != 'b'
            or rigid_vertices.shape != points.shape or region_id.shape != (len(points),)
            or region_id.dtype.kind not in 'iu' or np.any(faces < 0) or np.any(faces >= len(points))
            or not np.isfinite(points).all() or not np.isfinite(rigid_vertices).all()
            or omega.ndim != 2 or omega.shape[1:] != (3,) or not np.isfinite(omega).all()
            or np.any(owner < 0) or np.any(owner >= len(omega))
            or np.any(owner[faces] != face_owner[:, None])
            or np.any((region_id < 0) != rigid)
            or not np.isfinite((dt, radius)).all() or dt < 0 or radius <= 0):
        raise ValueError('Contact return context must preserve exact caller shape, owner, rigid and region identity.')
    # Every triangle's free corners must belong to one connected free body.
    regions = region_id[faces]
    for corner in range(3):
        for other in range(corner):
            both = (regions[:, corner] >= 0) & (regions[:, other] >= 0)
            if np.any(both & (regions[:, corner] != regions[:, other])):
                raise ValueError('Contact return context cannot split one free face between independent regions.')
    values = dict(full_points=points, full_faces=faces, full_owner=owner, full_omega=omega,
        full_face_owner=face_owner, full_rotation_vectors=omega[owner]*dt,
        full_rigid=rigid, full_rigid_vertices=rigid_vertices, full_region_id=region_id)
    if face_ids is not None:
        face_ids = np.asarray(face_ids)
        if face_ids.shape != (len(faces),) or face_ids.dtype.kind not in 'iu':
            raise ValueError('Contact return face IDs must align with this actual stage surface.')
        values['full_face_ids'] = face_ids
    stage = dict(kind='contact_final_return_geometry_context', version=1, radius=float(radius), stage_dt=float(dt))
    for name, value in values.items():
        stored = np.array(value, copy=True, order='C'); stored.setflags(write=False)
        stage[name] = stored
    return stage


def _contact_return_region(stage, vertices, selected, local_faces, region):
    """Keep global face positions and active-only assembly exact in one region."""
    vertices, selected, local_faces = (np.asarray(v) for v in (vertices, selected, local_faces))
    if (vertices.ndim != 1 or selected.ndim != 1 or vertices.dtype.kind not in 'iu'
            or selected.dtype.kind not in 'iu' or local_faces.shape != (len(selected), 3)
            or local_faces.dtype.kind not in 'iu' or not len(vertices) or not len(selected)
            or np.any(vertices < 0) or np.any(vertices >= len(stage['full_points']))
            or np.any(selected < 0) or np.any(selected >= len(stage['full_faces']))
            or len(np.unique(vertices)) != len(vertices) or len(np.unique(selected)) != len(selected)
            or np.any(local_faces < 0) or np.any(local_faces >= len(vertices))
            or not np.array_equal(vertices[local_faces], stage['full_faces'][selected])
            or not np.array_equal(vertices, np.unique(stage['full_faces'][selected]))
            or not isinstance(region, (int, np.integer)) or isinstance(region, (bool, np.bool_)) or region < 0):
        raise ValueError('Contact return region requires exact nonduplicated global/local face and vertex indices.')
    active = stage['full_region_id'][vertices] == region
    local_rigid = stage['full_rigid'][vertices]
    if not np.array_equal(~active, local_rigid) or not np.any(active):
        raise ValueError('Contact return region must contain only its free component and exact rigid anchors.')
    expected = np.flatnonzero(np.max(stage['full_region_id'][stage['full_faces']], axis=1) == region)
    if not np.array_equal(selected, expected):
        raise ValueError('Contact return region omitted or relabelled a caller face.')
    context = dict(stage, region_id=int(region))
    for name, value in dict(global_vertex_indices=vertices, global_face_indices=selected,
            local_points=stage['full_points'][vertices], local_faces=local_faces,
            active_mask=active, local_rigid=local_rigid).items():
        stored = np.array(value, copy=True, order='C'); stored.setflags(write=False)
        context[name] = stored
    return context


def _advance_return_region(points, faces, axis, velocity, radius, dt, old_area, minimum, maximum,
                           old_quality, max_substeps, max_backtracks, numerical_policy, contact_return_context):
    """Preferred path checked/charged on the exact caller return geometry."""
    from bounded_gravity import contact_return_map, validate_contact_return_context
    if numerical_policy is None or not numerical_policy.has_hard_bounds:
        raise ValueError('Mapped contact advancement requires its existing fixed hard policy.')
    if not isinstance(contact_return_context, dict):
        raise ValueError('Mapped contact advancement requires explicit immutable stage data.')
    if (not np.array_equal(points, contact_return_context.get('local_points'))
            or not np.array_equal(faces, contact_return_context.get('local_faces'))
            or radius != contact_return_context.get('radius') or dt != contact_return_context.get('stage_dt')):
        raise ValueError('Mapped preferred contact inputs do not belong to this fixed caller stage.')
    validate_contact_return_context(points, faces, contact_return_context.get('local_rigid'),
        dt, radius, contact_return_context)
    accepted = 0.; attempts = 0; backtracks = 0
    limited = np.zeros(len(faces), bool)
    moved = points.copy()
    edges = _edges(faces); a, b = edges.T
    length = np.arctan2(np.linalg.norm(np.cross(points[a], points[b]), axis=1), np.sum(points[a]*points[b], axis=1))*radius
    difference = np.linalg.norm(velocity[a]-velocity[b], axis=1)
    gradient = float(np.max(np.divide(difference, length, out=np.zeros_like(length), where=length > 0), initial=0))
    planned = max(1, min(int(max_substeps), int(np.ceil(gradient*dt/.1))))
    nominal = dt/planned
    quality_floor = np.minimum(.15, old_quality*.5)
    for attempts in range(1, int(max_substeps)+1):
        if accepted >= dt*(1-1e-12):
            attempts -= 1
            break
        step = min(nominal, dt-accepted)
        valid = False
        for _ in range(int(max_backtracks)):
            trial = _unit(rotate(points, axis*(accepted+step)))
            trial[~contact_return_context['active_mask']] = points[~contact_return_context['active_mask']]
            signed, quality = _quality(trial, faces)
            local_area = spherical_face_areas(trial, faces, radius)
            returned, current_area, _ = contact_return_map(trial, contact_return_context)
            returned_signed, returned_quality = _quality(returned, faces)
            area_bad = (~numerical_policy.feasible_mask(local_area, old_area*minimum, old_area*maximum)
                | ~numerical_policy.feasible_mask(current_area, old_area*minimum, old_area*maximum))
            bad = ((signed <= 1e-15) | (quality < quality_floor)
                | (returned_signed <= 1e-15) | (returned_quality < quality_floor) | area_bad)
            if not np.any(bad):
                valid = True
                break
            limited |= bad
            step *= .5; backtracks += 1
        if not valid or step < dt*1e-10:
            break
        accepted += step
        moved = trial
        numerical_policy = numerical_policy.advance(current_area, old_area*minimum, old_area*maximum)
    return moved, accepted/dt, attempts, backtracks, limited, numerical_policy

def _targets(points, owner, omega, boundaries, radius, width, shear_fraction):
    """Contact-side residual towards the local mean Euler motion."""
    keys = ('bmid', 'bn', 'bl', 'bp', 'bq')
    count = len(boundaries.get('bmid', []))
    if not count:
        return np.zeros((len(points), 3)), np.zeros(len(points)), np.full(len(points), -1, np.int32)
    if any(key not in boundaries for key in keys):
        raise ValueError('Deforming boundaries require bmid/bn/bl/bp/bq.')
    middle, normal = np.asarray(boundaries['bmid'], float), np.asarray(boundaries['bn'], float)
    length = np.asarray(boundaries['bl'], float)
    p, q = np.asarray(boundaries['bp']), np.asarray(boundaries['bq'])
    if (middle.shape != (count, 3) or normal.shape != middle.shape or length.shape != (count,)
            or p.shape != (count,) or q.shape != (count,) or p.dtype.kind not in 'iu' or q.dtype.kind not in 'iu'
            or np.any(p >= len(omega)) or np.any(q >= len(omega)) or np.any(p < 0) or np.any(q < 0)):
        raise ValueError('Boundary geometry and motion owners must align.')
    valid = np.asarray(boundaries.get('valid', np.ones(count, bool)), bool)
    admit_normal = np.asarray(boundaries.get('admit_normal', np.ones(count, bool)), bool)
    admit_shear = np.asarray(boundaries.get('admit_shear', np.ones(count, bool)), bool)
    if any(mask.shape != (count,) for mask in (valid, admit_normal, admit_shear)):
        raise ValueError('Boundary admission masks must align with segments.')
    valid = valid & (p != q) & (length > 0) & (np.linalg.norm(omega[q]-omega[p], axis=1) > 1e-14)
    valid &= admit_normal | (admit_shear & (shear_fraction > 0))
    selected = np.flatnonzero(valid)
    if not len(selected):
        return np.zeros((len(points), 3)), np.zeros(len(points)), np.full(len(points), -1, np.int32)
    # The helper's geographic bins are candidate acceleration only. Its exact
    # finite-arc distance and profile, not a raster value, determine the belt.
    scale = 6371./radius
    sampler_owner = np.asarray(boundaries.get('sampling_component', owner))
    sampler_a = np.asarray(boundaries.get('component_a', p))
    sampler_b = np.asarray(boundaries.get('component_b', q))
    belts = dict(mid=middle[selected], normal=normal[selected], length_km=length[selected]*scale,
                 owner_a=sampler_a[selected], owner_b=sampler_b[selected], values=np.ones((len(selected), 1)))
    weight, source = sample_belts(points, sampler_owner, belts, width_km=width*scale, return_sources=True)
    weight, source = weight[:, 0], source[:, 0]
    target = np.zeros((len(points), 3))
    present = np.flatnonzero(source >= 0)
    source[present] = selected[source[present]]
    if len(present):
        edge = source[present]
        other = np.where(owner[present] == p[edge], q[edge], p[edge])
        relative = np.cross(omega[other]-omega[owner[present]], points[present])*radius*.5
        local_normal = _unit(_tangent(normal[edge], points[present]))
        across = local_normal*np.sum(relative*local_normal, axis=1)[:, None]
        along = relative-across
        target[present] = (across*admit_normal[edge, None]+along*(admit_shear[edge]*shear_fraction)[:, None])*weight[present, None]
    if 'source_index' in boundaries:
        source[present] = np.asarray(boundaries['source_index'])[source[present]]
    return target, weight, source


def _smooth(points, faces, area, edges, target, rigid, driven, smoothing_length, iterations, tolerance):
    """Sparse SPD screened tangent graph with exact prescribed vertices."""
    count = len(points)
    free = ~(rigid | driven)
    fixed = np.where(driven[:, None], target, 0.)
    data = np.bincount(faces.ravel(), weights=np.repeat(area/3, 3), minlength=count)
    a, b = edges.T
    cosine = np.sum(points[a]*points[b], axis=1)
    length = np.arctan2(np.linalg.norm(np.cross(points[a], points[b]), axis=1), cosine)
    radius = np.sqrt(area.sum()/max(float(spherical_face_areas(points, faces, 1.).sum()), 1e-300))
    length *= radius
    # A contour intersection may put two vertices arbitrarily close while
    # their incident control areas remain finite. Using area/edge_length²
    # alone makes that unresolved tiny edge infinitely stiff and destroys
    # the screen's conditioning. The local dual-area length regularizes only
    # that sub-control-area scale; the physical smoothing length is unchanged.
    local_area = data[a]+data[b]
    weight = smoothing_length**2*local_area/(2*np.maximum(length*length+.005*local_area, 1e-24))
    diagonal = data+np.bincount(np.r_[a, b], weights=np.r_[weight, weight], minlength=count)
    connection = points[a]+points[b]
    denominator = np.maximum(1+cosine, 1e-12)

    def apply(value):
        # Parallel transport along each minor great-circle edge is orthogonal
        # between tangent planes, giving a symmetric positive graph energy.
        b_to_a = value[b]-connection*(np.sum(value[b]*points[a], axis=1)/denominator)[:, None]
        a_to_b = value[a]-connection*(np.sum(value[a]*points[b], axis=1)/denominator)[:, None]
        delta_a = (value[a]-b_to_a)*weight[:, None]
        delta_b = (value[b]-a_to_b)*weight[:, None]
        result = data[:, None]*value
        for column in range(3):
            result[:, column] += np.bincount(a, weights=delta_a[:, column], minlength=count)
            result[:, column] += np.bincount(b, weights=delta_b[:, column], minlength=count)
        return _tangent(result, points)

    if not free.any():
        return fixed, dict(converged=True, iterations=0, residual=0.)
    rhs = np.where(free[:, None], data[:, None]*target-apply(fixed), 0.)
    norm = float(np.linalg.norm(rhs))
    if norm <= 1e-25:
        return fixed, dict(converged=True, iterations=0, residual=0.)
    solution = np.where(free[:, None], target, 0.)
    residual = rhs-np.where(free[:, None], apply(solution), 0.)
    z = np.divide(residual, diagonal[:, None], out=np.zeros_like(residual), where=diagonal[:, None] > 0)
    direction = z.copy()
    rz = float(np.sum(residual*z))
    relative = float(np.linalg.norm(residual)/norm)
    used = 0
    for used in range(1, int(iterations)+1):
        if relative <= tolerance:
            used -= 1
            break
        action = np.where(free[:, None], apply(direction), 0.)
        denominator_cg = float(np.sum(direction*action))
        if denominator_cg <= 0 or not np.isfinite(denominator_cg):
            break
        alpha = rz/denominator_cg
        solution += alpha*direction
        residual -= alpha*action
        relative = float(np.linalg.norm(residual)/norm)
        z = np.divide(residual, diagonal[:, None], out=np.zeros_like(residual), where=diagonal[:, None] > 0)
        next_rz = float(np.sum(residual*z))
        direction = z+(next_rz/max(rz, 1e-300))*direction
        rz = next_rz
    converged = relative <= tolerance and np.isfinite(solution).all()
    # An unresolved numerical response cannot supply unverified deformation.
    result = _tangent(solution+fixed, points) if converged else np.zeros_like(target)
    result[rigid] = 0.
    result[driven] = target[driven] if converged else 0.
    return result, dict(converged=bool(converged), iterations=used, residual=relative)


class IncompleteContactStepError(ValueError):
    """A complete coupled step needs a shorter timestep or a resolved solver.

    No geometry has been committed when deform raises this exception. The
    caller must reject the WHOLE outer step, including its rigid rotations;
    dropping only the unaccepted residual contact motion changes the model.
    """


def _advance_final_contact_policy(policy, area, minimum, maximum, dt):
    """Reject only a valid contact proposal through the existing retry path.

    Final owner rotation/normalization is measured by the original admission
    predicate. This routing does not accept, repair or refund that proposal,
    and does not assert that a smaller coupled timestep will succeed. Invalid
    input/current policy and nonfinite/nonpositive geometry keep their original
    fatal ValueError. This helper is used only after the final policy constructor.
    """
    try:
        return policy.advance(area, minimum, maximum)
    except ValueError as original_error:
        if str(original_error) != 'Accepted geometry exhausted its persistent dimensional area budget.':
            raise
        proposed, nominal_minimum, nominal_maximum = policy._values(area, minimum, maximum)
        policy.validate(len(policy.reference_area_km2))
        if (not np.isfinite(proposed).all() or np.any(proposed <= 0)
                or not np.isfinite(nominal_minimum).all() or np.any(nominal_minimum <= 0)
                or not np.isfinite(nominal_maximum).all() or np.any(nominal_maximum < nominal_minimum)
                or not np.array_equal(nominal_minimum, policy.minimum_area_km2)
                or not np.array_equal(nominal_maximum, policy.maximum_area_km2)):
            raise
        admitted = policy.feasible_mask(proposed, nominal_minimum, nominal_maximum)
        if np.all(admitted):
            raise
        from numerical_accuracy import nominal_miss
        miss = nominal_miss(proposed, nominal_minimum, nominal_maximum)
        baseline = policy.baseline_miss_km2
        allowance = policy.allowance_km2
        increment = np.maximum(miss-baseline, 0.)
        predicted_spent = policy.spent_km2+increment
        if not np.isfinite(predicted_spent).all():
            raise
        nominal_admitted = miss <= allowance
        hard_admitted = policy.hard_feasible_mask(proposed)
        failure = IncompleteContactStepError(
            'Final returned contact geometry failed its unchanged dimensional area/strain admission; '
            'the complete coupled timestep must be rejected.')
        failure.diagnostics = dict(
            kind='final_contact_proposal_admission_rejection', version=1,
            stage='post_owner_rotation_and_rigid_restoration', dt_myr=float(dt),
            face_index_scope='all input deform contact-stage faces, before gravity or material adaptation',
            failed_faces=np.flatnonzero(~admitted).tolist(),
            nominal_failed_faces=np.flatnonzero(~nominal_admitted).tolist(),
            hard_failed_faces=np.flatnonzero(~hard_admitted).tolist(),
            final_area_km2=proposed.tolist(),
            nominal_minimum_area_km2=nominal_minimum.tolist(),
            nominal_maximum_area_km2=nominal_maximum.tolist(),
            current_area_km2=policy.current_area_km2.tolist(),
            reference_area_km2=policy.reference_area_km2.tolist(),
            baseline_nominal_miss_km2=baseline.tolist(),
            permitted_nominal_miss_km2=allowance.tolist(), nominal_miss_km2=miss.tolist(),
            previous_spent_km2=policy.spent_km2.tolist(), allocated_budget_km2=policy.budget_km2.tolist(),
            proposed_increment_km2=increment.tolist(), predicted_spent_km2=predicted_spent.tolist(),
            predicted_spent_within_budget_mask=(predicted_spent <= policy.budget_km2).tolist(),
            original_feasible_mask=admitted.tolist(), nominal_admission_mask=nominal_admitted.tolist(),
            hard_strain_admission_mask=hard_admitted.tolist(),
            hard_minimum_area_km2=None if not policy.has_hard_bounds else policy.hard_minimum_area_km2.tolist(),
            hard_maximum_area_km2=None if not policy.has_hard_bounds else policy.hard_maximum_area_km2.tolist(),
            hard_arithmetic_area_km2=None if not policy.has_hard_bounds else policy.hard_arithmetic_area_km2.tolist(),
            hard_stage_area_km2=None if not policy.has_hard_bounds else policy.hard_stage_area_km2.tolist(),
            reference_identity=policy.reference_identity,
            final_policy_validated=True, original_nominal_bounds_aligned=True,
            finite_positive_proposed_area=True, original_admission_criteria_unchanged=True,
            proposal_accepted=False, geometry_modified=False, credit_or_spent_modified=False,
            whole_coupled_rollback_required=True, smaller_timestep_success_claimed=False,
            original_exception_type=type(original_error).__name__, original_exception_message=str(original_error))
        raise failure from original_error


def deform(surface, omega, boundaries, dt, *, belt_width_km=400., shear_fraction=1.,
           smoothing_length_km=100., max_residual_speed_km_myr=100.,
           max_areal_strain_per_myr=.08, iterations=256, tolerance=1e-7,
           max_substeps=64, max_backtracks=16,
           face_min_area_ratio=None, face_max_area_ratio=None,
           require_material_contact=False, vertex_protected=None, vertex_response=None,
           mechanics_version=0, viscosity_weights=None, craton_rigid_vertices=None,
           redistribute_limited_contact=False, require_complete_contact=False, numerical_policy=None):
    """Return changed geometry and actual signed area strain, without mutation.

    omega is a slot-indexed rad/Myr table, dt Myr and residual speeds km/Myr.
    The full rigid Euler stage always advances exactly. Residual motion follows
    a frozen tangent direction in that rotating material frame; quality-limited
    accepted motion is reported explicitly. ``actual_vertex_velocity_km_myr``
    is the realized finite Cartesian chord rate, not an instantaneous tangent.
    Face area bounds are ratios to this step's original spherical areas and
    must include one. Columns should use the returned geometric ratio once.
    ``require_material_contact`` enables the runtime body/contact gate; the
    default also supports explicit kinematic fixtures that prescribe loading.
    Optional authored regional constraints fix vertices rigidly, or scale the
    admitted residual response from zero to 1.5. They do not add plate motion
    or remove cratonic constraints. Omitted controls preserve the baseline.
    ``require_complete_contact`` rejects the entire result if any contact
    region cannot cover dt; callers must roll back the coupled outer step.
    """
    points = np.asarray(surface['vertices'], float)
    if isinstance(mechanics_version, (bool, np.bool_)) or mechanics_version not in (0, 1):
        raise ValueError('Unsupported material mechanics version.')
    faces = np.asarray(surface['faces'])
    owner = np.asarray(surface['vertex_owner'])
    face_owner, kind = np.asarray(surface['face_owner']), np.asarray(surface['face_kind'])
    omega = np.asarray(omega, float)
    radius = float(surface.get('radius_km', 6371.))
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or np.any(np.abs(np.linalg.norm(points, axis=1)-1) > 1e-8)
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu'
            or np.any(faces < 0) or np.any(faces >= len(points))
            or owner.shape != (len(points),) or owner.dtype.kind not in 'iu'
            or face_owner.shape != (len(faces),) or kind.shape != (len(faces),)
            or omega.ndim != 2 or omega.shape[1] != 3 or not np.isfinite(omega).all()
            or np.any(owner < 0) or np.any(owner >= len(omega))
            or np.any(owner[faces] != face_owner[:, None])):
        raise ValueError('Deformation needs unit shared vertices and consistent separated motion owners.')
    controls = (dt, radius, belt_width_km, shear_fraction, smoothing_length_km,
                max_residual_speed_km_myr, max_areal_strain_per_myr, tolerance)
    if (not np.isfinite(controls).all() or dt < 0 or radius <= 0 or not 0 < belt_width_km < np.pi*radius
            or not 0 <= shear_fraction <= 1 or smoothing_length_km < 0 or max_residual_speed_km_myr <= 0
            or max_areal_strain_per_myr <= 0 or tolerance <= 0
            or any(isinstance(v, bool) or int(v) != v or v < 1 for v in (iterations, max_substeps, max_backtracks))):
        raise ValueError('Deformation controls must be finite, positive and bounded.')
    old_area = spherical_face_areas(points, faces, radius)
    old_signed, old_quality = _quality(points, faces)
    if numerical_policy is not None:
        from numerical_accuracy import NumericalPolicy
        if not isinstance(numerical_policy,NumericalPolicy) or mechanics_version!=1:raise ValueError('Numerical accuracy requires native material mechanics1.')
        numerical_policy.validate(len(faces));tolerance=numerical_policy.outer_tolerance
        if not np.allclose(old_area,numerical_policy.current_area_km2,rtol=2e-12,atol=0.):raise ValueError('Contact policy is not aligned with original material geometry.')
        accuracy_spent=numerical_policy.spent_km2.copy();accuracy_area=old_area.copy()
    if np.any(old_signed <= 1e-15) or np.any(old_area <= 0):
        raise ValueError('Deformation cannot start from degenerate or inverted material faces.')
    minimum = np.full(len(faces), .25) if face_min_area_ratio is None else np.asarray(face_min_area_ratio, float)
    maximum = np.full(len(faces), 4.) if face_max_area_ratio is None else np.asarray(face_max_area_ratio, float)
    if (minimum.shape != (len(faces),) or maximum.shape != (len(faces),) or not np.isfinite(minimum).all()
            or not np.isfinite(maximum).all() or np.any(minimum <= 0) or np.any(maximum<minimum)
            or (numerical_policy is None and (np.any(minimum > 1) or np.any(maximum < 1)))):
        raise ValueError('Each positive finite face area interval must include its current ratio one.')
    if numerical_policy is None:
        minimum = np.maximum(minimum, np.exp(-max_areal_strain_per_myr*dt))
        maximum = np.minimum(maximum, np.exp(max_areal_strain_per_myr*dt))
    else:
        # Nominal volume/column bounds stay unchanged for persistent spending.
        # Strain is an independent full-stage geometric restriction; apply the
        # existing ratio guard's +/-1e-12 arithmetic allowance exactly once.
        numerical_policy=numerical_policy.with_hard_bounds(
            old_area*np.exp(-max_areal_strain_per_myr*dt),
            old_area*np.exp(max_areal_strain_per_myr*dt),arithmetic_area=old_area*1e-12)
        numerical_policy.effective_bounds(old_area*minimum,old_area*maximum)
        if not numerical_policy.feasible(old_area,old_area*minimum,old_area*maximum):
            raise ValueError('Physical contact area/strain bounds conflict with the existing dimensional admission budget.')
    rigid_vertices = _unit(rotate(points, omega[owner]*dt))
    craton = np.zeros(len(points), bool)
    craton[np.unique(faces[kind == 2])] = True
    craton_fixed = craton.copy()
    if craton_rigid_vertices is not None:
        craton_fixed = np.asarray(craton_rigid_vertices)
        if (craton_fixed.shape != craton.shape or craton_fixed.dtype.kind != 'b'
                or np.any(craton_fixed & ~craton)):
            raise ValueError('Craton anchors must be a boolean subset of cratonic vertices.')
    target, weight, source = _targets(points, owner, omega, boundaries, radius, belt_width_km, shear_fraction)
    contact_diagnostics = dict(enabled=False)
    if require_material_contact:
        loading, contact_diagnostics = _contact_loading(surface, omega, boundaries, weight, belt_width_km, radius)
        target, weight, source = _targets(points, owner, omega, loading, radius, belt_width_km, shear_fraction)
    incident = np.bincount(faces.ravel(), minlength=len(points)) > 0
    # Version one transmits tensor stress through the connected body. The
    # finite belt supplies loading; its artificial outer contour is not a
    # clamped wall against transverse spreading. Basal drag supplies the
    # screened return to rigid plate velocity away from a contact.
    rigid = craton_fixed | ~incident
    if mechanics_version and require_material_contact:
        # Shared corner copies of different edge-connected bodies have zero
        # contact measure. Their common node must not transmit a point force.
        rigid |= np.asarray(loading['sampling_component']) < 0
    if not mechanics_version:
        rigid |= weight <= 1e-12
    if vertex_protected is not None:
        protected = np.asarray(vertex_protected)
        if protected.shape != (len(points),) or protected.dtype.kind != 'b':
            raise ValueError('Regional protection must be a boolean mask aligned with material vertices.')
        rigid |= protected
    if vertex_response is not None:
        response = np.asarray(vertex_response, float)
        if (response.shape != (len(points),) or not np.isfinite(response).all()
                or np.any(response < 0) or np.any(response > 1.5)):
            raise ValueError('Regional response must align with vertices and lie from zero to 1.5.')
        target *= response[:, None]
        rigid |= response == 0.
    target[rigid] = 0.
    driven = (weight >= .9999) & ~rigid
    edges = _edges(faces)
    if mechanics_version == 1:
        from viscous_sheet import solve
        normals = np.zeros_like(points)
        present = source >= 0
        if np.any(present):
            normals[present] = _unit(_tangent(np.asarray(boundaries['bn'])[source[present]], points[present]))
        velocity, solver = solve(points, faces, old_area, target, rigid, driven,
            smoothing_length_km, radius=radius, driven_normals=normals,
            iterations=iterations, tolerance=tolerance, viscosity_weights=viscosity_weights)
    else:
        velocity, solver = _smooth(points, faces, old_area, edges, target, rigid, driven,
                                   smoothing_length_km, iterations, tolerance)
    # A maximum principle bounds this screen in ideal arithmetic. Retain an
    # explicit numerical/parameter safety cap, never increase a target speed.
    cap = float(max_residual_speed_km_myr)
    if not mechanics_version:
        cap = min(cap, float(np.max(np.linalg.norm(target, axis=1), initial=0)))
    speed = np.linalg.norm(velocity, axis=1)
    if mechanics_version:
        # A single scale preserves the coupled flow direction; separate node
        # clipping would manufacture strain and spoil a virtual-work balance.
        speed_scale = min(1., cap/max(float(speed.max(initial=0.)), 1e-300))
        velocity *= speed_scale
        solver['velocity_safety_scale'] = speed_scale
    else:
        velocity *= np.minimum(1., np.divide(cap, speed, out=np.ones_like(speed), where=speed > 0))[:, None]
    commanded = velocity.copy()
    if dt > 0 and require_complete_contact and not solver['converged']:
        raise IncompleteContactStepError('Contact response failed its stationarity gate; the complete coupled timestep must be rejected.')
    axis = np.cross(points, velocity)/radius
    limited_faces = np.zeros(len(faces), bool)
    moved_local = points.copy()
    region_id = _regions(edges, ~rigid)
    face_region = np.max(region_id[faces], axis=1)
    fractions = np.zeros(len(points))
    region_rows = []
    contact_return_stage = None
    returned_area_witness = np.zeros(len(faces), bool)
    if numerical_policy is not None and numerical_policy.has_hard_bounds:
        contact_return_stage = _contact_return_stage(points, faces, owner, face_owner, omega, dt, radius,
            rigid, rigid_vertices, region_id, surface.get('face_id', surface.get('face_ids')))
    if dt > 0 and cap > 0 and solver['converged']:
        for region in np.unique(region_id[region_id >= 0]):
            selected = np.flatnonzero(face_region == region)
            vertices, inverse = np.unique(faces[selected], return_inverse=True)
            local_faces = inverse.reshape(-1, 3)
            region_policy=None if numerical_policy is None else numerical_policy.subset(selected)
            contact_return_context = (None if contact_return_stage is None else
                _contact_return_region(contact_return_stage, vertices, selected, local_faces, region))
            advance = _advance_region if contact_return_context is None else _advance_return_region
            advance_result = advance(
                points[vertices], local_faces, axis[vertices], velocity[vertices], radius, dt,
                old_area[selected], minimum[selected], maximum[selected], old_quality[selected],
                max_substeps, max_backtracks, **({} if region_policy is None else dict(numerical_policy=region_policy)),
                **({} if contact_return_context is None else dict(contact_return_context=contact_return_context)))
            if region_policy is None:trial,fraction,attempts,backtracks,limited=advance_result
            else:trial,fraction,attempts,backtracks,limited,accepted_region_policy=advance_result
            accommodation = None
            if redistribute_limited_contact and mechanics_version and fraction < 1.-1e-10:
                # Preserve the complete body's coupled response. Do not clip
                # individual vertex velocities or erase a saturated column.
                endpoint = _unit(rotate(points[vertices],axis[vertices]*dt))
                returned_quality_limited = False
                if contact_return_context is not None:
                    from bounded_gravity import contact_return_map
                    endpoint[~contact_return_context['active_mask']] = points[vertices][~contact_return_context['active_mask']]
                    returned_endpoint, endpoint_area, _ = contact_return_map(endpoint, contact_return_context)
                    returned_signed, returned_quality = _quality(returned_endpoint, local_faces)
                    returned_quality_limited = bool(np.any(returned_signed <= 1e-15)
                        or np.any(returned_quality < np.minimum(.15, old_quality[selected]*.5)))
                    local_endpoint_area = spherical_face_areas(endpoint, local_faces, radius)
                    local_area_limited = not region_policy.feasible(local_endpoint_area,
                        old_area[selected]*minimum[selected], old_area[selected]*maximum[selected])
                else:
                    endpoint_area = spherical_face_areas(endpoint,local_faces,radius)
                    local_area_limited = False
                area_limited = (np.any((endpoint_area < old_area[selected]*minimum[selected])
                    | (endpoint_area > old_area[selected]*maximum[selected])) if region_policy is None
                    else not region_policy.feasible(endpoint_area,old_area[selected]*minimum[selected],old_area[selected]*maximum[selected]))
                if area_limited or local_area_limited or returned_quality_limited or (region_policy is not None and np.any(_quality(endpoint,local_faces)[1]<np.minimum(.15,old_quality[selected]*.5))):
                    from contact_response import redistribute
                    from bounded_gravity import ConstraintSolveError
                    accommodation = dict(attempted=True,accepted=False)
                    try:
                        candidate, response_velocity, response_solver, detail = redistribute(
                            points[vertices],local_faces,old_area[selected],velocity[vertices],rigid[vertices],
                            old_area[selected]*minimum[selected],old_area[selected]*maximum[selected],dt,
                            radius=radius,length_km=smoothing_length_km,
                            viscosity_weights=None if viscosity_weights is None else np.broadcast_to(np.asarray(viscosity_weights),(len(faces),))[selected],
                            iterations=iterations,tolerance=tolerance,max_speed_km_myr=cap,
                            **({} if region_policy is None else dict(numerical_policy=region_policy)),
                            **({} if contact_return_context is None else dict(contact_return_context=contact_return_context)))
                        returned_geometry_valid = True
                        if contact_return_context is not None:
                            candidate[~contact_return_context['active_mask']] = points[vertices][~contact_return_context['active_mask']]
                            returned_candidate, candidate_area, _ = contact_return_map(candidate, contact_return_context)
                            returned_signed, returned_quality = _quality(returned_candidate, local_faces)
                            local_candidate_area = spherical_face_areas(candidate, local_faces, radius)
                            returned_geometry_valid = bool(np.all(returned_signed > 1e-15)
                                and np.all(returned_quality >= np.minimum(.15, old_quality[selected]*.5))
                                and region_policy.feasible(local_candidate_area,
                                    old_area[selected]*minimum[selected], old_area[selected]*maximum[selected]))
                        else:
                            candidate_area=spherical_face_areas(candidate,local_faces,radius)
                        signed, quality = _quality(candidate,local_faces)
                        area_valid=(np.all(candidate_area>=old_area[selected]*minimum[selected]*(1-1e-12))
                            and np.all(candidate_area<=old_area[selected]*maximum[selected]*(1+1e-12)) if region_policy is None
                            else region_policy.feasible(candidate_area,old_area[selected]*minimum[selected],old_area[selected]*maximum[selected]))
                        valid=(np.all(signed>1e-15) and np.all(quality>=np.minimum(.15,old_quality[selected]*.5)) and area_valid and returned_geometry_valid)
                        accommodation.update(detail,solver=response_solver)
                        if valid:
                            trial=candidate; fraction=1.
                            velocity[vertices]=response_velocity
                            accommodation['accepted']=True
                            if region_policy is not None:
                                # Complete redistributed motion replaces the partial preferred trial;
                                # discarded tentative path spending cannot leak into accepted state.
                                accepted_region_policy=region_policy.advance(candidate_area,old_area[selected]*minimum[selected],old_area[selected]*maximum[selected])
                        else:
                            accommodation['reason']='Endpoint failed unchanged orientation, quality or column bounds.'
                    except ConstraintSolveError as error:
                        accommodation['reason']=str(error)
                        if hasattr(error,'diagnostics'):accommodation['failed_solver_diagnostics']=error.diagnostics
            if require_complete_contact and fraction < 1.-1e-10:
                reason=(accommodation or {}).get('reason', 'Unchanged orientation or quality bounds limit the preferred motion.')
                failure=IncompleteContactStepError(
                    f'Contact region {int(region)} accepted only {fraction:.9g} of the {dt:.9g} Myr interval; '
                    f'the complete coupled timestep must be rejected. {reason}')
                if numerical_policy is not None:failure.diagnostics=dict(region=int(region),accepted_fraction=float(fraction),dt_myr=float(dt),contact_accommodation=accommodation)
                raise failure
            if region_policy is not None:
                accuracy_spent[selected]=accepted_region_policy.spent_km2
                accuracy_area[selected]=accepted_region_policy.current_area_km2
                if contact_return_context is not None:
                    returned_area_witness[selected] = fraction > 0.
            active = region_id[vertices] == region
            moved_local[vertices[active]] = trial[active]
            fractions[vertices[active]] = fraction
            limited_faces[selected] |= limited
            region_rows.append(dict(id=int(region), vertices=int(active.sum()), faces=len(selected),
                                    accepted_fraction=float(fraction), substeps=attempts,
                                    backtracks=backtracks, limited_faces=int(limited.sum())))
            if accommodation is not None:
                region_rows[-1]['contact_accommodation']=accommodation
    new = _unit(rotate(moved_local, omega[owner]*dt))
    new[rigid] = rigid_vertices[rigid]
    new_area = spherical_face_areas(new, faces, radius)
    if contact_return_stage is not None and not np.array_equal(new_area[returned_area_witness], accuracy_area[returned_area_witness]):
        raise ValueError('Exact full-layout returned contact areas differ from accepted regional witnesses; context or assembly is inconsistent.')
    ratio = np.divide(new_area, old_area, out=np.ones_like(old_area), where=old_area > 0)
    rigid_faces = np.all(rigid[faces], axis=1)
    unchanged_faces = np.all((rigid | (fractions == 0))[faces], axis=1)
    if numerical_policy is None:
        ratio[unchanged_faces] = 1.  # Legacy rigid/rejected stages have no physical strain.
    # The declared accuracy path records the actual measured spherical areas,
    # including rigid-rotation arithmetic roundoff. Exact V/A columns use this
    # same ratio; forcing it to one would disagree with the recorded geometry.
    strain = np.log(ratio)
    actual = (new-points)*radius/dt if dt > 0 else np.zeros_like(points)
    result = dict(vertices=new, rigid_vertices=rigid_vertices, face_area_km2=new_area,
                area_ratio=ratio, areal_strain=strain, face_areal_strain=strain,
                face_weight=weight[faces].mean(axis=1), face_rigid=rigid_faces, face_strain=strain,
                residual_velocity_km_myr=velocity*fractions[:, None], commanded_residual_velocity_km_myr=commanded,
                actual_vertex_velocity_km_myr=actual, rigid_mask=rigid, craton_mask=craton,
                region_weight=weight, source_boundary=source, boundary_target_mask=driven,
                region_id=region_id, accepted_residual_fraction=fractions, limited_face_mask=limited_faces,
                diagnostics=dict(model=('boundary-driven thin viscous sheet' if mechanics_version else 'boundary-driven screened material deformation'),
                                 mechanics_version=mechanics_version, solver=solver,
                                 rigid_plate_motion_fraction=1.,
                                 material_contact=contact_diagnostics,
                                 vertices=len(points), graph_edges=len(edges), rigid_vertices=int(rigid.sum()),
                                 craton_vertices=int(craton.sum()), deforming_vertices=int(np.count_nonzero(~rigid)),
                                 rigid_craton_vertices=int(craton_fixed.sum()),
                                 regions=region_rows, substeps=max([0, *[row['substeps'] for row in region_rows]]),
                                 total_substeps=sum(row['substeps'] for row in region_rows),
                                 backtracks=sum(row['backtracks'] for row in region_rows),
                                 accepted_residual_fraction=float(np.sum(fractions*speed)/max(float(speed.sum()), 1e-300)),
                                 limited_faces=int(limited_faces.sum()), max_residual_speed_km_myr=float(np.max(np.linalg.norm(velocity, axis=1)*fractions, initial=0)),
                                 redistributed_contact_regions=sum(bool(row.get('contact_accommodation',{}).get('accepted')) for row in region_rows),
                                 accepted_fraction_scope='elapsed contact interval; redistributed velocity may differ from preferred velocity',
                                 minimum_area_ratio=float(ratio.min(initial=1)), maximum_area_ratio=float(ratio.max(initial=1)),
                                 strain_measure='log of actual spherical face area ratio'))

    if numerical_policy is not None:
        final_policy=NumericalPolicy(numerical_policy.reference_area_km2,numerical_policy.budget_km2,accuracy_spent,accuracy_area,
            old_area*minimum,old_area*maximum,reference_identity=numerical_policy.reference_identity,
            **numerical_policy._hard_keywords())
        accepted_policy=_advance_final_contact_policy(final_policy,new_area,old_area*minimum,old_area*maximum,dt)
        result['diagnostics']['numerical_accuracy_policy']=accepted_policy.evidence(new_area,old_area*minimum,old_area*maximum)
        result['diagnostics']['numerical_accuracy_final_spent_km2']=accepted_policy.spent_km2.tolist()
    return result
