"""Native connected spherical geometry, independent of display rasters.

All distances and areas use great-circle geometry.  The point locator uses
Cartesian spatial bins only to select candidates; triangle containment and
interpolation never depend on a longitude/latitude map.  Its plain dictionaries
and arrays are checkpoint-safe, and uncovered material remains uncovered.
"""
from __future__ import annotations

import math
import numpy as np

RADIUS_KM = 6371.0


def _unit(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Spherical coordinates must have shape (N, 3).')
    length = np.linalg.norm(points, axis=1)
    if not np.all(np.isfinite(points)) or np.any(length <= 0):
        raise ValueError('Spherical coordinates must be finite and nonzero.')
    return points / length[:, None]


def _triangles(vertices, faces):
    vertices = _unit(vertices)
    faces = np.asarray(faces)
    if faces.size == 0:
        faces = np.empty((0, 3), dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError('Faces must be integer vertex triples.')
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError('A face references a missing vertex.')
    faces = faces.astype(np.int32, copy=False)
    triangle = vertices[faces]
    if len(faces):
        determinant = np.einsum('ij,ij->i', triangle[:, 0], np.cross(triangle[:, 1], triangle[:, 2]))
        if np.any(np.abs(determinant) < 1e-14):
            raise ValueError('Spherical triangles must have nonzero area.')
        centre = _unit(triangle.sum(axis=1))
        if np.any(np.einsum('fvi,fi->fv', triangle, centre) <= 1e-10):
            raise ValueError('Each spherical triangle must fit inside an open hemisphere.')
    return vertices, faces, triangle


def spherical_area(vertices, faces, radius_km=RADIUS_KM):
    """Exact minor spherical triangle areas in square kilometres."""
    if not np.isfinite(radius_km) or radius_km <= 0:
        raise ValueError('Sphere radius must be finite and positive.')
    _, _, triangle = _triangles(vertices, faces)
    a, b, c = triangle.transpose(1, 0, 2)
    numerator = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c)))
    denominator = (1. + np.einsum('ij,ij->i', a, b)
                   + np.einsum('ij,ij->i', b, c) + np.einsum('ij,ij->i', c, a))
    return 2. * np.arctan2(numerator, denominator) * float(radius_km)**2


def geometry(vertices, faces, radius_km=RADIUS_KM, *, require_closed=True):
    """Build exact face/edge geometry and connectivity from spherical triangles.

    Face neighbors correspond to local edges (v0,v1), (v1,v2), (v2,v0).
    ``edge_faces`` is ordered by face index.  ``edge_normal`` is tangent to
    the sphere, normal to the shared edge, and points from its first face
    toward its second.  Boundary edges of an open mesh use -1 for face two.
    """
    if not np.isfinite(radius_km) or radius_km <= 0:
        raise ValueError('Sphere radius must be finite and positive.')
    vertices, faces, triangle = _triangles(vertices, faces)
    if not len(faces):
        raise ValueError('Surface geometry requires at least one face.')
    signed = np.einsum('ij,ij->i', triangle[:, 0], np.cross(triangle[:, 1], triangle[:, 2]))
    if np.any(signed <= 0):
        raise ValueError('Surface faces must be consistently outward oriented.')
    xyz = _unit(triangle.sum(axis=1))
    n, nf = len(vertices), len(faces)
    directed = np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]))
    sorted_edges = np.sort(directed, axis=1)
    keys = sorted_edges[:, 0].astype(np.int64) * n + sorted_edges[:, 1]
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    if np.any(counts > 2) or (require_closed and np.any(counts != 2)):
        raise ValueError('Surface edges must have exactly two faces on a closed manifold.')
    edge_vertices = np.column_stack((unique // n, unique % n)).astype(np.int32)
    order = np.argsort(inverse, kind='stable')
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    edge_faces = np.full((len(unique), 2), -1, dtype=np.int32)
    edge_faces[:, 0] = order[starts] % nf
    paired = counts == 2
    edge_faces[paired, 1] = order[starts[paired] + 1] % nf
    edge_faces[paired] = np.sort(edge_faces[paired], axis=1)
    # Adjacent outward triangles must traverse their common edge oppositely.
    if np.any(paired):
        first = directed[order[starts[paired]]]
        second = directed[order[starts[paired] + 1]]
        if not np.all(first == second[:, ::-1]):
            raise ValueError('Adjacent surface faces have inconsistent edge orientation.')
    face_edges = inverse.reshape(3, nf).T.astype(np.int32)
    neighbors = np.where(edge_faces[face_edges, 0] == np.arange(nf)[:, None],
                         edge_faces[face_edges, 1], edge_faces[face_edges, 0])
    a, b = vertices[edge_vertices].transpose(1, 0, 2)
    mid = _unit(a + b)
    normal = _unit(np.cross(b - a, mid))
    toward = mid - xyz[edge_faces[:, 0]]
    toward[paired] = xyz[edge_faces[paired, 1]] - xyz[edge_faces[paired, 0]]
    normal *= np.where(np.einsum('ij,ij->i', normal, toward) >= 0., 1., -1.)[:, None]
    edge_length = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1),
                            np.einsum('ij,ij->i', a, b)) * radius_km
    area = spherical_area(vertices, faces, radius_km)
    return dict(vertices=vertices, faces=faces, xyz=xyz, face_xyz=xyz,
                face_area=area, area_km2=area, edge_vertices=edge_vertices,
                edge_faces=edge_faces, edge_mid=mid, edge_normal=normal,
                edge_length=edge_length, face_neighbors=neighbors.astype(np.int32),
                face_edges=face_edges, radius_km=float(radius_km))


def icosphere(level, radius_km=RADIUS_KM):
    """Connected nearly uniform surface with 20 * 4**level triangular cells."""
    if isinstance(level, (bool, np.bool_)) or int(level) != level or not 0 <= level <= 8:
        raise ValueError('Mesh subdivision level must be an integer from 0 to 8.')
    phi = (1. + math.sqrt(5.)) / 2.
    vertices = _unit(np.array([[-1,phi,0],[1,phi,0],[-1,-phi,0],[1,-phi,0],
                              [0,-1,phi],[0,1,phi],[0,-1,-phi],[0,1,-phi],
                              [phi,0,-1],[phi,0,1],[-phi,0,-1],[-phi,0,1]], float))
    faces = np.array([[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
                      [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
                      [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
                      [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]], np.int32)
    for _ in range(int(level)):
        count, nf = len(vertices), len(faces)
        edges = np.concatenate((faces[:, (0,1)], faces[:, (1,2)], faces[:, (2,0)]))
        edges.sort(axis=1)
        keys = edges[:, 0].astype(np.int64)*count + edges[:, 1]
        unique, inverse = np.unique(keys, return_inverse=True)
        middle = _unit(vertices[unique // count] + vertices[unique % count])
        ab, bc, ca = (inverse.reshape(3, nf) + count).astype(np.int32)
        a, b, c = faces.T
        vertices = np.concatenate((vertices, middle))
        faces = np.concatenate((np.column_stack((a,ab,ca)), np.column_stack((b,bc,ab)),
                                np.column_stack((c,ca,bc)), np.column_stack((ab,bc,ca))))
    return geometry(vertices, faces, radius_km)


def connected_components(values, edge_faces, mask=None):
    """Label connected cells with equal values; excluded cells receive -1.

    A boolean ``values`` array is treated as a selection mask unless an explicit
    mask is supplied.  For integer owner fields, both owner zero and all other
    owner values participate.  Labels are deterministic in first-cell order.
    """
    values = np.asarray(values)
    edges = np.asarray(edge_faces)
    if values.ndim != 1 or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError('Components require flat cell values and edge face pairs.')
    if not np.issubdtype(edges.dtype, np.integer) or np.any(edges < -1) or np.any(edges >= len(values)):
        raise ValueError('Component edges reference invalid cells.')
    selected = (values.copy() if values.dtype == np.bool_ and mask is None
                else np.ones(len(values), bool) if mask is None else np.asarray(mask, bool))
    if selected.shape != values.shape:
        raise ValueError('Component mask must match cell values.')
    labels = np.full(len(values), -1, dtype=np.int32)
    parent = np.arange(len(values), dtype=np.int32)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    valid = np.all(edges >= 0, axis=1)
    a, b = edges[valid].T
    valid = selected[a] & selected[b] & (values[a] == values[b])
    for left, right in zip(a[valid], b[valid]):
        left, right = find(int(left)), find(int(right))
        if left != right:
            parent[max(left, right)] = min(left, right)
    cells = np.flatnonzero(selected)
    roots = np.fromiter((find(int(i)) for i in cells), np.int32, count=len(cells))
    _, labels[cells] = np.unique(roots, return_inverse=True)
    return labels, int(labels.max() + 1) if len(labels) else 0


def _locator_box_entries(lower, upper, local_faces, volumes, resolution, *, max_entries=262144):
    """Expand integer boxes in original face/x/y/z order with bounded scratch.

    The final index still contains every original entry. Only its construction
    is batched; no geometric bounds, candidate order or large-face policy change.
    A single face may exceed the scratch target, but locator-local faces never
    exceed 4096 bins. The test-only small target exercises that boundary case.
    """
    counts = volumes[local_faces]
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    keys = np.empty(int(offsets[-1]), np.int64)
    face_indexes = np.empty(len(keys), np.int32)
    start = 0
    while start < len(local_faces):
        stop = max(start+1, int(np.searchsorted(offsets, offsets[start]+max_entries, side='right'))-1)
        first, last = int(offsets[start]), int(offsets[stop])
        faces = np.repeat(local_faces[start:stop], counts[start:stop])
        ordinal = np.arange(last-first, dtype=np.int64)-np.repeat(offsets[start:stop]-first, counts[start:stop])
        ny = upper[faces, 1]-lower[faces, 1]+1
        nz = upper[faces, 2]-lower[faces, 2]+1
        # NumPy's original (x,y,z) ravel order has z varying fastest.
        x = lower[faces, 0]+ordinal//(ny*nz)
        y = lower[faces, 1]+(ordinal//nz)%ny
        z = lower[faces, 2]+ordinal%nz
        keys[first:last] = x+resolution*(y+resolution*z)
        face_indexes[first:last] = faces
        start = stop
    return keys, face_indexes


def build_locator(vertices, faces, *, bin_resolution=None):
    """Build a bounded sparse Cartesian candidate index for material triangles.

    Open, disconnected and overlapping surfaces are accepted.  Each triangle
    must fit inside an open hemisphere.  Its bounding spherical cap is enclosed
    by a conservative Cartesian box, so poles and the map seam need no special
    cases.  The returned state contains only arrays and scalars.
    """
    vertices, faces, triangle = _triangles(vertices, faces)
    nf = len(faces)
    resolution = max(2, min(512, int(round(math.sqrt(max(nf, 1) / 8.))))) if bin_resolution is None else bin_resolution
    if isinstance(resolution, (bool, np.bool_)) or int(resolution) != resolution or not 1 <= resolution <= 2048:
        raise ValueError('Locator bin resolution must be an integer from 1 to 2048.')
    resolution = int(resolution)
    centre = _unit(triangle.sum(axis=1)) if nf else np.empty((0,3))
    radius = np.max(np.linalg.norm(triangle - centre[:, None, :], axis=2), axis=1) if nf else np.empty(0)
    lower = np.floor((centre - radius[:, None] - 1e-12 + 1.) * resolution * .5).astype(np.int64)
    upper = np.floor((centre + radius[:, None] + 1e-12 + 1.) * resolution * .5).astype(np.int64)
    lower, upper = np.clip(lower, 0, resolution-1), np.clip(upper, 0, resolution-1)
    # A very distorted face at fine resolution could occupy millions of bins.
    # Put such faces in a small global candidate list instead of making a huge
    # box expansion; point queries still process that list in bounded batches.
    volumes = np.prod(upper-lower+1, axis=1)
    global_faces = np.flatnonzero(volumes > 4096).astype(np.int32)
    local_faces = np.flatnonzero(volumes <= 4096)
    keys, face_indexes = _locator_box_entries(lower, upper, local_faces, volumes, resolution)
    if len(keys):
        order = np.argsort(keys, kind='stable')
        keys, face_indexes = keys[order], face_indexes[order]
        unique, starts = np.unique(keys, return_index=True)
        offsets = np.r_[starts, len(keys)].astype(np.int64)
    else:
        unique, offsets, face_indexes = np.empty(0, np.int64), np.array([0], np.int64), np.empty(0, np.int32)
    inverse = np.linalg.inv(triangle.transpose(0, 2, 1)) if nf else np.empty((0,3,3))
    return dict(version=1, resolution=resolution, keys=unique, offsets=offsets,
                candidates=face_indexes, global_faces=global_faces,
                inverse=inverse, face_vertices=triangle, faces=faces,
                vertex_count=len(vertices), face_count=nf)


def locate_points(points, locator, *, all_hits=False, chunk_size=8192):
    """Locate spherical points exactly and return radial barycentric weights.

    Ordinary result is ``(face_index[N], weights[N,3])``; uncovered points have
    face -1 and zero weights.  At a shared edge the lowest face index wins.
    ``all_hits=True`` returns sparse ``(point_index[K], face_index[K], weights[K,3])``
    in point/face order, including both sides of shared edges and true overlaps.
    Weights describe the ray's intersection with the planar vertex triangle;
    they sum to one and reproduce normalized barycentric source positions.
    """
    points = _unit(points)
    if isinstance(chunk_size, (bool, np.bool_)) or int(chunk_size) != chunk_size or chunk_size < 1:
        raise ValueError('Locator chunk size must be a positive integer.')
    count = len(points)
    result = np.full(count, -1, dtype=np.int32)
    weights = np.zeros((count, 3), dtype=np.float64)
    hits_point, hits_face, hits_weight = [], [], []
    if not count or not int(locator['face_count']):
        if all_hits:
            return np.empty(0, np.int64), np.empty(0, np.int32), np.empty((0,3))
        return result, weights
    resolution = int(locator['resolution'])
    bins = np.clip(np.floor((points+1.)*resolution*.5).astype(np.int64), 0, resolution-1)
    keys = bins[:,0] + resolution*(bins[:,1] + resolution*bins[:,2])
    order = np.argsort(keys, kind='stable')
    unique, starts = np.unique(keys[order], return_index=True)
    ends = np.r_[starts[1:], count]
    locations = np.searchsorted(locator['keys'], unique)
    global_faces = locator['global_faces']
    for key, start, end, location in zip(unique, starts, ends, locations):
        if location < len(locator['keys']) and locator['keys'][location] == key:
            candidates = locator['candidates'][locator['offsets'][location]:locator['offsets'][location+1]]
            if len(global_faces):
                candidates = np.sort(np.r_[candidates, global_faces])
        else:
            candidates = global_faces
        if not len(candidates):
            continue
        # Bound both dimensions.  Even a deliberately coarse bin or highly
        # distorted surface cannot allocate all-points by all-faces storage.
        for face_start in range(0, len(candidates), 512):
            face_part = candidates[face_start:face_start+512]
            batch = min(int(chunk_size), max(1, 262144 // len(face_part)))
            for point_start in range(int(start), int(end), batch):
                selected = order[point_start:min(int(end), point_start+batch)]
                coefficient = np.einsum('fij,pj->pfi', locator['inverse'][face_part], points[selected])
                inside = np.all(coefficient >= -2e-11, axis=2)
                pi, fi = np.nonzero(inside)
                if not len(pi):
                    continue
                barycentric = np.maximum(coefficient[pi, fi], 0.)
                barycentric /= barycentric.sum(axis=1)[:, None]
                query, face = selected[pi], face_part[fi]
                if all_hits:
                    hits_point.append(query)
                    hits_face.append(face)
                    hits_weight.append(barycentric)
                else:
                    # nonzero visits face candidates in sorted order.  Accept
                    # only its first match for each point, then compare with
                    # any match retained from an earlier face batch.
                    _, first = np.unique(pi, return_index=True)
                    query, face, barycentric = query[first], face[first], barycentric[first]
                    improve = (result[query] < 0) | (face < result[query])
                    result[query[improve]] = face[improve]
                    weights[query[improve]] = barycentric[improve]
    if not all_hits:
        return result, weights
    if not hits_point:
        return np.empty(0, np.int64), np.empty(0, np.int32), np.empty((0,3))
    query, face, barycentric = np.concatenate(hits_point), np.concatenate(hits_face), np.concatenate(hits_weight)
    order = np.lexsort((face, query))
    return query[order], face[order], barycentric[order]


def sample_vertices(values, points, locator, *, fill_value=np.nan):
    """Interpolate vertex fields without inventing coverage across material gaps."""
    values = np.asarray(values)
    if values.ndim < 1 or len(values) != int(locator['vertex_count']):
        raise ValueError('Vertex values must align with the source mesh.')
    face, weight = locate_points(points, locator)
    output = np.full((len(face),)+values.shape[1:], fill_value, dtype=np.result_type(values, float))
    valid = face >= 0
    local = values[locator['faces'][face[valid]]]
    output[valid] = np.sum(local * weight[valid].reshape((-1,3)+(1,)*(values.ndim-1)), axis=1)
    return output
