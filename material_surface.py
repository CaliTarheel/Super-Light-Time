"""Connected, material-attached spherical crust triangles under rigid motion.

Each retained face is an enduring piece of continent, craton or juvenile arc.
Neighbouring faces of one owner/region share actual indexed vertices. Faults
separate those vertices only when the neighbouring owners diverge. No raster,
point splat, mass tent or nearest-cell gap repair defines this material surface.

This first transport layer does not stretch faces, remesh, invent new crust or
choose which overlapping material is exposed. It returns every exact containing
triangle to the caller's exposure policy. Columns and provenance are per-face
arrays; changing motion ownership preserves them and their reference areas.
"""
from __future__ import annotations

import numpy as np

from ridge_geometry import rotate


SURFACE_VERSION = 1
RADIUS_KM = 6371.
_CHUNK = 65_536


def _vectors(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain finite Nx3 vectors.')
    return result


def _integers(value, count, name):
    result = np.asarray(value)
    if (result.shape != (count,) or not np.issubdtype(result.dtype, np.integer)
            or np.any(result < 0)):
        raise ValueError(f'{name} must contain one nonnegative integer per face.')
    return result.astype(np.int64, copy=True)


def spherical_face_areas(vertices, faces, radius_km=RADIUS_KM):
    """Physical areas of minor great-circle triangles, in square kilometres."""
    points = np.asarray(vertices, dtype=np.float64)
    cells = np.asarray(faces, dtype=np.int64)
    result = np.empty(len(cells), dtype=np.float64)
    for start in range(0, len(cells), _CHUNK):
        a, b, c = np.moveaxis(points[cells[start:start+_CHUNK]], 1, 0)
        # Translation of the determinant is exact algebraically. For metre-scale
        # newborn faces it avoids subtracting O(edge) products to obtain O(edge²).
        numerator = np.abs(np.einsum('ij,ij->i', a, np.cross(b-a, c-a)))
        denominator = (1.+np.einsum('ij,ij->i', a, b)
                       +np.einsum('ij,ij->i', b, c)+np.einsum('ij,ij->i', c, a))
        result[start:start+len(a)] = 2.*np.arctan2(numerator, denominator)*float(radius_km)**2
    return result


def face_centres(surface):
    """Unit-sphere face centres for sampling birth state and plate diagnostics."""
    centres = np.sum(surface['vertices'][surface['faces']], axis=1)
    return centres/np.maximum(np.linalg.norm(centres, axis=1)[:, None], 1e-30)


def _face_fields(supplied, count, retained, name, *, numeric):
    if supplied is None:
        return {}
    if not isinstance(supplied, dict):
        raise ValueError(f'{name} must be a dictionary of aligned per-face arrays.')
    result = {}
    for key, value in supplied.items():
        array = np.asarray(value)
        if (not isinstance(key, str) or not key or array.ndim < 1
                or len(array) != count or array.dtype.hasobject):
            raise ValueError(f'{name} requires named, non-object arrays aligned to all input faces.')
        if numeric and (array.ndim != 1 or array.dtype.kind not in 'biuf'):
            raise ValueError('Crustal columns must be scalar numeric per-face arrays.')
        if array.dtype.kind in 'fc' and not np.isfinite(array).all():
            raise ValueError(f'{name} contains non-finite values.')
        result[key] = array[retained].copy()
    return result


def initialize_surface(vertices, faces, face_owner, face_kind, *, face_id=None,
                       face_region=None, columns=None, provenance=None,
                       reference_area_km2=None, radius_km=RADIUS_KM):
    """Create independent checkpoint-safe state from indexed crust triangles.

    Input kind zero is ocean and is omitted; kinds 1/2/3 are retained unchanged.
    Columns/provenance and optional IDs/reference areas align with *input* faces.
    Vertices are compacted and shared within each original vertex/owner/region
    tuple. Different owners begin with coincident fault-side vertices, allowing
    exact subsequent independent rigid motion without tearing the faces.
    """
    points = _vectors(vertices, 'Material vertices')
    lengths = np.linalg.norm(points, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError('Material vertices must be nonzero spherical positions.')
    points = points/lengths[:, None]
    triangles = np.asarray(faces)
    if (triangles.ndim != 2 or triangles.shape[1] != 3
            or not np.issubdtype(triangles.dtype, np.integer)
            or np.any(triangles < 0) or np.any(triangles >= len(points))):
        raise ValueError('Material faces must contain valid integer vertex triples.')
    triangles = triangles.astype(np.int64, copy=True)
    count = len(triangles)
    owners = _integers(face_owner, count, 'Motion owners')
    kinds = _integers(face_kind, count, 'Crust kinds')
    if np.any(kinds > 3):
        raise ValueError('Crust kinds are ocean 0, continent 1, craton 2 or arc 3.')
    ids = np.arange(count, dtype=np.int64) if face_id is None else _integers(face_id, count, 'Face IDs')
    if len(np.unique(ids)) != count:
        raise ValueError('Persistent face IDs must be unique.')
    regions = np.zeros(count, np.int64) if face_region is None else _integers(face_region, count, 'Material regions')
    if not np.isscalar(radius_km) or not np.isfinite(radius_km) or radius_km <= 0:
        raise ValueError('The material sphere needs a positive finite radius in kilometres.')
    retained = np.flatnonzero(kinds > 0)
    column_state = _face_fields(columns, count, retained, 'Columns', numeric=True)
    provenance_state = _face_fields(provenance, count, retained, 'Provenance', numeric=False)
    provenance_state.setdefault('source_face_id', ids[retained].copy())
    reference = None
    if reference_area_km2 is not None:
        reference = np.asarray(reference_area_km2, dtype=np.float64)
        if reference.shape != (count,) or not np.isfinite(reference).all() or np.any(reference <= 0):
            raise ValueError('Reference areas must be positive finite values aligned to input faces.')
        reference = reference[retained].copy()
    triangles, owners, kinds, ids, regions = (value[retained] for value in (triangles, owners, kinds, ids, regions))
    if len(triangles):
        a, b, c = np.moveaxis(points[triangles], 1, 0)
        signed = np.einsum('ij,ij->i', a, np.cross(b, c))
        if np.any(np.abs(signed) < 1e-15):
            raise ValueError('Material faces must be nondegenerate spherical triangles.')
        # Keep geometry but give all faces consistent outward winding.
        reverse = signed < 0
        triangles[reverse] = triangles[reverse][:, [0, 2, 1]]
        keys = np.column_stack((triangles.ravel(), np.repeat(owners, 3), np.repeat(regions, 3)))
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        points = points[unique[:, 0]].copy()
        triangles = inverse.reshape(-1, 3).astype(np.int64)
        vertex_owner, vertex_region = unique[:, 1].copy(), unique[:, 2].copy()
    else:
        points = np.empty((0, 3), np.float64)
        vertex_owner, vertex_region = np.empty(0, np.int64), np.empty(0, np.int64)
    area = spherical_face_areas(points, triangles, radius_km)
    if np.any(area <= 0) or not np.isfinite(area).all():
        raise ValueError('Material triangles must have positive finite physical area.')
    return dict(version=SURFACE_VERSION, radius_km=float(radius_km), vertices=points,
                faces=triangles, vertex_owner=vertex_owner, vertex_region=vertex_region,
                face_owner=owners.copy(), face_kind=kinds.astype(np.uint8), face_id=ids.copy(),
                face_region=regions.copy(), area_km2=area, reference_area_km2=area.copy() if reference is None else reference,
                columns=column_state, provenance=provenance_state,
                next_face_id=int(ids.max())+1 if len(ids) else 0, geometry_revision=0)


def reassign_owners(surface, face_owner):
    """Change face owners without moving geometry or modifying material state.

    A vertex used by more than one new owner gets one copy per additional owner;
    every other vertex keeps its index and position. Existing separated faults
    are not silently welded. The caller chooses the split and protects intact
    cratons; this function does not invent or optimize a fracture path.
    """
    owners = _integers(face_owner, len(surface['faces']), 'Motion owners')
    if np.array_equal(owners, surface['face_owner']):
        return dict(duplicated_vertices=0, changed_faces=0)
    faces = surface['faces']
    if not len(faces):
        return dict(duplicated_vertices=0, changed_faces=0)
    changed = int(np.count_nonzero(owners != surface['face_owner']))
    keys = np.column_stack((faces.ravel(), np.repeat(owners, 3)))
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    original = unique[:, 0]
    first = np.r_[0, np.flatnonzero(np.diff(original))+1]
    keep = first.copy()
    preferred = np.flatnonzero(unique[:, 1] == surface['vertex_owner'][original])
    group = np.cumsum(np.r_[True, np.diff(original) != 0])-1
    keep[group[preferred]] = preferred
    duplicate = np.ones(len(unique), bool)
    duplicate[keep] = False
    copied = np.flatnonzero(duplicate)
    assigned = original.copy()
    assigned[copied] = len(surface['vertices'])+np.arange(len(copied))
    new_vertices = np.concatenate((surface['vertices'], surface['vertices'][original[copied]]), axis=0)
    new_vertex_owner = np.concatenate((surface['vertex_owner'], unique[copied, 1]))
    new_vertex_owner[original[keep]] = unique[keep, 1]
    new_vertex_region = np.concatenate((surface['vertex_region'], surface['vertex_region'][original[copied]]))
    # Commit only after every aligned array has been prepared successfully.
    surface.update(vertices=new_vertices, vertex_owner=new_vertex_owner,
                   vertex_region=new_vertex_region, faces=assigned[inverse].reshape(-1, 3),
                   face_owner=owners, geometry_revision=surface['geometry_revision']+1)
    return dict(duplicated_vertices=int(len(copied)), changed_faces=changed)


def append_surface(surface, vertices, faces, face_owner, face_kind, *, face_id=None,
                   face_region=None, columns=None, provenance=None, reference_area_km2=None):
    """Append a new connected crust patch without welding it to older material.

    Shared input indices remain shared within each new owner/region. Geometry
    is never matched by proximity against the existing surface, so unrelated
    or overlapping patches cannot acquire false connectivity. Optional column
    and provenance arrays must supply the existing per-face schema exactly.
    """
    if face_id is None:
        face_id = surface['next_face_id']+np.arange(len(faces), dtype=np.int64)
    added = initialize_surface(vertices, faces, face_owner, face_kind, face_id=face_id,
                               face_region=face_region, columns=columns, provenance=provenance,
                               reference_area_km2=reference_area_km2, radius_km=surface['radius_km'])
    if np.intersect1d(surface['face_id'], added['face_id']).size:
        raise ValueError('New material face IDs must not reuse an existing face ID.')
    combined = {}
    for group in ('columns', 'provenance'):
        if set(surface[group]) != set(added[group]):
            raise ValueError(f'Appended {group} must supply the same per-face fields.')
        combined[group] = {}
        for field, old in surface[group].items():
            new = added[group][field]
            if old.shape[1:] != new.shape[1:] or old.dtype != new.dtype:
                raise ValueError(f'Appended {group}.{field} must preserve field shape and dtype.')
            combined[group][field] = np.concatenate((old, new), axis=0)
    face_count, vertex_count = len(surface['faces']), len(surface['vertices'])
    for name in ('vertices', 'vertex_owner', 'vertex_region', 'face_owner', 'face_kind',
                 'face_id', 'face_region', 'area_km2', 'reference_area_km2'):
        combined[name] = np.concatenate((surface[name], added[name]), axis=0)
    combined['faces'] = np.concatenate((surface['faces'], added['faces']+vertex_count), axis=0)
    combined['next_face_id'] = max(surface['next_face_id'], added['next_face_id'])
    combined['geometry_revision'] = surface['geometry_revision']+1
    surface.update(combined)
    return dict(face_indices=np.arange(face_count, len(surface['faces']), dtype=np.int64),
                face_ids=added['face_id'].copy(), added_vertices=len(added['vertices']))


def refresh_geometry(surface):
    """Refresh physical areas/index revision after externally moved vertices.

    Vertices remain unit-sphere positions. This helper makes no inference about
    strain or remeshing and never changes reference areas or per-face columns.
    Degenerate, inverted or non-spherical moved triangles are rejected.
    """
    points = _vectors(surface['vertices'], 'Material vertices')
    if not np.allclose(np.linalg.norm(points, axis=1), 1., atol=1e-9, rtol=0):
        raise ValueError('Moved material vertices must remain on the unit sphere.')
    triangles = surface['faces']
    for start in range(0, len(triangles), _CHUNK):
        a, b, c = np.moveaxis(points[triangles[start:start+_CHUNK]], 1, 0)
        if np.any(np.einsum('ij,ij->i', a, np.cross(b, c)) <= 1e-15):
            raise ValueError('Moved material triangles are degenerate or inverted.')
    area = spherical_face_areas(points, triangles, surface['radius_km'])
    surface.update(area_km2=area, geometry_revision=surface['geometry_revision']+1)
    return area


def advect_surface(surface, omega_by_owner, dt):
    """Rotate each connected owner's shared vertices once by its finite stage.

    Angular velocities are Cartesian rad/Myr and dt is Myr. A row-indexed Nx3
    table or a dictionary keyed by owner ID is accepted. Signed finite dt is
    useful for exact round-trip checks. Scalar columns, IDs, provenance and
    reference areas never change during this rigid transport operation.
    """
    if not np.isscalar(dt) or not np.isfinite(dt):
        raise ValueError('Material advection needs one finite time interval.')
    owners = np.unique(surface['vertex_owner'])
    if isinstance(omega_by_owner, dict):
        if any(int(owner) not in omega_by_owner for owner in owners):
            raise ValueError('Every material owner needs an angular velocity.')
        velocities = _vectors([omega_by_owner[int(owner)] for owner in owners], 'Angular velocities') if len(owners) else np.empty((0, 3))
    else:
        table = _vectors(omega_by_owner, 'Angular velocities')
        if len(owners) and owners[-1] >= len(table):
            raise ValueError('Every material owner needs an angular velocity.')
        velocities = table[owners]
    if not len(owners) or float(dt) == 0. or not np.any(velocities):
        return
    points = surface['vertices'].copy()
    for owner, omega in zip(owners, velocities):
        chosen = np.flatnonzero(surface['vertex_owner'] == owner)
        for start in range(0, len(chosen), _CHUNK):
            indices = chosen[start:start+_CHUNK]
            moved = rotate(points[indices], omega*float(dt))
            points[indices] = moved/np.linalg.norm(moved, axis=1)[:, None]
    area = spherical_face_areas(points, surface['faces'], surface['radius_km'])
    surface.update(vertices=points, area_km2=area,
                   geometry_revision=surface['geometry_revision']+1)


def build_surface_locator(surface):
    """Build an optional reusable, plain-data index for this geometry revision."""
    from mesh_geometry import build_locator
    return dict(geometry=build_locator(surface['vertices'], surface['faces']),
                geometry_revision=surface['geometry_revision'], face_count=len(surface['faces']))


def sample_surface(surface, query_xyz, *, locator=None):
    """Return all exact containing material faces, including genuine overlaps.

    Internal edge/vertex queries can have several same-owner candidates. No
    exposure choice is made here. Uncovered points explicitly remain uncovered;
    there is no nearest-face extrapolation, raster repair or mass redistribution.
    ``weights`` are radial-projection barycentric weights for the hit triangle.
    """
    from mesh_geometry import locate_points
    points = _vectors(query_xyz, 'Query positions')
    if locator is None:
        locator = build_surface_locator(surface)
    if (locator['geometry_revision'] != surface['geometry_revision']
            or locator['face_count'] != len(surface['faces'])):
        raise ValueError('Rebuild the material locator after geometry or ownership changes.')
    query, face, weights = locate_points(points, locator['geometry'], all_hits=True)
    covered = np.zeros(len(points), bool)
    covered[query] = True
    return dict(query_index=query, face_index=face, face_id=surface['face_id'][face],
                owner=surface['face_owner'][face], kind=surface['face_kind'][face],
                weights=weights, uncovered_query_index=np.flatnonzero(~covered))
