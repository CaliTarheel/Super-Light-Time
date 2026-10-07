"""Exact material-chart correspondence across spherical deformation/remeshing.

Reference corners are homogeneous barycentric coordinates in an immutable
birth-triangle chart. Their row magnitudes matter: a spherical midpoint with
parent weights B inherits (B @ Hparent)/norm(B @ XYZparent). Ordinary normalized
barycentric corners would move material merely because a face was subdivided.
"""
from __future__ import annotations

import numpy as np

FIELDS = ('material_root_id', 'material_reference_corners', 'material_erosion_total_m')


def validate_fields(frame):
    count = len(frame['material_faces'])
    if any(name not in frame for name in FIELDS):
        raise ValueError('Deforming material history lacks its reference chart or cumulative erosion fields.')
    roots = np.asarray(frame['material_root_id'])
    corners = np.asarray(frame['material_reference_corners'], float)
    erosion = np.asarray(frame['material_erosion_total_m'], float)
    if (roots.shape != (count,) or not np.issubdtype(roots.dtype, np.integer)
            or np.any(roots < 0) or corners.shape != (count, 3, 3)
            or not np.isfinite(corners).all() or np.any(corners < -1e-12)
            or erosion.shape != (count,) or not np.isfinite(erosion).all()):
        raise ValueError('Invalid material reference chart or cumulative net erosion.')
    magnitude = np.linalg.norm(corners, axis=2)
    if np.any(magnitude <= 0):
        raise ValueError('Material reference corners must have nonzero homogeneous coordinates.')
    normalized = corners/magnitude[:, :, None]
    if np.any(np.abs(np.linalg.det(normalized)) < 1e-14):
        raise ValueError('Material reference triangles must remain invertible.')
    return roots, corners, erosion


def prepare(frame):
    """Prepare sparse, deterministic leaf searches within each birth root."""
    roots, corners, erosion = validate_fields(frame)
    ids = np.asarray(frame['material_face_id'])
    order = np.lexsort((ids, roots))
    unique, starts, counts = np.unique(roots[order], return_index=True, return_counts=True)
    by_root = {int(root): order[start:start+count] for root, start, count in zip(unique, starts, counts)}
    return dict(roots=roots, corners=corners, inverse=np.linalg.inv(corners), by_root=by_root,
                vertices=np.asarray(frame['material_vertices'], float), faces=np.asarray(frame['material_faces']),
                face_ids=ids, erosion_total_m=erosion)


def map_hits(first, second, source_faces, source_weights):
    """Map actual source hits into destination descendants, never nearest-fill.

The search is restricted to the same material root. Shared-edge ties choose
the lowest persistent destination face ID. An absent root/leaf is an explicit
error rather than guessed Euler motion or an unrelated overlapping sheet.
"""
    source_faces = np.asarray(source_faces)
    weights = np.asarray(source_weights, float)
    if (source_faces.ndim != 1 or not np.issubdtype(source_faces.dtype, np.integer)
            or np.any(source_faces < 0) or np.any(source_faces >= len(first['roots']))
            or weights.shape != (len(source_faces), 3) or not np.isfinite(weights).all()
            or np.any(weights < -1e-10)
            or not np.allclose(weights.sum(axis=1), 1., atol=1e-9, rtol=0.)):
        raise ValueError('Material transport requires actual source triangle hits.')
    roots = first['roots'][source_faces]
    references = np.einsum('ni,nij->nj', weights, first['corners'][source_faces])
    references /= references.sum(axis=1)[:, None]
    destination = np.full(len(source_faces), -1, np.int64)
    local = np.zeros((len(source_faces), 3))
    order = np.argsort(roots, kind='stable')
    unique, starts, counts = np.unique(roots[order], return_index=True, return_counts=True)
    for root, start, count in zip(unique, starts, counts):
        candidates = second['by_root'].get(int(root))
        if candidates is None:
            raise ValueError(f'Destination history is missing material root {int(root)}.')
        indices = order[start:start+count]
        for offset in range(0, len(indices), 65536):
            remaining = indices[offset:offset+65536]
            for face in candidates:
                if not len(remaining):
                    break
                bary = references[remaining]@second['inverse'][face]
                scale = np.abs(bary).sum(axis=1)
                inside = (bary.min(axis=1) >= -1e-10*np.maximum(scale, 1e-30)) & (bary.sum(axis=1) > 0)
                selected = remaining[inside]
                positive = np.maximum(bary[inside], 0.)
                destination[selected] = face
                local[selected] = positive/positive.sum(axis=1)[:, None]
                remaining = remaining[~inside]
            if len(remaining):
                raise ValueError(f'Destination material root {int(root)} has a missing reference-chart region.')
    xyz = np.einsum('ni,nij->nj', local, second['vertices'][second['faces'][destination]])
    xyz /= np.linalg.norm(xyz, axis=1)[:, None]
    return dict(xyz=xyz, face=destination, weights=local)
