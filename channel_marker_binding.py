"""Read-only trace-marker selection from persistent channel source regions.

Records use native material face IDs, not transient face-array positions. Each
record must carry the face triangle at the same source epoch as its disjoint
region polygons. Motion or remeshing requires an explicit geometry/history
remap before another binding; this helper never guesses marker ancestry.
"""

import numpy as np

import dense_crust
import mesh_coverage
from channel_upper_basal_reference import inherit_reference
from channel_region_geometry import spherical_partition_intersections


def bind_marker_regions(surface, records, trace_ids, trace_patch, trace_xyz):
    """Return each trace's local copied column at one checked material epoch.

    A marker on a regional boundary may name either adjacent region only when
    their complete column states agree exactly. Otherwise the boundary's
    unresolved physical state is rejected instead of choosing a phase history
    by input order. Inputs and stored histories are never mutated.
    """
    face_ids = np.asarray(surface['face_id'])
    vertices = np.asarray(surface['vertices'])
    faces = np.asarray(surface['faces'])
    if (face_ids.ndim != 1 or face_ids.dtype.kind not in 'iu'
            or len(face_ids) != len(faces) or len(np.unique(face_ids)) != len(face_ids)
            or not isinstance(records, (list, tuple)) or not records):
        raise ValueError('Marker binding needs unique native face IDs and source records.')
    ids = np.asarray(trace_ids)
    patches = np.asarray(trace_patch)
    points = np.asarray(trace_xyz, float)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu'
            or patches.shape != ids.shape or patches.dtype.kind not in 'iu'
            or points.shape != (len(ids), 3) or not np.isfinite(points).all()
            or not np.allclose(np.linalg.norm(points, axis=1), 1., rtol=0., atol=3e-12)
            or len(np.unique(ids)) != len(ids)):
        raise ValueError('Marker binding needs unique aligned trace IDs and unit points.')
    by_face = {int(face_id): index for index, face_id in enumerate(face_ids)}
    prepared = {}
    radius = float(surface.get('radius_km', 6371.))
    for record in records:
        if not isinstance(record, dict) or set(record) != {
                'face_id', 'face_triangle', 'history', 'polygons'}:
            raise ValueError('Each source record needs its face ID, epoch triangle, history and polygons.')
        face_id = record['face_id']
        if not isinstance(face_id, (int, np.integer)) or int(face_id) not in by_face or int(face_id) in prepared:
            raise ValueError('Source records need distinct existing material face IDs.')
        index = by_face[int(face_id)]
        triangle = vertices[faces[index]]
        if not np.array_equal(np.asarray(record['face_triangle']), triangle):
            raise ValueError('Source region geometry is stale after face motion or remeshing.')
        history = record['history']
        polygons = record['polygons']
        if (not isinstance(history, dict)
                or not isinstance(history.get('face_id'), (int, np.integer))
                or int(history['face_id']) != int(face_id)
                or not isinstance(history.get('regions'), list)
                or not isinstance(polygons, (list, tuple))
                or len(polygons) != len(history['regions'])):
            raise ValueError('Source region histories must align with the current face and polygons.')
        if any(not isinstance(region, dict)
               or not isinstance(region.get('region_id'), str)
               or not region['region_id'] or 'fraction' not in region
               or 'column' not in region for region in history['regions']):
            raise ValueError('Source regions need stable IDs, fractions and columns.')
        geometry = spherical_partition_intersections(
            triangle, polygons, polygons, radius_km=radius)
        fractions = np.asarray([region['fraction'] for region in history['regions']], float)
        region_ids = [region['region_id'] for region in history['regions']]
        if (len(set(region_ids)) != len(region_ids)
                or not np.allclose(fractions, geometry['old_fractions'], rtol=0., atol=2e-10)):
            raise ValueError('Source region identities or saved area fractions disagree with geometry.')
        for region in history['regions']:
            dense_crust.validate(region['column'])
            inherit_reference(region)
            if len(region['column']['thickness_km']) != 1:
                raise ValueError('Marker source regions need one retained-phase column each.')
        prepared[int(face_id)] = list(zip(history['regions'], polygons))
    if any(int(patch) not in prepared for patch in patches):
        raise ValueError('Marker has no material-attached source region record.')

    selected = []
    for trace_id, face_id, point in zip(ids, patches, points):
        candidates = []
        for region, polygon in prepared[int(face_id)]:
            planes = mesh_coverage._triangle_planes(np.asarray(polygon, float))
            if np.all(planes @ point >= -1e-11):
                candidates.append(region)
        if not candidates:
            raise ValueError('Marker left its material-attached source regions.')
        first = candidates[0]
        if any(not _same_column(first['column'], candidate['column'])
               or inherit_reference(first) != inherit_reference(candidate)
               for candidate in candidates[1:]):
            raise ValueError('Marker lies on an unresolved boundary between distinct phase histories.')
        chosen = min(candidates, key=lambda region: region['region_id'])
        selected.append(dict(trace_id=int(trace_id), face_id=int(face_id),
                             region_id=chosen['region_id'],
                             column={key: value.copy() for key, value in chosen['column'].items()},
                             **inherit_reference(chosen)))
    return selected


def _same_column(left, right):
    return (left.keys() == right.keys() and all(
        np.asarray(left[key]).dtype == np.asarray(right[key]).dtype
        and np.array_equal(left[key], right[key]) for key in left))
