"""Read-only, fixed-geography spherical transects through committed native history.

This measures sampled sections, not connected water bodies or geological basin
identities. The caller loads a committed manifest and selected saved frames; no
directory scanning, live API, engine mutation, or global flux inference occurs.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers

import numpy as np


VERSION = 1
BOUNDARY_TYPES = {1: 'ridge', 2: 'trench'}


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f'{name} must be an integer.')
    return int(value)


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f'{name} must be finite.')
    return float(value)


def _angle(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a*b, axis=-1))


def _xyz(coordinates):
    lon, lat = np.radians(coordinates).T
    return np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))


def _lon_lat(points):
    return np.degrees(np.column_stack((np.arctan2(points[:, 1], points[:, 0]),
                                      np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1)))))


def _transect(spec, spacing, radius):
    name = spec.get('id')
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Each transect requires a nonempty id.')
    coordinates = np.asarray(spec.get('waypoints_lon_lat'), float)
    if (coordinates.ndim != 2 or coordinates.shape[1:] != (2,) or len(coordinates) < 2
            or not np.isfinite(coordinates).all() or np.any(np.abs(coordinates[:, 0]) > 180)
            or np.any(np.abs(coordinates[:, 1]) > 90)):
        raise ValueError('Transect waypoints must be finite [longitude, latitude] degrees within globe bounds.')
    vertices = _xyz(coordinates)
    points, distances, legs = [], [], []
    offset = 0.
    for a, b in zip(vertices[:-1], vertices[1:]):
        angle = float(_angle(a, b))
        if angle < 1e-10 or angle > np.pi-1e-10:
            raise ValueError('Coincident or antipodal transect endpoints do not define a unique minor arc.')
        length = radius*angle
        count = int(math.ceil(length/spacing))
        fractions = np.linspace(0., 1., count+1)
        values = (np.sin((1-fractions)*angle)[:, None]*a
                  + np.sin(fractions*angle)[:, None]*b)/np.sin(angle)
        values /= np.linalg.norm(values, axis=1)[:, None]
        keep = slice(None) if not points else slice(1, None)
        points.extend(values[keep])
        distances.extend((offset+fractions*length)[keep])
        legs.append((a, b, offset, angle))
        offset += length
    return dict(id=name, waypoints_lon_lat=coordinates.tolist(), points=np.asarray(points),
                distance=np.asarray(distances), legs=legs, length_km=offset)


def _commit_selection(frames, manifest):
    """Validate membership in one explicit committed generation, never disk tails."""
    count = _integer(manifest.get('frame_count'), 'Committed frame_count')
    entries = manifest.get('frames')
    if count < 1 or not isinstance(entries, list) or len(entries) < count:
        raise ValueError('Committed manifest requires a complete indexed frame prefix.')
    entries = entries[:count]
    if not isinstance(manifest.get('run_id'), str) or not manifest['run_id']:
        raise ValueError('Committed manifest requires a run_id.')
    previous = -math.inf
    for index, entry in enumerate(entries):
        if _integer(entry.get('index'), 'Committed index') != index:
            raise ValueError('Committed frame prefix must be contiguous and indexed from zero.')
        epoch = _finite(entry.get('time_myr'), 'Committed epoch')
        if epoch < 0 or epoch <= previous:
            raise ValueError('Committed epochs must be nonnegative and strictly increasing.')
        previous = epoch
    frontier = manifest.get('time_myr')
    if frontier is not None and previous > _finite(frontier, 'Committed frontier')+1e-6:
        raise ValueError('Frame prefix extends beyond the committed time frontier.')
    selected = set()
    for frame in frames:
        index = _integer(frame.get('index'), 'Selected frame index')
        if index < 0 or index >= count or index in selected:
            raise ValueError('Selected frames must be distinct members of the committed prefix.')
        if _finite(frame.get('time_myr'), 'Selected epoch') != entries[index]['time_myr']:
            raise ValueError('Selected frame epoch does not match the committed prefix.')
        if frame.get('run_id', manifest['run_id']) != manifest['run_id']:
            raise ValueError('Selected frame belongs to a different run.')
        for key in ('engine_sha256', 'source_engine_sha256'):
            if frame.get(key) is not None and manifest.get('engine_sha256') is not None:
                if frame[key] != manifest['engine_sha256']:
                    raise ValueError('Selected frame engine provenance does not match the committed run.')
        selected.add(index)
    if not selected:
        raise ValueError('At least one committed frame must be selected.')


def _values(sampled, name, count, *, categorical=False):
    value = sampled.get(name)
    if value is None:
        return None
    values = np.asarray(value, float)
    if values.shape != (count,):
        raise ValueError(f'Sampled {name} must align with the transect points.')
    if categorical and (not np.isfinite(values).all() or np.any(values != np.floor(values))):
        raise ValueError(f'Sampled {name} must contain finite integer categories.')
    return values


def _json_numbers(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def _age_values(frame, sampled, count, crust):
    # Current physical-stack sampling returns the selected local ocean age. An
    # older frame can expose only its coarser control-cell age; label that limit.
    age = _values(sampled, 'selected_ocean_age_myr', count)
    source = 'selected_local_ocean_age_myr'
    if age is None:
        age = _values(sampled, 'ocean_age_myr', count)
        source = 'provided_ocean_age_myr'
    if age is None and sampled.get('native_face') is not None and frame.get('mesh_age_myr') is not None:
        cells = np.asarray(sampled['native_face'])
        grid_age = np.asarray(frame['mesh_age_myr'], float)
        if (cells.shape != (count,) or cells.dtype.kind not in 'iu' or grid_age.ndim != 1
                or np.any(cells < 0) or np.any(cells >= len(grid_age))):
            raise ValueError('Saved control-cell ocean age indices are invalid.')
        age, source = grid_age[cells], 'saved_control_cell_age_myr'
    if age is None or crust is None:
        return np.full(count, np.nan), 'unavailable'
    age = age.copy()
    age[(crust != 0) | ~np.isfinite(age) | (age < 0)] = np.nan
    return age, source


def _intervals(mask, distance, age):
    """Midpoint-classified runs with endpoint brackets, not sub-grid precision."""
    selected = np.flatnonzero(mask)
    if not len(selected):
        return []
    groups = np.split(selected, np.flatnonzero(np.diff(selected) != 1)+1)
    result = []
    for group in groups:
        first, last = int(group[0]), int(group[-1])
        left = [float(distance[max(first-1, 0)]), float(distance[first])]
        right = [float(distance[last]), float(distance[min(last+1, len(distance)-1)])]
        start, end = sum(left)/2, sum(right)/2
        values = age[group]
        valid = values[np.isfinite(values)]
        result.append(dict(start_km=start, end_km=end, sampled_width_km=end-start,
            start_transition_bracket_km=left, end_transition_bracket_km=right,
            width_uncertainty_km=(left[1]-left[0]+right[1]-right[0])/2,
            truncated_at_start=first == 0, truncated_at_end=last == len(distance)-1,
            sample_count=len(group), ocean_age_sample_count=len(valid),
            ocean_age_min_myr=float(valid.min()) if len(valid) else None,
            ocean_age_mean_myr=float(valid.mean()) if len(valid) else None,
            ocean_age_max_myr=float(valid.max()) if len(valid) else None))
    return result


def _boundary_geometry(frame):
    rows = frame.get('boundary_segments')
    if rows is None:
        return None
    if not isinstance(rows, list):
        raise ValueError('Saved boundary_segments must be a list when available.')
    arcs = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get('code') is None:
            return None  # Incomplete classification is not evidence of no ridge/trench.
        code = row.get('code')
        if code not in BOUNDARY_TYPES:
            continue
        if row.get('geometry_xyz') is None:
            return None
        vertices = np.asarray(row.get('geometry_xyz'), float)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 2 or not np.isfinite(vertices).all():
            raise ValueError('Ridge/trench boundary geometry is missing or malformed.')
        norms = np.linalg.norm(vertices, axis=1)
        if np.any(norms < 1e-12):
            raise ValueError('Boundary vertices must have nonzero spherical positions.')
        vertices = vertices/norms[:, None]
        for a, b in zip(vertices[:-1], vertices[1:]):
            angle = float(_angle(a, b))
            if angle >= np.pi-1e-10:
                raise ValueError('Antipodal boundary endpoints are ambiguous.')
            if angle < 1e-12:
                continue  # Repeated endpoints have no finite segment to cross.
            arcs.append((a, b, angle, index, int(code)))
    return arcs


def _crossings(transect, arcs, radius):
    if arcs is None:
        return dict(status='unavailable', intersections=None, collinear_overlaps=None)
    intersections, overlaps = [], []
    if not arcs:
        return dict(status='available', intersections=[], collinear_overlaps=[])
    aa, bb = np.array([a[0] for a in arcs]), np.array([a[1] for a in arcs])
    sizes = np.array([a[2] for a in arcs])
    normals = np.cross(aa, bb)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    for a, b, offset, angle in transect['legs']:
        normal = np.cross(a, b)
        normal /= np.linalg.norm(normal)
        axes = np.cross(normal, normals)
        lengths = np.linalg.norm(axes, axis=1)
        regular = np.flatnonzero(lengths > 1e-10)
        for sign in (-1., 1.):
            points = sign*axes[regular]/lengths[regular, None]
            along = _angle(a, points)
            on_path = np.abs(along+_angle(points, b)-angle) <= 1e-9
            on_boundary = np.abs(_angle(aa[regular], points)+_angle(points, bb[regular])-sizes[regular]) <= 1e-9
            for local in np.flatnonzero(on_path & on_boundary):
                source = int(regular[local])
                intersections.append(dict(distance_km=float(offset+along[local]*radius),
                    longitude_latitude_deg=_lon_lat(points[local:local+1])[0].tolist(),
                    type=BOUNDARY_TYPES[arcs[source][4]], boundary_segment_indices=[arcs[source][3]]))
        for source in np.flatnonzero(lengths <= 1e-10):
            candidates = [point for point in (a, b, aa[source], bb[source])
                if abs(float(_angle(a, point)+_angle(point, b))-angle) <= 1e-9
                and abs(float(_angle(aa[source], point)+_angle(point, bb[source]))-sizes[source]) <= 1e-9]
            if candidates:
                positions = [offset+float(_angle(a, point))*radius for point in candidates]
                overlaps.append(dict(start_km=min(positions), end_km=max(positions),
                    type=BOUNDARY_TYPES[arcs[source][4]], boundary_segment_index=arcs[source][3]))
    # Adjacent saved segments and adjacent transect legs can share a vertex.
    # Preserve all source references while counting one intersection per type.
    merged = []
    for row in sorted(intersections, key=lambda x: (x['distance_km'], x['type'])):
        match = next((old for old in reversed(merged[-8:]) if old['type'] == row['type']
                      and abs(old['distance_km']-row['distance_km']) < 1e-5), None)
        if match is None:
            merged.append(row)
        else:
            match['boundary_segment_indices'] = sorted(set(match['boundary_segment_indices']+row['boundary_segment_indices']))
    return dict(status='available', intersections=merged, collinear_overlaps=overlaps)


def review_frames(frames, transects, *, committed_manifest, sample_spacing_km=25.,
                  radius_km=6371., sampler=None):
    """Return a JSON-ready report for explicit fixed-geography transects.

    ``frames`` are selected flat dictionaries of saved JSON metadata and NPZ
    arrays. ``committed_manifest`` must be the one generation chosen by the
    caller's read-only loader, not a guessed prefix from files on disk. Optional
    frame hashes are copied as provenance. ``transects`` contain ``id`` and
    ``waypoints_lon_lat`` (degrees). An injectable ``sampler(frame, points)`` is
    available for independent analytic fixtures; production defaults to native
    saved-surface sampling, with one prepared locator per selected frame.
    """
    frames = list(frames)
    _commit_selection(frames, committed_manifest)
    spacing, radius = _finite(sample_spacing_km, 'Sample spacing'), _finite(radius_km, 'Radius')
    if spacing <= 0 or radius <= 0:
        raise ValueError('Sample spacing and radius must be positive.')
    paths = [_transect(spec, spacing, radius) for spec in transects]
    if not paths or len(set(path['id'] for path in paths)) != len(paths):
        raise ValueError('At least one uniquely identified transect is required.')
    points = np.concatenate([path['points'] for path in paths])
    definition = dict(radius_km=radius, sample_spacing_km=spacing, coordinate_frame='fixed_source_world',
        transects=[dict(id=path['id'], waypoints_lon_lat=path['waypoints_lon_lat']) for path in paths])
    definition_hash = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    epochs = []
    for frame in sorted(frames, key=lambda f: f['index']):
        if sampler is None:
            import native_frame_sampling
            prepared = native_frame_sampling.prepare(frame)
            sampled = native_frame_sampling.sample_frame(frame, points, prepared)
            del prepared
        else:
            sampled = sampler(frame, points)
        elevation = _values(sampled, 'elevation', len(points))
        if elevation is None or not np.isfinite(elevation).all():
            raise ValueError('Finite saved-surface elevation is required to measure a water gap.')
        crust = _values(sampled, 'crust', len(points), categorical=True)
        age, age_source = _age_values(frame, sampled, len(points), crust)
        arcs = _boundary_geometry(frame)
        results, offset = [], 0
        for path in paths:
            count = len(path['points'])
            part = slice(offset, offset+count)
            height, values = elevation[part], age[part]
            kinds = None if crust is None else crust[part]
            water = height < 0.
            oceanic = None if kinds is None else water & (kinds == 0)
            results.append(dict(id=path['id'], length_km=path['length_km'], sample_count=count,
                maximum_spacing_km=float(np.diff(path['distance']).max()),
                samples=dict(distance_km=path['distance'].tolist(), longitude_latitude_deg=_lon_lat(path['points']).tolist(),
                    elevation_m=height.tolist(), crust_kind=None if kinds is None else kinds.astype(int).tolist(),
                    ocean_age_myr=_json_numbers(values)), ocean_age_source=age_source,
                surface_water_intervals=_intervals(water, path['distance'], values),
                submerged_oceanic_crust_intervals=None if oceanic is None else _intervals(oceanic, path['distance'], values),
                boundaries=_crossings(path, arcs, radius)))
            offset += count
        epochs.append(dict(index=int(frame['index']), time_myr=float(frame['time_myr']),
            source_frame_sha256=frame.get('source_frame_sha256'),
            source_metadata_sha256=frame.get('source_metadata_sha256'), transects=results))
    source = dict(run_id=committed_manifest['run_id'], committed_frame_count=int(committed_manifest['frame_count']),
        committed_last_index=int(committed_manifest['frame_count'])-1,
        committed_last_epoch_myr=float(committed_manifest['frames'][int(committed_manifest['frame_count'])-1]['time_myr']),
        engine_sha256=committed_manifest.get('engine_sha256'),
        auxiliary_sources_sha256=committed_manifest.get('auxiliary_sources_sha256'))
    return dict(version=VERSION, source=source, definition=definition, definition_sha256=definition_hash,
        sampling_method='native_saved_surface' if sampler is None else 'injected_surface_sampler', epochs=epochs,
        limitations=[
            'Transects stay in source-world coordinates; endpoints do not follow plates or material.',
            'Water means elevation below the saved zero datum. Submerged oceanic crust additionally requires crust kind zero.',
            'Intervals are sampled sections, not geological basin identities or proof of ocean-wide gateway connectivity.',
            'Uncertainty assumes one transition between neighboring samples; narrower islands or gaps can be missed.',
            'Intersections include endpoint touches of saved minor-arc boundaries, not necessarily a change of plate; collinear contact is reported separately.',
            'Age summaries use valid oceanic samples with equal weights along the selected section, not basin-area means.',
            'Sparse epochs can miss intervening opening, closure, reversal or boundary events.',
            'No local consumption rate is inferred from global created or consumed totals.',
            'The caller certifies the committed manifest and file hashes; this function validates selection membership without accessing disk.'])
