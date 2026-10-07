"""Read-only, sparse mountain profiles and exact saved material-marker histories.

No simulation, nearest-face history matching, temporal interpolation or inferred
orogen identity is used. Native sampling is a reconstruction of saved geometry,
not additional physical resolution. Callers own committed-frame verification.
"""
from __future__ import annotations

import math
import numpy as np

RADIUS_KM = 6371.
MAX_NATIVE_POINTS_PER_EPOCH = 128
COUNTERS = ('uplift_m', 'extension_m', 'erosion_m', 'adjustment_m',
            'denudation_m', 'rebound_m', 'thermal_subsidence_m',
            'foreland_subsidence_m', 'ridge_uplift_m',
            'rift_extension_m', 'inversion_uplift_m')
FACE_FIELDS = ('root_id', 'parent_id', 'collision_sheet',
               'crustal_thickness_km', 'crustal_root_km',
               'rift_thermal_support_m', 'foreland_deflection_m',
               'erosion_total_m', 'erosion_rate_m_myr')
SAMPLE_FIELDS = ('raw_elevation_m', 'selected_sheet_height_m',
                 'physical_stack_support_m', 'display_thermal_support_m',
                 'thermal_support_m', 'clipping_delta_m',
                 'collision_surface_offset_m')


def _finite_number(value, label):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f'{label} must be a finite number.')
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{label} must be a finite number.') from None
    if not math.isfinite(result):
        raise ValueError(f'{label} must be a finite number.')
    return result


def _unit(point):
    value = np.asarray(point, float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError('Transect endpoints must be finite unit XYZ vectors.')
    norm = np.linalg.norm(value)
    if abs(norm - 1.) > 1e-5:
        raise ValueError('Transect endpoints must be unit XYZ vectors.')
    return value / norm


def transect_points(start_xyz, end_xyz, samples=63):
    """Unit XYZ and km along the minor great circle; no map-projection shortcut."""
    if type(samples) is not int or not 3 <= samples <= MAX_NATIVE_POINTS_PER_EPOCH:
        raise ValueError('Transect samples must be an integer from 3 to 128.')
    start, end = _unit(start_xyz), _unit(end_xyz)
    angle = float(np.arctan2(np.linalg.norm(np.cross(start, end)), np.dot(start, end)))
    if angle < 1e-10 or math.pi-angle < 1e-8:
        raise ValueError('Transect endpoints must be distinct and non-antipodal.')
    fraction = np.linspace(0., 1., samples)
    points = (np.sin((1-fraction)*angle)[:, None]*start
              + np.sin(fraction*angle)[:, None]*end) / math.sin(angle)
    points /= np.linalg.norm(points, axis=1)[:, None]
    return points, fraction*angle*RADIUS_KM


def _component(distance, height, peak, threshold):
    if height[peak] < threshold:
        return dict(width_km=0., left_km=None, right_km=None,
                    censored_left=False, censored_right=False), None
    left = right = peak
    while left > 0 and height[left-1] >= threshold:
        left -= 1
    while right < len(height)-1 and height[right+1] >= threshold:
        right += 1
    lo, hi = float(distance[left]), float(distance[right])
    if left > 0:
        lo = float(distance[left-1] + (distance[left]-distance[left-1])
                   * (threshold-height[left-1])/(height[left]-height[left-1]))
    if right < len(height)-1:
        hi = float(distance[right] + (distance[right+1]-distance[right])
                   * (height[right]-threshold)/(height[right]-height[right+1]))
    return dict(width_km=hi-lo, left_km=lo, right_km=hi,
                censored_left=left == 0, censored_right=right == len(height)-1), (left, right)


def profile_metrics(distance_km, elevation_m, *, highland_threshold_m=2000., plateau_drop_m=1000.):
    """Spatially interpolate threshold crossings, never unseen temporal history.

    Widths concern the connected sampled component containing the highest sample.
    A summit cap is the component within plateau_drop_m of that sample. A broad
    cap alone does not establish a geological plateau. Endpoint contact censors
    the corresponding width; the observed span is then only a lower bound.
    """
    distance, height = np.asarray(distance_km, float), np.asarray(elevation_m, float)
    if (distance.ndim != 1 or len(distance) < 3 or height.shape != distance.shape
            or not np.all(np.isfinite(distance)) or not np.all(np.isfinite(height))
            or np.any(np.diff(distance) <= 0)):
        raise ValueError('Profiles require at least three finite, strictly ordered distance/elevation pairs.')
    threshold = _finite_number(highland_threshold_m, 'Highland threshold')
    drop = _finite_number(plateau_drop_m, 'Summit cap drop')
    if drop <= 0:
        raise ValueError('Summit cap drop must be positive.')
    peak = int(np.argmax(height))  # first maximum is deterministic for tied peaks
    high, bounds = _component(distance, height, peak, threshold)
    # Do not turn an ocean-only profile into a putative mountain summit cap.
    cap, _ = _component(distance, height, peak, height[peak]-drop) if bounds else (
        dict(width_km=0., left_km=None, right_km=None, censored_left=False, censored_right=False), None)
    left_low = float(height[:bounds[0]].min()) if bounds and bounds[0] > 0 else None
    right_low = float(height[bounds[1]+1:].min()) if bounds and bounds[1] < len(height)-1 else None
    return dict(sampled_peak_m=float(height[peak]), peak_distance_km=float(distance[peak]),
                sampled_min_m=float(height.min()), sampled_relief_m=float(np.ptp(height)),
                highland_width_km=high['width_km'], summit_cap_width_km=cap['width_km'],
                highland_left_km=high['left_km'], highland_right_km=high['right_km'],
                summit_cap_left_km=cap['left_km'], summit_cap_right_km=cap['right_km'],
                highland_censored_left=high['censored_left'], highland_censored_right=high['censored_right'],
                cap_censored_left=cap['censored_left'], cap_censored_right=cap['censored_right'],
                left_adjacent_low_m=left_low, right_adjacent_low_m=right_low,
                peak_above_left_low_m=None if left_low is None else float(height[peak]-left_low),
                peak_above_right_low_m=None if right_low is None else float(height[peak]-right_low),
                max_sample_spacing_km=float(np.diff(distance).max()))


def _optional_scalar(frame, key, index, length, *, integer=False):
    if key not in frame:
        return None
    values = np.asarray(frame[key])
    if values.shape != (length,):
        raise ValueError(f'{key} must have one scalar per identity.')
    value = values[index]
    if not np.isfinite(value):
        return None
    if integer:
        if float(value) != int(value):
            raise ValueError(f'{key} must contain integral identities.')
        return int(value)
    return float(value)


def _ids(frame, key):
    if key not in frame:
        return np.empty(0, np.int64)
    values = np.asarray(frame[key])
    if values.ndim != 1 or values.dtype.kind not in 'iu' or len(np.unique(values)) != len(values):
        raise ValueError(f'{key} must contain unique integral identities.')
    return values


def _marker_histories(frames, marker_ids):
    histories = []
    lookups = []
    for frame in frames:
        ids = _ids(frame, 'trace_id')
        lookups.append((len(ids), {int(value): i for i, value in enumerate(ids)}))
    for trace_id in marker_ids:
        rows, previous, last_gain = [], None, None
        for frame, (count, lookup) in zip(frames, lookups):
            time = float(frame['time_myr'])
            index = lookup.get(trace_id)
            row = dict(time_myr=time, present=index is not None, xyz=None,
                       longitude_deg=None, latitude_deg=None, plate_uid=None,
                       patch_id=None, marker_age_myr=None, representative_relief_m=None,
                       counters_m={key: None for key in COUNTERS},
                       delta_counters_m={key: None for key in COUNTERS},
                       interval_myr=None, activity='unavailable',
                       structure_version=frame.get('structure_version'),
                       current_components_m={key: None for key in
                           ('rift_thermal_support_m', 'ridge_thermal_m', 'foreland_deflection_m')})
            if index is None:
                rows.append(row)
                previous = None  # absence is not an interpolated trajectory
                continue
            if 'trace_xyz' in frame:
                xyz = np.asarray(frame['trace_xyz'], float)
                if xyz.shape != (count, 3):
                    raise ValueError('trace_xyz must have one XYZ point per marker.')
                point = _unit(xyz[index])
                row.update(xyz=point.tolist(),
                           longitude_deg=float(np.degrees(np.arctan2(point[1], point[0]))),
                           latitude_deg=float(np.degrees(np.arcsin(np.clip(point[2], -1., 1.)))))
            for output, field in (('plate_uid', 'trace_plate_uid'), ('patch_id', 'trace_patch')):
                row[output] = _optional_scalar(frame, field, index, count, integer=True)
            birth = _optional_scalar(frame, 'trace_birth_myr', index, count)
            if birth is not None and 0 <= birth <= time:
                row['marker_age_myr'] = time-birth
            row['representative_relief_m'] = _optional_scalar(frame, 'trace_relief_m', index, count)
            for key in COUNTERS:
                row['counters_m'][key] = _optional_scalar(frame, 'trace_'+key, index, count)
            for key in row['current_components_m']:
                row['current_components_m'][key] = _optional_scalar(frame, 'trace_'+key, index, count)
            if previous is not None:
                row['interval_myr'] = [previous['time_myr'], time]
                for key in COUNTERS:
                    old, new = previous['counters_m'][key], row['counters_m'][key]
                    if old is not None and new is not None:
                        row['delta_counters_m'][key] = new-old
                uplift = row['delta_counters_m']['uplift_m']
                if uplift is not None:
                    if uplift > 1e-6:
                        row['activity'] = 'positive_gain_recorded_in_interval'
                        last_gain = list(row['interval_myr'])
                    elif uplift < -1e-6:
                        row['activity'] = 'counter_decrease_requires_review'
                    else:
                        extension, erosion = (row['delta_counters_m'][key] for key in ('extension_m', 'erosion_m'))
                        if any(value is not None and value < -1e-6 for value in (extension, erosion)):
                            row['activity'] = 'signed_lowering_counter_decrease_requires_review'
                        elif any(value is not None and value > 1e-6 for value in (extension, erosion)):
                            row['activity'] = 'lowering_recorded_without_positive_gain'
                        elif extension is not None and erosion is not None:
                            row['activity'] = 'no_recorded_gain_or_lowering'
                        else:
                            row['activity'] = 'no_recorded_positive_gain_other_counters_unavailable'
            rows.append(row)
            previous = row
        histories.append(dict(trace_id=trace_id, epochs=rows,
                              last_observed_positive_gain_interval_myr=last_gain,
                              cessation_time_myr=None))
    return histories


def _sample_values(sampled, key, total):
    if key not in sampled:
        return [None]*total
    values = np.asarray(sampled[key], float)
    if values.shape != (total,):
        raise ValueError(f'Native sample field {key} must have one scalar per point.')
    return [float(value) if np.isfinite(value) else None for value in values]


def review_frames(frames, *, transects=(), marker_ids=(), sampler=None):
    """Return JSON-ready review of chronological saved metadata+NPZ dictionaries.

    Transect specs: name, start_xyz, end_xyz, optional samples (63),
    highland_threshold_m (2000), plateau_drop_m (1000). Their total sampling
    budget is 128 points per epoch. All paths are sampled in one batch per epoch.
    The injected sampler(frame, points) follows native_frame_sampling.sample_frame.
    """
    frames = list(frames)
    times = [_finite_number(frame['time_myr'], 'Saved epoch') for frame in frames]
    if not times or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Review requires unique, increasing saved epochs.')
    run_ids = {frame['run_id'] for frame in frames if frame.get('run_id') is not None}
    if len(run_ids) > 1:
        raise ValueError('Material identities cannot be joined across different runs.')
    marker_ids = list(marker_ids)
    if any(type(value) is not int or value < 0 for value in marker_ids) or len(set(marker_ids)) != len(marker_ids):
        raise ValueError('Select unique nonnegative integer trace IDs.')
    profiles, all_points = [], []
    for spec in transects:
        name = spec['name']
        if not isinstance(name, str) or not name or name in {p['name'] for p in profiles}:
            raise ValueError('Transects require unique nonempty names.')
        points, distance = transect_points(spec['start_xyz'], spec['end_xyz'], spec.get('samples', 63))
        threshold = _finite_number(spec.get('highland_threshold_m', 2000.), 'Highland threshold')
        drop = _finite_number(spec.get('plateau_drop_m', 1000.), 'Summit cap drop')
        if drop <= 0:
            raise ValueError('Summit cap drop must be positive.')
        profiles.append(dict(name=name, reference_frame='fixed_geographic_minor_great_circle',
                             start_xyz=points[0].tolist(), end_xyz=points[-1].tolist(),
                             xyz=points.tolist(), distance_km=distance.tolist(),
                             highland_threshold_m=threshold, plateau_drop_m=drop, epochs=[]))
        all_points.append(points)
    total = sum(len(points) for points in all_points)
    if total > MAX_NATIVE_POINTS_PER_EPOCH:
        raise ValueError('Combined transects exceed 128 native points per epoch.')
    if total:
        if sampler is None:
            from native_frame_sampling import sample_frame
            sampler = sample_frame
        points = np.vstack(all_points)
        for frame, time in zip(frames, times):
            sampled = sampler(frame, points.copy())
            elevation = _sample_values(sampled, 'elevation', total)
            if any(value is None for value in elevation):
                raise ValueError('Native surface elevations must be recorded and finite at every profile point.')
            optional = {key: _sample_values(sampled, key, total) for key in SAMPLE_FIELDS}
            face_ids = _ids(frame, 'material_face_id')
            selected = np.asarray(sampled.get('material_face', np.full(total, -1)))
            if (selected.shape != (total,) or selected.dtype.kind not in 'iu'
                    or np.any(selected < -1) or np.any(selected >= len(face_ids))):
                raise ValueError('Native material selections must be saved face indices or -1 for ocean.')
            selected_ids = [int(face_ids[index]) if index >= 0 else None for index in selected]
            states = {key: [_optional_scalar(frame, 'material_'+key, index, len(face_ids),
                        integer=key in ('root_id', 'parent_id', 'collision_sheet')) if index >= 0 else None
                       for index in selected] for key in FACE_FIELDS}
            offset = 0
            for profile in profiles:
                count = len(profile['distance_km'])
                part = slice(offset, offset+count)
                row = dict(time_myr=time, native_surface_elevation_m=elevation[part],
                           material_face_id=selected_ids[part],
                           selected_face_state={key: values[part] for key, values in states.items()},
                           collision_surface_version=frame.get('collision_surface_version'),
                           arc_surface_version=frame.get('arc_surface_version'))
                row.update({('raw_surface_elevation_m' if key == 'raw_elevation_m' else key): values[part]
                            for key, values in optional.items()})
                parameters = {key: profile[key] for key in ('highland_threshold_m', 'plateau_drop_m')}
                row['metrics'] = profile_metrics(profile['distance_km'], row['native_surface_elevation_m'], **parameters)
                raw = row['raw_surface_elevation_m']
                row['raw_metrics'] = (profile_metrics(profile['distance_km'], raw, **parameters)
                                      if all(value is not None for value in raw) else None)
                profile['epochs'].append(row)
                offset += count
    return dict(schema_version=1, run_id=next(iter(run_ids), None), inspected_epochs_myr=times,
                native_points_per_epoch=total, radius_km=RADIUS_KM,
                profiles=profiles, material_histories=_marker_histories(frames, marker_ids),
                source_scope=[{key: frame.get(key) for key in
                              ('time_myr', 'index', 'source_frame_sha256', 'source_metadata_sha256')}
                              for frame in frames],
                interpretation=dict(
                    profiles='Fixed geographic paths encounter changing material; they are not tracked orogen identities.',
                    widths='Connected component containing the highest sample; linear spatial threshold interpolation only. Endpoint-censored widths are lower bounds.',
                    summit_cap='Width within the chosen drop below the sampled summit; a wide cap alone does not establish a plateau.',
                    adjacent_lows='Sample minima on either flank outside the central highland; no basin origin is inferred.',
                    elevation='Native sampled mapped surface. Raw physical surface is reported separately only when saved reconstruction supplies it; legacy clipping can differ.',
                    components='Selected-sheet, stack and thermal surface terms are native samples. Selected face state is a face scalar, not another surface-height term.',
                    marker_relief='Representative material relief is not total mapped elevation or summit height.',
                    activity='Positive gain counters mix shortening, magmatism, thermal effects and unloading. Activity is observed over a saved interval, not proof of active collision or a precise cessation date.',
                    budgets='Counters are not a complete elevation budget. Denudation/rebound, thermal/foreland subsidence and ridge/rift/inversion counters explain contributions already represented in aggregate gain/loss counters; do not add them twice.',
                    erosion='For structure_version 1, erosion_m is net denudation after rebound; legacy erosion_m is signed relief relaxation.',
                    age='Marker age is time since recorded simulation birth, not rock age or mountain age.'),
                limitations=[
                    'Sparse saved epochs cannot date onset or cessation between observations; unavailable values remain null.',
                    'Sampled extrema and widths are bounded probes, not global elevation bounds or whole-belt dimensions.',
                    'A material marker follows its exact trace_id; missing IDs are not replaced by nearby or descendant faces.',
                    'Profiles reconstruct existing geometry; narrow peaks and channels below the sampling or physical mesh scale can be missed.',
                    'This review does not model sediment routing, detailed erosion or goSPL terrain.'])
