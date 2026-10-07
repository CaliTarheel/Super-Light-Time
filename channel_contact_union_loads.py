"""Conservative read-only contact-cell loads on shared flexure strips.

Each cell has one physical footprint area and one signed buoyant force. The
hinge-line union supplies disjoint pieces of that cell on normal rays. Linear
normal-grid shape functions distribute the spherical-area-weighted pressure
to solver nodes, so constant and affine virtual work retain the same force.
"""

import math

import numpy as np


_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(4)


def lump_projected_contact_loads(line_union, offsets_m, zone_loads, *,
                                 zone_area_tolerance=1e-3):
    """Return nodal pressure for one conserved force per physical contact cell.

    `zone_loads` maps union zone IDs to records containing `physical_area_km2`,
    `mean_buoyancy_pa`, and `buoyancy_force_n`. These may come from a regional
    material inventory; this function never invents or transfers that mass.
    """
    if (not isinstance(line_union, dict) or not isinstance(zone_loads, dict)
            or not zone_loads):
        raise ValueError('Contact loads need a hinge union and named cell loads.')
    radius = float(line_union.get('radius_m', float('nan')))
    weights = np.asarray(line_union.get('line_weight_m'), float)
    rows = line_union.get('contact_intervals')
    offsets = np.asarray(offsets_m, float)
    tolerance = float(zone_area_tolerance)
    if (not np.isfinite(radius) or radius <= 0.
            or weights.ndim != 1 or not len(weights)
            or not np.isfinite(weights).all() or np.any(weights <= 0.)
            or not isinstance(rows, list) or len(rows) != len(weights)
            or offsets.ndim != 1 or len(offsets) < 5
            or not np.isfinite(offsets).all() or offsets[0] < 0.
            or np.any(np.diff(offsets) <= 0.)
            or offsets[-1] >= .5 * math.pi * radius
            or not np.allclose(np.diff(offsets),
                               (offsets[-1] - offsets[0]) / (len(offsets) - 1),
                               rtol=1e-5, atol=1e-5)
            or not np.isfinite(tolerance) or not 0. < tolerance < 1.):
        raise ValueError('Contact loads need aligned metre rays and a bounded area tolerance.')
    spacing = float((offsets[-1] - offsets[0]) / (len(offsets) - 1))
    pressures, source_areas, source_forces = {}, {}, {}
    for zone_id, record in zone_loads.items():
        if (not isinstance(zone_id, str) or not zone_id
                or not isinstance(record, dict)):
            raise ValueError('Contact loads need named finite cell records.')
        area = float(record.get('physical_area_km2', float('nan'))) * 1e6
        pressure = float(record.get('mean_buoyancy_pa', float('nan')))
        force = float(record.get('buoyancy_force_n', float('nan')))
        if (not np.isfinite([area, pressure, force]).all() or area <= 0.
                or not math.isclose(force, pressure * area,
                                    rel_tol=2e-12, abs_tol=1e-4)):
            raise ValueError('Contact cell pressure disagrees with its conserved force.')
        pressures[zone_id] = pressure
        source_areas[zone_id] = area
        source_forces[zone_id] = force
    union_area = {zone_id: 0. for zone_id in zone_loads}
    prepared_rows = []
    seen = set()
    for ray, intervals in enumerate(rows):
        if not isinstance(intervals, list) or not intervals:
            raise ValueError('Contact loads need occupied normal rays.')
        prepared = []
        previous_hi = None
        for interval in intervals:
            if not isinstance(interval, dict):
                raise ValueError('Contact loads need named interval records.')
            zone_id = interval.get('zone_id')
            if not isinstance(zone_id, str) or zone_id not in zone_loads:
                raise ValueError('Contact load has no matching material cell.')
            seen.add(zone_id)
            lo = float(interval.get('lower_offset_m', float('nan')))
            hi = float(interval.get('upper_offset_m', float('nan')))
            given_area = float(interval.get('represented_area_m2', float('nan')))
            if abs(lo - offsets[0]) <= 1e-5:
                lo = float(offsets[0])
            if abs(hi - offsets[-1]) <= 1e-5:
                hi = float(offsets[-1])
            if (not np.isfinite([lo, hi, given_area]).all()
                    or not offsets[0] <= lo < hi <= offsets[-1]
                    or given_area <= 0.):
                raise ValueError('Contact load interval exceeds the physical strip.')
            if previous_hi is not None and lo < previous_hi - 1e-5:
                raise ValueError('Contact load intervals overlap on one hinge ray.')
            previous_hi = hi
            exact_area = radius * weights[ray] * (
                math.sin(hi / radius) - math.sin(lo / radius))
            if not math.isclose(given_area, exact_area,
                                rel_tol=2e-10, abs_tol=1e-3):
                raise ValueError('Contact load interval has inconsistent spherical area.')
            union_area[zone_id] += given_area
            prepared.append((zone_id, lo, hi, exact_area))
        prepared_rows.append(prepared)
    if seen != set(zone_loads):
        raise ValueError('Contact load cells must all occur in the hinge union.')
    for zone_id, area in source_areas.items():
        if abs(union_area[zone_id] - area) > tolerance * area:
            raise ValueError('Contact cell hinge quadrature misses its physical area.')
    # The hinge quadrature is approximate, but its force inventory is exact:
    # normalize each cell's pressure by its represented area after verifying
    # that the area approximation is sufficiently resolved.
    effective_pressure = {zone_id: source_forces[zone_id] / union_area[zone_id]
                          for zone_id in zone_loads}
    nodal_force = np.zeros((len(weights), len(offsets)))
    represented_area = {zone_id: 0. for zone_id in zone_loads}
    represented_force = {zone_id: 0. for zone_id in zone_loads}
    for ray, intervals in enumerate(prepared_rows):
        for zone_id, lo, hi, exact_area in intervals:
            pressure = effective_pressure[zone_id]
            interior = offsets[(offsets > lo) & (offsets < hi)]
            bounds = np.r_[lo, interior, hi]
            integrated_length = 0.
            for left, right in zip(bounds[:-1], bounds[1:]):
                if right <= left:
                    continue
                index = int(np.searchsorted(offsets, .5 * (left + right), side='right') - 1)
                if not 0 <= index < len(offsets) - 1:
                    raise ValueError('Contact load lacks a normal-grid segment.')
                samples = .5 * (left + right) + .5 * (right - left) * _GAUSS_X
                area_weights = .5 * (right - left) * _GAUSS_W * np.cos(samples / radius)
                right_shape = (samples - offsets[index]) / spacing
                nodal_force[ray, index] += pressure * float(
                    area_weights @ (1. - right_shape))
                nodal_force[ray, index + 1] += pressure * float(
                    area_weights @ right_shape)
                integrated_length += float(np.sum(area_weights))
            quadrature_area = weights[ray] * integrated_length
            if not math.isclose(quadrature_area, exact_area,
                                rel_tol=2e-10, abs_tol=1e-3):
                raise ValueError('Contact load quadrature lost spherical area.')
            represented_area[zone_id] += quadrature_area
            represented_force[zone_id] += pressure * quadrature_area
    for zone_id in zone_loads:
        if (not math.isclose(represented_area[zone_id], union_area[zone_id],
                             rel_tol=2e-10, abs_tol=1e-3)
                or not math.isclose(represented_force[zone_id],
                                    source_forces[zone_id],
                                    rel_tol=2e-10, abs_tol=1e-4)):
            raise ValueError('Contact cell quadrature lost its conserved force.')
    normal_weights = np.full(len(offsets), spacing)
    normal_weights[[0, -1]] *= .5
    total = float(weights @ nodal_force.sum(axis=1))
    expected = math.fsum(source_forces.values())
    force_tolerance = max(1e-4, 2e-12 * math.fsum(
        abs(force) for force in source_forces.values()))
    if abs(total - expected) > force_tolerance:
        raise ValueError('Contact nodal loads lost source cell force.')
    return dict(buoyancy_pa=nodal_force / normal_weights[None, :],
                nodal_force_n_per_m=nodal_force,
                represented_area_m2_by_zone=represented_area,
                represented_force_n_by_zone=represented_force,
                input_pressure_pa_by_zone=pressures,
                effective_pressure_pa_by_zone=effective_pressure,
                total_represented_force_n=total,
                source_cell_force_n=math.fsum(source_forces.values()),
                scope='read-only spherical contact load; no native force or source commit')
