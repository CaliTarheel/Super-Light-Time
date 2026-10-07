"""Conservative, sphere-aware deposition of enduring buoyant crust.

Each material sample carries an oriented finite footprint, rather than a point
in longitude/latitude. The compact tent footprint is integrated approximately
against each destination cell's footprint. Per-sample weights sum to its exact
area, including at the poles. This is a numerical remap kernel, not extra crust
production or an operation that fills holes in a coastline.
"""
from __future__ import annotations

import numpy as np


def footprints(position, width, height):
    """Tangent east axes and angular tent half-widths at material creation."""
    longitude = np.arctan2(position[:, 1], position[:, 0])
    east = np.column_stack((-np.sin(longitude), np.cos(longitude), np.zeros(len(position))))
    cosine = np.linalg.norm(position[:, :2], axis=1)
    extent = np.column_stack((np.maximum(cosine, 1e-8) * np.pi / width,
                              np.full(len(position), .5 * np.pi / height)))
    return east, extent


def _repeat_range(start, count):
    index = np.repeat(np.arange(len(count)), count)
    local = np.arange(len(index)) - np.repeat(np.cumsum(count) - count, count)
    return index, start[index] + local


def deposit(position, east, extent, mass, kind, relief, suture, owner,
            width, height, xyz, cell_area, fallback_lookup, chunk_size=2048,
            extra_fields=None, owner_occupancy=None):
    """Return mass/property sums and the owner with greatest deposited area.

    Candidate cells come from a spherical cap around each oriented footprint.
    Its longitude interval becomes a full ring when it contains a pole; there
    is no seam cut or polar row clamp in the actual material deposition.
    Temporary arrays are bounded by parcel chunks, not a global parcel × grid
    search. A tiny, unresolved footprint still conserves its full area.

    When supplied, owner_occupancy is an output dictionary of owner -> int32
    cells at the existing 0.36 per-owner area-density threshold. It reuses the
    already computed plate_mass, never changes deposition, and leaves the
    historical return tuple unchanged. Callers must validate material geometry
    before reusing this derived sparse occupancy after later mutations.
    """
    size = width * height
    extra = {} if extra_fields is None else extra_fields
    for name, field in extra.items():
        if np.shape(field) != np.shape(mass) or not np.isfinite(field).all():
            raise ValueError(f'Extra material field {name} must align with finite parcel values.')
    if owner_occupancy is not None:
        owner_occupancy.clear()
    sums = np.zeros((5+len(extra), size), float)
    best = np.zeros(size)
    dominant = np.zeros(size, np.int16)
    dlon, dlat = 2 * np.pi / width, np.pi / height
    row_lat = np.pi / 2 - (np.arange(height) + .5) * dlat
    row_sin, row_cos = np.sin(row_lat), np.cos(row_lat)
    target_radius = .5 * np.hypot(dlon, dlat)
    for plate in np.unique(owner):
        selected = np.flatnonzero(owner == plate)
        plate_mass = np.zeros(size)
        for begin in range(0, len(selected), chunk_size):
            ids = selected[begin:begin + chunk_size]
            p, e, span = position[ids], east[ids], extent[ids]
            north = np.cross(p, e)
            lat = np.arcsin(np.clip(p[:, 2], -1, 1))
            longitude = np.arctan2(p[:, 1], p[:, 0])
            radius = np.minimum(np.hypot(span[:, 0], span[:, 1]) + target_radius, np.pi)
            first = np.maximum(0, np.ceil((np.pi / 2 - lat - radius) / dlat - .5).astype(int))
            last = np.minimum(height - 1, np.floor((np.pi / 2 - lat + radius) / dlat - .5).astype(int))
            row_source, rows = _repeat_range(first, np.maximum(0, last - first + 1))
            denominator = np.maximum(np.cos(lat[row_source]) * row_cos[rows], 1e-20)
            cosine = (np.cos(radius[row_source]) - np.sin(lat[row_source]) * row_sin[rows]) / denominator
            half_lon = np.arccos(np.clip(cosine, -1, 1))
            first_col = np.ceil((longitude[row_source] - half_lon + np.pi) / dlon - .5).astype(int)
            last_col = np.floor((longitude[row_source] + half_lon + np.pi) / dlon - .5).astype(int)
            column_count = np.minimum(width, np.maximum(0, last_col - first_col + 1))
            pair_row, cols = _repeat_range(first_col, column_count)
            source = row_source[pair_row]
            dest = rows[pair_row] * width + cols % width
            target = xyz[dest]
            # Integrating two finite footprints broadens the remap kernel by
            # the destination cell's projected size. This matters when an
            # originally narrow polar wedge travels toward the equator.
            target_cos = row_cos[rows[pair_row]]
            target_east = np.column_stack((-target[:, 1] / target_cos,
                                           target[:, 0] / target_cos, np.zeros(len(dest))))
            target_north = np.cross(target, target_east)
            se, sn = e[source], north[source]
            half_east = .5 * dlon * target_cos
            half_north = .5 * dlat
            a = np.sqrt(span[source, 0] ** 2
                        + (np.sum(target_east * se, axis=1) * half_east) ** 2
                        + (np.sum(target_north * se, axis=1) * half_north) ** 2)
            b = np.sqrt(span[source, 1] ** 2
                        + (np.sum(target_east * sn, axis=1) * half_east) ** 2
                        + (np.sum(target_north * sn, axis=1) * half_north) ** 2)
            u = np.abs(np.sum(target * se, axis=1)) / np.maximum(a, 1e-12)
            v = np.abs(np.sum(target * sn, axis=1)) / np.maximum(b, 1e-12)
            weight = np.maximum(1 - u, 0) * np.maximum(1 - v, 0) * cell_area[dest]
            valid = weight > 0
            source, dest, weight = source[valid], dest[valid], weight[valid]
            normalization = np.bincount(source, weights=weight, minlength=len(ids))
            missing = np.flatnonzero(normalization == 0)
            if len(missing):
                # This is only the unresolved-footprint limit; unlike the old
                # point-only method, resolved polar footprints cover full rings.
                extra_source, extra_dest, extra_weight = [], [], []
                for target_id, fraction in fallback_lookup(p[missing]):
                    extra_source.append(missing)
                    extra_dest.append(target_id)
                    extra_weight.append(fraction)
                source = np.concatenate((source, *extra_source))
                dest = np.concatenate((dest, *extra_dest))
                weight = np.concatenate((weight, *extra_weight))
                normalization[missing] = 1
            amount = mass[ids[source]] * weight / normalization[source]
            values = (amount, amount * relief[ids[source]], amount * suture[ids[source]],
                      amount * (kind[ids[source]] == 2), amount * (kind[ids[source]] == 3))
            values += tuple(amount*field[ids[source]] for field in extra.values())
            for total, value in zip(sums, values):
                np.add.at(total, dest, value)
            np.add.at(plate_mass, dest, amount)
        if owner_occupancy is not None:
            # Keep division and threshold identical to the independent
            # per-owner accretion calculation, including floating-point ties.
            owner_occupancy[int(plate)] = np.flatnonzero(plate_mass/cell_area >= .36).astype(np.int32)
        win = plate_mass > best
        best[win], dominant[win] = plate_mass[win], plate
    base = (*sums[:5], best, dominant)
    if extra_fields is not None:
        return (*base, {name: sums[5+i] for i, name in enumerate(extra)})
    return base
