"""Continue a continental split through attached oceanic lithosphere.

Continental crack geometry is supplied by the caller and is never changed here.
Ocean ownership follows the nearest fractured continental shore by shortest grid
paths within the parent plate. Paths use spherical distances, wrap longitude,
and continue across each polar cap. They terminate at existing plate boundaries.
This is a geometric worldbuilding closure, not elastic fracture propagation.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from fracture import component_labels


def extend_continental_partition(parent_mask, continental_mask, child_side,
                                 width, height, radius_km=6371.):
    """Return child-owned cells with the supplied continental sides unchanged.

Only continental cells of this parent act as coastline seeds. An ocean-only
disconnected component without such a shore keeps its original parent owner.
Parcel ownership is deliberately outside this routine: hidden buoyant material
and protected cratons retain the caller's coherent material classification.
"""
    width, height = int(width), int(height)
    if width < 2 or height < 2:
        raise ValueError('A spherical partition needs at least two rows and columns.')
    count = width * height
    parent = np.asarray(parent_mask, dtype=bool).reshape(-1)
    continent = np.asarray(continental_mask, dtype=bool).reshape(-1)
    side = np.asarray(child_side, dtype=bool).reshape(-1)
    if any(len(field) != count for field in (parent, continent, side)):
        raise ValueError('Partition rasters must have width times height cells.')
    continent = continent & parent
    ocean = parent & ~continent
    result = np.zeros(count, dtype=bool)
    result[continent] = side[continent]
    if not np.any(ocean) or not np.any(continent):
        return result

    # Only the shore can launch a path into ocean; interior continental cells
    # keep their already chosen sides and need no heap entries.
    ocean_grid = ocean.reshape(height, width)
    adjoining = np.roll(ocean_grid, 1, axis=1) | np.roll(ocean_grid, -1, axis=1)
    adjoining[1:] |= ocean_grid[:-1]
    adjoining[:-1] |= ocean_grid[1:]
    half = width // 2
    for shift in {half, (width + 1) // 2}:
        adjoining[0] |= np.roll(ocean_grid[0], shift)
        adjoining[-1] |= np.roll(ocean_grid[-1], shift)
    shore = np.flatnonzero(continent & adjoining.ravel())
    if not len(shore):
        return result
    distance = np.full(count, np.inf)
    distance[shore] = 0.
    queue = [(0., int(side[cell]), int(cell)) for cell in shore]
    heapq.heapify(queue)
    dlat, dlon = math.pi / height, 2 * math.pi / width
    latitudes = math.pi / 2 - (np.arange(height) + .5) * dlat
    east_cost = 2 * radius_km * np.arcsin(np.clip(np.cos(latitudes) * math.sin(dlon / 2), 0, 1))
    north_cost = radius_km * dlat
    # At an odd grid width the two nearest half-turn columns flank the exact
    # continuation. Their true spherical distance keeps the cap well behaved.
    polar_cost = {shift: radius_km * math.acos(float(np.clip(
        math.sin(latitudes[0])**2 + math.cos(latitudes[0])**2 * math.cos(shift * dlon), -1, 1)))
        for shift in {half, (width + 1) // 2}}

    while queue:
        travelled, label, cell = heapq.heappop(queue)
        if travelled > distance[cell] + 1e-9 or label != int(result[cell]):
            continue
        y, x = divmod(cell, width)
        neighbours = [(y * width + (x - 1) % width, east_cost[y]),
                      (y * width + (x + 1) % width, east_cost[y])]
        if y:
            neighbours.append((cell - width, north_cost))
        else:
            neighbours.extend((((x + shift) % width, cost) for shift, cost in polar_cost.items()))
        if y + 1 < height:
            neighbours.append((cell + width, north_cost))
        else:
            neighbours.extend(((y * width + (x + shift) % width, cost) for shift, cost in polar_cost.items()))
        for neighbour, cost in neighbours:
            if not ocean[neighbour]:
                continue
            proposed = travelled + cost
            if proposed < distance[neighbour] - 1e-9 or (abs(proposed - distance[neighbour]) <= 1e-9 and label < int(result[neighbour])):
                distance[neighbour] = proposed
                result[neighbour] = bool(label)
                heapq.heappush(queue, (proposed, label, neighbour))
    return result


def bounded_continental_partition(parent_mask, continental_mask, child_side,
                                  width, height, cell_area, edge_a, edge_b,
                                  edge_length, max_crumb_fraction=.025):
    """Use coastal continuation only when it shortens the added ocean break.

The original curved cut remains a fallback. Both alternatives preserve the
continental split, and the same connected-component bound used for fracture
acceptance excludes new substantial disconnected pieces. Comparing lengths
is a least-added-rupture heuristic; no existing boundary is suppressed.
"""
    parent = np.asarray(parent_mask, dtype=bool).reshape(-1)
    continent = np.asarray(continental_mask, dtype=bool).reshape(-1)
    original = parent & np.asarray(child_side, dtype=bool).reshape(-1)
    candidate = extend_continental_partition(parent, continent, original, width, height)
    ocean_edge = (parent[edge_a] & parent[edge_b]
                  & ~continent[edge_a] & ~continent[edge_b])

    def interface_length(region):
        return float(np.sum(edge_length[ocean_edge & (region[edge_a] != region[edge_b])]))

    original_length = interface_length(original)
    candidate_length = interface_length(candidate)
    details = dict(method='original curved cut', original_ocean_interface_km=original_length,
                   candidate_ocean_interface_km=candidate_length,
                   ocean_interface_km=original_length, ocean_cells_reassigned=0,
                   ocean_area_reassigned_km2=0.)
    if candidate_length >= original_length - 1e-6:
        return original, details
    if not np.any(candidate & continent) or not np.any(parent & ~candidate & continent):
        return original, details
    parent_labels, parent_count = component_labels(parent, width, height)
    disconnected = 0.
    for half in (candidate, parent & ~candidate):
        labels, count = component_labels(half, width, height)
        weights = np.bincount(labels[half], weights=cell_area[half], minlength=count)
        inherited = np.full(count, parent_count, dtype=np.int32)
        np.minimum.at(inherited, labels[half], parent_labels[half])
        largest = np.zeros(parent_count)
        np.maximum.at(largest, inherited, weights)
        total = np.bincount(parent_labels[half], weights=cell_area[half], minlength=parent_count)
        disconnected = max(disconnected, float(np.sum(total - largest) / max(total.sum(), 1.)))
    if disconnected > max_crumb_fraction:
        details['candidate_disconnected_area_fraction'] = disconnected
        return original, details
    changed = original != candidate
    details.update(method='coastal shortest-path continuation', ocean_interface_km=candidate_length,
                   ocean_cells_reassigned=int(np.count_nonzero(changed)),
                   ocean_area_reassigned_km2=float(np.sum(cell_area[changed])),
                   new_disconnected_area_fraction=disconnected)
    return candidate, details
