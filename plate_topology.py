"""Connected spherical plate domains and conservative fragmentation plans.

Only geometry is inspected here. The simulation owns plate identities, forces,
parcel histories and the application of these plans. In particular, a continent
is never transferred to a nearby plate merely because its parent is remote.
"""
from dataclasses import dataclass

import numpy as np


def domain_components(plate, width, height):
    """Return compact component IDs and their plate slots in one raster pass.

    Components include BOTH oceanic and continental lithosphere. Longitude
    wraps. An edge continued over either pole meets the same polar row half a
    turn away; the two polar caps never connect to one another. With odd widths
    the continued interval overlaps two columns, which are both considered.
    """
    grid = np.asarray(plate).reshape(height, width)
    labels = np.empty((height, width), np.int32)
    parent, owner = [], []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        if owner[a] != owner[b]:
            return
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    previous = []
    for y, row in enumerate(grid):
        breaks = np.r_[0, np.flatnonzero(row[1:] != row[:-1])+1, width]
        runs = []
        for start, stop in zip(breaks[:-1], breaks[1:]):
            label = len(parent)
            parent.append(label)
            owner.append(int(row[start]))
            labels[y, start:stop] = label
            runs.append((int(start), int(stop), label))
        if len(runs) > 1:
            union(runs[0][2], runs[-1][2])
        i = j = 0
        while i < len(previous) and j < len(runs):
            a, b = previous[i], runs[j]
            if max(a[0], b[0]) < min(a[1], b[1]):
                union(a[2], b[2])
            if a[1] <= b[1]:
                i += 1
            else:
                j += 1
        previous = runs
    for row in (labels[0], labels[-1]):
        for shift in set((width//2, (width+1)//2)):
            pairs = np.unique(np.column_stack((row, np.roll(row, shift))), axis=0)
            for a, b in pairs:
                union(int(a), int(b))
    roots = np.array([find(i) for i in range(len(parent))], np.int32)
    unique, compact = np.unique(roots, return_inverse=True)
    return compact[labels.ravel()].astype(np.int32), np.asarray(owner, np.int32)[unique]


def assign_material_components(labels, component_owner, width, height, cells,
                               owners, positions=None, xyz=None, max_rings=2):
    """Attribute material to a same-owner component without remote reassignment.

    Overlapping parcels can have centers just outside their visible plate. A
    bounded neighborhood resolves this using spherical distance when positions
    are supplied. A still-unresolved point remains -1, explicitly. The original
    parcel and its owner must be preserved by the caller.
    """
    cells = np.asarray(cells, np.int64)
    owners = np.asarray(owners)
    labels = np.asarray(labels)
    result = labels[cells].copy()
    result[component_owner[result] != owners] = -1
    indices = np.flatnonzero(result < 0)
    if not len(indices):
        return result
    # Bound the temporary arrays even on the largest supported simulation.
    for offset in range(0, len(indices), 32_768):
        subset = indices[offset:offset+32_768]
        base_y, base_x = np.divmod(cells[subset], width)
        best = np.full(len(subset), np.inf)
        choice = np.full(len(subset), -1, np.int32)
        for dy in range(-max_rings, max_rings+1):
            raw_y = base_y+dy
            crossing = (raw_y < 0) | (raw_y >= height)
            y = np.where(raw_y < 0, -raw_y-1,
                         np.where(raw_y >= height, 2*height-raw_y-1, raw_y))
            y = np.clip(y, 0, height-1)
            # Half-column continuation for odd widths has two valid neighbors.
            shifts = (width//2, (width+1)//2) if width % 2 else (width//2,)
            for shift in shifts:
                for dx in range(-max_rings, max_rings+1):
                    x = (base_x+dx+crossing*shift) % width
                    candidate_cell = y*width+x
                    candidate = labels[candidate_cell]
                    valid = component_owner[candidate] == owners[subset]
                    if positions is not None and xyz is not None:
                        distance = np.sum((positions[subset]-xyz[candidate_cell])**2, axis=1)
                    else:
                        distance = np.full(len(subset), float(dx*dx+dy*dy))
                    use = valid & (distance < best)
                    best[use] = distance[use]
                    choice[use] = candidate[use]
        if positions is not None and xyz is not None:
            # Longitude fans out near a pole. A cell at +90 degrees longitude
            # can be physically closer than the +/-2 column candidates above.
            # Search only the two-cell angular cap, in bounded matrix chunks.
            cap_rows = min(height, 2*max_rings+1)
            radius2 = 2-2*np.cos(max_rings*np.pi/height)
            for north in (True, False):
                polar = np.flatnonzero(base_y <= max_rings if north else base_y >= height-1-max_rings)
                row_start = 0 if north else height-cap_rows
                candidates = np.arange(row_start*width, (row_start+cap_rows)*width)
                candidate_components = labels[candidates]
                for start in range(0, len(polar), 64):
                    local = polar[start:start+64]
                    distance = 2-2*(positions[subset[local]] @ xyz[candidates].T)
                    matches = component_owner[candidate_components][None, :] == owners[subset[local], None]
                    distance[(~matches) | (distance > radius2+1e-12)] = np.inf
                    nearest = np.argmin(distance, axis=1)
                    value = distance[np.arange(len(local)), nearest]
                    use = value < best[local]
                    best[local[use]] = value[use]
                    choice[local[use]] = candidate_components[nearest[use]]
        result[subset] = choice
    return result


def _component_adjacency(labels, width, height):
    """Shared boundary angular lengths, excluding the zero-length polar edge."""
    grid = np.asarray(labels).reshape(height, width)
    a = np.r_[grid.ravel(), grid[:-1].ravel()]
    b = np.r_[np.roll(grid, -1, axis=1).ravel(), grid[1:].ravel()]
    use = a != b
    length = np.r_[np.full(width*height, np.pi/height),
                   np.repeat(2*np.pi/width*np.sin(np.arange(1, height)*np.pi/height), width)]
    pairs = np.sort(np.column_stack((a[use], b[use])), axis=1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    weights = np.bincount(inverse, weights=length[use], minlength=len(unique))
    return unique.astype(np.int32), weights


@dataclass
class DomainRepairPlan:
    labels: np.ndarray
    component_owner: np.ndarray
    area: np.ndarray
    material_mass: np.ndarray
    land_area: np.ndarray
    core_components: dict
    parcel_components: np.ndarray
    unresolved_mass_by_owner: dict
    adjacency: np.ndarray
    shared_boundary: np.ndarray
    fragments: list
    absorptions: list
    deferred: list


def plan_domain_repairs(plate, crust, width, height, cell_area, *,
                        parcel_cells=None, parcel_plate=None, parcel_mass=None,
                        parcel_xyz=None, xyz=None, parcel_groups=None, protected_slots=(),
                        available_slots=None, empty_ocean_fraction=.0001):
    """Describe detached domains with no mutation and no change of motion.

    The largest connected domain retains its plate identity. Detached buoyant
    material always qualifies for its own fragment identity. Only empty ocean
    remnants no larger than four cells and the larger of four average cells or
    0.01% of planetary area can be absorbed, and only across a shared boundary. Larger
    detached ocean domains also become fragments. A protected initial ocean
    stays a single kinematic plate, as requested by the worldbuilding model.

    Absorptions target a larger component (or a core), forming an acyclic chain;
    the caller resolves target component IDs AFTER fragment slot allocation.
    Capacity exhaustion is explicit in ``deferred``; it never causes continents
    to be welded to an unrelated neighbor. Inputs may all be flat arrays.

    Nonnegative ``parcel_groups`` protect coherent cratonic material. Within
    each group and existing owner, all parcels follow the mass-weighted mode
    of their valid local component assignments. This works for concave cratons
    whose geometric centroid lies in ocean; -1 means ordinary ungrouped crust.
    """
    plate = np.asarray(plate).reshape(-1)
    crust = np.asarray(crust).reshape(-1)
    cell_area = np.asarray(cell_area).reshape(-1)
    labels, owner = domain_components(plate, width, height)
    count = len(owner)
    area = np.bincount(labels, weights=cell_area, minlength=count)
    land_area = np.bincount(labels, weights=cell_area*(crust > 0), minlength=count)
    material_mass = np.zeros(count)
    parcel_components = np.empty(0, np.int32)
    unresolved_mass = {}
    if parcel_cells is not None:
        parcel_plate = np.asarray(parcel_plate)
        parcel_mass = np.asarray(parcel_mass)
        parcel_components = assign_material_components(labels, owner, width, height,
                                                      parcel_cells, parcel_plate, parcel_xyz, xyz)
        if parcel_groups is not None:
            groups = np.asarray(parcel_groups).reshape(-1)
            if len(groups) != len(parcel_components):
                raise ValueError('parcel_groups must have one value per parcel')
            protected = np.flatnonzero(groups >= 0)
            if len(protected):
                # Keep existing owners distinct even if malformed/inherited
                # input reuses a group ID on two already independent plates.
                keys = np.column_stack((groups[protected], parcel_plate[protected]))
                _, membership = np.unique(keys, axis=0, return_inverse=True)
                order = np.argsort(membership, kind='stable')
                breaks = np.r_[0, np.flatnonzero(np.diff(membership[order]))+1, len(order)]
                for start, stop in zip(breaks[:-1], breaks[1:]):
                    members = protected[order[start:stop]]
                    valid_members = members[parcel_components[members] >= 0]
                    if not len(valid_members):
                        continue
                    choices, inverse = np.unique(parcel_components[valid_members], return_inverse=True)
                    weights = np.bincount(inverse, weights=parcel_mass[valid_members], minlength=len(choices))
                    chosen = choices[np.argmax(weights)]
                    parcel_components[members] = chosen
        valid = parcel_components >= 0
        material_mass = np.bincount(parcel_components[valid], weights=parcel_mass[valid], minlength=count)
        for slot in np.unique(parcel_plate[~valid]):
            unresolved_mass[int(slot)] = float(parcel_mass[(~valid) & (parcel_plate == slot)].sum())
    cores = {}
    for slot in np.unique(owner):
        components = np.flatnonzero(owner == slot)
        cores[int(slot)] = int(components[np.argmax(area[components])])
    core_set, protected = set(cores.values()), set(protected_slots)
    pairs, length = _component_adjacency(labels, width, height)
    neighbors = [[] for _ in range(count)]
    for (a, b), weight in zip(pairs, length):
        neighbors[a].append((int(b), float(weight)))
        neighbors[b].append((int(a), float(weight)))
    earth_area = float(cell_area.sum())
    tiny_limit = max(4*earth_area/len(plate), empty_ocean_fraction*earth_area)
    cell_count = np.bincount(labels, minlength=count)
    detached = [i for i in range(count) if i not in core_set and int(owner[i]) not in protected]
    absorbable = {i for i in detached if area[i] <= tiny_limit and cell_count[i] <= 4 and
                  material_mass[i] <= earth_area*1e-14 and land_area[i] == 0}
    absorptions, fragments = [], []
    for component in detached:
        base = dict(component=int(component), parent_slot=int(owner[component]),
                    area_km2=float(area[component]), material_mass_km2=float(material_mass[component]),
                    land_area_km2=float(land_area[component]))
        if component in absorbable:
            candidates = [(neighbor, boundary) for neighbor, boundary in neighbors[component]
                          if neighbor not in absorbable or (area[neighbor], -neighbor) > (area[component], -component)]
            if candidates:
                target, boundary = max(candidates, key=lambda row: (row[1], area[row[0]], -row[0]))
                absorptions.append(dict(base, target_component=int(target), shared_boundary_radians=float(boundary)))
                continue
        fragments.append(base)
    # Meaningful material gets first choice when capacity is genuinely bounded.
    fragments.sort(key=lambda row: (max(row['material_mass_km2'], row['land_area_km2']) > 0,
                                    row['area_km2'], -row['component']), reverse=True)
    deferred = []
    if available_slots is not None:
        available_slots = max(0, int(available_slots))
        deferred, fragments = fragments[available_slots:], fragments[:available_slots]
        deferred = [dict(row, reason='plate capacity exhausted; material and parent preserved') for row in deferred]
    return DomainRepairPlan(labels, owner, area, material_mass, land_area, cores,
                            parcel_components, unresolved_mass, pairs, length,
                            fragments, absorptions, deferred)
