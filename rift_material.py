"""Persistent material membership and sparse contact for the rift mechanics mesh.

The mesh follows complete continental patches through advection, ownership
changes and array reordering.  It does not sample a fixed geographic stress
grid.  Only topology/new-material changes inspect the occupied material raster;
ordinary refreshes aggregate material properties and reuse reference contacts.
"""
from __future__ import annotations

import hashlib

import numpy as np

from material_geometry import patch_centres
from rift_mesh import assign_cells


def _unique_edges(values):
    if not values:
        return np.empty((0, 2), np.int64)
    pairs = np.sort(np.concatenate(values), axis=1)
    return np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)


def _cell_contacts(cells, members, domains, width, height, *, connectivity_only=False):
    """Sparse occupied-cell contact, with longitude seam and polar continuation.

    No link crosses an unoccupied cell.  ``domains`` specifies which members
    may touch (owner UID, or owner/reference-cell group during initialization).
    Connectivity-only contacts use spanning stars inside each occupied cell;
    exact cliques would waste quadratic storage while giving identical groups.
    Dense coincident supports also use spanning contacts above 64 candidates.
    """
    if not len(cells):
        return np.empty((0, 2), np.int64)
    occupancy = np.unique(np.column_stack((cells, members)), axis=0)
    occupied, starts, counts = np.unique(occupancy[:, 0], return_index=True, return_counts=True)
    links = []

    def join(first, second=None):
        if second is None:
            for group in np.unique(domains[first]):
                nodes = first[domains[first] == group]
                if len(nodes) < 2:
                    continue
                if connectivity_only or len(nodes)*(len(nodes)-1)//2 > 64:
                    links.append(np.column_stack((np.full(len(nodes)-1, nodes[0]), nodes[1:])))
                else:
                    a, b = np.triu_indices(len(nodes), 1)
                    links.append(np.column_stack((nodes[a], nodes[b])))
            return
        for group in np.intersect1d(domains[first], domains[second]):
            a, b = first[domains[first] == group], second[domains[second] == group]
            if connectivity_only:
                links.append(np.array([[a[0], b[0]]], np.int64))
            elif len(a)*len(b) > 64:
                links.append(np.column_stack((np.repeat(a[0], len(b)), b)))
                links.append(np.column_stack((a[1:], np.repeat(b[0], len(a)-1))))
            else:
                links.append(np.column_stack((np.repeat(a, len(b)), np.tile(b, len(a)))))

    for index in np.flatnonzero(counts > 1):
        join(occupancy[starts[index]:starts[index]+counts[index], 1])
    y, x = np.divmod(occupied, width)
    neighbor_sets = [y*width+(x+1) % width,
                     np.where(y+1 < height, occupied+width, -1)]
    for shift in sorted(set((width//2, (width+1)//2))):
        neighbor_sets.append(np.where((y == 0) | (y == height-1),
                                      y*width+(x+shift) % width, -1))
    for other in neighbor_sets:
        locations = np.searchsorted(occupied, other)
        valid = (other >= 0) & (locations < len(occupied))
        valid &= occupied[np.minimum(locations, len(occupied)-1)] == other
        source, target = np.flatnonzero(valid), locations[valid]
        single = (counts[source] == 1) & (counts[target] == 1)
        a = occupancy[starts[source[single]], 1]
        b = occupancy[starts[target[single]], 1]
        same = domains[a] == domains[b]
        if np.any(same):
            links.append(np.column_stack((a[same], b[same])))
        for ai, bi in zip(source[~single], target[~single]):
            join(occupancy[starts[ai]:starts[ai]+counts[ai], 1],
                 occupancy[starts[bi]:starts[bi]+counts[bi], 1])
    return _unique_edges(links)


def _group_new_patches(ids, centres, patch_owner, parcel_indices, inverse, cells,
                       new, width, height, budget):
    """Separate disconnected islands even when they occupy one coarse cell."""
    slots = np.flatnonzero(new)
    sites = assign_cells(centres[slots], budget)
    _, preliminary = np.unique(np.column_stack((patch_owner[slots], sites)), axis=0,
                               return_inverse=True)
    # Only new patches participate in their reference-connectivity grouping.
    selected = new[inverse]
    remap = np.full(len(ids), -1, np.int64)
    remap[slots] = np.arange(len(slots))
    links = _cell_contacts(cells[parcel_indices[selected]], remap[inverse[selected]],
                           preliminary, width, height, connectivity_only=True)
    parent = np.arange(len(slots), dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in links:
        a, b = find(int(a)), find(int(b))
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = np.array([find(i) for i in range(len(slots))], np.int64)
    _, grouped = np.unique(roots, return_inverse=True)
    return slots, grouped


def _current_edges(base_edges, bases, owner_uids, xyz):
    by_base = {}
    for index, (base, uid) in enumerate(zip(bases, owner_uids)):
        by_base.setdefault(int(base), {})[int(uid)] = index
    pairs = []
    for a, b in base_edges:
        left, right = by_base.get(int(a), {}), by_base.get(int(b), {})
        for uid in left.keys() & right.keys():
            pairs.append((left[uid], right[uid]))
    if not pairs:
        return np.empty((0, 2), np.int32)
    edges = np.unique(np.sort(np.asarray(pairs, np.int32), axis=1), axis=0)
    # Separate overlapping columns can have coincident aggregate centers. They
    # carry their own material, but a zero-length spring is not a valid edge.
    sine = np.linalg.norm(np.cross(xyz[edges[:, 0]], xyz[edges[:, 1]]), axis=1)
    return edges[sine >= 1e-10]


def refresh(s):
    """Return current material nodes and persistent reference contact edges.

    Fields: xyz, owners (slots), owner_uids, bases, area (reference km²), craton
    fraction, suture, thickness/reference_thickness (km), heat (m), parcel_node,
    trace_node, edges, edge_bases and topology_revision. Juvenile kind-3 parcels
    are -1 until they are accreted into continental material. The configured
    mechanics_nodes is a global sampling target, not a strict maximum: owner
    boundaries and disconnected islands must retain independent memberships.

    ``s.rift_material`` holds only checkpoint-safe dictionaries/scalars/arrays.
    A patch's base ID is assigned once; a node is (base ID, current owner UID),
    so a partial ownership transfer does not erase either side's inheritance.
    """
    if hasattr(s, 'material_surface'):
        from native_rift_material import refresh as native_refresh
        return native_refresh(s)
    budget = int(s.config.get('mechanics_nodes', 1024))
    state = getattr(s, 'rift_material', None)
    if state is None:
        state = dict(version=1, target_nodes=budget, next_base_id=0,
                     patch_ids=np.empty(0, np.int64), patch_bases=np.empty(0, np.int64),
                     base_edges=np.empty((0, 2), np.int64), topology_signature='',
                     topology_revision=0)
        s.rift_material = state
    if int(state['target_nodes']) != budget:
        raise ValueError('A running material mesh cannot change its reference node budget.')
    eligible = np.flatnonzero(((s.kind == 1) | (s.kind == 2)) & (s.mass > 0))
    ids, centres, inverse = patch_centres(s.pos[eligible], s.mass[eligible], s.parcel_patch[eligible])
    if len(ids):
        _, first = np.unique(inverse, return_index=True)
        patch_slots = s.parcel_plate[eligible[first]]
        if np.any(s.parcel_plate[eligible] != patch_slots[inverse]):
            raise ValueError('A continental material patch has been split between plate owners.')
        patch_owner = np.asarray(s.plate_uid[patch_slots], np.int64)
    else:
        patch_slots = np.empty(0, np.int32)
        patch_owner = np.empty(0, np.int64)
    known_ids = state['patch_ids']
    locations = np.searchsorted(known_ids, ids)
    known = ((locations < len(known_ids)) &
             (known_ids[np.minimum(locations, len(known_ids)-1)] == ids)) if len(known_ids) else np.zeros(len(ids), bool)
    patch_bases = np.full(len(ids), -1, np.int64)
    patch_bases[known] = state['patch_bases'][locations[known]]
    cells = np.asarray(s.parcel_cell) if hasattr(s, 'parcel_cell') and len(s.parcel_cell) == len(s.mass) else s._indices(s.pos)
    new = ~known
    if np.any(new):
        slots, grouping = _group_new_patches(ids, centres, patch_owner, eligible, inverse, cells,
                                             new, s.w, s.h, budget)
        patch_bases[slots] = int(state['next_base_id'])+grouping
        state['next_base_id'] += int(grouping.max())+1
        all_ids = np.concatenate((known_ids, ids[new]))
        all_bases = np.concatenate((state['patch_bases'], patch_bases[new]))
        order = np.argsort(all_ids)
        state['patch_ids'], state['patch_bases'] = all_ids[order], all_bases[order]
    signature = hashlib.blake2b(digest_size=16)
    signature.update(np.ascontiguousarray(ids).view(np.uint8))
    signature.update(np.ascontiguousarray(patch_owner).view(np.uint8))
    topology_signature = signature.hexdigest()
    if topology_signature != state['topology_signature']:
        # Contact is inferred from actual occupied material cells, never from
        # nearest continental centroids across intervening empty ocean.
        # One base can currently span multiple owners. Contact memberships are
        # therefore base/owner pairs, and only same-owner pairs can be joined.
        keys, contact_member = np.unique(np.column_stack((patch_bases, patch_owner)), axis=0,
                                         return_inverse=True)
        contact = _cell_contacts(cells[eligible], contact_member[inverse], keys[:, 1], s.w, s.h)
        added = keys[contact, 0] if len(contact) else np.empty((0, 2), np.int64)
        state['base_edges'] = _unique_edges([state['base_edges'], added])
        state['topology_signature'] = topology_signature
        state['topology_revision'] += 1
    keys, patch_node = np.unique(np.column_stack((patch_bases, patch_owner)), axis=0, return_inverse=True)
    count = len(keys)
    parcel_node = np.full(len(s.mass), -1, np.int32)
    parcel_node[eligible] = patch_node[inverse]
    selected_nodes = parcel_node[eligible]
    weights = s.mass[eligible]
    area = np.bincount(selected_nodes, weights=weights, minlength=count)

    def mean(value):
        return np.bincount(selected_nodes, weights=weights*np.asarray(value)[eligible], minlength=count)/area

    xyz = np.column_stack([mean(s.pos[:, axis]) for axis in range(3)])
    length = np.linalg.norm(xyz, axis=1)
    if np.any(length < 1e-10):
        raise ValueError('A persistent material node no longer has a local spherical centroid.')
    xyz /= length[:, None]
    owners = np.empty(count, np.int32)
    if count:
        _, first_patch = np.unique(patch_node, return_index=True)
        owners[:] = patch_slots[first_patch]
    trace_node = np.full(len(s.trace_patch), -1, np.int32)
    if len(ids) and len(trace_node):
        at = np.searchsorted(ids, s.trace_patch)
        valid = (at < len(ids)) & (ids[np.minimum(at, len(ids)-1)] == s.trace_patch)
        index = np.flatnonzero(valid)
        valid[index] &= (s.trace_kind[index] != 3) & (s.plate_uid[s.trace_plate[index]] == patch_owner[at[index]])
        trace_node[valid] = patch_node[at[valid]]
    edges = _current_edges(state['base_edges'], keys[:, 0], keys[:, 1], xyz)
    structure = s.structure
    result = dict(xyz=xyz, owners=owners, owner_uids=keys[:, 1], bases=keys[:, 0],
                  area=area, craton=mean(s.kind == 2), suture=mean(s.suture),
                  thickness=mean(structure['thickness_km']),
                  reference_thickness=mean(structure['reference_thickness_km']),
                  heat=mean(structure['rift_heat_m']), parcel_node=parcel_node,
                  trace_node=trace_node, edges=edges, edge_bases=keys[edges, 0],
                  topology_revision=int(state['topology_revision']))
    return result
