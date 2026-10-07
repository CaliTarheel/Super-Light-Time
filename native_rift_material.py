"""Persistent continental mechanics grouped from shared material triangle edges.

The moving connected surface is authoritative. A native control cell, a nearby
centroid or overlapping rendered material cannot create a mechanical contact.
Grouping into coarse nodes is an acceleration of this actual material graph;
the returned arrays retain the established progressive-rifting schema.
"""
from __future__ import annotations
import hashlib
import numpy as np

from mesh_geometry import connected_components
from rift_mesh import assign_cells


def shared_face_edges(faces):
    """Return face pairs sharing a full indexed material edge, not just a point."""
    triangles = np.asarray(faces)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in 'iu':
        raise ValueError('Material contact requires integer triangle vertex indices.')
    if not len(triangles):
        return np.empty((0, 2), np.int32)
    sides = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1)
    _, inverse, counts = np.unique(sides, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('More than two material faces share an indexed edge.')
    order = np.argsort(inverse, kind='stable')
    starts = np.r_[0, np.cumsum(counts)]
    paired = starts[:-1][counts == 2]
    face_index = np.tile(np.arange(len(triangles)), 3)
    result = np.column_stack((face_index[order[paired]], face_index[order[paired+1]]))
    return np.unique(np.sort(result, axis=1), axis=0).astype(np.int32)


def refresh(s):
    surface = s.material_surface
    faces = np.asarray(surface['faces'])
    ids = np.asarray(surface['face_id'], np.int64)
    owner = np.asarray(s.parcel_plate)
    n = len(ids)
    if (faces.shape != (n, 3) or len(np.unique(ids)) != n
            or not np.array_equal(ids, s.parcel_patch)
            or not np.array_equal(surface['face_owner'], owner)
            or not np.array_equal(surface['face_kind'], s.kind)
            or s.pos.shape != (n, 3) or np.asarray(s.mass).shape != (n,)
            or not np.isfinite(s.mass).all() or np.any(s.mass <= 0)):
        raise ValueError('Native material triangles, owners, kinds and one-column-per-face IDs must align.')
    budget = int(s.config.get('mechanics_nodes', 1024))
    state = getattr(s, 'rift_material', None)
    if state is None:
        state = dict(version=2, backend='native_material_triangles', target_nodes=budget,
                     next_base_id=0, patch_ids=np.empty(0, np.int64), patch_bases=np.empty(0, np.int64),
                     base_edges=np.empty((0, 2), np.int64), topology_signature='', topology_revision=0,
                     face_contacts=np.empty((0, 2), np.int32))
        s.rift_material = state
    if state.get('backend') != 'native_material_triangles':
        raise ValueError('A native material surface cannot reuse a raster material-reference graph.')
    if state['target_nodes'] != budget:
        raise ValueError('A running native material mesh cannot change its reference node budget.')
    eligible = ((s.kind == 1) | (s.kind == 2)) & (s.mass > 0)
    owner_uid = np.asarray(s.plate_uid[owner], np.int64)
    signature = hashlib.blake2b(digest_size=16)
    for value in (ids, faces, owner_uid, eligible):
        signature.update(np.ascontiguousarray(value).view(np.uint8))
    signature = signature.hexdigest()
    changed = signature != state['topology_signature']
    if changed:
        contact = shared_face_edges(faces)
        a, b = contact.T
        state['face_contacts'] = contact[eligible[a] & eligible[b] & (owner_uid[a] == owner_uid[b])]
        state['topology_signature'] = signature
        state['topology_revision'] += 1
    contact = state['face_contacts']

    known_ids = state['patch_ids']
    at = np.searchsorted(known_ids, ids)
    known = ((at < len(known_ids)) & (known_ids[np.minimum(at, len(known_ids)-1)] == ids)) if len(known_ids) else np.zeros(n, bool)
    bases = np.full(n, -1, np.int64)
    bases[known & eligible] = state['patch_bases'][at[known & eligible]]
    new = eligible & ~known
    if np.any(new):
        sites = np.full(n, -1, np.int64)
        sites[new] = assign_cells(s.pos[new], budget)
        keys = np.column_stack((owner_uid, sites))
        _, preliminary = np.unique(keys, axis=0, return_inverse=True)
        labels, count = connected_components(preliminary, contact, mask=new)
        bases[new] = int(state['next_base_id'])+labels[new]
        state['next_base_id'] += count
        all_ids = np.r_[known_ids, ids[new]]
        all_bases = np.r_[state['patch_bases'], bases[new]]
        order = np.argsort(all_ids)
        state['patch_ids'], state['patch_bases'] = all_ids[order], all_bases[order]

    # A later ownership transfer can separate two pieces of one old coarse
    # node. Reusing its base/owner key would silently reconnect them after a
    # motion-owner reunion. Refine that reference group once, retaining parent
    # IDs so the existing bond history can follow the material subdivision.
    if changed and np.any(eligible):
        _, domains = np.unique(np.column_stack((bases, owner_uid)), axis=0, return_inverse=True)
        labels, _ = connected_components(domains, contact, mask=eligible)
        for domain in np.unique(domains[eligible]):
            selected = np.flatnonzero(eligible & (domains == domain))
            fragments = np.unique(labels[selected])
            if len(fragments) <= 1:
                continue
            fragments = sorted(fragments, key=lambda group: (-float(s.mass[selected[labels[selected] == group]].sum()), int(group)))
            for group in fragments[1:]:
                separated = selected[labels[selected] == group]
                old_base = int(bases[separated[0]])
                new_base = int(state['next_base_id']); state['next_base_id'] += 1
                state.setdefault('base_parent', {})[new_base] = old_base
                bases[separated] = new_base
                positions = np.searchsorted(state['patch_ids'], ids[separated])
                state['patch_bases'][positions] = new_base

    keys, inverse = np.unique(np.column_stack((bases[eligible], owner_uid[eligible])), axis=0, return_inverse=True)
    count = len(keys)
    parcel_node = np.full(n, -1, np.int32); parcel_node[eligible] = inverse
    weights = s.mass[eligible]
    area = np.bincount(inverse, weights=weights, minlength=count)
    def mean(value):
        return np.bincount(inverse, weights=weights*np.asarray(value)[eligible], minlength=count)/area
    xyz = np.column_stack([mean(s.pos[:, axis]) for axis in range(3)])
    length = np.linalg.norm(xyz, axis=1)
    if np.any(length < 1e-10):
        raise ValueError('A native mechanical group no longer has a local material centroid.')
    xyz /= length[:, None]
    owners = np.empty(count, np.int32)
    if count:
        _, first = np.unique(inverse, return_index=True)
        owners[:] = owner[np.flatnonzero(eligible)[first]]
    if len(contact):
        edges = parcel_node[contact]
        edges = np.unique(np.sort(edges[edges[:, 0] != edges[:, 1]], axis=1), axis=0)
        sine = np.linalg.norm(np.cross(xyz[edges[:, 0]], xyz[edges[:, 1]]), axis=1)
        edges = edges[sine >= 1e-10]
    else:
        edges = np.empty((0, 2), np.int32)
    # Retain a historical base-contact catalogue for fault lineage, but only
    # CURRENT shared material edges above can transmit mechanical loading.
    if len(edges):
        state['base_edges'] = np.unique(np.sort(np.vstack((state['base_edges'], keys[edges, 0])), axis=1), axis=0)
    trace_node = np.full(len(s.trace_patch), -1, np.int32)
    if n and len(trace_node):
        order = np.argsort(ids); sorted_ids = ids[order]
        locations = np.searchsorted(sorted_ids, s.trace_patch)
        valid = (locations < n) & (sorted_ids[np.minimum(locations, n-1)] == s.trace_patch)
        indices = np.flatnonzero(valid)
        material = order[locations[valid]]
        matched = ((s.trace_kind[indices] != 3) & eligible[material]
                   & (s.plate_uid[s.trace_plate[indices]] == owner_uid[material]))
        trace_node[indices[matched]] = parcel_node[material[matched]]
    structure = s.structure
    return dict(xyz=xyz, owners=owners, owner_uids=keys[:, 1], bases=keys[:, 0], area=area,
                craton=mean(s.kind == 2), suture=mean(s.suture), thickness=mean(structure['thickness_km']),
                reference_thickness=mean(structure['reference_thickness_km']), heat=mean(structure['rift_heat_m']),
                parcel_node=parcel_node, trace_node=trace_node, edges=edges, edge_bases=keys[edges, 0],
                topology_revision=int(state['topology_revision']))
