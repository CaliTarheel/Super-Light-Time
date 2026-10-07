"""Bounded-memory spherical material graphs for continental deformation.

Nodes are persistent groups of material patches, not map pixels.  Membership is
assigned once in the material's reference configuration; rotating a plate must
move its existing nodes, rather than reassigning them to geographic cells.
These helpers infer no stress, create no plates and change no material state.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

EARTH_RADIUS_KM = 6371.0
_WORK_BYTES = 32 * 1024 * 1024


def _points(xyz):
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Material positions must have shape (N, 3).")
    lengths = np.linalg.norm(xyz, axis=1)
    if not np.all(np.isfinite(xyz)) or np.any(lengths <= 0):
        raise ValueError("Material positions must be finite, nonzero vectors.")
    return xyz / lengths[:, None]


@lru_cache(maxsize=8)
def _sites(budget):
    i = np.arange(budget, dtype=float)
    z = 1.0 - 2.0 * (i + 0.5) / budget
    angle = i * (np.pi * (3.0 - np.sqrt(5.0)))
    radius = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    sites = np.column_stack((radius*np.cos(angle), radius*np.sin(angle), z))
    sites.flags.writeable = False
    return sites


def fibonacci_sites(budget):
    """Return a copy of the deterministic unit-sphere reference sites."""
    if isinstance(budget, bool) or int(budget) != budget or budget < 4:
        raise ValueError("The material mesh budget must be an integer >= 4.")
    return _sites(int(budget)).copy()


def assign_cells(xyz, budget):
    """Assign points to their exact nearest Fibonacci reference site.

    Work arrays are bounded independently of the number of material patches.
    A cheap latitude-strip search supplies a distance bound; a second search
    includes EVERY site that can improve it, using |delta z| <= chord length.
    This avoids an N-by-budget global distance matrix, without an approximate
    neighbor lookup.  Cell IDs belong to the reference configuration, not the
    subsequently advected geographic locations.
    """
    sites = fibonacci_sites(budget)
    budget = int(budget)
    xyz = _points(xyz)
    output = np.empty(len(xyz), np.int32)
    half = min(budget, int(np.ceil(np.sqrt(budget))))
    offsets = np.arange(-half, half+1)
    # Three vector components plus distance, index and mask temporaries.
    batch = max(1, min(2048, _WORK_BYTES // (64 * len(offsets))))
    for start in range(0, len(xyz), batch):
        points = xyz[start:start+batch]
        centre = np.rint((1-points[:, 2])*budget*.5-.5).astype(np.int64)
        candidates = np.clip(centre[:, None]+offsets, 0, budget-1)
        delta = sites[candidates]-points[:, None, :]
        distance = np.einsum('ijk,ijk->ij', delta, delta)
        best = np.sqrt(np.min(distance, axis=1))
        lower = np.maximum(0, np.ceil((1-points[:, 2]-best)*budget*.5-.5-1e-10)).astype(np.int64)
        upper = np.minimum(budget-1, np.floor((1-points[:, 2]+best)*budget*.5-.5+1e-10)).astype(np.int64)
        # Sub-batch again if a pathological query makes the proven strip large.
        width = int(np.max(upper-lower+1))
        exact_batch = max(1, _WORK_BYTES // (64 * max(1, width)))
        for offset in range(0, len(points), exact_batch):
            stop = min(len(points), offset+exact_batch)
            indexes = lower[offset:stop, None]+np.arange(width)
            valid = indexes <= upper[offset:stop, None]
            indexes = np.minimum(indexes, budget-1)
            delta = sites[indexes]-points[offset:stop, None, :]
            distance = np.einsum('ijk,ijk->ij', delta, delta)
            distance[~valid] = np.inf
            selected = np.argmin(distance, axis=1)
            output[start+offset:start+stop] = indexes[np.arange(stop-offset), selected]
    return output


def _edge_array(edges, node_count):
    edges = np.asarray(edges)
    if edges.size == 0:
        return np.empty((0, 2), np.int32)
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("Graph edges must be integer node-index pairs.")
    if np.any(edges < 0) or np.any(edges >= node_count):
        raise ValueError("A graph edge references a missing material node.")
    edges = np.sort(edges.astype(np.int32), axis=1)
    return np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)


def build_graph(xyz, owners, area, craton, *, neighbors=6, support_edges=None,
                radius_km=EARTH_RADIUS_KM, reach=1.5):
    """Return nearby undirected links between material nodes of the same owner.

    ``area`` is each node's represented reference area in km².  Each link must
    lie within ``reach * (sqrt(area_i/pi) + sqrt(area_j/pi))`` km.  This finite
    support cap prevents nearest-neighbor links from crossing a wide empty
    ocean to a remote island.  Distances are spherical, including at the seam
    and poles.  Cratons remain connected mechanically; strength belongs in the
    loading solver, not in graph construction.

    Area/centroid inference cannot certify contact across a narrow strait.
    When available, ``support_edges`` is the authoritative candidate set from
    actual touching material footprints; those candidates still pass the
    owner and distance checks, but are not truncated to ``neighbors``.

    Without explicit support, complexity is O(sum(owner_node_count²)) only at
    graph construction, with bounded distance buffers.  Retain the resulting
    material links while a rigid plate rotates; do not rebuild every timestep.
    """
    xyz = _points(xyz)
    owners, area, craton = np.asarray(owners), np.asarray(area, float), np.asarray(craton)
    count = len(xyz)
    if owners.shape != (count,) or area.shape != (count,) or craton.shape != (count,):
        raise ValueError("Material graph fields must align with its nodes.")
    if not np.all(np.isfinite(area)) or np.any(area < 0):
        raise ValueError("Material node areas must be finite and nonnegative.")
    if int(neighbors) != neighbors or neighbors < 1 or not np.isfinite(reach) or reach <= 0:
        raise ValueError("Graph neighbors and finite support reach must be positive.")
    if not np.isfinite(radius_km) or radius_km <= 0:
        raise ValueError("Sphere radius must be finite and positive.")
    support_radius = np.sqrt(area/np.pi)/radius_km
    if support_edges is not None:
        edges = _edge_array(support_edges, count)
        a, b = edges.T
        sine = np.linalg.norm(np.cross(xyz[a], xyz[b]), axis=1)
        angle = np.arctan2(sine,
                           np.einsum('ij,ij->i', xyz[a], xyz[b]))
        valid = ((owners[a] >= 0) & (owners[a] == owners[b]) & (area[a] > 0) & (area[b] > 0)
                 & (sine >= 1e-10)
                 & (angle <= reach*(support_radius[a]+support_radius[b])+1e-12))
        return edges[valid]
    pairs = []
    for owner in np.unique(owners[owners >= 0]):
        nodes = np.flatnonzero((owners == owner) & (area > 0))
        if len(nodes) < 2:
            continue
        points = xyz[nodes]
        radii = support_radius[nodes]
        k = min(int(neighbors), len(nodes)-1)
        batch = max(1, min(512, _WORK_BYTES // (32*len(nodes))))
        for start in range(0, len(nodes), batch):
            stop = min(len(nodes), start+batch)
            # Dot products are rotation invariant; no longitude/latitude bins.
            distance = np.maximum(0.0, 2.0-2.0*(points[start:stop] @ points.T))
            distance[np.arange(stop-start), np.arange(start, stop)] = np.inf
            cap = np.minimum(np.pi, reach*(radii[start:stop, None]+radii[None, :]))
            chord_cap = 2*np.sin(cap*.5)
            distance[distance > chord_cap*chord_cap+1e-12] = np.inf
            # A regular mesh has exactly equidistant neighbors.  Floating point
            # roundoff after rotation must not switch which tied link survives
            # the degree limit. Quantization is well below one millimeter at
            # the 512--8192-node global spacing, not a map grid.
            ranking = np.round(distance, 12)
            ranking += np.arange(len(nodes))[None, :] * (2e-13/len(nodes))
            nearest = np.argpartition(ranking, k-1, axis=1)[:, :k]
            rows = np.repeat(np.arange(stop-start), k)
            targets = nearest.ravel()
            valid = np.isfinite(distance[rows, targets])
            sine = np.linalg.norm(np.cross(points[start+rows], points[targets]), axis=1)
            valid &= sine >= 1e-10
            pairs.append(np.column_stack((nodes[start+rows[valid]], nodes[targets[valid]])))
    return _edge_array(np.concatenate(pairs) if pairs else [], count)


def _components(node_count, edges, owners):
    parent = np.arange(node_count, dtype=np.int32)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in edges:
        if owners[a] != owners[b]:
            continue
        a, b = find(int(a)), find(int(b))
        if a != b:
            parent[max(a, b)] = min(a, b)
    return np.array([find(i) for i in range(node_count)], np.int32)


def coherent_cut(node_count, edges, failed_edges, owners, craton_groups=None,
                 *, area=None, min_nodes=3, min_fraction=.08):
    """Propose only new, substantial component partitions behind failed links.

    The result is a list of dictionaries with ``owner``, ``parent_nodes``,
    ``components`` and ``failed_edges``.  An isolated damaged link that leaves
    an alternate intact path produces no cut.  Existing isolated islands are
    never interpreted as a new fracture.  Every component must contain at least
    ``min_nodes`` positive-area nodes and ``min_fraction`` of the parent's
    reference area. Callers using a separate spatial-resolution guard may set
    ``min_fraction=0``; zero-area shards still cannot become daughters.
    If any enduring nonnegative craton group occurs on both sides, the entire
    proposal is rejected.  Callers must still verify complete patch/craton
    membership and actual material contact before creating a plate.

    ``failed_edges`` accepts a boolean mask aligned to the supplied edges, an
    integer vector of edge indices, or explicit endpoint pairs.
    """
    owners = np.asarray(owners)
    if int(node_count) != node_count or node_count < 0 or owners.shape != (node_count,):
        raise ValueError("Node count and material owners must align.")
    if min_nodes < 1 or not 0 <= min_fraction < .5:
        raise ValueError("Cut size safeguards must require positive nodes and a fraction below one half.")
    raw_edges = np.asarray(edges)
    graph = _edge_array(raw_edges, node_count)
    failed = np.asarray(failed_edges)
    if failed.dtype == bool:
        if failed.shape != (len(raw_edges),):
            raise ValueError("Failed-edge mask must align with supplied graph edges.")
        failed = _edge_array(raw_edges[failed], node_count)
    elif failed.ndim == 1:
        if not np.issubdtype(failed.dtype, np.integer) and failed.size:
            raise ValueError("Failed-edge indices must be integers.")
        if np.any(failed < 0) or np.any(failed >= len(raw_edges)):
            raise ValueError("A failed-edge index is outside the graph.")
        failed = _edge_array(raw_edges[failed.astype(np.int64)], node_count)
    else:
        failed = _edge_array(failed, node_count)
    keys = graph[:, 0].astype(np.int64)*max(1, node_count)+graph[:, 1]
    failed_keys = failed[:, 0].astype(np.int64)*max(1, node_count)+failed[:, 1]
    removed = np.isin(keys, failed_keys)
    if not np.any(removed):
        return []
    weights = np.ones(node_count) if area is None else np.asarray(area, float)
    groups = np.full(node_count, -1, np.int64) if craton_groups is None else np.asarray(craton_groups)
    if weights.shape != (node_count,) or groups.shape != (node_count,):
        raise ValueError("Cut areas and craton groups must align with material nodes.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Cut areas must be finite and nonnegative.")
    before = _components(node_count, graph, owners)
    after = _components(node_count, graph[~removed], owners)
    results = []
    for label in np.unique(before):
        parent = np.flatnonzero(before == label).astype(np.int32)
        owner = int(owners[parent[0]])
        if owner < 0:
            continue
        labels = np.unique(after[parent])
        if len(labels) < 2:
            continue
        total = float(weights[parent].sum())
        if total <= 0:
            continue
        # An undersized daughter retires itself, not the whole partition. Rejecting
        # every component because one is too small let a single detached node veto a
        # genuine two-way split: at 268 Myr the only severing set in the run produced
        # a [63, 1] partition and was discarded whole on account of the 1. Absorb
        # such slivers into the largest admissible daughter and judge what remains.
        assign = after[parent].copy()
        member = {int(item): (after[parent] == item) for item in labels}
        def admissible(mask):
            return (np.count_nonzero(weights[parent[mask]] > 0) >= min_nodes
                    and weights[parent[mask]].sum() >= min_fraction*total)
        keepers = [int(item) for item in labels if admissible(member[int(item)])]
        if len(keepers) < 2:
            continue
        if len(keepers) < len(labels):
            host = max(keepers, key=lambda item: weights[parent[member[item]]].sum())
            for item in labels:
                if int(item) not in keepers:
                    assign[member[int(item)]] = host
        components = [parent[assign == item] for item in keepers]
        protected = groups[parent] >= 0
        conflict = any(len(np.unique(assign[groups[parent] == group])) > 1
                       for group in np.unique(groups[parent][protected]))
        if conflict:
            continue
        broken = graph[removed]
        broken = broken[(before[broken[:, 0]] == label) & (before[broken[:, 1]] == label)]
        results.append(dict(owner=owner, parent_nodes=parent, components=components,
                            failed_edges=broken.copy()))
    return results
