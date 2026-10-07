"""Local, contributor-contained birth groups within qualified source histories.

Source history supplies adjacency. Distance never establishes membership.
Every candidate is anchored at an actual source and funded only by origins
inside its spherical footprint and connected in the induced source graph.
"""
from __future__ import annotations

import numpy as np

CONTAINMENT_TOLERANCE = 2e-11


def contained(patch, points):
    """Actual minor-triangle union containment, with locator-scale roundoff."""
    vertices=np.asarray(patch['vertices'],float);faces=np.asarray(patch['faces'],int)
    inverse=np.linalg.inv(vertices[faces].transpose(0,2,1))
    points=np.asarray(points,float);inside=np.zeros(len(points),bool)
    for start in range(0,len(points),4096):
        weights=np.einsum('fij,qj->qfi',inverse,points[start:start+4096])
        inside[start:start+4096]=np.any(np.all(weights>=-CONTAINMENT_TOLERANCE,axis=2),axis=1)
    return inside


def connected(indices, links, anchor):
    selected=set(map(int,indices))
    if int(anchor) not in selected:return np.empty(0,int)
    if links is None:return np.asarray(sorted(selected),int)
    neighbors={i:[] for i in selected}
    for a,b in links:
        a,b=int(a),int(b)
        if a in selected and b in selected:
            neighbors[a].append(b);neighbors[b].append(a)
    reached={int(anchor)};todo=[int(anchor)]
    while todo:
        for other in neighbors[todo.pop()]:
            if other not in reached:reached.add(other);todo.append(other)
    return np.asarray(sorted(reached),int)


def _adjacency(indices, links):
    # The source graph is invariant throughout this proposal search. Build it
    # once, preserving link multiplicity and input order.
    if links is None:
        return None
    neighbors = {i: [] for i in indices}
    for a, b in links:
        a, b = int(a), int(b)
        if a in neighbors and b in neighbors:
            neighbors[a].append(b)
            neighbors[b].append(a)
    return neighbors


def _connected(selected, neighbors, anchor):
    selected = selected if isinstance(selected, set) else set(map(int, selected))
    if anchor not in selected:
        return np.empty(0, int)
    if neighbors is None:
        return np.asarray(sorted(selected), int)
    reached = {anchor}
    todo = [anchor]
    while todo:
        for other in neighbors[todo.pop()]:
            if other in selected and other not in reached:
                reached.add(other)
                todo.append(other)
    return np.asarray(sorted(reached), int)


def _components(remaining, neighbors):
    # Each remaining origin receives the same sorted component the original
    # per-anchor traversal would return. Recompute after an accepted removal.
    unseen = set(remaining)
    result = {}
    while unseen:
        component = _connected(remaining, neighbors, next(iter(unseen)))
        for origin in component:
            result[int(origin)] = component
        unseen.difference_update(component)
    return result


def local_groups(points, areas, group, factory, capacity):
    """Partition a history component into geometrically funded local proposals.

    ``factory(anchor_index, summed_area)`` retains an actual source as centre.
    ``capacity(patch, anchor_index, summed_area)`` is the physical profile test.
    Pruning is monotone in source membership, so no source funds a footprint
    after it has been excluded. The search is conservative, not a guarantee of
    globally optimal volcanic plumbing or maximum source utilization.
    """
    points=np.asarray(points,float);areas=np.asarray(areas,float)
    indices=list(map(int,group['indices']));links=group.get('links')
    preferred=int(group['anchor_index'])
    centre=np.sum(points[indices]*areas[indices,None],axis=0)
    centre/=max(np.linalg.norm(centre),1e-30)
    # This only orders ACTUAL candidate anchors; it never defines membership.
    # A central existing source reduces arbitrary endpoint shifts when one
    # physical finite source trace is subdivided into more observations.
    first_position = {}
    for position, origin in enumerate(indices):
        first_position.setdefault(origin, position)
    order=sorted(indices,key=lambda i:(-float(points[i]@centre),i!=preferred,first_position[i]))
    neighbors = _adjacency(indices, links)
    remaining=set(indices);result=[]
    while remaining:
        components = _components(remaining, neighbors)
        chosen=None;last=None
        for anchor in order:
            if anchor not in remaining:continue
            take=components[anchor]
            while len(take):
                amount=float(areas[take].sum())
                patch=factory(anchor,amount)
                inside = contained(patch,points[take])
                # take is connected already; keeping every origin cannot
                # change its induced component. Preserve every geometry query.
                subset = take if bool(np.all(inside)) else _connected(take[inside], neighbors, anchor)
                if len(subset)==len(take):break
                take=subset
            if not len(take):continue
            # The exact final set funds the exact final footprint. There is no
            # neighbour distance bin and no credit from an excluded source.
            last=dict(indices=[anchor]+[int(i) for i in take if i!=anchor],anchor_index=anchor,links=links)
            if capacity(patch,anchor,float(areas[take].sum())):
                chosen=last;break
        if chosen is None:
            # These sources retain their full amounts. A later observation may
            # supply enough volume; unrelated groups cannot borrow the balance.
            result.extend(dict(indices=[i],anchor_index=i,links=[]) for i in order if i in remaining)
            break
        result.append(chosen);remaining.difference_update(chosen['indices'])
    return result
