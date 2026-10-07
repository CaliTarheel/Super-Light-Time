"""Explicit, conservative initial ocean plates and inherited ridge cooling ages.

The already fitted artwork is immutable geometry. Only whole water triangles
change owner; their shared edges are the declared ridge network. Connected
shortest-path cells avoid cutting or refitting any continental material. The
specified half spreading rate defines INITIAL AGE, never a plate velocity.
"""
from __future__ import annotations

from copy import deepcopy
import heapq
import numpy as np

VERSION = 1
DEFAULTS = dict(enabled=False, plate_count=4, half_spreading_rate_cm_yr=2., maximum_age_myr=180.)
RADIUS_KM = 6371.


def normalize(value=None):
    if value is None:
        return dict(DEFAULTS)
    if not isinstance(value, dict) or set(value)-set(DEFAULTS)-{'continental_attachments'}:
        raise ValueError('Invalid primordial ocean settings; continental_attachments is an optional explicit region-to-plate list.')
    result = dict(DEFAULTS, **value)
    if type(result['enabled']) is not bool:
        raise ValueError('Primordial ocean enabled must be boolean.')
    for name, low, high in (('plate_count', 2, 8), ('half_spreading_rate_cm_yr', .1, 20.),
                            ('maximum_age_myr', 1., 300.)):
        number = result[name]
        if (isinstance(number, (bool, np.bool_)) or not isinstance(number, (int, float, np.number))
                or not np.isscalar(number) or not np.isfinite(number) or not low <= number <= high
                or (name == 'plate_count' and number != int(number))):
            raise ValueError(f'Primordial ocean {name} must be finite and between {low} and {high}'
                             + (' and an integer.' if name == 'plate_count' else '.'))
        result[name] = int(number) if name == 'plate_count' else float(number)
    attachments = result.get('continental_attachments', [])
    if not isinstance(attachments, list):
        raise ValueError('Primordial ocean continental_attachments must be a list.')
    seen_regions, seen_uids, normalized = set(), set(), []
    for entry in attachments:
        if not isinstance(entry, dict) or set(entry) != {'region_index', 'continental_plate_uid'}:
            raise ValueError('Each continental attachment requires region_index and continental_plate_uid.')
        region, uid = entry['region_index'], entry['continental_plate_uid']
        if (isinstance(region, (bool, np.bool_)) or not isinstance(region, (int, np.integer))
                or not 0 <= region < result['plate_count']
                or isinstance(uid, (bool, np.bool_)) or not isinstance(uid, (int, np.integer)) or uid <= 0):
            raise ValueError('Attachment region_index must identify a zero-based water region and continental_plate_uid a positive integer.')
        if region in seen_regions or uid in seen_uids:
            raise ValueError('Continental attachments must have distinct regions and distinct continental plate UIDs.')
        seen_regions.add(region); seen_uids.add(uid)
        normalized.append(dict(region_index=int(region), continental_plate_uid=int(uid)))
    if attachments and not result['enabled']:
        raise ValueError('Continental attachments require primordial ocean initialization to be enabled.')
    if len(attachments) >= result['plate_count']:
        raise ValueError('Continental attachments must leave at least one independent primordial ocean region.')
    if 'continental_attachments' in result:
        result['continental_attachments'] = sorted(normalized, key=lambda entry: entry['region_index'])
    return result


def attachment_slots(s, fitted, partition, attachments):
    """Resolve explicit UID choices and require each resulting mixed plate connected.

    Region indices refer to the deterministic partition of this exact source
    world and plate count. A changed source must be reviewed as a new initial
    condition; indices are not persistent geographical names.
    """
    if not attachments:
        return {}
    faces = np.asarray(fitted['faces'])
    owners, kinds = np.asarray(fitted['face_owner']), np.asarray(fitted['face_kind'])
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts != 2):
        raise ValueError('Continental attachments require a closed conforming source sphere.')
    adjacent = np.argsort(inverse, kind='stable').reshape(-1, 2)//3
    graph = [[] for _ in faces]
    for a, b in adjacent:
        graph[a].append(int(b)); graph[b].append(int(a))
    result = {}
    for entry in attachments:
        region, uid = entry['region_index'], entry['continental_plate_uid']
        slots = np.flatnonzero(s.active & (s.plate_uid == uid))
        if len(slots) != 1:
            raise ValueError(f'Continental attachment UID {uid} is not an active plate in this starting world.')
        slot = int(slots[0])
        land = (owners == slot) & (kinds > 0)
        water = partition['face_labels'] == region
        if not np.any(land):
            raise ValueError(f'Continental attachment UID {uid} carries no continental source material.')
        left, right = adjacent.T
        coast = (water[left] & land[right]) | (land[left] & water[right])
        if not np.any(coast):
            raise ValueError(f'Water region {region} has no shared coast with continental plate UID {uid}.')
        selected = (owners == slot) | water
        start = int(np.flatnonzero(water)[0])
        seen, todo = {start}, [start]
        while todo:
            for other in graph[todo.pop()]:
                if selected[other] and other not in seen:
                    seen.add(other); todo.append(other)
        if len(seen) != int(selected.sum()):
            raise ValueError(f'Attachment of water region {region} leaves plate UID {uid} disconnected.')
        result[region] = slot
    return result


def _unit(points):
    points = np.asarray(points, float)
    return points/np.maximum(np.linalg.norm(points, axis=-1, keepdims=True), 1e-30)


def _arc(first, second):
    return RADIUS_KM*np.arctan2(np.linalg.norm(np.cross(first, second), axis=-1),
                              np.sum(first*second, axis=-1))


def _shortest_paths(adjacency, seeds):
    """Deterministic graph distances and seed labels, including exact ties."""
    distance = np.full(len(adjacency), np.inf)
    owner = np.full(len(adjacency), -1, np.int32)
    queue = []
    for node, offset, label in seeds:
        node, label, offset = int(node), int(label), float(offset)
        if offset < distance[node] or (offset == distance[node] and label < owner[node]):
            distance[node], owner[node] = offset, label
            heapq.heappush(queue, (offset, label, node))
    while queue:
        value, label, node = heapq.heappop(queue)
        if value != distance[node] or label != owner[node]:
            continue
        for neighbor, length in adjacency[node]:
            candidate = value+length
            if candidate < distance[neighbor] or (candidate == distance[neighbor] and label < owner[neighbor]):
                distance[neighbor], owner[neighbor] = candidate, label
                heapq.heappush(queue, (candidate, label, neighbor))
    return distance, owner


def partition_water(fitted, count):
    """Return connected water labels and ridge distances without changing input.

    Distances follow the water-face dual graph, with arc lengths through shared
    edge midpoints. Every disconnected basin receives a seed before additional
    farthest-point seeds are placed. Ridge cooling distances stay in their own
    ocean plate; they cannot take a shortcut across a continent or another plate.
    """
    vertices, faces = np.asarray(fitted['vertices']), np.asarray(fitted['faces'])
    water = np.asarray(fitted['face_kind']) == 0
    ids = np.flatnonzero(water)
    if len(ids) < count:
        raise ValueError('Too few fitted water faces for the requested initial ocean plate count.')
    centers = _unit(vertices[faces].sum(axis=1))
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    unique, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts != 2):
        raise ValueError('Initial ocean partition requires a closed conforming source sphere.')
    adjacent = np.argsort(inverse, kind='stable').reshape(-1, 2)//3
    left, right = adjacent.T
    local = np.full(len(faces), -1, np.int32)
    local[ids] = np.arange(len(ids))
    middle = _unit(vertices[unique].sum(axis=1))
    first = _arc(centers[left], middle)
    second = _arc(centers[right], middle)
    shared = water[left] & water[right]
    graph = [[] for _ in ids]
    for a, b, length in zip(local[left[shared]], local[right[shared]], (first+second)[shared]):
        if not np.isfinite(length) or length <= 0.:
            raise ValueError('Initial ocean partition has an invalid water adjacency distance.')
        graph[a].append((int(b), float(length)))
        graph[b].append((int(a), float(length)))
    components = []
    seen = np.zeros(len(ids), bool)
    for origin in range(len(ids)):
        if seen[origin]:
            continue
        todo, component = [origin], []
        seen[origin] = True
        while todo:
            node = todo.pop()
            component.append(node)
            for neighbor, _ in graph[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    todo.append(neighbor)
        components.append(np.sort(component))
    if len(components) > count:
        raise ValueError(f'The starting ocean has {len(components)} disconnected water components; '
                         f'primordial_ocean.plate_count={count} cannot seed every component.')
    areas = np.asarray(fitted['area_km2'])[ids]
    components.sort(key=lambda component: (-float(areas[component].sum()), int(component[0])))
    coast = water[left] != water[right]
    coast_nodes = np.where(water[left[coast]], local[left[coast]], local[right[coast]])
    coast_distance = np.where(water[left[coast]], first[coast], second[coast])
    inland, _ = _shortest_paths(graph, zip(coast_nodes, coast_distance, np.zeros(len(coast_nodes), int)))
    seeds = []
    for component in components:
        # Largest distance from the existing coast starts in a basin interior.
        # A wholly oceanic sphere has no coast: use a fixed Cartesian direction.
        score = inland[component]
        chosen = (component[np.argmax(score)] if np.isfinite(score).all()
                  else component[np.argmax(centers[ids[component], 0])])
        seeds.append(int(chosen))
    while True:
        nearest, labels = _shortest_paths(graph, ((node, 0., i) for i, node in enumerate(seeds)))
        if not np.isfinite(nearest).all():
            raise ValueError('An initial water component was left without a plate seed.')
        if len(seeds) == count:
            break
        candidate = int(np.argmax(nearest))
        if nearest[candidate] <= 0.:
            raise ValueError('The requested initial ocean plates cannot be represented separately.')
        seeds.append(candidate)
    face_labels = np.full(len(faces), -1, np.int16)
    face_labels[ids] = labels
    ridge = shared & (face_labels[left] != face_labels[right])
    if not np.any(ridge):
        raise ValueError('The requested ocean partition has no ocean-ocean ridge; increase the plate count.')
    # A sampled cell centre is older than its zero-age ridge edge. Do not reset
    # a whole triangle to zero or invent elapsed time/runtime spreading.
    age_graph = [[(other, length) for other, length in neighbors if labels[other] == labels[node]]
                 for node, neighbors in enumerate(graph)]
    age_seeds = [(int(local[a]), float(distance), 0)
                 for side, distances in ((left, first), (right, second))
                 for a, distance in zip(side[ridge], distances[ridge])]
    cooling_distance, _ = _shortest_paths(age_graph, age_seeds)
    return dict(face_labels=face_labels, water_faces=ids, seed_faces=ids[np.asarray(seeds)],
                cooling_distance_km=cooling_distance, component_count=len(components),
                ridge_vertex_indices=unique[ridge], ridge_face_indices=adjacent[ridge])


def initialize(s, fitted):
    """Allocate the explicit initial identities before support and material setup."""
    settings = normalize(s.config.get('primordial_ocean'))
    if not settings['enabled']:
        return fitted
    if s.t != 0. or s.steps != 0 or getattr(s, 'primordial_ocean_version', 0):
        raise ValueError('Primordial ocean plates are only initialized once in a fresh world.')
    original_uid = getattr(s, 'initial_ocean_plate_uid', None)
    parent = np.flatnonzero(s.active & (s.plate_uid == original_uid)) if original_uid is not None else []
    if len(parent) != 1:
        raise ValueError('Initial ocean splitting requires one identifiable source ocean plate.')
    parent = int(parent[0])
    partition = partition_water(fitted, settings['plate_count'])
    water = partition['water_faces']
    if np.any(np.asarray(fitted['face_owner'])[water] != parent):
        raise ValueError('Initial ocean splitting cannot reinterpret a world with multiple pre-existing ocean owners.')
    attachments = attachment_slots(s, fitted, partition, settings.get('continental_attachments', []))
    independent = [i for i in range(settings['plate_count']) if i not in attachments]
    new_count = len(independent)-1
    free = np.flatnonzero(~s.active)
    if len(free) < new_count:
        s._ensure_plate_capacity(s.capacity+new_count-len(free))
        free = np.flatnonzero(~s.active)
    if len(free) < new_count:
        raise ValueError('Insufficient plate capacity for the requested primordial ocean partition.')
    pure_slots = np.r_[parent, free[:new_count]].astype(np.int16)
    slots = np.empty(settings['plate_count'], np.int16)
    slots[independent] = pure_slots
    for region, slot in attachments.items():
        slots[region] = slot
    result = dict(fitted, face_owner=np.asarray(fitted['face_owner']).copy())
    result['face_owner'][water] = slots[partition['face_labels'][water]]
    centers = _unit(np.asarray(fitted['vertices'])[np.asarray(fitted['faces'])].sum(axis=1))
    for i, slot in enumerate(slots):
        if i not in attachments and slot != parent:
            s._new_plate_identity(int(slot), parent)
            s.active[slot] = True
            # Inherit the same provisional rigid rotation, adding no relative
            # velocity. reviewed_v1 subsequently solves all rotations from forces.
            s.omega[slot] = s.omega[parent]
            s.mantle[slot] = s.mantle[parent]
            s.polarity[slot, :] = s.polarity[:, slot] = -1
            s.collision_clock[slot, :] = s.collision_clock[:, slot] = 0.
            s.rift_clock[slot] = s.ocean_rift_clock[slot] = s.born[slot] = 0.
        selected = result['face_owner'] == slot
        center = _unit(np.sum(centers[selected]*np.asarray(fitted['area_km2'])[selected, None], axis=0))
        s.centres[slot] = center if np.linalg.norm(center) > .5 else centers[partition['seed_faces'][i]]
        if i not in attachments:
            s.names[slot] = f'Primordial ocean {i+1:02d}'
    s.count = int(np.count_nonzero(s.active))
    # Legacy ocean-only guards must never mistake a mixed continent/ocean
    # carrier for an independent primordial ocean plate.
    s.initial_ocean_plate_uids = np.asarray(s.plate_uid)[pure_slots].copy()
    s.primordial_ocean_carrier_uids = np.asarray(s.plate_uid)[slots].copy()
    if attachments:
        s.primordial_connectivity_version = 1
    age = np.zeros(len(fitted['faces']))
    age[water] = np.minimum(partition['cooling_distance_km']/(settings['half_spreading_rate_cm_yr']*10.),
                            settings['maximum_age_myr'])
    result['ocean_age_myr'] = age
    endpoints = np.asarray(fitted['vertices'])[partition['ridge_vertex_indices']]
    ridge_owners = result['face_owner'][partition['ridge_face_indices']]
    if np.any(ridge_owners[:, 0] == ridge_owners[:, 1]):
        raise ValueError('Continental attachment would turn a declared ridge into an internal interface.')
    length = _arc(endpoints[:, 0], endpoints[:, 1])
    s.primordial_ocean_version = VERSION
    s.primordial_ocean_ridges = dict(start=endpoints[:, 0].copy(), end=endpoints[:, 1].copy(),
        owner_uids=np.asarray(s.plate_uid)[ridge_owners].copy(), length_km=length)
    s.primordial_ocean_diagnostics = dict(version=VERSION, enabled=True, requested_plate_count=settings['plate_count'],
        realized_plate_count=len(slots), water_region_count=len(slots), plate_count=len(pure_slots),
        initial_ocean_plate_uids=s.initial_ocean_plate_uids.tolist(),
        independent_ocean_plate_count=len(pure_slots),
        water_region_carrier_uids=s.primordial_ocean_carrier_uids.tolist(),
        continental_attachments=deepcopy(settings.get('continental_attachments', [])),
        water_components=partition['component_count'], seed_xyz=centers[partition['seed_faces']].tolist(),
        ocean_area_km2=float(np.asarray(fitted['area_km2'])[water].sum()),
        plate_area_km2=[float(np.asarray(fitted['area_km2'])[result['face_owner'] == p].sum()) for p in slots],
        ridge_segments=len(length), ridge_length_km=float(length.sum()), ridge_age_myr=0.,
        half_spreading_rate_cm_yr=settings['half_spreading_rate_cm_yr'], maximum_age_myr=settings['maximum_age_myr'],
        basins_without_ridges_are_maximum_age=bool(np.any(~np.isfinite(partition['cooling_distance_km']))),
        land_geometry_unchanged=True, runtime_ocean_created_km2=0.,
        partition='Connected graph Voronoi of existing fitted water triangles; ridge arcs follow their shared edges.',
        cooling_age='Water-only, within-plate shortest distance from zero-age ridge edges divided by the initial half spreading rate.',
        motion='Initial ages and plate identities only; no imposed daughter relative velocity or runtime rift history.',
        connectivity='Each declared attachment joins a complete connected water region to one adjacent continental plate; the shared coast is internal. At least one independent ocean region is retained.')
    for event in s.events:
        if event.get('type') == 'initial' and event.get('time_myr') == 0.:
            event['description'] = event['description'].replace('The surrounding ocean begins as one plate.',
                f'The surrounding ocean begins as {len(slots)} connected water regions on {len(pure_slots)} independent ocean plates and {len(attachments)} mixed plates, with declared ridge cooling ages.')
            event['details'].update(initial_plates=s.count, initial_ocean_plate_uids=s.initial_ocean_plate_uids.tolist(),
                water_region_carrier_uids=s.primordial_ocean_carrier_uids.tolist())
    return result


def initialize_controls(s, fitted, intersections):
    """Area-average the declared water age using the exact owner intersections."""
    if getattr(s, 'primordial_ocean_version', 0) != VERSION:
        return
    source, cell = intersections['material_index'], intersections['control_index']
    areas = np.asarray(intersections['area_km2'])*(np.asarray(fitted['face_kind'])[source] == 0)
    water_area = np.bincount(cell, weights=areas, minlength=s.n)
    age_area = np.bincount(cell, weights=areas*np.asarray(fitted['ocean_age_myr'])[source], minlength=s.n)
    wet = water_area > 1e-12
    s.age[:] = 0.
    s.age[wet] = age_area[wet]/water_area[wet]
    s.plate[:] = np.argmax(s.support, axis=0)
    carrier_uids = getattr(s, 'primordial_ocean_carrier_uids', s.initial_ocean_plate_uids)
    slots = np.flatnonzero(np.isin(s.plate_uid, carrier_uids) & s.active)
    if np.any(~np.isin(slots, s.plate)):
        raise ValueError('The native control mesh cannot resolve every initial water-region carrier; refine mesh_level or reduce plate_count.')
    expected = float(np.asarray(fitted['area_km2'])@np.asarray(fitted['ocean_age_myr']))
    represented = float(water_area@s.age)
    if abs(represented-expected) > max(1e-3, expected*1e-10):
        raise ValueError('Initial ocean cooling age was not conservatively transferred to native controls.')
    s.primordial_ocean_diagnostics.update(initial_water_age_area_myr_km2=expected,
        initial_control_water_age_area_myr_km2=represented,
        initial_ocean_age_min_myr=float(s.age[wet].min()), initial_ocean_age_max_myr=float(s.age[wet].max()),
        initial_control_age_transfer_error_myr_km2=represented-expected)
    s._record('primordial_ocean', 'Connected initial ocean plates and ridge cooling ages preserve the fitted land map.',
              details=deepcopy(s.primordial_ocean_diagnostics))


def snapshot(s):
    if getattr(s, 'primordial_ocean_version', 0) != VERSION:
        return {}
    result = dict(primordial_ocean_version=VERSION, initial_ocean_plate_uids=s.initial_ocean_plate_uids.copy(),
                  primordial_ocean_diagnostics=deepcopy(s.primordial_ocean_diagnostics))
    if hasattr(s, 'primordial_ocean_carrier_uids'):
        result['primordial_ocean_carrier_uids'] = s.primordial_ocean_carrier_uids.copy()
    if getattr(s, 'primordial_connectivity_version', 0):
        result['primordial_connectivity_version'] = s.primordial_connectivity_version
    return result
