"""Plan local, coherent terrane accretion without moving any material.

The collision memory belongs to a local pair of material patches, not merely
two moving plates. Resolved buoyant footprints establish terrane connectivity;
all samples of a patch and each protected craton remain indivisible. Sparse
unresolved arc parcels cannot form a bridge across an otherwise empty ocean.

``plan_accretions(s, dt)`` updates local contact memory and links exact native
observations to their persistent collision records. It returns at most one plan;
the engine owns all material, motion, episode and support mutations.
``remap_contact_patches`` must be called when arc coalescence replaces patches.
"""
from __future__ import annotations

import math
import hashlib
from copy import deepcopy
import numpy as np

from crust_transport import deposit
from material_geometry import patch_centres
from plate_topology import domain_components
import collision_contacts
import localized_accretion

RADIUS_KM = 6371.
MEMORY_MYR = 35.
THRESHOLD = 18.
# Historical plate-age gate. It has no recorded derivation, it blocked every
# accretion for the first 70 Myr of a world, and because rift- and back-arc-born
# plates reset their birth time, each new microplate or arc waited another
# 70 Myr before it could dock. Kept for worlds without the age policy below.
MIN_PLATE_AGE_MYR = 70.
AGE_POLICY_VERSION = 1
# Welded-stack policy 1: distinct collision sheets that overlap on one plate
# (collision_contacts' 'accreted' pairs, above the 1 km2 unordered-overlap
# floor) are one accretion body. See collision_contacts.same_owner_overlap_pairs.
WELDED_STACK_VERSION = 1


def minimum_plate_age_myr(s):
    """Youngest plate age at which a contact may transfer ownership.

    Age policy 1 (fresh reviewed worlds with root-local welding) keeps only the
    protection the gate can justify: a plate may not use engagement older than
    itself. Welding memory is keyed by plate UID and a root gains at most one
    unit of loading per Myr of full engagement, so a plate younger than THRESHOLD
    Myr cannot have matured a root through its own contacts. Older worlds keep
    their recorded 70 Myr gate.
    """
    version = getattr(s, 'accretion_age_policy_version', 0)
    if isinstance(version, (bool, np.bool_)) or version not in (0, AGE_POLICY_VERSION):
        raise ValueError('Unsupported accretion age policy.')
    if version == AGE_POLICY_VERSION:
        if not localized_accretion.enabled(s):
            raise ValueError('The accretion age policy requires root-local welding.')
        return float(localized_accretion.THRESHOLD)
    return MIN_PLATE_AGE_MYR


def welded_stack_enabled(s):
    """True when overlapping same-plate sheets form one accretion body.

    Policy 0 (every saved world without the field) joins terranes only by shared
    edges and patch/craton IDs. Policy 1 is selected by fresh reviewed native
    worlds. It reads the exact overlap ledger and root-local welding, so it is
    refused without either.
    """
    version = getattr(s, 'accretion_welded_stack_version', 0)
    # Only an integer policy, as validate_frame requires of saved frames.
    if (isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer))
            or version not in (0, WELDED_STACK_VERSION)):
        raise ValueError('Unsupported welded-stack accretion policy.')
    if version != WELDED_STACK_VERSION:
        return False
    if not localized_accretion.enabled(s):
        raise ValueError('The welded-stack accretion policy requires root-local welding.')
    if not isinstance(getattr(s, 'native_mesh', None), dict):
        raise ValueError('The welded-stack accretion policy requires native material geometry.')
    return True


def occupancy_signature(s):
    """Fingerprint every input that determines deposited per-owner occupancy.

    Array identity cannot detect in-place arc mass additions or owner changes.
    Scan contents without retaining copies, including grid geometry at seams and
    poles. Non-contiguous fixtures are copied in bounded blocks only. Class,
    relief, patch and craton fields are intentionally excluded: they do not set
    the deposited area and are evaluated afresh by the component planner.
    """
    digest = hashlib.blake2b(digest_size=32)
    if isinstance(getattr(s, 'native_mesh', None), dict):
        # Actual triangle geometry is authoritative. A parcel-centre or raster
        # footprint signature cannot validate exact native triangle hits.
        digest.update(b'native-material-occupancy-v1')
        surface = s.material_surface
        inputs = [(name, surface[name]) for name in ('vertices', 'faces', 'face_id')]
        inputs += [(name, getattr(s, name)) for name in ('parcel_plate', 'mass', 'xyz')]
    else:
        digest.update(f'owner-occupancy-v1:.36:{s.w}:{s.h}'.encode('ascii'))
        inputs = [(name, getattr(s, name)) for name in ('pos', 'parcel_east', 'parcel_extent', 'mass', 'parcel_plate', 'xyz', 'cell_area')]
    for name, supplied in inputs:
        value = np.asarray(supplied)
        digest.update(f'{name}:{value.dtype.str}:{value.shape}'.encode('ascii'))
        if not value.size:
            continue
        if value.flags.c_contiguous:
            digest.update(memoryview(value).cast('B'))
        else:
            # At most roughly 1 MiB of temporary geometry, even for a strided
            # selection from a detailed global model.
            row_bytes = value.dtype.itemsize*max(1, math.prod(value.shape[1:]))
            rows = max(1, 1024*1024//row_bytes)
            for start in range(0, len(value), rows):
                digest.update(memoryview(np.ascontiguousarray(value[start:start+rows])).cast('B'))
    return digest.hexdigest()


def _cached_occupancy(s):
    """Return a current raster cache or leave the exact fallback in charge."""
    cached = getattr(s, '_owner_occupancy', None)
    signature = getattr(s, '_owner_occupancy_signature', None)
    if not isinstance(cached, dict) or not isinstance(signature, str):
        return None
    if occupancy_signature(s) != signature:
        return None
    # The local planner has always omitted nonpositive samples. Zero mass
    # cannot alter occupancy; a malformed negative sample must not make the
    # raster's signed sum replace that established local interpretation.
    if np.any(s.mass < 0):
        return None
    if isinstance(getattr(s, 'native_mesh', None), dict):
        hits = getattr(s, '_material_occupancy_hits', None)
        if not isinstance(hits, dict) or not {'query_index', 'face_index'} <= set(hits):
            return None
        return {'_material_hits': hits}
    return cached


def _unit(value):
    return value/max(float(np.linalg.norm(value)), 1e-30)


def _root(parent, value):
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _join(parent, a, b):
    a, b = _root(parent, a), _root(parent, b)
    if a != b:
        parent[max(a, b)] = min(a, b)


def _native_occupancy(s):
    """Exact native hits, never the legacy mass-deposition fallback."""
    cached = _cached_occupancy(s)
    if cached is not None:
        return cached
    from material_surface import sample_surface
    return {'_material_hits': sample_surface(s.material_surface, s.xyz)}


def _native_material_components(s, owner, occupancy, welded=None):
    """Connected indexed material faces, independent of their projected cells.

    Shared *edges* establish adjacency. A shared control cell, nearby centres or
    a point-only vertex contact cannot join two terranes. Explicit inherited
    craton and whole-patch IDs remain indivisible. Exact triangle hits supply
    the contact/footprint mapping, including hidden material; they do not
    supply connectivity.

    Under welded-stack policy 1 a weld between two sheets of this same owner
    (a thrust contact collision_contacts labels 'accreted', overlapping by more
    than the 1 km2 floor below which sheets merely abut) also joins them: the
    stack docks or stays as one body, so a docked terrane cannot be torn back
    out of its host's stack. Overlap never
    joins material of different plates, and never joins anything under policy 0.
    ``welded`` optionally supplies ``collision_contacts.same_owner_overlap_pairs``
    (face pairs and their sheet-pair contact areas) already computed for the
    current step.
    """
    surface = s.material_surface
    if (len(surface['faces']) != len(s.mass)
            or not np.array_equal(surface['face_id'], s.parcel_patch)):
        raise ValueError('Native material faces must align with persistent parcel-patch IDs.')
    ids = np.flatnonzero((s.parcel_plate == owner) & (s.mass > 0))
    if not len(ids):
        return None
    triangles = surface['faces'][ids]
    edge_vertices = np.sort(np.concatenate((triangles[:, [0,1]], triangles[:, [1,2]], triangles[:, [2,0]])), axis=1)
    edge_face = np.tile(np.arange(len(ids), dtype=np.int64), 3)
    order = np.lexsort((edge_vertices[:, 1], edge_vertices[:, 0]))
    edge_vertices, edge_face = edge_vertices[order], edge_face[order]
    same = np.flatnonzero(np.all(edge_vertices[1:] == edge_vertices[:-1], axis=1))
    parent = np.arange(len(ids), dtype=np.int32)
    for index in same:
        _join(parent, int(edge_face[index]), int(edge_face[index+1]))
    stacked = None
    if welded_stack_enabled(s):
        pairs, contact = (collision_contacts.same_owner_overlap_pairs(s, with_contact_area=True)
                          if welded is None else welded)
        pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
        local_index = np.full(len(s.mass), -1, np.int64)
        local_index[ids] = np.arange(len(ids))
        local_pairs = local_index[pairs]
        use = np.all(local_pairs >= 0, axis=1)
        stacked, stacked_contact, local_pairs = pairs[use], np.asarray(contact, float)[use], local_pairs[use]
        for a, b in local_pairs:
            _join(parent, int(a), int(b))
    for values, selected in ((s.parcel_patch[ids], np.ones(len(ids), bool)),
                             (s.parcel_craton[ids], s.parcel_craton[ids] >= 0)):
        seen = {}
        for index in np.flatnonzero(selected):
            group = int(values[index])
            if group in seen:
                _join(parent, seen[group], int(index))
            else:
                seen[group] = int(index)
    assigned = np.asarray([_root(parent, i) for i in range(len(ids))], np.int32)
    face_component = np.full(len(s.mass), -1, np.int32)
    face_component[ids] = assigned
    stacked_component = None if stacked is None else face_component[stacked[:, 0]]
    if occupancy is None or '_material_hits' not in occupancy:
        occupancy = _native_occupancy(s)
    hits = occupancy['_material_hits']
    hit_cells, hit_faces = np.asarray(hits['query_index']), np.asarray(hits['face_index'])
    use = (s.parcel_plate[hit_faces] == owner) & (s.mass[hit_faces] > 0)
    hit_cells, hit_faces = hit_cells[use], hit_faces[use]
    labels = np.full(s.n, -1, np.int32)
    if len(hit_cells):
        exposed = getattr(s, 'exposed_material', None)
        preferred = np.zeros(len(hit_cells), bool) if exposed is None else np.asarray(exposed)[hit_cells] == hit_faces
        order = np.lexsort((s.parcel_patch[hit_faces], ~preferred, hit_cells))
        _, first = np.unique(hit_cells[order], return_index=True)
        chosen = order[first]
        labels[hit_cells[chosen]] = face_component[hit_faces[chosen]]
    components = {}
    for component in np.unique(assigned):
        material = ids[assigned == component]
        patches, centers, _ = patch_centres(s.pos[material], s.mass[material], s.parcel_patch[material])
        relevant = face_component[hit_faces] == component
        coverage_cells, coverage_faces = hit_cells[relevant], hit_faces[relevant]
        components[int(component)] = dict(
            patch_ids=patches, patch_centers=centers, patch_cells=s._indices(centers),
            parcel_indices=material, area_km2=float(s.mass[material].sum()),
            center=_unit(np.sum(s.pos[material]*s.mass[material, None], axis=0)),
            anchor_patch=int(patches.min()), footprint_cells=np.unique(coverage_cells),
            coverage_cells=coverage_cells, coverage_patch_ids=s.parcel_patch[coverage_faces])
        if stacked is not None:
            # The welds inside this body, for its process zone (localized_accretion).
            inside = stacked_component == component
            components[int(component)]['welded_pairs'] = stacked[inside]
            # Summed overlap of each weld's sheet pair, km2, for the event record.
            components[int(component)]['welded_contact_km2'] = stacked_contact[inside]
    return dict(labels=labels, components=components, face_component=face_component)


def _material_components(s, owner, occupancy=None, welded=None):
    """Per-owner footprints also retain crust concealed by another plate.

    An exactly current per-owner raster cache avoids a second geometric deposit.
    Otherwise the extra deposit is restricted to owners participating in
    collision and uses the same production spherical footprint kernel.
    Continental patch centres supplement the thresholded footprint where grid
    sampling hides a thin original patch. Juvenile subthreshold centres do not.
    """
    if isinstance(getattr(s, 'native_mesh', None), dict):
        return _native_material_components(s, owner, occupancy, welded)
    ids = np.flatnonzero((s.parcel_plate == owner) & (s.mass > 0))
    if not len(ids):
        return None
    patches, centers, inverse = patch_centres(s.pos[ids], s.mass[ids], s.parcel_patch[ids])
    centers_cell = s._indices(centers)
    continental = np.bincount(inverse, weights=(s.kind[ids] != 3), minlength=len(patches)) > 0
    cells = None if occupancy is None else occupancy.get(owner)
    if cells is None:
        zero = np.zeros(len(ids))
        deposited = deposit(s.pos[ids], s.parcel_east[ids], s.parcel_extent[ids],
                            s.mass[ids], s.kind[ids], zero, zero,
                            np.zeros(len(ids), np.int16), s.w, s.h, s.xyz,
                            s.cell_area, s._sample_coordinates)[0]
        occupied = deposited/s.cell_area >= .36
    else:
        occupied = np.zeros(s.n, bool)
        occupied[cells] = True
    occupied[centers_cell[continental]] = True
    labels, values = domain_components(occupied.astype(np.uint8), s.w, s.h)
    labels[values[labels] == 0] = -1
    assigned = labels[centers_cell].copy()
    parent = np.arange(len(values), dtype=np.int32)
    # Connected cratons stay whole even if a raster gap divides their footprint.
    protected = s.parcel_craton[ids] >= 0
    for group in np.unique(s.parcel_craton[ids][protected]):
        members = np.unique(inverse[s.parcel_craton[ids] == group])
        groups = np.unique(assigned[members])
        groups = groups[groups >= 0]
        if len(groups):
            for component in groups[1:]:
                _join(parent, int(groups[0]), int(component))
    roots = np.array([_root(parent, i) for i in range(len(parent))], np.int32)
    valid = labels >= 0
    labels[valid] = roots[labels[valid]]
    valid_patch = assigned >= 0
    assigned[valid_patch] = roots[assigned[valid_patch]]
    components = {}
    parcel_component = assigned[inverse]
    parcel_order = np.argsort(parcel_component, kind='stable')
    patch_order = np.argsort(assigned, kind='stable')
    sorted_parcels, sorted_patches = parcel_component[parcel_order], assigned[patch_order]
    for component in np.unique(assigned[valid_patch]):
        pi = patch_order[np.searchsorted(sorted_patches, component):np.searchsorted(sorted_patches, component, side='right')]
        material = ids[parcel_order[np.searchsorted(sorted_parcels, component):np.searchsorted(sorted_parcels, component, side='right')]]
        components[int(component)] = dict(
            patch_ids=patches[pi], patch_centers=centers[pi], patch_cells=centers_cell[pi],
            parcel_indices=material, area_km2=float(s.mass[material].sum()),
            center=_unit(np.sum(s.pos[material]*s.mass[material, None], axis=0)),
            anchor_patch=int(patches[pi].min()))
    return dict(labels=labels, components=components)


def _cell_neighbors(s, cells):
    cells = np.asarray(cells, dtype=int)
    if isinstance(getattr(s, 'native_mesh', None), dict):
        values = np.asarray(s.native_mesh['face_neighbors'])[cells].ravel()
        return values[values >= 0]
    return np.concatenate((s.east[cells], s.west[cells], s.north[cells], s.south[cells]))


def _contact_clusters(s, edges):
    """Split one material-component pair into spatially connected contacts."""
    if not len(edges):
        return []
    parent = np.arange(len(edges), dtype=np.int32)
    cell_edge = {}
    for i, edge in enumerate(edges):
        for cell in (int(s.ba[edge]), int(s.bb[edge])):
            if cell in cell_edge:
                _join(parent, i, cell_edge[cell])
            else:
                cell_edge[cell] = i
    for cell, edge in cell_edge.items():
        for adjacent in _cell_neighbors(s, [cell]):
            other = cell_edge.get(int(adjacent))
            if other is not None:
                _join(parent, edge, other)
    labels = np.array([_root(parent, i) for i in range(len(edges))])
    return [edges[labels == label] for label in np.unique(labels)]


def _near_contact_patches(s, component, cells):
    ring = np.unique(np.concatenate((cells, _cell_neighbors(s, cells))))
    if 'coverage_cells' in component:
        return np.unique(component['coverage_patch_ids'][np.isin(component['coverage_cells'], ring)])
    return component['patch_ids'][np.isin(component['patch_cells'], ring)]


def _anchor(component, candidates, point):
    use = np.flatnonzero(np.isin(component['patch_ids'], candidates))
    return int(component['patch_ids'][use[np.argmax(component['patch_centers'][use] @ point)]])


def remap_contact_patches(s, old_patches, replacement_patches):
    """Keep actual patch anchors through the engine's same-cell arc coalescence.

    The two arrays describe exactly the existing old-to-new parcel-patch map.
    No spatial inference or reassignment to a different terrane occurs here.
    """
    old, new = np.asarray(old_patches), np.asarray(replacement_patches)
    if old.shape != new.shape or old.ndim != 1:
        raise ValueError('Coalesced patch ID arrays must align.')
    mapping, ambiguous = {}, set()
    for a, b in zip(old, new):
        a, b = int(a), int(b)
        if a in mapping and mapping[a] != b:
            ambiguous.add(a)
        mapping[a] = b
    for row in getattr(s, 'local_accretion_contacts', []):
        referenced = {int(row['p_patch']), int(row['q_patch'])}
        for name in ('p_patches', 'q_patches'):
            referenced.update(int(value) for value in row.get(name, []))
        if referenced & ambiguous:
            # A formerly shared patch split into different coalesced columns.
            # Its debt has no unambiguous local footprint; do not pick a child.
            row['loading'], row['completed'] = 0., True
            continue
        for name in ('p_patch', 'q_patch'):
            row[name] = mapping.get(int(row[name]), int(row[name]))
        for name in ('p_patches', 'q_patches'):
            if name in row:
                row[name] = sorted({mapping.get(int(value), int(value)) for value in row[name]})


def _contact_patch_sets(row):
    """Legacy records have one observed anchor; new records retain a local belt."""
    return (set(map(int, row.get('p_patches', [row['p_patch']]))),
            set(map(int, row.get('q_patches', [row['q_patch']]))))


def _persistent_contact_geometry(s):
    """Current exact overlaps admit concealed material without a grid edge."""
    if not isinstance(getattr(s, 'native_mesh', None), dict) or not hasattr(s, 'collision_contacts'):
        return []
    overlap = collision_contacts.current_overlap(s)
    if overlap is None:
        return []
    sheets = s.parcel_collision_sheet
    first, second, area = overlap['first'], overlap['second'], overlap['area_km2']
    result = []
    for row in s.collision_contacts:
        if row.get('state') != 'active' or row.get('last_seen_myr') != float(s.t):
            continue
        top, under = row['top_sheet'], row['under_sheet']
        use = (((sheets[first] == top) & (sheets[second] == under))
               | ((sheets[first] == under) & (sheets[second] == top)))
        use &= (area > 0.) & (s.mass[first] > 0.) & (s.mass[second] > 0.)
        use &= (s.kind[first] > 0) & (s.kind[second] > 0)
        if not np.any(use):
            continue
        for front in row.get('local_fronts', [None]):
            selected = np.flatnonzero(use)
            if front is not None:
                selected = np.asarray(front['overlap_indices'])
                if (selected.ndim != 1 or not np.issubdtype(selected.dtype,np.integer)
                        or np.any(selected < 0) or np.any(selected >= len(first))
                        or not np.all(use[selected])):
                    raise ValueError('Persistent contact front has invalid exact overlap references.')
            if not len(selected):
                continue
            a, b, weight = first[selected], second[selected], area[selected]
            p, q = sorted((int(s.parcel_plate[a[0]]), int(s.parcel_plate[b[0]])))
            if p == q or not s.active[p] or not s.active[q]:
                continue
            swap = s.parcel_plate[a] != p
            a, b = np.where(swap, b, a), np.where(swap, a, b)
            if np.any(s.parcel_plate[a] != p) or np.any(s.parcel_plate[b] != q):
                raise ValueError('Persistent collision sheets must each have one current owner.')
            result.append(dict(p=p, q=q, first=a, second=b, area=weight, record=row, front=front))
    return result


def _overlap_clusters(s, first, second):
    """Shared material edges connect a local overlap belt on either side."""
    parent = np.arange(len(first), dtype=np.int32)
    seen = {}
    for index, pair in enumerate(zip(first, second)):
        for side, face in enumerate(pair):
            triangle = s.material_surface['faces'][face]
            keys = [(side, 'face', int(face))]
            keys += [(side, 'edge', *sorted((int(a), int(b))))
                     for a, b in zip(triangle, np.roll(triangle, -1))]
            for key in keys:
                if key in seen:
                    _join(parent, index, seen[key])
                else:
                    seen[key] = index
    labels = np.array([_root(parent, i) for i in range(len(first))])
    return [np.flatnonzero(labels == label) for label in np.unique(labels)]


def _persistent_observations(s, contacts, geometry):
    observations = []
    roots = getattr(s, 'material_lineage', {}).get('root_id', s.parcel_patch)
    for contact in contacts:
        p, q = contact['p'], contact['q']
        if geometry[p] is None or geometry[q] is None:
            continue
        first, second, area = contact['first'], contact['second'], contact['area']
        components = np.column_stack((geometry[p]['face_component'][first],
                                      geometry[q]['face_component'][second]))
        groups, inverse = np.unique(components, axis=0, return_inverse=True)
        for group, (cp, cq) in enumerate(groups):
            if cp < 0 or cq < 0:
                continue
            candidates = np.flatnonzero(inverse == group)
            front = contact['front']
            # New front partitions come from connected clipped polygons. A
            # coarse face shared by disjoint overlaps must not join their clocks.
            clusters = ([np.arange(len(candidates))] if front is not None else
                        _overlap_clusters(s, first[candidates], second[candidates]))
            for cluster in clusters:
                take = candidates[cluster]
                a, b, weight = first[take], second[take], area[take]
                center = (_unit(np.asarray(front['center'])) if front is not None else
                          _unit(np.sum((s.pos[a]+s.pos[b])*weight[:,None], axis=0)))
                normal = np.asarray((front if front is not None else contact['record'])['normal'], float)
                normal = normal-center*np.dot(normal, center)
                if np.linalg.norm(normal) < 1e-12:
                    continue
                normal = _unit(normal)
                if p != int(contact['record']['top_owner']):
                    normal = -normal
                dv = np.cross(s.omega[q]-s.omega[p], center)*RADIUS_KM
                speed = float(dv@normal)
                shear = float(np.linalg.norm(dv-speed*normal))
                if speed >= -max(2., .35*shear):
                    continue
                if front is None:
                    vertices = s.material_surface['vertices'][s.material_surface['faces'][np.r_[a,b]]].reshape(-1,3)
                    radius = float(np.arccos(np.clip(vertices@center,-1.,1.)).max())*RADIUS_KM
                    length = min(2000.,2.*math.sqrt(float(weight.sum())))
                else:
                    radius = float(front['footprint_radius_km'])
                    length = float(front['length_km'])*float(weight.sum())/float(np.sum(area))
                observations.append(dict(p=p, q=q, cp=int(cp), cq=int(cq),
                    p_patches=np.unique(s.parcel_patch[a]), q_patches=np.unique(s.parcel_patch[b]),
                    p_roots=set(map(int, roots[a])), q_roots=set(map(int, roots[b])),
                    length=length, point=center, center=center, normal=normal, radius_km=radius,
                    persistent=contact['record']))
    return observations


def _same_front_footprint(s, previous, current):
    """Shared broad birth roots alone cannot lend a distant front their debt.

    Front indices are epoch-local, so matching uses physical bounds instead.
    Recorded angular motion allows a previously observed footprint to advect.
    Existing one-record-per-observation selection handles ambiguous splits by
    retaining at most one old debt; additional fronts begin with zero loading.
    """
    radius = previous.get('contact_radius_km')
    if radius is None:
        return False  # An older exact record has no unambiguous local footprint.
    old_center = _unit(np.asarray(previous['center'],float))
    distance = float(np.arccos(np.clip(old_center@current['center'],-1.,1.)))*RADIUS_KM
    elapsed = max(0.,float(s.t)-float(previous['last_seen_myr']))
    movement = max(np.linalg.norm(s.omega[current['p']]),np.linalg.norm(s.omega[current['q']]))*RADIUS_KM*elapsed
    return distance <= float(radius)+current['radius_km']+movement+1e-7


def plan_accretions(s, dt):
    """Return zero or one coherent, locally matured transfer plan.

    Required simulation fields are the current spherical boundary/grid and
    parcel/trace arrays already carried by Simulation. Plans contain source,
    target, their persistent UIDs, parcel_indices, trace_indices, patch_ids,
    footprint_cells (including overlapped material), surface_cells (only the
    source's currently visible buoyant cells), area_km2, center, contact_id,
    contact_length_km, loading and started_myr. The engine applies a plan once;
    use ``complete_contact`` after a successful transfer.
    """
    if not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError('Local contact integration needs a positive finite step.')
    records = getattr(s, 'local_accretion_contacts', [])
    for row in records:
        if row.get('collision_contact_id') is None:
            row['loading'] *= math.exp(-dt/MEMORY_MYR)
        elif row.get('decayed_myr', row.get('updated_myr')) != float(s.t):
            row['loading'] *= math.exp(-dt/MEMORY_MYR)
            row['decayed_myr'] = float(s.t)
    # Retain recent inactive contacts for their leaky memory, not indefinitely.
    records = [r for r in records if s.t-r['last_seen_myr'] <= 4*MEMORY_MYR]
    s.local_accretion_contacts = records
    s.next_accretion_contact_id = int(getattr(s, 'next_accretion_contact_id', 1))
    dv = np.cross(s.omega[s.bq]-s.omega[s.bp], s.bmid)*RADIUS_KM
    normal = np.sum(dv*s.bn, axis=1)
    shear = np.linalg.norm(dv-normal[:, None]*s.bn, axis=1)
    collision = ((s.bcode == 4) & (s.plate[s.ba] == s.bp) & (s.plate[s.bb] == s.bq)
                 & s.active[s.bp] & s.active[s.bq] & (s.crust[s.ba] > 0) & (s.crust[s.bb] > 0)
                 & (normal < -np.maximum(2., .35*shear)))
    exact = _persistent_contact_geometry(s)
    exact_pairs = {(contact['p'], contact['q']) for contact in exact}
    edges = np.flatnonzero(collision)
    if not len(edges) and not exact:
        localized_accretion.update(s, [], {}, dt)
        return []
    owners = np.unique(np.r_[s.bp[edges], s.bq[edges], [p for pair in exact_pairs for p in pair]])
    native = isinstance(getattr(s, 'native_mesh', None), dict)
    occupancy = _native_occupancy(s) if native else _cached_occupancy(s)
    welded = (collision_contacts.same_owner_overlap_pairs(s, with_contact_area=True)
              if native and welded_stack_enabled(s) else None)
    geometry = {int(p): _material_components(s, int(p), occupancy, welded) for p in owners}
    groups = {}
    for edge in edges:
        p, q = int(s.bp[edge]), int(s.bq[edge])
        a, b = int(s.ba[edge]), int(s.bb[edge])
        if p > q:
            p, q, a, b = q, p, b, a
        if geometry[p] is None or geometry[q] is None:
            continue
        cp, cq = int(geometry[p]['labels'][a]), int(geometry[q]['labels'][b])
        if cp not in geometry[p]['components'] or cq not in geometry[q]['components']:
            continue
        groups.setdefault((p, q, cp, cq), []).append(int(edge))
    observations = _persistent_observations(s, exact, geometry)
    exact_observations = list(observations)
    for (p, q, cp, cq), group_edges in sorted(groups.items()):
        a, b = geometry[p]['components'][cp], geometry[q]['components'][cq]
        for contact in _contact_clusters(s, np.asarray(group_edges, dtype=int)):
            p_cells = np.where(s.bp[contact] == p, s.ba[contact], s.bb[contact])
            q_cells = np.where(s.bp[contact] == q, s.ba[contact], s.bb[contact])
            p_patches, q_patches = _near_contact_patches(s, a, p_cells), _near_contact_patches(s, b, q_cells)
            if not len(p_patches) or not len(q_patches):
                continue
            length = float(s.bl[contact].sum())
            point = s.bmid[contact[np.argmax(s.bl[contact])]]
            center = _unit(np.sum(s.bmid[contact]*s.bl[contact, None], axis=0))
            # Suppress only a duplicate local belt. A distant touching front
            # between these plates retains its independent loading, even if it
            # has not yet developed a finite-area underthrust overlap.
            if any((p,q,cp,cq) == (seen['p'],seen['q'],seen['cp'],seen['cq'])
                   and np.intersect1d(p_patches,seen['p_patches']).size
                   and np.intersect1d(q_patches,seen['q_patches']).size
                   for seen in exact_observations):
                continue
            observations.append(dict(p=p, q=q, cp=cp, cq=cq, p_patches=p_patches,
                q_patches=q_patches, length=length, point=point, center=center))
    # Native policy 1 uses real clipped contact observations for root-local
    # engagement. Projected labels alone cannot authorize ownership transfer.
    localized_accretion.update(s, exact_observations, geometry, dt)
    ready, used_records = [], set()
    observed_patches = {r['id']: _contact_patch_sets(r) for r in records}
    for observation in observations:
        p, q, cp, cq = (observation[name] for name in ('p', 'q', 'cp', 'cq'))
        a, b = geometry[p]['components'][cp], geometry[q]['components'][cq]
        p_patches, q_patches = observation['p_patches'], observation['q_patches']
        length, point, center = (observation[name] for name in ('length', 'point', 'center'))
        persistent = observation.get('persistent')
        uid_p, uid_q = int(s.plate_uid[p]), int(s.plate_uid[q])
        current_p, current_q = set(map(int, p_patches)), set(map(int, q_patches))
        def local_overlap(row):
            if persistent is not None and row.get('collision_contact_id') is not None:
                return (row['collision_contact_id'] == persistent['id']
                        and set(row.get('p_roots', ())) & observation['p_roots']
                        and set(row.get('q_roots', ())) & observation['q_roots']
                        and _same_front_footprint(s,row,observation))
            return (observed_patches[row['id']][0] & current_p
                    and observed_patches[row['id']][1] & current_q)
        matching = [r for r in records if r['id'] not in used_records and not r.get('completed', False)
                    and r['p_uid'] == uid_p and r['q_uid'] == uid_q
                    and local_overlap(r)]
        if matching:
            def priority(candidate):
                prior_p, prior_q = observed_patches[candidate['id']]
                overlap = (len(prior_p & current_p)/len(prior_p | current_p)
                           + len(prior_q & current_q)/len(prior_q | current_q))
                distance = 1.-float(_unit(np.asarray(candidate.get('center', center))) @ center)
                # A merged contact retains the strongest existing history.
                # Never sum several debts or let one record feed two fronts.
                return (-candidate['loading'], -overlap, distance, candidate['id'])
            row = min(matching, key=priority)
        else:
            row = dict(id=s.next_accretion_contact_id, p_uid=uid_p, q_uid=uid_q,
                       p_patch=_anchor(a, p_patches, point), q_patch=_anchor(b, q_patches, point),
                       started_myr=float(s.t-dt), last_seen_myr=float(s.t), loading=0., completed=False)
            s.next_accretion_contact_id += 1
            records.append(row)
        used_records.add(row['id'])
        # Refresh the observed local belt as a front advances. A single
        # old grid-scale anchor may leave contact while the belt persists.
        row['p_patches'], row['q_patches'] = sorted(current_p), sorted(current_q)
        if row['p_patch'] not in current_p:
            row['p_patch'] = _anchor(a, p_patches, point)
        if row['q_patch'] not in current_q:
            row['q_patch'] = _anchor(b, q_patches, point)
        observed_patches[row['id']] = current_p, current_q
        rate = min(2., length/max(500., 1.4*math.sqrt(min(a['area_km2'], b['area_km2']))))
        increment = 0. if persistent is not None and row.get('updated_myr') == float(s.t) else dt*rate
        row.update(last_seen_myr=float(s.t), loading=float(row['loading']+increment),
                   contact_length_km=length, center=center.tolist())
        if persistent is not None:
            row.update(collision_contact_id=int(persistent['id']),
                collision_started_myr=float(persistent['started_myr']),
                p_roots=sorted(observation['p_roots']), q_roots=sorted(observation['q_roots']),
                geometry_source='exact persistent material overlap', updated_myr=float(s.t),
                contact_radius_km=float(observation['radius_km']))
            persistent['local_accretion_contact_ids'] = sorted(set(
                persistent.get('local_accretion_contact_ids', ())) | {int(row['id'])})
        if ((not localized_accretion.enabled(s) and row['loading'] < THRESHOLD)
                or s.t-max(s.born[p], s.born[q]) < minimum_plate_age_myr(s)):
            continue
        source, target, component_id, component = (p, q, cp, a) if (a['area_km2'], uid_p, a['anchor_patch']) < (b['area_km2'], uid_q, b['anchor_patch']) else (q, p, cq, b)
        # suture_strength lives on the PERSISTENT collision contact, never on this
        # local-accretion row -- the row carries loading and geometry only. Reading it
        # from `row` returned the 0. default on every call, so the suture eligibility
        # path could not fire: accreted_km2 stood at 29,475 from 80 to 344 Myr while
        # contact 4 held suture_strength 1.0 across 2,356 km of convergence.
        qualification = localized_accretion.qualify(s, component, source, target,
            suture_strength=float((persistent or {}).get('suture_strength', 0.)))
        if localized_accretion.enabled(s):
            s.native_accretion_diagnostics['qualification_checks'].append(deepcopy(qualification))
        if not qualification['eligible']:
            continue
        material = component['parcel_indices']
        patches = component['patch_ids']
        # Refuse malformed preexisting split groups rather than compound one.
        protected = np.unique(s.parcel_craton[material][s.parcel_craton[material] >= 0])
        if len(protected) and np.any(np.isin(s.parcel_craton, protected) & ~np.isin(np.arange(len(s.mass)), material)):
            continue
        if np.any(np.isin(s.parcel_patch, patches) & (s.parcel_plate != source)):
            continue
        traced = np.flatnonzero((s.trace_plate == source) & np.isin(s.trace_patch, patches))
        footprint = component.get('footprint_cells', np.flatnonzero(geometry[source]['labels'] == component_id))
        visible = footprint[(s.plate[footprint] == source) & (s.crust[footprint] > 0)]
        ready.append(dict(source=source, target=target, source_uid=int(s.plate_uid[source]),
                          target_uid=int(s.plate_uid[target]), parcel_indices=material.copy(),
                          trace_indices=traced, patch_ids=patches.copy(), footprint_cells=footprint,
                          surface_cells=visible, area_km2=component['area_km2'], center=component['center'].copy(),
                          contact_id=int(row['id']), contact_length_km=length, loading=float(row['loading']),
                          started_myr=float(row['started_myr']),
                          contact_patch_ids=(p_patches if source == p else q_patches).copy()))
        if persistent is not None:
            ready[-1]['collision_contact_id'] = int(persistent['id'])
        if localized_accretion.enabled(s):
            ready[-1]['local_welding_qualification'] = qualification
        if len(component.get('welded_contact_km2', ())):
            # The weakest weld holding this body together, so a docking can be
            # traced to the contacts that made it one body.
            ready[-1]['welded_stack_min_contact_km2'] = float(np.min(component['welded_contact_km2']))
    return sorted(ready, key=lambda p:(-p['loading'], p['contact_id']))[:1]


def complete_contact(s, contact_id):
    """A successful engine mutation consumes that contact's accumulated load."""
    for row in getattr(s, 'local_accretion_contacts', []):
        if row['id'] == contact_id:
            row['completed'], row['loading'] = True, 0.
            return
    raise ValueError('The accretion contact is no longer available.')


def _rehost_local_episodes(s, plan):
    """Host changes follow the selected local material, never the whole UID."""
    from ridge_interaction import reassign_ridge_episodes
    source_uid, target_uid = plan['source_uid'], plan['target_uid']
    footprint = np.zeros(s.n, bool)
    footprint[plan['footprint_cells']] = True
    episodes = getattr(s, 'ridge_episodes', [])
    if episodes:
        cells = s._indices(np.asarray([r['center'] for r in episodes]))
        reassign_ridge_episodes(s, source_uid, target_uid, footprint[cells])
    # A basin center is often on its arc/parent interface. Include that one
    # neighboring cell for record reconciliation, not for selecting material.
    edge_region = footprint.copy()
    cells = np.flatnonzero(footprint)
    edge_region[_cell_neighbors(s, cells)] = True
    for row in getattr(s, 'backarc_basins', []):
        if row.get('phase') == 'closed' or not edge_region[int(s._indices(np.asarray([row['center']]))[0])]:
            continue
        roles = [name for name in ('parent_plate_uid', 'arc_plate_uid') if row.get(name) == source_uid]
        if not roles:
            continue
        for role in roles:
            row[role] = target_uid
        row.setdefault('host_transfers', []).append(dict(time_myr=float(s.t),
            source_plate_uid=source_uid, target_plate_uid=target_uid,
            roles=roles, contact_id=int(plan['contact_id'])))
        arc = row.get('arc_plate_uid')
        if arc is not None and len({row['parent_plate_uid'], arc, row['downgoing_plate_uid']}) < 3:
            row['phase'] = 'closed'
            row['phase_serial'] = row.get('phase_serial', 0)+1
            row['extension_rate_km_myr'] = 0.
            row['loading_rate_km_myr'] = 0.
            s._record('backarc_closed', f"Local terrane accretion closed the separate plate boundary of back-arc basin {row['id']}.",
                      ('backarc', row['id'], row['phase_serial']), plates=(plan['source'], plan['target']),
                      xyz=row['center'], details=dict(backarc_id=row['id'], contact_id=int(plan['contact_id']),
                      parent_plate_uid=row['parent_plate_uid'], arc_plate_uid=arc,
                      downgoing_plate_uid=row['downgoing_plate_uid'], mechanism='local coherent terrane accretion'))


def _release_departed_welding(s, plan, material):
    """Loading built up by rock that has left the source plate leaves with it.

    Welding memory is keyed (source UID, target UID, root). Once a root has no
    face left on the source, a row keyed on that source describes a plate the
    rock no longer belongs to; kept, it would let the rock re-mature toward its
    old host within a step or two of any later contact. A root still partly on
    the source keeps its row. Returns the number of distinct roots released.
    """
    state = s.accretion_welding_state
    roots = localized_accretion._roots(s)
    moved = np.unique(roots[material])
    # Only rock with mass remains; an eroded or foundered face keeps no memory.
    remaining = np.unique(roots[(np.asarray(s.parcel_plate) == int(plan['source']))
                                & (np.asarray(s.mass) > 0.)])
    departed = np.setdiff1d(moved, remaining)
    drop = (state['source_uid'] == int(plan['source_uid'])) & np.isin(state['root_id'], departed)
    released = int(len(np.unique(state['root_id'][drop])))
    if np.any(drop):
        for name in localized_accretion.TABLE:
            state[name] = state[name][~drop]
    # validate_frame checks these counts against the saved root arrays.
    s.native_accretion_diagnostics.update(directed_root_records=len(state['root_id']),
        currently_engaged_roots=int(np.count_nonzero(state['active_weight'] > 0.)),
        mature_engaged_roots=int(np.count_nonzero((state['loading'] >= localized_accretion.THRESHOLD)
                                                  & (state['active_weight'] > 0.))))
    return released


def apply_accretion(s, plan):
    """Apply one valid local plan; return False if intervening changes stale it.

    Positions, area, relief and structural memories persist. The source retains
    its remote material, motion and ocean support. The receiver's existing
    area-weighted motion/background averaging uses only the transferred mass.
    Retirement remains the engine's separate empty-remnant decision.
    """
    from trench_history import transfer_overriding
    if not localized_accretion.validate_plan(s, plan):
        return False
    stack_policy = isinstance(getattr(s, 'native_mesh', None), dict) and welded_stack_enabled(s)
    source, target = int(plan['source']), int(plan['target'])
    material = np.asarray(plan['parcel_indices'], int)
    patches = np.asarray(plan['patch_ids'], np.int64)
    contacts = [r for r in getattr(s, 'local_accretion_contacts', []) if r['id'] == plan['contact_id'] and not r.get('completed', False)]
    if (source == target or not contacts or not s.active[source] or not s.active[target]
            or int(s.plate_uid[source]) != plan['source_uid'] or int(s.plate_uid[target]) != plan['target_uid']
            or not len(material) or np.any(material < 0) or np.any(material >= len(s.mass))):
        return False
    selected = np.flatnonzero(np.isin(s.parcel_patch, patches))
    if not np.array_equal(np.sort(material), selected) or np.any(s.parcel_plate[material] != source):
        return False
    protected = np.unique(s.parcel_craton[material][s.parcel_craton[material] >= 0])
    if len(protected) and np.any(np.isin(s.parcel_craton, protected) & ~np.isin(np.arange(len(s.mass)), material)):
        return False
    traced = np.flatnonzero((s.trace_plate == source) & np.isin(s.trace_patch, patches))
    mass = float(s.mass[material].sum())
    receiver_mass = float(s.mass[s.parcel_plate == target].sum())
    if mass <= 0 or not math.isfinite(mass):
        return False
    region = np.asarray(plan['surface_cells'], int)
    region = region[(s.plate[region] == source) & (s.crust[region] > 0)]
    # Existing merger weighting, with this terrane's actual mass in the numerator.
    s.omega[target] = (s.omega[source]*mass+s.omega[target]*receiver_mass)/max(mass+receiver_mass, 1.)
    s.mantle[target] = (s.mantle[source]*mass+s.mantle[target]*receiver_mass)/max(mass+receiver_mass, 1.)
    contact_patches = plan.get('contact_patch_ids', [])
    belt = material[np.isin(s.parcel_patch[material], contact_patches) & (s.relief[material] > 400)]
    traced_belt = traced[np.isin(s.trace_patch[traced], contact_patches) & (s.trace_relief_m[traced] > 400)]
    s.suture[belt] = np.maximum(s.suture[belt], .85)
    s.trace_suture[traced_belt] = np.maximum(s.trace_suture[traced_belt], .85)
    juvenile = material[s.kind[material] == 3]
    juvenile_traces = traced[s.trace_kind[traced] == 3]
    # Kind 3 has a -100 m datum offset. Reclassifying it alone must not
    # manufacture relief; retained structure dictionaries use full height.
    s.relief[juvenile] -= 100.
    s.trace_relief_m[juvenile_traces] -= 100.
    s.trace_adjustment_m[juvenile_traces] -= 100.
    s.kind[juvenile] = 1
    s.parcel_plate[material] = target
    s.trace_kind[juvenile_traces] = 1
    s.trace_plate[traced] = target
    s.plate[region] = target
    s.support[target, region] += s.support[source, region]
    s.support[source, region] = 0.
    transfer_overriding(s, source, target)
    _rehost_local_episodes(s, plan)
    s.process_totals['accreted_km2'] += mass
    stack_details = {}
    if stack_policy:
        stack_details = dict(accretion_welded_stack_version=WELDED_STACK_VERSION,
            welded_stack_sheets=int(len(np.unique(s.parcel_collision_sheet[material]))),
            released_welding_roots=_release_departed_welding(s, plan, material),
            **({'welded_stack_min_contact_km2': float(plan['welded_stack_min_contact_km2'])}
               if 'welded_stack_min_contact_km2' in plan else {}))
    # Other local fronts retain their own contact clocks and inherited loading;
    # a small incoming terrane does not reset the whole receiver's rift age.
    complete_contact(s, int(plan['contact_id']))
    s._record('accretion', f"Sustained local convergence accreted a connected terrane from {s.names[source]} onto {s.names[target]}.",
              ('local_accretion', int(plan['contact_id'])), plates=(source, target), xyz=plan['center'],
              details=dict(accreted_area_km2=mass, source_plate_uid=int(plan['source_uid']),
                           target_plate_uid=int(plan['target_uid']), contact_id=int(plan['contact_id']),
                           contact_started_myr=float(plan['started_myr']), contact_length_km=float(plan['contact_length_km']),
                           contact_loading=float(plan['loading']), material_patch_count=int(len(patches)),
                           terrane_anchor_patch=int(patches.min()),
                           source_material_remaining_km2=float(s.mass[s.parcel_plate == source].sum()),
                           mechanism='sustained local contact; coherent connected buoyant terrane transfer',
                           **({'native_accretion_version':1,
                               'local_welding_qualification':deepcopy(plan['local_welding_qualification'])}
                              if 'local_welding_qualification' in plan else {}),
                           **({'collision_contact_id':int(plan['collision_contact_id'])}
                              if 'collision_contact_id' in plan else {}),
                           **stack_details))
    return True
