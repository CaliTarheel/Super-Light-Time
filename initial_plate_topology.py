"""Optional initial plate ownership distinct from the crust-type artwork.

This is a starting-condition topology helper, not a tectonic process. A crust
map says what material exists; this module may assign nearby oceanic lithosphere
to the same rigid plate as part of a continent so a coastline can be a passive
margin rather than automatically being a plate boundary.

Only the explicit version-one passive-margin apron mode is supported. It adds no
plates, no motion and no crust. Ocean cells are reached outward from selected
coast segments, so every attached ocean cell remains connected to the continent
whose plate owns it. A global area cap and shrinking trial widths preserve one
connected independent-ocean reservoir.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from fracture import component_labels

RADIUS_KM = 6371.0
VERSION = 1
DEFAULTS = dict(version=VERSION, mode='passive_margin_aprons', seed=0,
                passive_coast_fraction=.65, apron_width_km=650.,
                max_attached_ocean_fraction=.45)


def normalize(value=None):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError('Initial plate topology has unknown fields.')
    result = dict(DEFAULTS, **value)
    if type(result['version']) is not int or result['version'] != VERSION:
        raise ValueError('Initial plate topology version must be 1.')
    if result['mode'] != 'passive_margin_aprons':
        raise ValueError('Initial plate topology mode must be passive_margin_aprons.')
    seed = result['seed']
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2**32:
        raise ValueError('Initial plate topology seed must be an unsigned 32-bit integer.')
    result['seed'] = int(seed)
    for key, low, high in (
        ('passive_coast_fraction', .05, .95),
        ('apron_width_km', 50., 2000.),
        ('max_attached_ocean_fraction', .05, .80),
    ):
        value = result[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f'Initial plate topology {key} must be a finite number.')
        value = float(value)
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'Initial plate topology {key} must lie between {low} and {high}.')
        result[key] = value
    return result


def _unit(value):
    value = np.asarray(value, float)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-30)


def _coast_seeds(s, settings, ocean):
    """Choose broad seeded coast sectors, weighted by physical boundary length."""
    a, b = np.asarray(s.edge_a), np.asarray(s.edge_b)
    land_a = s.initial_crust[a] > 0
    land_b = s.initial_crust[b] > 0
    coast = land_a ^ land_b
    edges = np.flatnonzero(coast)
    if not len(edges):
        return {}, dict(total_length_km=0., selected_length_km=0.)

    land_cell = np.where(land_a[edges], a[edges], b[edges])
    ocean_cell = np.where(land_a[edges], b[edges], a[edges])
    owner = s.plate[land_cell]
    midpoint = _unit(s.xyz[land_cell] + s.xyz[ocean_cell])
    seed_owner = {}
    selected_edge = np.zeros(len(edges), bool)
    selected_length = 0.
    total_length = float(np.asarray(s.edge_length)[edges].sum())

    for plate in np.unique(owner):
        local = np.flatnonzero(owner == plate)
        if not len(local) or plate == 0:
            continue
        land = np.flatnonzero((s.initial_crust > 0) & (s.plate == plate))
        if not len(land):
            continue
        center = _unit(np.sum(s.xyz[land] * s.cell_area[land, None], axis=0))
        reference = np.array([0., 0., 1.]) if abs(center[2]) < .85 else np.array([1., 0., 0.])
        east = _unit(np.cross(reference, center))
        north = np.cross(center, east)
        azimuth = np.arctan2(midpoint[local] @ north, midpoint[local] @ east)
        rng = np.random.default_rng(np.random.SeedSequence([settings['seed'], int(plate), 0x50415353]))
        phase = rng.uniform(-np.pi, np.pi, 3)
        score = (np.sin(2*azimuth + phase[0])
                 + .45*np.sin(5*azimuth + phase[1])
                 + .20*np.cos(7*azimuth + phase[2]))
        order = local[np.argsort(-score, kind='stable')]
        lengths = np.asarray(s.edge_length)[edges[order]]
        target = settings['passive_coast_fraction'] * lengths.sum()
        count = int(np.searchsorted(np.cumsum(lengths), target)) + 1
        chosen = order[:min(count, len(order))]
        selected_edge[chosen] = True
        selected_length += float(np.asarray(s.edge_length)[edges[chosen]].sum())
        for row in chosen:
            cell = int(ocean_cell[row])
            candidate = int(owner[row])
            # A rare cell touching two chosen continental plates uses the
            # stronger local score, then the lower stable slot as a tie break.
            quality = float(score[np.flatnonzero(local == row)[0]])
            current = seed_owner.get(cell)
            if current is None or (quality, -candidate) > (current[1], -current[0]):
                seed_owner[cell] = (candidate, quality)

    protected = np.zeros(s.n, bool)
    protected[np.unique(ocean_cell[~selected_edge])] = True
    if seed_owner:
        protected[np.fromiter(seed_owner, dtype=np.int64)] = False
    return ({cell: owner for cell, (owner, _) in seed_owner.items()}, protected,
            dict(total_length_km=total_length, selected_length_km=selected_length,
                 protected_coast_cells=int(np.count_nonzero(protected))))


def _distances(s, seeds, ocean, maximum):
    """Multi-source physical-distance flood through existing ocean cells."""
    distance = np.full(s.n, np.inf)
    owner = np.full(s.n, -1, np.int16)
    queue = []
    for cell, plate in seeds.items():
        if not ocean[cell]:
            continue
        if distance[cell] > 0. or (distance[cell] == 0. and plate < owner[cell]):
            distance[cell] = 0.
            owner[cell] = int(plate)
            heapq.heappush(queue, (0., int(plate), int(cell)))

    neighbors = (s.east, s.west, s.north, s.south)
    while queue:
        value, plate, cell = heapq.heappop(queue)
        if value != distance[cell] or plate != int(owner[cell]) or value > maximum:
            continue
        for table in neighbors:
            other = int(table[cell])
            if other == cell or not ocean[other]:
                continue
            angle = math.acos(float(np.clip(s.xyz[cell] @ s.xyz[other], -1., 1.)))
            proposed = value + RADIUS_KM*angle
            if proposed > maximum + 1e-10:
                continue
            if proposed < distance[other] - 1e-10 or (
                    abs(proposed-distance[other]) <= 1e-10 and (owner[other] < 0 or plate < owner[other])):
                distance[other] = proposed
                owner[other] = plate
                heapq.heappush(queue, (proposed, plate, other))
    return distance, owner


def _area_limited_mask(s, eligible, distance, max_area):
    cells = np.flatnonzero(eligible)
    if not len(cells):
        return np.zeros(s.n, bool)
    order = cells[np.lexsort((cells, distance[cells]))]
    cumulative = np.cumsum(s.cell_area[order])
    count = int(np.searchsorted(cumulative, max_area, side='right'))
    selected = np.zeros(s.n, bool)
    selected[order[:count]] = True
    return selected


def apply(s, settings):
    """Apply a topology-only passive-margin ownership seed to a fresh raster world."""
    settings = normalize(settings)
    if settings is None:
        return dict(version=0, enabled=False, mode='crust-derived plate ownership')
    land = np.asarray(s.initial_crust) > 0
    ocean = ~land
    if not np.any(land) or not np.any(ocean):
        return dict(version=VERSION, enabled=False, mode=settings['mode'],
                    reason='Passive-margin ownership needs both continental and oceanic crust.')

    if not np.all(s.plate[ocean] == 0):
        raise ValueError('Passive-margin topology must run before any other initial ocean ownership rewrite.')

    seeds, protected, coast = _coast_seeds(s, settings, ocean)
    if not seeds:
        raise ValueError('Passive-margin topology found no eligible continental coast.')
    distance, owner = _distances(s, seeds, ocean & ~protected, settings['apron_width_km'])
    ocean_area = float(s.cell_area[ocean].sum())
    max_area = settings['max_attached_ocean_fraction'] * ocean_area

    chosen = None
    realized_width = 0.
    # Preserve one connected independent reservoir. The initial requested width
    # gets first chance; progressively narrower physical bands are deterministic
    # fallbacks for narrow high-land seaways.
    # Continue down to the seeded coastline itself. At higher import resolution
    # the former 1/8-width floor could still isolate a single ocean cell, even
    # when the original ocean and the coast-only assignment are both connected.
    # This only extends failed trials; every previously successful trial keeps
    # its original ownership. The final connectivity check remains mandatory.
    for factor in (1., .80, .64, .50, .40, .32, .25, .20, .16, .125,
                   .10, .08, .064, .05, .04, .032, .025, .02, .016, 0.):
        width = settings['apron_width_km'] * factor
        eligible = ocean & (owner >= 0) & (distance <= width + 1e-10)
        selected = _area_limited_mask(s, eligible, distance, max_area)
        if not np.any(selected):
            continue
        _, components = component_labels(ocean & ~selected, s.w, s.h)
        if components == 1:
            chosen = selected
            realized_width = width
            break
    if chosen is None:
        raise ValueError('Passive-margin ownership could not preserve one connected independent ocean.')

    s.plate[chosen] = owner[chosen]
    attached_area = float(s.cell_area[chosen].sum())
    independent = ocean & ~chosen
    _, ocean_components = component_labels(independent, s.w, s.h)

    a, b = np.asarray(s.edge_a), np.asarray(s.edge_b)
    land_a = s.initial_crust[a] > 0
    land_b = s.initial_crust[b] > 0
    coast_mask = land_a ^ land_b
    coast_edges = np.flatnonzero(coast_mask)
    land_cell = np.where(land_a[coast_edges], a[coast_edges], b[coast_edges])
    ocean_cell = np.where(land_a[coast_edges], b[coast_edges], a[coast_edges])
    passive = s.plate[land_cell] == s.plate[ocean_cell]
    lengths = np.asarray(s.edge_length)[coast_edges]
    mixed = []
    for plate in np.unique(s.plate):
        if plate == 0:
            continue
        if np.any((s.plate == plate) & (s.initial_crust > 0)) and np.any((s.plate == plate) & ocean):
            mixed.append(int(plate))

    return dict(version=VERSION, enabled=True, mode=settings['mode'], seed=settings['seed'],
        requested_passive_coast_fraction=settings['passive_coast_fraction'],
        requested_apron_width_km=settings['apron_width_km'],
        realized_apron_width_km=float(realized_width),
        max_attached_ocean_fraction=settings['max_attached_ocean_fraction'],
        coast_length_km=float(lengths.sum()),
        seeded_coast_length_km=float(coast['selected_length_km']),
        protected_active_coast_cells=int(coast['protected_coast_cells']),
        passive_margin_length_km=float(lengths[passive].sum()),
        active_coast_length_km=float(lengths[~passive].sum()),
        passive_margin_fraction=float(lengths[passive].sum()/max(lengths.sum(), 1e-30)),
        attached_ocean_area_km2=attached_area,
        attached_ocean_fraction=float(attached_area/ocean_area),
        independent_ocean_area_km2=float(s.cell_area[independent].sum()),
        independent_ocean_fraction=float(s.cell_area[independent].sum()/ocean_area),
        independent_ocean_components=int(ocean_components),
        mixed_plate_count=len(mixed), mixed_plate_slots=mixed,
        ownership='Crust type is unchanged; selected neighboring ocean cells share continental rigid-plate ownership.',
        motion='No velocity is added. Mixed plates inherit the ordinary solved/seeded plate motion.')
