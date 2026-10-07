"""Optional ring-of-subduction initial condition around one supercontinent plate.

A circle on the sphere (a great circle by default) is declared as an inherited
trench. Everything inside it -- all continental material and the water between
the coast and the circle -- becomes ONE carrier plate. The water outside is the
distant ocean, which overrides. The carrier's own oceanic skirt is therefore the
incoming lithosphere at every point of the ring, so its slab pulls the plate
that carries the continent (Scotese Rule V: a continent pulled by an attached
slab after its ocean's ridge has been consumed).

A uniform pull around any circle exerts zero net torque about the circle's
axis, so in the rigid balance the pulls largely cancel: this startup loads the
supercontinent in tension instead of drifting it. Whether that tension breaks
the continent before the oceanic skirt is a strength question answered by
``plate_limit_analysis``; this module only declares geometry, ownership,
cooling ages and the ring's polarity. See GREAT-CIRCLE-STARTUP.md.

Like ``primordial_ocean``, only whole fitted water triangles change owner and
continental material is never refitted. Continental owners are merged into the
carrier; their former partition cuts remain as inherited ``seed_rift_id``
structure on the material, not as plate boundaries.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

VERSION = 1
RADIUS_KM = 6371.
DEFAULTS = dict(enabled=False, ring_radius_deg=90., pole_lat_lon_deg=None, minimum_skirt_km=300.,
                skirt_age_at_margin_myr=150., skirt_age_at_trench_myr=100., far_ocean_age_myr=100.)
_RANGES = dict(ring_radius_deg=(20., 160.), minimum_skirt_km=(0., 5000.),
               skirt_age_at_margin_myr=(1., 300.), skirt_age_at_trench_myr=(1., 300.),
               far_ocean_age_myr=(1., 300.))


def _number(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value) or not low <= value <= high):
        raise ValueError(f'Supercontinent ring {name} must be finite and between {low} and {high}.')
    return float(value)


def normalize(value=None):
    if value is None:
        return deepcopy(DEFAULTS)
    if not isinstance(value, dict) or set(value)-set(DEFAULTS):
        raise ValueError('Invalid supercontinent_ring settings; allowed keys are '+', '.join(sorted(DEFAULTS))+'.')
    result = dict(deepcopy(DEFAULTS), **value)
    if type(result['enabled']) is not bool:
        raise ValueError('Supercontinent ring enabled must be boolean.')
    for name, (low, high) in _RANGES.items():
        result[name] = _number(name, result[name], low, high)
    pole = result['pole_lat_lon_deg']
    if pole is not None:
        if not isinstance(pole, (list, tuple)) or len(pole) != 2:
            raise ValueError('Supercontinent ring pole_lat_lon_deg must be null or [latitude, longitude].')
        result['pole_lat_lon_deg'] = [_number('pole latitude', pole[0], -90., 90.),
                                      _number('pole longitude', pole[1], -360., 360.)]
    return result


def enabled(s):
    return getattr(s, 'supercontinent_ring_version', 0) == VERSION


def _unit(points):
    points = np.asarray(points, float)
    return points/np.maximum(np.linalg.norm(points, axis=-1, keepdims=True), 1e-30)


def _pole_xyz(lat_lon):
    lat, lon = np.radians(lat_lon)
    return np.array([math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat)])


def _nearest_angle(points, targets, chunk=4096):
    """Smallest great-circle angle from each point to any target point."""
    best = np.full(len(points), np.pi)
    for start in range(0, len(targets), chunk):
        dots = np.clip(points@targets[start:start+chunk].T, -1., 1.)
        best = np.minimum(best, np.arccos(dots.max(axis=1)))
    return best


def geometry(fitted, settings):
    """Classify fitted faces against the declared circle; pure, never mutates."""
    vertices, faces = np.asarray(fitted['vertices'], float), np.asarray(fitted['faces'])
    kind, area = np.asarray(fitted['face_kind']), np.asarray(fitted['area_km2'], float)
    centers = _unit(vertices[faces].sum(axis=1))
    land = kind > 0
    if not np.any(land):
        raise ValueError('A supercontinent ring requires continental material.')
    if settings['pole_lat_lon_deg'] is None:
        pole = _unit(np.sum(centers[land]*area[land, None], axis=0))
        if np.linalg.norm(np.sum(centers[land]*area[land, None], axis=0)) < 1e-9:
            raise ValueError('The continental centroid is undefined; specify pole_lat_lon_deg.')
    else:
        pole = _pole_xyz(settings['pole_lat_lon_deg'])
    radius = math.radians(settings['ring_radius_deg'])
    colatitude = np.arccos(np.clip(centers@pole, -1., 1.))
    inside = colatitude < radius
    # Every vertex of every land face must lie a declared skirt width inside
    # the ring, so no continental coast can touch the trench or the far ocean.
    land_vertices = np.unique(faces[land])
    land_colat = np.arccos(np.clip(vertices[land_vertices]@pole, -1., 1.))
    skirt_km = float((radius-land_colat.max())*RADIUS_KM)
    if skirt_km < settings['minimum_skirt_km']:
        raise ValueError(f'The supercontinent reaches within {skirt_km:.0f} km of the ring '
                         f'(minimum_skirt_km={settings["minimum_skirt_km"]:.0f}); enlarge ring_radius_deg '
                         'or move the pole.')
    water_inside = inside & ~land
    water_outside = ~inside & ~land
    if not np.any(water_inside) or not np.any(water_outside):
        raise ValueError('The ring must leave oceanic skirt inside and distant ocean outside.')
    # Linear age between coast and trench along the local coast-to-ring path.
    to_coast = _nearest_angle(centers[water_inside], centers[land])
    to_ring = np.maximum(radius-colatitude[water_inside], 0.)
    fraction = to_coast/np.maximum(to_coast+to_ring, 1e-12)
    age = np.zeros(len(faces))
    age[water_inside] = (settings['skirt_age_at_margin_myr']
                         + (settings['skirt_age_at_trench_myr']-settings['skirt_age_at_margin_myr'])*fraction)
    age[water_outside] = settings['far_ocean_age_myr']
    return dict(pole=pole, radius_rad=radius, inside=inside, land=land, water_inside=water_inside,
                water_outside=water_outside, ocean_age_myr=age, minimum_skirt_km=skirt_km,
                maximum_land_colatitude_deg=float(np.degrees(land_colat.max())))


def initialize(s, fitted):
    """Relabel owners before native support is built; returns a new fitted dict."""
    settings = normalize(s.config.get('supercontinent_ring'))
    if not settings['enabled']:
        return fitted
    if s.t != 0. or s.steps != 0 or enabled(s):
        raise ValueError('A supercontinent ring is only initialized once in a fresh world.')
    if getattr(s, 'primordial_ocean_version', 0):
        raise ValueError('supercontinent_ring and primordial_ocean are alternative ocean initializations.')
    original_uid = getattr(s, 'initial_ocean_plate_uid', None)
    ocean = np.flatnonzero(s.active & (s.plate_uid == original_uid)) if original_uid is not None else []
    if len(ocean) != 1:
        raise ValueError('A supercontinent ring requires one identifiable source ocean plate.')
    ocean = int(ocean[0])
    ring = geometry(fitted, settings)
    owners = np.asarray(fitted['face_owner'])
    area = np.asarray(fitted['area_km2'], float)
    if np.any(owners[~ring['land']] != ocean):
        raise ValueError('The supercontinent ring cannot reinterpret a world with several pre-existing ocean owners.')
    land_owners = np.unique(owners[ring['land']])
    land_area = np.bincount(owners[ring['land']], weights=area[ring['land']], minlength=s.capacity)
    carrier = int(land_owners[np.argmax(land_area[land_owners])])
    merged = [int(p) for p in land_owners if p != carrier]
    merged_uids = [int(s.plate_uid[p]) for p in merged]
    result = dict(fitted, face_owner=owners.copy(), ocean_age_myr=ring['ocean_age_myr'])
    result['face_owner'][ring['land'] | ring['water_inside']] = carrier
    result['face_owner'][ring['water_outside']] = ocean
    for p in merged:
        # Retire whole slots: no material, support or history belongs to them.
        s.active[p] = False
        s.omega[p] = s.mantle[p] = 0.
        s.polarity[p, :] = s.polarity[:, p] = -1
        s.collision_clock[p, :] = s.collision_clock[:, p] = 0.
    # The merged plate starts from the carrier's provisional rotation; the
    # reviewed balance subsequently solves every rotation from forces.
    centers = _unit(np.asarray(fitted['vertices'])[np.asarray(fitted['faces'])].sum(axis=1))
    for slot in (carrier, ocean):
        selected = result['face_owner'] == slot
        s.centres[slot] = _unit(np.sum(centers[selected]*area[selected, None], axis=0))
    s.names[carrier] = 'Supercontinent'
    s.names[ocean] = 'Distant ocean'
    s.count = int(np.count_nonzero(s.active))
    s.supercontinent_ring_version = VERSION
    s.supercontinent_ring_carrier_uid = int(s.plate_uid[carrier])
    s.supercontinent_ring_overriding_uid = int(s.plate_uid[ocean])
    s.supercontinent_ring_diagnostics = dict(version=VERSION, enabled=True,
        pole_xyz=ring['pole'].tolist(), ring_radius_deg=settings['ring_radius_deg'],
        great_circle=bool(abs(settings['ring_radius_deg']-90.) < 1e-12),
        pole_source='explicit' if settings['pole_lat_lon_deg'] is not None else 'area-weighted continental centroid',
        carrier_plate_uid=s.supercontinent_ring_carrier_uid,
        overriding_plate_uid=s.supercontinent_ring_overriding_uid,
        merged_continental_plate_uids=merged_uids,
        minimum_skirt_km=ring['minimum_skirt_km'], requested_minimum_skirt_km=settings['minimum_skirt_km'],
        maximum_land_colatitude_deg=ring['maximum_land_colatitude_deg'],
        continental_area_km2=float(area[ring['land']].sum()),
        skirt_area_km2=float(area[ring['water_inside']].sum()),
        distant_ocean_area_km2=float(area[ring['water_outside']].sum()),
        skirt_age_at_margin_myr=settings['skirt_age_at_margin_myr'],
        skirt_age_at_trench_myr=settings['skirt_age_at_trench_myr'],
        far_ocean_age_myr=settings['far_ocean_age_myr'],
        land_geometry_unchanged=True,
        ownership='All continental material and the water inside the ring form one carrier plate; water outside overrides.',
        cooling_age=('Skirt age varies linearly from margin to trench along the local coast-to-ring path; '
                     'the distant ocean has one declared age. Initial ages only, no runtime spreading.'),
        inherited_structure='Former continental partition cuts remain as material seed-rift identities, not plate boundaries.')
    for event in s.events:
        if event.get('type') == 'initial' and event.get('time_myr') == 0.:
            event['description'] = ('A ring of inherited subduction encloses one supercontinent plate; '
                                    'its own oceanic skirt sinks beneath the distant ocean.')
            event['details'].update(initial_plates=s.count, supercontinent_ring=True,
                                    merged_continental_plate_uids=merged_uids)
    return result


def initialize_controls(s, fitted, intersections):
    """Conservative area-average of declared water ages onto native controls."""
    if not enabled(s):
        return
    source, cell = intersections['material_index'], intersections['control_index']
    areas = np.asarray(intersections['area_km2'])*(np.asarray(fitted['face_kind'])[source] == 0)
    water_area = np.bincount(cell, weights=areas, minlength=s.n)
    age_area = np.bincount(cell, weights=areas*np.asarray(fitted['ocean_age_myr'])[source], minlength=s.n)
    wet = water_area > 1e-12
    s.age[:] = 0.
    s.age[wet] = age_area[wet]/water_area[wet]
    s.plate[:] = np.argmax(s.support, axis=0)
    slots = [int(np.flatnonzero(s.active & (s.plate_uid == uid))[0])
             for uid in (s.supercontinent_ring_carrier_uid, s.supercontinent_ring_overriding_uid)]
    if np.any(~np.isin(slots, s.plate)):
        raise ValueError('The native control mesh cannot resolve both ring plates; refine mesh_level.')
    water = np.asarray(fitted['face_kind']) == 0
    expected = float(np.asarray(fitted['area_km2'])[water]@np.asarray(fitted['ocean_age_myr'])[water])
    represented = float(water_area@s.age)
    if abs(represented-expected) > max(1e-3, expected*1e-10):
        raise ValueError('Ring cooling ages were not conservatively transferred to native controls.')
    s.supercontinent_ring_diagnostics.update(initial_control_age_transfer_error_myr_km2=represented-expected)
    s._record('supercontinent_ring', 'One supercontinent plate is enclosed by a declared ring trench.',
              details=deepcopy(s.supercontinent_ring_diagnostics))


def source_targets(s, source):
    """Source ring interfaces and their incoming owner (always the carrier).

    Only water-water interfaces between the carrier and the overriding ocean
    are ring trench; the carrier's coast is internal to it.
    """
    carrier = int(np.flatnonzero(s.active & (s.plate_uid == s.supercontinent_ring_carrier_uid))[0])
    over = int(np.flatnonzero(s.active & (s.plate_uid == s.supercontinent_ring_overriding_uid))[0])
    owner_a, owner_b = np.asarray(source['owner_a']), np.asarray(source['owner_b'])
    wet = (np.asarray(source['kind_a']) == 0) & (np.asarray(source['kind_b']) == 0)
    pair = ((owner_a == carrier) & (owner_b == over)) | ((owner_a == over) & (owner_b == carrier))
    if np.any(pair & ~wet):
        raise ValueError('Continental material touches the ring trench; enlarge the skirt.')
    return pair & wet, np.full(len(owner_a), carrier, int)


def snapshot(s):
    if not enabled(s):
        return {}
    return dict(supercontinent_ring_version=VERSION,
                supercontinent_ring_diagnostics=deepcopy(s.supercontinent_ring_diagnostics))
