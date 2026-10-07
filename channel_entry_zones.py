"""Read-only binding of regional entry zones to persistent native material.

This supplies face and contact identity, not vertical loads, heat, or a source
commit. Ordinary entry mechanics still reject active entry/stack overlap.
"""

import numpy as np

import channel_pair_zones
import channel_upper_region_zones
import channel_upper_rock_loads
import channel_native_pressure  # Keep explicit vertical-load assembly in run archives.
import collision_surface
import entry_regions


def partition_native_entry_zones(surface, sheets, owners, plate_uids, contacts,
                                 entry_spec, tracked_face_ids, *, epoch_myr,
                                 upper_records_by_face_id=None):
    """Derive complete tracked-face zones from current material and contact IDs.

    ``entry_spec`` is the validated output of ``entry_regions.specification``.
    Every active entry face must be tracked. An adjacent upper contact must be
    current, ordered, and owned by that entry's recorded overriding plate.
    """
    face_ids = np.asarray(surface['face_id'])
    faces = np.asarray(surface['faces'])
    vertices = np.asarray(surface['vertices'])
    sheets = np.asarray(sheets)
    owners = np.asarray(owners)
    plate_uids = np.asarray(plate_uids)
    tracked = np.asarray(tracked_face_ids)
    n = len(face_ids)
    if (face_ids.shape != (n,) or face_ids.dtype.kind not in 'iu'
            or len(np.unique(face_ids)) != n or faces.shape != (n, 3)
            or sheets.shape != (n,) or sheets.dtype.kind not in 'iu'
            or np.any(sheets <= 0) or owners.shape != (n,)
            or owners.dtype.kind not in 'iu' or np.any(owners < 0)
            or np.any(owners >= len(plate_uids))
            or plate_uids.ndim != 1 or plate_uids.dtype.kind not in 'iu'
            or len(np.unique(plate_uids)) != len(plate_uids)
            or tracked.ndim != 1 or tracked.dtype.kind not in 'iu'
            or len(np.unique(tracked)) != len(tracked)
            or not set(map(int, tracked)) <= set(map(int, face_ids))
            or not np.isfinite(epoch_myr)):
        raise ValueError('Native entry zones need aligned, unique material identities.')
    if entry_spec is None:
        raise ValueError('Native entry zones need active persistent entry material.')
    entry_ids = np.asarray(entry_spec['face_ids'])
    keys = ('hinge_normals', 'finite_midpoints', 'finite_half_lengths_km',
            'overriding_plate_uids')
    if (entry_ids.ndim != 1 or entry_ids.dtype.kind not in 'iu'
            or len(np.unique(entry_ids)) != len(entry_ids)
            or not set(map(int, entry_ids)) <= set(map(int, tracked))
            or any(len(np.asarray(entry_spec[key])) != len(entry_ids)
                   for key in keys)):
        raise ValueError('Every active entry face needs one tracked hinge history.')
    by_id = {int(face_id): index for index, face_id in enumerate(face_ids)}
    entry = {int(face_id): row for row, face_id in enumerate(entry_ids)}
    graph = collision_surface.descendants(contacts)
    pairs = collision_surface.ordered_overlap_pairs(surface, sheets, graph)
    upper_by_lower = {}
    for upper, lower in zip(pairs['upper_face'], pairs['lower_face']):
        upper_by_lower.setdefault(int(lower), []).append(int(upper))
    upper_faces = set(map(int, pairs['upper_face']))
    active = {}
    for contact in contacts:
        if (contact.get('state') != 'active'
                or contact.get('last_seen_myr') != float(epoch_myr)):
            continue
        key = (int(contact['top_sheet']), int(contact['under_sheet']))
        if key in active:
            raise ValueError('Native entry contacts have duplicate sheet identities.')
        active[key] = float(contact['overlap_area_km2'])
    observed = {}
    for upper, lower, area in zip(pairs['upper_face'], pairs['lower_face'],
                                  pairs['area_km2']):
        key = (int(sheets[upper]), int(sheets[lower]))
        observed[key] = observed.get(key, 0.) + float(area)
    def current_contact(key):
        return (key in active and key in observed
                and np.isclose(active[key], observed[key], rtol=2e-9, atol=1e-6))
    triangles = vertices[faces]
    triangles_by_id = {int(face_id): triangles[index]
                       for index, face_id in enumerate(face_ids)}
    radius = float(surface.get('radius_km', 6371.))
    zones_by_face = {}
    ledgers = {}
    for face_id in map(int, tracked):
        index = by_id[face_id]
        if face_id not in entry:
            zones_by_face[face_id] = [dict(zone_id='unentered-0',
                                           polygon=triangles[index].copy())]
            continue
        row = entry[face_id]
        if index in upper_faces:
            raise ValueError('Entered face is above another material sheet; polarity handoff is unresolved.')
        upper_indices = sorted(set(upper_by_lower.get(index, ())),
                               key=lambda upper: (int(sheets[upper]), int(face_ids[upper])))
        for upper in upper_indices:
            key = (int(sheets[upper]), int(sheets[index]))
            if not current_contact(key):
                raise ValueError('Entry contact is not current, active and geometrically matched.')
        half_length = float(entry_spec['finite_half_lengths_km'][row])
        finite = np.isfinite(half_length)
        report = channel_pair_zones.partition_entry_stack(
            triangles[index], triangles[upper_indices], sheets[upper_indices],
            int(sheets[index]), graph, entry_spec['hinge_normals'][row],
            radius_km=radius,
            finite_midpoint=(entry_spec['finite_midpoints'][row] if finite else None),
            finite_half_length_km=(half_length if finite else None),
            upper_face_ids=face_ids[upper_indices])
        report = dict(report, zones=channel_upper_region_zones.refine_by_upper_regions(
            report['zones'], triangles_by_id,
            {} if upper_records_by_face_id is None else upper_records_by_face_id,
            report['face_area_km2'], radius_km=radius))
        overriding_uid = int(entry_spec['overriding_plate_uids'][row])
        for zone in report['zones']:
            if zone['kind'] != 'entered-contact':
                continue
            stack = zone['covering_sheets_top_to_bottom']
            if any(not current_contact((top, below))
                   for top, below in zip(stack, stack[1:])):
                raise ValueError('Covering entry stack has a stale upper-sheet contact.')
            adjacent = by_id[zone['adjacent_upper_face_id']]
            if int(plate_uids[owners[adjacent]]) != overriding_uid:
                raise ValueError('Entry contact needs an explicit overriding-owner handoff.')
        zones_by_face[face_id] = [dict(zone_id=zone['zone_id'],
                                       polygon=zone['polygon'])
                                  for zone in report['zones']]
        ledgers[face_id] = report
    return dict(zones_by_face=zones_by_face, entry_faces=ledgers,
                scope='native face/contact geometry only; no basal depth, heat or source commit')


def prepare_native_entry_zone_remap(s):
    """Compose checked native entry geometry with the regional remap stage."""
    import channel_region_native

    if not channel_region_native.enabled(s) or not entry_regions.enabled(s):
        raise ValueError('Native entry remap needs installed regions and persistent entry.')
    surface = s.material_surface
    state = s.continental_entry_regions
    if (not np.array_equal(state['face_ids'], s.parcel_patch)
            or not np.array_equal(surface['face_id'], s.parcel_patch)
            or np.asarray(s.parcel_entry_region).shape != np.asarray(s.parcel_patch).shape):
        raise ValueError('Native entry remap cannot repair stale material alignment.')
    spec = entry_regions.specification(s)
    tracked = [record['face_id'] for record in s.channel_region_store['records']]
    upper_records = {int(record['face_id']): record
                     for record in s.channel_region_store['records']}
    geometry = partition_native_entry_zones(
        surface, s.parcel_collision_sheet, s.parcel_plate, s.plate_uid,
        s.collision_contacts, spec, tracked, epoch_myr=s.t,
        upper_records_by_face_id=upper_records)
    stage = channel_region_native.prepare_zone_remap(s, geometry['zones_by_face'])
    rock = channel_upper_rock_loads.measure_native_upper_rock(s, geometry)
    return dict(stage, geometry=geometry, upper_rock=rock)
