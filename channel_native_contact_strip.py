"""Read-only trench-normal upper strip from a current native contact zone."""

import numpy as np

import channel_entry_zones
import collision_surface
import entry_regions
import mesh_coverage
from channel_native_upper_strip import bind_upper_strip, trench_normal_points


def bind_contact_zone_upper_strip(s, lower_face_id, zone_id,
                                  contact_xyz, offsets_m):
    """Bind one supplied contact quadrature site to nearby upper material.

    The current entry/contact partition proves that the point belongs to the
    named entered-contact zone. We then follow its inherited hinge normal and
    identify each upper material face geometrically. This is a diagnostic
    strip, not an integration rule or a native force/source transaction.
    """
    if (not isinstance(lower_face_id, (int, np.integer))
            or not isinstance(zone_id, str) or not zone_id):
        raise ValueError('Contact strip needs a lower material face and zone ID.')
    point = np.asarray(contact_xyz, float)
    if (point.shape != (3,) or not np.isfinite(point).all()
            or not np.isclose(np.linalg.norm(point), 1., rtol=0., atol=3e-12)):
        raise ValueError('Contact strip needs a unit quadrature point.')
    if not hasattr(s, 'channel_region_store'):
        raise ValueError('Contact strip needs current native regional histories.')
    spec = entry_regions.specification(s)
    if spec is None:
        raise ValueError('Contact strip needs persistent entry material.')
    lower_ids = np.asarray(spec['face_ids'])
    rows = np.flatnonzero(lower_ids == int(lower_face_id))
    if len(rows) != 1:
        raise ValueError('Contact strip needs one active lower entry face.')
    tracked = [record['face_id'] for record in s.channel_region_store['records']]
    records = {int(record['face_id']): record
               for record in s.channel_region_store['records']}
    surface = s.material_surface
    geometry = channel_entry_zones.partition_native_entry_zones(
        surface, s.parcel_collision_sheet, s.parcel_plate, s.plate_uid,
        s.collision_contacts, spec, tracked, epoch_myr=s.t,
        upper_records_by_face_id=records)
    zones = geometry['entry_faces'][int(lower_face_id)]['zones']
    selected = [zone for zone in zones if zone['zone_id'] == zone_id]
    if len(selected) != 1 or selected[0]['kind'] != 'entered-contact':
        raise ValueError('Contact strip needs one current entered-contact zone.')
    zone = selected[0]
    if len(zone['covering_sheets_top_to_bottom']) != 1:
        raise ValueError('Contact strip needs exactly one overriding sheet.')
    planes = mesh_coverage._triangle_planes(zone['polygon'])
    if not np.all(planes @ point >= -1e-11):
        raise ValueError('Contact strip point is outside its current contact zone.')
    upper_sheet = int(zone['adjacent_upper_sheet_id'])
    face_ids = np.asarray(surface['face_id'])
    upper_indices = np.flatnonzero(face_ids == zone['adjacent_upper_face_id'])
    if len(upper_indices) != 1:
        raise ValueError('Contact strip lost its adjacent upper material face.')
    upper_owner = int(s.parcel_plate[upper_indices[0]])
    graph = collision_surface.descendants(s.collision_contacts)
    above = {int(sheet) for sheet, descendants in graph.items()
             if upper_sheet in descendants}
    faces = np.asarray(surface['faces'])
    vertices = np.asarray(surface['vertices'])
    sheets = np.asarray(s.parcel_collision_sheet)
    owners = np.asarray(s.parcel_plate)
    relevant = np.flatnonzero((sheets == upper_sheet) | np.isin(sheets, list(above)))
    relevant_planes = np.asarray([
        mesh_coverage._triangle_planes(vertices[faces[index]])
        for index in relevant])
    def owner_at(sample):
        matching = relevant[np.all(relevant_planes @ sample >= -1e-11, axis=1)]
        if any(int(sheets[index]) in above for index in matching):
            raise ValueError('Contact strip encounters another upper sheet.')
        candidates = [index for index in matching if int(sheets[index]) == upper_sheet]
        if len(candidates) != 1 or int(owners[candidates[0]]) != upper_owner:
            raise ValueError('Contact strip needs one unambiguous upper material face per point.')
        return int(face_ids[candidates[0]])
    if owner_at(point) != int(zone['adjacent_upper_face_id']):
        raise ValueError('Contact strip upper face disagrees with its contact zone.')
    # Compute the path once before binding its identified native material.
    path = trench_normal_points(spec['hinge_normals'][int(rows[0])], point,
                                offsets_m, surface['radius_km'])
    upper_ids = np.array([owner_at(sample) for sample in path['xyz']], np.int64)
    result = bind_upper_strip(s, upper_ids, path['xyz'])
    return dict(result, lower_face_id=int(lower_face_id), zone_id=zone_id,
                adjacent_upper_face_id=int(zone['adjacent_upper_face_id']),
                hinge_xyz=path['hinge_xyz'],
                contact_offset_m=path['contact_offset_m'], xyz=path['xyz'],
                scope='read-only current contact-zone strip; no native force or source commit')
