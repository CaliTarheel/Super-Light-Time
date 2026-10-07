"""Read-only composition of native entry zones and ordered column sources.

The contact solver must supply each zone's vertical intervals and an
independent thermal depth. This preflight binds their rock masses to current
native regional material, then stages phase and heat without committing any
mechanical work, upper response, mantle return or material ownership.
"""

from copy import copy

import channel_entry_zones
import channel_native_pressure


def prepare_ordered_entry_source(s, paths_by_face_zone, other_loads_by_face,
                                 elapsed_myr, *, constitutive_parameters=None):
    """Stage complete native regional sources after exact contact-zone remap.

    Every entry face needs one explicit path per new zone. Other tracked faces,
    including overriding faces, need their own full region-load lists; this
    function does not silently freeze or evolve their source histories.
    """
    import channel_region_native

    if (not isinstance(paths_by_face_zone, dict)
            or not isinstance(other_loads_by_face, dict)):
        raise ValueError('Ordered native entry source needs complete face load maps.')
    zone_stage = channel_entry_zones.prepare_native_entry_zone_remap(s)
    geometry = zone_stage['geometry']
    entry_ids = set(geometry['entry_faces'])
    records = zone_stage['store']['records']
    tracked = {int(record['face_id']) for record in records}
    if (set(paths_by_face_zone) != entry_ids
            or set(other_loads_by_face) != tracked - entry_ids):
        raise ValueError('Ordered native entry source needs every entry zone and other tracked face.')
    loads_by_face = {}
    pressure_by_face_zone = {}
    rock_by_face = zone_stage['upper_rock']['by_lower_face']
    for record in records:
        face_id = int(record['face_id'])
        if face_id not in entry_ids:
            loads_by_face[face_id] = other_loads_by_face[face_id]
            continue
        zone_rock = {zone['zone_id']: zone for zone in rock_by_face[face_id]}
        supplied = paths_by_face_zone[face_id]
        if (not isinstance(supplied, dict)
                or len(zone_rock) != len(rock_by_face[face_id])
                or set(supplied) != set(zone_rock)):
            raise ValueError('Ordered native entry source needs one path per exact contact zone.')
        pressure = {}
        for zone_id, path in supplied.items():
            if (not isinstance(path, dict)
                    or set(path) != {'lower_top_depth_m', 'segments',
                                     'effective_thermal_depth_km'}):
                raise ValueError('Ordered native entry source needs an independent vertical and thermal path.')
            pressure[zone_id] = channel_native_pressure.pressure_from_measured_rock(
                zone_rock[zone_id], path['lower_top_depth_m'], path['segments'])
        loads = []
        for region in record['history']['regions']:
            zone_id = region['zone_id']
            path = supplied[zone_id]
            measured = pressure[zone_id]
            order = tuple(layer['sheet_id'] for layer in
                          zone_rock[zone_id]['rock_layers_top_to_bottom'])
            loads.append(dict(region_id=region['region_id'],
                              lower_top_depth_m=path['lower_top_depth_m'],
                              segments=measured['validated_segments'],
                              covering_sheets_top_to_bottom=order,
                              effective_thermal_depth_km=
                                  path['effective_thermal_depth_km']))
        loads_by_face[face_id] = loads
        pressure_by_face_zone[face_id] = pressure
    staged = copy(s)
    staged.channel_region_store = zone_stage['store']
    source = channel_region_native.prepare_source(
        staged, loads_by_face, elapsed_myr,
        constitutive_parameters=constitutive_parameters)
    return dict(zone_stage=zone_stage, source_stage=source,
                pressure_by_face_zone=pressure_by_face_zone,
                scope='read-only native ordered entry sources; no mechanics, upper work or owner commit')
