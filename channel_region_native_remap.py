"""Atomic read-only native refinement and sibling-coarsening preflight.

This is the unchanged regional remap transaction, separated from store
installation, source evolution and same-topology material motion.
"""

import json

import numpy as np

import crust_inventory
import dense_crust
import mesh_coverage
from channel_upper_basal_reference import inherit_reference
from channel_region_geometry import spherical_partition_intersections
from channel_region_refinement import split_history_to_children


def prepare_remap(s, mapping, new_ids, surface, structure, trace_patch):
    """Stage and audit native child columns before adaptivity commits them."""
    from channel_region_native import (
        VERSION, enabled, coarsening_compatible, _same_column,
        _reference_mass, _project, _markers, _check_trace_alignment)
    if not enabled(s):
        return None
    store = s.channel_region_store
    if (store.get('version') != VERSION or store.get('epoch_myr') != float(s.t)
            or store.get('geometry_revision') != int(s.material_surface['geometry_revision'])):
        raise ValueError('Regional source state is stale for native material remapping.')
    old_ids = np.asarray(s.material_surface['face_id'])
    by_id = {int(face_id): index for index, face_id in enumerate(old_ids)}
    records = []
    if 'source_face' in mapping:
        for record in store['records']:
            old_index = by_id[int(record['face_id'])]
            new_indices = np.flatnonzero(mapping['source_face'] == old_index)
            if len(new_indices) == 0:
                raise ValueError('Regional source face disappeared during native remapping.')
            children = [dict(face_id=int(new_ids[index]), face_index=int(index),
                             triangle=surface['vertices'][surface['faces'][index]])
                        for index in new_indices]
            records.extend(split_history_to_children(record, children,
                           radius_km=float(surface.get('radius_km', 6371.)))['records'])
    else:
        if not coarsening_compatible(s, mapping):
            raise ValueError('Regional coarsening would mix different source histories.')
        tracked = {int(record['face_id']): record for record in store['records']}
        starts = mapping['source_indptr']
        for index in range(len(new_ids)):
            sources = mapping['source_indices'][starts[index]:starts[index+1]]
            group = [tracked.get(int(old_ids[source])) for source in sources]
            if not any(record is not None for record in group):
                continue
            triangle = surface['vertices'][surface['faces'][index]]
            child = dict(face_id=int(new_ids[index]), face_index=int(index),
                         triangle=triangle)
            if len(sources) == 1:
                records.extend(split_history_to_children(group[0], [child],
                               radius_km=float(surface.get('radius_km', 6371.)))['records'])
                continue
            radius = float(surface.get('radius_km', 6371.))
            old_polygons = [polygon for record in group for polygon in record['polygons']]
            spherical_partition_intersections(
                triangle, old_polygons, old_polygons, radius_km=radius)
            first = group[0]['history']['regions'][0]
            exact = (all(len(record['history']['regions']) == 1 for record in group)
                     and all(record['history']['regions'][0].get(
                         'zone_id', record['history']['regions'][0]['region_id'])
                         == first.get('zone_id', first['region_id'])
                         and _same_column(first['column'],
                                          record['history']['regions'][0]['column'])
                         and inherit_reference(first) == inherit_reference(
                             record['history']['regions'][0])
                         for record in group[1:]))
            if exact:
                union_id = json.dumps(['native-exact-union', sorted(
                    record['history']['regions'][0]['region_id'] for record in group)],
                    separators=(',', ':'))
                regions = [dict(region_id=union_id,
                                zone_id=first.get('zone_id', first['region_id']),
                                fraction=1.,
                                column={key: value.copy() for key, value in first['column'].items()},
                                **inherit_reference(first))]
                polygons = [triangle.copy()]
            else:
                parent_area = mesh_coverage._polygon_area(triangle, radius)
                regions = []
                polygons = []
                for record in group:
                    for region, polygon in zip(record['history']['regions'], record['polygons']):
                        regions.append(dict(
                            region_id=json.dumps(['native-retain', int(record['face_id']),
                                                  region['region_id']], separators=(',', ':')),
                            zone_id=region.get('zone_id', region['region_id']),
                            fraction=float(mesh_coverage._polygon_area(polygon, radius) / parent_area),
                            column={key: value.copy() for key, value in region['column'].items()},
                            **inherit_reference(region)))
                        polygons.append(np.asarray(polygon).copy())
            records.append(dict(face_id=child['face_id'], face_triangle=triangle.copy(),
                                history=dict(face_id=child['face_id'], face_index=index,
                                             regions=regions,
                                             scope='native sibling geometry coarsening with local history retained'),
                                polygons=polygons))
    fields = set(structure)
    staged = {name: value.copy() for name, value in structure.items()}
    new_mass = np.asarray(surface['reference_area_km2']).copy()
    radius = float(surface.get('radius_km', 6371.))
    for record in records:
        index = record['history']['face_index']
        new_mass[index] = _reference_mass(record, radius)
        mean = _project(record, fields)
        for name in fields:
            staged[name][index] = mean[name][0]
    crust_inventory.validate(staged)
    dense_crust.validate(staged)
    old_mass = np.asarray(s.mass)
    if not np.isclose(old_mass.sum(), new_mass.sum(), rtol=2e-12, atol=1e-6):
        raise ValueError('Native regional remap changed reference mass.')
    for name in (*crust_inventory.FIELDS, *dense_crust.FIELDS):
        if not np.isclose(old_mass @ s.structure[name], new_mass @ staged[name],
                          rtol=2e-12, atol=1e-6):
            raise ValueError('Native regional remap changed conserved crust or heat: ' + name)
    old_volume = float(old_mass @ (s.structure['area_factor'] * s.structure['thickness_km']))
    new_volume = float(new_mass @ (staged['area_factor'] * staged['thickness_km']))
    if not np.isclose(old_volume, new_volume, rtol=2e-12, atol=1e-5):
        raise ValueError('Native regional remap changed column volume.')
    old_bound = _markers(s, [record['face_id'] for record in store['records']])
    _check_trace_alignment(s, old_bound)
    old_trace_ids = {row['trace_id'] for row in old_bound}
    new_trace_ids = {row['trace_id'] for row in _markers(
        s, [record['face_id'] for record in records], surface=surface,
        trace_patch=trace_patch, records=records)}
    if old_trace_ids != new_trace_ids:
        raise ValueError('Native regional remap lost a material trace marker.')
    return dict(structure=staged, mass=new_mass,
                store=dict(version=VERSION, epoch_myr=float(s.t),
                           geometry_revision=int(surface['geometry_revision']),
                           records=records), remapped_markers=len(new_trace_ids))


