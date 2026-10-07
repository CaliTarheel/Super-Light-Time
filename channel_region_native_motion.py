"""Read-only native material motion with local regional strain and traces."""

import numpy as np

import crust_inventory
import crustal_structure
import dense_crust
from channel_upper_basal_reference import inherit_reference
from channel_region_geometry import (
    spherical_partition_intersections, transport_material_polygon)


def prepare_motion(s, moved_surface, moved_trace_xyz):
    """Audit a same-topology local area change without committing a timestep.

    Each material subregion keeps its own reference area and composition. Its
    physical area change adjusts its area factor and inverse thickness before
    the face is projected. Native mechanics, source heating and entry/owner
    transfer still need a coupled transaction before this stage can be used.
    """
    from channel_region_native import (
        VERSION, enabled, _reference_mass, _project, _markers,
        _check_trace_alignment, _trace_rows)
    if not enabled(s):
        raise ValueError('Regional motion needs an installed native store.')
    old = s.material_surface
    store = s.channel_region_store
    if (store.get('version') != VERSION or store.get('epoch_myr') != float(s.t)
            or store.get('geometry_revision') != int(old['geometry_revision'])
            or moved_surface['geometry_revision'] != old['geometry_revision'] + 1
            or moved_surface['radius_km'] != old['radius_km']
            or not np.array_equal(moved_surface['face_id'], old['face_id'])
            or not np.array_equal(moved_surface['faces'], old['faces'])
            or not np.array_equal(moved_surface['face_owner'], old['face_owner'])
            or not np.array_equal(moved_surface['vertex_owner'], old['vertex_owner'])
            or np.asarray(moved_trace_xyz).shape != np.asarray(s.trace_xyz).shape
            or not np.array_equal(moved_surface['reference_area_km2'],
                                  old['reference_area_km2'])):
        raise ValueError('Regional motion needs fresh same-topology material geometry and markers.')
    radius = float(old['radius_km'])
    by_id = {int(face_id): index for index, face_id in enumerate(old['face_id'])}
    staged = {name: value.copy() for name, value in s.structure.items()}
    records = []
    for record in store['records']:
        face_id = int(record['face_id'])
        index = by_id[face_id]
        old_triangle = old['vertices'][old['faces'][index]]
        new_triangle = moved_surface['vertices'][moved_surface['faces'][index]]
        if (record['history'].get('face_index') != index
                or not np.array_equal(record['face_triangle'], old_triangle)):
            raise ValueError('Regional source geometry is stale for material motion.')
        old_geometry = spherical_partition_intersections(
            old_triangle, record['polygons'], record['polygons'], radius_km=radius)
        polygons = [transport_material_polygon(
            old_triangle, new_triangle, polygon, radius_km=radius)
            for polygon in record['polygons']]
        new_geometry = spherical_partition_intersections(
            new_triangle, polygons, polygons, radius_km=radius)
        if (not np.isclose(old_geometry['face_area_km2'], old['area_km2'][index],
                           rtol=2e-12, atol=1e-6)
                or not np.isclose(new_geometry['face_area_km2'],
                                  moved_surface['area_km2'][index],
                                  rtol=2e-12, atol=1e-6)):
            raise ValueError('Regional motion geometry and face areas disagree.')
        old_fractions = old_geometry['old_fractions']
        if not np.allclose(old_fractions,
                           [region['fraction'] for region in record['history']['regions']],
                           rtol=0., atol=2e-10):
            raise ValueError('Regional source fractions are stale for material motion.')
        regions = []
        for region, old_fraction, new_fraction in zip(
                record['history']['regions'], old_fractions,
                new_geometry['old_fractions']):
            old_area = old_geometry['face_area_km2'] * old_fraction
            new_area = new_geometry['face_area_km2'] * new_fraction
            ratio = new_area / old_area
            column = {key: value.copy() for key, value in region['column'].items()}
            column['area_factor'] *= ratio
            column['thickness_km'] /= ratio
            crustal_structure.elevation(column)
            regions.append(dict(region_id=region['region_id'],
                                zone_id=region.get('zone_id', region['region_id']),
                                fraction=float(new_fraction), column=column,
                                **inherit_reference(region)))
        moved = dict(face_id=face_id, face_triangle=new_triangle.copy(),
                     history=dict(face_id=face_id, face_index=index, regions=regions,
                                  scope='read-only material motion; no native source commit'),
                     polygons=polygons)
        if (not np.isclose(_reference_mass(record, radius), s.mass[index],
                           rtol=2e-10, atol=1e-6)
                or not np.isclose(_reference_mass(moved, radius), s.mass[index],
                                  rtol=2e-10, atol=1e-6)):
            raise ValueError('Regional motion changed material reference area.')
        mean = _project(moved, set(staged))
        for name in staged:
            staged[name][index] = mean[name][0]
        if not np.isclose(
                old_geometry['face_area_km2'] * s.structure['thickness_km'][index],
                new_geometry['face_area_km2'] * staged['thickness_km'][index],
                rtol=2e-10, atol=1e-5):
            raise ValueError('Regional motion changed physical column volume.')
        records.append(moved)
    crust_inventory.validate(staged)
    dense_crust.validate(staged)
    for name in (*crust_inventory.FIELDS, *dense_crust.FIELDS):
        if not np.isclose(s.mass @ s.structure[name], s.mass @ staged[name],
                          rtol=2e-10, atol=1e-6):
            raise ValueError('Regional motion changed conserved crust or heat: ' + name)
    old_bound = _markers(s, [record['face_id'] for record in store['records']])
    _check_trace_alignment(s, old_bound)
    old_markers = {row['trace_id'] for row in old_bound}
    new_bound = _markers(
        s, [record['face_id'] for record in records], surface=moved_surface,
        records=records, trace_xyz=moved_trace_xyz)
    new_markers = {row['trace_id'] for row in new_bound}
    if old_markers != new_markers:
        raise ValueError('Regional motion lost a material trace marker.')
    traces = {name: value.copy() for name, value in s.trace_structure.items()}
    for index, column in _trace_rows(s, new_bound):
        for name in traces:
            traces[name][index] = column[name][0]
    dense_crust.validate(traces)
    return dict(structure=staged,
                trace_structure=traces,
                store=dict(version=VERSION, epoch_myr=float(s.t),
                           geometry_revision=int(moved_surface['geometry_revision']),
                           records=records), remapped_markers=len(new_markers),
                scope='read-only material motion; native source and mechanics not committed')


