"""Validate native lower phase regions before projected-contact accounting."""

import math

import numpy as np

import column_density
import dense_crust
from channel_region_geometry import spherical_partition_intersections


def bind_lower_regions(s, face_id):
    """Return independent current physical inventories for one native face."""
    import channel_region_native

    if (not isinstance(face_id, (int, np.integer))
            or isinstance(face_id, (bool, np.bool_))
            or not channel_region_native.enabled(s)):
        raise ValueError('Lower inventory needs an installed native material face.')
    surface = s.material_surface
    store = s.channel_region_store
    if (store.get('version') != channel_region_native.VERSION
            or store.get('epoch_myr') != float(s.t)
            or store.get('geometry_revision') != int(surface['geometry_revision'])):
        raise ValueError('Lower inventory needs current native regional geometry.')
    ids = np.asarray(surface['face_id'])
    matches = np.flatnonzero(ids == int(face_id))
    records = [record for record in store['records']
               if int(record['face_id']) == int(face_id)]
    if len(matches) != 1 or len(records) != 1:
        raise ValueError('Lower inventory needs one tracked native face and record.')
    index = int(matches[0])
    record = records[0]
    triangle = np.asarray(surface['vertices'])[surface['faces'][index]]
    if not np.array_equal(np.asarray(record['face_triangle']), triangle):
        raise ValueError('Lower inventory material geometry is stale.')
    history = record['history']
    regions = history['regions']
    if (not isinstance(regions, list) or not regions
            or len(regions) != len(record['polygons'])
            or ('face_id' in history and int(history['face_id']) != int(face_id))
            or ('face_index' in history and int(history['face_index']) != index)):
        raise ValueError('Lower inventory regions lost their native face identity.')
    radius = float(surface.get('radius_km', float('nan')))
    geometry = spherical_partition_intersections(
        triangle, record['polygons'], record['polygons'], radius_km=radius)
    face_area = float(geometry['face_area_km2'])
    fractions = np.asarray([region['fraction'] for region in regions], float)
    if (not np.isfinite(fractions).all()
            or not np.allclose(fractions, geometry['old_fractions'],
                               rtol=0., atol=2e-10)
            or len({region['region_id'] for region in regions}) != len(regions)
            or any(not isinstance(region['region_id'], str)
                   or not region['region_id'] for region in regions)):
        raise ValueError('Lower inventory regions disagree with their material partition.')
    expected_fields = set(s.structure)
    projected = channel_region_native._project(record, expected_fields)
    if any(not np.allclose(projected[name][0], s.structure[name][index],
                           rtol=2e-10, atol=1e-8)
           for name in expected_fields):
        raise ValueError('Lower inventory disagrees with its current native column.')
    bound = []
    for region, polygon, fraction in zip(regions, record['polygons'], fractions):
        column = region['column']
        dense_crust.validate(column)
        if not dense_crust.present(column) or len(column['thickness_km']) != 1:
            raise ValueError('Lower inventory needs one retained-phase column per region.')
        thickness = float(column['thickness_km'][0])
        factor = float(column['area_factor'][0])
        dense_thickness = float(column[dense_crust.DENSE][0]) / factor
        area = face_area * float(fraction)
        if (not np.isfinite([thickness, factor, dense_thickness, area]).all()
                or thickness <= 0. or factor <= 0.
                or not 0. <= dense_thickness <= thickness):
            raise ValueError('Lower inventory area, strain or dense phase is inconsistent.')
        ordinary_volume = (thickness - dense_thickness) * area
        dense_volume = dense_thickness * area
        volume = ordinary_volume + dense_volume
        mass = 1e9 * (column_density.RHO_CRUST * ordinary_volume
                      + column_density.RHO_DENSE * dense_volume)
        equivalent = float(dense_crust.mass_volume(column)[0]) / factor
        if not math.isclose(mass, 1e9 * column_density.RHO_CRUST
                            * equivalent * area, rel_tol=2e-12):
            raise ValueError('Lower inventory lost retained dense-crust mass.')
        bound.append(dict(region_id=region['region_id'],
                          polygon=np.asarray(polygon, float).copy(),
                          physical_area_km2=area,
                          material_reference_area_km2=area / factor,
                          thickness_km=thickness,
                          dense_thickness_km=dense_thickness,
                          physical_volume_km3=volume,
                          ordinary_volume_km3=ordinary_volume,
                          dense_volume_km3=dense_volume,
                          crust_mass_kg=mass))
    reference_area = math.fsum(row['material_reference_area_km2'] for row in bound)
    native_area = float(surface['area_km2'][index])
    native_reference = float(surface['reference_area_km2'][index])
    native_mass = float(s.mass[index])
    if (not np.isfinite([reference_area, native_area,
                         native_reference, native_mass]).all()
            or not math.isclose(native_area, face_area, rel_tol=2e-10)
            or not math.isclose(native_reference, reference_area, rel_tol=2e-10)
            or not math.isclose(native_mass, reference_area, rel_tol=2e-10)):
        raise ValueError('Lower inventory area, strain or dense phase is inconsistent.')
    return dict(face_id=int(face_id), face_index=index,
                lower_triangle=triangle.copy(), radius_km=radius,
                physical_area_km2=face_area,
                material_reference_area_km2=reference_area,
                regions=bound,
                scope='read-only current native phase regions; no contact or source commit')
