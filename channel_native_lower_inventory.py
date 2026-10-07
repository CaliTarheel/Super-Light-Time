"""Read-only uniform native lower inventory for projected contact loads.

This compatibility entry point accepts one region only. Mixed native phase
histories must be intersected with contact cells before force assembly.
"""

from channel_native_lower_regions import bind_lower_regions


def bind_uniform_lower_inventory(s, face_id):
    """Return one current native phase column without averaging histories."""
    bound = bind_lower_regions(s, face_id)
    if len(bound['regions']) != 1:
        raise ValueError('Lower inventory needs one uniform regional phase history.')
    region = bound['regions'][0]
    return dict(face_id=bound['face_id'], face_index=bound['face_index'],
                region_id=region['region_id'],
                lower_triangle=bound['lower_triangle'], radius_km=bound['radius_km'],
                physical_area_km2=bound['physical_area_km2'],
                material_reference_area_km2=bound['material_reference_area_km2'],
                physical_volume_km3=region['physical_volume_km3'],
                ordinary_volume_km3=region['ordinary_volume_km3'],
                dense_volume_km3=region['dense_volume_km3'],
                crust_mass_kg=region['crust_mass_kg'],
                mean_density_kg_m3=(region['crust_mass_kg']
                                    / (region['physical_volume_km3'] * 1e9)),
                scope='read-only uniform native lower phase; no contact or source commit')
