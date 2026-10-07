"""Material-attached regional column histories for the isolated channel source.

The caller supplies old/new partitions and load histories. This module keeps
local phase/heat columns distinct and conservatively remaps them; it does not
commit a native timestep or infer a contact pressure or thermal path.
"""

import json

import numpy as np

import crustal_structure
import dense_crust
from channel_upper_basal_reference import inherit_reference
from channel_region_geometry import (
    spherical_partition_intersections, transport_material_polygon,
    material_intersection_cells)
from entry_channel_column import advance_column, advance_ordered_column, _coalesce_samples

def initialize_region_history(state, face_index, regions, *, face_id=None):
    """Copy a fixed disjoint face partition into independently evolving columns.

    Stable string IDs and physical-area fractions are supplied by the caller;
    this oracle cannot discover, move or conservatively remap material regions.
    """
    dense_crust.validate(state)
    if (not isinstance(face_index, (int, np.integer))
            or not 0 <= face_index < len(state['thickness_km'])
            or not isinstance(regions, (list, tuple)) or not regions):
        raise ValueError('Region history needs an existing face and nonempty partition.')
    if any(not isinstance(region, dict)
           or set(region) != {'region_id', 'fraction'}
           or not isinstance(region['region_id'], str)
           or not region['region_id'] for region in regions):
        raise ValueError('Every source region needs a stable string ID and fraction.')
    ids = [region['region_id'] for region in regions]
    if len(set(ids)) != len(ids):
        raise ValueError('Source region IDs must be unique.')
    if face_id is not None and (not isinstance(face_id, (int, np.integer)) or face_id < 0):
        raise ValueError('Persistent source face ID must be a nonnegative integer.')
    weights = np.asarray([region['fraction'] for region in regions], float)
    if (not np.isfinite(weights).all() or np.any(weights <= 0.)
            or abs(float(weights.sum()) - 1.) > 2e-10):
        raise ValueError('Source history fractions must partition one face.')
    identity = dict(face_index=int(face_index))
    if face_id is not None:
        identity['face_id'] = int(face_id)
    return dict(**identity, regions=[
        dict(region_id=region['region_id'], zone_id=region['region_id'],
             fraction=float(weight),
             column={key: np.asarray(value)[face_index:face_index+1].copy()
                     for key, value in state.items()})
        for region, weight in zip(regions, weights)],
        scope='fixed material-region histories only; no geometry remap or native commit')


def advance_region_history(history, loads, elapsed_myr, *,
                           constitutive_parameters=None, face_area_km2=None):
    """Advance persistent fixed-area regions without erasing subface history.

    The complete ID-to-load map must be supplied for each step. A changed
    partition or exchanged material needs a separate conservative remap and is
    rejected here by construction. The input history is never mutated.
    """
    if (not isinstance(history, dict) or not isinstance(history.get('regions'), list)
            or not history['regions'] or not isinstance(loads, (list, tuple))):
        raise ValueError('Source history needs fixed regions and a load for each.')
    regions = history['regions']
    ids = [region['region_id'] for region in regions]
    weights = np.asarray([region['fraction'] for region in regions], float)
    if (len(set(ids)) != len(ids) or not np.isfinite(weights).all()
            or np.any(weights <= 0.) or abs(float(weights.sum()) - 1.) > 2e-10):
        raise ValueError('Source history has an invalid fixed partition.')
    common = {'region_id', 'lower_top_depth_m', 'effective_thermal_depth_km'}
    single = common | {'current_upper_base_depth_m',
                       'upper_column_mass_kg_m2'}
    ordered = common | {'segments', 'covering_sheets_top_to_bottom'}
    def valid_load(load):
        return (isinstance(load, dict)
                and (single <= load.keys() <= single | {'water_depth_m'}
                     or set(load) == ordered))
    if (len(loads) != len(regions) or any(
            not valid_load(load) for load in loads)
            or {load['region_id'] for load in loads} != set(ids)):
        raise ValueError('Source history needs one pressure and thermal path per region ID.')
    if face_area_km2 is not None and (
            not np.isscalar(face_area_km2) or not np.isfinite(face_area_km2)
            or face_area_km2 <= 0.):
        raise ValueError('Source history face area must be positive and finite.')
    by_id = {load['region_id']: load for load in loads}
    samples = []
    for region in regions:
        dense_crust.validate(region['column'])
        if len(region['column']['thickness_km']) != 1:
            raise ValueError('Each source history region needs exactly one column.')
        load = by_id[region['region_id']]
        if 'segments' in load:
            sample = advance_ordered_column(
                region['column'], 0, load['lower_top_depth_m'],
                load['segments'], load['covering_sheets_top_to_bottom'],
                load['effective_thermal_depth_km'], elapsed_myr,
                constitutive_parameters=constitutive_parameters)
        else:
            sample = advance_column(
                region['column'], 0, load['lower_top_depth_m'],
                load['current_upper_base_depth_m'],
                load['upper_column_mass_kg_m2'],
                load['effective_thermal_depth_km'], elapsed_myr,
                water_depth_m=load.get('water_depth_m', 0.),
                constitutive_parameters=constitutive_parameters)
        samples.append(sample)
    merged = _coalesce_samples(samples, weights)
    returned = float(weights @ [sample['returned_km'] for sample in samples])
    result = dict(
        history=dict(**_face_identity(history), regions=[
            dict(region_id=region['region_id'], fraction=region['fraction'],
                 zone_id=region.get('zone_id', region['region_id']),
                 column=sample['column'], **inherit_reference(region))
            for region, sample in zip(regions, samples)], scope=history.get('scope')),
        column=merged, returned_km=returned,
        contraction_km=float(weights @ [sample['contraction_km'] for sample in samples]),
        withheld_equivalent_km=float(weights @ [
            sample['withheld_equivalent_km'] for sample in samples]),
        top_pressure_pa=np.array([sample['top_pressure_pa'] for sample in samples]),
        maximum_ordinary_pressure_pa=max(
            sample['maximum_ordinary_pressure_pa'] for sample in samples),
        region_count=len(samples),
        scope='read-only fixed subface histories; no moving remap or native commit',
    )
    if face_area_km2 is not None:
        result['returned_volume_km3'] = returned * float(face_area_km2)
    return result


def remap_region_history(history, new_zones, overlap_fractions):
    """Split histories by caller-supplied old/new physical-area intersections.

    Rows are old material regions; columns are new contact zones. Each positive
    intersection inherits an independent copy of its old material column.
    Mixing old columns before applying nonlinear sources would discard local
    history. This is a read-only remap; it does not calculate moving geometry.
    """
    if (not isinstance(history, dict) or not isinstance(history.get('regions'), list)
            or not history['regions'] or not isinstance(new_zones, (list, tuple))
            or not new_zones):
        raise ValueError('Moving source remap needs old regions and new zones.')
    old = history['regions']
    old_ids = [region['region_id'] for region in old]
    if len(set(old_ids)) != len(old_ids):
        raise ValueError('Old source region IDs must be unique.')
    if any(not isinstance(zone, dict) or set(zone) != {'zone_id', 'fraction'}
           or not isinstance(zone['zone_id'], str) or not zone['zone_id']
           for zone in new_zones):
        raise ValueError('Each new source zone needs a stable string ID and fraction.')
    zone_ids = [zone['zone_id'] for zone in new_zones]
    if len(set(zone_ids)) != len(zone_ids):
        raise ValueError('New source zone IDs must be unique.')
    old_weight = np.asarray([region['fraction'] for region in old], float)
    new_weight = np.asarray([zone['fraction'] for zone in new_zones], float)
    overlap = np.asarray(overlap_fractions, float)
    if (overlap.shape != (len(old), len(new_zones))
            or not np.isfinite(old_weight).all() or not np.isfinite(new_weight).all()
            or not np.isfinite(overlap).all() or np.any(old_weight <= 0.)
            or np.any(new_weight <= 0.) or np.any(overlap < 0.)
            or abs(float(old_weight.sum()) - 1.) > 2e-10
            or abs(float(new_weight.sum()) - 1.) > 2e-10
            or not np.allclose(overlap.sum(axis=1), old_weight,
                               rtol=0., atol=2e-10)
            or not np.allclose(overlap.sum(axis=0), new_weight,
                               rtol=0., atol=2e-10)):
        raise ValueError('Old/new overlap areas must exactly partition both physical faces.')
    result = []
    for i, region in enumerate(old):
        dense_crust.validate(region['column'])
        if len(region['column']['thickness_km']) != 1:
            raise ValueError('Each remapped source history needs one column.')
        for j, zone in enumerate(new_zones):
            area = float(overlap[i, j])
            if area > 0.:
                result.append(dict(
                    region_id=json.dumps([region['region_id'], zone['zone_id']],
                                         separators=(',', ':')),
                    zone_id=zone['zone_id'], fraction=area,
                    column={key: value.copy() for key, value in region['column'].items()},
                    **inherit_reference(region)))
    return dict(**_face_identity(history), regions=result,
                scope='read-only exact intersection split; no geometry inference, coarsening or native commit')


def expand_zone_loads(history, zone_loads):
    """Assign one zone pressure/thermal path to every inherited child region."""
    if (not isinstance(history, dict) or not isinstance(history.get('regions'), list)
            or not history['regions'] or not isinstance(zone_loads, (list, tuple))):
        raise ValueError('Zone loads need nonempty remapped source history.')
    expected = {region.get('zone_id', region['region_id'])
                for region in history['regions']}
    if (len(zone_loads) != len(expected) or any(
            not isinstance(load, dict) or 'zone_id' not in load
            or not isinstance(load['zone_id'], str) for load in zone_loads)
            or {load['zone_id'] for load in zone_loads} != expected):
        raise ValueError('Zone loads must match every new contact zone exactly once.')
    by_zone = {load['zone_id']: load for load in zone_loads}
    return [dict(region_id=region['region_id'], **{
        key: value for key, value in by_zone[
            region.get('zone_id', region['region_id'])].items()
        if key != 'zone_id'}) for region in history['regions']]


def remap_history_from_polygons(history, face_triangle, old_polygons,
                                new_zones, *, radius_km=6371.):
    """Bind co-registered spherical intersection geometry to source histories."""
    if (not isinstance(history, dict) or not isinstance(history.get('regions'), list)
            or not isinstance(old_polygons, (list, tuple))
            or len(old_polygons) != len(history['regions'])
            or not isinstance(new_zones, (list, tuple)) or not new_zones
            or any(not isinstance(zone, dict)
                   or set(zone) != {'zone_id', 'polygon'} for zone in new_zones)):
        raise ValueError('Source polygons must align with old regions and named new zones.')
    geometry = spherical_partition_intersections(
        face_triangle, old_polygons, [zone['polygon'] for zone in new_zones],
        radius_km=radius_km)
    old_fractions = np.asarray([
        region['fraction'] for region in history['regions']], float)
    if not np.allclose(old_fractions, geometry['old_fractions'],
                       rtol=0., atol=2e-10):
        raise ValueError('Old source polygon areas do not match saved material fractions.')
    zones = [dict(zone_id=zone['zone_id'], fraction=float(fraction))
             for zone, fraction in zip(new_zones, geometry['new_fractions'])]
    remapped = remap_region_history(
        history, zones, geometry['overlap_fractions'])
    return dict(history=remapped, geometry=geometry)


def remap_history_after_face_motion(history, old_face_triangle,
                                    new_face_triangle, old_polygons,
                                    new_zones, *, radius_km=6371.):
    """Transport old material polygons before the new-frame overlap audit.

    Rigid plate motion preserves their physical fractions. Nonrigid subface
    strain is rejected by the saved-fraction audit until its local mass/heat
    accounting has a material-reference measure.
    """
    if not isinstance(old_polygons, (list, tuple)):
        raise ValueError('Old material polygons must be an ordered partition.')
    transported = [transport_material_polygon(
        old_face_triangle, new_face_triangle, polygon, radius_km=radius_km)
        for polygon in old_polygons]
    report = remap_history_from_polygons(
        history, new_face_triangle, transported, new_zones,
        radius_km=radius_km)
    report['transported_old_polygons'] = transported
    return report


def remap_history_with_local_strain(history, old_face_triangle,
                                    new_face_triangle, old_polygons,
                                    new_zones, *, radius_km=6371.):
    """Split material cells and conserve crust/heat under their local area change.

    This pays only the geometric area-factor and inverse thickness transaction.
    Native strain heating, flexure, mechanics and source-epoch commit are not
    inferred from a polygon map.
    """
    if (not isinstance(history, dict) or not isinstance(history.get('regions'), list)
            or not isinstance(old_polygons, (list, tuple))
            or len(old_polygons) != len(history['regions'])
            or not isinstance(new_zones, (list, tuple)) or not new_zones
            or any(not isinstance(zone, dict)
                   or set(zone) != {'zone_id', 'polygon'}
                   or not isinstance(zone['zone_id'], str)
                   or not zone['zone_id'] for zone in new_zones)
            or len({zone['zone_id'] for zone in new_zones}) != len(new_zones)):
        raise ValueError('Local strain remap needs aligned old regions and named new zones.')
    geometry = material_intersection_cells(
        old_face_triangle, new_face_triangle, old_polygons,
        [zone['polygon'] for zone in new_zones], radius_km=radius_km)
    old_fractions = np.asarray([
        region['fraction'] for region in history['regions']], float)
    if not np.allclose(old_fractions,
                       geometry['old_geometry']['old_fractions'],
                       rtol=0., atol=2e-10):
        raise ValueError('Old source polygon areas do not match saved material fractions.')
    old_area = geometry['old_geometry']['face_area_km2']
    new_area = geometry['new_geometry']['face_area_km2']
    children = []
    for cell in geometry['cells']:
        parent = history['regions'][cell['old_index']]
        column = {key: value.copy() for key, value in parent['column'].items()}
        ratio = cell['old_area_km2'] / cell['new_area_km2']
        column['thickness_km'] *= ratio
        column['area_factor'] /= ratio
        # Validate the same material floor/ceiling used by native columns.
        crustal_structure.elevation(column)
        children.append(dict(
            region_id=json.dumps([
                parent['region_id'], new_zones[cell['new_index']]['zone_id']],
                separators=(',', ':')),
            zone_id=new_zones[cell['new_index']]['zone_id'],
            fraction=cell['new_area_km2'] / new_area, column=column,
            **inherit_reference(parent)))
    old_mass = sum(
        old_area * fraction * dense_crust.mass_volume(region['column'])[0]
        / region['column']['area_factor'][0]
        for fraction, region in zip(old_fractions, history['regions']))
    new_mass = sum(
        cell['new_area_km2'] * dense_crust.mass_volume(region['column'])[0]
        / region['column']['area_factor'][0]
        for cell, region in zip(geometry['cells'], children))
    old_heat = sum(
        old_area * fraction * region['column'][dense_crust.HEAT][0]
        / region['column']['area_factor'][0]
        for fraction, region in zip(old_fractions, history['regions']))
    new_heat = sum(
        cell['new_area_km2'] * region['column'][dense_crust.HEAT][0]
        / region['column']['area_factor'][0]
        for cell, region in zip(geometry['cells'], children))
    if (not np.isclose(old_mass, new_mass, rtol=2e-10, atol=1e-12)
            or not np.isclose(old_heat, new_heat, rtol=2e-10, atol=1e-9)):
        raise ValueError('Local strain remap changed conserved crust or heat inventory.')
    return dict(
        history=dict(**_face_identity(history), regions=children,
                     scope='read-only local strain and source-history remap; no native commit'),
        geometry=geometry,
        old_crust_volume_km3=float(old_mass), new_crust_volume_km3=float(new_mass),
        old_stored_heat_km3_c=float(old_heat),
        new_stored_heat_km3_c=float(new_heat),
        scope='conservative material area/thickness transaction only; no native mechanics or checkpoint',
    )


def coalesce_identical_region_histories(history):
    """Reduce child count only for exactly equal columns in the same zone.

    This is exact source-state equivalence, not tolerance-based phase mixing.
    Regions in different contact zones can receive different future loads and
    must remain separate even if their columns happen to match today.
    """
    if (not isinstance(history, dict) or not isinstance(history.get('regions'), list)
            or not history['regions']):
        raise ValueError('Exact source coalescence needs nonempty region history.')
    rows = history['regions']
    ids = [region['region_id'] for region in rows]
    weights = np.asarray([region['fraction'] for region in rows], float)
    if (len(set(ids)) != len(ids) or not np.isfinite(weights).all()
            or np.any(weights <= 0.) or abs(float(weights.sum()) - 1.) > 2e-10):
        raise ValueError('Exact source coalescence needs unique IDs and complete area.')

    def identical(a, b):
        return (a.keys() == b.keys() and all(
            np.asarray(a[key]).dtype == np.asarray(b[key]).dtype
            and np.array_equal(a[key], b[key]) for key in a))

    groups = []
    for region in sorted(rows, key=lambda row: (
            row.get('zone_id', row['region_id']), row['region_id'])):
        dense_crust.validate(region['column'])
        if len(region['column']['thickness_km']) != 1:
            raise ValueError('Exact source coalescence needs single-column regions.')
        zone = region.get('zone_id', region['region_id'])
        found = next((group for group in groups
                      if group['zone_id'] == zone
                      and identical(group['column'], region['column'])
                      and group['reference'] == inherit_reference(region)), None)
        if found is None:
            groups.append(dict(zone_id=zone, column=region['column'],
                               reference=inherit_reference(region),
                               fraction=0., member_ids=[]))
            found = groups[-1]
        found['fraction'] += float(region['fraction'])
        found['member_ids'].append(region['region_id'])
    result = [dict(
        region_id=(group['member_ids'][0] if len(group['member_ids']) == 1
                   else json.dumps(['exact-union', group['member_ids']],
                                   separators=(',', ':'))),
        zone_id=group['zone_id'], fraction=group['fraction'],
        column={key: value.copy() for key, value in group['column'].items()},
        **group['reference'])
        for group in groups]
    return dict(history=dict(
        **_face_identity(history), regions=result,
        scope='exact same-zone source-state coalescence; no approximate phase mixing'),
        original_region_count=len(rows), result_region_count=len(result),
        scope='read-only exact region count reduction; no native checkpoint update')


def _face_identity(history):
    identity = dict(face_index=history['face_index'])
    if 'face_id' in history:
        identity['face_id'] = history['face_id']
    return identity
