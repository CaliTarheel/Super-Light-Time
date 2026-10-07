"""Isolated retained-phase update with a layered channel top load.

These are column samples from already supplied contact geometry. They reuse
the native pure phase/thermal kernel but do not partition a moving face,
change ownership, or commit a model timestep.
"""

import numpy as np

import dense_crust
import phase_evolution
import phase_region_projection
from entry_channel_pressure import (
    layered_top_pressure, ordered_top_pressure)


def advance_column(state, face_index, lower_top_depth_m,
                   current_upper_base_depth_m, upper_column_mass_kg_m2,
                   effective_thermal_depth_km, elapsed_myr, *,
                   water_depth_m=0., constitutive_parameters=None):
    """Advance a copied material column under separate pressure/thermal paths.

    The thermal depth is deliberately supplied independently of geometric
    burial; a contact reaction or ambient mantle depth does not determine the
    slab's temperature history. Top overburden is ordered upper rock, water,
    and only the mantle gap below the current upper base. The native phase
    kernel adds the incoming column's own internal overburden.
    """
    thermal_depth, duration = _validated_source_inputs(
        state, face_index, effective_thermal_depth_km, elapsed_myr)
    pressure = layered_top_pressure(
        lower_top_depth_m, current_upper_base_depth_m,
        upper_column_mass_kg_m2, water_depth_m=water_depth_m)
    return _advance_at_top_pressure(
        state, face_index, pressure, thermal_depth, duration,
        constitutive_parameters)


def advance_ordered_column(state, face_index, lower_top_depth_m, segments,
                           covering_sheets_top_to_bottom,
                           effective_thermal_depth_km, elapsed_myr, *,
                           constitutive_parameters=None):
    """Advance one copied column under an explicit ordered vertical load."""
    thermal_depth, duration = _validated_source_inputs(
        state, face_index, effective_thermal_depth_km, elapsed_myr)
    pressure = ordered_top_pressure(
        lower_top_depth_m, segments, covering_sheets_top_to_bottom)
    result = _advance_at_top_pressure(
        state, face_index, pressure, thermal_depth, duration,
        constitutive_parameters)
    result['rock_sheet_ids'] = pressure['rock_sheet_ids']
    return result


def _validated_source_inputs(state, face_index, effective_thermal_depth_km,
                             elapsed_myr):
    dense_crust.validate(state)
    count = len(state['thickness_km'])
    thermal_depth = float(effective_thermal_depth_km)
    duration = float(elapsed_myr)
    if (not isinstance(face_index, (int, np.integer))
            or not 0 <= face_index < count
            or not np.isfinite(thermal_depth) or thermal_depth < 0.
            or not np.isfinite(duration) or duration <= 0.):
        raise ValueError('Channel source needs an existing column, thermal path and positive elapsed time.')
    return thermal_depth, duration


def _advance_at_top_pressure(state, face_index, pressure, thermal_depth,
                             duration, constitutive_parameters):
    top_mass = np.asarray(pressure['overburden_mass_kg_m2'])
    if top_mass.ndim != 0:
        raise ValueError('A channel source sample needs scalar top overburden.')
    controls = phase_evolution.parameters(constitutive_parameters)
    (updated, returned, contraction, steps, maximum_rate, maximum_pressure,
     withheld, reserve_limited) = phase_evolution._advance_regions(
        state, np.array([face_index]), np.array([thermal_depth]),
        np.array([float(top_mass)]), controls, duration)
    dense_crust.validate(updated)
    return dict(
        column=updated, returned_km=float(returned[0]),
        contraction_km=float(contraction[0]),
        withheld_equivalent_km=float(withheld[0]),
        reserve_limited=bool(reserve_limited[0]),
        substeps=steps, maximum_drainage_rate_myr=maximum_rate,
        maximum_ordinary_pressure_pa=maximum_pressure,
        top_pressure_pa=float(pressure['pressure_pa']),
        top_overburden_mass_kg_m2=float(top_mass),
        effective_thermal_depth_km=thermal_depth,
        scope='isolated copied column; no native face projection, owner or checkpoint update',
    )


def project_face_regions(state, face_index, regions, elapsed_myr, *,
                         constitutive_parameters=None, face_area_km2=None):
    """Evolve separate physical-area regions and coalesce their inventories.

    Every region begins from the same input face column and carries its own
    contact geometry, top overburden and thermal path. Nonlinear phase/heat
    sources are evaluated *before* physical-area weighting; averaging depths
    or pressures first would erase locally eligible material. The weights
    are supplied by an external disjoint geometry partition and must total
    one. No native face/marker state is mutated or committed here.
    """
    if not isinstance(regions, (list, tuple)) or not regions:
        raise ValueError('Channel source projection needs disjoint face regions.')
    required = {'fraction', 'lower_top_depth_m',
                'current_upper_base_depth_m', 'upper_column_mass_kg_m2',
                'effective_thermal_depth_km'}
    allowed = required | {'water_depth_m'}
    if any(not isinstance(region, dict)
           or not required <= region.keys()
           or not region.keys() <= allowed for region in regions):
        raise ValueError('Channel source region lacks an explicit pressure or thermal path.')
    weights = np.asarray([region['fraction'] for region in regions], float)
    if (not np.isfinite(weights).all() or np.any(weights <= 0.)
            or abs(float(weights.sum()) - 1.) > 2e-10):
        raise ValueError('Channel source region physical-area fractions must partition one face.')
    if face_area_km2 is not None and (
            not np.isscalar(face_area_km2)
            or not np.isfinite(face_area_km2) or face_area_km2 <= 0.):
        raise ValueError('Channel source face area must be positive and finite.')
    samples = [advance_column(
        state, face_index, region['lower_top_depth_m'],
        region['current_upper_base_depth_m'],
        region['upper_column_mass_kg_m2'],
        region['effective_thermal_depth_km'], elapsed_myr,
        water_depth_m=region.get('water_depth_m', 0.),
        constitutive_parameters=constitutive_parameters)
        for region in regions]
    merged = _coalesce_samples(samples, weights)
    returned = float(weights @ [sample['returned_km'] for sample in samples])
    contraction = float(weights @ [sample['contraction_km'] for sample in samples])
    result = dict(
        column=merged, returned_km=returned, contraction_km=contraction,
        withheld_equivalent_km=float(weights @ [
            sample['withheld_equivalent_km'] for sample in samples]),
        region_count=len(samples),
        maximum_ordinary_pressure_pa=max(
            sample['maximum_ordinary_pressure_pa'] for sample in samples),
        top_pressure_pa=np.array([sample['top_pressure_pa']
                                  for sample in samples]),
        scope='read-only disjoint region projection; no native face, marker or checkpoint update',
    )
    if face_area_km2 is not None:
        result['returned_volume_km3'] = returned * float(face_area_km2)
    return result


def _coalesce_samples(samples, weights):
    expanded = {
        key: np.concatenate([sample['column'][key] for sample in samples])
        for key in samples[0]['column']}
    groups = np.zeros(len(samples), dtype=int)
    merged = phase_region_projection.coalesce_regions(
        expanded, groups, weights, 1, measure='physical')
    dense_crust.validate(merged)
    return merged
