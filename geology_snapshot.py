"""Scalar geological state on the same finite footprints as buoyant crust."""
import numpy as np
import crustal_structure as columns
import progressive_rifting


GRID_FIELDS = ('crustal_thickness_km', 'crustal_root_km', 'rift_thermal_support_m',
               'rift_cooling_age_myr', 'foreland_deflection_m', 'erosion_rate_m_myr',
               'rift_damage', 'rift_strength_relative')
# foundered_m is the only record of eclogitic mass loss a marker carries:
# crustal_root_km cannot show it, because the neutral reference thins with the
# column so that the relaxation solver does not read a foundered root as
# compressed. Appended last, and skipped on a state that predates the field.
TRACE_DIAGNOSTICS = ('denudation_m', 'rebound_m', 'thermal_subsidence_m', 'foreland_subsidence_m',
                     'foundered_m')


def material_fields(state):
    cold = dict(state, rift_heat_m=np.zeros_like(state['rift_heat_m']))
    return dict(crustal_thickness_km=state['thickness_km'],
                crustal_root_km=np.maximum(state['thickness_km']-state['reference_thickness_km'], 0.),
                rift_thermal_support_m=np.maximum(columns.elevation(state)-columns.elevation(cold), 0.),
                rift_cooling_age_myr=state['rift_age_myr'],
                foreland_deflection_m=state['foreland_m'],
                erosion_rate_m_myr=np.maximum(state['erosion_rate_m_myr'],0.))


def deposit_fields(s):
    fields = material_fields(s.structure)
    fields.update(progressive_rifting.material_fields(s))
    age = fields['rift_cooling_age_myr']
    fields['rift_cooling_valid'] = (age >= 0).astype(float)
    fields['rift_cooling_age_myr'] = np.maximum(age, 0.)
    return fields


def finish_grid(s, sums):
    denominator = np.maximum(s.land_mass, 1e-20)
    s.structure_grid = {name: sums[name]/denominator for name in GRID_FIELDS}
    valid = sums['rift_cooling_valid'] >= .5*denominator
    s.structure_grid['rift_cooling_age_myr'] = np.divide(
        sums['rift_cooling_age_myr'], sums['rift_cooling_valid'],
        out=np.full(s.n, -1.), where=valid & (sums['rift_cooling_valid'] > 0))


def snapshot(s):
    ocean = s.crust == 0
    fields = {}
    for name, field in s.structure_grid.items():
        result = field.astype(np.float32).copy()
        result[ocean] = 7. if name == 'crustal_thickness_km' else -1. if name == 'rift_cooling_age_myr' else 1. if name == 'rift_strength_relative' else 0.
        if name == 'rift_strength_relative':
            result = np.maximum(result, .01)
        fields[name] = result
    for name, field in material_fields(s.trace_structure).items():
        if name != 'erosion_rate_m_myr':
            fields['trace_'+name] = field.copy()
    for name in TRACE_DIAGNOSTICS:
        if name in s.trace_structure:
            fields['trace_'+name] = s.trace_structure[name].copy()
    # A map cell can touch multiple trench faces. The largest mature weight
    # wins, with ID as deterministic tie-breaker; full local geometry remains
    # in trench_systems rather than being collapsed into this display index.
    grid = np.zeros(s.n, np.int32)
    order = np.lexsort((s.trench_id, s.trench_maturity))
    for edge in order:
        if s.trench_id[edge] > 0:
            grid[s.ba[edge]] = grid[s.bb[edge]] = s.trench_id[edge]
    fields['trench'] = grid
    land = ~ocean
    stats = dict(
        mean_crustal_thickness_km=float(np.average(fields['crustal_thickness_km'][land], weights=s.cell_area[land])) if np.any(land) else 0.,
        max_crustal_thickness_km=float(fields['crustal_thickness_km'][land].max()) if np.any(land) else 0.,
        cooling_margin_area_km2=float(s.cell_area[land & (fields['rift_cooling_age_myr'] >= 0)].sum()),
        foreland_basin_area_km2=float(s.cell_area[land & (fields['foreland_deflection_m'] >= 100)].sum()))
    return dict(structure_version=columns.STRUCTURE_VERSION, **fields), stats
