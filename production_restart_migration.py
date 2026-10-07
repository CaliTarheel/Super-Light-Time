"""Explicit production-policy restart of the repaired 146 Myr world.

This boundary adds a declared thermal initial condition, not recovered history.
The reference bath uses the existing geotherm at each current stack region;
its area-weighted face mean is the initial finite-volume column temperature.
An unburied-reference sensitivity uses only each column's own thickness. Both
start with zero retained dense phase and leave previous foundering untouched.

Moving-hinge version 1 is the reduced work closure in OVERRIDING-RESPONSE.md,
not mantle suction or a prescribed velocity. It accords with Scotese Rules I
and VII through explicit forces and relative motion. The reference conductive
bath and finite-volume averaging are computational assumptions, not Scotese
claims or a reconstruction of 146 Myr of heating. No contact, geometry, floor,
or conservation guard is relaxed to activate these policies.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import numpy as np

import dense_crust
import evolution_policy
import mesh_coverage
import phase_evolution
import plate_balance
import retained_phase_profile
import restart_boundary

VERSION = 1
EPOCH_MYR = 146.
TEMPERATURE_POLICIES = ('reference_bath', 'unburied_reference')


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _arrays(value, path='state'):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('Migration does not support object arrays.')
        return {path: dict(dtype=value.dtype.str, shape=list(value.shape),
            sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(_arrays(item, path+'/'+str(key)))
        return result
    if isinstance(value, (tuple, list)):
        result = {}
        for index, item in enumerate(value):
            result.update(_arrays(item, path+'/'+str(index)))
        return result
    if type(value).__name__ == 'PersistentDomainTracker':
        return _arrays(vars(value), path+'/domain_tracker')
    return {}


def _requirements(s):
    if float(s.t) != EPOCH_MYR:
        raise ValueError('Production restart migration is scoped to 146 Myr.')
    if getattr(s, 'production_restart_migration', None) is not None:
        raise ValueError('Production restart migration is already recorded.')
    expected = dict(plate_balance_version=1, plate_resistance_version=1,
        material_mechanics_version=1, collision_surface_version=1,
        foundering_version=2, foundering_depth_version=1,
        same_sheet_nonpenetration_version=1)
    for name, version in expected.items():
        value = getattr(s, name, None)
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value != version):
            raise ValueError('Repaired basic migration requires '+name+'='+str(version)+'.')
    if phase_evolution.enabled(s):
        raise ValueError('Cannot overwrite an existing retained thermal history.')
    if plate_balance.subduction_response_version(s) != 0:
        raise ValueError('Expected the basic migration with fixed-trench response.')
    retreat = np.asarray(getattr(s, 'trench_retreat_speed', ()), float)
    if not np.isfinite(retreat).all() or np.any(retreat != 0.):
        raise ValueError('Moving-hinge migration requires zero heuristic trench retreat.')
    if bool(getattr(s, 'foreland_loading_enabled', True)):
        raise ValueError('Basic migration must disable the regional foreland correction.')
    for name in ('continental_entry_regions', 'entry_phase_depth', 'channel_region_store',
                 'slab_tether_evolution'):
        if getattr(s, name, None) is not None:
            raise ValueError('Production restart cannot inherit experimental '+name+'.')
    surface = s.material_surface
    overlap = mesh_coverage.same_sheet_overlaps(surface['vertices'], surface['faces'],
        s.parcel_collision_sheet, radius_km=surface.get('radius_km', 6371.))
    if len(overlap['area_km2']):
        raise ValueError('Production restart requires repaired geometry with zero same-sheet overlap.')


def _thermal_initial_condition(s, controls, policy):
    """Validate real stack regions and declare a face-mean initial temperature."""
    thickness = np.asarray(s.structure['thickness_km'], float)
    if (thickness.shape != np.asarray(s.mass).shape or not len(thickness)
            or not np.isfinite(thickness).all() or np.any(thickness <= 0.)):
        raise ValueError('Thermal initialization requires aligned positive physical columns.')
    own = np.minimum(controls['mantle_temperature_c'], controls['surface_temperature_c']
        +controls['geotherm_c_per_km']*controls['thermal_depth_fraction']*thickness)
    # The temporary phase state exists only to use the production region guard.
    # It is discarded; neither initialization advances a source step or RNG.
    probe = deepcopy(s)
    phase_evolution.upgrade(probe, own, constitutive_parameters=controls)
    prepared = phase_evolution.prepare(probe)
    rows = prepared['regions']
    index = np.asarray([row['face'] for row in rows], int)
    weight = np.asarray([row['weight'] for row in rows], float)
    upper = np.asarray([thickness[row['upper']].sum() for row in rows], float)
    regional = np.minimum(controls['mantle_temperature_c'], controls['surface_temperature_c']
        +controls['geotherm_c_per_km']*(upper+controls['thermal_depth_fraction']*thickness[index]))
    closure = np.bincount(index, weights=weight, minlength=len(thickness))
    if not np.allclose(closure, 1., rtol=2e-9, atol=2e-11):
        raise ValueError('Thermal initialization requires a complete disjoint stack partition.')
    reference = np.bincount(index, weights=weight*regional, minlength=len(thickness))/closure
    initial = reference if policy == 'reference_bath' else own
    reference_formula = 'area-weighted mean of min(T_mantle, T_surface + geotherm * (local_overburden_km + depth_fraction * own_thickness_km))'
    unburied_formula = 'min(T_mantle, T_surface + geotherm * depth_fraction * own_thickness_km)'
    diagnostic = dict(policy=policy, initialized_myr=float(s.t),
        interpretation='Declared new thermal initial condition; no inherited heating history recovered.',
        formula=reference_formula if policy == 'reference_bath' else unburied_formula,
        reference_bath_formula=reference_formula,
        sensitivity_formula=unburied_formula,
        sensitivity_interpretation='Unburied-reference assumption before equilibration to present burial; not an age estimate.',
        face_temperature_averaging=(
            'Current stack regions are averaged by their physical area within each face before initializing its uniform column temperature.'
            if policy == 'reference_bath' else
            'Only own-column thickness sets the uniform face temperature; current overburden is excluded.'),
        trace_temperature_policy='Markers inherit their owning material face temperature through the production phase migration.',
        inherited_thermal_history_reconstructed=False, initial_retained_dense_volume_km3=0.,
        constitutive_parameters=deepcopy(controls), stack_regions=len(rows),
        ordered_overlap_pairs=len(prepared['upper']),
        maximum_local_overburden_km=float(upper.max(initial=0.)),
        buried_material_faces=int(len(np.unique(index[upper > 0.]))),
        minimum_temperature_c=float(initial.min()), maximum_temperature_c=float(initial.max()),
        maximum_burial_temperature_increment_c=float((reference-own).max(initial=0.)),
        maximum_applied_burial_temperature_increment_c=float((initial-own).max(initial=0.)),
        temperature_sha256=_arrays(initial)['state']['sha256'],
        stack_geometry_validated=True, same_sheet_guard_preserved=True)
    return initial, diagnostic


def migrate_production(simulation, *, temperature_policy='reference_bath',
                       constitutive_parameters=None):
    """Return a detached state and JSON receipt; never step or mutate the input.

    Input must already have the basic balance/weld/sink migration and an actual
    conservative geometry repair. A version marker alone cannot certify that
    repair: exact same-sheet intersections and regional stack closure are checked.
    """
    if temperature_policy not in TEMPERATURE_POLICIES:
        raise ValueError('Temperature policy must be reference_bath or unburied_reference.')
    _requirements(simulation)
    controls = phase_evolution.parameters(constitutive_parameters)
    before = _arrays(vars(simulation))
    config_hash = _hash(simulation.config)
    rng_hash = _hash(simulation.rng.bit_generator.state)
    events_hash = _hash(simulation.events)
    epoch = (float(simulation.t), int(simulation.steps))
    volume = float(np.asarray(simulation.material_surface['area_km2'])
                   @np.asarray(simulation.structure['thickness_km']))
    s = deepcopy(simulation)
    initial, thermal = _thermal_initial_condition(s, controls, temperature_policy)
    phase_report = phase_evolution.upgrade(s, initial, constitutive_parameters=controls)
    s.subduction_response_version = 1
    s.erosion_relief_version = 2
    policy_report = evolution_policy.upgrade(s)
    phase_evolution.require_current_floor(s)
    for state in (s.structure, s.trace_structure):
        dense_crust.validate(state)
        if np.any(state[dense_crust.DENSE] != 0.):
            raise ValueError('Migration must start with zero retained dense phase.')
    after = _arrays(vars(s))
    changed = [name for name, record in before.items() if after.get(name) != record]
    if changed:
        raise ValueError('Migration changed pre-existing arrays: '+', '.join(changed[:12]))
    if (_hash(s.config) != config_hash or _hash(s.rng.bit_generator.state) != rng_hash
            or _hash(s.events) != events_hash or (float(s.t), int(s.steps)) != epoch):
        raise ValueError('Migration changed configuration, RNG, clocks, or event history.')
    if float(np.asarray(s.material_surface['area_km2'])@s.structure['thickness_km']) != volume:
        raise ValueError('Migration changed physical material volume.')
    receipt = dict(version=VERSION, epoch_myr=float(s.t), epoch_steps=int(s.steps),
        initialization_boundary=restart_boundary.record(s),
        kind='explicit_repaired_production_restart', thermal_initial_condition=thermal,
        phase_migration=phase_report, evolution_policy=policy_report,
        moving_hinge=dict(from_version=0, to_version=1, passive_resistance_version=1,
            heuristic_retreat_zero=True, imposed_velocity=False,
            closure='Constrained upper-plate hinge and one shared slab gravitational-work source; not a solved mantle-suction field.'),
        erosion_relief_version=2, retained_phase_floor_version=phase_evolution.FLOOR_VERSION,
        prior_physics_profile_version=int(getattr(simulation, 'physics_profile_version', 0)),
        experimental_full_stack_enabled=False,
        preservation=dict(all_preexisting_arrays_exact=True, preexisting_arrays=len(before),
            preexisting_array_inventory_sha256=_hash(before), added_array_paths=sorted(set(after)-set(before)),
            rng_sha256=rng_hash, config_sha256=config_hash, events_sha256=events_hash,
            clocks_unchanged=True, physical_column_volume_km3=volume),
        limitations=['Thermal initial conditions are declared at the restart boundary, not reconstructed history.',
            'A face retains one mean temperature; subface thermal heterogeneity is not stored initially.',
            'Moving-hinge response is a reduced instantaneous work closure; it does not add rupture, entry/contact channels, or an evolving slab-potential-energy ledger.',
            'Past frames retain their original physics; only subsequent evolution uses these selected policies.'])
    s.production_restart_migration = deepcopy(receipt)
    s.retained_phase_profile_diagnostics = dict(enabled=True,
        profile='production_restart_'+temperature_policy,
        **deepcopy(thermal))
    return s, _jsonable(receipt)


def snapshot(s):
    """Report actual saved activation without claiming fresh-profile history."""
    result = retained_phase_profile.snapshot(s)
    result['subduction_response_version'] = plate_balance.subduction_response_version(s)
    if hasattr(s, 'production_restart_migration'):
        result['production_restart_migration'] = deepcopy(s.production_restart_migration)
    return result
