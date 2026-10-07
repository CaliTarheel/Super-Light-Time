"""Explicit fresh-world activation of the retained thermal/phase closure.

The thermal_v1 initial condition places each initially unburied continental
column at its own reference conductive-bath temperature. The reference bath is
surface_temperature + geotherm * thermal_depth_fraction * column_thickness,
capped at the mantle temperature. This is a declared initial thermal state,
not a reconstruction of inherited heating or a resolved geotherm. Existing
checkpoints retain their saved phase state and never run this initializer.
"""
from copy import deepcopy
import numpy as np

VERSION = 1
NAME = 'thermal_v1'


def normalize(value='disabled'):
    if not isinstance(value, str) or value not in ('disabled', NAME):
        raise ValueError('Retained phases must be disabled or thermal_v1.')
    return value


def initialize(s):
    """Select reference PR12 physics once, before the fresh force solve."""
    setting = normalize(s.config.get('retained_phases', 'disabled'))
    if setting == 'disabled':
        return dict(enabled=False, profile='disabled')
    if s.config.get('physics_profile') != 'reviewed_v1':
        raise ValueError('Retained thermal phases require reviewed_v1.')
    if float(s.t) != 0. or int(s.steps) != 0:
        raise ValueError('Retained thermal phase selection is fresh-world initialization, not a resume migration.')
    if getattr(s, 'retained_phase_profile_version', 0) == VERSION:
        return deepcopy(s.retained_phase_profile_diagnostics)
    if getattr(s, 'retained_dense_crust_version', 0):
        raise ValueError('Fresh retained-phase initialization cannot reset an existing phase history.')
    import phase_evolution
    controls = phase_evolution.parameters()
    thickness = np.asarray(s.structure['thickness_km'], float)
    if (thickness.shape != np.asarray(s.mass).shape or not np.isfinite(thickness).all()
            or np.any(thickness <= 0.)):
        raise ValueError('Initial reference temperatures require aligned positive physical column thickness.')
    # Fresh source material tiles the sphere without overlying material sheets.
    # Do not silently use an unburied bath for an already stacked initial state.
    import eclogite_sink
    _, _, overlap_area, _ = eclogite_sink._depth_pairs(s)
    if np.any(np.asarray(overlap_area) > 0.):
        raise ValueError('The thermal_v1 fresh bath requires initially unburied columns; supply explicit stack temperatures separately.')
    initial = np.minimum(controls['mantle_temperature_c'],
                         controls['surface_temperature_c']+controls['geotherm_c_per_km']
                         *controls['thermal_depth_fraction']*thickness)
    report = phase_evolution.upgrade(s, initial, constitutive_parameters=controls)
    # Fresh opt-in only: existing typed checkpoints keep their saved erosion
    # version, including historical version 1's explicit dense-state rejection.
    s.erosion_relief_version = 2
    diagnostic = dict(enabled=True, version=VERSION, profile=NAME, initialized_myr=0.,
        initial_temperature_policy='Initially unburied columns equilibrated with the declared reference conductive bath.',
        temperature_formula='min(mantle_temperature_c, surface_temperature_c + geotherm_c_per_km * thermal_depth_fraction * thickness_km)',
        inherited_thermal_history_reconstructed=False,
        initial_temperature_min_c=float(initial.min()) if len(initial) else None,
        initial_temperature_max_c=float(initial.max()) if len(initial) else None,
        constitutive_parameters=deepcopy(controls), phase_migration=deepcopy(report),
        mechanics='Existing ordinary/dense mass supplies one shared plate/sheet gravitational functional.',
        column_floor=dict(version=phase_evolution.FLOOR_VERSION,
            invariant='Current mass-equivalent thickness retains an 8 km mechanics reserve; mantle return and erosion spend it.'),
        drainage='Retained dense-phase evolution replaces the historical immediate foundering sink.',
        erosion_relief_version=2,
        surface_erosion='Net relief decay removes ordinary upper crust only, capped before dense basement and by the phase-free-equivalent 8 km mechanics reserve; unmet relief is recorded. Dense basement remains subject to phase evolution/drainage.',
        limitation='Reference thermal bath and strength-limited drainage; not resolved mantle flow or calibrated instability.')
    s.retained_phase_profile_version = VERSION
    s.retained_phase_profile_diagnostics = diagnostic
    return deepcopy(diagnostic)


def snapshot(s):
    """Report actual saved activation; a config edit cannot create phase state."""
    import phase_evolution
    enabled = phase_evolution.enabled(s)
    if not enabled:
        return dict(retained_dense_crust_version=0, retained_phase_profile=dict(enabled=False, profile='disabled'))
    profile = deepcopy(getattr(s, 'retained_phase_profile_diagnostics',
                               dict(enabled=True, profile='explicit_external_initialization')))
    return dict(retained_dense_crust_version=phase_evolution.VERSION,
                retained_phase_floor_version=phase_evolution.floor_version(s),
                retained_phase_profile=profile)
