"""Explicit fresh-world selection of reviewed reduced physics.

Selecting this profile is not a checkpoint migration. It starts from the same
imported artwork and random seed, with a different, recorded constitutive law.
Inherited slabs require a separate explicit geological initial condition;
no artificial startup traction is supplied.
"""
from copy import deepcopy
import numpy as np

VERSION = 1
NAME = 'reviewed_v1'


def normalize(value='legacy'):
    if not isinstance(value, str) or value not in ('legacy', NAME):
        raise ValueError('Physics profile must be legacy or reviewed_v1.')
    return value


def normalize_subduction_response(value='fixed_trench'):
    if not isinstance(value, str) or value not in ('fixed_trench', 'moving_hinge_v1'):
        raise ValueError('Subduction response must be fixed_trench or moving_hinge_v1.')
    return value


def polarity_enabled(s):
    version = getattr(s, 'subduction_polarity_version', 0)
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer)) or version not in (0, 1):
        raise ValueError('Unsupported subduction polarity policy.')
    return version == 1


def stall_buoyant_incoming(s):
    """A blocked incoming plate stalls; its opposite does not inherit a trench.

    Finite native water geometry admits any surviving oceanic part of a front;
    the consumption operator still clips out actual continental material.
    This is a reduced stall closure, not a breakoff or reverse-initiation solve.
    """
    if not polarity_enabled(s):
        return
    import native_subduction
    import normal_partition
    candidates = normal_partition.convergence(s) & ((s.down == s.bp) | (s.down == s.bq))
    below = np.where(s.down == s.bp, s.ba, s.bb)
    water = (native_subduction.edge_ocean_fraction(s, s.down) > 1e-10
             if native_subduction.enabled(s) else s.crust[below] == 0)
    blocked = candidates & ~water
    s.bcode[blocked] = 4
    s.down[blocked] = -1
    if hasattr(s, 'trench_maturity'):
        s.trench_maturity[blocked] = 0.
    previous = getattr(s, '_subduction_stalled_mask', None)
    if previous is None or np.shape(previous) != np.shape(blocked):
        previous = np.zeros_like(blocked)
    s._subduction_stalled_mask = previous | blocked
    import effective_subduction
    effective = effective_subduction.enabled(s)
    s.subduction_polarity_diagnostics = dict(version=1,
        stalled_edges=int(np.count_nonzero(s._subduction_stalled_mask)),
        stalled_length_km=float(np.asarray(s.bl)[s._subduction_stalled_mask].sum()),
        policy='Historical incoming polarity stalls on continental arrival; no automatic reversal.',
        shutdown=('Declared effective trace persists; drive and intake stop locally at fully buoyant incoming material.'
                  if effective else 'Existing local quiet/shutdown law; retained slab mass retires through its own ledger.'),
        limitation=('Replacement-trench initiation and independent rollback are not resolved.'
                    if effective else 'Slab breakoff and initiation of a replacement trench are not resolved.'))


def initialize(s):
    """Activate only after all native geometry, columns and contacts exist."""
    if normalize(s.config.get('physics_profile', 'legacy')) != NAME:
        return
    if getattr(s, 'physics_profile_version', 0) == VERSION:
        return
    if float(s.t) != 0. or int(s.steps) != 0:
        raise ValueError('The reviewed physics profile is a fresh-world initialization, not a resume migration.')
    import migration_balance_weld_sink
    import trench_history
    import primordial_subduction
    import retained_phase_profile
    import effective_subduction
    s.physics_profile_version = VERSION
    # An explicit fresh-world policy; old typed checkpoints retain their saved
    # resistance version (or the missing/zero legacy law).
    s.plate_resistance_version = 1
    s.subduction_response_version = int(normalize_subduction_response(
        s.config.get('subduction_response', 'fixed_trench')) == 'moving_hinge_v1')
    import backarc
    # The moving hinge has no forearc sweep, so its back-arcs load from the
    # overriding plate's motion away from an anchored slab. Fresh Lite worlds
    # use the same motion relative to their declared effective reservoir;
    # saved worlds keep their recorded driver (missing field reads 0).
    effective_selected = effective_subduction.normalize(s.config.get('effective_subduction'))['enabled']
    s.backarc_driver_version = (backarc.EFFECTIVE_DRIVER_VERSION if effective_selected else
                               backarc.DRIVER_VERSION if s.subduction_response_version == 1 else 0)
    s.subduction_polarity_version = 1
    # Fresh-world choice; saved worlds without the field keep the kinematic law.
    s.trench_persistence_version = int(s.config.get('trench_persistence', 'kinematic') == 'attached_slab_v1')
    s.slab_allocation_version = int(s.config.get('slab_allocation', 'uniform') == 'fed_v1')
    s.erosion_relief_version = 1
    s.foreland_loading_enabled = False
    import boundary_labels
    boundary_labels.upgrade(s)
    upgrades = migration_balance_weld_sink.upgrade_state(s)
    import localized_accretion
    import local_accretion
    if localized_accretion.enabled(s):
        # Fresh reviewed worlds replace the unexplained 70 Myr plate-age gate
        # with the youngest age at which a plate's own contacts can mature.
        s.accretion_age_policy_version = local_accretion.AGE_POLICY_VERSION
        if isinstance(getattr(s, 'native_mesh', None), dict):
            # Overlapping sheets of one plate (a welded thrust stack) dock or
            # stay as one body, so docked terranes cannot ping-pong back.
            s.accretion_welded_stack_version = local_accretion.WELDED_STACK_VERSION
    # Fresh reviewed worlds gate continental breakup on realized kilometres of
    # extension rather than a link-length-dependent strain. The rift mechanics
    # labels written before this selection are refreshed to match.
    import progressive_rifting
    s.rupture_criterion_version = progressive_rifting.RUPTURE_CRITERION_VERSION
    s.rift_mechanics.pop('rupture_strain_threshold', None)
    s.rift_mechanics.update(progressive_rifting._rupture_calibration(s))
    # A terrane transfer on one plate no longer vetoes every continental cut
    # of its step; only cuts whose own loading unit changed wait a step.
    s.rift_commit_version = progressive_rifting.RIFT_COMMIT_VERSION
    s.rift_mechanics.update(progressive_rifting._commit_labels(s))
    retained_phases = retained_phase_profile.initialize(s)
    # Initial seeded velocities supplied the legacy first-frame classifications.
    # They are not inherited mechanical history for the force-balanced world.
    if s.trench_systems:
        raise ValueError('Fresh profile activation cannot reinterpret existing trench history.')
    # The artwork importer used the legacy engine's seeded velocities. Its
    # time-zero kinematic events are not events of the balanced native world,
    # and stale deduplication keys would suppress the genuine initial trenches.
    s.events = [event for event in s.events if not (
        event.get('time_myr') == 0. and (event['type'].startswith('trench_') or
        (event['type'] == 'breakup' and event['description'].startswith('Initial extensional Euler'))))]
    for ident, event in enumerate(s.events, 1):
        event['id'] = ident
    s.event_keys = {key for key in s.event_keys if not (isinstance(key, tuple) and key and key[0] == 'trench')}
    s.polarity[:] = -1
    effective = effective_subduction.initialize(s)
    inherited_subduction = effective or primordial_subduction.initialize(s)
    import slab_anchors
    if not effective and slab_anchors.enabled(s):
        # Declared inherited slabs start uniform along their own traces.
        slab_anchors.ensure(s)
    s._forces(s.config['dt_myr'])
    s._boundaries()
    s._trench_last_update_myr = None
    if inherited_subduction:
        # These rows already represent the specified pre-existing mechanism.
        # Zero elapsed model time must not classify them quiet or reset maturity.
        s._trench_last_update_myr = 0.
        trench_history.prepare(s)
    else:
        trench_history.update(s, 0.)
    if not effective:
        primordial_subduction.record_initial_motion(s)
    if s.backarc_driver_version:
        backarc.loading_speed(s)  # initial diagnostics from the first solved motion
    speed = np.linalg.norm(np.cross(s.omega[s.plate], s.xyz), axis=1)*6371.*.1
    s.physics_profile_diagnostics = dict(version=VERSION, name=NAME,
        subduction_response_version=s.subduction_response_version,
        subduction_response=s.config.get('subduction_response', 'fixed_trench'),
        trench_persistence_version=s.trench_persistence_version,
        slab_allocation_version=s.slab_allocation_version,
        plate_resistance_version=s.plate_resistance_version,
        backarc_driver_version=s.backarc_driver_version,
        accretion_age_policy_version=getattr(s, 'accretion_age_policy_version', 0),
        accretion_welded_stack_version=getattr(s, 'accretion_welded_stack_version', 0),
        rupture_criterion_version=getattr(s, 'rupture_criterion_version', 0),
        rift_commit_version=getattr(s, 'rift_commit_version', 0),
        initialized_myr=0., upgrades=deepcopy(upgrades),
        initial_mean_speed_cm_yr=float(np.average(speed, weights=s.cell_area)),
        initial_max_speed_cm_yr=float(speed.max(initial=0.)),
        initial_trenches=len(s.trench_systems),
        initial_slab_mass_kg=float(sum(r.get('slab_retained_excess_mass_kg', 0.) for r in s.trench_systems)),
        motion='Instantaneous force balance; legacy imposed basal drive, velocity lag and speed cap removed.',
        startup=('Declared effective incoming-plate trench traction and represented ridge gradients drive the ordinary balance; speeds remain solved.'
                 if effective else
                 'Declared inherited oceanic slabs and represented ridge gradients drive the ordinary balance; speeds remain solved.'
                 if inherited_subduction else
                 'Only represented ridge gradients and physical inventories drive motion; startup can be nearly stationary.'),
        primordial_subduction=deepcopy(getattr(s, 'primordial_subduction_diagnostics', {'enabled': False})),
        effective_subduction=deepcopy(getattr(s, 'effective_subduction_diagnostics', {'enabled': False})),
        erosion=('Version 2 net surface-relief decay removes ordinary upper crust only, capped before dense basement and by the phase-free-equivalent 8 km mechanics reserve; unmet relief is recorded.'
                 if s.erosion_relief_version == 2 else
                 'Version 1 net surface-relief decay with explicit gross removal and rebound.'),
        erosion_relief_version=s.erosion_relief_version,
        foreland_loading='Regional trough disabled until a balanced flexural operator is available.',
        retained_dense_crust=('Retained dense phases with the reference thermal bath and strength-limited drainage.'
                              if retained_phases['enabled'] else
                              'Conservative inventory is active; retained thermal/phase coupling is disabled.'),
        retained_phases=deepcopy(retained_phases),
        limitations=['No resolved mantle convection or spontaneous subduction-initiation model.',
                     'No resolved slab breakoff or automatic replacement trench.',
                     'Interface, foundering and erosion coefficients remain reduced-model assumptions.'])
    s._record('physics_profile', 'Fresh world uses the reviewed force-balanced physics profile.',
              details=deepcopy(s.physics_profile_diagnostics))


def snapshot(s):
    if not getattr(s, 'physics_profile_version', 0):
        return {}
    import primordial_subduction
    import retained_phase_profile
    import effective_subduction
    retained = retained_phase_profile.snapshot(s)
    # Only fresh worlds record a rupture criterion; frames of saved reviewed
    # worlds (criterion 0) keep exactly their earlier profile block.
    rupture = getattr(s, 'rupture_criterion_version', 0)
    rupture = dict(rupture_criterion_version=rupture) if rupture else {}
    if getattr(s, 'rift_commit_version', 0):
        rupture['rift_commit_version'] = s.rift_commit_version
    return dict(primordial_subduction.snapshot(s), **effective_subduction.snapshot(s), **retained, physics_profile_version=VERSION,
                subduction_response_version=getattr(s, 'subduction_response_version', 0),
                plate_resistance_version=getattr(s, 'plate_resistance_version', 0),
                physics_profile=dict(version=VERSION, name=NAME,
                    subduction_response_version=getattr(s, 'subduction_response_version', 0),
                    plate_resistance_version=getattr(s, 'plate_resistance_version', 0),
                    backarc_driver_version=getattr(s, 'backarc_driver_version', 0),
                    plate_balance=True, normal_partition=True, weld=True,
                    interface_shear=True, foundering_version=getattr(s, 'foundering_version', 0),
                    retained_dense_crust_version=getattr(s, 'retained_dense_crust_version', 0),
                    retained_phase_floor_version=retained.get('retained_phase_floor_version', 0),
                    erosion_relief_version=getattr(s, 'erosion_relief_version', 0),
                    **rupture,
                    foreland_loading_enabled=bool(getattr(s, 'foreland_loading_enabled', True))),
                physics_profile_diagnostics=deepcopy(s.physics_profile_diagnostics),
                subduction_polarity_version=getattr(s, 'subduction_polarity_version', 0),
                subduction_polarity_diagnostics=deepcopy(getattr(s, 'subduction_polarity_diagnostics', {})),
                erosion_relief_version=getattr(s, 'erosion_relief_version', 0),
                erosion_relief_diagnostics=deepcopy(getattr(s, 'erosion_relief_diagnostics', {})),
                foreland_loading_enabled=bool(getattr(s, 'foreland_loading_enabled', True)),
                timestep_diagnostics=deepcopy(getattr(s, 'timestep_diagnostics', {})))
