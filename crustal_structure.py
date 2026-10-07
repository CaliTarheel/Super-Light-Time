"""Reduced crustal columns, post-rift cooling and foreland accommodation.

These functions carry scalar material state; they do not change plate motion,
parcel positions, footprints or reference-area weights. Thickness and reciprocal
effective column area represent unresolved horizontal strain. Airy compensation
is local and instantaneous. The thermal and foreland closures are deliberately
bounded worldbuilding approximations, not a lithosphere or sediment solver.

Basis: McKenzie (1978), doi:10.1016/0012-821X(78)90071-7, stretching plus
post-rift cooling; https://www.fatiando.org/harmonica/latest/api/generated/
harmonica.isostatic_moho_airy.html, Airy column and water-load accounting.
The legacy foreland trough is flexure-inspired. Native loading version1 supplies
a distributed regional correction through foreland_loading; scalar column state,
volume and loading/unloading history here retain their existing meaning.
"""
from __future__ import annotations

import numpy as np
import crust_inventory
import dense_crust

from bounded_gravity import FEASIBILITY_TOLERANCE
from ridge_geometry import candidate_cells

STRUCTURE_VERSION = 1
STATE_FIELDS = ('thickness_km', 'reference_thickness_km', 'reference_elevation_m',
                'area_factor', 'rift_heat_m', 'rift_age_myr', 'foreland_m')
RADIUS_KM = 6371.
RHO_MANTLE = 3300.
RHO_CRUST = 2800.
RHO_WATER = 1030.
AIRY_M_PER_KM = 1000.*(RHO_MANTLE-RHO_CRUST)/RHO_MANTLE
WATER_FACTOR = RHO_MANTLE/(RHO_MANTLE-RHO_WATER)
MIN_THICKNESS_KM, MAX_THICKNESS_KM = dense_crust.MECHANICAL_FLOOR_KM, 75.
STRAIN_PARTITION = .25
MAX_STRAIN_PER_MYR = .08
COOLING_MYR = 63.
HEATING_COUPLING = 1.4
MAX_RIFT_HEAT_M = 3200./WATER_FACTOR  # air-equivalent metres, not temperature
MAX_FORELAND_M = 1800.
FORELAND_LOADING_MYR = 12.
FORELAND_UNLOADING_MYR = 100.


def _finite(value, name):
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite.')
    return result


def restored_thickness(state):
    """Current mass-equivalent thickness before retained-phase contraction.

    The 8 km mechanics reserve applies to this virtual phase-free column. The
    physical column may be thinner only by the mass-conserving contraction of
    dense material that remains in the column.
    """
    return dense_crust.restored_thickness(state)


def conserved_volume_floor(state):
    """Physical thickness at the mechanics floor under reciprocal area strain."""
    thickness = np.asarray(state['thickness_km'], float)
    restored = restored_thickness(state)
    return thickness*MIN_THICKNESS_KM/restored


def fixed_area_floor(state):
    """Physical floor at fixed area after restoring retained-phase contraction."""
    thickness = np.asarray(state['thickness_km'], float)
    floor = thickness+MIN_THICKNESS_KM-restored_thickness(state)
    return np.maximum(floor, np.nextafter(0., 1.))


def fixed_area_removal_capacity(state):
    """Physical rock removable at fixed area before spending the reserve.

    Ordinary upper crust is removed first. Dense basal crust carries more
    ordinary-equivalent mass per physical kilometre, so once erosion reaches it
    each physical kilometre spends ``1/RATIO`` kilometres of restored capacity.
    """
    thickness = np.asarray(state['thickness_km'], float)
    equivalent = np.maximum(restored_thickness(state)-MIN_THICKNESS_KM, 0.)
    if not dense_crust.present(state):
        return np.minimum(equivalent, thickness)
    dense = np.asarray(state[dense_crust.DENSE], float)/np.asarray(state['area_factor'], float)
    ordinary = np.maximum(thickness-dense, 0.)
    ordinary_removed = np.minimum(equivalent, ordinary)
    dense_removed = np.minimum(dense, np.maximum(equivalent-ordinary_removed, 0.)*dense_crust.RATIO)
    return ordinary_removed+dense_removed


def _water_load(air_height):
    return np.where(air_height >= 0, air_height, air_height*WATER_FACTOR)


def _air_height(surface_height):
    return np.where(surface_height >= 0, surface_height, surface_height/WATER_FACTOR)


def _state(state):
    values = np.broadcast_arrays(*[_finite(state[name], name) for name in STATE_FIELDS])
    result = dict(zip(STATE_FIELDS, values))
    import numerical_accuracy
    if numerical_accuracy.present(state):
        result.update({name: _finite(state[name], name) for name in numerical_accuracy.FIELDS})
    if crust_inventory.present(state):
        result.update({name: np.asarray(state[name], float) for name in crust_inventory.FIELDS})
    if dense_crust.present(state):
        result.update({name: np.asarray(state[name], float) for name in dense_crust.FIELDS})
        dense_crust.validate(result)
    crust_inventory.validate(result)
    if 'lip_heat_m' in state:
        result['lip_heat_m'] = np.broadcast_to(_finite(state['lip_heat_m'], 'LIP thermal support'), values[0].shape)
        if np.any(result['lip_heat_m'] < 0):
            raise ValueError('LIP thermal support must be nonnegative.')
    # Report WHICH bound failed and by how much. The bare message this used to
    # raise stopped a 342 Myr run and told nobody anything: every quantity it
    # guards is clamped where it is computed, so a violation means some other
    # writer bypassed a clamp, and finding that writer needs the field name, the
    # offending value and the count. Diagnosing it from the message alone was
    # not possible.
    violations = []
    thickness = np.asarray(result['thickness_km'], float)
    restored = restored_thickness(result)
    dimensional = numerical_accuracy.validate_columns(result, restored, MIN_THICKNESS_KM, MAX_THICKNESS_KM)
    under = ((thickness <= 0) if dimensional else
        (thickness <= 0) | (restored < MIN_THICKNESS_KM-dense_crust.FLOOR_TOLERANCE_KM))
    over = np.zeros_like(under) if dimensional else thickness > MAX_THICKNESS_KM+1e-9
    if under.any() or over.any():
        violations.append('thickness_km: %d of %d outside the positive, phase-adjusted [%g, %g] bounds; '
                          'observed physical [%.6g, %.6g], restored [%.6g, %.6g]'
                          % (int(under.sum()+over.sum()), thickness.size, MIN_THICKNESS_KM,
                             MAX_THICKNESS_KM, float(thickness.min()), float(thickness.max()),
                             float(restored.min()), float(restored.max())))
    for name, low, high in (('reference_thickness_km', np.nextafter(0., 1.), None),
                            ('area_factor', np.nextafter(0., 1.), None),
                            ('rift_heat_m', -1e-9, MAX_RIFT_HEAT_M+1e-9),
                            ('rift_age_myr', -1., None),
                            ('foreland_m', -1e-9, MAX_FORELAND_M+1e-9)):
        value = np.asarray(result[name], float)
        under = value < low
        over = np.zeros_like(under) if high is None else value > high
        if under.any() or over.any():
            violations.append('%s: %d of %d outside [%g, %s], observed [%.6g, %.6g]'
                              % (name, int(under.sum()+over.sum()), value.size, low,
                                 'inf' if high is None else '%g' % high,
                                 float(value.min()), float(value.max())))
    if violations:
        raise ValueError('Crustal column state is outside its documented bounds. '
                         + '; '.join(violations))
    return result


def _elevation(state):
    air = (_air_height(state['reference_elevation_m'])
           + AIRY_M_PER_KM*(state['thickness_km']-state['reference_thickness_km'])
           + state['rift_heat_m'])
    if 'lip_heat_m' in state:
        air = air+state['lip_heat_m']
    air = air+dense_crust.elevation_correction(state)
    return _water_load(air)-state['foreland_m']


def elevation(state):
    """Compensated bed elevation in metres relative to the zero sea datum.

    Water loading applies to the Airy/thermal column. The separate foreland
    deflection is prescribed in actual metres; its water/flexural feedback is
    not solved a second time. Reference elevations anchor the neutral columns.
    """
    return _elevation(_state(state))


def denudation_for_net_loss(state, net_loss_m):
    """Invert local Airy rebound for a prescribed nonnegative surface loss.

    This is a calibration adapter, not an erosion or sediment-transport model.
    It leaves thermal support, foreland displacement and reference offsets
    fixed, and handles the water-loaded branch and crossing of its sea datum.
    The caller still limits removal to available rock and books the inventory.
    A retained dense phase needs a separate erosion closure: removing it can
    raise the surface, so silently applying the ordinary-crust inverse is wrong.
    """
    values = _state(state)
    loss = np.broadcast_to(_finite(net_loss_m, 'Net erosion loss'), values['thickness_km'].shape)
    if np.any(loss < 0):
        raise ValueError('Net erosion loss must be nonnegative.')
    if dense_crust.present(values) and np.any(values[dense_crust.DENSE] > 0):
        raise ValueError('Net-relief erosion is not calibrated for retained dense crust.')
    loaded = _elevation(values)+values['foreland_m']
    return (_air_height(loaded)-_air_height(loaded-loss))*(1000./AIRY_M_PER_KM)


def ordinary_denudation_for_net_loss(state, net_loss_m):
    """Bound a relief request to ordinary upper crust above a retained dense base.

    This explicit closure holds area, phase volume, thermal support and reference
    offsets fixed during surface erosion. The dense isostatic correction is then
    constant, so the ordinary Airy inverse (including water loading) is valid only
    up to the ordinary/dense interface. Stop there or at the phase-free-equivalent
    8 km mechanics reserve,
    whichever comes first; do not erode dense basement to chase a surface target.
    Phase reaction/drainage remains responsible for the retained basal material.

    Return the achievable removal and relief shortfall without changing state.
    This is a reduced surface-erosion assumption, not a general inverse through
    dense crust: removing that basement can raise rather than lower the surface.
    """
    values = _state(state)
    if not dense_crust.present(values):
        raise ValueError('Ordinary-only erosion requires explicit retained phase state.')
    loss = np.broadcast_to(_finite(net_loss_m, 'Net erosion loss'), values['thickness_km'].shape)
    if np.any(loss < 0):
        raise ValueError('Net erosion loss must be nonnegative.')
    loaded = _elevation(values)+values['foreland_m']
    requested = (_air_height(loaded)-_air_height(loaded-loss))*(1000./AIRY_M_PER_KM)
    ordinary_volume = np.maximum(values['thickness_km']*values['area_factor']-values[dense_crust.DENSE], 0.)
    ordinary_m = ordinary_volume/values['area_factor']*1000.
    floor_m = fixed_area_removal_capacity(values)*1000.
    removed = np.minimum(requested, np.minimum(ordinary_m, floor_m))
    # Round toward the ordinary side if metre/km/reference-area conversion would
    # otherwise export even one representational ulp of retained dense material.
    too_deep = removed/1000.*values['area_factor'] > ordinary_volume
    while np.any(too_deep):
        removed = np.where(too_deep, np.nextafter(removed, 0.), removed)
        too_deep = removed/1000.*values['area_factor'] > ordinary_volume
    after = dict(values, thickness_km=values['thickness_km']-removed/1000.)
    realized = _elevation(values)-_elevation(after)
    return dict(denudation_m=removed, requested_net_loss_m=loss.copy(),
                realized_net_loss_m=realized, unmet_net_loss_m=np.maximum(loss-realized, 0.),
                limited=removed < requested, available_ordinary_m=ordinary_m)


def initialize_structure(kind, initial_elevation_m=220.):
    """Initialize buoyant kinds 1/2/3 while preserving every supplied height.

    Neutral reference columns are 35/42/25 km for mobile continental crust,
    cratons and juvenile arcs. These are starting assumptions, not inferred
    petrology. Different reference elevations absorb unresolved initial keel
    and buoyancy structure without imposing an artificial craton plateau.
    """
    kind, initial = np.broadcast_arrays(np.asarray(kind), _finite(initial_elevation_m, 'Initial elevation'))
    if not np.isin(kind, (1, 2, 3)).all():
        raise ValueError('Crustal structure belongs to buoyant material kinds 1, 2 or 3.')
    reference = np.where(kind == 2, 42., np.where(kind == 3, 25., 35.))
    return dict(thickness_km=reference.copy(), reference_thickness_km=reference.copy(),
                reference_elevation_m=initial.copy(), area_factor=np.ones(kind.shape),
                rift_heat_m=np.zeros(kind.shape), rift_age_myr=np.full(kind.shape, -1.),
                foreland_m=np.zeros(kind.shape))


def _validate_exact_volume_transaction(old, new, old_area, new_area):
    """No future per-face error can be hidden by an aggregate cancellation."""
    if (not np.allclose(new_area*new['thickness_km'],old_area*old['thickness_km'],rtol=2e-12,atol=1e-8)
        or not np.allclose(new['area_factor']*new['thickness_km'],
            old['area_factor']*old['thickness_km'],rtol=2e-12,atol=1e-8)):
        raise ValueError('Exact geometric column transaction failed per-face physical/reference volume conservation.')


def evolve_structure(state, dt_myr, *, shortening_km_myr=0., extension_km_myr=0.,
                     denudation_m=0., strength=1., belt_width_km=400., foreland_target_m=0.,
                     geometric_log_area=None, geometric_accuracy=None):
    """Return independent new state and an exactly closing height budget.

    Speeds are normal shortening/extension in km/Myr over a physical belt.
    A .25 partition and strength scale convert them to unresolved strain, capped
    at .08/Myr. Mechanical T*A is conserved; denudation removes actual rock
    thickness and volume. Thickness limits prevent unbounded roots or complete
    numerical loss of continental columns; they do not oceanize material.

    Rift support is an air-equivalent height with bounded strain-driven heating
    and 63-Myr cooling. Foreland load approaches its target in 12 Myr and unloads
    in 100 Myr. All returned height-budget terms are actual surface metres.

    Ledger: total_delta = mechanical_delta + thermal_delta + foreland_delta
                          - net_erosion_loss.
    Also net_erosion_loss = denudation - rebound. If trace_erosion stores NET
    loss, rebound must remain diagnostic and must NOT be added to trace_uplift.
    """
    if np.ndim(dt_myr) or not np.isfinite(float(dt_myr)) or dt_myr <= 0:
        raise ValueError('Structure time step must be a finite positive scalar.')
    dt = float(dt_myr)
    old = _state(state)
    shape = old['thickness_km'].shape
    def parameter(value, name):
        return np.broadcast_to(_finite(value, name), shape)
    short = parameter(shortening_km_myr, 'Shortening')
    extension = parameter(extension_km_myr, 'Extension')
    removal = parameter(denudation_m, 'Denudation')
    resistance = parameter(strength, 'Strength')
    width = parameter(belt_width_km, 'Deformation width')
    target = parameter(foreland_target_m, 'Foreland target')
    if (np.any(short < 0) or np.any(extension < 0) or np.any(removal < 0)
            or np.any(resistance < 0) or np.any(resistance > 1)
            or np.any(width <= 0) or np.any(target < 0)):
        raise ValueError('Rates, denudation and load must be nonnegative; strength lies in [0,1], width is positive.')
    new = {name: value.copy() for name, value in old.items()}
    start_height = _elevation(old)
    strain = np.clip(STRAIN_PARTITION*resistance*(short-extension)/width,
                     -MAX_STRAIN_PER_MYR, MAX_STRAIN_PER_MYR)
    # Clip the logarithmic change before exponentiation, including enormous
    # but finite test durations. Bounds are based on the available actual column.
    old_restored = restored_thickness(old)
    log_change = np.clip(strain*dt, np.log(MIN_THICKNESS_KM/old_restored),
                          np.log(MAX_THICKNESS_KM/old['thickness_km']))
    import numerical_accuracy
    if numerical_accuracy.present(old) and geometric_log_area is None:
        if np.any(strain!=0):raise ValueError('Numerical material columns require actual geometric mechanical strain.')
        geometric_log_area=np.zeros(shape)
        geometric_accuracy={name:old[name].copy() for name in numerical_accuracy.FIELDS}
    if numerical_accuracy.present(old) and geometric_accuracy is None:
        raise ValueError('Numerical material columns require their accepted area accounting transaction.')
    if geometric_log_area is not None:
        # A moving material surface has already realized its horizontal strain.
        # Charge precisely that area change, without the unresolved .25 factor
        # or a second strain estimate from the boundary speeds. Geometry must
        # respect column limits before this transaction is committed.
        log_change = -parameter(geometric_log_area, 'Geometric logarithmic area change')
        target_thickness = old['thickness_km']*np.exp(log_change)
        target_restored = old_restored*np.exp(log_change)
        # One tolerance, shared with the geometric feasibility search that
        # produced this area change (bounded_gravity.FEASIBILITY_TOLERANCE), and
        # RELATIVE like it: an absolute 1e-8 km is 1.3e-9 of the 8 km bound but
        # only 1.3e-10 of the 75 km bound, so the two ends of the same guard were
        # being held to different strictness, and the thick end -- where a
        # collision region sits -- was the loose one. The clamp below still
        # absorbs the representational roundoff between this writer and the
        # absolute 1e-9 that _state reads back at.
        if geometric_accuracy is None and (np.any(target_restored < MIN_THICKNESS_KM*(1.-FEASIBILITY_TOLERANCE))
                or np.any(target_thickness > MAX_THICKNESS_KM*(1.+FEASIBILITY_TOLERANCE))):
            raise ValueError('Geometric deformation exceeded the available crustal column bounds.')
    # The guard above rejects a genuine excursion at the shared relative
    # tolerance, but _state() reads back at an absolute 1e-9, so a value landing
    # in the gap between those tolerances passed the writer and failed the
    # reader. At 342 Myr exactly one face of 21,191 did that: its violation was
    # smaller than six significant figures, and it halted the run. Clamp to the
    # same bounds the guard just verified -- this removes only representational
    # roundoff, never a real excursion, because anything past the guard has
    # already raised. The sibling path in structure_engine._surface_change clips
    # identically.
    if geometric_accuracy is not None:
        import numerical_accuracy
        if geometric_log_area is None or not numerical_accuracy.present(old) or set(geometric_accuracy)!=set(numerical_accuracy.FIELDS):
            raise ValueError('Exact geometric columns require their complete dimensional accuracy ledger.')
        new.update({name: parameter(geometric_accuracy[name], name).copy() for name in numerical_accuracy.FIELDS})
        for name in (numerical_accuracy.REFERENCE,numerical_accuracy.BUDGET):
            if not np.array_equal(new[name],old[name]):
                raise ValueError('Geometric deformation renewed a material numerical area allocation.')
        if np.any(new[numerical_accuracy.SPENT]<old[numerical_accuracy.SPENT]):
            raise ValueError('Geometric deformation refunded cumulative numerical area spending.')
        old_area=old[numerical_accuracy.AREA];new_area=new[numerical_accuracy.AREA]
        if np.any(new_area<=0) or not np.allclose(np.log(new_area/old_area),-log_change,rtol=1e-10,atol=1e-14):
            raise ValueError('Geometric column transaction disagrees with its accepted area change.')
        # Physical V/A is exact to arithmetic precision. The nominal limits are
        # tested in area units, never repaired by discarding crust volume.
        new['thickness_km']=old_area*old['thickness_km']/new_area
        new['area_factor']=old['area_factor']*(new_area/old_area)
        _validate_exact_volume_transaction(old,new,old_area,new_area)
        numerical_accuracy.validate_columns(new,restored_thickness(new),MIN_THICKNESS_KM,MAX_THICKNESS_KM)
    else:
        new['thickness_km'] = np.clip(old['thickness_km']*np.exp(log_change),
                                      conserved_volume_floor(old), MAX_THICKNESS_KM)
        new['area_factor'] *= old['thickness_km']/new['thickness_km']
    mechanical_height = _elevation(new)
    actual_extension = np.maximum(-log_change/dt, 0.)
    heating = HEATING_COUPLING*actual_extension
    rate = heating+1./COOLING_MYR
    equilibrium = MAX_RIFT_HEAT_M*heating/rate
    decay = np.exp(-rate*dt)
    new['rift_heat_m'] = equilibrium+(old['rift_heat_m']-equilibrium)*decay
    new['rift_heat_m'] = np.clip(new['rift_heat_m'], 0., MAX_RIFT_HEAT_M)
    old_age = old['rift_age_myr']
    new['rift_age_myr'] = np.where(actual_extension > 1e-14, 0., np.where(old_age >= 0, old_age+dt, -1.))
    thermal_height = _elevation(new)
    removed = np.minimum(removal, fixed_area_removal_capacity(new)*1000.)
    crust_inventory.erode(new, removed/1000.)
    new['thickness_km'] -= removed/1000.
    eroded_height = _elevation(new)
    net_loss = thermal_height-eroded_height
    target = np.minimum(target, MAX_FORELAND_M)
    response = np.where(target > old['foreland_m'], FORELAND_LOADING_MYR, FORELAND_UNLOADING_MYR)
    new['foreland_m'] = old['foreland_m']+(target-old['foreland_m'])*(-np.expm1(-dt/response))
    final_height = _elevation(new)
    mechanical = mechanical_height-start_height
    budget = dict(mechanical_delta_m=mechanical,
                  shortening_gain_m=np.maximum(mechanical, 0.),
                  extension_loss_m=np.maximum(-mechanical, 0.),
                  thermal_delta_m=thermal_height-mechanical_height,
                  denudation_m=removed, rebound_m=removed-net_loss,
                  net_erosion_loss_m=net_loss,
                  foreland_delta_m=final_height-eroded_height,
                  total_delta_m=final_height-start_height,
                  final_elevation_m=final_height,
                  mechanical_thickness_delta_km=old['thickness_km']*(np.exp(log_change)-1),
                  removed_volume_km_per_reference_km2=removed/1000.*new['area_factor'])
    return new, budget


def coalesce_structure(state, weights, groups):
    """Merge same-owner juvenile columns without losing height or volume.

    Groups must be contiguous zero-based IDs supplied by the caller's existing
    arc-coalescence decision. No continental/cratonic or ownership decision is
    made here. Reference offsets are re-anchored after mixing so weighted mean
    surface height is unchanged, including mixed submerged/emerged columns.
    """
    old = _state(state)
    weight = _finite(weights, 'Reference-area weights')
    group = np.asarray(groups)
    if (weight.ndim != 1 or old['thickness_km'].shape != weight.shape or group.shape != weight.shape
            or not np.issubdtype(group.dtype, np.integer) or np.any(group < 0) or np.any(weight <= 0)):
        raise ValueError('Coalescence needs aligned positive weights and nonnegative integer groups.')
    if not len(weight):
        return {name: value.copy() for name, value in old.items()}
    count = int(group.max())+1
    totals = np.bincount(group, weights=weight, minlength=count)
    if np.any(totals == 0):
        raise ValueError('Coalescence group IDs must be contiguous.')
    mean = lambda values: np.bincount(group, weights=weight*values, minlength=count)/totals
    new = {name: mean(value) for name,value in old.items()}
    import numerical_accuracy
    if numerical_accuracy.present(old):
        for name in numerical_accuracy.FIELDS:
            new[name]=np.bincount(group,weights=old[name],minlength=count)
        # Coalescence cannot renew the per-identity maximum. A caller must keep
        # these identities separate if their accumulated allocations do not fit.
        if np.any(new[numerical_accuracy.BUDGET]>numerical_accuracy.allowance(new[numerical_accuracy.REFERENCE])):
            raise ValueError('Coalescence would combine incompatible numerical area allocations.')
    new['thickness_km'] = mean(old['thickness_km']*old['area_factor'])/new['area_factor']
    valid_age = old['rift_age_myr'] >= 0
    age_weight = np.bincount(group, weights=weight*valid_age, minlength=count)
    new['rift_age_myr'] = np.divide(
        np.bincount(group, weights=weight*np.where(valid_age, old['rift_age_myr'], 0.), minlength=count),
        age_weight, out=np.full(count, -1.), where=age_weight > 0)
    desired = mean(_elevation(old))
    reference_air = (_air_height(desired+new['foreland_m'])-new['rift_heat_m']
                     - AIRY_M_PER_KM*(new['thickness_km']-new['reference_thickness_km']))
    if 'lip_heat_m' in new:
        reference_air -= new['lip_heat_m']
    reference_air -= dense_crust.elevation_correction(new)
    new['reference_elevation_m'] = _water_load(reference_air)
    return new


def foreland_profile(distance_km, load_height_m, *, width_km=250., peak_belt_km=60.,
                     max_depth_m=MAX_FORELAND_M):
    """Positive compact trough beside a load; this is not a flexure solution.

    Zero through the central peak belt, deepest at width_km, and zero again at
    2*width_km-peak_belt_km. Relief below 500 m supplies no mountain load. The
    amplitude is one quarter of additional relief, bounded by max_depth_m.
    """
    if not 0 <= peak_belt_km < width_km or max_depth_m < 0:
        raise ValueError('Foreland widths and depth must describe a positive bounded trough.')
    distance, load = np.broadcast_arrays(_finite(distance_km, 'Foreland distance'),
                                         _finite(load_height_m, 'Mountain load'))
    phase = (distance-peak_belt_km)/(2*(width_km-peak_belt_km))
    shape = np.where((phase > 0) & (phase < 1), np.sin(np.pi*np.clip(phase, 0, 1))**2, 0.)
    return np.minimum(max_depth_m, .25*np.maximum(load-500., 0.))*shape


def foreland_target(xyz, owner_slots, collision_mid, collision_normal, collision_length_km,
                    plate_a, plate_b, load_height_m, *, width_km=250., peak_belt_km=60.,
                    max_depth_m=MAX_FORELAND_M, chunk_size=8192):
    """Sample the nearest same-owner finite collision belt, with bounded memory.

    Call with CURRENT collision segments (not all boundaries). Caller-supplied
    load is the nearby achieved mountain height in metres, not convergence rate.
    A 128x64 spatial bin index limits each segment to nearby material points;
    finite great-circle distances then determine the trough. Points outside the
    relevant owner/locality receive zero. No plate-wide depression is applied.
    """
    points, mid, normal = (_finite(v, n) for v, n in
                           ((xyz, 'Material coordinates'), (collision_mid, 'Collision centres'),
                            (collision_normal, 'Collision normals')))
    owners, pa, pb = map(np.asarray, (owner_slots, plate_a, plate_b))
    length, load = _finite(collision_length_km, 'Collision lengths'), _finite(load_height_m, 'Mountain loads')
    if (points.ndim != 2 or points.shape[1] != 3 or owners.shape != (len(points),)
            or mid.ndim != 2 or mid.shape[1] != 3 or normal.shape != mid.shape
            or any(v.shape != (len(mid),) for v in (pa, pb, length, load))
            or np.any(length < 0) or chunk_size < 1):
        raise ValueError('Collision geometry and owner arrays must align.')
    # Validate the requested trough even when no points/segments are eligible.
    foreland_profile(np.empty(0), np.empty(0), width_km=width_km,
                     peak_belt_km=peak_belt_km, max_depth_m=max_depth_m)
    result = np.zeros(len(points))
    if not len(points) or not len(mid):
        return result
    point_norm, mid_norm = np.linalg.norm(points, axis=1), np.linalg.norm(mid, axis=1)
    if np.any(point_norm == 0) or np.any(mid_norm == 0):
        raise ValueError('Spherical coordinates must be nonzero.')
    points = points/point_norm[:, None]
    mid = mid/mid_norm[:, None]
    normal = normal-mid*np.sum(mid*normal, axis=1)[:, None]
    normal_norm = np.linalg.norm(normal, axis=1)
    if np.any(normal_norm < 1e-12):
        raise ValueError('Collision normals must be tangential to their centres.')
    normal /= normal_norm[:, None]
    tangent = np.cross(normal, mid)
    w, h = 128, 64
    lon = np.arctan2(points[:, 1], points[:, 0])
    lat = np.arcsin(np.clip(points[:, 2], -1, 1))
    cells = np.clip(((np.pi/2-lat)*h/np.pi).astype(int), 0, h-1)*w+(((lon+np.pi)*w/(2*np.pi)).astype(int) % w)
    order = np.argsort(cells, kind='stable')
    counts = np.bincount(cells, minlength=w*h)
    starts = np.r_[0, np.cumsum(counts)]
    cutoff = 2*width_km-peak_belt_km
    # One latitude-cell diagonal is a conservative radius to any bin point,
    # including bins containing a pole. Exact geometry filters excess candidates.
    padding = np.sqrt(2.)*np.pi/h*RADIUS_KM
    segment, bins = candidate_cells(mid, cutoff+length/2+padding, w, h)
    segment_starts = np.r_[0, np.cumsum(np.bincount(segment, minlength=len(mid)))]
    nearest = np.full(len(points), np.inf)
    best_load = np.zeros(len(points))
    for k in range(len(mid)):
        selected_bins = bins[segment_starts[k]:segment_starts[k+1]]
        parts = [order[starts[b]:starts[b+1]] for b in selected_bins if counts[b]]
        if not parts:
            continue
        relevant = np.concatenate(parts)
        relevant = relevant[(owners[relevant] == pa[k]) | (owners[relevant] == pb[k])]
        half = length[k]/(2*RADIUS_KM)
        end_a = np.cos(half)*mid[k]+np.sin(half)*tangent[k]
        end_b = np.cos(half)*mid[k]-np.sin(half)*tangent[k]
        for begin in range(0, len(relevant), int(chunk_size)):
            selected = relevant[begin:begin+int(chunk_size)]
            x = points[selected]
            cross = x@normal[k]
            along = np.arctan2(x@tangent[k], x@mid[k])
            distance = np.where(np.abs(along) <= half,
                                np.abs(np.arcsin(np.clip(cross, -1, 1))),
                                np.arccos(np.clip(np.maximum(x@end_a, x@end_b), -1, 1)))*RADIUS_KM
            better = (distance < nearest[selected]-1e-8) & (distance <= cutoff)
            ties = np.abs(distance-nearest[selected]) <= 1e-8
            best_load[selected] = np.where(better, load[k], np.where(ties, np.maximum(best_load[selected], load[k]), best_load[selected]))
            nearest[selected] = np.minimum(nearest[selected], distance)
    valid = np.isfinite(nearest) & (nearest <= cutoff)
    result[valid] = foreland_profile(nearest[valid], best_load[valid], width_km=width_km,
                                     peak_belt_km=peak_belt_km, max_depth_m=max_depth_m)
    return result
