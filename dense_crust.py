"""Conservative retained dense-crust phase and a finite thermal reservoir.

The low-level column transaction is independent of geometry. Its caller supplies
the eligible fraction of unconverted mafic material and a conductive thermal
bath for each column/region. This is a reduced two-stage reaction/drainage model,
not a thermodynamic database or resolved Rayleigh-Taylor instability solver.

Drainage is reserve-limited: mantle return can spend only the ordinary-equivalent
reserve above the caller's phase-adjusted mechanical floor. Dense material that
the reserve cannot release remains in the column, and the withheld amount is
reported. See PHASE-ADJUSTED-COLUMN-FLOOR.md.
"""
import numpy as np
import column_density as density
import crust_inventory as inventory

VERSION = 1
FLOOR_VERSION = 2
RATIO = density.RHO_CRUST/density.RHO_DENSE
MECHANICAL_FLOOR_KM = 8.
FLOOR_TOLERANCE_KM = 1e-9
DENSE = density.DENSE_VOLUME_FIELD
CONVERTED = 'dense_converted_km_per_reference_km2'
ERODED = 'dense_eroded_km_per_reference_km2'
RETURNED = 'dense_returned_km_per_reference_km2'
HEAT = 'crust_heat_km_c_per_reference_km2'
HEAT_BASELINE = 'crust_heat_baseline_km_c_per_reference_km2'
HEAT_ADDED = 'crust_heat_added_km_c_per_reference_km2'
HEAT_ERODED = 'crust_heat_eroded_km_c_per_reference_km2'
HEAT_RETURNED = 'crust_heat_returned_km_c_per_reference_km2'
HEAT_BATH = 'crust_heat_bath_km_c_per_reference_km2'
FIELDS = (DENSE, CONVERTED, ERODED, RETURNED, HEAT, HEAT_BASELINE, HEAT_ADDED,
          HEAT_ERODED, HEAT_RETURNED, HEAT_BATH)
FRAME_FIELDS={'material_'+name:name for name in FIELDS}
FRAME_FIELDS.update(material_dense_column_thickness_km='thickness_km',
                    material_dense_column_area_factor='area_factor')
ARRAY_FIELDS={name:np.float64 for name in FRAME_FIELDS}
ARRAY_FIELDS['material_crust_temperature_c']=np.float64
REACTION_TEMPERATURE_C = 600.
REACTION_TAU_MYR = 10.
DETACHMENT_TAU_MYR = 20.
HEAT_CAPACITY_J_KG_K = 1200.


def present(state):
    count = sum(name in state for name in FIELDS)
    if count not in (0, len(FIELDS)):
        raise ValueError('Incomplete retained dense-crust/thermal state.')
    return bool(count)


def mass_volume(state):
    """Mass divided by ordinary-crust density, per reference area (km3/km2)."""
    physical = np.asarray(state['thickness_km'])*np.asarray(state['area_factor'])
    return physical+(1./RATIO-1.)*state[DENSE]


def phase_volume_loss(state):
    """Mass-conserving contraction of retained dense crust, per reference area.

    Only dense material still present in the column is restored here. Mantle
    return and surface erosion remove mass and must therefore spend the ordinary
    equivalent mechanical reserve rather than creating virtual floor capacity.
    """
    if not present(state):
        return np.zeros_like(np.asarray(state['thickness_km'], float))
    return np.asarray(state[DENSE], float)*(1./RATIO-1.)


def restored_thickness(state):
    """Current mass-equivalent thickness with retained-phase contraction restored."""
    return (np.asarray(state['thickness_km'], float)
            +phase_volume_loss(state)/np.asarray(state['area_factor'], float))


def temperature(state):
    return np.asarray(state[HEAT])/mass_volume(state)


def validate(state):
    if not present(state):
        return
    shape = np.asarray(state['thickness_km']).shape
    for key in FIELDS:
        value = np.asarray(state[key])
        if value.shape != shape or not np.isfinite(value).all() or (key != HEAT_BATH and np.any(value < 0)):
            raise ValueError('Invalid dense-crust/thermal state: '+key)
    physical = np.asarray(state['thickness_km'])*np.asarray(state['area_factor'])
    if not np.isfinite(physical).all() or np.any(physical <= 0) or np.any(state[DENSE] > physical+1e-12):
        raise ValueError('Retained dense phase exceeds its physical column.')
    restored = restored_thickness(state)
    if not np.isfinite(restored).all() or np.any(restored < MECHANICAL_FLOOR_KM-FLOOR_TOLERANCE_KM):
        raise ValueError('Retained-phase column exceeds its phase-adjusted mechanical floor.')
    if not inventory.present(state):
        raise ValueError('Retained phase requires the conservative composition inventory.')
    if np.any(state[DENSE]/RATIO > state[inventory.REMAINING]+1e-12):
        raise ValueError('Retained dense phase exceeds the remaining mafic composition.')
    # CONVERTED only ever grows, so its rounding accumulates with the RUN and
    # not with the value: measured on run SEP21T the worst residual rises dead
    # linearly at about one ULP per step (7.69e-14 at 16 Myr, 6.41e-13 at 128,
    # 9.97e-13 at 192) and crossed a fixed 1e-12 at 194 Myr, stopping the run
    # on 2 cells of 36,955 at 1.000834e-12 - 0.08% over, with no jump and no
    # leak. Where a cell's own value is small the relative term contributes
    # nothing, so this floor is the only allowance an accumulator has. 1e-9 km
    # is a micron on a column kilometres thick, and matches the sensible-heat
    # inventory immediately below, which has always used it.
    if not np.allclose(state[CONVERTED], state[DENSE]+state[ERODED]+state[RETURNED], rtol=3e-12, atol=1e-9):
        raise ValueError('Converted dense material is not conserved.')
    sources = state[HEAT_BASELINE]+state[HEAT_ADDED]+state[HEAT_BATH]
    sinks = state[HEAT]+state[HEAT_ERODED]+state[HEAT_RETURNED]
    if not np.allclose(sources, sinks, rtol=3e-12, atol=1e-9):
        raise ValueError('Crustal sensible-heat inventory is not conserved.')
    inventory.validate(state)


def initialize(state, temperature_c):
    """Explicit boundary: no inherited phase or unrecorded hot-root history."""
    if any(key in state for key in FIELDS):
        raise ValueError('Cannot overwrite retained phase or thermal history.')
    inventory.validate(state)
    if not inventory.present(state):
        raise ValueError('Initialize the composition inventory before the retained phase.')
    t = np.broadcast_to(np.asarray(temperature_c, float), state['thickness_km'].shape)
    if not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError('An explicit finite nonnegative initial temperature is required.')
    state.update({key: np.zeros_like(t) for key in FIELDS})
    state[HEAT] = state['thickness_km']*state['area_factor']*t
    state[HEAT_BASELINE] = state[HEAT].copy()
    validate(state)


def _chain(unconverted, dense_equivalent, rate, elapsed, detach_rate):
    """Exact U -> D -> R solution for constant rates; all values in mass units."""
    a, b, t = np.broadcast_arrays(rate, detach_rate, elapsed)
    decay_a, decay_b = np.exp(-a*t), np.exp(-b*t)
    difference = b-a
    # expm1 avoids cancellation when the two rates nearly coincide. Rewriting
    # the factor with the slower exponential avoids overflow at long times.
    delta = np.abs(difference)*t
    factor = np.divide(-np.expm1(-delta), np.abs(difference), out=t.copy(), where=np.abs(difference) > 1e-15)
    feed = unconverted*a*np.exp(-np.minimum(a, b)*t)*factor
    ordinary = unconverted*decay_a
    retained = dense_equivalent*decay_b+feed
    returned = np.maximum(unconverted+dense_equivalent-ordinary-retained, 0.)
    return ordinary, retained, returned


def advance(state, eligible_fraction, bath_temperature_c, thermal_tau_myr, dt, *,
            reaction_tau_myr=REACTION_TAU_MYR, detachment_tau_myr=DETACHMENT_TAU_MYR,
            detachment_fraction=0., minimum_thickness_km=0., drainage_rate_myr=None):
    """Stage and commit one conservative column reaction/drainage transaction.

    Thermal relaxation is analytic. Reaction acts only for the part of this
    interval actually above 600 C. The caller must explicitly admit detachment;
    density alone is not proof that a strong lithosphere permits drainage. An
    admitted dense phase drains after it forms, including cooling intervals;
    it is never deleted at the moment of conversion. Sensible
    heat exported by drained material is evaluated at the endpoint temperature
    (explicit operator splitting), and the conductive bath exchange is recorded.
    Reactions are athermal in this closure: no latent heat is silently invented.
    """
    validate(state)
    if not present(state):
        raise ValueError('Retained phase advance requires an explicit initial state.')
    if not np.isfinite(dt) or dt < 0:
        raise ValueError('Retained phase advance needs finite nonnegative elapsed time.')
    controls = np.array([reaction_tau_myr, detachment_tau_myr, minimum_thickness_km], float)
    if not np.isfinite(controls).all() or np.any(controls[:2] <= 0) or controls[2] < 0:
        raise ValueError('Reaction/drainage times must be positive and the column floor nonnegative.')
    if np.any(restored_thickness(state) < minimum_thickness_km-FLOOR_TOLERANCE_KM):
        raise ValueError('Retained-phase state is already below the requested phase-adjusted floor.')
    shape = state['thickness_km'].shape
    fraction, bath, tau, detach = [np.broadcast_to(np.asarray(v, float), shape) for v in
                           (eligible_fraction, bath_temperature_c, thermal_tau_myr, detachment_fraction)]
    if (not np.isfinite(fraction).all() or np.any((fraction < 0) | (fraction > 1))
            or not np.isfinite(bath).all() or np.any(bath < 0)
            or not np.isfinite(detach).all() or np.any((detach < 0) | (detach > 1))
            or not np.isfinite(tau).all() or np.any(tau <= 0)):
        raise ValueError('Finite eligibility, bath temperature and positive thermal time must align.')
    original_mass = mass_volume(state)
    start_t = temperature(state)
    end_t = bath+(start_t-bath)*np.exp(-dt/tau)
    initially_hot = start_t >= REACTION_TEMPERATURE_C
    finally_hot = end_t >= REACTION_TEMPERATURE_C
    crossing = initially_hot != finally_hot
    ratio = np.divide(REACTION_TEMPERATURE_C-bath, start_t-bath,
                      out=np.ones(shape), where=crossing)
    at = np.clip(-tau*np.log(np.maximum(ratio, np.finfo(float).tiny)), 0., dt)
    hot_duration = np.where(initially_hot, np.where(finally_hot, dt, at),
                            np.where(finally_hot, dt-at, 0.))
    before_hot = np.where(~initially_hot & finally_hot, at, 0.)
    after_hot = dt-before_hot-hot_duration
    unconverted = np.maximum(state[inventory.REMAINING]-state[DENSE]/RATIO, 0.)
    old_dense_eq = state[DENSE]/RATIO
    detach_rate = detach/detachment_tau_myr
    if drainage_rate_myr is not None:
        detach_rate=np.broadcast_to(np.asarray(drainage_rate_myr,float),shape)
        if not np.isfinite(detach_rate).all() or np.any(detach_rate<0):
            raise ValueError('Mechanical drainage rate must be finite and nonnegative.')
    # Cold time before/after reaction still unloads an already dense root.
    first_dense = old_dense_eq*np.exp(-before_hot*detach_rate)
    ordinary, dense_eq, _ = _chain(unconverted, first_dense, fraction/reaction_tau_myr,
                                  hot_duration, detach_rate)
    dense_eq *= np.exp(-after_hot*detach_rate)
    converted_eq = np.maximum(unconverted-ordinary, 0.)
    # Compare transfers on the dense-phase scale. Adding the much larger
    # unchanged ordinary inventory before subtracting it can manufacture a
    # positive return even when both reaction and drainage are exactly zero.
    returned_eq = np.maximum((old_dense_eq-dense_eq)+converted_eq, 0.)
    area = state['area_factor']
    # Reserve-limited drainage. Mantle return spends the ordinary-equivalent
    # reserve above the phase-adjusted mechanical floor and can never take a
    # column below it. Dense material that the reserve cannot release stays in
    # the column as retained dense phase: nothing is deleted, rescaled or moved
    # to another ledger, and conversion kinetics are untouched. A column with
    # no reserve therefore keeps its whole dense root until explicit parcel
    # retirement/oceanization exists. The withheld amount is reported so that
    # this closure is visible in every frame rather than silently absorbed.
    floor = max(float(minimum_thickness_km), MECHANICAL_FLOOR_KM)
    reserve_eq = np.maximum((restored_thickness(state)-floor)*area, 0.)
    withheld_eq = np.maximum(returned_eq-reserve_eq, 0.)
    returned_eq = returned_eq-withheld_eq
    dense_eq = dense_eq+withheld_eq
    contraction = converted_eq*(1.-RATIO)
    physical_return = returned_eq*RATIO
    requested = contraction+physical_return
    remaining_physical = state['thickness_km']*area-requested
    if np.any(remaining_physical <= 0):
        # The phase-adjusted floor below distinguishes conservative contraction
        # from actual mass return. Exhausting the material parcel itself still
        # needs an explicit retirement/oceanization transaction.
        raise ValueError('Phase conversion/drainage exhausts a physical material column; extend parcel lifecycle mechanics.')
    staged = {key: np.asarray(value).copy() for key, value in state.items()}
    staged[DENSE] = np.maximum(dense_eq*RATIO, 0.)
    staged[CONVERTED] += converted_eq*RATIO
    staged[RETURNED] += physical_return
    staged[inventory.RETURNED] += returned_eq
    staged[inventory.REMAINING] = np.maximum(staged[inventory.REMAINING]-returned_eq, 0.)
    staged['thickness_km'] -= requested/area
    if 'reference_thickness_km' in staged:
        # Conversion/removal changes the stress-free material column as well
        # as its strained thickness. Preserve the geometric strain ratio, and
        # re-anchor the neutral elevation so this bookkeeping does not add or
        # cancel the actual mass/density-driven surface response.
        import crustal_structure as columns
        old_reference = staged['reference_thickness_km'].copy()
        staged['reference_thickness_km'] *= staged['thickness_km']/state['thickness_km']
        air = columns._air_height(staged['reference_elevation_m'])
        air += columns.AIRY_M_PER_KM*(staged['reference_thickness_km']-old_reference)
        staged['reference_elevation_m'] = columns._water_load(air)
    bath_exchange = original_mass*(end_t-start_t)
    exported_heat = returned_eq*end_t
    staged[HEAT_BATH] += bath_exchange
    staged[HEAT_RETURNED] += exported_heat
    staged[HEAT] = mass_volume(staged)*end_t
    if 'foundered_m' in staged:
        staged['foundered_m'] += physical_return/area*1000.
    validate(staged)
    if np.any(restored_thickness(staged) < minimum_thickness_km-FLOOR_TOLERANCE_KM):
        raise ValueError('Phase transaction exceeded the requested phase-adjusted floor.')
    inventory.validate(staged)
    state.update(staged)
    return dict(converted_equivalent_km=converted_eq/area,
                contraction_km=contraction/area, returned_km=physical_return/area,
                returned_equivalent_km=returned_eq/area, hot_duration_myr=hot_duration,
                bath_heat_km_c_per_reference_km2=bath_exchange,
                withheld_equivalent_km=withheld_eq/area, reserve_limited=withheld_eq > 0.)


def erosion(state, removed_km):
    """Remove ordinary upper crust first, then dense basal crust; before T changes.

    Updates the composition and phase/heat inventories together. Remaining
    unconverted mafic material is mixed within the ordinary upper column.
    """
    validate(state)
    removed = np.broadcast_to(np.asarray(removed_km, float), state['thickness_km'].shape)*state['area_factor']
    total = state['thickness_km']*state['area_factor']
    if not np.isfinite(removed).all() or np.any(removed < 0) or np.any(removed > total+1e-12):
        raise ValueError('Erosion exceeds the retained physical column.')
    ordinary = total-state[DENSE]
    dense_removed = np.maximum(removed-ordinary, 0.)
    ordinary_removed = np.minimum(removed, ordinary)
    unconverted = state[inventory.REMAINING]-state[DENSE]/RATIO
    eligible_removed = (np.divide(unconverted*ordinary_removed, ordinary,
                                  out=np.zeros_like(ordinary), where=ordinary > 0)+dense_removed/RATIO)
    exported_mass = ordinary_removed+dense_removed/RATIO
    exported_heat = exported_mass*temperature(state)
    state[HEAT] -= exported_heat
    state[HEAT_ERODED] += exported_heat
    state[DENSE] = np.maximum(state[DENSE]-dense_removed, 0.)
    state[ERODED] += dense_removed
    state[inventory.REMAINING] = np.maximum(state[inventory.REMAINING]-eligible_removed, 0.)
    state[inventory.ERODED] += eligible_removed


def add_heat(state, volume, temperature_c):
    """Admit ordinary-crust magma's sensible heat after physical volume addition."""
    volume, t = np.broadcast_arrays(np.asarray(volume, float), np.asarray(temperature_c, float))
    if not np.isfinite(volume).all() or not np.isfinite(t).all() or np.any(volume < 0) or np.any(t < 0):
        raise ValueError('Magma heat admission requires actual volume and a finite temperature.')
    supplied = volume*t
    state[HEAT] += supplied
    state[HEAT_ADDED] += supplied


def elevation_correction(state):
    """Air-equivalent metres from replacing ordinary material with dense crust."""
    if not present(state):
        return 0.
    return -(density.RHO_DENSE-density.RHO_CRUST)/density.RHO_MANTLE*1000.*state[DENSE]/state['area_factor']
