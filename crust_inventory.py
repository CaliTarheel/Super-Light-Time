"""Conservative removable-crust inventory, in km3 per reference km2.

The reference area follows a material parcel, so strain changes thickness but
does not change this inventory. This is a bulk, well-mixed composition closure:
new crust supplies 60% removable material; erosion exports its current mixture.
Optional dense_crust state retains converted material as a separate basal phase.
With that state, inventories use ordinary-crust-equivalent volume (mass divided
by 2800 kg/m3), and erosion reaches the dense base only after the upper material.
"""
import numpy as np

FRACTION = .6
FIELDS = tuple('foundering_'+name+'_km_per_reference_km2' for name in
               ('baseline', 'added', 'eroded', 'returned', 'remaining'))
BASELINE, ADDED, ERODED, RETURNED, REMAINING = FIELDS


def present(state):
    count = sum(name in state for name in FIELDS)
    if count not in (0, len(FIELDS)):
        raise ValueError('Incomplete removable-crust inventory.')
    return bool(count)


def validate(state):
    if not present(state):
        return
    shape = np.asarray(state['thickness_km']).shape
    thickness = np.asarray(state['thickness_km'])
    area = np.asarray(state['area_factor'])
    if (area.shape != shape or not np.isfinite(thickness).all() or not np.isfinite(area).all()
            or np.any(thickness <= 0) or np.any(area <= 0)):
        raise ValueError('Removable-crust inventory needs positive finite columns and areas.')
    for name in FIELDS:
        value = np.asarray(state[name])
        if value.shape != shape or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError('Invalid removable-crust inventory: '+name)
    supplied = state[BASELINE]+state[ADDED]
    accounted = state[ERODED]+state[RETURNED]+state[REMAINING]
    # The same accumulator problem as dense_crust.validate: BASELINE+ADDED
    # only grows, and its residual on SEP21T had already reached 1.229e-12 by
    # 192 Myr. It survives today only because these values are large enough
    # (145.9) for the relative term to carry it; a cell whose own inventory is
    # small has nothing but this floor. Raised before it stops a run rather
    # than after.
    if not np.allclose(supplied, accounted, rtol=2e-12, atol=1e-9):
        raise ValueError('Removable-crust inventory does not close.')
    volume = state['thickness_km']*state['area_factor']
    # With a retained dense phase, these composition inventories remain mass
    # divided by the ordinary-crust density. Physical volume contracts during
    # conversion; that must not be mistaken for loss of mafic material.
    import dense_crust
    if dense_crust.present(state):
        volume = dense_crust.mass_volume(state)
    if np.any(state[REMAINING] > volume+2e-12*np.maximum(volume, 1.)):
        raise ValueError('Removable crust exceeds retained column volume.')


def initialize(state, *, legacy=False):
    """Explicit baseline; legacy conversion preserves the old remaining cap.

    Historical removal cannot be reconstructed from the old thickness counter
    after strain. The migration boundary is the start of the new volume ledger.
    """
    if any(name in state for name in FIELDS):
        raise ValueError('Cannot reset an existing removable-crust inventory.')
    thickness = np.asarray(state['thickness_km'], float)
    allowance = FRACTION*thickness
    if legacy:
        allowance = np.maximum(allowance-(1.-FRACTION)*state['foundered_m']/1000., 0.)
    volume = allowance*state['area_factor']
    state.update({name: np.zeros_like(thickness) for name in FIELDS})
    state[BASELINE] = volume.copy()
    state[REMAINING] = volume.copy()
    validate(state)


def erode(state, removed_km):
    """Book erosion BEFORE changing thickness; export the current bulk mixture."""
    if not present(state):
        return
    import dense_crust
    if dense_crust.present(state):
        dense_crust.erosion(state, removed_km)
        return
    validate(state)
    removed = np.broadcast_to(np.asarray(removed_km, float), state['thickness_km'].shape)
    if not np.isfinite(removed).all() or np.any(removed < 0) or np.any(removed > state['thickness_km']):
        raise ValueError('Erosion exceeds the retained column.')
    exported = state[REMAINING]*(removed/state['thickness_km'])
    state[ERODED] = state[ERODED]+exported
    state[REMAINING] = np.maximum(state[REMAINING]-exported, 0.)


def add(state, volume, *, temperature_c=1200.):
    """Book actual new crust AFTER changing thickness, per reference km2."""
    if not present(state):
        return
    volume = np.broadcast_to(np.asarray(volume, float), state['thickness_km'].shape)
    if not np.isfinite(volume).all() or np.any(volume < 0):
        raise ValueError('Added crust volume must be finite and nonnegative.')
    supplied = FRACTION*volume
    state[ADDED] = state[ADDED]+supplied
    state[REMAINING] = state[REMAINING]+supplied
    import dense_crust
    if dense_crust.present(state):
        dense_crust.add_heat(state, volume, temperature_c)
    validate(state)


def founder(state, removed_km):
    """Book mantle return BEFORE changing thickness."""
    if not present(state):
        return
    import dense_crust
    if dense_crust.present(state):
        raise ValueError('A retained dense phase must use its conservative reaction/drainage transaction.')
    validate(state)
    volume = np.asarray(removed_km, float)*state['area_factor']
    if (not np.isfinite(volume).all() or np.any(volume < 0)
            or np.any(volume > state[REMAINING]+1e-12)):
        raise ValueError('Mantle return exceeds removable crust.')
    state[RETURNED] = state[RETURNED]+volume
    state[REMAINING] = np.maximum(state[REMAINING]-volume, 0.)


def grow_reference(state, indices, old_area, new_area, added_volume):
    """Dilute per-reference ledgers and supply magma at an arc-growth boundary.

    Areas are reference km2; added_volume is physical km3. The caller has
    already assigned the new physical thickness and area factor.
    """
    if not present(state):
        return
    subset = {name: value[indices].copy() for name, value in state.items()}
    import dense_crust
    fields = FIELDS+(dense_crust.FIELDS if dense_crust.present(state) else ())
    for name in fields:
        subset[name] *= old_area/new_area
    add(subset, added_volume/new_area)
    for name in fields:
        state[name][indices] = subset[name]
