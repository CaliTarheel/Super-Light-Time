"""Unilateral vertical channel element for a descending conserved crust column.

This local constitutive element is not installed in the native solver. A
horizontal overlap does not by itself determine which crust is above, the
vertical reference, or the rate at which a contact changes into a channel.
Those geometry and source transactions must be supplied before this element
can replace either the existing stack or entry potential.
"""

import numpy as np

import column_density


GRAVITY_M_S2 = 9.81


def buoyancy_surface_load(volume_km3, mass_kg, area_km2):
    """Signed upward buoyancy per horizontal area from conserved lower crust.

    Positive values oppose descent. A dense column can give a negative value.
    No material is created: the mantle displacement uses the supplied volume.
    """
    volume, mass, area = np.broadcast_arrays(
        np.asarray(volume_km3, float), np.asarray(mass_kg, float),
        np.asarray(area_km2, float))
    if (not np.isfinite(volume).all() or not np.isfinite(mass).all()
            or not np.isfinite(area).all() or np.any(volume <= 0.)
            or np.any(mass <= 0.) or np.any(area <= 0.)):
        raise ValueError('Channel buoyancy needs positive finite volume, mass and area.')
    load = GRAVITY_M_S2 * (column_density.RHO_MANTLE * volume * 1e9 - mass) / (area * 1e6)
    if not np.isfinite(load).all():
        raise ValueError('Channel buoyancy exceeds the finite force range.')
    return load


def equilibrate(preferred_depth_m, upper_base_depth_m, buoyancy_n_m2,
                bending_stiffness_n_m3, *, upper_restoring_stiffness_n_m3=None):
    """Minimize a local lower-path spring, buoyancy and upper uplift energy.

    Depth is positive downward. The upper base is at ``H`` and the descending
    crust's top must satisfy ``z+w >= H`` where ``w`` is upper uplift. Its
    unconstrained preferred path is ``z0``. For local path stiffness ``K``,
    upper restoring stiffness ``Ku`` and signed upward buoyancy ``B``, the
    energy per horizontal area is ``K/2 * (z-z0)**2 + B*z + Ku/2 * w**2``.
    The reaction ``lambda`` is nonnegative and acts only at contact. If Ku is
    omitted, the upper base is rigid (w=0), recovering the original element.
    Envelope derivatives return forces conjugate to the preferred entry path
    and to the moving, unloaded upper base.

    The energy datum is the surface, so callers must also account for any
    previous stack energy when material changes mode. This function neither
    makes that transfer nor prescribes a value of K. The pointwise spring is
    not a nonlocal plate-bending operator; its physical closure is audited in
    CHANNEL_FLEXURE_AUDIT.md.
    """
    values = [np.asarray(value, float) for value in
              (preferred_depth_m, upper_base_depth_m, buoyancy_n_m2,
               bending_stiffness_n_m3)]
    if upper_restoring_stiffness_n_m3 is not None:
        values.append(np.asarray(upper_restoring_stiffness_n_m3, float))
    preferred, base, buoyancy, stiffness, *upper = np.broadcast_arrays(*values)
    if (not all(np.isfinite(value).all() for value in (preferred, base, buoyancy, stiffness))
            or np.any(preferred < 0.) or np.any(base < 0.) or np.any(stiffness <= 0.)):
        raise ValueError('Channel depths and stiffness must be finite, with nonnegative depths and positive stiffness.')
    upper_stiffness = upper[0] if upper else None
    if upper_stiffness is not None:
        if not np.isfinite(upper_stiffness).all() or np.any(upper_stiffness <= 0.):
            raise ValueError('Upper vertical restoring stiffness must be finite and positive.')
    free_depth = preferred - buoyancy / stiffness
    contact = free_depth <= base
    deficit = np.maximum(base - free_depth, 0.)
    if upper_stiffness is None:
        reaction = stiffness * deficit
        uplift = np.zeros_like(reaction)
    else:
        reduced = 1. / (1. / stiffness + 1. / upper_stiffness)
        reaction = reduced * deficit
        uplift = reaction / upper_stiffness
    depth = free_depth + reaction / stiffness
    energy = .5 * stiffness * (depth - preferred)**2 + buoyancy * depth
    if upper_stiffness is not None:
        energy += .5 * upper_stiffness * uplift**2
    if not all(np.isfinite(value).all() for value in (depth, uplift, reaction, energy)):
        raise ValueError('Channel equilibrium exceeds the finite energy or force range.')
    return dict(depth_m=depth, energy_j_m2=energy, contact=contact,
                reaction_pa=reaction, upper_uplift_m=uplift,
                mantle_gap_m=depth+uplift-base,
                preferred_depth_derivative_n_m2=stiffness * (preferred-depth),
                upper_base_derivative_n_m2=reaction,
                buoyancy_derivative_m=depth)
