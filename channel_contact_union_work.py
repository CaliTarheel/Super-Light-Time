"""Read-only shared-strip work oracle for a union of contact intervals.

All plate paths, material stiffnesses and upper basal references are explicit
inputs. This composition tests geometry, dimensions and reciprocal work; it
is not a native force, phase-source or ownership transaction.
"""

import numpy as np

from channel_two_plate_flexure import anchored_two_plate_contact


def _per_ray(value, count, name):
    result = np.asarray(value, float)
    if result.shape == ():
        result = np.full(count, float(result))
    if (result.shape != (count,) or not np.isfinite(result).all()
            or np.any(result <= 0.)):
        raise ValueError(name + ' must be positive for every hinge ray.')
    return result


def integrate_two_plate_contact_union(line_union, offsets_m,
                                      preferred_depth_m,
                                      unloaded_upper_base_depth_m,
                                      buoyancy_pa, lower_rigidity_n_m,
                                      upper_rigidity_n_m,
                                      upper_foundation_n_m3, *,
                                      free_upper_hinge=False):
    """Integrate one full bending solve per unique along-hinge position.

    Contact is enabled only at nodes inside the union's normal intervals.
    A normal grid that misses any interval fails closed; convergence of the
    pointwise constraint at interval edges still needs refinement checks.
    """
    if not isinstance(line_union, dict):
        raise ValueError('Shared-strip work needs a checked hinge-line union.')
    weights = np.asarray(line_union.get('line_weight_m'), float)
    intervals = line_union.get('contact_intervals')
    offsets = np.asarray(offsets_m, float)
    preferred = np.asarray(preferred_depth_m, float)
    base = np.asarray(unloaded_upper_base_depth_m, float)
    buoyancy = np.asarray(buoyancy_pa, float)
    if (weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all()
            or np.any(weights <= 0.) or not isinstance(intervals, list)
            or len(intervals) != len(weights)
            or offsets.ndim != 1 or len(offsets) < 5
            or not np.isfinite(offsets).all() or offsets[0] < 0.
            or np.any(np.diff(offsets) <= 0.)
            or not np.allclose(np.diff(offsets),
                               (offsets[-1] - offsets[0]) / (len(offsets) - 1),
                               rtol=1e-5, atol=1e-5)
            or preferred.shape != (len(weights), len(offsets))
            or base.shape != preferred.shape
            or buoyancy.shape not in ((), preferred.shape)
            or not np.isfinite(preferred).all() or not np.isfinite(base).all()
            or not np.isfinite(buoyancy).all()
            or np.any(preferred < 0.) or np.any(base < 0.)):
        raise ValueError('Shared-strip work needs aligned line, depth and force fields.')
    if type(free_upper_hinge) is not bool:
        raise ValueError('Shared-strip hinge boundary must be explicitly boolean.')
    buoyancy = np.broadcast_to(buoyancy, preferred.shape)
    dl = _per_ray(lower_rigidity_n_m, len(weights), 'Lower rigidity')
    du = _per_ray(upper_rigidity_n_m, len(weights), 'Upper rigidity')
    foundation = _per_ray(upper_foundation_n_m3, len(weights), 'Upper foundation')
    spacing = float(np.mean(np.diff(offsets)))
    masks = np.zeros_like(base, bool)
    for index, row in enumerate(intervals):
        if not isinstance(row, list) or not row:
            raise ValueError('Every hinge ray needs disjoint contact intervals.')
        merged = []
        for interval in row:
            if (not isinstance(interval, dict)
                    or not {'lower_offset_m', 'upper_offset_m'} <= set(interval)):
                raise ValueError('Shared-strip work needs offset-bounded contact intervals.')
            lo = float(interval['lower_offset_m'])
            hi = float(interval['upper_offset_m'])
            if abs(lo-offsets[0]) <= 1e-5:
                lo = float(offsets[0])
            if abs(hi-offsets[-1]) <= 1e-5:
                hi = float(offsets[-1])
            if (not np.isfinite([lo, hi]).all() or not offsets[0] <= lo < hi <= offsets[-1]):
                raise ValueError('Contact interval extends beyond the anchored strip.')
            if merged and lo < merged[-1][1] - 1e-5:
                raise ValueError('Normal nodes miss or duplicate a contact interval.')
            if merged and lo <= merged[-1][1] + 1e-5:
                merged[-1][1] = max(merged[-1][1],hi)
            else:
                merged.append([lo,hi])
        # Adjacent source cells share one mechanical contact interval. Requiring
        # a grid node in every tiny source cell would make the bending energy
        # depend on how the same region was triangulated.
        for lo,hi in merged:
            selected = (offsets >= lo) & (offsets <= hi)
            if not np.any(selected) or np.any(masks[index] & selected):
                raise ValueError('Normal nodes miss or duplicate a contact interval.')
            masks[index] |= selected
    solutions = [anchored_two_plate_contact(
        preferred[index], base[index], buoyancy[index],
        dl[index], du[index], foundation[index], spacing,
        free_upper_hinge=free_upper_hinge, contact_mask=masks[index])
        for index in range(len(weights))]
    line_energy = np.asarray([row['energy_j_per_m'] for row in solutions])
    reaction = np.asarray([row['reaction_pa'] for row in solutions])
    basal_force = np.asarray([row['basal_reference_derivative_n_per_m']
                              for row in solutions])
    return dict(energy_j=float(weights @ line_energy),
                line_energy_j_per_m=line_energy,
                contact_mask=masks,
                reaction_pa=reaction,
                basal_reference_derivative_n=basal_force*weights[:,None]*masks,
                hinge_line_reaction_n_per_m=np.asarray(
                    [row['hinge_line_reaction_n_per_m'] for row in solutions]),
                depth_m=np.asarray([row['depth_m'] for row in solutions]),
                upper_uplift_m=np.asarray([row['upper_uplift_m'] for row in solutions]),
                minimum_gap_m=min(row['minimum_gap_m'] for row in solutions),
                maximum_free_force_residual_n_per_m=max(
                    row['free_force_residual_n_per_m'] for row in solutions),
                scope='read-only shared one-dimensional strips; no native force or source commit')
