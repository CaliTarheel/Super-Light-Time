"""Shared conservative projection of local phase-source columns to faces."""

import numpy as np

import crustal_structure


def coalesce_regions(expanded, groups, weights, face_count, *, measure):
    """Coalesce regional columns with an explicit area-weight convention.

    Native phase rows are fractions of one face with the same area factor, so
    their existing reference-area weights and non-structure counter sums are
    preserved exactly. Independent material subregions can acquire different
    area factors under deformation; physical-area fractions are then converted
    to reference-area weights before mixing per-reference inventories.
    """
    group = np.asarray(groups)
    weight = np.asarray(weights, float)
    if (measure not in ('reference', 'physical')
            or group.ndim != 1 or weight.shape != group.shape
            or not np.issubdtype(group.dtype, np.integer)
            or isinstance(face_count, (bool, np.bool_))
            or not isinstance(face_count, (int, np.integer))
            or face_count < 1 or np.any(group < 0)
            or np.any(group >= face_count) or not np.isfinite(weight).all()
            or np.any(weight <= 0.)):
        raise ValueError('Region projection needs positive aligned area weights and face groups.')
    if measure == 'physical':
        factors = np.asarray(expanded['area_factor'], float)
        if (factors.shape != weight.shape or not np.isfinite(factors).all()
                or np.any(factors <= 0.)):
            raise ValueError('Physical region projection needs positive local area factors.')
        reference_weight = weight / factors
    else:
        reference_weight = weight
    merged = crustal_structure.coalesce_structure(
        expanded, reference_weight, group)
    totals = np.bincount(group, weights=reference_weight,
                         minlength=face_count)
    for key, value in expanded.items():
        if key not in merged:
            numerator = np.bincount(
                group, weights=reference_weight * value,
                minlength=face_count)
            merged[key] = (numerator if measure == 'reference'
                           else numerator / totals)
    return merged
