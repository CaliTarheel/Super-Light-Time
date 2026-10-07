"""Coherent material decisions for the spherical crust representation.

Four quadrature samples represent one original crust cell. They describe its
area and material properties; they are not four independent pieces that a new
plate boundary may send in four directions. Fractures therefore classify the
area-weighted spherical centre of each persistent patch. Original connected
cratons also retain a material identity even when overlapping crust hides them
from the categorical surface raster.
"""
from __future__ import annotations

import numpy as np


def patch_centres(position, mass, patch_ids):
    """Return patch IDs, spherical centroids, and parcel-to-patch indices.

    IDs may be sparse and need not start at zero. Longitudes are never averaged:
    centroids remain well defined across the seam and over either pole. Each
    patch is local, so an antipodal/zero-result centroid indicates invalid data.
    """
    position = np.asarray(position, dtype=float)
    mass = np.asarray(mass, dtype=float)
    patch_ids = np.asarray(patch_ids)
    if position.shape != (len(mass), 3) or patch_ids.shape != mass.shape:
        raise ValueError("Patch positions, masses and IDs must describe the same parcels.")
    if not len(mass):
        return patch_ids.copy(), np.empty((0, 3), float), np.empty(0, np.int64)
    ids, inverse = np.unique(patch_ids, return_inverse=True)
    centres = np.column_stack([
        np.bincount(inverse, weights=mass * position[:, axis], minlength=len(ids))
        for axis in range(3)
    ])
    length = np.linalg.norm(centres, axis=1)
    if np.any(length <= 0) or not np.all(np.isfinite(length)):
        raise ValueError("A material patch needs a finite, nonzero spherical centroid.")
    centres /= length[:, None]
    return ids, centres, inverse


def coherent_partition(position, mass, patch_ids, classifier):
    """Classify each complete material patch, then broadcast to its parcels.

    ``classifier`` maps spherical points to a signed fracture distance. A centre
    exactly on the crack stays on its nonpositive side, matching grid cuts.
    """
    _, centres, inverse = patch_centres(position, mass, patch_ids)
    return (np.asarray(classifier(centres)) > 0)[inverse]


def splits_protected_groups(group_ids, sides):
    """Whether a binary cut divides any enduring, nonnegative craton ID.

    A negative ID marks unprotected mobile/juvenile material. Decisions use all
    material in the candidate owner, including parcels concealed beneath another
    plate's raster cell. They must not depend on visible craton pixels alone.
    """
    group_ids = np.asarray(group_ids)
    sides = np.asarray(sides, dtype=bool)
    if group_ids.shape != sides.shape:
        raise ValueError("Protected group IDs and fracture sides must align.")
    protected = group_ids >= 0
    if not np.any(protected):
        return False
    _, inverse = np.unique(group_ids[protected], return_inverse=True)
    counts = np.bincount(inverse)
    positive = np.bincount(inverse, weights=sides[protected])
    return bool(np.any((positive > 0) & (positive < counts)))
