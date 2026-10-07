"""Broad seeded basement topography, sampled once at material birth.

This is an initial condition, not a geological process or fine terrain model.
Smooth three-dimensional waves avoid longitude seams, polar singularities and
independent cell noise. The caller carries the sampled scalar with its material;
it must not resample this field at the material's later world position.
"""
from __future__ import annotations

import numpy as np


def initial_basement_relief(xyz, seed):
    """Return relief in [0, 220] m at unit-sphere positions, independent of grid.

    Every mode has zero spherical mean because its wavenumber is an integer
    multiple of pi: the spherical mean of sin(k dot x + phase) is
    sin(phase) * sin(|k|) / |k|. Positive weights summing to one therefore give
    a whole-sphere mean of 110 m and an analytic 0--220 m bound. A selected
    continent need not have that mean. The shortest wavelength is ~4,250 km
    on an Earth-sized sphere. Seeded orientations and phases have their own
    random stream, so evaluating relief cannot perturb subsequent tectonics.
    """
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Basement relief positions must be finite Nx3 vectors.')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x43A16F2D]))
    axes = rng.normal(size=(6, 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    phases = rng.uniform(0., 2*np.pi, len(axes))
    wavenumbers = np.pi*np.array([1., 1., 2., 2., 3., 3.])
    weights = np.array([.34, .24, .17, .11, .08, .06])
    relief = np.empty(len(points), dtype=np.float64)
    # Bound temporary allocations independently of the material parcel count.
    for start in range(0, len(points), 65_536):
        block = points[start:start+65_536]
        field = np.zeros(len(block), dtype=np.float64)
        for axis, phase, wavenumber, weight in zip(axes, phases, wavenumbers, weights):
            field += weight*np.sin(wavenumber*(block @ axis)+phase)
        relief[start:start+len(block)] = 110.*(1.+field)
    return relief
