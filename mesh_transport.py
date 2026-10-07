"""Conservative finite-volume transport on closed spherical triangular cells.

The integrated normal flux of rigid Euler motion along a great-circle edge is
R² omega dot (end-start). These endpoint differences telescope around each
closed face, preserving a uniform field without a latitude/pole correction.
No display raster or interpolation back from a rendered map is used.
"""
from __future__ import annotations
import numpy as np

# Resource guard, not a physical speed cap. Reject the entire request before
# integration instead of silently shortening its interval or clipping motion.
MAX_TRANSPORT_SUBSTEPS = 4096


def edge_flux(mesh, omega):
    vertices = mesh['vertices'][mesh['edge_vertices']]
    tangent = vertices[:, 1]-vertices[:, 0]
    right = np.cross(tangent, mesh['edge_mid'])
    sign = np.where(np.sum(right*mesh['edge_normal'], axis=1) >= 0, 1., -1.)
    radius = float(mesh.get('radius_km', 6371.))
    return (tangent@np.asarray(omega, float))*sign*radius**2


def advect(mesh, values, omega, dt, *, cfl=.45):
    """Transport cell-average scalar columns, preserving their area integrals.

    Upwind flux is intentionally monotone. Ocean transitions may diffuse at
    finite mesh resolution; continental geometry uses separate moving faces.
    A CFL subdivision changes numerical integration only, never saved time.
    """
    field = np.asarray(values, float)
    scalar = field.ndim == 1
    if scalar:
        field = field[:, None]
    area = np.asarray(mesh['area_km2'])
    if (field.ndim != 2 or len(field) != len(area) or not np.isfinite(field).all()
            or not np.isfinite(dt) or dt < 0 or not 0 < cfl < 1):
        raise ValueError('Finite mesh-aligned scalar fields and nonnegative time are required.')
    flux = edge_flux(mesh, omega)
    if not np.isfinite(flux).all() or np.any(area <= 0.) or not np.isfinite(area).all():
        raise ValueError('Transport requires finite fluxes and positive finite cell areas.')
    a, b = np.asarray(mesh['edge_faces']).T
    outgoing = np.bincount(a, weights=np.maximum(flux, 0), minlength=len(area))
    outgoing += np.bincount(b, weights=np.maximum(-flux, 0), minlength=len(area))
    count = max(1, int(np.ceil(dt*np.max(outgoing/area, initial=0)/cfl)))
    if count > MAX_TRANSPORT_SUBSTEPS:
        raise ValueError(f'Transport needs {count} CFL substeps, above the resource limit '
                         f'{MAX_TRANSPORT_SUBSTEPS}; reject the step and inspect plate forces.')
    result = field.copy()
    upwind = np.where(flux >= 0, a, b)
    for _ in range(count):
        amount = result[upwind]*(flux*dt/count)[:, None]
        for column in range(result.shape[1]):
            change = np.bincount(b, weights=amount[:, column], minlength=len(area))
            change -= np.bincount(a, weights=amount[:, column], minlength=len(area))
            result[:, column] += change/area
    return result[:, 0] if scalar else result


def face_vertex_stencil(mesh):
    """Area-weighted face-to-vertex adjacency, padded with zero weights."""
    faces = np.asarray(mesh['faces'])
    vertex = faces.ravel()
    source = np.repeat(np.arange(len(faces)), 3)
    order = np.argsort(vertex, kind='stable')
    counts = np.bincount(vertex, minlength=len(mesh['vertices']))
    width = int(counts.max(initial=0))
    indices = np.zeros((len(counts), width), np.int32)
    weights = np.zeros((len(counts), width))
    starts = np.r_[0, np.cumsum(counts)]
    for v in range(len(counts)):
        selected = source[order[starts[v]:starts[v+1]]]
        indices[v, :len(selected)] = selected
        values = mesh['area_km2'][selected]
        weights[v, :len(selected)] = values/values.sum()
    return indices, weights


def sample_coordinates(mesh, locator, stencil, points):
    """Continuous native reconstruction for nonconservative diagnostic probes.

    Finite-volume advection above never uses this reconstruction. Trench
    gradient queries use neighboring spherical cells instead of raster pixels.
    """
    from mesh_geometry import locate_points
    face, bary = locate_points(np.asarray(points), locator)
    if np.any(face < 0):
        raise ValueError('A closed computational mesh must contain every spherical query.')
    indices, weights = stencil
    vertices = mesh['faces'][face]
    return [(indices[vertices[:, corner], k], bary[:, corner]*weights[vertices[:, corner], k])
            for corner in range(3) for k in range(indices.shape[1])]
