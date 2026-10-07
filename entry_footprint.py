"""Read-only horizontal footprint of conserved along-slab entry material.

An admitted material strip keeps its reference area and down-dip length. For
the existing constant dip, its distance ``q`` past a great-circle hinge is an
along-slab conveyor coordinate: depth is ``q sin(dip)``, while horizontal
distance from the hinge is ``q cos(dip)``. This module maps points and local
area measure only; it does not change native face geometry or collision work.
"""

import math
import numpy as np


def _coordinates(points, hinge_normal, dip_degrees):
    points = np.asarray(points, float)
    normal = np.asarray(hinge_normal, float)
    dip = float(dip_degrees)
    if (points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all()
            or np.any(np.abs(np.linalg.norm(points, axis=-1) - 1.) > 1e-10)
            or normal.shape != (3,) or not np.isfinite(normal).all()
            or abs(float(np.linalg.norm(normal)) - 1.) > 1e-10
            or not math.isfinite(dip) or not 0. < dip < 90.):
        raise ValueError('Conveyor footprint needs unit points, a unit hinge and dip in (0,90).')
    sine = np.clip(points @ normal, -1., 1.)
    cosine = np.sqrt(np.maximum(0., 1. - sine*sine))
    if np.any(cosine <= 1e-10):
        raise ValueError('Conveyor footprint is singular at a hinge-coordinate pole.')
    return points, normal, np.arcsin(sine), cosine, math.cos(math.radians(dip))


def project_points(points, hinge_normal, dip_degrees):
    """Map admitted material to its constant-dip horizontal surface footprint.

    Non-entered points stay put. Hinge-parallel angle is unchanged; positive
    hinge-normal angle is shortened by cos(dip). The map of a geodesic face
    edge is generally curved, so mapping just its three vertices does not
    define an exact projected contact polygon.
    """
    points, normal, angle, cosine, factor = _coordinates(points, hinge_normal, dip_degrees)
    hinge = (points - np.expand_dims(np.sin(angle), -1)*normal) / np.expand_dims(cosine, -1)
    projected = (hinge*np.expand_dims(np.cos(angle*factor), -1)
                 + normal*np.expand_dims(np.sin(angle*factor), -1))
    return np.where(np.expand_dims(angle > 0., -1), projected, points.copy())


def unproject_points(points, hinge_normal, dip_degrees):
    """Recover conserved material coordinates from a projected contact point."""
    points, normal, angle, cosine, factor = _coordinates(points, hinge_normal, dip_degrees)
    if np.any((angle > 0.) & (angle >= factor*math.pi/2. - 1e-10)):
        raise ValueError('Projected contact leaves the invertible conveyor domain.')
    hinge = (points - np.expand_dims(np.sin(angle), -1)*normal) / np.expand_dims(cosine, -1)
    original_angle = angle/factor
    unprojected = (hinge*np.expand_dims(np.cos(original_angle), -1)
                   + normal*np.expand_dims(np.sin(original_angle), -1))
    return np.where(np.expand_dims(angle > 0., -1), unprojected, points.copy())


def projected_area_ratio(points, hinge_normal, dip_degrees):
    """Horizontal footprint area divided by represented spherical material area.

    In signed hinge latitude ``a`` and along-hinge angle ``phi``, material
    area is R² cos(a) da dphi. The projected area replaces ``a`` with
    ``a cos(dip)``, yielding the exact local Jacobian below for a>0.
    """
    _, _, angle, cosine, factor = _coordinates(points, hinge_normal, dip_degrees)
    return np.where(angle > 0., factor*np.cos(factor*angle)/cosine, 1.)


def reference_to_footprint_ratio(triangle, barycentric, hinge_normal, dip_degrees,
                                 *, radius_km=6371.):
    """Footprint area per material-reference area at one barycentric site."""
    from entry_overlap_geometry import projective_area_ratio

    triangle = np.asarray(triangle, float)
    barycentric = np.asarray(barycentric, float)
    material_ratio = projective_area_ratio(triangle, barycentric, radius_km=radius_km)
    direction = barycentric @ triangle
    point = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
    return material_ratio*projected_area_ratio(point, hinge_normal, dip_degrees)


def depth_m(points, hinge_normal, dip_degrees, *, radius_km=6371.):
    """Depth of the same conserved material coordinate used by the footprint."""
    _, _, angle, _, _ = _coordinates(points, hinge_normal, dip_degrees)
    if not np.isfinite(radius_km) or radius_km <= 0.:
        raise ValueError('Conveyor depth needs a positive finite radius.')
    return np.maximum(angle, 0.)*float(radius_km)*1000.*math.sin(math.radians(dip_degrees))
