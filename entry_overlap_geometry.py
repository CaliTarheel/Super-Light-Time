"""Shared spherical geometry for entered continental material/stack pairs.

This module decides only where material triangles intersect after positive
hinge-depth and optional finite-trench clipping. Vertical order, contact,
energy, force and source accounting belong to their respective callers.
"""

import numpy as np

import finite_entry_arc
import mesh_coverage


def projective_area_ratio(triangle, barycentric, *, radius_km=6371.):
    """Physical spherical area per unit material-reference area.

    The material face maps affine barycentric weights `w` to the unit sphere
    by normalizing `r = w @ triangle`. Its physical Jacobian is
    `R² det(v0,v1,v2) / |r|³` per barycentric-coordinate area. The model's
    existing material-reference measure is `2 A_face du dv`, where
    `A_face` is the exact whole spherical face area. Their ratio is one only
    in the infinitesimal planar limit; its reference-area mean is exactly one.
    """
    triangle = np.asarray(triangle, float)
    barycentric = np.asarray(barycentric, float)
    radius_km = float(radius_km)
    if (triangle.shape != (3, 3) or not np.isfinite(triangle).all()
            or np.any(np.abs(np.linalg.norm(triangle, axis=1) - 1.) > 1e-10)
            or barycentric.ndim < 1 or barycentric.shape[-1] != 3
            or not np.isfinite(barycentric).all()
            or np.any(np.abs(barycentric.sum(axis=-1) - 1.) > 1e-10)
            or np.any(barycentric < -2e-10)
            or not np.isfinite(radius_km) or radius_km <= 0.):
        raise ValueError('Projective area ratio needs a unit face and material barycentric points.')
    determinant = float(np.dot(
        triangle[0], np.cross(triangle[1] - triangle[0],
                              triangle[2] - triangle[0])))
    area_km2 = mesh_coverage._polygon_area(triangle, radius_km)
    if determinant <= 0. or area_km2 <= 0.:
        raise ValueError('Projective area ratio needs an outward nondegenerate face.')
    directions = barycentric @ triangle
    norm = np.linalg.norm(directions, axis=-1)
    ratio = radius_km**2 * determinant / (2. * area_km2 * norm**3)
    return float(ratio) if ratio.ndim == 0 else ratio


def spherical_area_and_field_gradient(triangle, fields, *, radius_km=6371.):
    """Differentiate a clipped spherical footprint in affine nodal fields.

    The lower material triangle is held fixed while differentiating its
    clipped barycentric polygon. Under a rigid rotation of that whole lower
    face, its explicit geometric rotation does zero area work; callers still
    must chain the field derivatives through lower and upper plate motion.
    At a clipping topology change the derivative is only a one-sided local
    value, not a smooth force law.
    """
    triangle = np.asarray(triangle, float)
    fields = np.asarray(fields, float)
    radius_km = float(radius_km)
    if (triangle.shape != (3, 3) or not np.isfinite(triangle).all()
            or np.any(np.abs(np.linalg.norm(triangle, axis=1) - 1.) > 1e-10)
            or not np.isfinite(radius_km) or radius_km <= 0.):
        raise ValueError('Spherical clipped area needs a unit lower face and positive radius.')
    determinant = float(np.dot(
        triangle[0], np.cross(triangle[1] - triangle[0],
                              triangle[2] - triangle[0])))
    if determinant <= 0.:
        raise ValueError('Spherical clipped area needs an outward nondegenerate face.')
    polygon = finite_entry_arc.clip_fields(fields)
    width = 3 * len(fields)
    gradient = np.zeros(width)
    if len(polygon) < 3:
        return 0., gradient.reshape(fields.shape), polygon

    vertices = []
    derivatives = []
    for barycentric, jacobian in polygon:
        direction = barycentric @ triangle
        derivative = triangle.T @ jacobian
        length = np.linalg.norm(direction)
        point = direction / length
        derivative = (derivative - point[:, None]
                      * (point @ derivative)[None, :]) / length
        vertices.append(point)
        derivatives.append(derivative)
    for second in range(1, len(vertices) - 1):
        a, b, c = (vertices[index] for index in (0, second, second + 1))
        da, db, dc = (derivatives[index]
                      for index in (0, second, second + 1))
        u, v = b - a, c - a
        du, dv = db - da, dc - da
        cross = np.cross(u, v)
        cross_gradient = (np.cross(du.T, v).T
                          + np.cross(u, dv.T).T)
        determinant = float(a @ cross)
        determinant_gradient = (da.T @ cross + a @ cross_gradient)
        denominator = 1. + a @ b + b @ c + c @ a
        denominator_gradient = (da.T @ (b + c) + db.T @ (a + c)
                                + dc.T @ (a + b))
        numerator = abs(determinant)
        numerator_gradient = np.sign(determinant) * determinant_gradient
        squared = denominator**2 + numerator**2
        if squared <= 0.:
            raise ValueError('Spherical clipped area has an invalid solid angle.')
        gradient += (2. * radius_km**2
                     * (denominator * numerator_gradient
                        - numerator * denominator_gradient) / squared)
    points = np.asarray(vertices)
    area = mesh_coverage._polygon_area(points, radius_km)
    if not np.isfinite(gradient).all():
        raise ValueError('Spherical clipped area derivative is not finite.')
    return area, gradient.reshape(fields.shape), polygon


def clipped_entry_stack_intersections(points, faces, sheets, indices, normals,
                                      entry_fractions, *, radius_km,
                                      finite_midpoints=None,
                                      finite_half_lengths_km=None,
                                      include_polygon=False):
    """Yield local entered/stack face intersections and their spherical area.

    A stack elsewhere on that face is not an entry/stack overlap. The clipping
    uses the same reference-linear positive coordinate and finite endpoints
    as the energy, projected radially onto the spherical material triangle.
    A pair is reported only above the existing 1e-8 km2 geometry tolerance.
    """
    selected = {int(indices[local]): int(local)
                for local in np.flatnonzero(entry_fractions > 0.)}
    if not selected:
        return
    vertices = np.asarray(points, float)
    triangles = vertices[np.asarray(faces)]
    overlap = mesh_coverage.material_overlaps(
        vertices, faces, sheets, radius_km=radius_km)
    pieces = {}
    for a, b in zip(overlap['first'], overlap['second']):
        for entered, stack in ((int(a), int(b)), (int(b), int(a))):
            local = selected.get(entered)
            if local is None:
                continue
            if entered not in pieces:
                triangle = triangles[entered]
                q = radius_km * 1000. * np.arcsin(
                    np.clip(triangle @ normals[local], -1., 1.))
                left = right = np.ones(3)
                if (finite_half_lengths_km is not None and
                        np.isfinite(finite_half_lengths_km[local])):
                    l, r, _, _ = finite_entry_arc.endpoint_planes(
                        normals[local], finite_midpoints[local],
                        finite_half_lengths_km[local], radius_km)
                    left, right = triangle @ l, triangle @ r
                _, _, _, _, _, parts = finite_entry_arc.integrate(q, left, right)
                pieces[entered] = []
                for _, bary in parts:
                    geometry = bary @ triangle
                    geometry /= np.linalg.norm(geometry, axis=1)[:, None]
                    pieces[entered].append(geometry)
            for piece in pieces[entered]:
                intersection = mesh_coverage.clip_triangle(
                    piece, triangles[stack])
                area = mesh_coverage._polygon_area(intersection, radius_km)
                if area > 1e-8:
                    row = dict(entered_face=entered, stack_face=stack,
                               area_km2=float(area))
                    if include_polygon:
                        row['polygon'] = intersection
                    yield row
