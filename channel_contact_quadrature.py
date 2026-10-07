"""Read-only area-conserving quadrature sites on a spherical contact zone."""

import numpy as np

import mesh_coverage
from channel_region_geometry import _checked_polygon


def _unit(point):
    return point / np.linalg.norm(point)


def _split(triangle):
    a, b, c = triangle
    ab, bc, ca = _unit(a + b), _unit(b + c), _unit(c + a)
    return (np.array((a, ab, ca)), np.array((ab, b, bc)),
            np.array((ca, bc, c)), np.array((ab, bc, ca)))


def spherical_contact_quadrature(polygon, *, radius_km=6371.,
                                 refinement_level=0, max_nodes=4096):
    """Return spherical triangle centroids with exact physical-area weights.

    A convex zone is fanned from its first vertex and optionally subdivided
    along great-circle edges. The weights sum to its native spherical area;
    pointwise energy/force convergence still requires a separate refinement
    check, especially across contact/free or phase boundaries.
    """
    radius = float(radius_km)
    if (not np.isfinite(radius) or radius <= 0.
            or type(refinement_level) is not int or not 0 <= refinement_level <= 10
            or type(max_nodes) is not int or max_nodes < 1):
        raise ValueError('Contact quadrature needs positive radius and bounded refinement.')
    points, _ = _checked_polygon(polygon, 'Contact zone', radius)
    count = (len(points) - 2) * 4**refinement_level
    if count > max_nodes:
        raise ValueError('Contact quadrature exceeds its node budget.')
    triangles = [np.array((points[0], points[index], points[index + 1]))
                 for index in range(1, len(points) - 1)]
    for _ in range(refinement_level):
        triangles = [child for triangle in triangles for child in _split(triangle)]
    areas = np.asarray([mesh_coverage._polygon_area(triangle, radius)
                        for triangle in triangles], float)
    area = float(mesh_coverage._polygon_area(points, radius))
    if (not np.isfinite(areas).all() or np.any(areas <= 0.)
            or not np.isclose(float(areas.sum()), area, rtol=2e-10, atol=1e-11)):
        raise ValueError('Contact quadrature did not conserve spherical area.')
    sites = np.asarray([_unit(triangle.sum(axis=0)) for triangle in triangles])
    return dict(xyz=sites, area_m2=areas * 1e6,
                polygon_area_m2=area * 1e6, refinement_level=refinement_level,
                scope='read-only spherical-area sites; no force or source integration')
