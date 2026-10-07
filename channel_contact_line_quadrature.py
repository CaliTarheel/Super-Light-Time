"""Along-trench measure for a nonlocal strip on a spherical contact zone.

The two-plate flexure benchmark returns joules per metre of trench. Its strips
must be integrated over hinge arclength, not multiplied by contact area. The
spherical coarea Jacobian supplies an independent area closure for the sampled
zone cross sections: dA = R cos(theta) dtheta * R d(alpha).
"""

import numpy as np

import mesh_coverage
from channel_region_geometry import _checked_polygon


def _cross_section(hinge, normal, planes):
    lower, upper = 0., .5 * np.pi
    for plane in planes:
        at_hinge = float(plane @ hinge)
        at_pole = float(plane @ normal)
        if at_hinge < 0. and at_pole < 0.:
            raise ValueError('Contact zone misses an along-trench quadrature ray.')
        if at_hinge < 0. <= at_pole:
            lower = max(lower, float(np.arctan2(-at_hinge, at_pole)))
        elif at_pole < 0. <= at_hinge:
            upper = min(upper, float(np.arctan2(at_hinge, -at_pole)))
    if not lower < upper:
        raise ValueError('Contact zone has no positive entry cross section.')
    return lower, upper


def spherical_contact_line_quadrature(polygon, hinge_normal, *,
                                      radius_km=6371., area_tolerance=1e-4,
                                      max_panels=4096):
    """Return hinge-length weights and one interior contact point per strip.

    Midpoint panels double until the coarea estimate matches the independent
    native spherical polygon area within the requested relative tolerance.
    This tests geometric coverage, not convergence of the future strip energy.
    """
    normal = np.asarray(hinge_normal, float)
    radius = float(radius_km) * 1000.
    tolerance = float(area_tolerance)
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or not np.isclose(np.linalg.norm(normal), 1., rtol=0., atol=3e-12)
            or not np.isfinite(radius) or radius <= 0.
            or not np.isfinite(tolerance) or not 0. < tolerance < 1.
            or type(max_panels) is not int or max_panels < 4):
        raise ValueError('Contact line quadrature needs a unit hinge and bounded area tolerance.')
    points, planes = _checked_polygon(polygon, 'Contact zone', radius_km)
    entry = points @ normal
    projections = points - entry[:, None] * normal
    lengths = np.linalg.norm(projections, axis=1)
    if (np.any(entry < -1e-11) or np.any(entry >= 1. - 1e-10)
            or np.any(lengths <= 1e-8)):
        raise ValueError('Contact line quadrature needs one positive entry hemisphere.')
    origin = projections.sum(axis=0)
    magnitude = float(np.linalg.norm(origin))
    if magnitude <= 1e-8:
        raise ValueError('Contact line quadrature has no unique along-trench origin.')
    origin /= magnitude
    tangent = np.cross(normal, origin)
    angles = np.arctan2(points @ tangent, points @ origin)
    start, stop = float(angles.min()), float(angles.max())
    if not 0. < stop - start < np.pi:
        raise ValueError('Contact line quadrature needs a local convex hinge span.')
    true_area = float(mesh_coverage._polygon_area(points, radius_km) * 1e6)
    panels = 4
    while panels <= max_panels:
        step = (stop - start) / panels
        mid = start + (np.arange(panels) + .5) * step
        hinge = np.cos(mid)[:, None] * origin + np.sin(mid)[:, None] * tangent
        bounds = np.array([_cross_section(ray, normal, planes) for ray in hinge])
        lo, hi = bounds.T
        contact = (np.cos(.5 * (lo + hi))[:, None] * hinge
                   + np.sin(.5 * (lo + hi))[:, None] * normal)
        if np.any(contact @ planes.T < -1e-10):
            raise ValueError('Contact line quadrature site left its source zone.')
        line_weights = np.full(panels, radius * step)
        represented = radius * line_weights * (np.sin(hi) - np.sin(lo))
        estimate = float(represented.sum())
        if abs(estimate - true_area) <= tolerance * true_area:
            return dict(contact_xyz=contact, hinge_xyz=hinge,
                        line_weight_m=line_weights,
                        cross_section_offset_bounds_m=bounds * radius,
                        represented_area_m2=represented,
                        polygon_area_m2=true_area,
                        relative_area_residual=(estimate - true_area) / true_area,
                        panels=panels,
                        scope='read-only along-trench strip measure; no flexure work or source commit')
        panels *= 2
    raise ValueError('Contact line quadrature did not close spherical area within its panel budget.')
