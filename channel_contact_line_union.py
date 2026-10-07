"""One hinge-length measure for disjoint zones crossed by the same strip.

Several contact polygons can occupy one along-trench coordinate. Each normal
strip bends once, with a union of its disjoint contact intervals. Integrating
a complete strip separately for each polygon would duplicate bending work.
"""

import numpy as np

import mesh_coverage
from channel_contact_line_quadrature import _cross_section
from channel_region_geometry import _checked_polygon


def spherical_contact_line_union(zones, hinge_normal, *, radius_km=6371.,
                                 area_tolerance=1e-4, max_panels=4096):
    """Return unique hinge rays and their zone-attributed normal intervals."""
    normal = np.asarray(hinge_normal, float)
    radius = float(radius_km) * 1000.
    tolerance = float(area_tolerance)
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or not np.isclose(np.linalg.norm(normal), 1., rtol=0., atol=3e-12)
            or not np.isfinite(radius) or radius <= 0.
            or not np.isfinite(tolerance) or not 0. < tolerance < 1.
            or type(max_panels) is not int or max_panels < 4
            or not isinstance(zones, (list, tuple)) or not zones):
        raise ValueError('Contact line union needs zones, a unit hinge and bounded tolerance.')
    prepared = []
    for zone in zones:
        if (not isinstance(zone, dict) or set(zone) != {'zone_id', 'polygon'}
                or not isinstance(zone['zone_id'], str) or not zone['zone_id']):
            raise ValueError('Contact line union needs named convex zone polygons.')
        points, planes = _checked_polygon(zone['polygon'], 'Contact zone', radius_km)
        entry = points @ normal
        projected = points - entry[:, None] * normal
        if (np.any(entry < -1e-11) or np.any(entry >= 1. - 1e-10)
                or np.any(np.linalg.norm(projected, axis=1) <= 1e-8)):
            raise ValueError('Contact line union needs one positive entry hemisphere.')
        prepared.append(dict(zone_id=zone['zone_id'], points=points,
                             planes=planes, projected=projected,
                             area_m2=float(mesh_coverage._polygon_area(
                                 points, radius_km) * 1e6)))
    if len({row['zone_id'] for row in prepared}) != len(prepared):
        raise ValueError('Contact line union needs distinct zone IDs.')
    true_area = sum(row['area_m2'] for row in prepared)
    for index, first in enumerate(prepared):
        for second in prepared[index + 1:]:
            overlap = mesh_coverage._clip_planes(first['points'], second['planes'])
            if mesh_coverage._polygon_area(overlap, radius_km) * 1e6 > 2e-10 * true_area:
                raise ValueError('Contact line union zones overlap in physical area.')
    origin = np.sum([row['projected'].sum(axis=0) for row in prepared], axis=0)
    magnitude = float(np.linalg.norm(origin))
    if magnitude <= 1e-8:
        raise ValueError('Contact line union has no unique along-trench origin.')
    origin /= magnitude
    tangent = np.cross(normal, origin)
    knots = []
    for row in prepared:
        angles = np.arctan2(row['points'] @ tangent, row['points'] @ origin)
        row['start'], row['stop'] = float(angles.min()), float(angles.max())
        if not 0. < row['stop'] - row['start'] < np.pi:
            raise ValueError('Contact line union needs local convex hinge spans.')
        knots.extend((row['start'], row['stop']))
    knots = np.unique(knots)
    if knots[-1] - knots[0] >= np.pi:
        raise ValueError('Contact line union exceeds one unambiguous hinge hemisphere.')
    segments = [(float(a), float(b)) for a, b in zip(knots[:-1], knots[1:])
                if b - a > 1e-12 and any(
                    row['start'] < .5 * (a + b) < row['stop'] for row in prepared)]
    if not segments:
        raise ValueError('Contact line union has no positive hinge span.')
    per_segment = 4
    while per_segment * len(segments) <= max_panels:
        rays, weights, intervals, areas = [], [], [], []
        for start, stop in segments:
            step = (stop - start) / per_segment
            for alpha in start + (np.arange(per_segment) + .5) * step:
                hinge = np.cos(alpha) * origin + np.sin(alpha) * tangent
                weight = radius * step
                occupied = []
                for row in prepared:
                    if not row['start'] < alpha < row['stop']:
                        continue
                    lo, hi = _cross_section(hinge, normal, row['planes'])
                    contact = (np.cos(.5 * (lo + hi)) * hinge
                               + np.sin(.5 * (lo + hi)) * normal)
                    if np.any(row['planes'] @ contact < -1e-10):
                        raise ValueError('Contact line union site left its source zone.')
                    occupied.append(dict(zone_id=row['zone_id'],
                                         lower_offset_m=radius * lo,
                                         upper_offset_m=radius * hi,
                                         contact_xyz=contact,
                                         represented_area_m2=radius * weight
                                             * (np.sin(hi) - np.sin(lo))))
                occupied.sort(key=lambda interval: interval['lower_offset_m'])
                if any(left['upper_offset_m'] > right['lower_offset_m'] + 1e-5
                       for left, right in zip(occupied, occupied[1:])):
                    raise ValueError('Contact line union has overlapping normal intervals.')
                rays.append(hinge)
                weights.append(weight)
                intervals.append(occupied)
                areas.append(sum(row['represented_area_m2'] for row in occupied))
        estimated = float(sum(areas))
        if abs(estimated - true_area) <= tolerance * true_area:
            return dict(hinge_xyz=np.asarray(rays),
                        radius_m=radius,
                        line_weight_m=np.asarray(weights),
                        contact_intervals=intervals,
                        represented_area_m2=np.asarray(areas),
                        polygon_area_m2=true_area,
                        relative_area_residual=(estimated - true_area) / true_area,
                        panels=len(rays),
                        scope='read-only union of contact intervals; one nonlocal strip per hinge ray')
        per_segment *= 2
    raise ValueError('Contact line union did not close spherical area within its panel budget.')
