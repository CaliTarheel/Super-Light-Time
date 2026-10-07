"""Exact upper material-region intersections for lower contact-source zones.

An overriding face can retain different phase and heat columns in adjacent
polygons. A lower source zone must not average those columns before pressure
is evaluated. This is geometry and identity only; it assigns no vertical load.
"""

import json

import numpy as np

import mesh_coverage
from channel_region_geometry import (
    _checked_polygon, remove_roundoff_vertices,
    spherical_partition_intersections)


def _checked_upper_record(face_id, triangle, record, radius_km):
    if (int(record['face_id']) != face_id
            or record['history'].get('face_id') != face_id
            or not np.array_equal(record['face_triangle'], triangle)):
        raise ValueError('Upper regional source geometry is stale for its native face.')
    regions = record['history']['regions']
    polygons = record['polygons']
    region_ids = [region['region_id'] for region in regions]
    if (not regions or len(regions) != len(polygons)
            or len(set(region_ids)) != len(region_ids)):
        raise ValueError('Upper regional source polygons lack aligned histories.')
    geometry = spherical_partition_intersections(
        triangle, polygons, polygons, radius_km=radius_km)
    if not np.allclose(
            geometry['old_fractions'], [region['fraction'] for region in regions],
            rtol=0., atol=2e-10):
        raise ValueError('Upper regional source fractions are stale.')
    return [(region['region_id'], polygon) for region, polygon in zip(regions, polygons)]


def refine_by_upper_regions(zones, triangles_by_face_id, records_by_face_id,
                            lower_face_area_km2, *, radius_km=6371.):
    """Split each covered lower zone by every covering upper region polygon.

    Region bindings follow the persistent top-to-bottom face order. An upper
    face without a regional record remains one native face column and has a
    ``None`` region ID. Every refinement must conserve its parent zone area.
    """
    if not isinstance(records_by_face_id, dict):
        raise ValueError('Upper regional sources need a face-ID record map.')
    checked = {}
    refined = []
    for zone in zones:
        pieces = [dict(zone, upper_region_bindings_top_to_bottom=())]
        for face_id in zone['covering_upper_face_ids_top_to_bottom']:
            face_id = int(face_id)
            record = records_by_face_id.get(face_id)
            if record is None:
                pieces = [dict(piece, upper_region_bindings_top_to_bottom=
                               piece['upper_region_bindings_top_to_bottom']
                               + ((face_id, None),)) for piece in pieces]
                continue
            if face_id not in checked:
                checked[face_id] = _checked_upper_record(
                    face_id, triangles_by_face_id[face_id], record, radius_km)
            next_pieces = []
            for piece in pieces:
                polygon, _ = _checked_polygon(
                    piece['polygon'], 'Lower contact zone', radius_km)
                parent_area = mesh_coverage._polygon_area(polygon, radius_km)
                children = []
                for region_id, upper_polygon in checked[face_id]:
                    clipped = remove_roundoff_vertices(mesh_coverage._clip_planes(
                        polygon, mesh_coverage._triangle_planes(upper_polygon)))
                    if len(clipped) < 3:
                        continue
                    area = mesh_coverage._polygon_area(clipped, radius_km)
                    if area <= 0.:
                        continue
                    children.append((region_id, clipped, area))
                if not np.isclose(sum(area for _, _, area in children), parent_area,
                                  rtol=0., atol=2e-10 * parent_area):
                    raise ValueError('Upper regional polygons do not cover their lower contact zone.')
                for region_id, clipped, area in children:
                    next_pieces.append(dict(
                        piece, polygon=clipped,
                        zone_id=json.dumps([piece['zone_id'], face_id, region_id],
                                           separators=(',', ':')),
                        area_km2=float(area),
                        fraction=float(area / lower_face_area_km2),
                        upper_region_bindings_top_to_bottom=
                            piece['upper_region_bindings_top_to_bottom']
                            + ((face_id, region_id),)))
            pieces = next_pieces
        refined.extend(pieces)
    if not np.isclose(sum(piece['area_km2'] for piece in refined),
                      lower_face_area_km2, rtol=0.,
                      atol=2e-10 * lower_face_area_km2):
        raise ValueError('Upper regional refinement changed lower material area.')
    return refined
