"""Exact native material/contact-zone intersections before local sources."""

import numpy as np

import channel_region_history
import mesh_coverage
from channel_region_geometry import remove_roundoff_vertices


def prepare_zone_remap(s, zones_by_face):
    """Stage new contact zones without mixing old material phase histories.

    The caller supplies a complete, co-registered spherical polygon partition
    for each tracked face. Each positive old-region/new-zone intersection keeps
    its old column and a new stable compound ID. This does not infer the zones
    from contact geometry or apply their pressure/thermal source.
    """
    from channel_region_native import (
        VERSION, enabled, _markers, _check_trace_alignment,
        _reference_mass, _project)
    if not enabled(s) or not isinstance(zones_by_face, dict):
        raise ValueError('Native source-zone remap needs an installed store and face zones.')
    store = s.channel_region_store
    surface = s.material_surface
    if (store.get('version') != VERSION or store.get('epoch_myr') != float(s.t)
            or store.get('geometry_revision') != int(surface['geometry_revision'])):
        raise ValueError('Native source-zone remap needs the current material epoch.')
    records = store['records']
    face_ids = {int(record['face_id']) for record in records}
    if set(zones_by_face) != face_ids:
        raise ValueError('Native source-zone remap needs zones for every tracked face.')
    old_bound = _markers(s, sorted(face_ids))
    _check_trace_alignment(s, old_bound)
    by_id = {int(face_id): index for index, face_id in enumerate(surface['face_id'])}
    radius = float(surface.get('radius_km', 6371.))
    next_records = []
    fields = set(s.structure)
    for record in records:
        face_id = int(record['face_id'])
        index = by_id[face_id]
        triangle = surface['vertices'][surface['faces'][index]]
        if (record['history'].get('face_index') != index
                or not np.array_equal(record['face_triangle'], triangle)
                or not np.isclose(_reference_mass(record, radius), s.mass[index],
                                  rtol=2e-10, atol=1e-6)):
            raise ValueError('Native source-zone history is stale for its material face.')
        zones = zones_by_face[face_id]
        report = channel_region_history.remap_history_from_polygons(
            record['history'], triangle, record['polygons'], zones,
            radius_km=radius)
        pieces = []
        for old_index, polygon in enumerate(record['polygons']):
            for zone_index, zone in enumerate(zones):
                if report['geometry']['overlap_fractions'][old_index, zone_index] <= 0.:
                    continue
                piece = remove_roundoff_vertices(mesh_coverage._clip_planes(
                    np.asarray(polygon),
                    mesh_coverage._triangle_planes(np.asarray(zone['polygon']))))
                if mesh_coverage._polygon_area(piece, radius) <= 0.:
                    raise ValueError('Native source-zone intersection lost positive material area.')
                pieces.append(piece)
        next_record = dict(face_id=face_id, face_triangle=triangle.copy(),
                           history=report['history'], polygons=pieces)
        if (len(pieces) != len(report['history']['regions'])
                or not np.isclose(_reference_mass(next_record, radius), s.mass[index],
                                  rtol=2e-10, atol=1e-6)):
            raise ValueError('Native source-zone remap changed material reference area.')
        mean = _project(next_record, fields)
        for name in fields:
            if not np.isclose(mean[name][0], s.structure[name][index],
                              rtol=2e-10, atol=1e-8):
                raise ValueError('Native source-zone remap changed a face column: ' + name)
        next_records.append(next_record)
    bound = _markers(s, sorted(face_ids), records=next_records)
    _check_trace_alignment(s, bound)
    if {row['trace_id'] for row in bound} != {row['trace_id'] for row in old_bound}:
        raise ValueError('Native source-zone remap lost a material trace marker.')
    return dict(store=dict(version=VERSION, epoch_myr=float(s.t),
                           geometry_revision=int(surface['geometry_revision']),
                           records=next_records),
                remapped_markers=len(bound),
                scope='read-only exact material/contact-zone intersections; no source commit')
