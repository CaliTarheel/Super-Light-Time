"""Conservative read-only subdivision of material-attached channel histories.

The caller supplies a genuine partition of one old spherical material face by
new child triangles and their persistent IDs. This module does not create a
native remesh, change the source law, or combine histories during coarsening.
"""

import json

import numpy as np

import dense_crust
import mesh_coverage
from channel_upper_basal_reference import inherit_reference
from channel_region_geometry import (
    spherical_partition_intersections, remove_roundoff_vertices)


def split_history_to_children(record, children, *, radius_km=6371.):
    """Split every old regional phase column into its child-face intersections.

    Children cover exactly the old face in its current material frame. Each
    positive cell retains its parent's complete column state, including heat
    and phase inventories; only its physical-area fraction and material face
    identity change. A native caller must separately provide a valid remesh
    map and project child means into committed face and marker state.
    """
    if (not isinstance(record, dict) or set(record) != {
            'face_id', 'face_triangle', 'history', 'polygons'}
            or not isinstance(children, (list, tuple)) or not children):
        raise ValueError('Region refinement needs a saved face record and child triangles.')
    old_id = record['face_id']
    history = record['history']
    polygons = record['polygons']
    if (not isinstance(old_id, (int, np.integer)) or old_id < 0
            or not isinstance(history, dict) or history.get('face_id') != int(old_id)
            or not isinstance(history.get('regions'), list)
            or not isinstance(polygons, (list, tuple))
            or len(polygons) != len(history['regions'])):
        raise ValueError('Region refinement needs aligned persistent old-face histories.')
    if any(not isinstance(child, dict) or set(child) != {
            'face_id', 'face_index', 'triangle'}
           or not isinstance(child['face_id'], (int, np.integer))
           or not isinstance(child['face_index'], (int, np.integer))
           or child['face_id'] < 0 or child['face_index'] < 0
           for child in children):
        raise ValueError('Every child needs a persistent face ID, index and triangle.')
    child_ids = [int(child['face_id']) for child in children]
    child_indices = [int(child['face_index']) for child in children]
    if len(set(child_ids)) != len(children) or len(set(child_indices)) != len(children):
        raise ValueError('Child material face IDs and indices must be unique.')
    geometry = spherical_partition_intersections(
        record['face_triangle'], polygons, [child['triangle'] for child in children],
        radius_km=radius_km)
    old_regions = history['regions']
    saved_fractions = np.asarray([region['fraction'] for region in old_regions], float)
    if not np.allclose(saved_fractions, geometry['old_fractions'], rtol=0., atol=2e-10):
        raise ValueError('Saved regional area fractions disagree with their material polygons.')
    for region in old_regions:
        if (not isinstance(region, dict)
                or not isinstance(region.get('region_id'), str)
                or not region['region_id'] or 'column' not in region):
            raise ValueError('Source regions need stable identities and retained columns.')
        dense_crust.validate(region['column'])
        if len(region['column']['thickness_km']) != 1:
            raise ValueError('Each old source region needs one column.')
    if len({region['region_id'] for region in old_regions}) != len(old_regions):
        raise ValueError('Old source region IDs must be unique.')

    face_area = geometry['face_area_km2']
    old_crust = sum(_inventory(region['column'], dense_crust.mass_volume)
                    * face_area * fraction
                    for region, fraction in zip(old_regions, saved_fractions))
    old_heat = sum(_inventory(region['column'], lambda column: column[dense_crust.HEAT])
                   * face_area * fraction
                   for region, fraction in zip(old_regions, saved_fractions))
    new_crust = 0.
    new_heat = 0.
    roundoff_area = 0.
    output = []
    for j, child in enumerate(children):
        triangle = np.asarray(child['triangle'], float)
        planes = mesh_coverage._triangle_planes(triangle)
        child_area = face_area * geometry['new_fractions'][j]
        regions = []
        pieces = []
        for i, parent in enumerate(old_regions):
            piece = remove_roundoff_vertices(mesh_coverage._clip_planes(
                np.asarray(polygons[i], float), planes))
            area = mesh_coverage._polygon_area(piece, radius_km)
            if 0. < area <= face_area * 1e-14:
                roundoff_area += area
                continue
            if area <= 0.:
                continue
            column = {key: value.copy() for key, value in parent['column'].items()}
            region_id = (parent['region_id'] if len(children) == 1
                         and int(child['face_id']) == int(old_id)
                         else json.dumps([parent['region_id'], int(child['face_id'])],
                                         separators=(',', ':')))
            regions.append(dict(region_id=region_id,
                                zone_id=parent.get('zone_id', parent['region_id']),
                                fraction=float(area / child_area), column=column,
                                **inherit_reference(parent)))
            pieces.append(piece)
            new_crust += area * _inventory(column, dense_crust.mass_volume)
            new_heat += area * _inventory(column, lambda state: state[dense_crust.HEAT])
        if (not regions or abs(sum(region['fraction'] for region in regions) - 1.) > 2e-10):
            raise ValueError('Child material face lost a regional source area.')
        output.append(dict(face_id=int(child['face_id']), face_triangle=triangle.copy(),
                           history=dict(face_id=int(child['face_id']),
                                        face_index=int(child['face_index']), regions=regions,
                                        scope='read-only exact material refinement; no native commit'),
                           polygons=pieces))
    if (roundoff_area > face_area * 1e-12
            or not np.isclose(old_crust, new_crust, rtol=2e-10, atol=1e-8)
            or not np.isclose(old_heat, new_heat, rtol=2e-10, atol=1e-5)):
        raise ValueError('Regional refinement changed the absolute crust or heat inventory.')
    return dict(records=output, old_crust_volume_km3=float(old_crust),
                new_crust_volume_km3=float(new_crust),
                old_stored_heat_km3_c=float(old_heat),
                new_stored_heat_km3_c=float(new_heat),
                discarded_roundoff_area_km2=float(roundoff_area),
                scope='read-only source-region refinement; no native face or marker commit')


def _inventory(column, field):
    return float(np.asarray(field(column))[0] / np.asarray(column['area_factor'])[0])
