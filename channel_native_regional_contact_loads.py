"""Read-only phase-aware native loads for projected contact cells.

Every lower material cell is intersected with the native regional phase
partition before mass and buoyancy are calculated. The result preserves each
history rather than averaging dense crust across a mixed face.
"""

import math

import numpy as np

import entry_contact_channel
import mesh_coverage
from channel_native_lower_regions import bind_lower_regions


def bind_regional_projected_loads(s, lower_face_id, projected_pair, *,
                                  area_mismatch_tolerance=1e-3):
    """Return force and physical pressure on each projected contact cell.

    The affine material-area ledger in `projected_pair` must agree closely
    with its spherical source preimage. The native phase inventory is measured
    in that spherical source area; the mismatch remains an explicit diagnostic
    until the native entry/source law uses one common area measure.
    """
    bound = bind_lower_regions(s, lower_face_id)
    tolerance = float(area_mismatch_tolerance)
    if (not isinstance(projected_pair, dict)
            or projected_pair.get('coordinate_discretization') != 'affine_nodal'
            or not isinstance(projected_pair.get('contact_cells'), list)
            or not projected_pair['contact_cells']
            or not np.isfinite(tolerance) or not 0. < tolerance < 1.):
        raise ValueError('Regional contact loads need occupied affine-projected cells.')
    face_area = bound['physical_area_km2']
    radius = bound['radius_km']
    if not math.isclose(float(projected_pair['face_reference_area_km2']),
                        face_area, rel_tol=2e-10):
        raise ValueError('Regional contact source face area disagrees with native material.')
    triangle = bound['lower_triangle']
    face_planes = mesh_coverage._triangle_planes(triangle)
    regions = bound['regions']
    region_planes = {row['region_id']: mesh_coverage._triangle_planes(row['polygon'])
                     for row in regions}
    represented = {row['region_id']: 0. for row in regions}
    cell_loads = []
    max_mismatch = 0.
    max_outside = 0.
    spherical_total = affine_total = physical_total = 0.
    for index, cell in enumerate(projected_pair['contact_cells']):
        if not isinstance(cell, dict):
            raise ValueError('Regional contact needs complete projected cell records.')
        material = np.asarray(cell['material_polygon'], float)
        bary = np.asarray(cell['material_barycentric_polygon'], float)
        physical = np.asarray(cell['physical_polygon'], float)
        affine_area = float(cell['material_reference_area_km2'])
        physical_area = float(cell['physical_area_km2'])
        if (material.ndim != 2 or material.shape[1] != 3 or len(material) < 3
                or bary.shape != (len(material), 3)
                or physical.ndim != 2 or physical.shape[1] != 3
                or len(physical) < 3
                or not np.isfinite([affine_area, physical_area]).all()
                or affine_area <= 0. or physical_area <= 0.
                or not np.isfinite(bary).all()
                or np.any(bary < -1e-3)
                or not np.allclose(bary.sum(axis=1), 1., rtol=0., atol=1e-9)):
            raise ValueError('Regional contact has invalid material or physical cell geometry.')
        expected_material = bary @ triangle
        expected_material /= np.linalg.norm(expected_material, axis=1, keepdims=True)
        if not np.allclose(expected_material, material, rtol=0., atol=3e-10):
            raise ValueError('Regional contact cell belongs to another material face.')
        untrimmed_area = mesh_coverage._polygon_area(material, radius)
        inside = mesh_coverage._clip_planes(material, face_planes)
        spherical_area = mesh_coverage._polygon_area(inside, radius)
        measured_physical = mesh_coverage._polygon_area(physical, radius)
        if (spherical_area <= 0.
                or not math.isclose(measured_physical, physical_area,
                                    rel_tol=2e-10, abs_tol=1e-7)):
            raise ValueError('Regional contact cell lost its measured physical area.')
        outside = max(0., untrimmed_area - spherical_area) / untrimmed_area
        max_outside = max(max_outside, outside)
        if outside > tolerance:
            raise ValueError('Regional contact cell leaves its native material face.')
        mismatch = abs(spherical_area - affine_area) / spherical_area
        max_mismatch = max(max_mismatch, mismatch)
        if mismatch > tolerance:
            raise ValueError('Regional contact affine and spherical source areas disagree.')
        contributions = []
        covered = 0.
        for region in regions:
            piece = mesh_coverage._clip_planes(
                inside, region_planes[region['region_id']])
            area = mesh_coverage._polygon_area(piece, radius)
            if area <= 0.:
                continue
            represented[region['region_id']] += area
            covered += area
            volume = area * region['thickness_km']
            ordinary_volume = area * (region['thickness_km']
                                      - region['dense_thickness_km'])
            dense_volume = area * region['dense_thickness_km']
            mass = (region['crust_mass_kg'] / region['physical_area_km2']) * area
            force = float(entry_contact_channel.buoyancy_surface_load(
                volume, mass, area)) * area * 1e6
            contributions.append(dict(
                region_id=region['region_id'], material_polygon=piece.copy(),
                material_area_km2=area,
                physical_volume_km3=volume,
                ordinary_volume_km3=ordinary_volume,
                dense_volume_km3=dense_volume,
                crust_mass_kg=mass, buoyancy_force_n=force))
        if abs(covered - spherical_area) > max(1e-7, face_area * 2e-10):
            raise ValueError('Regional contact cell did not close its native phase partition.')
        force = math.fsum(row['buoyancy_force_n'] for row in contributions)
        cell_loads.append(dict(cell_index=index,
                               physical_area_km2=physical_area,
                               mean_buoyancy_pa=force / (physical_area * 1e6),
                               buoyancy_force_n=force,
                               spherical_material_area_km2=spherical_area,
                               affine_material_area_km2=affine_area,
                               regional_contributions=contributions))
        spherical_total += spherical_area
        affine_total += affine_area
        physical_total += physical_area
    if (any(represented[row['region_id']]
            > row['physical_area_km2'] + max(1e-7, face_area * 2e-10)
            for row in regions)
            or not math.isclose(physical_total,
                                float(projected_pair['physical_contact_area_km2']),
                                rel_tol=2e-10, abs_tol=1e-7)
            or not math.isclose(affine_total,
                                float(projected_pair['material_reference_contact_area_km2']),
                                rel_tol=2e-10, abs_tol=1e-7)):
        raise ValueError('Regional projected loads exceed their material or physical budget.')
    return dict(face_id=bound['face_id'], cell_loads=cell_loads,
                contact_buoyancy_force_n=math.fsum(
                    row['buoyancy_force_n'] for row in cell_loads),
                spherical_material_contact_area_km2=spherical_total,
                affine_material_contact_area_km2=affine_total,
                physical_contact_area_km2=physical_total,
                represented_material_area_km2_by_region=represented,
                max_affine_spherical_area_mismatch=max_mismatch,
                max_outside_source_area_fraction=max_outside,
                scope='read-only native regional loads; no phase, heat or owner commit')
