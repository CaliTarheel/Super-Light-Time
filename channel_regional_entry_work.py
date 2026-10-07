"""Read-only phase-resolved entry work on projected material contact cells.

The installed native entry potential still uses a face-mean column. This
oracle keeps each regional phase's signed buoyancy with its material polygon
and reports the difference from that installed potential before any source
handoff is attempted. It does not change the native timestep.
"""

import math

import numpy as np

import continental_entry
import entry_affine_footprint
import entry_contact_channel
import mesh_coverage
from channel_native_lower_regions import bind_lower_regions
from channel_native_regional_contact_loads import bind_regional_projected_loads
from entry_projected_loads import uniform_contact_entry_work


def _children(triangle):
    a, b, c = triangle
    ab = (a + b) / np.linalg.norm(a + b)
    bc = (b + c) / np.linalg.norm(b + c)
    ca = (c + a) / np.linalg.norm(c + a)
    return (np.array([a, ab, ca]), np.array([ab, b, bc]),
            np.array([ca, bc, c]), np.array([ab, bc, ca]))


def _spherical_affine_mean(polygon, inverse_face, nodal_q,
                           radius_km, refinement):
    """Area-weighted affine q at spherical subtriangle centroids."""
    triangles = [np.asarray(polygon, float)[[0, index, index + 1]]
                 for index in range(1, len(polygon) - 1)]
    for _ in range(refinement):
        triangles = [child for triangle in triangles
                     for child in _children(triangle)]
    areas = []
    moments = []
    for triangle in triangles:
        area = mesh_coverage._polygon_area(triangle, radius_km)
        if area <= 0.:
            continue
        centroid = triangle.sum(axis=0)
        centroid /= np.linalg.norm(centroid)
        bary = inverse_face @ centroid
        bary /= bary.sum()
        if not np.isfinite(bary).all() or np.min(bary) < -1e-6:
            raise ValueError('Regional entry quadrature left its material face.')
        areas.append(area)
        moments.append(area * float(bary @ nodal_q))
    total = math.fsum(areas)
    if total <= 0.:
        raise ValueError('Regional entry quadrature has no material area.')
    return math.fsum(moments) / total, total


def regional_contact_entry_work(s, lower_face_id, projected_pair,
                                hinge_normal, dip_degrees, *,
                                refinement=3, coordinate_tolerance=1e-4):
    """Compare phase-aware spherical work with the installed mean-face law.

    Both calculations use the existing nodal-affine along-slab coordinate.
    The phase-aware branch uses spherical material area and validates a second
    quadrature level. Its discrepancy from the installed law is separated into
    area/coordinate measure and spatial phase-localization contributions.
    """
    if (not isinstance(refinement, (int, np.integer))
            or isinstance(refinement, (bool, np.bool_))
            or not 0 <= refinement <= 5
            or not math.isfinite(coordinate_tolerance)
            or not 0. < coordinate_tolerance < 1.):
        raise ValueError('Regional entry needs bounded quadrature settings.')
    bound = bind_lower_regions(s, lower_face_id)
    loads = bind_regional_projected_loads(s, lower_face_id, projected_pair)
    face = bound['lower_triangle']
    normal = np.asarray(hinge_normal, float)
    dip = float(dip_degrees)
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or abs(float(np.linalg.norm(normal)) - 1.) > 1e-10
            or not math.isfinite(dip) or not 0. < dip < 90.):
        raise ValueError('Regional entry needs a signed unit hinge and finite dip.')
    footprint = entry_affine_footprint.AffineConveyorFootprint(
        face, normal, dip, radius_km=bound['radius_km'])
    for cell in projected_pair['contact_cells']:
        expected = footprint.project(cell['material_barycentric_polygon'])
        if not np.allclose(expected, cell['physical_polygon'],
                           rtol=0., atol=3e-10):
            raise ValueError('Regional entry hinge disagrees with projected contact.')
    volume = math.fsum(row['physical_volume_km3'] for row in bound['regions'])
    mass = math.fsum(row['crust_mass_kg'] for row in bound['regions'])
    native_volume, native_mass = continental_entry.native_inventory(s)
    face_index = bound['face_index']
    if (not math.isclose(float(native_volume[face_index]), volume,
                         rel_tol=2e-12)
            or not math.isclose(float(native_mass[face_index]), mass,
                                rel_tol=2e-12)):
        raise ValueError('Regional entry phase mass disagrees with the installed native potential.')
    face_area = bound['physical_area_km2']
    nodal_q = bound['radius_km'] * 1000. * np.arcsin(
        np.clip(face @ normal, -1., 1.))
    inverse = np.linalg.inv(face.T)
    depth_factor = math.sin(math.radians(dip))
    phase_work = []
    uniform_area_moment = []
    max_coordinate_error_m = 0.
    for cell in loads['cell_loads']:
        for phase in cell['regional_contributions']:
            mean_q, coarse_area = _spherical_affine_mean(
                phase['material_polygon'], inverse, nodal_q,
                bound['radius_km'], refinement)
            finer_q, finer_area = _spherical_affine_mean(
                phase['material_polygon'], inverse, nodal_q,
                bound['radius_km'], refinement + 1)
            if (not math.isclose(coarse_area, phase['material_area_km2'],
                                 rel_tol=2e-10, abs_tol=1e-7)
                    or not math.isclose(finer_area, phase['material_area_km2'],
                                        rel_tol=2e-10, abs_tol=1e-7)):
                raise ValueError('Regional entry quadrature lost phase area.')
            error = abs(finer_q - mean_q)
            max_coordinate_error_m = max(max_coordinate_error_m, error)
            if error > coordinate_tolerance * max(1., float(np.max(np.abs(nodal_q)))):
                raise ValueError('Regional entry coordinate quadrature is unresolved.')
            mean_depth = finer_q * depth_factor
            work = phase['buoyancy_force_n'] * mean_depth
            phase_work.append(work)
            uniform_area_moment.append(
                phase['material_area_km2'] * 1e6 * mean_depth)
    face_pressure = float(entry_contact_channel.buoyancy_surface_load(
        volume, mass, face_area))
    uniform_spherical = face_pressure * math.fsum(uniform_area_moment)
    old = uniform_contact_entry_work(
        face, normal, dip, volume, mass, projected_pair,
        radius_km=bound['radius_km'])
    phase_total = math.fsum(phase_work)
    old_total = float(old['contact_entry_work_j'])
    force = math.fsum(cell['buoyancy_force_n']
                      for cell in loads['cell_loads'])
    if not math.isclose(force, loads['contact_buoyancy_force_n'],
                        rel_tol=2e-12, abs_tol=1e-4):
        raise ValueError('Regional entry work lost projected phase force.')
    return dict(phase_resolved_work_j=phase_total,
                old_mean_face_work_j=old_total,
                uniform_spherical_work_j=uniform_spherical,
                phase_localization_difference_j=phase_total - uniform_spherical,
                geometry_measure_difference_j=uniform_spherical - old_total,
                total_work_difference_j=phase_total - old_total,
                contact_buoyancy_force_n=force,
                max_coordinate_refinement_difference_m=max_coordinate_error_m,
                scope='read-only phase-aware entry work; native source remains face-mean')
