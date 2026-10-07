"""Read-only refinement oracle for projected continental entry/contact area.

The conserved lower material triangle is clipped to its positive conveyor
coordinate and optional finite trench. Each barycentric subtriangle is mapped
to a horizontal footprint, then intersected with one upper spherical face.
Mapped geodesic edges are curved, so every returned contact area is explicitly
labelled by its subdivision level. It is not a native contact acceptance test.
"""

import numpy as np

import entry_affine_footprint
import entry_footprint
import finite_entry_arc
import mesh_coverage
import slab_memory
from channel_region_geometry import _checked_polygon


def _reference_fraction(polygon):
    if len(polygon) < 3:
        return 0.
    x, y = polygon[:, 1], polygon[:, 2]
    return float(abs(np.sum(x*np.roll(y, -1) - y*np.roll(x, -1))))


def _children(triangle):
    a, b, c = triangle
    ab, bc, ca = (a+b)/2., (b+c)/2., (c+a)/2.
    return (np.array([a, ab, ca]), np.array([ab, b, bc]),
            np.array([ca, bc, c]), np.array([ab, bc, ca]))


def projected_pair_areas(lower_triangle, upper_triangle, hinge_normal, dip_degrees,
                         *, radius_km=6371., finite_midpoint=None,
                         finite_half_length_km=None, refinement=0,
                         include_cells=False,
                         coordinate_discretization='spherical_point'):
    """Return both conserved-reference and physical-footprint pair areas.

    `refinement` is the number of four-way barycentric subdivisions of each
    entered piece. Call at several levels to measure the curved-edge error;
    this function never silently accepts a chosen tolerance as exact.
    """
    if (not isinstance(refinement, (int, np.integer)) or isinstance(refinement, (bool, np.bool_))
            or not 0 <= refinement <= 7 or not np.isfinite(radius_km) or radius_km <= 0.
            or (finite_midpoint is None) != (finite_half_length_km is None)
            or type(include_cells) is not bool
            or not isinstance(coordinate_discretization,str)
            or coordinate_discretization not in ('spherical_point','affine_nodal')):
        raise ValueError('Projected pair needs a radius, bounded refinement and matched finite endpoints.')
    lower, _ = _checked_polygon(lower_triangle, 'Lower material face', radius_km)
    upper, _ = _checked_polygon(upper_triangle, 'Upper material face', radius_km)
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError('Projected pair needs two spherical triangles.')
    # Validate dip and hinge even if no entered material is present.
    entry_footprint.project_points(lower, hinge_normal, dip_degrees)
    affine = (entry_affine_footprint.AffineConveyorFootprint(
        lower,hinge_normal,dip_degrees,radius_km=radius_km)
        if coordinate_discretization=='affine_nodal' else None)
    normal = np.asarray(hinge_normal, float)
    q = radius_km*1000.*np.arcsin(np.clip(lower@normal, -1., 1.))
    # This reduced model only couples the slab above 660 km depth. Farther
    # conveyor travel needs a separately resolved deep-slab/contact law; near
    # the hinge-coordinate pole the projection is also singular.
    limit_m = slab_memory.upper_mantle_length_km(dip_degrees)*1000.
    if np.any(q > limit_m*(1.+1e-12)):
        raise ValueError('Projected entry exceeds the represented upper-mantle slab window.')
    left = right = np.ones(3)
    if finite_midpoint is not None:
        l, r, _, _ = finite_entry_arc.endpoint_planes(
            normal, finite_midpoint, finite_half_length_km, radius_km)
        left, right = lower@l, lower@r
    _, _, _, _, fraction, parts = finite_entry_arc.integrate(q, left, right)
    face_area = mesh_coverage._polygon_area(lower, radius_km)
    physical_entered = physical_contact = reference_contact = 0.
    unprojected_contact = 0.
    reference_tiles = 0.
    coordinate_mismatch_m = 0.
    cells = [] if include_cells else None
    panels = 0
    for _, barycentric in parts:
        original_piece = barycentric@lower
        original_piece /= np.linalg.norm(original_piece,axis=1,keepdims=True)
        unprojected_contact += mesh_coverage._polygon_area(
            mesh_coverage.clip_triangle(original_piece,upper),radius_km)
        triangles = [barycentric]
        for _ in range(refinement):
            triangles = [child for triangle in triangles for child in _children(triangle)]
        for bary in triangles:
            material = bary@lower
            material /= np.linalg.norm(material, axis=1, keepdims=True)
            # Native entry energy uses an affine interpolant of nodal hinge
            # distance. The spherical projection uses the exact distance at
            # each point; expose their discrepancy rather than hiding it.
            sites = np.vstack((bary,bary.mean(axis=0)))
            unit = sites@lower
            unit /= np.linalg.norm(unit,axis=1,keepdims=True)
            spherical_q = radius_km*1000.*np.arcsin(np.clip(unit@normal,-1.,1.))
            coordinate_mismatch_m = max(coordinate_mismatch_m,
                                        float(np.max(np.abs(spherical_q-sites@q))))
            projected = (affine.project(bary) if affine is not None else
                         entry_footprint.project_points(material, normal, dip_degrees))
            reference_tiles += face_area*_reference_fraction(bary)
            physical_entered += mesh_coverage._polygon_area(projected, radius_km)
            polygon = mesh_coverage.clip_triangle(projected, upper)
            intersection_area = mesh_coverage._polygon_area(polygon, radius_km)
            physical_contact += intersection_area
            if len(polygon) >= 3 and intersection_area > 0.:
                if affine is None:
                    original = entry_footprint.unproject_points(polygon, normal, dip_degrees)
                    weights = np.linalg.solve(lower.T, original.T).T
                    weights /= weights.sum(axis=1, keepdims=True)
                else:
                    weights = affine.unproject(polygon)
                    original = weights@lower
                    original /= np.linalg.norm(original,axis=1,keepdims=True)
                if not np.isfinite(weights).all():
                    raise ValueError('Projected contact cannot recover material coordinates.')
                reference_area = face_area*_reference_fraction(weights)
                reference_contact += reference_area
                if cells is not None:
                    cells.append(dict(material_barycentric_polygon=weights,
                                      material_polygon=original,
                                      physical_polygon=polygon,
                                      material_reference_area_km2=reference_area,
                                      physical_area_km2=intersection_area))
            panels += 1
    reference_entered = face_area*float(fraction)
    if (abs(reference_tiles-reference_entered) > max(1e-7, face_area*2e-10)
            or reference_contact > reference_entered+max(1e-7,face_area*2e-8)
            or physical_contact > physical_entered+max(1e-7,physical_entered*2e-8)):
        raise ValueError('Projected contact lost its material or physical area budget.')
    return dict(material_reference_entered_area_km2=reference_entered,
                material_reference_contact_area_km2=reference_contact,
                physical_entered_area_km2=physical_entered,
                physical_contact_area_km2=physical_contact,
                legacy_unprojected_contact_area_km2=unprojected_contact,
                face_reference_area_km2=face_area, refinement=int(refinement),
                panels=panels,max_affine_coordinate_mismatch_m=coordinate_mismatch_m,
                coordinate_discretization=coordinate_discretization,
                contact_cells=cells,
                scope='read-only curved-edge refinement; no native contact acceptance')
