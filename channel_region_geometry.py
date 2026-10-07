"""Read-only co-registered spherical partitions for source-history remapping.

All polygons must refer to one material face in the same spherical coordinate
frame. A caller must transport the old partition into that frame before this
module computes intersections; physical positions at different times cannot be
intersected directly to infer material identity.
"""

import numpy as np

import mesh_coverage


def remove_roundoff_vertices(polygon):
    """Drop repeated adjacent spherical vertices within binary64 representation."""
    tolerance = 8. * np.finfo(float).eps
    result = []
    for point in polygon:
        if not result or np.max(np.abs(point - result[-1])) > tolerance:
            result.append(point)
    if len(result) > 1 and np.max(np.abs(result[0] - result[-1])) <= tolerance:
        result.pop()
    return np.asarray(result)


def _checked_polygon(polygon, label, radius_km):
    points = np.asarray(polygon, float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 3
            or not np.isfinite(points).all()
            or not np.allclose(np.linalg.norm(points, axis=1), 1.,
                               rtol=0., atol=3e-12)):
        raise ValueError(label + ' must be a finite unit-sphere polygon.')
    planes = mesh_coverage._triangle_planes(points)
    if (not np.isfinite(planes).all()
            or np.any(points @ planes.T < -3e-12)
            or mesh_coverage._polygon_area(points, radius_km) <= 0.):
        raise ValueError(label + ' must be a nondegenerate outward convex polygon.')
    return points, planes


def spherical_partition_intersections(face_triangle, old_polygons,
                                      new_polygons, *, radius_km=6371.):
    """Return old/new physical-area fractions and their intersection matrix.

    Uses the native great-circle clipping/solid-angle area kernel. Both input
    partitions must be disjoint, wholly inside the face, and cover it. The
    old/new polygons must already be in one co-moving material frame.
    """
    if (not np.isfinite(radius_km) or radius_km <= 0.
            or not isinstance(old_polygons, (list, tuple)) or not old_polygons
            or not isinstance(new_polygons, (list, tuple)) or not new_polygons):
        raise ValueError('Spherical source partitions need a radius and polygons.')
    face, face_planes = _checked_polygon(face_triangle, 'Material face', radius_km)
    if len(face) != 3:
        raise ValueError('Material face must be a spherical triangle.')
    face_area = mesh_coverage._polygon_area(face, radius_km)
    tolerance = 2e-10 * face_area

    def checked_partition(polygons, label):
        checked = [_checked_polygon(polygon, f'{label} {index}', radius_km)
                   for index, polygon in enumerate(polygons)]
        areas = np.array([mesh_coverage._polygon_area(polygon, radius_km)
                          for polygon, _ in checked])
        for polygon, _ in checked:
            inside = mesh_coverage._clip_planes(polygon, face_planes)
            if abs(mesh_coverage._polygon_area(inside, radius_km)
                   - mesh_coverage._polygon_area(polygon, radius_km)) > tolerance:
                raise ValueError(label + ' extends beyond its material face.')
        for i in range(len(checked)):
            for j in range(i + 1, len(checked)):
                overlap = mesh_coverage._clip_planes(
                    checked[i][0], checked[j][1])
                if mesh_coverage._polygon_area(overlap, radius_km) > tolerance:
                    raise ValueError(label + ' polygons overlap each other.')
        if abs(float(areas.sum()) - face_area) > tolerance:
            raise ValueError(label + ' polygons do not cover their material face.')
        return checked, areas

    old, old_area = checked_partition(old_polygons, 'Old source')
    new, new_area = checked_partition(new_polygons, 'New source')
    overlap = np.array([
        [mesh_coverage._polygon_area(
            mesh_coverage._clip_planes(old_polygon, new_planes), radius_km)
         for _, new_planes in new]
        for old_polygon, _ in old])
    if (not np.allclose(overlap.sum(axis=1), old_area,
                        rtol=0., atol=tolerance)
            or not np.allclose(overlap.sum(axis=0), new_area,
                               rtol=0., atol=tolerance)):
        raise ValueError('Spherical source intersections do not conserve both partitions.')
    return dict(face_area_km2=face_area,
                old_fractions=old_area / face_area,
                new_fractions=new_area / face_area,
                overlap_fractions=overlap / face_area,
                scope='co-registered spherical material face; no motion map or native commit')


def transport_material_polygon(old_face_triangle, new_face_triangle, polygon,
                               *, radius_km=6371.):
    """Carry a polygon by its radial barycentric coordinates on one face.

    The vertex correspondence is material identity. For a deforming face this
    changes subface physical area; source inventories need a separate local
    strain transaction before such transported polygons can be committed.
    """
    old, old_planes = _checked_polygon(
        old_face_triangle, 'Old material face', radius_km)
    new, new_planes = _checked_polygon(
        new_face_triangle, 'New material face', radius_km)
    points, _ = _checked_polygon(polygon, 'Old material region', radius_km)
    if len(old) != 3 or len(new) != 3:
        raise ValueError('Material polygon transport needs matched face triangles.')
    if np.any(points @ old_planes.T < -3e-12):
        raise ValueError('Old material region lies outside its source face.')
    coefficients = np.linalg.solve(old.T, points.T).T
    weights = coefficients / coefficients.sum(axis=1)[:, None]
    if not np.isfinite(weights).all() or np.any(weights < -2e-10):
        raise ValueError('Old material region has invalid radial barycentric points.')
    mapped = weights @ new
    mapped /= np.linalg.norm(mapped, axis=1)[:, None]
    _checked_polygon(mapped, 'Transported material region', radius_km)
    if np.any(mapped @ new_planes.T < -3e-12):
        raise ValueError('Transported material region lies outside its new face.')
    return mapped


def material_intersection_cells(old_face_triangle, new_face_triangle,
                                old_polygons, new_polygons, *, radius_km=6371.):
    """Measure each new contact cell and its material preimage on the old face.

    A cell's old/new physical-area ratio determines its local area-factor and
    thickness transaction. Both partitions and both sets of area margins are
    checked; the source history and crust fields are not changed here.
    """
    old_geometry = spherical_partition_intersections(
        old_face_triangle, old_polygons, old_polygons,
        radius_km=radius_km)
    transported = [transport_material_polygon(
        old_face_triangle, new_face_triangle, polygon,
        radius_km=radius_km) for polygon in old_polygons]
    new_geometry = spherical_partition_intersections(
        new_face_triangle, transported, new_polygons,
        radius_km=radius_km)
    cells = []
    old_sum = np.zeros(len(old_polygons))
    new_sum = np.zeros(len(new_polygons))
    for i, polygon in enumerate(transported):
        for j, zone in enumerate(new_polygons):
            new_piece = mesh_coverage._clip_planes(
                polygon, mesh_coverage._triangle_planes(zone))
            new_area = mesh_coverage._polygon_area(new_piece, radius_km)
            if new_area <= 0.:
                continue
            old_piece = transport_material_polygon(
                new_face_triangle, old_face_triangle, new_piece,
                radius_km=radius_km)
            old_area = mesh_coverage._polygon_area(old_piece, radius_km)
            if old_area <= 0.:
                raise ValueError('Source intersection lost its material preimage.')
            old_sum[i] += old_area
            new_sum[j] += new_area
            cells.append(dict(old_index=i, new_index=j,
                              old_area_km2=old_area, new_area_km2=new_area,
                              old_polygon=old_piece, new_polygon=new_piece))
    old_face_area = old_geometry['face_area_km2']
    new_face_area = new_geometry['face_area_km2']
    if (not np.allclose(old_sum, old_geometry['old_fractions'] * old_face_area,
                        rtol=0., atol=2e-10 * old_face_area)
            or not np.allclose(new_sum, new_geometry['new_fractions'] * new_face_area,
                               rtol=0., atol=2e-10 * new_face_area)):
        raise ValueError('Material source cells do not conserve old and new areas.')
    return dict(cells=cells, old_geometry=old_geometry,
                new_geometry=new_geometry, transported_old_polygons=transported,
                scope='read-only projective material intersections; no crust update')
