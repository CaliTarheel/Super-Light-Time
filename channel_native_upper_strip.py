"""Read-only material binding for an overriding-plate flexure strip.

The caller supplies an ordered set of points and their material face IDs.
This module checks that they form one equally spaced great-circle strip on
one collision sheet, then retrieves each region's saved pre-contact basal
depth. It does not infer a trench normal, rigidity, or contact force.
"""

import numpy as np

from channel_marker_binding import bind_marker_regions
from channel_upper_basal_reference import REFERENCE


def bind_upper_strip(s, upper_face_ids, upper_xyz):
    """Return physical spacing and unloaded bases for native upper material."""
    import channel_region_native

    if not channel_region_native.enabled(s):
        raise ValueError('Upper strip needs an installed native regional store.')
    store = s.channel_region_store
    surface = s.material_surface
    if (store.get('version') != channel_region_native.VERSION
            or store.get('epoch_myr') != float(s.t)
            or store.get('geometry_revision') != int(surface['geometry_revision'])):
        raise ValueError('Upper strip needs current material geometry.')
    face_ids = np.asarray(upper_face_ids)
    points = np.asarray(upper_xyz, float)
    if (face_ids.ndim != 1 or face_ids.dtype.kind not in 'iu'
            or len(face_ids) < 5 or points.shape != (len(face_ids), 3)
            or not np.isfinite(points).all()
            or not np.allclose(np.linalg.norm(points, axis=1), 1., rtol=0., atol=3e-12)):
        raise ValueError('Upper strip needs at least five aligned unit material points.')
    native_ids = np.asarray(surface['face_id'])
    sheets = np.asarray(getattr(s, 'parcel_collision_sheet', []))
    owners = np.asarray(getattr(s, 'parcel_plate', []))
    surface_owners = np.asarray(surface['face_owner'])
    if (native_ids.ndim != 1 or sheets.shape != native_ids.shape
            or sheets.dtype.kind not in 'iu'
            or owners.shape != native_ids.shape or owners.dtype.kind not in 'iu'
            or surface_owners.shape != native_ids.shape
            or not np.array_equal(owners, surface_owners)
            or len(np.unique(native_ids)) != len(native_ids)):
        raise ValueError('Upper strip needs aligned native sheet and plate owners.')
    sheet_by_face = {int(face_id): int(sheet)
                     for face_id, sheet in zip(native_ids, sheets)}
    owner_by_face = {int(face_id): int(owner)
                     for face_id, owner in zip(native_ids, owners)}
    if (any(int(face_id) not in sheet_by_face for face_id in face_ids)
            or len({sheet_by_face[int(face_id)] for face_id in face_ids}) != 1
            or len({owner_by_face[int(face_id)] for face_id in face_ids}) != 1):
        raise ValueError('Upper strip must stay on one identified sheet and plate owner.')
    radius = float(surface.get('radius_km', np.nan))
    if not np.isfinite(radius) or radius <= 0.:
        raise ValueError('Upper strip needs a finite positive native radius.')

    first, last = points[0], points[-1]
    axis = np.cross(first, last)
    magnitude = float(np.linalg.norm(axis))
    if magnitude <= 1e-12:
        raise ValueError('Upper strip needs distinct non-antipodal endpoints.')
    axis /= magnitude
    total_angle = float(np.arctan2(magnitude, first @ last))
    angles = np.arctan2(np.cross(first, points) @ axis, points @ first)
    increments = np.diff(angles)
    if (np.max(np.abs(points @ axis)) > 1e-8
            or np.any(increments <= 0.)
            or np.any(angles < -1e-10)
            or np.any(angles > total_angle + 1e-10)
            or not np.allclose(increments, total_angle / (len(points) - 1),
                               rtol=1e-5, atol=1e-10)):
        raise ValueError('Upper strip must be ordered and equally spaced on one great circle.')

    bound = bind_marker_regions(surface, store['records'],
                                np.arange(len(points), dtype=np.int64),
                                face_ids, points)
    if any(REFERENCE not in row for row in bound):
        raise ValueError('Upper strip needs every material region to have an unloaded base.')
    epochs = {row[REFERENCE]['epoch_myr'] for row in bound}
    if len(epochs) != 1 or next(iter(epochs)) > float(s.t):
        raise ValueError('Upper strip needs one past pre-contact reference epoch.')
    return dict(spacing_m=radius * 1000. * total_angle / (len(points) - 1),
                distance_m=radius * 1000. * angles.copy(),
                unloaded_upper_base_depth_m=np.asarray(
                    [row[REFERENCE]['depth_m'] for row in bound], float),
                face_id=face_ids.copy(),
                region_id=tuple(row['region_id'] for row in bound),
                collision_sheet=sheet_by_face[int(face_ids[0])],
                plate_owner=owner_by_face[int(face_ids[0])],
                reference_epoch_myr=next(iter(epochs)),
                scope='read-only material strip; no trench normal or native contact solve')


def trench_normal_points(hinge_normal, entered_contact_xyz, offsets_m, radius_km):
    """Sample a great circle normal to a persistent entry hinge.

    The supplied contact point fixes along-trench position. Positive signed
    offsets follow the entry side of the hinge, where the existing entry law
    measures depth. A caller must separately verify that the point belongs to
    a current entered-contact polygon; this function only checks geometry.
    """
    normal = np.asarray(hinge_normal, float)
    contact = np.asarray(entered_contact_xyz, float)
    offsets = np.asarray(offsets_m, float)
    radius = float(radius_km) * 1000.
    if (normal.shape != (3,) or contact.shape != (3,)
            or not np.isfinite(normal).all() or not np.isfinite(contact).all()
            or not np.allclose([np.linalg.norm(normal), np.linalg.norm(contact)],
                               1., rtol=0., atol=3e-12)
            or offsets.ndim != 1 or len(offsets) < 5
            or not np.isfinite(offsets).all()
            or not np.isfinite(radius) or radius <= 0.
            or offsets[0] < 0. or offsets[-1] >= .5 * np.pi * radius
            or np.any(np.diff(offsets) <= 0.)
            or not np.allclose(np.diff(offsets),
                               (offsets[-1] - offsets[0]) / (len(offsets) - 1),
                               rtol=1e-5, atol=1e-5)):
        raise ValueError('Trench-normal strip needs unit geometry and ordered metre offsets.')
    entry_sine = float(contact @ normal)
    if entry_sine <= 1e-12 or entry_sine >= 1. - 1e-12:
        raise ValueError('Trench-normal strip needs contact on the positive entry side.')
    contact_offset = radius * np.arcsin(entry_sine)
    if not offsets[0] <= contact_offset <= offsets[-1]:
        raise ValueError('Trench-normal strip must bracket its contact point.')
    hinge = contact - entry_sine * normal
    hinge /= np.linalg.norm(hinge)
    angles = offsets / radius
    points = np.cos(angles)[:, None] * hinge + np.sin(angles)[:, None] * normal
    return dict(xyz=points, hinge_xyz=hinge,
                contact_offset_m=contact_offset,
                scope='read-only signed hinge geometry; contact polygon not verified')


def bind_trench_normal_upper_strip(s, upper_face_ids, hinge_normal,
                                   entered_contact_xyz, offsets_m):
    """Compose signed hinge sampling with current native upper material."""
    geometry = trench_normal_points(
        hinge_normal, entered_contact_xyz, offsets_m,
        s.material_surface['radius_km'])
    bound = bind_upper_strip(s, upper_face_ids, geometry['xyz'])
    return dict(bound, hinge_xyz=geometry['hinge_xyz'],
                contact_offset_m=geometry['contact_offset_m'],
                xyz=geometry['xyz'],
                scope='read-only hinge-normal upper material; contact zone and forces external')
