"""Continuous scalar reconstruction on the actual connected material surface.

Face columns remain authoritative. Area-weighted shared-vertex values and
radial barycentric interpolation provide a bounded C0 display/export surface;
they are not new physical degrees of freedom or conservative column remapping.
Only indexed vertices shared by the same owner exchange values. Coincident
coordinates on disconnected sheets, ocean gaps, and fault sides never blend.
"""
from __future__ import annotations

import numpy as np
import mesh_geometry


def prepare(vertices, faces, owner, *, area_km2=None):
    """Build a checkpoint-safe face-to-shared-vertex reconstruction stencil."""
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces)
    owner = np.asarray(owner)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,)
            or not np.isfinite(vertices).all()):
        raise ValueError('Material vertices must be finite xyz coordinates.')
    if (faces.ndim != 2 or faces.shape[1:] != (3,)
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError('Material faces must reference existing vertex triples.')
    if owner.shape != (len(faces),) or not np.issubdtype(owner.dtype, np.integer):
        raise ValueError('Material owners must be an integer per face.')
    area = (mesh_geometry.spherical_area(vertices, faces) if area_km2 is None
            else np.asarray(area_km2, float))
    if area.shape != (len(faces),) or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError('Material face areas must be finite and positive.')
    # A well-formed material surface already duplicates fault-side vertices.
    # Including owner here also protects imported histories with shared indices
    # across an owner boundary, without mutating their geometry or face IDs.
    if len(owner) and np.all(owner == owner[0]):
        # One-owner patches (including arc capacity trials) need only sort
        # vertex IDs. This gives exactly the same lexical node order as the
        # general (vertex, owner) keys, without sorting structured rows.
        vertex_index, inverse = np.unique(faces.ravel(), return_inverse=True)
        unique = np.column_stack((vertex_index, np.full(len(vertex_index), owner[0], dtype=owner.dtype)))
    else:
        keys = np.column_stack((faces.ravel(), np.repeat(owner, 3)))
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    face_vertices = inverse.reshape((-1, 3))
    corner_area = np.repeat(area, 3)
    total = np.bincount(inverse, weights=corner_area, minlength=len(unique))
    return dict(face_vertices=face_vertices, vertex_index=unique[:, 0],
                vertex_owner=unique[:, 1], face_area_km2=area.copy(),
                corner_weight=corner_area/total[inverse], vertex_count=len(unique))


def vertex_values(stencil, face_values):
    """Area-weighted scalar values, constant-exact and bounded by contributors.

This averaging is not in general linear-exact from face-centre observations.
The following barycentric interpolation is linear in the reconstructed nodal
values, but it does not preserve each original face's mean or volume.
"""
    values = np.asarray(face_values, float)
    if values.shape != (len(stencil['face_vertices']),) or not np.isfinite(values).all():
        raise ValueError('Material reconstruction needs one finite scalar per face.')
    return np.bincount(stencil['face_vertices'].ravel(),
                       weights=np.repeat(values, 3)*stencil['corner_weight'],
                       minlength=stencil['vertex_count'])


def sample_hits(stencil, nodal_values, face_index, weights):
    """Interpolate already selected exact triangle hits; never fill a gap."""
    values = np.asarray(nodal_values, float)
    faces = np.asarray(face_index)
    weights = np.asarray(weights, float)
    if (faces.ndim != 1 or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= len(stencil['face_vertices']))):
        raise ValueError('Reconstruction requires actual containing material faces.')
    if values.shape != (stencil['vertex_count'],) or not np.isfinite(values).all():
        raise ValueError('Invalid reconstructed material vertex values.')
    if (weights.shape != (len(faces), 3) or not np.isfinite(weights).all()
            or np.any(weights < -1e-10)
            or not np.allclose(weights.sum(axis=1), 1., rtol=0., atol=1e-9)):
        raise ValueError('Material hit weights must be barycentric coordinates.')
    # Locator tolerances can admit roundoff-sized negatives at an edge.
    weights = np.maximum(weights, 0.)
    weights /= weights.sum(axis=1)[:, None]
    return np.einsum('ij,ij->i', values[stencil['face_vertices'][faces]], weights)
