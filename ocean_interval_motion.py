"""Experimental, face-local spherical endpoint maps with explicit attachments.

This supplies one geometric map for probes, material and local area change.
It does not choose velocities, integrate time, certify a complete trajectory,
or accept a simulation step. A caller must additionally check continuity,
global intersections and the source/work budget before using an interval.
No production evolution path imports this module.

For source and endpoint triangles S and T (vertices stored as columns), the
map is F(x)=A x/|A x|, A=T inv(S). Its spherical area Jacobian is det(A)/|A x|^3.
This is a declared projective finite-element map, not an exact solution of a
time-dependent velocity field. It maps edges to their endpoint great-circle
arcs and has an exact inverse inside each attached donor triangle. Explicit
donor and owner identities prevent a point being silently attached to a
different overlapping plate by nearest-point sampling.
"""
from __future__ import annotations

import hashlib
import numpy as np

VERSION = 1
_FIELDS = ('source_triangles_xyz', 'mapped_triangles_xyz', 'owner_uids',
           'donor_ids', 'source_epoch_myr', 'duration_myr', 'radius_m', 'version',
           'source_inverse_condition', 'mapped_inverse_condition')


def _fingerprint(context):
    digest = hashlib.sha256()
    for name in _FIELDS:
        value = np.ascontiguousarray(context[name])
        digest.update(repr((name, value.dtype.str, value.shape)).encode('ascii'))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _number(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or np.asarray(value).dtype.kind not in 'iuf'
            or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(name + ' must be finite and ' + ('positive.' if positive else 'nonnegative.'))
    return float(value)


def _ids(value, count, name):
    array = np.asarray(value)
    if array.shape != (count,) or array.dtype.kind not in 'iu' or np.any(array < 0):
        raise ValueError(name + ' must contain aligned nonnegative integer identities.')
    # Avoid overflowing uint64 into a different donor identity.
    if np.any(array > np.iinfo(np.int64).max):
        raise ValueError(name + ' exceeds the supported identity range.')
    return array.astype(np.int64, copy=True)


def _triangles(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf':
        raise ValueError(name + ' must contain real triangle coordinates.')
    array = np.asarray(value, float)
    if (array.ndim != 3 or array.shape[1:] != (3, 3) or not len(array)
            or not np.isfinite(array).all()
            or not np.allclose(np.linalg.norm(array, axis=2), 1., rtol=0., atol=2e-12)):
        raise ValueError(name + ' must contain finite unit-sphere triangles.')
    # Outward orientation alone does not establish a minor, well-resolved
    # triangle. Reject near-collinear vertices before inversion as well.
    centre = array.sum(axis=1)
    singular = np.linalg.svd(array, compute_uv=False)
    if (np.any(np.linalg.det(array) <= 0.)
            or np.any(np.einsum('fvi,fi->fv', array, centre) <= 0.)
            or np.any(singular[:, -1] <= singular[:, 0]*1e-10)):
        raise ValueError(name + ' needs outward, resolved triangles inside a minor hemisphere.')
    return array.copy()


def _attachment_keys(owners, donors):
    keys = np.empty(len(donors), dtype=[('owner', '<i8'), ('donor', '<i8')])
    keys['owner'], keys['donor'] = owners, donors
    return keys


def prepare(source_triangles_xyz, mapped_triangles_xyz, owner_uids, donor_ids,
            *, source_epoch_myr, duration_myr, radius_m):
    """Build a serializable face-local map; no global motion is certified.

    Owner UID / donor ID pairs are unique within this interval. Shared edges
    must be checked by the caller: two locally valid maps can still separate,
    overlap or exchange positions. Endpoint validity does not certify the
    intervening trajectory. The returned state contains ordinary arrays and
    scalars, suitable for the existing typed checkpoint codec. Recorded inverse
    condition numbers describe roundoff amplification; this kernel does not
    promise a fixed mapping error for highly elongated triangles. A caller's
    finite-step acceptance must account for its required geometric accuracy.
    """
    source = _triangles(source_triangles_xyz, 'Source geometry')
    mapped = _triangles(mapped_triangles_xyz, 'Endpoint geometry')
    if mapped.shape != source.shape:
        raise ValueError('Source and endpoint triangles must align.')
    donors = _ids(donor_ids, len(source), 'Donor IDs')
    owners = _ids(owner_uids, len(source), 'Owner UIDs')
    if np.any(owners == 0):
        raise ValueError('Owner UIDs must be positive.')
    if len(np.unique(_attachment_keys(owners, donors))) != len(donors):
        raise ValueError('Owner UID / donor ID pairs must be unique within the interval.')
    context = dict(version=VERSION, source_triangles_xyz=source,
        mapped_triangles_xyz=mapped, donor_ids=donors,
        owner_uids=owners,
        source_epoch_myr=_number(source_epoch_myr, 'Source epoch'),
        duration_myr=_number(duration_myr, 'Interval duration', positive=True),
        radius_m=_number(radius_m, 'Sphere radius', positive=True),
        source_inverse_condition=np.linalg.cond(source),
        mapped_inverse_condition=np.linalg.cond(mapped))
    context['fingerprint'] = _fingerprint(context)
    return context


def _validate(context):
    try:
        valid = (type(context['version']) is int and context['version'] == VERSION
                 and context['fingerprint'] == _fingerprint(context))
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError('The prepared source interval or endpoint geometry has changed.')


def _attached(context, donor_ids, owner_uids, points, inverse):
    _validate(context)
    if type(inverse) is not bool:
        raise ValueError('Map direction must be an explicit boolean.')
    raw = np.asarray(points)
    if raw.dtype.kind not in 'iuf':
        raise ValueError('Attached points must be real unit-sphere vectors.')
    points = np.asarray(points, float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or not np.allclose(np.linalg.norm(points, axis=1), 1., atol=2e-12, rtol=0.)):
        raise ValueError('Attached points must be finite unit-sphere vectors.')
    donor = _ids(donor_ids, len(points), 'Point donor IDs')
    owner = _ids(owner_uids, len(points), 'Point owner UIDs')
    keys = _attachment_keys(context['owner_uids'], context['donor_ids'])
    order = np.argsort(keys)
    sorted_keys = keys[order]
    requested = _attachment_keys(owner, donor)
    index = np.searchsorted(sorted_keys, requested)
    if np.any(index == len(order)) or np.any(sorted_keys[index] != requested):
        raise ValueError('An attached owner / donor identity is absent from this interval.')
    index = order[index]
    source = context['mapped_triangles_xyz' if inverse else 'source_triangles_xyz'][index]
    mapped = context['source_triangles_xyz' if inverse else 'mapped_triangles_xyz'][index]
    # Geometric containment must not amplify inverse conditioning into false
    # exclusions of exact vertices. Evaluate oriented great-circle distances
    # using local differences, including a floating-point arithmetic bound.
    edges = np.roll(source, -1, axis=1)-source
    planes = np.cross(source, edges)
    planes /= np.linalg.norm(planes, axis=2)[:, :, None]
    offset = points[:, None, :]-source
    distance = np.einsum('nvi,nvi->nv', planes, offset)
    tolerance = 32.*np.finfo(float).eps*(np.linalg.norm(offset, axis=2)+np.linalg.norm(edges, axis=2))
    if np.any(distance < -tolerance):
        raise ValueError('A point lies outside its explicitly attached donor triangle.')
    matrix = np.linalg.inv(source.transpose(0, 2, 1))
    operator = mapped.transpose(0, 2, 1)@matrix
    # Evaluate the point by local oriented-volume coordinates, not A@x, whose
    # cancellation can place a mapped boundary point outside its own triangle.
    # Only geometric incidence within the arithmetic bound above is snapped;
    # this does not broaden donor membership by an inverse-condition factor.
    determinant = np.einsum('ni,ni->n', source[:, 0],
        np.cross(source[:, 1]-source[:, 0], source[:, 2]-source[:, 0]))
    from_point = source-points[:, None, :]
    weights = np.column_stack([
        np.einsum('ni,ni->n', points,
            np.cross(from_point[:, (i+1)%3], from_point[:, (i+2)%3]))/determinant
        for i in range(3)])
    for edge in range(3):
        weights[np.abs(distance[:, edge]) <= tolerance[:, edge], (edge+2)%3] = 0.
    weights = np.maximum(weights, 0.)
    unscaled = np.einsum('nvi,nv->ni', mapped, weights)
    length = np.linalg.norm(unscaled, axis=1)
    if np.any(length <= 0.) or not np.isfinite(length).all():
        raise ValueError('The endpoint map is singular at an attached point.')
    return points, operator, unscaled/length[:, None], length


def map_points(context, donor_ids, owner_uids, points_xyz, *, inverse=False):
    """Map explicitly attached points to endpoints, or pull endpoints back."""
    return _attached(context, donor_ids, owner_uids, points_xyz, inverse)[2]


def area_jacobian(context, donor_ids, owner_uids, points_xyz, *, inverse=False):
    """Return local destination/source area, not a donor-average area ratio."""
    _, operator, _, length = _attached(context, donor_ids, owner_uids, points_xyz, inverse)
    jacobian = np.linalg.det(operator)/length**3
    if np.any(jacobian <= 0.) or not np.isfinite(jacobian).all():
        raise ValueError('The local area map must preserve orientation.')
    return jacobian


def push_tangent(context, donor_ids, owner_uids, points_xyz, tangent_vectors, *, inverse=False):
    """Push a spatial differential through this endpoint map.

    Units of the supplied vector are preserved. This is the spatial derivative
    of F, not an instantaneous or mean physical velocity over the interval.
    """
    points, operator, mapped, length = _attached(context, donor_ids, owner_uids, points_xyz, inverse)
    raw = np.asarray(tangent_vectors)
    if raw.dtype.kind not in 'iuf':
        raise ValueError('Spatial tangent vectors must be real.')
    vectors = np.asarray(tangent_vectors, float)
    if vectors.shape != points.shape or not np.isfinite(vectors).all():
        raise ValueError('Spatial differentials must be aligned finite tangent vectors.')
    scale = np.max(np.abs(vectors), axis=1, initial=0.)
    scaled = vectors/np.where(scale > 0., scale, 1.)[:, None]
    if np.any(np.abs(np.einsum('ni,ni->n', scaled, points))
              > 2e-11*np.linalg.norm(scaled, axis=1)):
        raise ValueError('Spatial differentials must be aligned finite tangent vectors.')
    pushed = np.einsum('nij,nj->ni', operator, scaled)
    derivative = (pushed-mapped*np.einsum('ni,ni->n', pushed, mapped)[:, None])/length[:, None]
    with np.errstate(over='ignore', invalid='ignore'):
        result = derivative*scale[:, None]
    if not np.isfinite(result).all():
        raise ValueError('The mapped spatial differential exceeds finite numerical range.')
    return result
