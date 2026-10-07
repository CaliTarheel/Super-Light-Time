"""Finite registered-entry provenance for new collision-sheet order only.

This does not initiate entry, create a slab, release an attachment, or permit
active entry/stack mechanics. The existing entry energy guard remains required.
"""
from copy import copy, deepcopy
from fractions import Fraction

import numpy as np

import entry_regions
import finite_entry_arc

VERSION = 3


def _identity(value, name, *, minimum=0):
    # Check the original scalar: int(1.75), bool and NumPy negative indexing
    # must never turn malformed provenance into a different valid identity.
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not minimum <= value <= np.iinfo(np.int64).max):
        raise ValueError(f'{name} must be an original bounded integer identity.')
    return int(value)


def _integers(value, name, *, shape=None, minimum=0, upper=None, unique=False):
    array = np.asarray(value)
    if (array.dtype.kind not in 'iu' or array.dtype.itemsize > 8
            or (shape is None and array.ndim != 1)
            or (shape is not None and array.shape != shape)
            or np.any(array < minimum) or np.any(array > np.iinfo(np.int64).max)
            or (upper is not None and np.any(array >= upper))
            or (unique and len(np.unique(array)) != array.size)):
        raise ValueError(f'{name} needs aligned bounded original integers.')
    return array


def _inputs(s, one, two):
    """Validate identity/geometry before indexing or registry integer coercion."""
    one = _identity(one, 'First collision sheet', minimum=1)
    two = _identity(two, 'Second collision sheet', minimum=1)
    if one == two:
        raise ValueError('Inherited polarity needs two distinct collision sheets.')
    ids = _integers(s.parcel_patch, 'Persistent material identities', unique=True)
    n = len(ids)
    uids = _integers(s.plate_uid, 'Plate UIDs', unique=True)
    owners = _integers(s.parcel_plate, 'Material owner indices', shape=(n,), upper=len(uids))
    sheets = _integers(s.parcel_collision_sheet, 'Collision sheet identities', shape=(n,), minimum=1)
    if np.asarray(s.active).shape != uids.shape:
        raise ValueError('Active plates must align with plate UIDs.')
    mesh = s.material_surface
    mesh_ids = _integers(mesh['face_id'], 'Surface material identities', shape=(n,), unique=True)
    mesh_owners = _integers(mesh['face_owner'], 'Surface owner indices', shape=(n,), upper=len(uids))
    if not np.array_equal(ids, mesh_ids) or not np.array_equal(owners, mesh_owners):
        raise ValueError('Surface material identities and owners must match the native parcels.')
    vertices = np.asarray(mesh['vertices'])
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.dtype.kind not in 'iuf'
            or vertices.dtype.itemsize > 8 or not np.isfinite(vertices).all()
            or np.any(np.abs(np.linalg.norm(vertices, axis=1)-1.) > 1e-10)):
        raise ValueError('Inherited polarity needs finite unit-sphere material vertices.')
    faces = _integers(mesh['faces'], 'Material face vertex indices', shape=(n, 3), upper=len(vertices))
    if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise ValueError('Material faces need three distinct vertex identities.')
    radius = mesh.get('radius_km', 6371.)
    if (isinstance(radius, (bool, np.bool_)) or not np.isscalar(radius)
            or np.asarray(radius).dtype.kind not in 'iuf' or not np.isfinite(radius) or radius <= 0.):
        raise ValueError('Material sphere radius must be finite and positive.')
    state = s.continental_entry_regions
    old = _integers(state['face_ids'], 'Registered material identities', unique=True)
    _integers(s.parcel_entry_region, 'Entry membership identities', shape=old.shape)
    # _validate uses int keys internally. Reject non-integer originals first,
    # including row identities that happen to compare equal after truncation.
    for row in state['regions']:
        _identity(row['id'], 'Entry region identity', minimum=1)
        _identity(row['source_trench_id'], 'Entry source trench identity', minimum=1)
        _identity(row['overriding_plate_uid'], 'Entry overriding plate UID')
    for row in getattr(s, 'trench_systems', ()):
        _identity(row['id'], 'Source trench identity', minimum=1)
        _identity(row['downgoing_plate_uid'], 'Source incoming plate UID')
        _identity(row['overriding_plate_uid'], 'Source overriding plate UID')
    return one, two, ids, sheets, uids[owners], vertices[faces], float(radius)


def _fraction(x):
    return Fraction(float(x))


def _det(a, b, c):
    return (a[0]*(b[1]*c[2]-b[2]*c[1])
            -a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]))


def _clip(polygon, field):
    if not polygon:
        return []
    result = []
    previous = polygon[-1]
    old = sum(a*b for a, b in zip(previous, field))
    for current in polygon:
        new = sum(a*b for a, b in zip(current, field))
        if (old < 0) != (new < 0):
            t = old/(old-new)
            result.append(tuple(a+t*(b-a) for a, b in zip(previous, current)))
        if new >= 0:
            result.append(current)
        previous, old = current, new
    return result


def _area(polygon):
    # Fraction of the complete material reference triangle, not physical km2.
    return sum(a[1]*b[2]-a[2]*b[1] for a, b in zip(polygon, polygon[1:]+polygon[:1])) if polygon else Fraction(0)


def _overlap(triangle, other):
    """Exact represented projective intersection in the first face's chart."""
    a = [tuple(map(_fraction, row)) for row in triangle]
    b = [tuple(map(_fraction, row)) for row in other]
    if _det(*a) <= 0 or _det(*b) <= 0:
        raise ValueError('Inherited polarity needs outward nondegenerate material triangles.')
    polygon = [tuple(Fraction(int(i == j)) for i in range(3)) for j in range(3)]
    for i in range(3):
        field = tuple(_det(b[i], b[(i+1) % 3], point) for point in a)
        polygon = _clip(polygon, field)
    if _area(polygon) <= 0:
        raise ValueError('Reported collision pair has no exact represented positive overlap.')
    return polygon


def _entered_part(triangle, polygon, row, radius):
    normal = np.asarray(row['hinge_normal'])
    q = radius*1000.*np.arcsin(np.clip(triangle@normal, -1., 1.))
    if np.any(np.abs(triangle@normal) >= 1.-1e-12):
        raise ValueError('Inherited entry witness reaches a hinge-coordinate pole.')
    entered = _clip(polygon, tuple(map(_fraction, q)))
    if _area(entered) <= 0:
        return Fraction(0)
    if 'finite_midpoint' not in row or 'finite_half_length_km' not in row:
        raise ValueError('New collision polarity requires finite registered entry provenance.')
    left, right, _, _ = finite_entry_arc.endpoint_planes(
        normal, row['finite_midpoint'], row['finite_half_length_km'], radius)
    # These are precisely the represented nodal fields used by native entry
    # energy. Rational clipping changes no field, distance or endpoint law.
    for field in (triangle@left, triangle@right):
        entered = _clip(entered, tuple(map(_fraction, field)))
    return _area(entered)


def choose(s, one, two, first, second):
    """Return one fully witnessed order, None for unrelated contact, or reject.

    Every actual positive overlap pair must be completely covered by a
    consistent finite entered footprint. Partial or opposed regional histories
    cannot be expressed by the current one-order-per-sheet-pair ledger.
    """
    if not entry_regions.enabled(s):
        return None
    one, two, ids, sheets, owners, triangles, radius = _inputs(s, one, two)
    first, second = np.asarray(first), np.asarray(second)
    n = len(ids)
    if (first.ndim != 1 or second.shape != first.shape or not len(first)
            or first.dtype.kind not in 'iu' or second.dtype.kind not in 'iu'
            or np.any(first < 0) or np.any(second < 0)
            or np.any(first >= n) or np.any(second >= n) or np.any(first == second)):
        raise ValueError('Inherited polarity needs nonempty aligned integer overlap pairs.')
    pair_sheets = np.sort(np.column_stack((sheets[first], sheets[second])), axis=1)
    if not np.all(pair_sheets == np.sort([one, two])):
        raise ValueError('Overlap pairs do not belong to the requested collision sheets.')
    # Registry validation can align genuine newly born faces. Keep that read
    # detached even when admit_contact is called directly rather than refresh.
    probe = copy(s)
    probe.continental_entry_regions = deepcopy(s.continental_entry_regions)
    probe.parcel_entry_region = np.asarray(s.parcel_entry_region).copy()
    regions, _ = entry_regions._validate(probe)
    sources = {int(row['id']): row for row in getattr(s, 'trench_systems', ())}
    if len(sources) != len(getattr(s, 'trench_systems', ())):
        raise ValueError('Inherited polarity source trench identities are not unique.')
    membership = probe.parcel_entry_region
    decisions = []
    evidence = []
    any_witness = False
    for a, b in zip(first, second):
        a, b = int(a), int(b)
        choices = []
        for lower, upper in ((a, b), (b, a)):
            ident = int(membership[lower])
            if not ident:
                continue
            region = regions[ident]
            polygon = _overlap(triangles[lower], triangles[upper])
            total = _area(polygon)
            entered = _entered_part(triangles[lower], polygon, region, radius)
            if entered <= 0:
                continue
            any_witness = True
            source = sources.get(int(region['source_trench_id']))
            if (source is None or source.get('phase') not in ('initiating', 'mature', 'quiet')
                    or int(source['downgoing_plate_uid']) != int(owners[lower])
                    or int(source['overriding_plate_uid']) != int(region['overriding_plate_uid'])):
                raise ValueError('Finite entry lost its actual inherited trench polarity; explicit handoff required.')
            if int(owners[upper]) != int(region['overriding_plate_uid']):
                raise ValueError('Entered collision needs an explicit overriding-owner handoff.')
            order = (int(sheets[upper]), int(sheets[lower]))
            if set(order) != {int(one), int(two)}:
                raise ValueError('Finite entry witness does not belong to this collision pair.')
            fraction = entered/total
            choices.append((order, fraction == 1))
            evidence.append(dict(source_trench_id=int(source['id']), entry_region_id=ident,
                incoming_face_id=int(s.parcel_patch[lower]), overriding_face_id=int(s.parcel_patch[upper]),
                incoming_plate_uid=int(owners[lower]), overriding_plate_uid=int(owners[upper]),
                overlap_reference_fraction=[str(total.numerator), str(total.denominator)],
                covered_fraction=[str(fraction.numerator), str(fraction.denominator)]))
        decisions.append(choices)
    if not any_witness:
        return None
    orders = {order for choices in decisions for order, _ in choices}
    if len(orders) != 1:
        raise ValueError('Conflicting finite inherited polarities need regional contact ordering.')
    if any(len(choices) != 1 or not choices[0][1] for choices in decisions):
        raise ValueError('Partial finite entry provenance cannot order every collision overlap region.')
    top, under = orders.pop()
    return dict(top_sheet=top, under_sheet=under, polarity_policy_version=VERSION,
                polarity_source='complete finite registered entry provenance',
                polarity_entry_witnesses=evidence)
