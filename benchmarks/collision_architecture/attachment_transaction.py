"""Conservative preflight for explicitly authored intact finite material bonds.

Research infrastructure only: no engine callsite, attachment admission, strength,
release, policy activation, state mutation or physical trajectory is supplied.
The two endpoints map the SAME reference triangle into two birth-root charts.
Exact rational arithmetic audits represented real integer/binary64 charts; an
unresolved gap/overlap fails closed rather than becoming an attachment tolerance.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math

import numpy as np


VERSION = 1
SCOPE = 'explicitly-authored-intact-finite-bonds'
DOMAIN = ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)),
          (Fraction(0), Fraction(1)))


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(name+' must be a nonnegative integer.')
    return int(value)


def _matrix(value, name):
    array = np.asarray(value)
    # Inspect the authored representation before conversion: object bytes are
    # pointers, and casting complex values would discard retained geometry.
    # Binary16/32 values embed exactly in binary64; wider floats do not.
    if array.dtype.kind not in 'iuf' or (array.dtype.kind == 'f' and array.dtype.itemsize > 8):
        raise ValueError(name+' must use real integer or at most binary64 floating-point coordinates.')
    if array.shape != (3, 3) or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(name+' must contain nonnegative finite homogeneous root coordinates.')
    scalar = int if array.dtype.kind in 'iu' else float
    result = tuple(tuple(Fraction(scalar(v)) for v in row) for row in array)
    if _det(result) <= 0:
        raise ValueError(name+' must have positive finite area and orientation.')
    return result


def _det(a):
    return (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
            - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
            + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))


def _inverse(a):
    determinant = _det(a)
    return tuple(tuple((a[(j+1)%3][(i+1)%3]*a[(j+2)%3][(i+2)%3]
                        - a[(j+1)%3][(i+2)%3]*a[(j+2)%3][(i+1)%3])/determinant
                       for j in range(3)) for i in range(3))


def _multiply(a, b):
    return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))


def _area(polygon):
    return sum(a[0]*b[1]-b[0]*a[1]
               for a, b in zip(polygon, polygon[1:]+polygon[:1]))/2 if len(polygon) >= 3 else Fraction(0)


def _clip(polygon, plane):
    """Exact closed halfplane clip in the shared event triangle, no area floor."""
    if not polygon:
        return ()
    def distance(point):
        return plane[0]+plane[1]*point[0]+plane[2]*point[1]
    result = []
    previous = polygon[-1]; old = distance(previous)
    for current in polygon:
        new = distance(current)
        if (old < 0) != (new < 0):
            fraction = old/(old-new)
            result.append(tuple(previous[k]+fraction*(current[k]-previous[k]) for k in range(2)))
        if new >= 0:
            result.append(current)
        previous, old = current, new
    cleaned = []
    for point in result:
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return tuple(cleaned) if _area(tuple(cleaned)) > 0 else ()


def _intersection(first, second):
    result = first
    for a, b in zip(second, second[1:]+second[:1]):
        # Positive oriented polygon: left side of each directed edge.
        result = _clip(result, (a[0]*b[1]-a[1]*b[0], a[1]-b[1], b[0]-a[0]))
        if not result:
            break
    return result


def _leaf_rows(leaves):
    if not isinstance(leaves, dict) or set(leaves) != {'face_id', 'root_id', 'reference_corners', 'owner_uid'}:
        raise ValueError('Leaf state requires exact face/root/chart/owner fields.')
    face, root, owners = (np.asarray(leaves[k]) for k in ('face_id', 'root_id', 'owner_uid'))
    n = len(face) if face.ndim == 1 else -1
    if n < 1 or any(v.shape != (n,) or v.dtype.kind not in 'iu' or np.any(v < 0)
                    for v in (face, root, owners)) or len(np.unique(face)) != n:
        raise ValueError('Leaf IDs and owners must be aligned unique-face nonnegative integers.')
    charts = np.asarray(leaves['reference_corners'])
    if charts.shape != (n, 3, 3):
        raise ValueError('Leaf root charts must align with faces.')
    result = {}
    for f, r, owner, chart in zip(face, root, owners, charts):
        result.setdefault(int(r), []).append((int(f), int(owner), _matrix(chart, 'Leaf chart')))
    for rows in result.values():
        rows.sort(key=lambda row: row[0])
    return result


def _records(store):
    if (not isinstance(store, dict) or set(store) != {'version', 'scope', 'bonds'}
            or type(store['version']) is not int or store['version'] != VERSION
            or store['scope'] != SCOPE or not isinstance(store['bonds'], list)):
        raise ValueError('An explicit intact finite-bond store is required; legacy state is not inferred.')
    seen = set()
    records = []
    for record in store['bonds']:
        if not isinstance(record, dict) or set(record) != {'bond_id', 'creation_event_id', 'created_myr', 'source', 'receiver'}:
            raise ValueError('Each bond requires identity, authored event provenance and paired finite endpoints.')
        ident = _integer(record['bond_id'], 'Bond ID')
        _integer(record['creation_event_id'], 'Creation event ID')
        epoch = record['created_myr']
        if isinstance(epoch, (bool, np.bool_)) or not np.isscalar(epoch) or not math.isfinite(epoch) or epoch < 0:
            raise ValueError('Bond creation time must be finite and nonnegative.')
        if ident in seen:
            raise ValueError('Bond IDs must be unique.')
        seen.add(ident)
        endpoints = []
        for side in ('source', 'receiver'):
            endpoint = record[side]
            if not isinstance(endpoint, dict) or set(endpoint) != {'root_id', 'corners'}:
                raise ValueError('A finite endpoint requires its root and homogeneous triangle corners.')
            endpoints.append((_integer(endpoint['root_id'], 'Endpoint root'),
                              _matrix(endpoint['corners'], 'Endpoint footprint')))
        if endpoints[0][0] == endpoints[1][0]:
            raise ValueError('This primitive requires two distinct material birth roots.')
        records.append((ident, *endpoints))
    return sorted(records, key=lambda row: row[0])


def _endpoint_partition(endpoint, leaves):
    root, corners = endpoint
    pieces = []
    for face, owner, chart in leaves.get(root, ()):
        mapping = _multiply(corners, _inverse(chart))
        polygon = DOMAIN
        for j in range(3):
            # lambda=(1-x-y,x,y), lambda*corners*inverse(chart) >= 0.
            polygon = _clip(polygon, (mapping[0][j], mapping[1][j]-mapping[0][j],
                                      mapping[2][j]-mapping[0][j]))
        if polygon:
            pieces.append((face, owner, polygon))
    for i, (_, _, first) in enumerate(pieces):
        for _, _, second in pieces[:i]:
            if _area(_intersection(first, second)) > 0:
                raise ValueError('A finite bond endpoint has duplicate positive-area leaf coverage.')
    if sum((_area(p) for _, _, p in pieces), Fraction(0)) != Fraction(1, 2):
        raise ValueError('A finite bond endpoint has missing or unresolved represented leaf coverage.')
    return pieces


def _fraction(value):
    return [str(value.numerator), str(value.denominator)]


def _resolve(store, leaves):
    indexed = _leaf_rows(leaves)
    output = []
    for ident, source, receiver in _records(store):
        first = _endpoint_partition(source, indexed)
        second = _endpoint_partition(receiver, indexed)
        paired = []
        total = Fraction(0)
        for one, owner_one, a in first:
            for two, owner_two, b in second:
                polygon = _intersection(a, b)
                measure = 2*_area(polygon)
                if measure <= 0:
                    continue
                if owner_one != owner_two:
                    raise ValueError('Unsupported owner cut through an explicitly intact finite bond.')
                total += measure
                paired.append(dict(source_face_id=one, receiver_face_id=two, owner_uid=owner_one,
                    fraction_exact=_fraction(measure),
                    event_polygon_exact=[[_fraction(x), _fraction(y)] for x, y in polygon]))
        if total != 1:
            raise ValueError('Paired finite footprints do not conserve the entire authored bond.')
        output.append(dict(bond_id=ident, paired_cells=paired, covered_fraction_exact=['1', '1']))
    return output


def _canonical(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'iuf' or (value.dtype.kind == 'f' and value.dtype.itemsize > 8):
            raise ValueError('Finite attachment signatures require serialization-safe real numeric arrays.')
        return dict(dtype=value.dtype.str, shape=list(value.shape), bytes=value.tobytes().hex())
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        return {'float_hex': value.hex()}
    return value


def _signature(store, leaves):
    data = json.dumps(_canonical([store, leaves]), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def prepare_transaction(store, current_leaves, proposed_leaves):
    """Stage unchanged durable bonds plus conservative derived leaf incidence.

    The caller explicitly supplies an intact authored ledger. No bonds are born,
    deleted, weakened, or inferred here. Both representations must cover each
    paired footprint once, and every finite paired portion must share its owner.
    Neither input nor any world object is modified, including on failure.
    """
    current = _resolve(store, current_leaves)
    proposed = _resolve(store, proposed_leaves)
    return dict(version=VERSION, scope=SCOPE, store=deepcopy(store),
        current_signature=_signature(store, current_leaves),
        proposed_signature=_signature(store, proposed_leaves),
        current_incidence=current, proposed_incidence=proposed,
        measure='exact shared event-triangle fraction; not present physical area or bond strength')


def validate_prepared(prepared, store, current_leaves, proposed_leaves):
    """Revalidate a staged transaction before its caller mutates any world state."""
    fresh = prepare_transaction(store, current_leaves, proposed_leaves)
    if _canonical(prepared) != _canonical(fresh):
        raise ValueError('Finite attachment preflight is stale or was modified.')
    return deepcopy(fresh['store'])
