"""Positive basal-domain partition for a supplied finite control triangle.

Research integration component, not a native callsite or subcell-owner model.
Only the bottom material layer touches mantle. A native constant fractional
plate measure cannot be replaced by a full material footprint on a mixed cell.
Such states reject explicitly rather than subtracting matrices or changing
ownership. Caller must provide every potentially intersecting material face.
"""
from fractions import Fraction
from functools import wraps
import math
import numpy as np

from . import stable_basal_metric


class UnsupportedBasalOwnership(ValueError):
    """The supplied native support cannot resolve the physical basal owner."""


def _finite_arithmetic(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise', under='raise'):
                result = function(*args, **kwargs)
            def visit(value):
                if isinstance(value, np.ndarray) and value.dtype.kind in 'fcu':
                    if not np.isfinite(value).all():
                        raise ValueError('Derived basal measure or SI drag is nonfinite.')
                elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    raise ValueError('Derived basal measure or SI drag is nonfinite.')
                elif isinstance(value, dict):
                    for child in value.values(): visit(child)
                elif isinstance(value, (list, tuple)):
                    for child in value: visit(child)
            visit(result)
            return result
        except (FloatingPointError, OverflowError, ZeroDivisionError, np.linalg.LinAlgError) as error:
            raise ValueError('Basal allocation exceeds its resolved numerical domain.') from error
    return checked


def _triangle(value, name):
    a = np.asarray(value)
    if a.dtype.kind not in 'fiu' or (a.dtype.kind == 'f' and a.dtype.itemsize > 8):
        raise ValueError(f'{name} requires real binary64-compatible coordinates.')
    a = np.asarray(a, float)
    if a.shape != (3, 3) or not np.isfinite(a).all():
        raise ValueError(f'{name} requires three finite three-vectors.')
    if not np.allclose(np.linalg.norm(a, axis=1), 1., rtol=0., atol=2e-12):
        raise ValueError(f'{name} requires unit spherical coordinates.')
    rational = [[Fraction(float(x)) for x in row] for row in a]
    if _dot(rational[0], _cross(rational[1], rational[2])) <= 0:
        raise ValueError(f'{name} requires positive finite winding.')
    # The chosen triangle is the minor convex intersection of its edge planes.
    if np.any(a@a.sum(axis=0) <= 0):
        raise ValueError(f'{name} must lie in a common open hemisphere.')
    return a.copy(), rational


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Fraction(0))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _area(poly):
    return sum((a[0]*b[1]-a[1]*b[0] for a, b in zip(poly, poly[1:]+poly[:1])), Fraction(0))/2


def _clean(poly):
    result = []
    for point in poly:
        if not result or point != result[-1]:
            result.append(point)
    if len(result) > 1 and result[0] == result[-1]:
        result.pop()
    changed = True
    while changed and len(result) >= 3:
        changed = False
        for i in range(len(result)):
            a, b, c = result[i-1], result[i], result[(i+1) % len(result)]
            if (b[0]-a[0])*(c[1]-b[1]) == (b[1]-a[1])*(c[0]-b[0]):
                result.pop(i); changed = True
                break
    return result


def _split(poly, coefficients):
    """Exact shared crossing for positive/negative halves in one cell chart."""
    d = [coefficients[0]+x*(coefficients[1]-coefficients[0])
         +y*(coefficients[2]-coefficients[0]) for x, y in poly]
    if all(v >= 0 for v in d):
        return poly, []
    if all(v <= 0 for v in d):
        return [], poly
    pos, neg = [], []
    for i, point in enumerate(poly):
        previous = (i-1) % len(poly)
        if d[i]*d[previous] < 0:
            t = d[previous]/(d[previous]-d[i])
            crossing = tuple((1-t)*a+t*b for a, b in zip(poly[previous], point))
            pos.append(crossing); neg.append(crossing)
        if d[i] >= 0:
            pos.append(point)
        if d[i] <= 0:
            neg.append(point)
    return _clean(pos), _clean(neg)


def _partition(poly, planes):
    inside, outside = poly, []
    for coefficients in planes:
        inside, piece = _split(inside, coefficients)
        if len(piece) >= 3 and _area(piece) > 0:
            outside.append(piece)
        if len(inside) < 3 or _area(inside) == 0:
            return [], outside
    return inside, outside


def _identifier(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer.')
    return int(value)


def _order(above):
    graph = {}
    for pair in above:
        if len(pair) != 2:
            raise ValueError('Sheet order requires upper/lower pairs.')
        upper, lower = (_identifier(v, 'sheet ID') for v in pair)
        if upper == lower:
            raise ValueError('Sheet order cannot contain a cycle.')
        graph.setdefault(upper, set()).add(lower)
    result = {}
    def visit(node, path):
        if node in path:
            raise ValueError('Sheet order cannot contain a cycle.')
        if node not in result:
            children = graph.get(node, set())
            descendants = set(children)
            for child in children:
                descendants.update(visit(child, path | {node}))
            result[node] = descendants
        return result[node]
    for node in graph:
        visit(node, set())
    return result


def _unit_polygon(poly, basis):
    # Compatibility diagnostic only. Callers needing positive work must retain
    # the complete stable record; a global binary64 metric may lose its weak mode.
    value = stable_basal_metric.build(np.asarray(basis, float), poly)
    return value.polygon, value.metric_global, value.area_unit


@_finite_arithmetic
def partition_cell(control_triangle, material, above, *, radius_m, saved_cell_area_m2,
                   support, basal_drag_pa_s_per_m, relative_tolerance=2e-10,
                   allocation_policy='preserve-native'):
    """Partition geometry with explicit native-preserving or resolved allocation.

    ``material`` records contain face_id, sheet_id, owner and triangle. ``above``
    consists of persistent (upper_sheet, lower_sheet) pairs, including same-owner
    layers. ``support`` maps native plate IDs to constant cell fractions.
    Native saved area/geometric area is retained as an explicit measure density.
    Default preserve-native rejects mixed support under material. The explicit
    resolved-material-bottom policy CHANGES native owner allocation on covered
    pieces; it never silently falls back or claims unchanged per-owner work.

    Returns positive material pieces (only bottom faces), plate-only complement
    pieces, and their rigid-work metrics. Total material velocity still needs
    integration over each returned polygon by the joint solver. No material
    basal term should simultaneously be added at every full-face node.
    """
    if allocation_policy not in ('preserve-native', 'resolved-material-bottom'):
        raise ValueError('Unknown basal allocation policy.')
    for name, value in [('radius_m', radius_m), ('saved_cell_area_m2', saved_cell_area_m2),
                        ('basal_drag_pa_s_per_m', basal_drag_pa_s_per_m),
                        ('relative_tolerance', relative_tolerance)]:
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    control, basis = _triangle(control_triangle, 'Control cell')
    native = {}
    if not isinstance(support, dict) or not support:
        raise ValueError('Native support must be an explicit plate fraction mapping.')
    for key, value in support.items():
        owner = _identifier(key, 'support owner')
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError('Native support must be finite and nonnegative.')
        native[owner] = float(value)
    if not math.isclose(math.fsum(native.values()), 1., rel_tol=0., abs_tol=64*np.finfo(float).eps):
        raise ValueError('Native support does not represent a normalized control-cell measure.')
    graph = _order(above)
    records = {}
    pieces = [([(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(1))], ())]
    for record in material:
        face = _identifier(record['face_id'], 'face ID')
        if face in records:
            raise ValueError('Material face IDs must be unique.')
        sheet = _identifier(record['sheet_id'], 'sheet ID')
        owner = _identifier(record['owner'], 'material owner')
        _, triangle = _triangle(record['triangle'], f'Material face {face}')
        records[face] = dict(sheet=sheet, owner=owner)
        planes = [tuple(_dot(_cross(a, b), point) for point in basis)
                  for a, b in zip(triangle, triangle[1:]+triangle[:1])]
        updated = []
        for polygon, covers in pieces:
            inside, outside = _partition(polygon, planes)
            updated.extend((p, covers) for p in outside)
            if inside:
                updated.append((inside, covers+(face,)))
        pieces = updated
        if sum((_area(p) for p, _ in pieces), Fraction(0)) != Fraction(1, 2):
            raise ValueError('Exact cell-chart partition lost or duplicated area.')

    control_metric = stable_basal_metric.build(control, ((0,0),(1,0),(0,1)))
    cell_metric, cell_area = control_metric.metric_global.copy(), control_metric.area_unit
    radius2 = float(radius_m)**2
    geometric_area = cell_area*radius2
    density = float(saved_cell_area_m2)/geometric_area
    cell_metric *= radius2*density
    metric_scale = radius2*density
    local_cell_metric = control_metric.metric_local*metric_scale
    local_factor = control_metric.factor_local*np.sqrt(metric_scale)
    result = []
    for polygon, covers in pieces:
        bottom = None
        if covers:
            sheets = [records[f]['sheet'] for f in covers]
            if len(set(sheets)) != len(sheets):
                raise ValueError('Positive overlap within one material sheet is invalid.')
            candidates = [f for f in covers if all(other == f or records[f]['sheet'] in
                          graph.get(records[other]['sheet'], ()) for other in covers)]
            if len(candidates) != 1:
                raise ValueError('Basal contact requires a unique ordered bottom material layer.')
            bottom = candidates[0]
            owner = records[bottom]['owner']
            if owner not in native:
                raise UnsupportedBasalOwnership(f'Bottom face {bottom} owner {owner} is not in the supplied active owner map.')
            if allocation_policy == 'preserve-native' and (native.get(owner, 0.) != 1. or any(weight != 0. for p, weight in native.items() if p != owner)):
                raise UnsupportedBasalOwnership(
                    f'Bottom face {bottom}, owner {owner}, has positive coverage in mixed/mismatched '
                    f'native support {native}; subcell mantle ownership is not available.')
        stable = stable_basal_metric.build(control, polygon)
        spherical, metric, area = stable.polygon, stable.metric_global, stable.area_unit
        fractions = {records[bottom]['owner']: 1.} if bottom is not None else native.copy()
        result.append(dict(polygon=spherical, chart_polygon=tuple(polygon),
            chart_area=_area(polygon), cover_faces=covers, bottom_face=bottom,
            owner=records[bottom]['owner'] if bottom is not None else None,
            fractions=fractions, area_m2=area*metric_scale,
            metric_m2=metric*metric_scale, stable_metric=stable,
            metric_control_frame_m2=stable.metric_in(control_metric)*metric_scale))

    area_error = math.fsum(row['area_m2'] for row in result)-float(saved_cell_area_m2)
    def matrix_sum(values):
        values = list(values)
        return np.array([[math.fsum(value[i,j] for value in values) for j in range(3)] for i in range(3)])
    metric_sum = matrix_sum(row['metric_control_frame_m2'] for row in result)
    metric_error = stable_basal_metric._whitened_error(metric_sum-local_cell_metric,local_factor)
    if abs(area_error) > relative_tolerance*saved_cell_area_m2 or metric_error > relative_tolerance:
        raise ValueError('Exact-chart local metrics fail full cell area/rigid-work closure.')
    # Positive pieces only: never global-minus-material matrix subtraction.
    material_metric = {p: np.zeros((3, 3)) for p in native}
    remaining_metric = {p: np.zeros((3, 3)) for p in native}
    for row in result:
        destination = material_metric if row['bottom_face'] is not None else remaining_metric
        for owner, fraction in row['fractions'].items():
            destination[owner] += fraction*row['metric_m2']
    deltas = {}
    local_deltas = {}
    for owner, fraction in native.items():
        deltas[owner] = material_metric[owner]+remaining_metric[owner]-fraction*cell_metric
        local_allocated = matrix_sum(row['fractions'].get(owner,0.)*row['metric_control_frame_m2'] for row in result)
        local_deltas[owner] = local_allocated-fraction*local_cell_metric
        closure = stable_basal_metric._whitened_error(local_deltas[owner],local_factor)
        if allocation_policy == 'preserve-native' and closure > relative_tolerance*fraction:
            raise ValueError('Allocated plate measure does not preserve native rigid work.')
    common_error = stable_basal_metric._whitened_error(matrix_sum(local_deltas.values()),local_factor)
    if common_error > relative_tolerance:
        raise ValueError('Allocated total basal measure fails common-rotation closure.')
    reassigned = [row for row in result if row['bottom_face'] is not None and
                  (native[row['owner']] != 1. or any(w != 0. for p,w in native.items() if p != row['owner']))]
    return dict(pieces=result, geometric_cell_area_m2=geometric_area,
        native_measure_density=density, native_cell_area_m2=float(saved_cell_area_m2),
        native_cell_metric_m2=cell_metric, support=native,
        control_stable_metric=control_metric,
        material_metric_m2=material_metric, remaining_metric_m2=remaining_metric,
        remaining_rotational_drag_n_m_s={p: basal_drag_pa_s_per_m*radius2*m for p, m in remaining_metric.items()},
        basal_drag_pa_s_per_m=float(basal_drag_pa_s_per_m), radius_m=float(radius_m),
        area_closure_error_m2=area_error, scaled_metric_closure_error=metric_error,
        allocation_policy=allocation_policy, per_owner_legacy_metric_change_m2=deltas,
        scaled_common_rotation_closure_error=common_error,
        reassigned_material_area_m2=math.fsum(row['area_m2'] for row in reassigned),
        reassigned_material_pieces=len(reassigned),
        scope='Supplied control cell and complete supplied candidate list; no global coverage certificate or subcell-support reconstruction.')
