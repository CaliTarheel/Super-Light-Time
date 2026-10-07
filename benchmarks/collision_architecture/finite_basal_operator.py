"""Finite bottom-material basal resistance on the frozen joint SI space.

Caller-supplied PR210 allocations choose the physical measure and policy. This
module integrates that measure; it does not discover contacts, choose owners,
calibrate coefficients, or certify global coverage. ``replace_basal`` removes
the former nodal basal term before installing this one.
"""
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
import math

import numpy as np

from collision_interface import _precise_rotation_integral
from convex_partition import Edge
from . import shared_contact as joint
from .joint_resistance import _system_signature
from .sparse_shared_contact import BlockMap
from .framed_basal_map import FramedBlockMap
from .stable_basal_metric import StableMetric


def _real(value, name):
    result = np.asarray(value)
    if result.dtype.kind not in 'fiu' or (result.dtype.kind == 'f' and result.dtype.itemsize > 8):
        raise ValueError(f'{name} must contain real binary64-representable numbers.')
    result = result.astype(float)
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite.')
    return result


def _positive(value, name):
    result = _real(value, name)
    if result.shape != () or result <= 0:
        raise ValueError(f'{name} must be a finite positive scalar.')
    return float(result)


def _vector(value, count):
    result = _real(value, 'generalized velocity')
    if result.shape != (count,):
        raise ValueError('Finite generalized velocity must align with the operator.')
    return result


def _mapping(value, limit, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be an explicit ID mapping.')
    result = {}
    for key, item in value.items():
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < 0
               for x in (key, item)) or item >= limit:
            raise ValueError(f'{name} must contain valid integer IDs and slots.')
        result[int(key)] = int(item)
    if len(set(result.values())) != len(result):
        raise ValueError(f'{name} must not alias distinct IDs to one slot.')
    return result


def _identifier(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer ID.')
    return int(value)


def _fractions(value, name):
    if not isinstance(value, dict) or not value:
        raise ValueError(f'{name} requires an explicit nonempty fraction mapping.')
    result = {}
    for key, weight in value.items():
        owner = _identifier(key, name+' owner')
        fraction = _real(weight, name+' fraction')
        if fraction.shape != () or fraction < 0 or fraction > 1:
            raise ValueError(f'{name} fractions must be scalar values from zero to one.')
        result[owner] = float(fraction)
    if not math.isclose(math.fsum(result.values()), 1., rel_tol=0., abs_tol=64*np.finfo(float).eps):
        raise ValueError(f'{name} fractions must sum to one.')
    return result


def _diagonal(design):
    if isinstance(design, FramedBlockMap):
        return design.diagonal()
    return np.bincount(design.columns.ravel(), weights=np.sum(design.blocks**2, axis=1).ravel(),
                       minlength=design.shape[1])


def _absolute_action(design, value):
    if isinstance(design, FramedBlockMap):
        return design.absolute_action(value)
    local = np.einsum('bij,bj->bi', abs(design.blocks), abs(value[design.columns]))
    transpose = np.einsum('bij,bi->bj', abs(design.blocks), local)
    return np.bincount(design.columns.ravel(), weights=transpose.ravel(), minlength=design.shape[1])


def _absolute_forward(design, value):
    if isinstance(design, FramedBlockMap):
        return design.absolute_forward(value)
    local = np.einsum('bij,bj->bi', abs(design.blocks), abs(value[design.columns]))
    return np.bincount(design.rows.ravel(), weights=local.ravel(), minlength=design.shape[0])


def _finite(*values):
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError('Finite basal operator exceeds its resolved arithmetic range.')


def _metric_error(observed, expected):
    """Compare every rigid direction, including the weak centroid-spin axis."""
    lower = np.linalg.cholesky(expected)
    normalized = np.linalg.solve(lower, observed-expected)
    normalized = np.linalg.solve(lower, normalized.T).T
    return float(np.linalg.norm(normalized, 2))


def _polygon(value):
    p = _real(value, 'basal polygon')
    if (p.ndim != 2 or p.shape[1:] != (3,) or len(p) < 3
            or np.max(abs(np.linalg.norm(p, axis=1)-1.)) > 2e-12):
        raise ValueError('Basal pieces require finite unit-sphere polygons.')
    # These are the represented PR210 corners; do not silently re-normalize,
    # re-anchor, enlarge, or discard thin pieces here.
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        if np.any(Edge(a, b).distances(p) < 0.):
            raise ValueError('Basal polygon is not resolved convex positive geometry.')
    metric, area = _precise_rotation_integral(p)
    if area <= 0 or np.linalg.eigvalsh(metric).min() <= 0:
        raise ValueError('Basal polygon has unresolved positive area or rigid work.')
    return p, metric, area


def _trace_rule(system, face, polygon, order, density, beta):
    """Positive radial Duffy rule, compressed by QR without truncating modes."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    u, v = np.meshgrid(.5*(nodes+1.), .5*(nodes+1.), indexing='ij')
    wu, wv = np.meshgrid(.5*weights, .5*weights, indexing='ij')
    u, v, wu, wv = (x.ravel() for x in (u, v, wu, wv))
    basis = system['tangent_basis'][face]
    points = system['points'][face]
    factor = np.empty((0, 9)); area_sum = 0.; metric_sum = np.zeros((3, 3))
    for b, c in zip(polygon[1:-1], polygon[2:]):
        a = polygon[0]
        determinant = float(Edge(a, b).distances(c[None])[0])
        if determinant <= 0:
            raise ValueError('Every represented basal fan must have positive winding.')
        chord = a+u[:, None]*(b-a)+(1-u)[:, None]*v[:, None]*(c-a)
        length = np.linalg.norm(chord, axis=1)
        q = chord/length[:, None]
        area = system['radius_m']**2*density*wu*wv*(1-u)*determinant/length**3
        # The same homogeneous trace solved directly. Forming an inverse and
        # multiplying adds avoidable cancellation for narrow native faces;
        # retain the unchanged sign and reconstruction acceptance gates.
        alpha = np.linalg.solve(points.T, q.T).T
        band = 256*np.finfo(float).eps*np.maximum(1., np.max(abs(alpha), axis=1))
        if (np.any(area <= 0.) or np.any(alpha < -band[:, None])
                or np.max(np.linalg.norm(alpha@points-q, axis=1)) > 256*np.finfo(float).eps):
            raise ValueError('Basal quadrature has unresolved finite-face trace geometry.')
        projector = np.eye(3)[None]-q[:, :, None]*q[:, None, :]
        trace = np.zeros((len(q), 3, 9))
        trace[:, 0, 1] = q[:, 2]; trace[:, 0, 2] = -q[:, 1]
        trace[:, 1, 0] = -q[:, 2]; trace[:, 1, 2] = q[:, 0]
        trace[:, 2, 0] = q[:, 1]; trace[:, 2, 1] = -q[:, 0]
        for corner in range(3):
            trace[:, :, 3+2*corner:5+2*corner] = (
                alpha[:, corner, None, None]*(projector@basis[corner]))
        weighted = (np.sqrt(beta*area)[:, None, None]*trace).reshape(-1, 9)
        _finite(weighted)
        # At most one fan rule plus a9x9 previous factor is retained. Positive
        # factor construction avoids subtracting full-cell and covered drag.
        uncompressed = factor.T@factor+weighted.T@weighted
        factor = np.linalg.qr(np.vstack((factor, weighted)), mode='r')
        difference = factor.T@factor-uncompressed
        scale = np.sqrt(np.diag(uncompressed))
        denominator = scale[:, None]*scale[None]
        if np.any((denominator == 0.) & (difference != 0.)):
            raise ValueError('Basal factor compression changed an inactive coordinate.')
        normalized = np.divide(difference, denominator, out=np.zeros_like(difference), where=denominator > 0.)
        if np.max(abs(normalized)) > 256*9*np.finfo(float).eps:
            raise ValueError('Positive basal factor compression failed its Gram check.')
        area_sum += float(np.sum(area))
        metric_sum += np.einsum('n,nij->ij', area, projector)
    _finite(factor, area_sum, metric_sum)
    return factor, area_sum, metric_sum


def _corner_trace_coordinates(material_points,geometry):
    """Original homogeneous trace at exact ray corners, before local rounding.

    Cramer's numerators are exact Fractions, so a true edge incidence is zero
    and a small positive coefficient cannot be erased by a subtractive solve.
    Only positive ray normalization uses Decimal. Agreement of two precisions
    is a numerical guard, not a directed-rounding proof.
    """
    geometry.validate()
    represented=_real(material_points,'material trace points')
    if represented.shape!=(3,3):raise ValueError('A material trace requires one original triangle.')
    p=[tuple(Fraction(float(x)) for x in row) for row in represented]
    dot=lambda a,b:sum((x*y for x,y in zip(a,b)),Fraction(0))
    cross=lambda a,b:(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    normals=(cross(p[1],p[2]),cross(p[2],p[0]),cross(p[0],p[1]))
    determinant=dot(p[0],normals[0])
    if determinant<=0:raise ValueError('Original material trace has nonpositive winding.')
    rays=_piece_rays(geometry)
    coefficients=[[dot(normal,ray)/determinant for normal in normals] for ray in rays]
    if any(value<0 for row in coefficients for value in row):
        raise ValueError('Exact basal corner leaves its original material face.')
    lengths_squared=[dot(ray,ray) for ray in rays]
    previous=None
    for precision in (max(80,geometry.precision),max(80,geometry.precision)+32):
        with localcontext() as context:
            context.prec=precision
            decimal=lambda value:Decimal(value.numerator)/Decimal(value.denominator)
            result=np.array([[float(decimal(v)/decimal(norm).sqrt()) for v in row]
                             for row,norm in zip(coefficients,lengths_squared)])
        if (not np.isfinite(result).all() or any(v!=0 and result[i,j]<=0.
                for i,row in enumerate(coefficients) for j,v in enumerate(row))):
            raise ValueError('Exact positive material trace coefficient has an unresolved binary64 cast.')
        if previous is not None and not np.array_equal(result,previous):
            raise ValueError('Original corner trace normalization did not resolve at both precisions.')
        previous=result
    result.setflags(write=False)
    return result


def _local_trace_rule(system, face, geometry, order, density, beta, *, corner_alpha=None):
    """Integrate the original radial trace in its positive piece's local frame.

    Local coordinates are formed from the exact chart before global rounding.
    The small normal-spin columns and diagonal retain squared transverse
    coordinates; they are never obtained by subtracting nearly equal ones.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    u, v = np.meshgrid(.5*(nodes+1.), .5*(nodes+1.), indexing='ij')
    wu, wv = np.meshgrid(.5*weights, .5*weights, indexing='ij')
    u, v, wu, wv = (x.ravel() for x in (u, v, wu, wv))
    polygon = geometry.local_polygon
    # Material source points are transformed without normalization: the same
    # original homogeneous trace and tangent degrees of freedom are retained.
    points = geometry.localize(system['points'][face])
    basis = geometry.localize(system['tangent_basis'][face].transpose(0, 2, 1)).transpose(0, 2, 1)
    if corner_alpha is None:corner_alpha=_corner_trace_coordinates(system['points'][face],geometry)
    factor = np.empty((0, 9)); area_sum = 0.; metric_sum = np.zeros((3, 3))
    for fan,(b, c) in enumerate(zip(polygon[1:-1], polygon[2:])):
        a = polygon[0]
        determinant = float(Edge(a, b).distances(c[None])[0])
        if determinant <= 0:
            raise ValueError('Every local basal fan must retain positive winding.')
        chord = a+u[:, None]*(b-a)+(1-u)[:, None]*v[:, None]*(c-a)
        length = np.linalg.norm(chord, axis=1)
        q = chord/length[:, None]
        area = system['radius_m']**2*density*wu*wv*(1-u)*determinant/length**3
        # P alpha(q)=q is linear in the unnormalized ray. Interpolating the
        # original exact-corner coefficients avoids solving an almost-edge
        # system independently at every rounded quadrature point.
        weights=np.column_stack(((1-u)*(1-v),u,(1-u)*v))
        alpha=(weights@corner_alpha[[0,fan+1,fan+2]])/length[:,None]
        band = 256*np.finfo(float).eps*np.maximum(1., np.max(abs(alpha), axis=1))
        if (np.any(area <= 0.) or np.any(alpha < -band[:, None])
                or np.max(np.linalg.norm(alpha@points-q, axis=1)) > 256*np.finfo(float).eps):
            raise ValueError('Local basal quadrature has unresolved finite-face trace geometry.')
        trace = np.zeros((len(q), 3, 9))
        trace[:, 0, 1] = q[:, 2]; trace[:, 0, 2] = -q[:, 1]
        trace[:, 1, 0] = -q[:, 2]; trace[:, 1, 2] = q[:, 0]
        trace[:, 2, 0] = q[:, 1]; trace[:, 2, 1] = -q[:, 0]
        rotation = trace[:, :, :3]
        # For unit q, cross(q).T cross(q) equals its tangent projector. This
        # positive construction resolves the normal diagonal qx²+qy².
        projector = np.einsum('nki,nkj->nij', rotation, rotation)
        for corner in range(3):
            trace[:, :, 3+2*corner:5+2*corner] = (
                alpha[:, corner, None, None]*(projector@basis[corner]))
        weighted = (np.sqrt(beta*area)[:, None, None]*trace).reshape(-1, 9)
        _finite(weighted)
        uncompressed = factor.T@factor+weighted.T@weighted
        factor = np.linalg.qr(np.vstack((factor, weighted)), mode='r')
        difference = factor.T@factor-uncompressed
        scale = np.sqrt(np.diag(uncompressed)); denominator = scale[:, None]*scale[None]
        if np.any((denominator == 0.) & (difference != 0.)):
            raise ValueError('Local basal factor compression changed an inactive coordinate.')
        normalized = np.divide(difference, denominator, out=np.zeros_like(difference), where=denominator > 0.)
        if np.max(abs(normalized)) > 256*9*np.finfo(float).eps:
            raise ValueError('Positive local basal factor compression failed its Gram check.')
        area_sum += float(np.sum(area))
        metric_sum += np.einsum('n,nij->ij', area, projector)
    _finite(factor, area_sum, metric_sum)
    return factor, area_sum, metric_sum


def _material_factor(system, face, geometry, density, beta, area, metric, tolerance, orders):
    # Establish containment on the original exact chart and represented
    # material face. Rounded display corners or interior quadrature alone
    # cannot establish that every positive piece is on its declared bottom.
    points = [[Fraction(float(x)) for x in row] for row in system['points'][face]]
    rays = _piece_rays(geometry)
    for a, b in zip(points, points[1:]+points[:1]):
        normal = (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
        if any(sum((x*y for x, y in zip(normal, ray)), Fraction(0)) < 0 for ray in rays):
            raise ValueError('Exact basal chart leaves its declared bottom material face.')
    previous = None; scale = None; records = []
    corner_alpha=_corner_trace_coordinates(system['points'][face],geometry)
    for order in orders:
        factor, measured_area, measured_metric = _local_trace_rule(system, face, geometry, order, density, beta,
            corner_alpha=corner_alpha)
        gram = factor.T@factor
        if scale is None:
            scale = np.sqrt(np.diag(gram))
        if previous is not None:
            denominator = scale[:, None]*scale[None, :]
            delta = gram-previous
            if np.any((denominator == 0.) & (delta != 0.)):
                raise ValueError('Initial basal quadrature missed an active direction.')
            scaled = np.divide(delta, denominator, out=np.zeros_like(delta), where=denominator > 0.)
            estimate = float(np.linalg.norm(scaled, 2))
            area_error = abs(measured_area-area)/area
            metric_error = _metric_error(measured_metric, metric)
            compressed_metric_error = _metric_error(gram[:3, :3]/beta, metric)
            records.append(dict(order=order, operator_relative_change=estimate,
                area_relative_error=area_error, full_rigid_metric_relative_error=metric_error,
                compressed_rigid_metric_relative_error=compressed_metric_error))
            if max(estimate, area_error, metric_error, compressed_metric_error) <= tolerance:
                return factor, records
        previous = gram
    raise RuntimeError('Finite basal quadrature did not converge at the supplied orders.')


def _piece_rays(geometry):
    basis = [[Fraction(float(x)) for x in row] for row in geometry.control_triangle]
    rays = []
    for x, y in geometry.chart_polygon:
        weights = (1-x-y, x, y)
        ray = tuple(sum((weights[j]*basis[j][k] for j in range(3)), Fraction(0)) for k in range(3))
        rays.append(ray)
    return rays


def _piece_key(geometry):
    """Exact oriented rays distinguish chart geometry before display rounding."""
    rays = []
    for ray in _piece_rays(geometry):
        scale = next(abs(value) for value in ray if value)
        rays.append(tuple(value/scale for value in ray))
    return tuple(sorted(rays))


@dataclass(frozen=True)
class FactorGram:
    design: BlockMap

    @property
    def shape(self):
        return (self.design.shape[1],)*2

    def __matmul__(self, value):
        return self.design.rmatvec(self.design@value)

    def to_dense(self):
        if self.shape[0] > 256:
            raise ValueError('Dense conversion is limited to small parity audits.')
        return np.column_stack([self@column for column in np.eye(self.shape[0])])


class FiniteBasal:
    def __init__(self, system, design, reference, diagnostics):
        self.system_signature = _system_signature(system)
        self.basal_design = design
        self.reference_factor_sqrt_w = reference.copy()
        self.hessian_n_s_m = FactorGram(design)
        self.diagonal_n_s_m = _diagonal(design)
        self.load_n = design.rmatvec(reference)
        self.constant_w = .5*float(reference@reference)
        self.diagnostics = tuple(diagnostics)
        _finite(self.diagonal_n_s_m, self.load_n, self.constant_w)
        for value in (design.blocks, design.rows, design.columns, self.reference_factor_sqrt_w,
                      self.diagonal_n_s_m, self.load_n):
            value.setflags(write=False)

    def validate_system(self, system):
        if _system_signature(system) != self.system_signature:
            raise ValueError('Finite basal operator belongs to a different joint geometry.')

    def absolute_action(self, value):
        result = _absolute_action(self.basal_design, _vector(value, self.basal_design.shape[1]))
        _finite(result)
        return result

    def evaluate(self, value):
        y = _vector(value, self.basal_design.shape[1])
        slip = self.basal_design@y-self.reference_factor_sqrt_w
        potential = .5*float(slip@slip)
        gradient = self.basal_design.rmatvec(slip)
        _finite(potential, gradient)
        return potential, gradient

    def work(self, value):
        y = _vector(value, self.basal_design.shape[1])
        speed = self.basal_design@y; reference = self.reference_factor_sqrt_w
        slip = speed-reference
        constituent = _absolute_forward(self.basal_design, y)+abs(reference)
        result = dict(basal_dissipation_w=float(slip@slip),
            basal_reference_input_power_w=-float(slip@reference),
            generalized_gradient_work_w=float(slip@speed), objective_w=.5*float(slip@slip),
            arithmetic_constituent_work_bound_w=float(constituent@constituent))
        _finite(*result.values())
        return result


def build(system, allocations, face_id_to_index, *, owner_to_plate_slot,
          mantle_omega_rad_s, quadrature_relative_tolerance, max_order):
    """Integrate a supplied disjoint set of frozen positive basal allocations.

    ``mantle_omega_rad_s`` is an explicitly specified GLOBAL Euler field, not
    nodal extrapolation. IDs map explicitly to the joint system's face/owner
    slots. Allocation policy, beta and saved-area density come from every
    allocation; no coefficient or policy is inferred here. This function does
    not install anything. Use ``replace_basal`` for a replacement functional.
    """
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='raise'):
            return _build(system, allocations, face_id_to_index, owner_to_plate_slot,
                          mantle_omega_rad_s, quadrature_relative_tolerance, max_order)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        raise ValueError('Finite basal integration exceeds its resolved numerical domain.') from error


def _build(system, allocations, face_ids, owner_ids, mantle, tolerance, max_order):
    radius = _positive(system['radius_m'], 'radius_m')
    owners = _mapping(owner_ids, system['plate_count'], 'owner_to_plate_slot')
    faces = _mapping(face_ids, len(system['faces']), 'face_id_to_index')
    omega = _real(mantle, 'mantle_omega_rad_s')
    if omega.shape != (3,):
        raise ValueError('Mantle reference must be one explicit global Euler vector.')
    mantle_speed = radius*omega
    tolerance = _positive(tolerance, 'quadrature_relative_tolerance')
    if (tolerance >= .01 or isinstance(max_order, (bool, np.bool_))
            or not isinstance(max_order, (int, np.integer))
            or max_order not in (16, 32, 64, 128)):
        raise ValueError('A relative tolerance below .01 and max_order16/32/64/128 are required.')
    orders = tuple(order for order in (8, 16, 32, 64, 128) if order <= max_order)
    blocks = []; columns = []; diagnostics = []; seen = set()
    frames_hi = []; frames_lo = []
    allocations = tuple(allocations)
    if not allocations:
        raise ValueError('At least one explicit basal allocation is required.')
    for cell_index, allocation in enumerate(allocations):
        policy = allocation['allocation_policy']
        if policy not in ('preserve-native', 'resolved-material-bottom'):
            raise ValueError('Basal allocation must retain an explicit reviewed policy.')
        if _positive(allocation['radius_m'], 'allocation radius_m') != radius:
            raise ValueError('Basal allocation and joint sphere radius differ.')
        support = _fractions(allocation['support'], 'Native support')
        density = _positive(allocation['native_measure_density'], 'native_measure_density')
        beta = _positive(allocation['basal_drag_pa_s_per_m'], 'basal_drag_pa_s_per_m')
        cell_area = _positive(allocation['native_cell_area_m2'], 'native_cell_area_m2')
        control_geometry = allocation.get('control_stable_metric')
        if not isinstance(control_geometry, StableMetric):
            raise ValueError('Basal allocation requires a bound stable control metric.')
        control_geometry.validate()
        measure_scale = radius**2*density
        cell_metric = control_geometry.metric_local*measure_scale
        if (abs(control_geometry.area_unit*measure_scale-cell_area) > 512*np.finfo(float).eps*cell_area
                or not np.array_equal(_real(allocation['native_cell_metric_m2'], 'native_cell_metric_m2'), control_geometry.metric_global*measure_scale)):
            raise ValueError('Basal control geometry and saved measure disagree.')
        area_sum = 0.; metric_sum = np.zeros((3, 3))
        for piece_index, piece in enumerate(allocation['pieces']):
            geometry = piece.get('stable_metric')
            if not isinstance(geometry, StableMetric):
                raise ValueError('Basal pieces require bound exact-chart local metrics.')
            geometry.validate()
            if (not np.array_equal(geometry.control_triangle, control_geometry.control_triangle)
                    or tuple(piece['chart_polygon']) != geometry.chart_polygon
                    or not np.array_equal(piece['polygon'], geometry.polygon)):
                raise ValueError('Basal piece geometry and chart provenance disagree.')
            # Exact repeated rays cannot be two disjoint pieces. Global display
            # coordinates are insufficient to distinguish very narrow pieces.
            key = _piece_key(geometry)
            if key in seen:
                raise ValueError('Repeated basal polygon would duplicate physical drag.')
            seen.add(key)
            area = geometry.area_unit*measure_scale
            metric = geometry.metric_local*measure_scale
            saved_area = _positive(piece['area_m2'], 'piece area_m2')
            saved_metric = _real(piece['metric_m2'], 'piece metric_m2')
            if (abs(area-saved_area) > 512*np.finfo(float).eps*area
                    or not np.array_equal(saved_metric, geometry.metric_global*measure_scale)):
                raise ValueError('Basal piece geometry and saved measure disagree.')
            area_sum += area; metric_sum += geometry.metric_in(control_geometry)*measure_scale
            bottom = piece['bottom_face']
            fractions = _fractions(piece['fractions'], 'Piece support')
            if bottom is not None:
                bottom = _identifier(bottom, 'bottom_face')
                owner = _identifier(piece['owner'], 'piece owner')
                if bottom not in faces or owner not in owners:
                    raise ValueError('Basal material ID is absent from explicit system mappings.')
                face_index = faces[bottom]; face = system['faces'][face_index]
                slot = owners[owner]
                if np.any(system['vertex_plate'][face] != slot) or fractions != {owner: 1.}:
                    raise ValueError('Bottom material trace and allocated owner are inconsistent.')
                if policy == 'preserve-native' and (support.get(owner) != 1.
                        or any(weight != 0. for other, weight in support.items() if other != owner)):
                    raise ValueError('Preserve-native bottom material requires pure matching native support.')
                factor, record = _material_factor(system, face, geometry, density, beta, area, metric, tolerance, orders)
                column = np.r_[3*slot+np.arange(3),
                    (system['nplate']+2*face[:, None]+np.arange(2)).ravel()]
                blocks.append(factor); columns.append(column)
                frames_hi.append(geometry.frame_hi); frames_lo.append(geometry.frame_lo)
                diagnostics.append(dict(cell=cell_index, piece=piece_index, kind='bottom-material',
                    bottom_face=bottom, policy=policy, area_m2=area, quadrature=record))
            else:
                if piece['owner'] is not None:
                    raise ValueError('Uncovered basal pieces must not name a material owner.')
                if fractions != support:
                    raise ValueError('Uncovered basal piece must retain its supplied native support.')
                if not math.isclose(math.fsum(fractions.values()), 1., rel_tol=0., abs_tol=64*np.finfo(float).eps):
                    raise ValueError('Uncovered support fractions must sum to one.')
                for owner, fraction in fractions.items():
                    if owner not in owners or not np.isfinite(fraction) or fraction < 0:
                        raise ValueError('Uncovered basal support is invalid or unmapped.')
                    if fraction == 0:
                        continue
                    factor = np.zeros((9, 9))
                    factor[:3, :3] = np.sqrt(beta*fraction)*np.linalg.cholesky(metric).T
                    column = np.r_[3*owners[owner]+np.arange(3), np.zeros(6, dtype=np.int64)]
                    blocks.append(factor); columns.append(column)
                    frames_hi.append(geometry.frame_hi); frames_lo.append(geometry.frame_lo)
                    diagnostics.append(dict(cell=cell_index, piece=piece_index, kind='uncovered-rigid',
                        owner=owner, policy=policy, fraction=fraction, area_m2=area*fraction))
        if (abs(area_sum-cell_area) > 2e-10*cell_area
                or _metric_error(metric_sum, cell_metric) > 2e-10):
            raise ValueError('Supplied basal pieces do not close their native cell measure.')
    size = len(system['load_n']); count = len(blocks)
    design = FramedBlockMap(np.asarray(blocks).reshape(count, 9, 9),
        np.arange(9*count).reshape(count, 9), np.asarray(columns, dtype=np.int64).reshape(count, 9),
        (9*count, size), np.asarray(frames_hi).reshape(count, 3, 3), np.asarray(frames_lo).reshape(count, 3, 3))
    mantle_field = np.zeros(size)
    mantle_field[:system['nplate']] = np.tile(mantle_speed, system['plate_count'])
    return FiniteBasal(system, design, design@mantle_field, diagnostics)


@dataclass(frozen=True)
class _ReplacementGram:
    basal: FactorGram
    strain: BlockMap
    other: np.ndarray

    @property
    def shape(self):
        return self.basal.shape

    def __matmul__(self, value):
        y = _vector(value, self.shape[0])
        result = self.basal@y+self.strain.rmatvec(self.strain@y)
        cut = self.other.shape[1]
        result[:cut] += self.other.T@(self.other@y[:cut])
        _finite(result)
        return result


def replace_basal(system, component, *, other_drag_excludes_allocated_basal):
    """Return a shallow frozen-geometry system with old basal drag REPLACED.

    Retains existing strain, independent other-domain drag and external loads.
    The caller must ensure ``other_plate_drag_factor`` is not ALSO the physical
    complement already supplied in allocations. It may represent only additional
    non-basal resistance. The required declaration records that responsibility;
    the operator cannot independently establish the provenance of a matrix.
    No native solver/callsite is modified.
    """
    if other_drag_excludes_allocated_basal is not True:
        raise ValueError('Explicit exclusion of allocated basal drag from other-domain drag is required.')
    component.validate_system(system)
    if 'finite_basal' in system:
        raise ValueError('A finite basal replacement has already been installed.')
    result = dict(system)
    strain = system['strain_design']; other = system['other_plate_drag_factor']
    operator = _ReplacementGram(component.hessian_n_s_m, strain, other)
    diagonal = component.diagonal_n_s_m+_diagonal(strain)
    diagonal[:system['nplate']] += np.sum(other**2, axis=0)
    external = system['nodal_load_n']+system['torque_load_n']
    result.update(finite_basal=component, hessian_n_s_m=operator,
        hessian_diagonal_n_s_m=diagonal, load_n=external+component.load_n,
        mantle_load_n=component.load_n, objective_constant_w=component.constant_w,
        validate_system=component.validate_system)
    # Obsolete nodal-work routines must fail rather than report zero or doubled
    # basal power on a system whose physical measure has changed.
    result.pop('basal_drag_weight_n_s_m', None)
    result.pop('mantle_velocity_m_s', None)

    def absolute_action(value):
        y = _vector(value, len(result['load_n']))
        action = component.absolute_action(y)+_absolute_action(strain, y)
        action[:system['nplate']] += abs(other).T@(abs(other)@abs(y[:system['nplate']]))
        _finite(action)
        return action

    def physical_work(value):
        y = _vector(value, len(result['load_n']))
        work = component.work(y)
        viscous = float(np.linalg.norm(strain@y)**2)
        other_work = float(np.linalg.norm(other@y[:system['nplate']])**2)
        external_work = float(external@y)
        basal = work['basal_dissipation_w']; mantle = work['basal_reference_input_power_w']
        power = dict(basal_dissipation_w=basal, viscous_dissipation_w=viscous,
            other_drag_dissipation_w=other_work, external_power_w=external_work,
            basal_reference_input_power_w=mantle,
            unsolved_power_defect_w=basal+viscous+other_work-external_work-mantle,
            generalized_gradient_work_w=float(y@(operator@y-result['load_n'])),
            objective_w=.5*(basal+viscous+other_work)-external_work)
        _finite(*power.values())
        return power

    def absolute_work(value):
        y = _vector(value, len(result['load_n']))
        amount = component.work(y)['arithmetic_constituent_work_bound_w']
        amount += float(abs(y)@_absolute_action(strain, y))
        amount += float(np.linalg.norm(abs(other)@abs(y[:system['nplate']]))**2)
        amount += float(abs(external)@abs(y))
        _finite(amount)
        return amount

    result.update(hessian_absolute_action=absolute_action, physical_work=physical_work,
                  physical_work_absolute_scale_w=absolute_work)
    _finite(diagonal, result['load_n'])
    return result
