"""Fixed finite material-pair resistance on the joint SI velocity space.

The caller supplies admitted adjacent interfaces and physical free-edge pieces,
already split wherever either material face changes. This is not an admission,
attachment, rupture or native domain-allocation policy. Quadrature is assembled
once, so every nonlinear evaluation uses one unchanged functional.
"""
import hashlib
import math

import numpy as np

import collision_interface
from convex_partition import Edge
import weld_geometry
from . import shared_contact as joint


def _system_signature(system):
    digest = hashlib.sha256()
    for key in ('points', 'faces', 'vertex_plate', 'tangent_basis', 'radius_m', 'nplate'):
        value = np.asarray(system[key])
        digest.update(key.encode()); digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _pair(system, first, second):
    faces = system['faces']
    if (isinstance(first, (bool, np.bool_)) or isinstance(second, (bool, np.bool_))
            or not isinstance(first, (int, np.integer)) or not isinstance(second, (int, np.integer))
            or not 0 <= first < len(faces) or not 0 <= second < len(faces) or first == second):
        raise ValueError('Two distinct admitted material faces are required.')
    return faces[first], faces[second]


def _point(value):
    p = np.asarray(value, float)
    if p.shape != (3,) or not np.isfinite(p).all() or abs(np.linalg.norm(p)-1.) > 2e-12:
        raise ValueError('Finite unit-sphere geometry is required.')
    return p/np.linalg.norm(p)


def _trace_difference(system, first, second, point):
    """Common-point total velocity, including same-owner residual slip."""
    faces = _pair(system, first, second)
    c = _point(point)
    result = np.zeros((3, len(system['load_n'])))
    projector = np.eye(3)-np.outer(c, c)
    cut = system['nplate']
    for sign, face in zip((1., -1.), faces):
        alpha = joint._trace(system['points'], face, c)
        owner = int(system['vertex_plate'][face[0]])
        result[:, 3*owner:3*owner+3] -= sign*joint._cross_matrix(c)
        for node, coefficient in zip(face, alpha):
            result[:, cut+2*node:cut+2*node+2] += sign*coefficient*projector@system['tangent_basis'][node]
    return result


def _mode_minimum(modes, start, end):
    cuts = [start, end]
    phase = math.atan2(float(modes[1]), float(modes[0]))
    for k in range(math.ceil((start-phase)/math.pi), math.floor((end-phase)/math.pi)+1):
        cuts.append(phase+k*math.pi)
    return min(float(modes@np.array([math.cos(t), math.sin(t)])) for t in cuts)


def weld_arc(system, *, first_face, second_face, basis, tangent, outward_normal,
             start, end, capacity_n_per_m, measure_weight):
    """Exact existing one-sided weld law on one finite paired free-edge arc.

    Positive rate is outward advance of first relative to second (closure);
    negative rate peels first away. ``measure_weight`` is explicit: the native
    perimeter convention assigns half to each represented side. Arc endpoints
    must lie on one true first-face boundary edge, within both finite faces.
    """
    face_a, face_b = _pair(system, first_face, second_face)
    b, t, n = (_point(p) for p in (basis, tangent, outward_normal))
    if max(abs(b@t), abs(b@n), abs(t@n)) > 2e-12:
        raise ValueError('Arc basis/tangent/normal must be mutually orthogonal.')
    if not np.isfinite([start, end]).all() or not 0 < end-start < math.pi:
        raise ValueError('One positive minor finite-face arc is required.')
    capacity = joint._scalar(capacity_n_per_m, 'capacity_n_per_m', positive=False)
    weight = joint._scalar(measure_weight, 'measure_weight')
    modes = np.zeros((2, len(system['load_n'])))
    first_alpha = None
    for sign, face in ((1., face_a), (-1., face_b)):
        alpha = np.linalg.solve(system['points'][face].T, np.column_stack((b, t)))
        band = 256*np.finfo(float).eps*max(1., float(np.max(np.abs(alpha))))
        if any(_mode_minimum(a, start, end) < -band for a in alpha):
            raise ValueError('Whole arc must lie within both supplied finite faces.')
        if sign == 1.:
            first_alpha = alpha
        owner = int(system['vertex_plate'][face[0]])
        modes[:, 3*owner:3*owner+3] += sign*np.cross(np.array([b, t]), n)
        for node, a in zip(face, alpha):
            modes[:, system['nplate']+2*node:system['nplate']+2*node+2] += (
                sign*a[:, None]*(n@system['tangent_basis'][node])[None])
    edge_candidates = np.flatnonzero(np.linalg.norm(first_alpha, axis=1) < 2e-12)
    if len(edge_candidates) != 1 or n@system['points'][face_a].mean(axis=0) >= 0.:
        raise ValueError('Weld requires an outward normal on a first-face edge.')
    edge = np.delete(face_a, edge_candidates[0])
    incidence = np.any(system['faces'] == edge[0], axis=1)&np.any(system['faces'] == edge[1], axis=1)
    if np.count_nonzero(incidence) != 1:
        raise ValueError('An internal material mesh edge is not a physical weld edge.')
    return dict(kind='weld', operator=modes, start=float(start), end=float(end),
                coefficient_n=capacity*weight*system['radius_m'], first_face=int(first_face),
                second_face=int(second_face), capacity_n_per_m=capacity, measure_weight=weight,
                system_signature=_system_signature(system))


def _positive_determinant(triangle):
    # Shared physical predicate uses exact binary64 signs when needed.
    a, b, c = np.asarray(triangle)
    determinant = float(Edge(a, b).distances(c[None])[0])
    if determinant <= 0 or not np.isfinite(determinant):
        raise ValueError('Interface fan must retain resolved positive winding.')
    return determinant


def _interface_rule(system, first, second, polygon, coefficient, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = .5*(nodes+1.), .5*weights
    factors = []; points = []; measures = []
    a = polygon[0]
    for b, c in zip(polygon[1:-1], polygon[2:]):
        determinant = _positive_determinant([a, b, c])
        for u, wu in zip(nodes, weights):
            for v, wv in zip(nodes, weights):
                chord = a+u*(b-a)+(1-u)*v*(c-a)
                length = np.linalg.norm(chord)
                point = chord/length
                area = system['radius_m']**2*wu*wv*(1-u)*determinant/length**3
                if not np.isfinite(area) or area <= 0.:
                    raise ValueError('Interface quadrature must have finite positive area.')
                trace = _trace_difference(system, first, second, point)
                factors.append(np.sqrt(coefficient*area)*trace)
                points.append(point); measures.append(area)
    factor = np.concatenate(factors)
    return factor, factor.T@factor, np.asarray(points), np.asarray(measures)


def _weld_factor(arc, y, delta):
    """Exact positive factors of active trigonometric second moments."""
    operator = arc['operator']; modes = operator@y
    amplitude = float(np.linalg.norm(modes))
    cuts = [arc['start'], arc['end']]
    if amplitude:
        phase = math.atan2(float(modes[1]), float(modes[0]))
        for level in (0., -delta):
            if abs(level) <= amplitude:
                angle = math.acos(float(np.clip(level/amplitude, -1., 1.)))
                for root in (phase-angle, phase+angle):
                    for k in range(math.ceil((arc['start']-root)/(2*math.pi)), math.floor((arc['end']-root)/(2*math.pi))+1):
                        if arc['start'] < root+2*k*math.pi < arc['end']:
                            cuts.append(root+2*k*math.pi)
    cuts = sorted(set(cuts)); rows = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        span = end-start; middle = .5*(start+end)
        direction = np.array([math.cos(middle), math.sin(middle)])
        speed = float(modes@direction)
        if not -delta < speed < 0.:
            continue
        perpendicular = (.5*(span-math.sin(span)) if span >= 1e-3 else
                         span**3*(1/12-span**2/240+span**4/10080))
        for moment, axis in ((.5*(span+math.sin(span)), direction),
                             (perpendicular, np.array([-direction[1], direction[0]]))):
            rows.append(np.sqrt(arc['coefficient_n']*moment/delta)*(axis@operator))
    return np.asarray(rows).reshape(-1, len(y))


def interface_patch(system, *, first_face, second_face, polygon, viscosity_pa_s,
                    thickness_m, relative_tolerance=1e-10, orders=(8, 16, 32, 64)):
    """Converged positive quadrature of the existing finite Couette shear law.

    Native caller must supply an adjacent-layer polygon. Same plate ownership
    does not remove inter-sheet residual slip. Convergence is an a posteriori
    estimate on a fixed energy scaling, not certified integration error.
    """
    _pair(system, first_face, second_face)
    p = np.asarray(polygon, float)
    if p.ndim != 2 or p.shape[1:] != (3,) or len(p) < 3:
        raise ValueError('An admitted finite interface polygon is required.')
    p = np.array([_point(row) for row in p])
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        if np.any(Edge(a, b).distances(p) < 0.):
            raise ValueError('Interface polygon must have convex positive winding.')
    for point in p:
        _trace_difference(system, first_face, second_face, point)
    eta = joint._scalar(viscosity_pa_s, 'interface viscosity_pa_s')
    thickness = joint._scalar(thickness_m, 'interface thickness_m')
    tolerance = joint._scalar(relative_tolerance, 'quadrature relative_tolerance')
    if tolerance >= 1e-2 or len(orders) < 2 or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in orders) or any(b <= a for a, b in zip(orders, orders[1:])):
        raise ValueError('Increasing quadrature orders and a relative tolerance below .01 are required.')
    # Independent analytic rigid-rotation metric also validates convex winding.
    metric = collision_interface.rotation_metric(p, system['radius_m']/1000.)
    exact_area = float(np.trace(metric)/2.)
    signature = _system_signature(system)
    previous = None; fixed_scale = None; records = []
    for order in orders:
        factor, gram, points, measures = _interface_rule(system, first_face, second_face, p, eta/thickness, order)
        if not np.isfinite(gram).all():
            raise ValueError('Interface Gram matrix is nonfinite.')
        if fixed_scale is None:
            fixed_scale = np.sqrt(np.diag(gram))
        denominator = fixed_scale[:, None]*fixed_scale[None]
        if previous is not None:
            difference = gram-previous
            if np.any((denominator == 0.) & (difference != 0.)):
                raise ValueError('Initial interface energy scaling missed an active direction.')
            normalized = np.divide(difference, denominator, out=np.zeros_like(gram), where=denominator > 0.)
            estimate = float(np.linalg.norm(normalized, ord=2))
            area_error = abs(float(measures.sum())-exact_area)/exact_area
            integrated_metric = sum(w*(np.eye(3)-np.outer(c, c)) for w, c in zip(measures, points))
            metric_error = float(np.linalg.norm(integrated_metric-metric, ord=2)/np.linalg.norm(metric, ord=2))
            centroid_axis = _point(np.sum(p, axis=0)/np.linalg.norm(np.sum(p, axis=0)))
            weak_work = float(measures@np.sum(np.cross(centroid_axis, points)**2, axis=1))
            analytic_work = float(centroid_axis@metric@centroid_axis)
            weak_error = abs(weak_work-analytic_work)
            representation_allowance = 64*np.finfo(float).eps*float(np.linalg.norm(metric, ord=2))
            records.append(dict(order=order, operator_relative_change=estimate, area_relative_error=area_error,
                                rigid_metric_relative_error=metric_error, centroid_spin_work_error_m2=weak_error))
            if (estimate <= tolerance and area_error <= tolerance and metric_error <= tolerance
                    and weak_error <= tolerance*weak_work+representation_allowance):
                return dict(kind='interface', factor=factor, hessian_n_s_m=gram,
                    points=points, area_weights_m2=measures, area_m2=exact_area,
                    coefficient_pa_s_m=eta/thickness, first_face=int(first_face), second_face=int(second_face),
                    quadrature=records, quadrature_relative_tolerance=tolerance, system_signature=signature)
        previous = gram
    raise RuntimeError('Finite interface quadrature did not converge at supplied orders.')


class Resistance:
    """One immutable-in-use collection of fixed finite contact functionals."""
    def __init__(self, system, *, welds=(), interfaces=(), smoothing_speed_m_s):
        from copy import deepcopy
        self.size = len(system['load_n'])
        self.welds = deepcopy(tuple(welds)); self.interfaces = deepcopy(tuple(interfaces))
        self.smoothing = joint._scalar(smoothing_speed_m_s, 'smoothing_speed_m_s')
        signature = _system_signature(system)
        self.system_signature = signature
        for row in self.welds:
            if (row.get('kind') != 'weld' or row.get('system_signature') != signature
                    or row['operator'].shape != (2, self.size) or not np.isfinite(row['operator']).all()):
                raise ValueError('Weld operator belongs to a different joint system.')
            joint._scalar(row['coefficient_n'], 'weld coefficient_n', positive=False)
            if not np.isfinite([row['start'], row['end']]).all() or not 0 < row['end']-row['start'] < math.pi:
                raise ValueError('Weld arc interval is invalid.')
        for row in self.interfaces:
            if (row.get('kind') != 'interface' or row.get('system_signature') != signature
                    or row['hessian_n_s_m'].shape != (self.size, self.size)
                    or row['factor'].ndim != 2 or row['factor'].shape[1] != self.size
                    or not np.isfinite(row['hessian_n_s_m']).all() or not np.isfinite(row['factor']).all()):
                raise ValueError('Interface operator belongs to a different joint system.')
            reference = np.abs(row['factor']).T@np.abs(row['factor'])
            if np.any(np.abs(row['factor'].T@row['factor']-row['hessian_n_s_m']) > 64*np.finfo(float).eps*reference):
                raise ValueError('Interface factor and Hessian are inconsistent.')

    def validate_system(self, system):
        if _system_signature(system) != self.system_signature:
            raise ValueError('Finite-contact resistance belongs to a different joint geometry.')

    def evaluate(self, velocity):
        y = np.asarray(velocity, float)
        if y.shape != (self.size,) or not np.isfinite(y).all():
            raise ValueError('Finite aligned joint velocities are required.')
        value = 0.; gradient = np.zeros(self.size); hessian = np.zeros((self.size, self.size)); work = 0.
        details = []; factors = []
        for arc in self.welds:
            phi, g, h, opening = weld_geometry.integrate(arc['operator'], y, arc['start'], arc['end'], self.smoothing)
            coefficient = arc['coefficient_n']
            value += coefficient*phi; gradient += coefficient*g; hessian += coefficient*h
            power = float(coefficient*(g@y)); work += power
            factors.append(_weld_factor(arc, y, self.smoothing))
            details.append(dict(kind='weld', potential_w=float(coefficient*phi), resisting_power_w=power,
                                opening_integral_m_s=float(opening), generalized_reaction_n=-coefficient*g))
        for patch in self.interfaces:
            speed = patch['factor']@y
            g = patch['factor'].T@speed
            power = float(speed@speed)
            value += .5*power; gradient += g; hessian += patch['hessian_n_s_m']; work += power
            factors.append(patch['factor'])
            details.append(dict(kind='interface', potential_w=.5*power, resisting_power_w=power,
                                generalized_reaction_n=-g, area_m2=patch['area_m2'], quadrature=patch['quadrature']))
        if not all(np.isfinite(x).all() for x in (value, gradient, hessian, work)):
            raise RuntimeError('Finite-contact resistance evaluation is nonfinite.')
        return dict(potential_w=float(value), gradient_n=gradient, hessian_n_s_m=hessian,
                    hessian_factor_sqrt_n_s_m=(np.concatenate(factors) if factors else np.empty((0, self.size))),
                    resisting_power_w=float(work), diagnostics=dict(terms=details))
