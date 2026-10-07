"""Finite collision laws condensed onto their actual material-face columns.

The physical laws and quadrature are those of joint_resistance. Only the
representation changes: no per-contact global row or material-sized Hessian.
Admission, adjacency, physical edge multiplicity and native policy stay external.
"""
from dataclasses import dataclass
import math

import numpy as np

import weld_geometry
from . import joint_resistance as laws
from . import shared_contact as joint


def _readonly(value, dtype=float):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _finite(*values):
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError('Finite local collision arithmetic is required.')


@dataclass(frozen=True)
class Weld:
    columns: np.ndarray
    operator: np.ndarray
    coefficient_n: float
    start: float
    end: float
    signature: str


@dataclass(frozen=True)
class Interface:
    columns: np.ndarray
    factor: np.ndarray
    area_m2: float
    signature: str
    quadrature_order: int
    compression_relative_error: float


class Geometry:
    """One detached geometry snapshot, reused for any number of finite pairs."""
    def __init__(self, system):
        self.system = {key: _readonly(system[key], dtype=np.asarray(system[key]).dtype)
                       for key in ('points', 'faces', 'vertex_plate', 'tangent_basis')}
        self.system.update(radius_m=system['radius_m'], nplate=system['nplate'])
        self.signature = laws._system_signature(self.system)
        self.size = len(system['load_n'])
        edges = np.sort(np.concatenate([self.system['faces'][:, [0, 1]],
            self.system['faces'][:, [1, 2]], self.system['faces'][:, [2, 0]]]), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        self.free_edges = frozenset(map(tuple, unique[counts == 1]))

    def _local(self, first, second):
        a, b = laws._pair(self.system, first, second)
        nodes, inverse = np.unique(np.r_[a, b], return_inverse=True)
        owners, owner_inverse = np.unique(self.system['vertex_plate'][nodes], return_inverse=True)
        columns = np.r_[(3*owners[:, None]+np.arange(3)).ravel(),
                        (self.system['nplate']+2*nodes[:, None]+np.arange(2)).ravel()]
        local = dict(points=self.system['points'][nodes], faces=inverse.reshape(2, 3),
            vertex_plate=owner_inverse, tangent_basis=self.system['tangent_basis'][nodes],
            radius_m=self.system['radius_m'], nplate=3*len(owners), load_n=np.zeros(len(columns)))
        return local, _readonly(columns, np.int64)

    def weld_arc(self, **parameters):
        args = dict(parameters)
        first, second = args.pop('first_face'), args.pop('second_face')
        local, columns = self._local(first, second)
        row = laws.weld_arc(local, first_face=0, second_face=1, **args)
        # The tiny descriptor alone cannot distinguish a physical free edge from
        # an internal edge incident on an omitted face. Check full-mesh topology.
        face = self.system['faces'][first]
        alpha = np.linalg.solve(self.system['points'][face].T,
                                np.column_stack((laws._point(args['basis']), laws._point(args['tangent']))))
        candidates = np.flatnonzero(np.linalg.norm(alpha, axis=1) < 2e-12)
        if len(candidates) != 1 or tuple(sorted(np.delete(face, candidates[0]))) not in self.free_edges:
            raise ValueError('An internal material mesh edge is not a physical weld edge.')
        _finite(row['operator'], row['coefficient_n'])
        return Weld(columns, _readonly(row['operator']), row['coefficient_n'],
                    row['start'], row['end'], self.signature)

    def interface_patch(self, **parameters):
        args = dict(parameters)
        first, second = args.pop('first_face'), args.pop('second_face')
        local, columns = self._local(first, second)
        row = laws.interface_patch(local, first_face=0, second_face=1, **args)
        # Thin QR is confined to <=18 columns. It preserves the positive Gram
        # without retaining every quadrature row or clipping small eigenvalues.
        factor = np.linalg.qr(row['factor'], mode='r')
        scale = np.sqrt(np.diag(row['hessian_n_s_m']))
        denominator = np.outer(scale, scale)
        difference = abs(factor.T@factor-row['hessian_n_s_m'])
        if np.any((denominator == 0.) & (difference != 0.)):
            raise ValueError('Local factor compression changed an inactive coordinate.')
        error = np.max(np.divide(difference, denominator, out=np.zeros_like(difference), where=denominator > 0.))
        _finite(factor, error)
        if error > 256*np.finfo(float).eps*max(len(columns), 1):
            raise ValueError('Local positive-factor compression failed its Gram check.')
        return Interface(columns, _readonly(factor), row['area_m2'], self.signature,
                         row['quadrature'][-1]['order'], float(error))


class LocalGram:
    """Sum of F.T F blocks; every block touches at most two material faces."""
    def __init__(self, size, blocks):
        self.shape = (size, size)
        self.blocks = tuple(blocks)
        self.diagonal = np.zeros(size)
        for columns, factor in self.blocks:
            np.add.at(self.diagonal, columns, np.sum(factor*factor, axis=0))
        _finite(self.diagonal)

    def _action(self, value, absolute):
        y = np.asarray(value, float)
        if y.shape != (self.shape[0],):
            raise ValueError('Aligned joint velocity required.')
        _finite(y)
        result = np.zeros_like(y)
        for columns, factor in self.blocks:
            if absolute:
                factor = abs(factor)
            np.add.at(result, columns, factor.T@(factor@(abs(y[columns]) if absolute else y[columns])))
        _finite(result)
        return result

    def __matmul__(self, value):
        return self._action(value, False)

    def absolute_matvec(self, value):
        return self._action(value, True)


class Resistance:
    """Fixed finite-contact functional with geometry binding and local reactions."""
    def __init__(self, system, *, welds=(), interfaces=(), smoothing_speed_m_s):
        self.size = len(system['load_n'])
        self.signature = laws._system_signature(system)
        self.smoothing = joint._scalar(smoothing_speed_m_s, 'smoothing_speed_m_s')
        self.welds = tuple(Weld(_readonly(row.columns, row.columns.dtype), _readonly(row.operator),
            row.coefficient_n, row.start, row.end, row.signature) for row in welds)
        self.interfaces = tuple(Interface(_readonly(row.columns, row.columns.dtype), _readonly(row.factor),
            row.area_m2, row.signature, row.quadrature_order, row.compression_relative_error) for row in interfaces)
        for row in self.welds+self.interfaces:
            if (row.signature != self.signature or row.columns.ndim != 1
                    or row.columns.dtype.kind not in 'iu' or len(row.columns) > 18
                    or len(np.unique(row.columns)) != len(row.columns)
                    or np.any(row.columns < 0) or np.any(row.columns >= self.size)):
                raise ValueError('Local collision record belongs to a different joint geometry.')
        for row in self.welds:
            if (row.operator.shape != (2, len(row.columns)) or row.coefficient_n < 0.
                    or not 0. < row.end-row.start < math.pi):
                raise ValueError('Invalid finite weld record.')
            _finite(row.operator, row.coefficient_n, row.start, row.end)
        for row in self.interfaces:
            if row.factor.ndim != 2 or row.factor.shape[1] != len(row.columns) or row.area_m2 <= 0.:
                raise ValueError('Invalid finite interface record.')
            _finite(row.factor, row.area_m2)

    def validate_system(self, system):
        if laws._system_signature(system) != self.signature:
            raise ValueError('Finite-contact resistance belongs to a different joint geometry.')

    def evaluate(self, velocity):
        y = np.asarray(velocity, float)
        if y.shape != (self.size,):
            raise ValueError('Aligned joint velocities required.')
        _finite(y)
        gradient = np.zeros(self.size); absolute_gradient = np.zeros(self.size)
        value = 0.; power = 0.; blocks = []; details = []
        for row in self.welds:
            local_y = y[row.columns]; modes = row.operator@local_y
            phi, g, _, opening = weld_geometry.integrate(np.eye(2), modes, row.start, row.end, self.smoothing)
            local_g = row.operator.T@(row.coefficient_n*g)
            np.add.at(gradient, row.columns, local_g)
            np.add.at(absolute_gradient, row.columns, abs(row.operator).T@abs(row.coefficient_n*g))
            p = float(modes@(row.coefficient_n*g))
            value += row.coefficient_n*phi; power += p
            arc = dict(operator=row.operator, coefficient_n=row.coefficient_n, start=row.start, end=row.end)
            blocks.append((row.columns, laws._weld_factor(arc, local_y, self.smoothing)))
            details.append(dict(kind='weld', resisting_power_w=p, opening_integral_m_s=float(opening)))
        for row in self.interfaces:
            speed = row.factor@y[row.columns]; local_g = row.factor.T@speed
            np.add.at(gradient, row.columns, local_g)
            # Include cancellation in BOTH factor actions. Using |F y| alone
            # erases the products that computed a nearly null modal speed.
            np.add.at(absolute_gradient, row.columns,
                      abs(row.factor).T@(abs(row.factor)@abs(y[row.columns])))
            p = float(speed@speed); power += p; value += .5*p
            blocks.append((row.columns, row.factor))
            details.append(dict(kind='interface', resisting_power_w=p, area_m2=row.area_m2,
                                quadrature_order=row.quadrature_order))
        hessian = LocalGram(self.size, blocks)
        _finite(value, power, gradient, absolute_gradient)
        return dict(potential_w=float(value), gradient_n=gradient,
            gradient_absolute_scale_n=absolute_gradient, resisting_power_w=float(power),
            hessian_n_s_m=hessian, diagonal_n_s_m=hessian.diagonal,
            diagnostics=dict(terms=details))
