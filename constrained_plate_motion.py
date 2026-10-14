"""Convex instantaneous plate motion with explicit linear intake constraints.

The constraint is C x >= b in cm/yr-scaled Euler coordinates. Its nonnegative
reaction is an additional source, not extra basal drag or a replacement for
slab pull. The power supplied by it is lambda.Cx (lambda.b at equilibrium).
This module does not alter geometry, history, topology, or saved velocities.

SciPy supplies feasibility and an initial constrained minimizer. An independent
Newton/KKT refinement and acceptance check qualify the result; an optimizer's
success flag alone is insufficient. Nothing is accepted by clipping velocities.
"""
from __future__ import annotations

import numpy as np

import plate_balance as pb

FEASIBILITY_TOLERANCE_CM_YR = 1e-9
FORCE_TOLERANCE = 1e-9


class IncompatibleIntake(ValueError):
    """No rigid motion can satisfy all retained ocean intake constraints."""


def _scipy():
    try:
        from scipy.optimize import linprog, minimize, nnls
    except ImportError as error:
        raise ValueError('Persistent ocean intake requires SciPy; install requirements-intake.txt.') from error
    return linprog, minimize, nnls


def _dual(rows, gradient, gaps, scale):
    """Recover nonnegative reactions only at physically tight constraints."""
    _, _, nnls = _scipy()
    active = np.flatnonzero(gaps <= 1e-7)
    reaction = np.zeros(len(rows))
    if len(active):
        weights, _ = nnls(rows[active].T, gradient/scale, maxiter=10000)
        reaction[active] = weights*scale
    return reaction


def residual(model, gradient, rows, reaction, *, cost=None):
    reference = np.asarray(model._force_reference, float).copy()
    if cost is not None:
        reference += np.abs(cost)
    reference += np.abs(rows).T@reaction
    balanced = gradient-rows.T@reaction
    # Each whole plate and the possible final daughter coordinate must pass.
    result = 0.
    for start in range(0, len(gradient), 3):
        end = min(start+3, len(gradient))
        metric = np.sqrt(np.diag(model.stiffness)[start:end])
        numerator = float(np.linalg.norm(balanced[start:end]/metric))
        denominator = float(np.linalg.norm(reference[start:end]/metric))
        result = max(result, numerator/denominator if denominator else
                     (0. if numerator == 0. else np.inf))
    return result


def solve(model, rows, minimum, *, linear_cost=None, initial=None):
    """Minimize the existing potential, plus optional cost, subject to Cx>=b.

    Returns coordinates, nonnegative reactions, and measured solver evidence.
    The caller decides which established ocean intervals belong in C. A bound
    on a daughter coordinate can be included as another row with b=0.
    """
    linprog, minimize, _ = _scipy()
    rows, minimum = np.asarray(rows, float), np.asarray(minimum, float)
    if (rows.ndim != 2 or rows.shape[1] != model.size or minimum.shape != (len(rows),)
            or not np.isfinite(rows).all() or not np.isfinite(minimum).all()
            or not len(rows) or np.any(np.linalg.norm(rows, axis=1) == 0.)):
        raise ValueError('Intake needs finite nonzero rows and aligned finite bounds.')
    cost = np.zeros(model.size) if linear_cost is None else np.asarray(linear_cost, float)
    if cost.shape != (model.size,) or not np.isfinite(cost).all():
        raise ValueError('The constrained motion cost must be finite and aligned.')
    # Opposed local pulls can have zero net rigid torque. Their finite basal
    # metric still sets a scale when the intake constraint starts from rest.
    scale = max(float(np.linalg.norm(model.torque)), float(np.linalg.norm(model.stiffness)),
                float(np.linalg.norm(cost)), 1.)
    feasible = linprog(np.zeros(model.size), A_ub=-rows, b_ub=-minimum,
                       bounds=[(None, None)]*model.size, method='highs')
    if feasible.status == 2:
        raise IncompatibleIntake('Established ocean intake is incompatible with the current rigid plates; '
                                 'a resolved deformation or plate split is required.')
    if not feasible.success:
        raise ValueError('Intake feasibility was not established: '+feasible.message)
    x = feasible.x
    if initial is not None:
        trial = np.asarray(initial, float)
        if trial.shape == x.shape and np.isfinite(trial).all() and np.all(rows@trial >= minimum):
            x = trial.copy()
    iterations = 0
    for delta in np.asarray(pb.HUBER_CONTINUATION_KM_MYR)*pb.KM_MYR_CM_YR:
        def objective(value):
            options = dict(want_hessian=False) if isinstance(model, pb.Balance) else {}
            potential, gradient, *_ = model._evaluate(value, delta, **options)
            return (potential+cost@value)/scale, (gradient+cost)/scale
        result = minimize(objective, x, jac=True, method='SLSQP', constraints=[dict(
            type='ineq', fun=lambda value: rows@value-minimum, jac=lambda value: rows)],
            options=dict(ftol=1e-13, maxiter=1000))
        iterations += int(result.nit)
        if not result.success:
            raise ValueError('Constrained plate balance failed: '+result.message)
        x = result.x

    delta = pb.HUBER_CONTINUATION_KM_MYR[-1]*pb.KM_MYR_CM_YR
    refinements = 0
    for _ in range(20):
        value, gradient, hessian, *_ = model._evaluate(x, delta)
        gradient = gradient+cost
        gaps = rows@x-minimum
        reaction = _dual(rows, gradient, gaps, scale)
        error = residual(model, gradient, rows, reaction, cost=cost)
        if error <= FORCE_TOLERANCE and float(gaps.min()) >= -FEASIBILITY_TOLERANCE_CM_YR:
            break
        active = np.flatnonzero(reaction > scale*1e-14)
        a = rows[active]
        system = np.block([[hessian/scale, -a.T], [a, np.zeros((len(a), len(a)))]])
        rhs = np.r_[-gradient/scale+a.T@(reaction[active]/scale), -gaps[active]]
        correction, *_ = np.linalg.lstsq(system, rhs, rcond=1e-13)
        step = correction[:model.size]
        accepted = False
        for halving in range(30):
            alpha = .5**halving
            trial = x+alpha*step
            trial_gaps = rows@trial-minimum
            if float(trial_gaps.min()) < -FEASIBILITY_TOLERANCE_CM_YR:
                continue
            trial_value, trial_gradient, *_ = model._evaluate(trial, delta)
            trial_gradient = trial_gradient+cost
            trial_reaction = _dual(rows, trial_gradient, trial_gaps, scale)
            trial_error = residual(model, trial_gradient, rows, trial_reaction, cost=cost)
            roundoff = 64*np.finfo(float).eps*max(abs(value+cost@x), scale)
            if (trial_error < error and trial_value+cost@trial <= value+cost@x+roundoff):
                x, accepted = trial, True
                refinements += 1
                break
        if not accepted:
            raise ValueError(f'Constrained motion KKT refinement failed ({error:g}); no motion accepted.')
    else:
        raise ValueError('Constrained motion exceeded its KKT refinement budget.')
    # Recompute every gate at the returned point, including complementarity.
    gradient = model._evaluate(x, delta)[1]+cost
    gaps = rows@x-minimum
    reaction = _dual(rows, gradient, gaps, scale)
    error = residual(model, gradient, rows, reaction, cost=cost)
    complement = float(np.max(np.abs(reaction*gaps), initial=0.))/max(
        scale, float(np.sum(np.abs(reaction*(rows@x)))), 1.)
    if (not np.isfinite(x).all() or error > FORCE_TOLERANCE or complement > FORCE_TOLERANCE
            or float(gaps.min()) < -FEASIBILITY_TOLERANCE_CM_YR or np.any(reaction < 0.)):
        raise ValueError('Constrained motion failed the independent KKT acceptance gates.')
    if np.any((minimum > 0.) & (rows@x <= 0.)):
        raise ValueError('Positive intake was not resolved above floating-point tolerance.')
    return x, reaction, dict(method='convex constrained Stokes balance',
        iterations=iterations, kkt_refinements=refinements, force_relative_residual=error,
        maximum_intake_violation_cm_yr=max(-float(gaps.min()), 0.),
        complementarity_relative_residual=complement,
        active_constraints=int(np.count_nonzero(reaction)), constraints=len(rows),
        reaction_source_power_w=float(reaction@(rows@x)),
        prescribed_intake_source_power_w=float(reaction@minimum))
