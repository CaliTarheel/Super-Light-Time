"""Paid work along the force-rift model's existing necking path.

The intact cut capacities are already selected by ``plate_limit_analysis``.
This module neither picks a cut nor supplies motion or a force coefficient.
For a frozen mode, normal/shear displacement per unit scalar opening gives a
generalized intact capacity C in newtons. Existing force-rift settings weaken
that capacity to C/(1 + opening/width). Its exact positive resisting work is
C*width*log((width + opening + increment)/(width + opening)).

All coordinates here are metres, capacities newtons, and work joules. The
caller must separately establish physical loading/yield, account for viscous
and other resisting work, and preserve an irreversible cumulative paid-work
ledger. Healing of the existing opening coordinate cannot refund past work.
The passive-margin experiment's principle of charging strength loss inside
resisting work is retained; its different linear critical-slip law, residual
friction, geometry and startup source transactions are not imported.
"""

from __future__ import annotations

import math
import numbers


def _scalar(value, name, *, nonnegative=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f'{name} must be a finite real number in the stated SI units.')
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite real number in the stated SI units.') from exc
    if (not math.isfinite(result) or (nonnegative and result < 0.)
            or (positive and result <= 0.)):
        qualifier = 'positive ' if positive else 'nonnegative ' if nonnegative else ''
        raise ValueError(f'{name} must be a finite {qualifier}real number in the stated SI units.')
    return result


def _finite(value, name):
    if not math.isfinite(value):
        raise ValueError(f'{name} exceeds the finite SI arithmetic range.')
    return value


def _coordinate(opening_m, width_m):
    opening = _scalar(opening_m, 'Opening', nonnegative=True)
    width = _scalar(width_m, 'Rift width', positive=True)
    return opening, width, _finite(width + opening, 'Necking coordinate')


def mixed_capacity_n(length_m, strengths_n_per_m, normal_rate, shear_rate):
    """Intact generalized capacity of one edge in a fixed mixed mode.

    ``strengths_n_per_m`` contains tensile, compressive and shear strengths.
    Positive normal displacement opens; negative normal displacement closes.
    Both rates are metres displacement per metre scalar mode increment.
    Signed shear has the same resisting cost in either direction. Summing this
    function over the actual cut gives its intact generalized capacity.
    """
    length = _scalar(length_m, 'Edge length', nonnegative=True)
    try:
        strengths = tuple(strengths_n_per_m)
    except TypeError as exc:
        raise ValueError('Cut strength requires tension, compression and shear capacities.') from exc
    if len(strengths) != 3:
        raise ValueError('Cut strength requires tension, compression and shear capacities.')
    tension, compression, shear = (
        _scalar(v, 'Cut strength', nonnegative=True) for v in strengths)
    normal = _scalar(normal_rate, 'Normal mode rate')
    tangent = _scalar(shear_rate, 'Shear mode rate')
    terms = (tension * max(normal, 0.), compression * max(-normal, 0.),
             shear * abs(tangent))
    for term in terms:
        _finite(term, 'Mixed cut capacity')
    try:
        result = length * math.fsum(terms)
    except OverflowError as exc:
        raise ValueError('Mixed cut capacity exceeds the finite SI arithmetic range.') from exc
    return _finite(result, 'Mixed cut capacity')


def necking_multiplier(opening_m, width_m):
    """The existing S/S0 = 1/(1 + opening/width), with finite SI inputs."""
    _, width, base = _coordinate(opening_m, width_m)
    return width / base


def necking_force_n(capacity_n, opening_m, width_m):
    """Derivative of cumulative resisting work with respect to opening."""
    capacity = _scalar(capacity_n, 'Intact generalized capacity', nonnegative=True)
    return capacity * necking_multiplier(opening_m, width_m)


def necking_work_j(capacity_n, opening_m, increment_m, width_m):
    """Exact positive cut resistance charged once over an accepted increment.

    This is total resistance along the existing necking law, not an additional
    fracture surcharge to add to a plastic cut resistance already charged.
    The frozen cut and mode rates must remain the same over the increment.
    """
    capacity = _scalar(capacity_n, 'Intact generalized capacity', nonnegative=True)
    increment = _scalar(increment_m, 'Opening increment', nonnegative=True)
    _, width, base = _coordinate(opening_m, width_m)
    if capacity == 0. or increment == 0.:
        return 0.
    # Multiplying width by log1p first avoids an unnecessary C*width overflow
    # and preserves very short accepted increments against a large opening.
    distance = width * math.log1p(increment / base)
    return _finite(capacity * distance, 'Paid cut work')


def increment_for_work_m(capacity_n, opening_m, width_m, available_work_j,
                         maximum_increment_m):
    """Largest finite increment whose cut resistance fits an available budget.

    This bounds a caller's proposed increment. It does not establish a force,
    a yield crossing, a time-dependent trajectory, or a complete energy
    balance. A zero-capacity cut has zero cost; separate caller gates still
    decide whether anything moves. Returned work never exceeds the supplied
    budget under this module's same floating-point work evaluation.
    """
    capacity = _scalar(capacity_n, 'Intact generalized capacity', nonnegative=True)
    budget = _scalar(available_work_j, 'Available cut work', nonnegative=True)
    limit = _scalar(maximum_increment_m, 'Maximum opening increment', nonnegative=True)
    _, width, base = _coordinate(opening_m, width_m)
    if limit == 0.:
        return 0.
    full_work = necking_work_j(capacity, opening_m, limit, width)
    if budget >= full_work:
        return limit
    if budget == 0.:
        return 0.
    # Use the analytical inverse as a first trial, then bracket rounding so
    # the accepted cut cannot spend a fraction more work than is available.
    exponent = (budget / capacity) / width
    try:
        trial = base * math.expm1(exponent)
    except OverflowError:
        trial = limit
    if not math.isfinite(trial):
        trial = limit
    trial = min(max(trial, 0.), limit)
    trial_work = necking_work_j(capacity, opening_m, trial, width)
    low, high = (trial, limit) if trial_work <= budget else (0., trial)
    # A fixed finite bound also handles an underflowed analytical inverse.
    # Ordinary model SI ranges converge to adjacent floats much sooner.
    for _ in range(1076):
        middle = low + (high - low) * .5
        if middle == low or middle == high:
            break
        if necking_work_j(capacity, opening_m, middle, width) <= budget:
            low = middle
        else:
            high = middle
    return low
