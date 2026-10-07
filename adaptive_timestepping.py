"""Transactional timestep refinement for explicitly reviewed native worlds.

A rejected contact update must not leave rigid motion, RNG draws or any other
process ahead of its material deformation. Retry the whole coupled step, never
just the failed operator. Physical acceptance criteria remain unchanged.
"""
from copy import deepcopy
import math
import budget

from deforming_regions import IncompleteContactStepError

MIN_DT_MYR = 0.001
MAX_DEPTH = 10
MAX_TRIALS = 128


def advance(simulation, dt, raw_step):
    dt = float(dt)
    if not math.isfinite(dt) or not 0 < dt <= 5:
        raise ValueError('Step length must be positive and at most 5 Myr.')
    # Event-boundary splitting inside the native step is part of this transaction.
    if getattr(simulation, '_adaptive_step_active', False):
        return raw_step(dt)
    budget.require_physical_time('adaptive_begin')
    before = deepcopy(vars(simulation))
    start = float(simulation.t)
    diagnostics = dict(version=1, requested_dt_myr=dt, completed_dt_myr=0.,
                       accepted_substeps=0, rejected_trials=0, minimum_dt_myr=dt,
                       last_rejection=None,
                       substep_scope='Accepted coupled retry intervals; internal event-boundary splits may be smaller.')
    trials = 0
    simulation._adaptive_step_active = True

    def restore(state):
        simulation.__dict__.clear()
        simulation.__dict__.update(state)

    def attempt(interval, depth):
        nonlocal trials
        budget.require_physical_time('adaptive_trial')
        budget.record('adaptive_trial', event='entry', dt_myr=interval, depth=depth, source_time_myr=float(simulation.t))
        trials += 1
        if trials > MAX_TRIALS:
            raise IncompleteContactStepError('Coupled timestep exceeded its bounded retry budget.')
        checkpoint = deepcopy(vars(simulation))
        old_time = float(simulation.t)
        try:
            raw_step(interval)
        except IncompleteContactStepError as error:
            restore(checkpoint)
            diagnostics['rejected_trials'] += 1
            diagnostics['last_rejection'] = str(error)
            budget.record('adaptive_trial', event='rejected', dt_myr=interval, depth=depth, message=str(error))
            if depth >= MAX_DEPTH or interval / 2 < MIN_DT_MYR:
                raise IncompleteContactStepError(
                    f'Complete contact response remains unavailable at {interval:g} Myr; '
                    'the entire requested timestep has been rolled back. ' + str(error)) from error
            attempt(interval / 2, depth + 1)
            attempt(interval / 2, depth + 1)
            return
        elapsed = float(simulation.t) - old_time
        if not math.isclose(elapsed, interval, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError('A coupled timestep returned without completing its requested physical interval.')
        budget.require_physical_time('adaptive_trial:completed')
        budget.record('adaptive_trial', event='returned', dt_myr=interval, destination_time_myr=float(simulation.t))
        diagnostics['accepted_substeps'] += 1
        diagnostics['minimum_dt_myr'] = min(diagnostics['minimum_dt_myr'], interval)

    try:
        attempt(dt, 0)
        diagnostics['completed_dt_myr'] = float(simulation.t) - start
        if not math.isclose(diagnostics['completed_dt_myr'], dt, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError('Adaptive timestepping did not complete the requested physical interval.')
        simulation.timestep_diagnostics = diagnostics
    except BaseException:
        # Preserve the last complete epoch even after a later half-step fails.
        restore(before)
        raise
    finally:
        simulation.__dict__.pop('_adaptive_step_active', None)
