"""Explicit cooperative wall budget, external to physical simulation state.

One outer interval owns the clock. Nested retries/event splits and workers inherit
the same absolute perf_counter deadlines. This is not a hard process watchdog.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
import time

POLICY = 'budgeted_history_1800s_per_myr_v1'
MAX_EVENTS = 512
_active = ContextVar('deep_time_step_budget', default=None)
_clock = time.perf_counter


class BudgetExhausted(BaseException):
    """Abort the transaction; never enter numerical retry/halving handling."""


@dataclass
class StepBudget:
    dt_myr: float
    started: float
    deadline: float
    physical_deadline: float
    force_refinement_deadline: float
    events: list = field(default_factory=list)
    dropped_events: int = 0


def _positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(label + ' must be finite and positive')
    return float(value)


def current():
    return _active.get()


@contextmanager
def step_budget(dt_myr, seconds_per_myr=1800, reserve_seconds=120):
    dt = _positive(dt_myr, 'dt_myr')
    rate = _positive(seconds_per_myr, 'seconds_per_myr')
    reserve = _positive(reserve_seconds, 'reserve_seconds')
    inherited = current()
    if inherited is not None:
        yield inherited
        return
    total = dt * rate
    if not math.isfinite(total):
        raise ValueError('Budget duration overflow')
    # A fractional outer interval receives the same reserve proportion, capped
    # at120s; inner adaptive halves never create a new budget.
    reserve = min(reserve, total / 15.)
    started = _clock()
    state = StepBudget(dt, started, started + total, started + total - reserve,
                       min(started + total * .5, started + total - reserve))
    token = _active.set(state)
    try:
        yield state
    finally:
        _active.reset(token)


def refinement_allowed(stage=None):
    state = current()
    if state is None:
        return True
    limit = (state.force_refinement_deadline if stage in ('force', 'gravity')
             else state.physical_deadline)
    return _clock() < limit


def require_physical_time(stage):
    state = current()
    if state is not None and _clock() >= state.physical_deadline:
        record(stage, status='physical_budget_exhausted')
        raise BudgetExhausted('Physical completion reserve reached: ' + str(stage))


def require_finalization_time(stage):
    state = current()
    if state is not None and _clock() >= state.deadline:
        record(stage, status='whole_step_budget_exhausted')
        raise BudgetExhausted('Whole-step wall budget exhausted: ' + str(stage))


def _scalar(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError('Budget observations must be finite JSON scalars')


def record(stage, **facts):
    state = current()
    if state is None:
        return
    if not isinstance(stage, str) or len(stage) > 160:
        raise ValueError('Invalid budget stage')
    event = {'stage': stage, 'elapsed_seconds': max(0., _clock() - state.started)}
    event.update({str(key): _scalar(value) for key, value in facts.items()})
    if len(state.events) < MAX_EVENTS:
        state.events.append(event)
    else:
        state.dropped_events += 1


def snapshot():
    state = current()
    if state is None:
        return None
    return {'policy': POLICY, 'dt_myr': state.dt_myr, 'started': state.started,
            'deadline': state.deadline, 'physical_deadline': state.physical_deadline,
            'force_refinement_deadline': state.force_refinement_deadline}


def _validate_snapshot(value):
    required = {'policy', 'dt_myr', 'started', 'deadline', 'physical_deadline',
                'force_refinement_deadline'}
    if not isinstance(value, dict) or set(value) != required or value['policy'] != POLICY:
        raise ValueError('Invalid step budget snapshot')
    vals = {key: _positive(value[key], key) for key in required - {'policy'}}
    if not (vals['started'] < vals['force_refinement_deadline'] <=
            vals['physical_deadline'] < vals['deadline']):
        raise ValueError('Invalid absolute deadline order')
    return vals


@contextmanager
def installed(value):
    if value is None:
        if current() is not None:
            raise ValueError('Cannot clear an active budget in a worker scope')
        yield None
        return
    vals = _validate_snapshot(value)
    inherited = current()
    if inherited is not None:
        if snapshot() != value:
            raise ValueError('Cannot replace an active interval budget')
        yield inherited
        return
    state = StepBudget(**vals)
    token = _active.set(state)
    try:
        yield state
    finally:
        _active.reset(token)


def evidence():
    state = current()
    if state is None:
        return None
    return {**snapshot(), 'elapsed_seconds': max(0., _clock() - state.started),
            'remaining_seconds': max(0., state.deadline - _clock()),
            'events': [dict(item) for item in state.events],
            'dropped_events': state.dropped_events,
            'enforcement': 'cooperative; blocking native calls require external watchdog'}


def merge_evidence(value):
    if value is None:
        return
    state = current()
    if state is None or not isinstance(value, dict):
        raise ValueError('Worker evidence requires active matching interval')
    if any(value.get(key) != val for key, val in snapshot().items()):
        raise ValueError('Worker evidence changed interval identity/deadline')
    events = value.get('events')
    dropped = value.get('dropped_events')
    if not isinstance(events, list) or len(events) > MAX_EVENTS or type(dropped) is not int or dropped < 0:
        raise ValueError('Invalid bounded worker evidence')
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get('stage'), str):
            raise ValueError('Invalid worker event')
        copied = {str(key): _scalar(val) for key, val in event.items()}
        if len(state.events) < MAX_EVENTS:
            state.events.append({**copied, 'worker_observation': True})
        else:
            state.dropped_events += 1
    state.dropped_events += dropped
