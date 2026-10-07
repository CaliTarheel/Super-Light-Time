"""Active processing time and a recent-work ETA for background jobs.

Managers own their locks and lifecycle. Keep this object in memory, merge
``snapshot()`` into status responses, and persist its plain elapsed_seconds.
Recreate it with that elapsed offset when loading a paused job; monotonic clock
readings and rate samples must never be persisted across processes.
"""
from collections import deque
import math
import time


def _nonnegative(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be finite and nonnegative.') from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative.')
    return value


class ProgressTiming:
    """Estimate remaining active seconds from recently completed work units.

    Work may be integration Myr, item counts, or weighted progress from 0 to 1.
    It must be monotonic within one job. The default estimate needs three
    advances at distinct clock times and at least two seconds of active work.
    Setup time is included, then ages out of a target sixty-second window.
    Sparse callbacks retain at least the required advances and their baseline,
    so long integration steps can still produce an estimate.

    A status request includes time since the last callback in the measured
    rate, so a stalled operation cannot count its ETA down to a false zero.
    Pause freezes elapsed time and hides ETA. Resume starts a fresh rate
    window while retaining the accumulated elapsed time and completed work.
    This class is not internally locked; call it under the manager's lock.
    """

    def __init__(self, total_units=1., completed_units=0., elapsed_seconds=0.,
                 running=True, clock=time.perf_counter, window_seconds=60.,
                 min_samples=3, min_seconds=2., max_samples=64):
        self.total = _nonnegative(total_units, 'Total work')
        self.completed = _nonnegative(completed_units, 'Completed work')
        if self.completed > self.total:
            raise ValueError('Completed work cannot exceed total work.')
        self._elapsed = _nonnegative(elapsed_seconds, 'Elapsed seconds')
        self.window_seconds = _nonnegative(window_seconds, 'Rate window')
        self.min_seconds = _nonnegative(min_seconds, 'Minimum sample duration')
        if self.window_seconds <= 0 or self.min_seconds <= 0:
            raise ValueError('Rate window and minimum sample duration must be positive.')
        if (isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1
                or isinstance(max_samples, bool) or not isinstance(max_samples, int)
                or max_samples < min_samples+1):
            raise ValueError('Sample limits must be integers with room for a baseline and the required advances.')
        self.min_samples = min_samples
        self.max_samples = max_samples
        self._clock = clock
        self._last_clock = None
        self._segment_start = self._now()
        self.running = bool(running)
        # All work units may be done while final files are still being saved.
        # Completion is an explicit lifecycle event, never inferred from 100%.
        self.finished = False
        self._samples = deque([(self._elapsed, self.completed)])

    def _now(self):
        try:
            now = float(self._clock())
        except (TypeError, ValueError, OverflowError):
            raise ValueError('Progress clock must return a finite monotonic number.') from None
        if not math.isfinite(now) or (self._last_clock is not None and now < self._last_clock):
            raise ValueError('Progress clock must return a finite monotonic number.')
        self._last_clock = now
        return now

    def _active_elapsed(self, now):
        return self._elapsed + (now-self._segment_start if self.running else 0.)

    def _trim_samples(self, elapsed):
        # Keep one point at/before the window edge to account for the complete
        # operation spanning that edge, plus enough timed advances to estimate
        # sparse work. A step can legitimately take longer than the window.
        cutoff = elapsed-self.window_seconds
        while len(self._samples) > self.min_samples+1 and self._samples[1][0] <= cutoff:
            self._samples.popleft()
        while len(self._samples) > self.max_samples:
            self._samples.popleft()

    def update(self, completed_units):
        """Record completed work, returning current timing fields."""
        completed = _nonnegative(completed_units, 'Completed work')
        if not self.completed <= completed <= self.total:
            raise ValueError('Completed work must advance monotonically without exceeding its total.')
        if not self.running and completed != self.completed:
            raise ValueError('Resume timing before recording more completed work.')
        now = self._now()
        elapsed = self._active_elapsed(now)
        if completed > self.completed:
            # Bursts at the same instant are one timed observation, not enough
            # evidence to bypass warm-up or infer an infinite processing rate.
            if self._samples[-1][0] == elapsed:
                self._samples[-1] = (elapsed, completed)
            else:
                self._samples.append((elapsed, completed))
            self.completed = completed
            self._trim_samples(elapsed)
        return self._snapshot(now)

    def pause(self):
        """Freeze active elapsed time; call after the worker actually stops."""
        now = self._now()
        self._elapsed = self._active_elapsed(now)
        self.running = False
        return self._snapshot(now)

    def resume(self):
        """Resume the active clock and discard rate samples from the old session."""
        now = self._now()
        if not self.running and not self.finished:
            self._segment_start = now
            self.running = True
            self._samples.clear()
            self._samples.append((self._elapsed, self.completed))
        return self._snapshot(now)

    def finish(self):
        """Freeze elapsed time and report zero remaining work."""
        now = self._now()
        self._elapsed = self._active_elapsed(now)
        self.running = False
        self.completed = self.total
        self.finished = True
        return self._snapshot(now)

    def snapshot(self):
        """Return live elapsed_seconds and nullable eta_seconds, JSON-safe."""
        return self._snapshot(self._now())

    def _snapshot(self, now):
        elapsed = self._active_elapsed(now)
        eta = 0. if self.finished else None
        if self.running and not self.finished and self.completed < self.total:
            self._trim_samples(elapsed)
            if len(self._samples)-1 >= self.min_samples:
                first_time, first_work = self._samples[0]
                duration = elapsed-first_time
                work = self.completed-first_work
                if duration >= self.min_seconds and work > 0:
                    estimate = (self.total-self.completed)*duration/work
                    if math.isfinite(estimate):
                        eta = estimate
        return {'elapsed_seconds': round(elapsed, 2),
                'eta_seconds': None if eta is None else round(eta, 2)}
