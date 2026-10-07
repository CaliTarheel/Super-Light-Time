"""Runtime worker limits for one ordered batch on a caller-owned executor.

Only dispatch policy changes live. Lowering the limit never interrupts an
already submitted job. The caller owns executor creation, process/thread count,
input immutability and physical-state commits; workers should return proposals.
"""
from __future__ import annotations

from concurrent.futures import Executor, Future
from threading import Condition
from typing import Callable, Iterable, TypeVar
import budget

T = TypeVar('T')
R = TypeVar('R')
_EMPTY = object()
_POLL_SECONDS = 0.05


def _integer(value: int, label: str, maximum: int | None = None) -> int:
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        upper = f' through {maximum}' if maximum is not None else ' or greater'
        raise ValueError(f'{label} must be an integer from 1{upper}.')
    return value


class WorkerPolicy:
    """Shared dispatch budget. The default is two workers with a six-worker cap.

    ``in_flight`` counts submitted jobs whose completion callback has not yet
    run, including jobs queued in the executor. Revision changes only when the
    requested limit changes. One batch at a time may use a policy.
    """

    def __init__(self, max_workers: int = 6, initial_workers: int = 2):
        self._maximum = _integer(max_workers, 'max_workers')
        self._requested = _integer(initial_workers, 'initial_workers', self._maximum)
        self._in_flight = 0
        self._revision = 0
        self._batch_active = False
        self._condition = Condition()

    def _status_locked(self) -> dict[str, int]:
        return dict(max_workers=self._maximum, requested_workers=self._requested,
                    in_flight=self._in_flight, revision=self._revision)

    def status(self) -> dict[str, int]:
        with self._condition:
            return self._status_locked()

    def set_limit(self, n: int) -> dict[str, int]:
        n = _integer(n, 'Worker limit', self._maximum)
        with self._condition:
            if n != self._requested:
                self._requested = n
                self._revision += 1
                self._condition.notify_all()
            return self._status_locked()

    def _completed(self, future: Future) -> None:
        with self._condition:
            self._in_flight -= 1
            self._condition.notify_all()


def bounded_ordered_map(executor: Executor, fn: Callable[[T], R],
                        items: Iterable[T], policy: WorkerPolicy) -> list[R]:
    """Return results in input order, honoring a limit that can change live.

    Dispatch is bounded; the result list itself grows with the input. Lowering
    a limit permits previously submitted work to drain before new submissions.
    Increasing wakes dispatch immediately, with a 50-ms fallback poll. Actual
    parallelism is also limited by the injected executor's capacity.

    On any error, cancel this batch's queued jobs and join its running jobs
    before raising. Jobs must terminate cooperatively: Python cannot forcibly
    stop an arbitrary running future. The executor is never shut down here.
    """
    with policy._condition:
        if policy._batch_active:
            raise RuntimeError('Worker policy already has an owning batch.')
        policy._batch_active = True
    pending: dict[Future, int] = {}
    results: list[R] = []
    buffered = _EMPTY
    exhausted = False
    try:
        iterator = iter(items)
        while True:
            budget.require_physical_time('worker_batch')
            # Collect all completed jobs, including failures behind a slow
            # earlier job, before admitting another group of submissions.
            for future, index in tuple(pending.items()):
                if future.done():
                    results[index] = future.result()
                    del pending[future]
            if exhausted and not pending:
                return results

            while not exhausted:
                with policy._condition:
                    capacity = len(pending) < policy._requested
                if any(future.done() for future in pending):
                    break
                if not capacity:
                    break
                if buffered is _EMPTY:
                    try:
                        buffered = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                # The iterator runs outside the lock so slow input generation
                # cannot block chat/API updates to the requested limit.
                with policy._condition:
                    if len(pending) >= policy._requested:
                        break
                    budget.require_physical_time('worker_dispatch')
                    future = executor.submit(fn, buffered)
                    pending[future] = len(results)
                    results.append(None)  # Replaced by the corresponding result.
                    buffered = _EMPTY
                    policy._in_flight += 1
                    future.add_done_callback(policy._completed)
                # Do not fill a free slot past an already visible failure.
                if future.done():
                    break

            with policy._condition:
                if pending and not any(future.done() for future in pending):
                    policy._condition.wait(timeout=_POLL_SECONDS)
    except BaseException:
        # Never cancel or wait on unrelated jobs sharing the supplied executor.
        for future in pending:
            future.cancel()
        for future in pending:
            try:
                future.result()
            except BaseException:
                pass
        raise
    finally:
        with policy._condition:
            # Future.result() may wake before its callbacks have returned.
            while policy._in_flight:
                policy._condition.wait(timeout=_POLL_SECONDS)
            policy._batch_active = False
            policy._condition.notify_all()
