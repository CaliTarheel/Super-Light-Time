"""Performance-only reusable worker service; no simulation state is retained.

Only the thread inside ``activate`` dispatches work. Processes receive immutable
stage inputs and return ordered proposals; all physical commits stay in the
simulation thread. A failed task drains its batch and propagates to the caller.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import ctypes
import multiprocessing
from multiprocessing import shared_memory
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock, local

from runtime_workers import WorkerPolicy, bounded_ordered_map
import budget

_ACTIVE = local()
_BLAS_VARIABLES = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                   'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')


def configure_worker_environment():
    """Spawned interpreters inherit these before importing NumPy."""
    for name in _BLAS_VARIABLES:
        os.environ[name] = '1'


def physical_core_count():
    """Detect physical cores without treating SMT threads as extra cores.

    Unknown hosts conservatively use one worker. The six-core development host
    is detected by Windows topology rather than a hard-coded machine identity.
    """
    try:
        if os.name == 'nt':
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            query = kernel.GetLogicalProcessorInformationEx
            query.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
            query.restype = ctypes.c_int
            size = ctypes.c_ulong()
            query(0, None, ctypes.byref(size))  # RelationProcessorCore
            if not size.value:
                raise OSError('Processor topology size unavailable.')
            buffer = ctypes.create_string_buffer(size.value)
            if not query(0, buffer, ctypes.byref(size)):
                raise OSError(ctypes.get_last_error(), 'Processor topology unavailable.')
            offset = count = 0
            while offset < size.value:
                relation = ctypes.c_ulong.from_buffer(buffer, offset).value
                length = ctypes.c_ulong.from_buffer(buffer, offset+4).value
                if length < 8 or offset+length > size.value:
                    raise ValueError('Malformed processor topology record.')
                count += relation == 0
                offset += length
        elif sys.platform == 'darwin':
            count = int(subprocess.check_output(['sysctl', '-n', 'hw.physicalcpu'], text=True).strip())
        else:
            roots = Path('/sys/devices/system/cpu').glob('cpu[0-9]*/topology')
            allowed = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else None
            cores = set()
            for root in roots:
                if allowed is not None and int(root.parent.name[3:]) not in allowed:
                    continue
                cores.add(((root/'physical_package_id').read_text().strip(),
                           (root/'core_id').read_text().strip()))
            count = len(cores)
        if count < 1:
            raise ValueError('No physical cores detected.')
        return min(int(count), int(os.cpu_count() or count))
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        return 1


def initialize_worker():
    configure_worker_environment()
    sys.dont_write_bytecode = True
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise OSError(ctypes.get_last_error(), 'Cannot set worker below-normal priority.')
    elif hasattr(os, 'nice'):
        os.nice(5)


def worker_probe(delay=0.):
    """Read-only process diagnostic used by runtime verification."""
    import time
    if delay:
        time.sleep(delay)
    priority = None
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetPriorityClass.argtypes = [ctypes.c_void_p]
        priority = int(kernel.GetPriorityClass(kernel.GetCurrentProcess()))
    return dict(pid=os.getpid(), blas_environment={key: os.environ.get(key) for key in _BLAS_VARIABLES},
                priority_class=priority)


def _budgeted_worker_call(payload):
    function, item, inherited = payload
    with budget.installed(inherited):
        budget.require_physical_time('worker_entry')
        try:
            result = function(item)
        except BaseException as error:
            try:
                error.runtime_budget_evidence = budget.evidence()
            except (AttributeError, TypeError):
                pass
            raise
        budget.require_physical_time('worker_return')
        return result, budget.evidence()


class RuntimePool:
    def __init__(self, workers=None, *, max_workers=None):
        detected = physical_core_count()
        # CPython's Windows process executor supports at most 61 children.
        supported = min(detected,61) if os.name == 'nt' else detected
        maximum = supported if max_workers is None else max_workers
        if type(maximum) is not int or not 1 <= maximum <= supported:
            raise ValueError(f'max_workers must be an integer from 1 through {supported}.')
        self.policy = WorkerPolicy(max_workers=maximum,
                                   initial_workers=min(2, maximum) if workers is None else workers)
        self._pool = None
        self._lock = Lock()
        self._closed = False
        self._batches = 0
        configure_worker_environment()

    def status(self):
        return dict(self.policy.status(), pool_started=self._pool is not None,
                    pool_closed=self._closed, completed_batches=self._batches,
                    physical_cores=physical_core_count())

    @contextmanager
    def activate(self):
        previous = getattr(_ACTIVE, 'runtime', None)
        if previous is not None and previous is not self:
            raise RuntimeError('A different parallel runtime already owns this thread.')
        if self._closed:
            raise RuntimeError('Parallel runtime is closed.')
        _ACTIVE.runtime = self
        try:
            yield self
        finally:
            _ACTIVE.runtime = previous

    def map(self, fn, items):
        with self._lock:
            if self._closed:
                raise RuntimeError('Parallel runtime is closed.')
            if self._pool is None:
                self._pool = ProcessPoolExecutor(max_workers=self.policy.status()['max_workers'],
                    mp_context=multiprocessing.get_context('spawn'), initializer=initialize_worker)
            pool = self._pool
        inherited = budget.snapshot()
        if inherited is None:
            result = bounded_ordered_map(pool, fn, items, self.policy)
        else:
            try:
                completed = bounded_ordered_map(pool, _budgeted_worker_call,
                    ((fn, item, inherited) for item in items), self.policy)
            except BaseException as error:
                if hasattr(error, 'runtime_budget_evidence'):
                    budget.merge_evidence(error.runtime_budget_evidence)
                raise
            result = []
            for value, report in completed:
                budget.merge_evidence(report)
                result.append(value)
        self._batches += 1
        return result

    def close(self):
        with self._lock:
            self._closed = True
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        self._activation = self.activate()
        self._activation.__enter__()
        return self

    def __exit__(self, *args):
        self._activation.__exit__(*args)
        self.close()


def active_runtime():
    return getattr(_ACTIVE, 'runtime', None)


@dataclass(frozen=True)
class SharedArray:
    name: str
    shape: tuple
    dtype: str


@contextmanager
def share_inputs(value):
    """Publish one read-only stage input tree; unlink after every job drains."""
    import numpy as np
    blocks = []
    def pack(item):
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            if array.dtype.hasobject or array.dtype.fields is not None:
                raise ValueError('Worker inputs require plain numerical arrays, not object or structured arrays.')
            block = shared_memory.SharedMemory(create=True, size=max(1, array.nbytes))
            blocks.append(block)
            np.ndarray(array.shape, dtype=array.dtype, buffer=block.buf)[...] = array
            return SharedArray(block.name, array.shape, array.dtype.str)
        if isinstance(item, dict):
            return {key: pack(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return tuple(pack(child) for child in item)
        if isinstance(item, list):
            return [pack(child) for child in item]
        return item
    try:
        yield pack(value)
    finally:
        for block in reversed(blocks):
            block.close()
            block.unlink()


@contextmanager
def read_inputs(value):
    """Attach a shared stage tree in a worker; workers cannot mutate arrays."""
    import numpy as np
    blocks = []
    def unpack(item):
        if isinstance(item, SharedArray):
            block = shared_memory.SharedMemory(name=item.name)
            blocks.append(block)
            array = np.ndarray(item.shape, dtype=np.dtype(item.dtype), buffer=block.buf)
            array.flags.writeable = False
            return array
        if isinstance(item, dict):
            return {key: unpack(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return tuple(unpack(child) for child in item)
        if isinstance(item, list):
            return [unpack(child) for child in item]
        return item
    try:
        yield unpack(value)
    finally:
        for block in reversed(blocks):
            block.close()
