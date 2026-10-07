"""Optional exact ordered native flow loop; reference backend remains available.

The generated Python import contains the C source, compiler provenance and DLL
bytes, so the existing frozen helper-source/checkpoint hashes cover the binary.
Only a verified content-addressed DLL is loaded; an available broken backend
raises instead of silently changing execution policy.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import base64
import ctypes
import hashlib
import os
import operator
from pathlib import Path
import tempfile
from threading import Lock
import uuid
import numpy as np

try:
    import _flow_native_build as build
except ImportError:
    build = None

_REFERENCE = ContextVar('flow_reference', default=False)
_LOCK = Lock()
_LIBRARY = None


@contextmanager
def reference_backend():
    token = _REFERENCE.set(True)
    try:
        yield
    finally:
        _REFERENCE.reset(token)


def provenance():
    if not active():
        return dict(name='python_ordered_dinic', version=1)
    return dict(name='native_ordered_dinic', version=2,
                binary_sha256=build.BINARY_SHA256, source_sha256=build.SOURCE_SHA256,
                compiler=build.COMPILER, flags=build.FLAGS)


def _library():
    global _LIBRARY
    with _LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        payload = base64.b64decode(build.BINARY_BASE64, validate=True)
        if hashlib.sha256(payload).hexdigest() != build.BINARY_SHA256:
            raise ValueError('Native flow binary does not match its frozen fingerprint.')
        if hashlib.sha256(build.C_SOURCE.encode('utf-8')).hexdigest() != build.SOURCE_SHA256:
            raise ValueError('Native flow source does not match its frozen fingerprint.')
        directory = Path(tempfile.gettempdir())/'DeepTime-flow'/build.BINARY_SHA256
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/'flow.dll'
        if not path.exists():
            temporary = directory/(uuid.uuid4().hex+'.tmp')
            try:
                temporary.write_bytes(payload)
                try:
                    temporary.replace(path)
                except PermissionError:
                    if not path.exists() or path.read_bytes() != payload:
                        raise
            finally:
                temporary.unlink(missing_ok=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != build.BINARY_SHA256:
            raise ValueError('Cached native flow DLL has an invalid fingerprint.')
        library = ctypes.CDLL(str(path))
        pointer = ctypes.c_void_p
        library.flow_run.argtypes = [ctypes.c_int64, ctypes.c_int64, pointer, pointer,
            pointer, pointer, ctypes.c_int64, ctypes.c_int64, ctypes.c_double,
            pointer, pointer, pointer]
        library.flow_run.restype = ctypes.c_int
        library.flow_run32.argtypes = [ctypes.c_int32, ctypes.c_int32, pointer, pointer,
            pointer, pointer, ctypes.c_int32, ctypes.c_int32, ctypes.c_double,
            pointer, pointer, pointer]
        library.flow_run32.restype = ctypes.c_int
        _LIBRARY = library
        return library


def active():
    """True when run/run_arrays would execute the native backend."""
    return build is not None and os.name == 'nt' and not _REFERENCE.get()


def run(count, head, to, cap, source, sink, tolerance):
    if not active():
        return None
    offset = np.empty(count+1, np.int64); offset[0] = 0
    np.cumsum([len(edges) for edges in head], out=offset[1:])
    adjacency = np.fromiter((e for edges in head for e in edges), np.int64,
                            count=int(offset[-1]))
    if len(head) != count:
        raise ValueError('Invalid native flow graph or capacities.')
    total, reachable, capacity, counters = run_arrays(
        count, offset, adjacency, np.asarray(to, np.int64), np.asarray(cap, np.float64),
        source, sink, tolerance)
    return total, reachable, capacity.tolist(), counters


def run_arrays(count, offset, adjacency, to, cap, source, sink, tolerance):
    """Native flow on a graph already laid out as _Flow would hold it.

    offset (count+1) and adjacency list each node's edge ids in insertion
    order; edge 2k+1 is the reverse of 2k. cap is copied, never modified.
    Returns (total, reachable, residual capacities, counters).
    """
    if not active():
        raise RuntimeError('Native flow backend is not active.')
    try:
        count, source, sink = map(operator.index, (count, source, sink))
    except TypeError as exc:
        raise ValueError('Native flow needs integer node counts and terminals.') from exc
    if not (0 <= source < count and 0 <= sink < count and source != sink):
        raise ValueError('Native flow needs distinct valid source and sink nodes.')
    # Check dimensions and integer types before conversion; never truncate a
    # malformed floating index or wrap a uint64 value during narrowing.
    indices = [np.asarray(value) for value in (offset, adjacency, to)]
    if any(value.ndim != 1 or (value.size and value.dtype.kind not in 'iu')
           or np.any(value < 0) or np.any(value > np.iinfo(np.int64).max)
           for value in indices) or np.asarray(cap).ndim != 1:
        raise ValueError('Invalid native flow graph or capacities.')
    offset = np.ascontiguousarray(offset, np.int64)
    adjacency = np.ascontiguousarray(adjacency, np.int64)
    destinations = np.ascontiguousarray(to, np.int64)
    capacity = np.array(cap, np.float64, order='C')
    if (len(offset) != count+1 or offset[0] != 0 or np.any(np.diff(offset) < 0) or
            offset[-1] != len(adjacency) or len(destinations) != len(capacity) or
            len(capacity) % 2 or np.any(destinations < 0) or np.any(destinations >= count) or
            np.any(adjacency < 0) or np.any(adjacency >= len(capacity)) or
            not np.isfinite(capacity).all() or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError('Invalid native flow graph or capacities.')
    reachable = np.zeros(count, np.uint8)
    total = np.zeros(1, np.float64); counters = np.zeros(2, np.uint64)
    pointer = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    # All indices are bounded above before casting. Larger graphs retain the
    # original 64-bit kernel. Floating capacities and counters keep their ABI.
    compact = max(count, len(capacity), len(adjacency)) <= np.iinfo(np.int32).max
    kernel = _library().flow_run
    if compact:
        offset, adjacency, destinations = (
            np.ascontiguousarray(value, np.int32) for value in (offset, adjacency, destinations))
        kernel = _library().flow_run32
    status = kernel(count, len(capacity), pointer(offset), pointer(adjacency),
        pointer(destinations), pointer(capacity), source, sink, tolerance,
        pointer(total), pointer(reachable), pointer(counters))
    if status:
        raise RuntimeError(f'Native flow rejected its graph or allocation (status {status}).')
    return float(total[0]), reachable.astype(bool), capacity, counters.tolist()
