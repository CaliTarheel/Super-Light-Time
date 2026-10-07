"""Optional exact ordered material operator; NumPy reference remains available.

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
from pathlib import Path
import tempfile
from threading import Lock
import uuid
import numpy as np

try:
    import _material_native_build as build
except ImportError:
    build = None

_REFERENCE = ContextVar('material_reference', default=False)
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
        return dict(name='numpy_ordered_material', version=1)
    return dict(name='native_ordered_material', version=1,
                binary_sha256=build.BINARY_SHA256, source_sha256=build.SOURCE_SHA256,
                compiler=build.COMPILER, flags=build.FLAGS)


def _library():
    global _LIBRARY
    with _LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        payload = base64.b64decode(build.BINARY_BASE64, validate=True)
        if hashlib.sha256(payload).hexdigest() != build.BINARY_SHA256:
            raise ValueError('Native material binary does not match its frozen fingerprint.')
        if hashlib.sha256(build.C_SOURCE.encode('utf-8')).hexdigest() != build.SOURCE_SHA256:
            raise ValueError('Native material source does not match its frozen fingerprint.')
        directory = Path(tempfile.gettempdir())/'DeepTime-material'/build.BINARY_SHA256
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/'material.dll'
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
            raise ValueError('Cached native material DLL has an invalid fingerprint.')
        library = ctypes.CDLL(str(path))
        pointer = ctypes.c_void_p
        library.apply.argtypes = [ctypes.c_int64, ctypes.c_int64]+[pointer]*9
        library.apply.restype = ctypes.c_int
        _LIBRARY = library
        return library


def active():
    """True when the ordered material operator is available."""
    return build is not None and os.name == 'nt' and not _REFERENCE.get()


class _OrderedOperator:
    def __init__(self, arrays):
        self.a, self.b, self.da, self.db, self.transverse, self.stiffness, self.base = arrays
        self.count, self.edges = len(self.base), len(self.a)
        self.library = _library()

    def __call__(self, velocity):
        velocity = np.ascontiguousarray(velocity, np.float64)
        if velocity.shape != (self.count, 3) or not np.isfinite(velocity).all():
            raise ValueError('Material operator needs one finite 3-vector per node.')
        result = np.empty_like(velocity)
        pointer = lambda value: value.ctypes.data_as(ctypes.c_void_p)
        status = self.library.apply(self.count, self.edges, pointer(self.a), pointer(self.b),
            pointer(velocity), pointer(self.da), pointer(self.db), pointer(self.transverse),
            pointer(self.stiffness), pointer(self.base), pointer(result))
        if status:
            raise RuntimeError(f'Native material operator rejected its allocation (status {status}).')
        return result


def prepare_operator(a, b, direction_a, direction_b, transverse, stiffness, base):
    """Freeze this solve's coefficients once; no geometry persists across steps.

    The returned callable computes the raw operator. Tangent/Euler projection,
    convergence, tolerance and all scientific diagnostics remain in Python.
    """
    if not active():
        return None
    base = np.asarray(base)
    n = len(base) if base.ndim == 1 else 0
    a, b = np.asarray(a), np.asarray(b)
    if (base.ndim != 1 or a.ndim != 1 or b.shape != a.shape or
            any(value.size and value.dtype.kind not in 'iu' for value in (a, b)) or
            np.any(a < 0) or np.any(b < 0) or np.any(a >= n) or np.any(b >= n)):
        raise ValueError('Native material operator needs valid integer endpoints and node coefficients.')
    m = len(a)
    vectors = [np.asarray(value) for value in (direction_a, direction_b, transverse)]
    stiffness = np.asarray(stiffness)
    if (any(value.shape != (m, 3) for value in vectors) or stiffness.shape != (m,) or
            not all(np.isfinite(value).all() for value in [*vectors, stiffness, base])):
        raise ValueError('Native material coefficients have invalid shape or nonfinite values.')
    if not n or not m:
        return None  # Keep the reference's signed-zero additions on empty graphs.
    arrays = [np.array(value, dtype=dtype, order='C', copy=True) for value, dtype in
              zip([a, b, *vectors, stiffness, base], [np.int64]*2+[np.float64]*5)]
    for value in arrays:
        value.flags.writeable = False
    return _OrderedOperator(arrays)

