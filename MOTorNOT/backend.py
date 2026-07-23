''' CPU/GPU array-backend selection for MOTorNOT.

    The numeric kernels in MOTorNOT are written to be *array-agnostic*: they
    call `get_array_module(X)` to obtain either NumPy (CPU) or CuPy (GPU)
    depending on where their input arrays live, following the pattern
    recommended by CuPy. As a result the very same code runs on the GPU when it
    is handed CuPy arrays and on the CPU otherwise, with no duplicated logic.

    - On a machine with a working CUDA GPU and CuPy installed, `GPU_AVAILABLE`
      is True and `xp` defaults to CuPy.
    - Otherwise everything falls back to NumPy transparently.

    You can override detection with the environment variable
    `MOTORNOT_BACKEND = cpu | gpu | auto` (default auto), or at runtime with
    `use_cpu()` / `use_gpu()`.

    scipy-based routines (adaptive `solve_ivp`, the elliptic-integral coil
    fields, root finding) run on the CPU regardless; convert with `asnumpy`
    at those boundaries.
'''
import os
import numpy as np

# ---- try to bring up CuPy -------------------------------------------------
_cp = None
_gpu_ok = False


def _probe_gpu():
    global _cp, _gpu_ok
    try:
        import warnings
        with warnings.catch_warnings():
            # CuPy warns about CUDA_PATH when the toolkit comes from pip wheels;
            # headers are still found, so silence this benign message.
            warnings.filterwarnings('ignore', message='.*CUDA path.*')
            import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            return
        # make sure kernels actually compile/run on this machine
        float((cp.arange(4, dtype=cp.float64) * 2).sum())
        _cp = cp
        _gpu_ok = True
    except Exception:
        _cp = None
        _gpu_ok = False


_backend_env = os.environ.get('MOTORNOT_BACKEND', 'auto').lower()
if _backend_env != 'cpu':
    _probe_gpu()

# Whether the GPU may be used right now (can be toggled at runtime).
GPU_AVAILABLE = _gpu_ok
_enabled = _gpu_ok and _backend_env != 'cpu'

# Default array module (used when creating fresh arrays).
xp = _cp if _enabled else np


def use_gpu():
    ''' Route new array creation and dispatch to the GPU (if available). '''
    global _enabled, xp
    if not GPU_AVAILABLE:
        raise RuntimeError('No CUDA GPU / CuPy available on this machine.')
    _enabled = True
    xp = _cp
    return xp


def use_cpu():
    ''' Force all computation onto the CPU (NumPy). '''
    global _enabled, xp
    _enabled = False
    xp = np
    return xp


def using_gpu():
    ''' True if the GPU backend is currently active. '''
    return _enabled


def get_array_module(*arrays):
    ''' Return the array module (CuPy or NumPy) appropriate for `arrays`.

        Mirrors cupy.get_array_module but honours the CPU override: if the GPU
        is disabled, always returns NumPy. If enabled, returns CuPy when any
        argument already lives on the GPU, else NumPy (so CPU inputs stay on
        the CPU without surprise transfers).
    '''
    if _enabled:
        return _cp.get_array_module(*arrays)
    return np


def asarray(a, dtype=None):
    ''' Put `a` on the active backend (GPU array if enabled, else NumPy). '''
    return xp.asarray(a, dtype=dtype)


def asnumpy(a):
    ''' Bring `a` back to the host as a NumPy array (no-op for NumPy input). '''
    if _cp is not None and isinstance(a, _cp.ndarray):
        return _cp.asnumpy(a)
    return np.asarray(a)


def to_backend(a, xp_module, dtype=float):
    ''' Coerce `a` to live in the given array module. '''
    if xp_module is np:
        return np.asarray(asnumpy(a), dtype=dtype)
    return xp_module.asarray(a, dtype=dtype)


def device_name():
    ''' Human-readable name of the active compute device. '''
    if _enabled:
        props = _cp.cuda.runtime.getDeviceProperties(0)
        return props['name'].decode()
    return 'CPU (NumPy)'
