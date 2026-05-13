"""Compact storage for LPD NN data.

On disk:
  data/lpd/nn.joblib   — the shared NearestNeighbors model (saved once)
  data/lpd/loops.npz   — zlib-compressed loop arrays, one entry per instrument

Callers refer to instruments by name: "drums", "piano", "guitar", "bass".
"""
import os
import joblib
import numpy as np

_DIR = "data/lpd"
_NN_PATH = os.path.join(_DIR, "nn.joblib")
_LOOPS_PATH = os.path.join(_DIR, "loops.npz")

INSTRUMENTS = ("drums", "piano", "guitar", "bass")


def _check_instrument(name):
    if name not in INSTRUMENTS:
        raise ValueError(
            f"Unknown instrument {name!r}; expected one of {INSTRUMENTS}"
        )


def _read_loops():
    if not os.path.exists(_LOOPS_PATH):
        return {}
    with np.load(_LOOPS_PATH) as z:
        return {k: z[k] for k in z.files}


def exists(instrument):
    """True if `nn.joblib` and the instrument's array are both on disk."""
    _check_instrument(instrument)
    if not (os.path.exists(_NN_PATH) and os.path.exists(_LOOPS_PATH)):
        return False
    with np.load(_LOOPS_PATH) as z:
        return instrument in z.files


def load(instrument):
    """Return (nn, loops_array) for the given instrument."""
    _check_instrument(instrument)
    if not (os.path.exists(_NN_PATH) and os.path.exists(_LOOPS_PATH)):
        raise FileNotFoundError(
            f"{_NN_PATH} or {_LOOPS_PATH} not found — run migrate.py."
        )
    nn = joblib.load(_NN_PATH)
    with np.load(_LOOPS_PATH) as z:
        arr = z[instrument]
    return nn, arr


def dump(nn, arr, instrument):
    """Save the shared nn and merge `arr` into loops.npz under `instrument`.

    Safe to call repeatedly for different instruments — existing keys are
    preserved.
    """
    _check_instrument(instrument)
    os.makedirs(_DIR, exist_ok=True)
    joblib.dump(nn, _NN_PATH)
    arrays = _read_loops()
    arrays[instrument] = arr
    np.savez_compressed(_LOOPS_PATH, **arrays)


def dump_all(nn, arrays):
    """Save the shared nn and all instrument arrays in one pass.

    `arrays` is a dict like {"drums": ..., "piano": ..., ...}. Faster than
    four `dump` calls because loops.npz is written exactly once.
    """
    for name in arrays:
        _check_instrument(name)
    os.makedirs(_DIR, exist_ok=True)
    joblib.dump(nn, _NN_PATH)
    np.savez_compressed(_LOOPS_PATH, **arrays)
