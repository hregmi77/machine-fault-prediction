"""Load vibration recordings (real CWRU data, or synthetic fallback) and cut them into windows."""
from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat

from src.config import CWRU_FILES, RAW_DIR, STEP, WINDOW
from src.simulate import simulate


@dataclass
class Recording:
    rec_id: str
    label: str
    severity: float
    signal: np.ndarray
    source: str  # "cwru" or "synthetic"


def _read_mat(path) -> np.ndarray:
    mat = loadmat(path)
    key = next(k for k in mat if k.endswith("_DE_time"))  # drive-end accelerometer
    return mat[key].ravel().astype(float)


def load_recordings(prefer_real: bool = True) -> list[Recording]:
    recs = []
    for num, (label, sev) in CWRU_FILES.items():
        path = RAW_DIR / f"{num}.mat"
        if prefer_real and path.exists():
            recs.append(Recording(f"CWRU-{num}", label, sev, _read_mat(path), "cwru"))
        else:
            sig = simulate(label, severity=sev or 0.007, seconds=10, seed=num)
            recs.append(Recording(f"SIM-{num}", label, sev, sig, "synthetic"))
    return recs


def windows(signal: np.ndarray, size: int = WINDOW, step: int = STEP) -> np.ndarray:
    """Split a 1-D signal into overlapping windows, shape (n_windows, size)."""
    if len(signal) < size:
        raise ValueError(f"signal shorter than one window ({len(signal)} < {size})")
    n = 1 + (len(signal) - size) // step
    idx = np.arange(size)[None, :] + step * np.arange(n)[:, None]
    return signal[idx]
