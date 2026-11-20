"""Feature engineering for vibration windows.

Three families of features, the same ones a reliability engineer looks at:
  1. Time-domain statistics (RMS, kurtosis, crest factor ...): overall health.
  2. Frequency-band energy: where the energy sits in the spectrum.
  3. Envelope spectrum at the bearing defect frequencies: *which* part is failing.
"""
import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.stats import kurtosis, skew

from src.config import BPFI, BPFO, BSF, FS

BANDS = [(0, 500), (500, 2000), (2000, 4000), (4000, 6000)]


def time_features(x: np.ndarray) -> dict:
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    mean_abs = np.mean(np.abs(x))
    return {
        "rms": rms,
        "peak": peak,
        "crest_factor": peak / rms,
        "kurtosis": kurtosis(x, fisher=False),
        "skewness": skew(x),
        "std": np.std(x),
        "peak_to_peak": np.ptp(x),
        "impulse_factor": peak / mean_abs,
        "shape_factor": rms / mean_abs,
    }


def spectrum(x: np.ndarray, fs: int = FS):
    """One-sided amplitude spectrum with a Hann window."""
    w = np.hanning(len(x))
    mag = np.abs(np.fft.rfft(x * w)) * 2 / w.sum()
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    return freqs, mag


def envelope_spectrum(x: np.ndarray, fs: int = FS, band=(2000, 5000)):
    """Band-pass around the resonance, take the Hilbert envelope, then its spectrum."""
    sos = sps.butter(4, band, btype="bandpass", fs=fs, output="sos")
    env = np.abs(sps.hilbert(sps.sosfiltfilt(sos, x)))
    return spectrum(env - env.mean(), fs)


def _peak_near(freqs, mag, f0, tol=0.04):
    m = (freqs > f0 * (1 - tol)) & (freqs < f0 * (1 + tol))
    return float(mag[m].max()) if m.any() else 0.0


def freq_features(x: np.ndarray, fs: int = FS) -> dict:
    freqs, mag = spectrum(x, fs)
    power = mag ** 2
    total = power.sum() + 1e-12
    feats = {f"band_{lo}_{hi}": power[(freqs >= lo) & (freqs < hi)].sum() / total
             for lo, hi in BANDS}
    feats["spectral_centroid"] = float((freqs * power).sum() / total)

    ef, em = envelope_spectrum(x, fs)
    ref = np.median(em) + 1e-12  # normalise against the envelope noise floor
    feats["env_bpfi"] = _peak_near(ef, em, BPFI) / ref
    feats["env_bpfo"] = _peak_near(ef, em, BPFO) / ref
    feats["env_bsf"] = _peak_near(ef, em, BSF) / ref
    return feats


def extract(x: np.ndarray, fs: int = FS) -> dict:
    return {**time_features(x), **freq_features(x, fs)}


def build_table(recordings, window_fn) -> pd.DataFrame:
    """One row per window: features + label + recording id + position in recording."""
    rows = []
    for rec in recordings:
        w = window_fn(rec.signal)
        for i, seg in enumerate(w):
            rows.append({**extract(seg), "label": rec.label, "severity": rec.severity,
                         "rec_id": rec.rec_id, "source": rec.source,
                         "position": i / max(len(w) - 1, 1)})
    return pd.DataFrame(rows)


FEATURE_COLS = list(time_features(np.random.randn(256)).keys()) + \
    [f"band_{lo}_{hi}" for lo, hi in BANDS] + \
    ["spectral_centroid", "env_bpfi", "env_bpfo", "env_bsf"]
