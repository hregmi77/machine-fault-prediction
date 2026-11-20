import numpy as np
import pytest

from src.config import BPFI, BPFO, FS, WINDOW
from src.data import windows
from src.features import FEATURE_COLS, envelope_spectrum, extract, time_features
from src.simulate import simulate


def test_windows_shape_and_overlap():
    x = np.arange(10_000, dtype=float)
    w = windows(x, size=1000, step=500)
    assert w.shape == (19, 1000)
    assert w[1, 0] == 500


def test_windows_rejects_short_signal():
    with pytest.raises(ValueError):
        windows(np.zeros(10), size=100)


def test_sine_time_features():
    t = np.arange(FS) / FS
    f = time_features(np.sin(2 * np.pi * 50 * t))
    assert f["rms"] == pytest.approx(1 / np.sqrt(2), rel=1e-3)
    assert f["crest_factor"] == pytest.approx(np.sqrt(2), rel=1e-3)
    assert f["kurtosis"] == pytest.approx(1.5, rel=1e-2)


def test_extract_returns_all_features():
    feats = extract(np.random.default_rng(0).standard_normal(WINDOW))
    assert list(feats) == FEATURE_COLS
    assert all(np.isfinite(v) for v in feats.values())


@pytest.mark.parametrize("label,f0", [("inner_race", BPFI), ("outer_race", BPFO)])
def test_envelope_finds_fault_frequency(label, f0):
    x = simulate(label, seconds=2, seed=1)
    freqs, mag = envelope_spectrum(x)
    band = (freqs > 50) & (freqs < 300)
    peak = freqs[band][np.argmax(mag[band])]
    assert abs(peak - f0) < 3  # within FFT resolution


def test_fault_raises_kurtosis():
    healthy = time_features(simulate("normal", seconds=2, seed=2))["kurtosis"]
    faulty = time_features(simulate("outer_race", severity=0.021, seconds=2, seed=2))["kurtosis"]
    assert faulty > healthy  # impacts make the signal "spikier" than Gaussian noise
