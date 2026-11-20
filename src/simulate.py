"""Physics-inspired synthetic bearing vibration, used when the CWRU files are absent.

A bearing defect produces a short impact each time a rolling element passes it.
Each impact rings the structure at a resonance and decays. The impact rate
equals the defect frequency (BPFI, BPFO, or BSF), which is what envelope
analysis later recovers.
"""
import numpy as np

from src.config import BPFI, BPFO, BSF, FS, SHAFT_HZ

FAULT_FREQ = {"inner_race": BPFI, "outer_race": BPFO, "ball": BSF}


def simulate(label: str, severity: float = 0.007, seconds: float = 10.0,
             seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS

    # Healthy machine: shaft harmonics + broadband noise
    x = 0.05 * np.sin(2 * np.pi * SHAFT_HZ * t)
    x += 0.02 * np.sin(2 * np.pi * 2 * SHAFT_HZ * t + 0.3)
    x += rng.uniform(0.15, 0.25) * rng.standard_normal(n)  # sensor + machine noise

    if label == "normal":
        return x

    f_fault = FAULT_FREQ[label]
    amp = 0.25 + 15 * severity          # bigger defect, harder impacts
    resonance = 3_000 + rng.uniform(-200, 200)
    decay = 900.0
    ring = np.arange(int(0.004 * FS)) / FS
    impulse = np.exp(-decay * ring) * np.sin(2 * np.pi * resonance * ring)

    period = FS / f_fault
    k = 0.0
    while k < n - len(impulse):
        start = int(k + rng.normal(0, 0.01 * period))  # slight slip
        if 0 <= start < n - len(impulse):
            a = amp
            if label == "inner_race":    # inner race rotates in/out of load zone
                a *= 0.6 + 0.4 * np.cos(2 * np.pi * SHAFT_HZ * start / FS)
            elif label == "ball":         # ball faults are weaker and irregular
                a *= 0.35 * rng.uniform(0.1, 1.0)
            x[start:start + len(impulse)] += a * impulse
        k += period
    return x
