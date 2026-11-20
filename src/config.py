"""Project-wide settings: paths, sampling, and bearing geometry."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

# CWRU drive-end accelerometer, 12 kHz, 0 HP load, ~1797 rpm
FS = 12_000
SHAFT_RPM = 1797
SHAFT_HZ = SHAFT_RPM / 60

# SKF 6205-2RS drive-end bearing: defect frequencies as multiples of shaft speed
BPFI = 5.4152 * SHAFT_HZ  # inner race
BPFO = 3.5848 * SHAFT_HZ  # outer race
BSF = 2.3570 * SHAFT_HZ   # ball spin

WINDOW = 2048   # samples per analysis window (~0.17 s)
STEP = 1024     # 50% overlap

CLASSES = ["normal", "inner_race", "ball", "outer_race"]

# CWRU file number -> (label, fault diameter in inches). Source:
# https://engineering.case.edu/bearingdatacenter/download-data-file
CWRU_FILES = {
    97: ("normal", 0.0),
    105: ("inner_race", 0.007), 169: ("inner_race", 0.014), 209: ("inner_race", 0.021),
    118: ("ball", 0.007), 185: ("ball", 0.014), 222: ("ball", 0.021),
    130: ("outer_race", 0.007), 197: ("outer_race", 0.014), 234: ("outer_race", 0.021),
}
CWRU_URL = "https://engineering.case.edu/sites/default/files/{num}.mat"
