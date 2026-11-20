"""Download the CWRU bearing recordings used in this project.

Usage:
    python -m src.download_cwru
"""
import sys

import requests

from src.config import CWRU_FILES, CWRU_URL, RAW_DIR


def download(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for num, (label, size) in CWRU_FILES.items():
        dest = RAW_DIR / f"{num}.mat"
        if dest.exists() and not force:
            print(f"  skip {dest.name} (exists)")
            continue
        url = CWRU_URL.format(num=num)
        print(f"  get  {url}  [{label}, {size} in]")
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
        except requests.RequestException as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            continue
        dest.write_bytes(r.content)
    print(f"Done. Files are in {RAW_DIR}")


if __name__ == "__main__":
    download(force="--force" in sys.argv)
