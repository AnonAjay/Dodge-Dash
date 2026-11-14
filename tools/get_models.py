"""Download the MediaPipe pose model into models/ (not committed to git).
    python -m tools.get_models            # lite (fast)
    python -m tools.get_models --variant full
If the download fails, get the .task file from the MediaPipe pose landmarker docs
and save it as models/pose_landmarker_<variant>.task
"""
import argparse
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["lite", "full", "heavy"], default="lite")
    args = ap.parse_args()
    url = f"{BASE}/pose_landmarker_{args.variant}/float16/latest/pose_landmarker_{args.variant}.task"
    dest = Path("models") / f"pose_landmarker_{args.variant}.task"
    dest.parent.mkdir(exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
