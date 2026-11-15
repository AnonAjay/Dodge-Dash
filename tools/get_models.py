"""Download the MediaPipe pose model once (needs internet).

Run from the repo root:  python -m tools.get_models
The .task file is git-ignored: every laptop needs its own copy (or copy it by USB for the fest).
"""
import urllib.request
from pathlib import Path

URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
       "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
DEST = Path("models/pose_landmarker_lite.task")


def main():
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists():
        print("Already downloaded:", DEST)
        return
    print("Downloading...")
    urllib.request.urlretrieve(URL, DEST)
    print("Saved to", DEST)


if __name__ == "__main__":
    main()
