"""Run on EVERY laptop and send the numbers to the lead.
    python -m tools.get_models
    python -m tools.timing_test [--variant lite|full] [--camera 0]

Uses the MediaPipe Tasks API (mp.solutions no longer exists in current mediapipe).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

N_FRAMES = 150


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="lite")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    model = Path("models") / f"pose_landmarker_{args.variant}.task"
    if not model.exists():
        sys.exit(f"Missing {model}. Run: python -m tools.get_models --variant {args.variant}")

    options = vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        sys.exit("Could not open the camera.")

    t_cam = t_pose = 0.0
    last_ts = -1
    detected = 0
    wall_start = time.perf_counter()
    for _ in range(N_FRAMES):
        a = time.perf_counter()
        ok, frame = cap.read()
        b = time.perf_counter()
        if not ok:
            sys.exit("Camera read failed.")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), last_ts + 1)   # must strictly increase
        last_ts = ts
        result = landmarker.detect_for_video(image, ts)
        c = time.perf_counter()
        t_cam += b - a
        t_pose += c - b
        detected += 1 if result.pose_landmarks else 0
    wall = time.perf_counter() - wall_start
    cap.release()

    print(f"variant={args.variant}  frames={N_FRAMES}  person detected in {detected}/{N_FRAMES}")
    print(f"camera read : {1000 * t_cam / N_FRAMES:6.1f} ms/frame")
    print(f"pose        : {1000 * t_pose / N_FRAMES:6.1f} ms/frame")
    print(f"serial loop : {N_FRAMES / wall:6.1f} FPS  (threads should beat this)")


if __name__ == "__main__":
    main()
