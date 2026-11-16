"""Floor calibration. Run once per venue, again if the camera moves.

    python -m tools.calibrate_floor --floor-w 1.245 --floor-h 1.245

1. Click the 4 floor markers IN THIS ORDER, as you see them on screen:
   near-left, near-right, far-right, far-left.
2. A grid is drawn over the live feed. If it sits on your tape, press ENTER to save
   calibration.local.json (git-ignored). If not, press R and click again.
See .agent/skills/floor-homography/SKILL.md
"""
from __future__ import annotations

# keep pose_core imports above cv2 (pose_core/__init__ sets a Windows camera env var)
import argparse
import sys
import time

from pose_core.capture import CameraPoseSource
from pose_core.floor_map import CORNER_NAMES, FloorMapper, setup_warnings

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_grid(frame, mapper: FloorMapper, step_m: float = 0.25):
    """Draw the floor grid (every step_m metres) over the camera image."""
    def line(a, b, col, th):
        pa, pb = mapper.to_pixels(*a), mapper.to_pixels(*b)
        if max(abs(v) for v in (*pa, *pb)) < 1e5:
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), col, th)

    g = 0.0
    while g <= mapper.w_m + 1e-6:
        line((g, 0), (g, mapper.h_m), (0, 255, 255), 1)
        g += step_m
    g = 0.0
    while g <= mapper.h_m + 1e-6:
        line((0, g), (mapper.w_m, g), (0, 255, 255), 1)
        g += step_m
    for a, b in (((0, 0), (mapper.w_m, 0)), ((mapper.w_m, 0), (mapper.w_m, mapper.h_m)),
                 ((mapper.w_m, mapper.h_m), (0, mapper.h_m)), ((0, mapper.h_m), (0, 0))):
        line(a, b, (0, 255, 0), 2)


def calibrate_floor(get_frame, w_m: float, h_m: float, grid_m: float = 0.25, save: bool = True):
    """get_frame() -> latest BGR camera frame (a copy) or None. Returns a FloorMapper or None."""
    win = "Calibrate floor"
    clicks: list[tuple[int, int]] = []
    state = {"mapper": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state["mapper"] is None and len(clicks) < 4:
            clicks.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        frame = get_frame()
        if frame is None:
            cv2.waitKey(10)
            continue
        mapper = state["mapper"]
        if mapper is None:
            for i, p in enumerate(clicks):
                cv2.circle(frame, p, 6, (0, 255, 0), -1)
                cv2.putText(frame, f"{i + 1} {CORNER_NAMES[i]}", (p[0] + 8, p[1] - 8), FONT, 0.5, (0, 255, 0), 2)
                if i > 0:
                    cv2.line(frame, clicks[i - 1], p, (0, 255, 0), 2)
            if len(clicks) == 4:
                cv2.line(frame, clicks[3], clicks[0], (0, 255, 0), 2)
                msg = "ENTER = show grid, R = redo"
                warns = setup_warnings(clicks, frame.shape[1], frame.shape[0])
                if not warns:
                    cv2.putText(frame, "Setup check: looks OK", (10, 50), FONT, 0.5, (0, 220, 0), 2)
                for i, w_ in enumerate(warns):
                    cv2.putText(frame, "WARNING: " + w_, (10, 50 + 22 * i), FONT, 0.5, (0, 165, 255), 2)
            else:
                msg = f"Click corner {len(clicks) + 1}/4: {CORNER_NAMES[len(clicks)]}  (R = redo, Q = quit)"
        else:
            draw_grid(frame, mapper, grid_m)
            msg = "Does the grid sit on your tape?  ENTER = save, R = redo"
        cv2.putText(frame, msg, (10, 25), FONT, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, frame)

        k = cv2.waitKey(20) & 0xFF
        if k == ord("r"):
            clicks.clear()
            state["mapper"] = None
        elif k in (13, 32):
            if mapper is None and len(clicks) == 4:
                state["mapper"] = FloorMapper(clicks, w_m, h_m)
            elif mapper is not None:
                if save:
                    mapper.save()
                cv2.destroyWindow(win)
                return mapper
        elif k == ord("q"):
            cv2.destroyWindow(win)
            return None


def main():
    ap = argparse.ArgumentParser(description="Calibrate the floor box (once per venue).")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--floor-w", type=float, default=1.245, help="box width in metres (49 in = 1.245)")
    ap.add_argument("--floor-h", type=float, default=1.245, help="box depth in metres")
    ap.add_argument("--grid", type=float, default=0.25, help="grid spacing in metres")
    args = ap.parse_args()

    src = CameraPoseSource(args.camera)
    try:
        src.start()
    except RuntimeError as e:
        sys.exit(str(e))
    try:
        t0 = time.monotonic()
        while src.latest_frame() is None:
            if time.monotonic() - t0 > 5:
                sys.exit("No frames from the camera.")
            time.sleep(0.05)
        mapper = calibrate_floor(src.latest_frame, args.floor_w, args.floor_h, args.grid)
    finally:
        src.stop()
        cv2.destroyAllWindows()
    print("Saved calibration.local.json" if mapper else "Cancelled, nothing saved.")


if __name__ == "__main__":
    main()
