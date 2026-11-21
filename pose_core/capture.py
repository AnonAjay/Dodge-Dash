"""Camera + pose inference in ONE background thread. The game only calls latest().

The shared piece of data is the newest PoseFrame, guarded by a lock. The thread overwrites it
~30x per second; the game reads it at 60 FPS and never waits for the camera.
(see .agent/skills/pose-tracking/SKILL.md)

Implements shared.contracts.PoseProvider.
"""
from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from pose_core.floor_map import FloorMapper
from pose_core.landmarks import JumpDetector, feet_info
from pose_core.skeleton import Skeleton3D, SkeletonBuilder
from shared.contracts import PoseFrame

# ---- tunables ----------------------------------------------------------------
SMOOTH_ALPHA = 0.5     # legacy fallback
TELEPORT_PX = 150      # a bigger jump than this = a different person, don't smooth across it
LOST_GRACE_S = 0.5     # keep the last position this long when detection blinks out
ONE_EURO_MIN_CUTOFF = 0.8  # Hz: heavier smoothing at rest to kill sensor noise
ONE_EURO_BETA = 0.02       # Speed coefficient: eliminates lag during fast movement
ONE_EURO_D_CUTOFF = 1.0    # Hz: cutoff for derivative/speed estimation


class OneEuroFilter:
    """1-Euro adaptive low-pass filter for 2D points (Casiez et al., CHI 2012).
    Dynamically balances jitter elimination at rest and zero lag at speed.
    """

    def __init__(self, min_cutoff: float = ONE_EURO_MIN_CUTOFF, beta: float = ONE_EURO_BETA, d_cutoff: float = ONE_EURO_D_CUTOFF):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None
        self.t_prev: float | None = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / max(dt, 1e-4))

    def filter(self, x: np.ndarray | tuple[float, float], t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt <= 1e-4:
            return self.x_prev

        # Teleport guard: if tracked point snaps across room, reset immediately
        if np.linalg.norm(x - self.x_prev) > TELEPORT_PX:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        # 1. Estimate filtered derivative (velocity in px/s)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # 2. Dynamic cutoff based on speed
        speed = float(np.linalg.norm(dx_hat))
        cutoff = self.min_cutoff + self.beta * speed

        # 3. Filter position
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.t_prev = t
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


def open_camera(index: int = 0, width: int = 640, height: int = 480):
    """Try several Windows camera backends until one actually delivers frames."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for b in backends:
        cap = cv2.VideoCapture(index, b)
        if cap.isOpened():
            for _ in range(15):  # cameras need a moment to warm up
                ok, _f = cap.read()
                if ok:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    print(f"Camera {index} opened (backend {b})")
                    return cap
                time.sleep(0.1)
        cap.release()
    raise RuntimeError(
        f"No frames from camera {index}. End every python.exe in Task Manager, "
        "close Teams/Zoom/Camera app, or try another camera index.")


def create_landmarker(model_path: str):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,  # so we can pick the person standing in the box
    )
    return mp, vision.PoseLandmarker.create_from_options(options)


class PoseTracker:
    """Turns landmark lists into PoseFrames. No camera or MediaPipe inside: easy to test."""

    def __init__(self, mapper: FloorMapper):
        self.mapper = mapper
        self.jump = JumpDetector()
        self.sm: dict[str, OneEuroFilter] = {}
        self.last_floor = (mapper.w_m / 2, mapper.h_m / 2)
        self.grounded = None            # last (centre, left, right) floor positions on the ground
        self.last_pf: PoseFrame | None = None
        self.last_good_ts: float | None = None
        self.debug: dict = {}
        self.skeleton: Skeleton3D | None = None
        self.skeleton_builder = SkeletonBuilder()

    def _smooth(self, key: str, px, ts: float | None = None) -> np.ndarray:
        if key not in self.sm:
            self.sm[key] = OneEuroFilter()
        current_ts = ts if ts is not None else (self.last_good_ts or 0.0)
        return self.sm[key].filter(px, current_ts)

    def update(self, people, img_w: int, img_h: int, ts: float,
               world_people: list | None = None) -> PoseFrame:
        self.debug = {"foot_px": None, "pts": [], "lift": 0.0, "body_ok": False, "landmarks_px": []}

        cands = []
        for idx, lms in enumerate(people):
            c = feet_info(lms, img_w, img_h)
            if c is None:
                continue
            fx, fy = self.mapper.to_floor(*c["centre_px"])
            dist = math.hypot(fx - self.last_floor[0], fy - self.last_floor[1])
            c["rank"] = (0 if self.mapper.inside(fx, fy) else 1, dist)  # in the box first, then nearest
            c["idx"] = idx
            c["lms"] = lms
            cands.append(c)

        if not cands:
            self.sm.clear()
            self.jump.reset()
            if self.last_pf is not None and self.last_good_ts is not None \
                    and ts - self.last_good_ts < LOST_GRACE_S:
                if self.skeleton is not None:
                    self.skeleton = replace(self.skeleton, timestamp=ts)
                return replace(self.last_pf, timestamp=ts, airborne=False)  # short blink-out: hold
            self.skeleton = None
            self.skeleton_builder.reset()
            return PoseFrame(timestamp=ts)  # tracked=False, no position

        c = min(cands, key=lambda k: k["rank"])

        airborne, lift = self.jump.update(c["yl"], c["yr"], c["torso"], c["y_hip"])

        # The homography is only valid for feet ON the floor. From the first rising frame
        # (before the jump is even confirmed) hold the last grounded position, and keep the
        # smoothing state untouched so the lifted feet don't contaminate it.
        if (airborne or self.jump.rising) and self.grounded is not None:
            pos, lf, rf = self.grounded
            centre = np.asarray(c["centre_px"], dtype=float)   # raw, for the debug view only
        else:
            centre = self._smooth("c", c["centre_px"], ts)
            for key, name in (("l", "left_px"), ("r", "right_px")):
                if c[name] is not None:
                    self._smooth(key, c[name], ts)
                else:
                    self.sm.pop(key, None)
            pos = self.mapper.to_floor(*centre)
            lf = self.mapper.to_floor(*self.sm["l"].x_prev) if ("l" in self.sm and self.sm["l"].x_prev is not None) else None
            rf = self.mapper.to_floor(*self.sm["r"].x_prev) if ("r" in self.sm and self.sm["r"].x_prev is not None) else None
            self.grounded = (pos, lf, rf)
        self.last_floor = pos

        selected_idx = c["idx"]
        if world_people is not None and selected_idx < len(world_people):
            world_lms = world_people[selected_idx]
            self.skeleton = self.skeleton_builder.build(
                world_lms,
                pos,
                hip_lift=self.jump.hip_lift,
                ts=ts,
                is_jumping=(airborne or self.jump.rising),
            )
        else:
            self.skeleton = None

        pf = PoseFrame(timestamp=ts, tracked=True, floor_x_m=pos[0], floor_y_m=pos[1],
                       left_foot_m=lf, right_foot_m=rf, arms_up=c["arms_up"], airborne=airborne)
        self.last_pf, self.last_good_ts = pf, ts
        landmarks_px = [(l.x * img_w, l.y * img_h, getattr(l, "visibility", 1.0)) for l in c["lms"]]
        self.debug = {"foot_px": tuple(centre), "pts": c["pts"], "lift": lift,
                      "body_ok": c["torso"] is not None, "landmarks_px": landmarks_px}
        return pf


class CameraPoseSource:
    """Camera thread -> PoseFrame. Without a mapper it only supplies camera frames
    (used by the calibration tool); give it one with set_mapper() to start tracking."""

    def __init__(self, camera_index: int = 0,
                 model_path: str = "models/pose_landmarker_lite.task",
                 mapper: FloorMapper | None = None):
        self.camera_index = camera_index
        self.model_path = model_path
        self._lock = threading.Lock()
        self._latest = PoseFrame()
        self._latest_skeleton: Skeleton3D | None = None
        self._frame = None
        self._debug: dict = {}
        self._pending = mapper          # applied inside the thread
        self._tracker: PoseTracker | None = None
        self._mp = None
        self._landmarker = None
        self._last_ms = 0
        self._cap = None
        self._thread: threading.Thread | None = None
        self._running = False
        self.pose_fps = 0.0
        self.error: str | None = None

    # ---- PoseProvider ----------------------------------------------------------
    def start(self) -> None:
        if self._pending is not None:
            self._check_model()
        self._cap = open_camera(self.camera_index)
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def latest(self) -> PoseFrame:
        with self._lock:
            return self._latest

    def latest_skeleton(self) -> Skeleton3D | None:
        with self._lock:
            return self._latest_skeleton

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

    # ---- extras for tools and debug views --------------------------------------
    def set_mapper(self, mapper: FloorMapper) -> None:
        """Start tracking, or switch to a new calibration while running."""
        self._check_model()
        with self._lock:
            self._pending = mapper

    def snapshot(self):
        """(PoseFrame, camera frame or None, debug dict), all from the same moment."""
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return self._latest, frame, dict(self._debug)

    def snapshot3d(self):
        """(PoseFrame, Skeleton3D or None, camera frame or None, debug dict), all from the same moment."""
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return self._latest, self._latest_skeleton, frame, dict(self._debug)

    def latest_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    # ---- internals ---------------------------------------------------------------
    def _check_model(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run: python -m tools.get_models")

    def _apply_pending(self):
        with self._lock:
            mapper, self._pending = self._pending, None
        if mapper is None:
            return
        if self._landmarker is None:
            self._mp, self._landmarker = create_landmarker(self.model_path)
        self._tracker = PoseTracker(mapper)

    def _detect(self, frame, ts: float):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ms = max(int(ts * 1000), self._last_ms + 1)  # must keep increasing
        self._last_ms = ms
        res = self._landmarker.detect_for_video(img, ms)
        return res.pose_landmarks, res.pose_world_landmarks

    def _run(self):
        last = time.monotonic()
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)  # mirror, so moving right moves right on screen
            ts = time.monotonic()
            try:
                self._apply_pending()
                if self._tracker is not None:
                    people, world_people = self._detect(frame, ts)
                    pf = self._tracker.update(people, frame.shape[1], frame.shape[0], ts, world_people=world_people)
                    skel = self._tracker.skeleton
                    debug = dict(self._tracker.debug)
                else:
                    pf, skel, debug = PoseFrame(timestamp=ts), None, {}
            except Exception as e:  # keep the thread alive, report the problem
                self.error = f"{type(e).__name__}: {e}"
                pf, skel, debug = PoseFrame(timestamp=ts), None, {}
            with self._lock:
                self._latest, self._latest_skeleton, self._frame, self._debug = pf, skel, frame, debug
            dt = ts - last
            last = ts
            if dt > 0:
                self.pose_fps = 0.9 * self.pose_fps + 0.1 * (1.0 / dt)
