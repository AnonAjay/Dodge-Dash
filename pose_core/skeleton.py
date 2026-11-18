"""3D skeleton representation in floor coordinates.

Provides the 35 MediaPipe pose bone connections, the Skeleton3D contract dataclass,
and the pure SkeletonBuilder which transforms MediaPipe pose_world_landmarks into
the game's metric floor coordinate frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# MediaPipe 33-landmark pose connections (35 bones)
BONES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

UP_SIGN_OVERRIDE: int | None = None
CAMERA_PITCH_DEG: float = 0.0
SKELETON_ALPHA: float = 0.5


@dataclass
class Skeleton3D:
    """3D stick figure connecting all 33 MediaPipe pose landmarks in floor coordinates.

    Conventions:
      - Floor frame in metres:
        x grows to the right
        y grows AWAY from the camera
        (0, 0) = near-left corner of the taped box (see pose_core/floor_map.py)
        z = UP, with z = 0 being the floor plane.
      - points_m: list of 33 (x, y, z) coordinate tuples in the floor frame.
      - visibility: list of 33 float visibility scores (0..1).
      - jump_height_m: estimated vertical lift (metres) above the floor (0.0 when standing).
    """
    timestamp: float
    points_m: list[tuple[float, float, float]]
    visibility: list[float]
    jump_height_m: float


class SkeletonBuilder:
    """Transforms 33-point world landmarks into 3D floor coordinates."""

    def __init__(self, camera_pitch_deg: float = CAMERA_PITCH_DEG):
        self.camera_pitch_deg = camera_pitch_deg
        self.up_sign: int | None = UP_SIGN_OVERRIDE
        self._samples: list[float] = []
        self._smoothed_points: list[tuple[float, float, float]] | None = None
        self._jump_ema: float | None = None

    def reset(self) -> None:
        """Reset point smoothing and jump state (e.g. when tracking is lost)."""
        self._smoothed_points = None
        self._jump_ema = None

    @staticmethod
    def _get_attr(lm: Any, name: str, default: float = 0.0) -> float:
        if hasattr(lm, name):
            return getattr(lm, name)
        if isinstance(lm, dict) and name in lm:
            return lm[name]
        if isinstance(lm, (list, tuple)):
            mapping = {"x": 0, "y": 1, "z": 2, "visibility": 3}
            idx = mapping.get(name)
            if idx is not None and idx < len(lm):
                return lm[idx]
        return default

    def build(
        self,
        world_lms: list[Any],
        floor_pos: Any,
        hip_lift: float = 0.0,
        ts: float = 0.0,
        is_jumping: bool | None = None,
    ) -> Skeleton3D:
        """Build a Skeleton3D from 33 world landmarks and 2D floor position."""
        if len(world_lms) < 33:
            raise ValueError(f"Expected 33 world landmarks, got {len(world_lms)}")

        # 1. Estimate up_sign (once from the first 15 frames, or via UP_SIGN_OVERRIDE)
        y11 = self._get_attr(world_lms[11], "y")
        y12 = self._get_attr(world_lms[12], "y")
        y23 = self._get_attr(world_lms[23], "y")
        y24 = self._get_attr(world_lms[24], "y")
        diff = ((y11 + y12) / 2.0) - ((y23 + y24) / 2.0)

        if UP_SIGN_OVERRIDE is not None:
            effective_up_sign = UP_SIGN_OVERRIDE
            if self.up_sign is None:
                self.up_sign = UP_SIGN_OVERRIDE
                print(f"SkeletonBuilder: up_sign={self.up_sign} (override: world y points {'DOWN' if self.up_sign == -1 else 'UP'})")
        elif self.up_sign is not None:
            effective_up_sign = self.up_sign
        else:
            self._samples.append(diff)
            effective_up_sign = -1 if np.median(self._samples) < 0 else 1
            if len(self._samples) == 15:
                self.up_sign = effective_up_sign
                print(f"SkeletonBuilder: up_sign={self.up_sign} (estimated: world y points {'DOWN' if self.up_sign == -1 else 'UP'})")

        # 2. Camera pitch correction
        theta = math.radians(self.camera_pitch_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        wx_list = []
        wz_level_list = []
        up_list = []
        vis_list = []

        for lm in world_lms:
            wx = float(self._get_attr(lm, "x"))
            wy = float(self._get_attr(lm, "y"))
            wz = float(self._get_attr(lm, "z"))
            v = float(self._get_attr(lm, "visibility", 1.0))

            # Convert point to level frame
            y_l = wy * cos_t + wz * sin_t
            z_l = -wy * sin_t + wz * cos_t

            wx_list.append(wx)
            wz_level_list.append(z_l)
            up_list.append(effective_up_sign * y_l)
            vis_list.append(v)

        # 4. Anchor horizontally: centroid of feet landmarks (27..32) sits at floor_pos
        feet_indices = range(27, 33)
        feet_wx = float(np.mean([wx_list[i] for i in feet_indices]))
        feet_wz = float(np.mean([wz_level_list[i] for i in feet_indices]))

        fpx = floor_pos[0] if isinstance(floor_pos, (tuple, list)) else floor_pos.x
        fpy = floor_pos[1] if isinstance(floor_pos, (tuple, list)) else floor_pos.y

        X_list = [fpx + (wx_list[i] - feet_wx) for i in range(len(world_lms))]
        Y_list = [fpy + (wz_level_list[i] - feet_wz) for i in range(len(world_lms))]

        # 6. jump_height_m: max(0, hip_lift) * torso_m while rising or airborne
        s11 = np.array([self._get_attr(world_lms[11], "x"), self._get_attr(world_lms[11], "y"), self._get_attr(world_lms[11], "z")])
        s12 = np.array([self._get_attr(world_lms[12], "x"), self._get_attr(world_lms[12], "y"), self._get_attr(world_lms[12], "z")])
        h23 = np.array([self._get_attr(world_lms[23], "x"), self._get_attr(world_lms[23], "y"), self._get_attr(world_lms[23], "z")])
        h24 = np.array([self._get_attr(world_lms[24], "x"), self._get_attr(world_lms[24], "y"), self._get_attr(world_lms[24], "z")])
        torso_m = float(np.linalg.norm(((s11 + s12) / 2.0) - ((h23 + h24) / 2.0)))

        jumping = (hip_lift > 0.0) if is_jumping is None else is_jumping
        if not jumping:
            jump_height_m = 0.0
            self._jump_ema = 0.0
        else:
            raw_h = max(0.0, float(hip_lift)) * torso_m
            if raw_h < 0.03:
                target_h = 0.0
            else:
                target_h = min(0.8, raw_h)

            if self._jump_ema is None or self._jump_ema == 0.0:
                self._jump_ema = target_h
            else:
                self._jump_ema = 0.5 * target_h + 0.5 * self._jump_ema
            jump_height_m = float(self._jump_ema)

        # 5. Anchor vertically: lowest foot point sits at z = jump_height_m
        feet_up_min = min(up_list[i] for i in feet_indices)
        Z_list = [(up_list[i] - feet_up_min) + jump_height_m for i in range(len(world_lms))]

        # 7. Smooth every point with an EMA (SKELETON_ALPHA = 0.5)
        raw_points = [(X_list[i], Y_list[i], Z_list[i]) for i in range(len(world_lms))]

        if self._smoothed_points is None:
            smoothed_points = list(raw_points)
        else:
            smoothed_points = []
            for i in range(len(world_lms)):
                if vis_list[i] < 0.5:
                    smoothed_points.append(self._smoothed_points[i])
                else:
                    px, py, pz = raw_points[i]
                    prev_x, prev_y, prev_z = self._smoothed_points[i]
                    sx = SKELETON_ALPHA * px + (1.0 - SKELETON_ALPHA) * prev_x
                    sy = SKELETON_ALPHA * py + (1.0 - SKELETON_ALPHA) * prev_y
                    sz = SKELETON_ALPHA * pz + (1.0 - SKELETON_ALPHA) * prev_z
                    smoothed_points.append((float(sx), float(sy), float(sz)))

        self._smoothed_points = smoothed_points

        return Skeleton3D(
            timestamp=ts,
            points_m=smoothed_points,
            visibility=vis_list,
            jump_height_m=jump_height_m,
        )
