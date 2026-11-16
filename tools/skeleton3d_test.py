"""3D skeleton and falling hazard test viewer.

Renders an interactive 3D perspective scene with:
  - Neon floor grid corresponding to the calibrated taped box
  - 33-point stick figure in real-world metric floor coordinates
  - Faint shadow on the floor
  - Wireframe hazard cubes falling linearly from 2.5 m to 0
  - Orbit camera with presets (1: default, 2: side view, 3: top-down)

Run from repo root:
    python -m tools.skeleton3d_test --keyboard
    python -m tools.skeleton3d_test
"""
from __future__ import annotations

import argparse
import atexit
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import tools.drop_box_test as dbt
from pose_core.capture import CameraPoseSource
from pose_core.floor_map import FloorMapper
from pose_core.skeleton import (
    BONES,
    CAMERA_PITCH_DEG,
    Skeleton3D,
    SkeletonBuilder,
)
from shared.contracts import PoseFrame
from tools.drop_box_test import draw_debug

# Plausible 33-point standing human template (~1.66 m tall, hip-centered, y-down)
SYNTHETIC_WORLD_LANDMARKS: list[tuple[float, float, float, float]] = [
    # 0..10: Head
    (0.00, -0.75, -0.10, 1.0),   # 0: nose
    (0.02, -0.78, -0.08, 1.0),   # 1: left eye inner
    (0.04, -0.78, -0.08, 1.0),   # 2: left eye
    (0.06, -0.78, -0.08, 1.0),   # 3: left eye outer
    (-0.02, -0.78, -0.08, 1.0),  # 4: right eye inner
    (-0.04, -0.78, -0.08, 1.0),  # 5: right eye
    (-0.06, -0.78, -0.08, 1.0),  # 6: right eye outer
    (0.09, -0.75, 0.00, 1.0),    # 7: left ear
    (-0.09, -0.75, 0.00, 1.0),   # 8: right ear
    (0.03, -0.70, -0.08, 1.0),   # 9: mouth left
    (-0.03, -0.70, -0.08, 1.0),  # 10: mouth right
    # 11..12: Shoulders
    (0.20, -0.50, 0.00, 1.0),    # 11: left shoulder
    (-0.20, -0.50, 0.00, 1.0),   # 12: right shoulder
    # 13..16: Arms
    (0.25, -0.25, 0.00, 1.0),    # 13: left elbow
    (-0.25, -0.25, 0.00, 1.0),   # 14: right elbow
    (0.28, 0.00, 0.00, 1.0),     # 15: left wrist
    (-0.28, 0.00, 0.00, 1.0),    # 16: right wrist
    # 17..22: Hands
    (0.30, 0.05, 0.00, 1.0),     # 17: left pinky
    (-0.30, 0.05, 0.00, 1.0),    # 18: right pinky
    (0.29, 0.06, 0.00, 1.0),     # 19: left index
    (-0.29, 0.06, 0.00, 1.0),    # 20: right index
    (0.27, 0.02, 0.00, 1.0),     # 21: left thumb
    (-0.27, 0.02, 0.00, 1.0),    # 22: right thumb
    # 23..24: Hips
    (0.10, 0.00, 0.00, 1.0),     # 23: left hip
    (-0.10, 0.00, 0.00, 1.0),    # 24: right hip
    # 25..28: Legs
    (0.12, 0.45, 0.00, 1.0),     # 25: left knee
    (-0.12, 0.45, 0.00, 1.0),    # 26: right knee
    (0.12, 0.85, 0.00, 1.0),     # 27: left ankle
    (-0.12, 0.85, 0.00, 1.0),    # 28: right ankle
    # 29..32: Feet
    (0.12, 0.88, 0.05, 1.0),     # 29: left heel
    (-0.12, 0.88, 0.05, 1.0),    # 30: right heel
    (0.12, 0.88, -0.12, 1.0),    # 31: left toe
    (-0.12, 0.88, -0.12, 1.0),   # 32: right toe
]

TORSO_BONES = {(11, 12), (11, 23), (12, 24), (23, 24)}


@dataclass
class OrbitCamera:
    """Virtual orbit camera parameterized by look-at target, distance, yaw, and pitch."""
    target: tuple[float, float, float]
    distance: float
    yaw_deg: float
    pitch_deg: float
    fov_deg: float = 45.0

    @classmethod
    def default(cls, w_m: float = 1.245, h_m: float = 1.245, fov_deg: float = 45.0) -> OrbitCamera:
        """Default: near side, centered on x=w/2, elevated, pitched down ~30 deg, looking at (w/2, h/2, 0.5)."""
        target = (w_m / 2.0, h_m / 2.0, 0.5)
        dist = auto_frame_distance(w_m, h_m, yaw_deg=0.0, pitch_deg=30.0, target=target, fov_deg=fov_deg)
        return cls(
            target=target,
            distance=dist,
            yaw_deg=0.0,
            pitch_deg=30.0,
            fov_deg=fov_deg,
        )

    @classmethod
    def side(cls, w_m: float = 1.245, h_m: float = 1.245, fov_deg: float = 45.0) -> OrbitCamera:
        """Side view: looking along x to check the figure stands upright."""
        target = (w_m / 2.0, h_m / 2.0, 0.75)
        dist = auto_frame_distance(w_m, h_m, yaw_deg=-90.0, pitch_deg=0.0, target=target, fov_deg=fov_deg)
        return cls(
            target=target,
            distance=dist,
            yaw_deg=-90.0,
            pitch_deg=0.0,
            fov_deg=fov_deg,
        )

    @classmethod
    def top_down(cls, w_m: float = 1.245, h_m: float = 1.245, fov_deg: float = 45.0) -> OrbitCamera:
        """Top-down view: looking straight down from above the box."""
        target = (w_m / 2.0, h_m / 2.0, 0.0)
        dist = auto_frame_distance(w_m, h_m, yaw_deg=0.0, pitch_deg=89.0, target=target, fov_deg=fov_deg)
        return cls(
            target=target,
            distance=dist,
            yaw_deg=0.0,
            pitch_deg=89.0,
            fov_deg=fov_deg,
        )


def project(points_xyz: Any, cam: OrbitCamera, size: tuple[int, int] = (800, 800)) -> np.ndarray:
    """Project 3D floor coordinates (N, 3) to 2D image pixels (N, 2)."""
    theta = math.radians(cam.pitch_deg)
    phi = math.radians(cam.yaw_deg)

    # Camera eye position in floor frame
    cx = cam.target[0] + cam.distance * math.cos(theta) * math.sin(phi)
    cy = cam.target[1] - cam.distance * math.cos(theta) * math.cos(phi)
    cz = cam.target[2] + cam.distance * math.sin(theta)
    C = np.array([cx, cy, cz], dtype=float)
    T = np.array(cam.target, dtype=float)

    f = T - C
    f_norm = np.linalg.norm(f)
    f = np.array([0.0, 1.0, 0.0]) if f_norm < 1e-6 else (f / f_norm)

    # Right vector (perpendicular to f and world up (0, 0, 1))
    r = np.cross(f, np.array([0.0, 0.0, 1.0]))
    r_norm = np.linalg.norm(r)
    r = np.array([1.0, 0.0, 0.0]) if r_norm < 1e-6 else (r / r_norm)

    # Camera up vector
    u = np.cross(r, f)
    u_norm = np.linalg.norm(u)
    u = np.array([0.0, 0.0, 1.0]) if u_norm < 1e-6 else (u / u_norm)

    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    D = pts - C

    x_cam = np.dot(D, r)
    y_cam = np.dot(D, u)
    z_cam = np.dot(D, f)

    W, H = size
    fov_rad = math.radians(cam.fov_deg)
    f_px = (W / 2.0) / math.tan(fov_rad / 2.0)
    cx_img = W / 2.0
    cy_img = H / 2.0

    out = np.full((len(pts), 2), np.nan, dtype=float)
    valid = z_cam > 0.05
    out[valid, 0] = cx_img + f_px * (x_cam[valid] / z_cam[valid])
    out[valid, 1] = cy_img - f_px * (y_cam[valid] / z_cam[valid])
    return out


def auto_frame_distance(
    w_m: float,
    h_m: float,
    yaw_deg: float,
    pitch_deg: float,
    target: tuple[float, float, float],
    fov_deg: float = 45.0,
    size: tuple[int, int] = (800, 800),
    target_span: float = 0.60,
    figure_height_m: float = 1.9,
    margin_px: float = 10.0,
) -> float:
    """Choose camera distance so floor rectangle spans ~60% width and 1.9m figure is fully visible."""
    W, H = size
    cx, cy = w_m / 2.0, h_m / 2.0
    floor_corners = [
        (0.0, 0.0, 0.0),
        (w_m, 0.0, 0.0),
        (w_m, h_m, 0.0),
        (0.0, h_m, 0.0),
    ]
    # Points defining a 1.9 m tall figure standing at box centre
    fig_pts = [(cx, cy, z) for z in np.linspace(0.0, figure_height_m, 11)]
    fig_pts.extend([
        (cx - 0.25, cy, figure_height_m * 0.8),
        (cx + 0.25, cy, figure_height_m * 0.8),
        (cx, cy - 0.2, figure_height_m * 0.8),
        (cx, cy + 0.2, figure_height_m * 0.8),
    ])

    best_r = 3.5
    best_diff = float("inf")

    # Numeric loop: search distances from 0.5m to 15.0m
    for r in np.arange(0.5, 15.0, 0.02):
        cam = OrbitCamera(target=target, distance=float(r), yaw_deg=yaw_deg, pitch_deg=pitch_deg, fov_deg=fov_deg)
        pts_fig = project(fig_pts, cam, size)
        if np.isnan(pts_fig).any():
            continue
        u_f, v_f = pts_fig[:, 0], pts_fig[:, 1]
        fig_ok = bool(
            (u_f >= margin_px).all() and (u_f <= W - margin_px).all() and
            (v_f >= margin_px).all() and (v_f <= H - margin_px).all()
        )
        if not fig_ok:
            continue

        pts_floor = project(floor_corners, cam, size)
        if np.isnan(pts_floor).any():
            continue
        span_w = (np.max(pts_floor[:, 0]) - np.min(pts_floor[:, 0])) / W
        span_h = (np.max(pts_floor[:, 1]) - np.min(pts_floor[:, 1])) / H
        span = max(span_w, span_h) if pitch_deg > 70.0 else span_w
        diff = abs(span - target_span)
        if diff < best_diff:
            best_diff = diff
            best_r = float(r)

    return round(float(best_r), 2)


def render_scene(
    skeleton: Skeleton3D | None,
    game: Any | None,
    cam: OrbitCamera,
    size: tuple[int, int] = (800, 800),
    w_m: float = 1.245,
    h_m: float = 1.245,
    grid_m: float = 0.25,
) -> np.ndarray:
    """Render the 3D floor scene, neon grid, falling hazards, and stick figure."""
    W, H = size
    img = np.full((H, W, 3), (22, 22, 26), dtype=np.uint8)

    def draw_seg(p1, p2, color, thick):
        if not (np.isnan(p1).any() or np.isnan(p2).any()):
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])
            # Clip roughly to screen boundary to avoid OpenCV integer overflow
            if -W <= x1 <= 2 * W and -H <= y1 <= 2 * H and -W <= x2 <= 2 * W and -H <= y2 <= 2 * H:
                cv2.line(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

    # 1. Neon floor grid
    grid_pts = []
    x = 0.0
    while x <= w_m + 1e-5:
        grid_pts.extend([(x, 0.0, 0.0), (x, h_m, 0.0)])
        x += grid_m
    y = 0.0
    while y <= h_m + 1e-5:
        grid_pts.extend([(0.0, y, 0.0), (w_m, y, 0.0)])
        y += grid_m

    if grid_pts:
        proj_grid = project(grid_pts, cam, size)
        for i in range(0, len(proj_grid), 2):
            draw_seg(proj_grid[i], proj_grid[i + 1], (140, 110, 0), 1)

    # Taped box outer boundary (bright cyan)
    corners = [(0.0, 0.0, 0.0), (w_m, 0.0, 0.0), (w_m, h_m, 0.0), (0.0, h_m, 0.0)]
    proj_corners = project(corners, cam, size)
    for i in range(4):
        draw_seg(proj_corners[i], proj_corners[(i + 1) % 4], (255, 230, 0), 2)

    # Scale indicators & reference marks
    # Metric tick marks and labels every 0.5 m along near edge (y=0) and left edge (x=0)
    x_tick = 0.5
    while x_tick <= w_m + 1e-5:
        p_base = project([(x_tick, 0.0, 0.0)], cam, size)[0]
        p_ext = project([(x_tick, -0.04, 0.0)], cam, size)[0]
        draw_seg(p_base, p_ext, (180, 180, 180), 1)
        if not np.isnan(p_ext).any():
            cv2.putText(
                img, f"{x_tick:.1f}m", (int(p_ext[0]) - 14, int(p_ext[1]) + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA,
            )
        x_tick += 0.5

    y_tick = 0.5
    while y_tick <= h_m + 1e-5:
        p_base = project([(0.0, y_tick, 0.0)], cam, size)[0]
        p_ext = project([(-0.04, y_tick, 0.0)], cam, size)[0]
        draw_seg(p_base, p_ext, (180, 180, 180), 1)
        if not np.isnan(p_ext).any():
            cv2.putText(
                img, f"{y_tick:.1f}m", (int(p_ext[0]) - 38, int(p_ext[1]) + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA,
            )
        y_tick += 0.5

    # Scale reference: thin vertical line 1.0 m tall at far-left corner (0, h_m, 0), labelled "1 m"
    ref_base = project([(0.0, h_m, 0.0)], cam, size)[0]
    ref_top = project([(0.0, h_m, 1.0)], cam, size)[0]
    draw_seg(ref_base, ref_top, (220, 220, 220), 1)
    if not np.isnan(ref_top).any():
        cv2.putText(
            img, "1 m", (int(ref_top[0]) - 32, int(ref_top[1]) + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
        )

    # Real floor dimension text
    cv2.putText(
        img, f"floor {w_m:.3f} x {h_m:.3f} m", (15, H - 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
    )

    # 2. Falling hazards
    if game is not None:
        for hz in getattr(game, "hazards", []):
            hs = hz.size / 2.0
            cx, cy = hz.cx, hz.cy
            floor_sq = [
                (cx - hs, cy - hs, 0.0),
                (cx + hs, cy - hs, 0.0),
                (cx + hs, cy + hs, 0.0),
                (cx - hs, cy + hs, 0.0),
            ]
            proj_sq = project(floor_sq, cam, size)

            if hz.age < dbt.WARN_S:
                # Warning phase: orange landing outline + falling wireframe cube
                for i in range(4):
                    draw_seg(proj_sq[i], proj_sq[(i + 1) % 4], (0, 165, 255), 2)

                z_bot = 2.5 * max(0.0, 1.0 - (hz.age / dbt.WARN_S))
                z_top = z_bot + hz.size
                cube_pts = [
                    (cx - hs, cy - hs, z_bot),
                    (cx + hs, cy - hs, z_bot),
                    (cx + hs, cy + hs, z_bot),
                    (cx - hs, cy + hs, z_bot),
                    (cx - hs, cy - hs, z_top),
                    (cx + hs, cy - hs, z_top),
                    (cx + hs, cy + hs, z_top),
                    (cx - hs, cy + hs, z_top),
                ]
                proj_cube = project(cube_pts, cam, size)
                # Bottom square
                for i in range(4):
                    draw_seg(proj_cube[i], proj_cube[(i + 1) % 4], (0, 200, 255), 2)
                # Top square
                for i in range(4):
                    draw_seg(proj_cube[i + 4], proj_cube[((i + 1) % 4) + 4], (0, 200, 255), 2)
                # Vertical pillars
                for i in range(4):
                    draw_seg(proj_cube[i], proj_cube[i + 4], (0, 200, 255), 2)
            else:
                # Impact phase: filled red square on floor + BOOM text
                if not np.isnan(proj_sq).any():
                    pts_poly = proj_sq.astype(np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(img, [pts_poly], (0, 0, 200))
                    for i in range(4):
                        draw_seg(proj_sq[i], proj_sq[(i + 1) % 4], (255, 255, 255), 2)

                mid_pt = project([(cx, cy, hz.size * 0.5)], cam, size)[0]
                if not np.isnan(mid_pt).any():
                    cv2.putText(
                        img, "BOOM", (int(mid_pt[0]) - 30, int(mid_pt[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                    )

    # 3. Stick figure + faint shadow
    if skeleton is not None and len(skeleton.points_m) == 33:
        # Faint floor shadow (same points with z = 0)
        shadow_pts_3d = [(p[0], p[1], 0.0) for p in skeleton.points_m]
        proj_shadow = project(shadow_pts_3d, cam, size)
        for i, j in BONES:
            draw_seg(proj_shadow[i], proj_shadow[j], (45, 45, 52), 2)

        # Neck shadow on floor
        p11 = skeleton.points_m[11]
        p12 = skeleton.points_m[12]
        neck_base_shadow_3d = ((p11[0] + p12[0]) / 2.0, (p11[1] + p12[1]) / 2.0, 0.0)
        proj_neck_base_shadow = project([neck_base_shadow_3d], cam, size)[0]
        draw_seg(proj_neck_base_shadow, proj_shadow[0], (45, 45, 52), 2)

        # 33-point stick figure
        proj_skel = project(skeleton.points_m, cam, size)

        # Draw bones
        for i, j in BONES:
            vi = skeleton.visibility[i] if i < len(skeleton.visibility) else 1.0
            vj = skeleton.visibility[j] if j < len(skeleton.visibility) else 1.0
            is_dim = (vi < 0.5 or vj < 0.5)

            if (i, j) in TORSO_BONES or (j, i) in TORSO_BONES:
                thick = 4
                color = (70, 110, 70) if is_dim else (0, 255, 128)
            else:
                thick = 2
                color = (70, 90, 110) if is_dim else (0, 230, 255)

            draw_seg(proj_skel[i], proj_skel[j], color, thick)

        # Draw neck line (drawing only) from shoulder midpoint (11, 12) to nose (0)
        neck_base_3d = (
            (p11[0] + p12[0]) / 2.0,
            (p11[1] + p12[1]) / 2.0,
            (p11[2] + p12[2]) / 2.0,
        )
        proj_neck_base = project([neck_base_3d], cam, size)[0]
        v11 = skeleton.visibility[11] if 11 < len(skeleton.visibility) else 1.0
        v12 = skeleton.visibility[12] if 12 < len(skeleton.visibility) else 1.0
        v0 = skeleton.visibility[0] if 0 < len(skeleton.visibility) else 1.0
        neck_dim = (v11 < 0.5 or v12 < 0.5 or v0 < 0.5)
        neck_color = (70, 110, 70) if neck_dim else (0, 255, 128)
        draw_seg(proj_neck_base, proj_skel[0], neck_color, 3)

        # Head circle around nose/ears
        nose = proj_skel[0]
        if not np.isnan(nose).any():
            cv2.circle(img, (int(nose[0]), int(nose[1])), 14, (0, 255, 255), 2, cv2.LINE_AA)

        # Joint dots
        for idx, pt in enumerate(proj_skel):
            if not np.isnan(pt).any():
                v = skeleton.visibility[idx] if idx < len(skeleton.visibility) else 1.0
                jcolor = (100, 100, 110) if v < 0.5 else (255, 255, 255)
                cv2.circle(img, (int(pt[0]), int(pt[1])), 3, jcolor, -1, cv2.LINE_AA)

    return img


def is_foot_out(pt: tuple[float, float], w_m: float, h_m: float, margin_m: float = 0.0) -> bool:
    """Check if foot point is outside or on the boundary margin."""
    x, y = pt
    return (x <= margin_m) or (x >= w_m - margin_m) or (y <= margin_m) or (y >= h_m - margin_m)


def extract_foot_points(pf: PoseFrame | None, kb_pos: tuple[float, float] | None = None) -> list[tuple[float, float]]:
    """Extract foot points from PoseFrame or keyboard position."""
    if kb_pos is not None:
        return [kb_pos]
    if pf is None:
        return []
    if pf.left_foot_m is not None and pf.right_foot_m is not None:
        return [pf.left_foot_m, pf.right_foot_m]
    elif pf.left_foot_m is not None:
        return [pf.left_foot_m]
    elif pf.right_foot_m is not None:
        return [pf.right_foot_m]
    elif pf.floor_x_m is not None and pf.floor_y_m is not None:
        return [(pf.floor_x_m, pf.floor_y_m)]
    return []


@dataclass
class ViewerState:
    game: dbt.DropBoxGame
    cam: OrbitCamera
    builder: SkeletonBuilder
    w_m: float
    h_m: float
    edge_margin: float = 0.0
    out_hold: float = 0.2
    arm_s: float = 1.0
    grid_m: float = 0.25
    mapper: FloorMapper | None = None
    keyboard: bool = False

    # Dynamic state
    armed: bool = False
    arm_timer: float = 0.0
    out_timer: float = 0.0
    game_over: bool = False
    offending_point: tuple[float, float] | None = None
    last_canvas: np.ndarray | None = None
    frozen_skel: Skeleton3D | None = None
    frozen_cam_view: np.ndarray | None = None
    last_pf: PoseFrame | None = None

    # Keyboard navigation state
    kb_x: float = 0.6225
    kb_y: float = 0.6225
    jump_start_t: float = 0.0
    jump_duration: float = 0.5
    pose_fps: float = 60.0

    def reset(self) -> None:
        """Reset game and viewer state back to disarmed."""
        self.game.reset()
        self.armed = False
        self.arm_timer = 0.0
        self.out_timer = 0.0
        self.game_over = False
        self.offending_point = None
        self.last_canvas = None
        self.frozen_skel = None
        self.frozen_cam_view = None
        self.last_pf = None


def frame_step(state: ViewerState, source: Any | None, dt: float, now: float | None = None) -> np.ndarray:
    """Execute a single frame step of game update, out-of-box checking, and scene rendering."""
    if now is None:
        now = time.monotonic()

    # If already in GAME OVER, freeze scene: keep rendering the last frame without updating
    if state.game_over:
        canvas = render_scene(
            state.frozen_skel,
            state.game,
            state.cam,
            size=(800, 800),
            w_m=state.w_m,
            h_m=state.h_m,
            grid_m=state.grid_m,
        )
        if state.frozen_cam_view is not None:
            preview = state.frozen_cam_view.copy()
            if state.mapper is not None:
                preview = dbt.draw_debug(preview, state.mapper, state.last_pf or PoseFrame(), {}, False)
            thumb_w, thumb_h = 200, 150
            thumb = cv2.resize(preview, (thumb_w, thumb_h))
            canvas[10:10 + thumb_h, canvas.shape[1] - 10 - thumb_w:canvas.shape[1] - 10] = thumb
            cv2.rectangle(
                canvas, (canvas.shape[1] - 10 - thumb_w, 10), (canvas.shape[1] - 10, 10 + thumb_h),
                (180, 180, 180), 1
            )
        if state.offending_point is not None:
            ox, oy = state.offending_point
            proj_off = project([(ox, oy, 0.0)], state.cam, (canvas.shape[1], canvas.shape[0]))[0]
            if not np.isnan(proj_off).any():
                cv2.circle(canvas, (int(proj_off[0]), int(proj_off[1])), 8, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (int(proj_off[0]), int(proj_off[1])), 12, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "OUT", (int(proj_off[0]) + 12, int(proj_off[1]) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

        overlay = canvas.copy()
        cv2.rectangle(overlay, (80, 260), (720, 540), (14, 14, 18), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (80, 260), (720, 540), (0, 0, 255), 3)

        cv2.putText(canvas, "GAME OVER", (canvas.shape[1] // 2 - 165, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
        final_txt = f"Score: {state.game.score}   Dodges: {state.game.dodges}   Hits: {state.game.hits}"
        cv2.putText(canvas, final_txt, (canvas.shape[1] // 2 - 215, 395),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        reason_txt = "Player stepped out of the box!"
        cv2.putText(canvas, reason_txt, (canvas.shape[1] // 2 - 160, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)
        restart_txt = "Press R to restart, Q to quit"
        cv2.putText(canvas, restart_txt, (canvas.shape[1] // 2 - 165, 490),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        key_help = "J/L: yaw | I/K: pitch | U/O: zoom | 1: default 2: side 3: top | R: restart | Q: quit"
        cv2.putText(canvas, key_help, (15, canvas.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        state.last_canvas = canvas
        return canvas

    # Active frame
    cam_view = None
    debug = {}
    pose_fps = state.pose_fps

    if state.keyboard:
        in_jump = (now - state.jump_start_t) < state.jump_duration
        if in_jump:
            tau = (now - state.jump_start_t) / state.jump_duration
            arc = 4.0 * tau * (1.0 - tau)
            hip_lift = 0.8 * arc
        else:
            hip_lift = 0.0

        skel = state.builder.build(
            SYNTHETIC_WORLD_LANDMARKS,
            floor_pos=(state.kb_x, state.kb_y),
            hip_lift=hip_lift,
            ts=now,
            is_jumping=in_jump,
        )
        pf = PoseFrame(
            timestamp=now,
            tracked=True,
            floor_x_m=state.kb_x,
            floor_y_m=state.kb_y,
            airborne=in_jump,
        )
        feet = extract_foot_points(pf, kb_pos=(state.kb_x, state.kb_y))
        pose_fps = 60.0
    else:
        if source is not None and getattr(source, "error", None):
            sys.exit(f"Tracking error: {source.error}")
        if source is not None:
            pf, skel, frame, debug = source.snapshot3d()
            cam_view = frame
            pose_fps = getattr(source, "pose_fps", 0.0)
            feet = extract_foot_points(pf)
        else:
            pf = PoseFrame()
            skel = None
            feet = []

    state.last_pf = pf

    # Out condition check
    any_out = any(is_foot_out(pt, state.w_m, state.h_m, state.edge_margin) for pt in feet) if feet else False
    offending = next((pt for pt in feet if is_foot_out(pt, state.w_m, state.h_m, state.edge_margin)), None)

    all_inside = (len(feet) > 0) and not any_out
    if not state.armed:
        if pf.tracked and not pf.airborne and all_inside:
            state.arm_timer += dt
            if state.arm_timer >= state.arm_s:
                state.armed = True
        else:
            state.arm_timer = 0.0
        pl_in_box = False
    else:
        pl_in_box = pf.tracked
        if pf.tracked and not pf.airborne:
            if any_out:
                state.out_timer += dt
                if state.out_timer >= state.out_hold:
                    state.game_over = True
                    state.offending_point = offending
                    state.frozen_skel = skel
                    state.frozen_cam_view = cam_view
            else:
                state.out_timer = 0.0
        else:
            pass

    # Update game if not game_over
    if not state.game_over:
        px = pf.floor_x_m if (pf.floor_x_m is not None) else (state.w_m / 2.0)
        py = pf.floor_y_m if (pf.floor_y_m is not None) else (state.h_m / 2.0)
        pl = dbt.Player(
            tracked=bool(pf.tracked),
            in_box=pl_in_box,
            x=float(px),
            y=float(py),
            airborne=bool(pf.airborne),
        )
        state.game.update(dt, pl)

    # Render scene
    render_skel = state.frozen_skel if state.game_over else skel
    canvas = render_scene(
        render_skel,
        state.game,
        state.cam,
        size=(800, 800),
        w_m=state.w_m,
        h_m=state.h_m,
        grid_m=state.grid_m,
    )

    # Camera preview with floor box and foot points drawn
    render_cam_view = state.frozen_cam_view if state.game_over else cam_view
    if render_cam_view is not None:
        preview = render_cam_view.copy()
        if state.mapper is not None:
            preview = dbt.draw_debug(preview, state.mapper, pf, debug, pl_in_box)
        thumb_w, thumb_h = 200, 150
        thumb = cv2.resize(preview, (thumb_w, thumb_h))
        canvas[10:10 + thumb_h, canvas.shape[1] - 10 - thumb_w:canvas.shape[1] - 10] = thumb
        cv2.rectangle(
            canvas, (canvas.shape[1] - 10 - thumb_w, 10), (canvas.shape[1] - 10, 10 + thumb_h),
            (180, 180, 180), 1
        )

    # HUD text
    score_txt = f"Score: {state.game.score}   Dodges: {state.game.dodges}   Hits: {state.game.hits}   Pose: {pose_fps:.0f} FPS"
    cv2.putText(canvas, score_txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    up_str = str(state.builder.up_sign) if state.builder.up_sign is not None else "evaluating"
    jump_h = render_skel.jump_height_m if render_skel is not None else 0.0
    axis_txt = f"up_sign: {up_str} | pitch: {CAMERA_PITCH_DEG:.1f} deg | jump_z: {jump_h:.2f} m"
    cv2.putText(canvas, axis_txt, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

    # Disarmed notification
    if not state.armed and not state.game_over:
        cv2.putText(
            canvas, "STAND INSIDE THE BOX TO START",
            (canvas.shape[1] // 2 - 240, canvas.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA,
        )

    # If game over triggered in this frame, draw overlay
    if state.game_over:
        if state.offending_point is not None:
            ox, oy = state.offending_point
            proj_off = project([(ox, oy, 0.0)], state.cam, (canvas.shape[1], canvas.shape[0]))[0]
            if not np.isnan(proj_off).any():
                cv2.circle(canvas, (int(proj_off[0]), int(proj_off[1])), 8, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (int(proj_off[0]), int(proj_off[1])), 12, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "OUT", (int(proj_off[0]) + 12, int(proj_off[1]) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

        overlay = canvas.copy()
        cv2.rectangle(overlay, (80, 260), (720, 540), (14, 14, 18), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (80, 260), (720, 540), (0, 0, 255), 3)

        cv2.putText(canvas, "GAME OVER", (canvas.shape[1] // 2 - 165, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
        final_txt = f"Score: {state.game.score}   Dodges: {state.game.dodges}   Hits: {state.game.hits}"
        cv2.putText(canvas, final_txt, (canvas.shape[1] // 2 - 215, 395),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        reason_txt = "Player stepped out of the box!"
        cv2.putText(canvas, reason_txt, (canvas.shape[1] // 2 - 160, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)
        restart_txt = "Press R to restart, Q to quit"
        cv2.putText(canvas, restart_txt, (canvas.shape[1] // 2 - 165, 490),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

    key_help = "J/L: yaw | I/K: pitch | U/O: zoom | 1: default 2: side 3: top | R: restart | Q: quit"
    cv2.putText(canvas, key_help, (15, canvas.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    state.last_canvas = canvas
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description="3D skeleton viewer & drop box falling test.")
    ap.add_argument("--keyboard", action="store_true", help="no camera, WASD + SPACE")
    ap.add_argument("--camera", type=int, default=0, help="camera device index (default: 0)")
    ap.add_argument("--floor-w", type=float, default=1.245, help="taped box width (m)")
    ap.add_argument("--floor-h", type=float, default=1.245, help="taped box depth (m)")
    ap.add_argument("--grid", type=float, default=0.25, help="grid spacing (m)")
    ap.add_argument("--fall-s", type=float, default=1.5, help="fall warning duration (s)")
    ap.add_argument("--spawn-s", type=float, default=2.0, help="seconds between hazards")
    ap.add_argument("--view-fov", type=float, default=45.0, help="virtual camera lens field of view (deg)")
    ap.add_argument("--edge-margin", type=float, default=0.0, help="edge margin in metres for out-of-box (default: 0.0)")
    ap.add_argument("--out-hold", type=float, default=0.2, help="seconds out of box before game over (default: 0.2)")
    ap.add_argument("--arm-s", type=float, default=1.0, help="seconds inside box to arm the game (default: 1.0)")
    args = ap.parse_args()

    # Configure drop_box_test module attributes before instantiating DropBoxGame
    dbt.WARN_S = args.fall_s
    dbt.SPAWN_EVERY_S = args.spawn_s
    dbt.IMPACT_S = 0.3

    w_m, h_m = args.floor_w, args.floor_h
    source = mapper = None

    if not args.keyboard:
        mapper = FloorMapper.load()
        if mapper is None:
            sys.exit("Run: python -m tools.calibrate_floor --floor-w 1.245 --floor-h 1.245")
        if not Path("models/pose_landmarker_lite.task").exists():
            sys.exit("Run: python -m tools.get_models")
        w_m, h_m = mapper.w_m, mapper.h_m

        try:
            source = CameraPoseSource(args.camera, mapper=mapper)
            source.start()
        except FileNotFoundError:
            sys.exit("Run: python -m tools.get_models")
        except RuntimeError as e:
            sys.exit(str(e))
        atexit.register(source.stop)

        print("Waiting for camera and pose model to initialize...")
        t0 = time.monotonic()
        while source.latest_frame() is None:
            if source.error:
                sys.exit(f"Tracking error: {source.error}")
            if time.monotonic() - t0 > 25.0:
                sys.exit("No frames from the camera.")
            time.sleep(0.05)

    game = dbt.DropBoxGame(w_m, h_m)
    cam = OrbitCamera.default(w_m, h_m, fov_deg=args.view_fov)
    builder = SkeletonBuilder(camera_pitch_deg=CAMERA_PITCH_DEG)

    state = ViewerState(
        game=game,
        cam=cam,
        builder=builder,
        w_m=w_m,
        h_m=h_m,
        edge_margin=args.edge_margin,
        out_hold=args.out_hold,
        arm_s=args.arm_s,
        grid_m=args.grid,
        mapper=mapper,
        keyboard=args.keyboard,
        kb_x=w_m / 2.0,
        kb_y=h_m / 2.0,
    )

    last_t = time.monotonic()

    print("Controls:")
    print("  J / L: rotate yaw left/right")
    print("  I / K: tilt pitch up/down")
    print("  U / O: zoom in/out")
    print("  1: default view | 2: side view | 3: top-down view")
    print("  R: restart | Q: quit")
    print("  WASD + SPACE (keyboard mode)")

    try:
        while True:
            now = time.monotonic()
            dt = min(now - last_t, 0.1)
            last_t = now

            canvas = frame_step(state, source, dt, now=now)

            cv2.imshow("3D Skeleton & Falling Boxes", canvas)
            k = cv2.waitKey(20) & 0xFF

            if k == ord("q"):
                break
            elif k in (ord("r"), ord("R")):
                state.reset()
            elif k == ord("1"):
                state.cam = OrbitCamera.default(state.w_m, state.h_m, fov_deg=state.cam.fov_deg)
            elif k == ord("2"):
                state.cam = OrbitCamera.side(state.w_m, state.h_m, fov_deg=state.cam.fov_deg)
            elif k == ord("3"):
                state.cam = OrbitCamera.top_down(state.w_m, state.h_m, fov_deg=state.cam.fov_deg)
            elif k == ord("j"):
                state.cam.yaw_deg = (state.cam.yaw_deg + 5.0) % 360.0
            elif k == ord("l"):
                state.cam.yaw_deg = (state.cam.yaw_deg - 5.0) % 360.0
            elif k == ord("i"):
                state.cam.pitch_deg = min(89.0, state.cam.pitch_deg + 5.0)
            elif k == ord("k"):
                state.cam.pitch_deg = max(-89.0, state.cam.pitch_deg - 5.0)
            elif k == ord("u"):
                state.cam.distance = max(0.5, state.cam.distance - 0.2)
            elif k == ord("o"):
                state.cam.distance = min(10.0, state.cam.distance + 0.2)

            if state.keyboard and not state.game_over:
                step = 0.06
                if k == ord("a"):
                    state.kb_x = max(-0.1, state.kb_x - step)
                if k == ord("d"):
                    state.kb_x = min(state.w_m + 0.1, state.kb_x + step)
                if k == ord("w"):
                    state.kb_y = min(state.h_m + 0.1, state.kb_y + step)
                if k == ord("s"):
                    state.kb_y = max(-0.1, state.kb_y - step)
                if k == ord(" "):
                    state.jump_start_t = time.monotonic()
    finally:
        if source is not None:
            source.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
