"""Wall Shape (Game B) baseline test harness and interactive pose matcher.

Follows Hard Rules:
  - NO game rules: NO total score, NO lives, NO game over.
  - Displays: Live match score (0..100%), Hold progress, and Per-wall result.
  - Mirrored camera view with oncoming wall cut-out silhouette and color-coded skeleton.
  - Real-time joint breakdown table with directional adjustment hints.
  - Interactive --keyboard mode for offline testing without a webcam.
"""
from __future__ import annotations

import argparse
import atexit
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from pose_core.floor_map import FloorMapper
from pose_core.pose_matcher import (
    DEFAULT_TEMPLATES,
    JointComparison,
    PoseMatchResult,
    PoseTemplate,
    evaluate_pose,
    extract_angles_from_skeleton,
    extract_pose_angles,
    snapshot_pose_match,
)
from shared.contracts import PoseFrame

# ==============================================================================
# UI & Theme Constants
# ==============================================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX
STAGE_W, STAGE_H = 680, 560
SIDEBAR_W = 420
CANVAS_W, CANVAS_H = STAGE_W + SIDEBAR_W, STAGE_H

# Colors (BGR)
C_BG = (16, 17, 22)
C_PANEL_BG = (22, 24, 32)
C_CARD_BG = (28, 32, 42)
C_BORDER = (45, 52, 68)

C_TEXT_WHITE = (245, 248, 255)
C_TEXT_GREY = (160, 168, 185)
C_TEXT_MUTED = (110, 118, 135)

C_PASS = (0, 230, 115)         # Neon Green
C_WARN = (0, 170, 255)         # Bright Amber / Orange
C_FAIL = (60, 60, 240)         # Crimson Red
C_CYAN = (255, 225, 0)         # Cyan (BGR)
C_WALL_ACCENT = (255, 210, 40) # Glowing Cyan/Sky Blue
C_WALL_FILL = (35, 45, 60)     # Translucent Wall Tint

# Standard MediaPipe Joint Indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

CORE_BONES: list[tuple[int, int]] = [
    (11, 12),           # Shoulders
    (11, 13), (13, 15), # Left Arm
    (12, 14), (14, 16), # Right Arm
    (11, 23), (12, 24), # Torso sides
    (23, 24),           # Hips
    (23, 25), (25, 27), # Left Leg
    (24, 26), (26, 28), # Right Leg
    (0, 11),  (0, 12),  # Neck / Head
]


# ==============================================================================
# Reference 2D Posture Generator (Target Wall Cut-Out Silhouettes)
# ==============================================================================

def make_template_ref_posture(template_id: str) -> list[tuple[float, float, float]]:
    """Return 33 normalized 2D/3D reference coordinates for target wall silhouettes.

    Mirrored display coordinates:
      - x = 0.0 is screen-left (player's left limbs)
      - x = 1.0 is screen-right (player's right limbs)
      - y = 0.0 is top, y = 1.0 is bottom
    """
    pts = [(0.0, 0.0, 0.0)] * 33
    pts[0] = (0.50, 0.25, 0.0)  # Nose/Head

    # Torso
    pts[11] = (0.43, 0.38, 0.0)  # L_SHOULDER
    pts[12] = (0.57, 0.38, 0.0)  # R_SHOULDER
    pts[23] = (0.45, 0.60, 0.0)  # L_HIP
    pts[24] = (0.55, 0.60, 0.0)  # R_HIP

    # Default standing legs
    pts[25] = (0.45, 0.76, 0.0)  # L_KNEE
    pts[26] = (0.55, 0.76, 0.0)  # R_KNEE
    pts[27] = (0.45, 0.92, 0.0)  # L_ANKLE
    pts[28] = (0.55, 0.92, 0.0)  # R_ANKLE

    tid = template_id.lower()
    if tid == "t_pose":
        pts[13] = (0.29, 0.38, 0.0); pts[15] = (0.15, 0.38, 0.0)
        pts[14] = (0.71, 0.38, 0.0); pts[16] = (0.85, 0.38, 0.0)
    elif tid == "star":
        pts[13] = (0.33, 0.28, 0.0); pts[15] = (0.23, 0.18, 0.0)
        pts[14] = (0.67, 0.28, 0.0); pts[16] = (0.77, 0.18, 0.0)
        pts[25] = (0.40, 0.76, 0.0); pts[27] = (0.35, 0.92, 0.0)
        pts[26] = (0.60, 0.76, 0.0); pts[28] = (0.65, 0.92, 0.0)
    elif tid == "cactus":
        pts[13] = (0.29, 0.38, 0.0); pts[15] = (0.29, 0.24, 0.0)
        pts[14] = (0.71, 0.38, 0.0); pts[16] = (0.71, 0.24, 0.0)
    elif tid == "disco_right":
        pts[14] = (0.67, 0.26, 0.0); pts[16] = (0.77, 0.14, 0.0)
        pts[13] = (0.39, 0.52, 0.0); pts[15] = (0.35, 0.66, 0.0)
    elif tid == "disco_left":
        pts[13] = (0.33, 0.26, 0.0); pts[15] = (0.23, 0.14, 0.0)
        pts[14] = (0.61, 0.52, 0.0); pts[16] = (0.65, 0.66, 0.0)
    elif tid == "tree_pose":
        pts[13] = (0.33, 0.28, 0.0); pts[15] = (0.23, 0.18, 0.0)
        pts[14] = (0.67, 0.28, 0.0); pts[16] = (0.77, 0.18, 0.0)
        pts[26] = (0.63, 0.70, 0.0); pts[28] = (0.48, 0.76, 0.0)
    elif tid == "l_pose":
        pts[13] = (0.43, 0.24, 0.0); pts[15] = (0.43, 0.10, 0.0)
        pts[14] = (0.71, 0.38, 0.0); pts[16] = (0.85, 0.38, 0.0)
    elif tid == "squat":
        pts[11] = (0.43, 0.44, 0.0); pts[12] = (0.57, 0.44, 0.0)
        pts[23] = (0.45, 0.64, 0.0); pts[24] = (0.55, 0.64, 0.0)
        pts[25] = (0.37, 0.75, 0.0); pts[27] = (0.45, 0.90, 0.0)
        pts[26] = (0.63, 0.75, 0.0); pts[28] = (0.55, 0.90, 0.0)
        pts[13] = (0.31, 0.45, 0.0); pts[15] = (0.18, 0.45, 0.0)
        pts[14] = (0.69, 0.45, 0.0); pts[16] = (0.82, 0.45, 0.0)
    else:
        # Default T-Pose fallback
        pts[13] = (0.29, 0.38, 0.0); pts[15] = (0.15, 0.38, 0.0)
        pts[14] = (0.71, 0.38, 0.0); pts[16] = (0.85, 0.38, 0.0)

    return pts


# ==============================================================================
# Wall Shape Test Engine State (Rule 3 Compliant: No Total Score, No Lives)
# ==============================================================================

@dataclass
class WallShapeTestState:
    templates: list[PoseTemplate] = field(default_factory=lambda: list(DEFAULT_TEMPLATES))
    template_idx: int = 0
    angle_tolerance_deg: float = 20.0
    required_hold_s: float = 1.0
    wall_approach_s: float = 7.0

    # Live approach loop
    time_to_impact_s: float = 7.0
    hold_elapsed_s: float = 0.0

    # Per-Wall Result (Transient banner, Rule 3)
    wall_state: str = "APPROACHING"  # APPROACHING | PASSED | MISSED
    result_banner_msg: str = ""
    result_banner_col: tuple[int, int, int] = C_PASS
    result_display_s: float = 0.0
    last_wall_result: str = "None"

    # Synthetic Keyboard Player State
    keyboard_mode: bool = False
    kb_posture_id: str = "t_pose"
    kb_arm_offset_deg: float = 0.0
    kb_pos_m: tuple[float, float] = (0.62, 0.62)

    def current_template(self) -> PoseTemplate:
        return self.templates[self.template_idx % len(self.templates)]

    def next_wall(self) -> None:
        self.template_idx = (self.template_idx + 1) % len(self.templates)
        self.time_to_impact_s = self.wall_approach_s
        self.hold_elapsed_s = 0.0
        self.wall_state = "APPROACHING"

    def select_template(self, index: int) -> None:
        if 0 <= index < len(self.templates):
            self.template_idx = index
            self.time_to_impact_s = self.wall_approach_s
            self.hold_elapsed_s = 0.0
            self.wall_state = "APPROACHING"

    def update(self, dt: float, match_result: PoseMatchResult) -> None:
        """Update wall approach countdown, hold timer, and per-wall result transitions."""
        if self.wall_state == "APPROACHING":
            self.time_to_impact_s = max(0.0, self.time_to_impact_s - dt)

            # Hold progress accumulates if pose matches
            if match_result.is_full_match:
                self.hold_elapsed_s = min(self.required_hold_s, self.hold_elapsed_s + dt)
            else:
                # Slight decay on imperfect pose to reward sustained posture
                self.hold_elapsed_s = max(0.0, self.hold_elapsed_s - 1.2 * dt)

            # Wall Outcome Judgement (Rule 3: Per-wall result only, no game over)
            if self.hold_elapsed_s >= self.required_hold_s:
                self.wall_state = "PASSED"
                self.result_banner_msg = "WALL PASSED! (Match Held)"
                self.result_banner_col = C_PASS
                self.result_display_s = 1.5
                self.last_wall_result = f"PASSED ({self.current_template().name})"
            elif self.time_to_impact_s <= 0.0:
                self.wall_state = "MISSED"
                self.result_banner_msg = "WALL MISSED! (Timed Out)"
                self.result_banner_col = C_FAIL
                self.result_display_s = 1.5
                self.last_wall_result = f"MISSED ({self.current_template().name})"

        elif self.wall_state in ("PASSED", "MISSED"):
            self.result_display_s -= dt
            if self.result_display_s <= 0.0:
                self.next_wall()


# ==============================================================================
# Rendering Pipeline
# ==============================================================================

def render_stage_view(
    frame: np.ndarray | None,
    state: WallShapeTestState,
    match_res: PoseMatchResult,
    player_pts: Sequence[Any] | None,
    mapper: FloorMapper | None = None,
    pf: PoseFrame | None = None,
    debug: dict | None = None,
) -> np.ndarray:
    """Render mirrored camera feed, floor grid/standing zone, oncoming cut-out wall, and player skeleton."""
    if frame is not None:
        view = cv2.resize(frame, (STAGE_W, STAGE_H))
    else:
        # Dark studio stage canvas for keyboard / offline mode
        view = np.full((STAGE_H, STAGE_W, 3), C_BG, dtype=np.uint8)
        # Perspective floor grid
        for gy in range(STAGE_H // 2, STAGE_H, 24):
            cv2.line(view, (0, gy), (STAGE_W, gy), (28, 30, 40), 1, cv2.LINE_AA)
        for gx in range(0, STAGE_W, 60):
            cv2.line(view, (STAGE_W // 2, STAGE_H // 2), (gx, STAGE_H), (28, 30, 40), 1, cv2.LINE_AA)

    # 1. Perspective Floor Grid & Standing Target Zone on Live Video
    if mapper is not None and frame is not None:
        w_m, h_m = mapper.w_m, mapper.h_m
        grid_overlay = view.copy()

        # 3x3 floor grid
        cols, rows = 3, 3
        dx, dy = w_m / cols, h_m / rows
        for c in range(cols + 1):
            x = min(w_m, c * dx)
            p1 = mapper.to_pixels(x, 0.0)
            p2 = mapper.to_pixels(x, h_m)
            cv2.line(grid_overlay, (int(round(p1[0])), int(round(p1[1]))),
                     (int(round(p2[0])), int(round(p2[1]))), (140, 130, 60), 1, cv2.LINE_AA)

        for r in range(rows + 1):
            y = min(h_m, r * dy)
            p1 = mapper.to_pixels(0.0, y)
            p2 = mapper.to_pixels(w_m, y)
            cv2.line(grid_overlay, (int(round(p1[0])), int(round(p1[1]))),
                     (int(round(p2[0])), int(round(p2[1]))), (140, 130, 60), 1, cv2.LINE_AA)

        # Center target standing zone (Center 1x1 tile)
        c_corners_m = [
            (dx, dy), (2 * dx, dy), (2 * dx, 2 * dy), (dx, 2 * dy)
        ]
        c_corners_px = np.array([mapper.to_pixels(x, y) for x, y in c_corners_m], dtype=np.int32)
        in_pos = match_res.pos_matched
        tile_tint = (0, 90, 0) if in_pos else (90, 70, 0)
        cv2.fillPoly(grid_overlay, [c_corners_px], tile_tint)

        # Floor arena boundary
        poly = mapper.corners_px.astype(np.int32).reshape(-1, 1, 2)
        border_col = C_PASS if in_pos else C_WARN
        cv2.polylines(grid_overlay, [poly], True, border_col, 2, cv2.LINE_AA)

        cv2.addWeighted(grid_overlay, 0.45, view, 0.55, 0, view)

        # "STAND HERE" Target Label on floor
        c_px = mapper.to_pixels(w_m / 2.0, h_m / 2.0)
        tag_col = C_PASS if in_pos else C_CYAN
        cv2.circle(view, (int(round(c_px[0])), int(round(c_px[1]))), 6, tag_col, -1, cv2.LINE_AA)
        cv2.putText(view, "STAND HERE", (int(round(c_px[0])) - 46, int(round(c_px[1])) + 4),
                    FONT, 0.38, tag_col, 1, cv2.LINE_AA)

        # Player live feet markers on floor (L & R)
        if pf is not None:
            if getattr(pf, "left_foot_m", None):
                lx, ly = mapper.to_pixels(*pf.left_foot_m)
                cv2.circle(view, (int(round(lx)), int(round(ly))), 6, (255, 120, 0), -1, cv2.LINE_AA)
                cv2.circle(view, (int(round(lx)), int(round(ly))), 8, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(view, "L", (int(round(lx)) - 3, int(round(ly)) + 3), FONT, 0.32, (0, 0, 0), 1)
            if getattr(pf, "right_foot_m", None):
                rx, ry = mapper.to_pixels(*pf.right_foot_m)
                cv2.circle(view, (int(round(rx)), int(round(ry))), 6, (255, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(view, (int(round(rx)), int(round(ry))), 8, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(view, "R", (int(round(rx)) - 3, int(round(ry)) + 3), FONT, 0.32, (0, 0, 0), 1)

    # 2. Oncoming Wall Approaching Border
    tmpl = state.current_template()
    approach_ratio = max(0.0, min(1.0, 1.0 - (state.time_to_impact_s / state.wall_approach_s)))
    depth_pad_x = int(35 * (1.0 - approach_ratio))
    depth_pad_y = int(25 * (1.0 - approach_ratio))

    wall_box = (
        depth_pad_x + 10,
        depth_pad_y + 10,
        STAGE_W - 2 * (depth_pad_x + 10),
        STAGE_H - 2 * (depth_pad_y + 10),
    )
    cv2.rectangle(
        view,
        (wall_box[0], wall_box[1]),
        (wall_box[0] + wall_box[2], wall_box[1] + wall_box[3]),
        C_WALL_ACCENT,
        2,
        cv2.LINE_AA,
    )

    # 3. Target Cut-Out Silhouette (Dashed Cyan/Neon Reference Wireframe)
    ref_pts = make_template_ref_posture(tmpl.id)
    ref_px: list[tuple[int, int]] = []
    w_scale = 1.0 + 0.12 * approach_ratio
    cx, cy = STAGE_W // 2, int(STAGE_H * 0.44)

    for rx, ry, _ in ref_pts:
        px = int(cx + (rx - 0.5) * (STAGE_W * 0.72) * w_scale)
        py = int(cy + (ry - 0.5) * (STAGE_H * 0.68) * w_scale)
        ref_px.append((px, py))

    # Draw target cut-out bones (Translucent thick silhouette)
    silhouette_layer = view.copy()
    for j1, j2 in CORE_BONES:
        if j1 < len(ref_px) and j2 < len(ref_px):
            cv2.line(silhouette_layer, ref_px[j1], ref_px[j2], C_CYAN, 8, cv2.LINE_AA)
    for px, py in ref_px:
        cv2.circle(silhouette_layer, (px, py), 6, C_CYAN, -1, cv2.LINE_AA)
    cv2.addWeighted(silhouette_layer, 0.40, view, 0.60, 0, view)

    # 4. Draw Player Skeleton (Color-coded: Green = within tolerance, Red/Amber = adjust)
    if player_pts and len(player_pts) >= 29:
        player_px: list[tuple[int, int]] = []

        # Check if 3D metric skeleton points from camera
        is_3d_skel = False
        if len(player_pts) > 0 and isinstance(player_pts[0], (tuple, list)) and len(player_pts[0]) >= 3:
            if abs(player_pts[0][0]) < 2.5 and abs(player_pts[0][1]) < 2.5 and (abs(player_pts[0][2]) > 0.05 or any(abs(p[2]) > 0.1 for p in player_pts)):
                is_3d_skel = True

        if is_3d_skel:
            # Determine real anchor pixel on screen
            foot_anchor = None
            if debug and debug.get("foot_px"):
                foot_anchor = (float(debug["foot_px"][0]), float(debug["foot_px"][1]))
            elif pf and getattr(pf, "floor_x_m", None) is not None and mapper is not None:
                foot_anchor = mapper.to_pixels(pf.floor_x_m, pf.floor_y_m)
            else:
                foot_anchor = (float(cx), float(STAGE_H - 100))

            # Compute reliable scale in pixels per metre from arena bounds
            if mapper is not None:
                pts = mapper.corners_px
                w_near_px = math.hypot(pts[0][0] - pts[3][0], pts[0][1] - pts[3][1])
                w_far_px = math.hypot(pts[1][0] - pts[2][0], pts[1][1] - pts[2][1])
                scale_px_m = (w_near_px + w_far_px) / (2.0 * max(0.1, mapper.w_m))
            else:
                scale_px_m = 230.0

            # Scale each 3D joint relative to foot anchor
            fpx = pf.floor_x_m if (pf and getattr(pf, "floor_x_m", None) is not None) else 0.62
            for lm in player_pts:
                dx_m = lm[0] - fpx
                dz_m = lm[2]
                px = int(round(foot_anchor[0] + dx_m * scale_px_m))
                py = int(round(foot_anchor[1] - dz_m * scale_px_m))
                player_px.append((px, py))
        else:
            # 2D normalized points (keyboard mode)
            for lm in player_pts:
                if hasattr(lm, "x") and hasattr(lm, "y"):
                    px = int(lm.x * STAGE_W)
                    py = int(lm.y * STAGE_H)
                elif isinstance(lm, (tuple, list)):
                    rx, ry = lm[0], lm[1]
                    px = int(cx + (rx - 0.5) * (STAGE_W * 0.72) * w_scale)
                    py = int(cy + (ry - 0.5) * (STAGE_H * 0.68) * w_scale)
                else:
                    px, py = 0, 0
                player_px.append((px, py))

        def is_joint_ok(j_idx: int) -> bool:
            """Unconstrained joints (Head, Hips, Wrists, Ankles) default to green unless requested."""
            name_map = {
                11: "left_shoulder",
                12: "right_shoulder",
                13: "left_elbow",
                14: "right_elbow",
                23: "left_hip",
                24: "right_hip",
                25: "left_knee",
                26: "right_knee",
            }
            j_name = name_map.get(j_idx)
            if j_name is None:
                return True
            if j_name in match_res.joint_results:
                return match_res.joint_results[j_name].is_matched
            return True

        # Draw bones
        for j1, j2 in CORE_BONES:
            if j1 < len(player_px) and j2 < len(player_px):
                p1, p2 = player_px[j1], player_px[j2]
                if 0 <= p1[0] < STAGE_W and 0 <= p1[1] < STAGE_H and 0 <= p2[0] < STAGE_W and 0 <= p2[1] < STAGE_H:
                    m1 = is_joint_ok(j1)
                    m2 = is_joint_ok(j2)
                    bone_col = C_PASS if (m1 and m2) else C_WARN
                    cv2.line(view, p1, p2, bone_col, 3, cv2.LINE_AA)

        # Draw joints
        for idx in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if idx < len(player_px):
                pt = player_px[idx]
                if 0 <= pt[0] < STAGE_W and 0 <= pt[1] < STAGE_H:
                    is_ok = is_joint_ok(idx)
                    j_col = C_PASS if is_ok else C_FAIL
                    cv2.circle(view, pt, 6, j_col, -1, cv2.LINE_AA)
                    cv2.circle(view, pt, 8, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. Top Guidance Banner & Per-Wall Result (Rule 3)
    if state.wall_state in ("PASSED", "MISSED"):
        # Big Per-Wall Announcement Banner
        overlay = view.copy()
        cv2.rectangle(overlay, (0, 0), (STAGE_W, 70), (10, 10, 15), -1)
        cv2.addWeighted(overlay, 0.75, view, 0.25, 0, view)
        cv2.putText(view, state.result_banner_msg, (STAGE_W // 2 - 190, 45), FONT, 0.85, state.result_banner_col, 2, cv2.LINE_AA)
    else:
        # Live directional cue
        cue_msg = match_res.primary_hint if match_res.primary_hint else "HOLD POSE!"
        cue_col = C_PASS if match_res.is_full_match else C_WARN
        cv2.rectangle(view, (0, 0), (STAGE_W, 44), (12, 14, 20), -1)
        cv2.putText(view, cue_msg, (16, 29), FONT, 0.60, cue_col, 2, cv2.LINE_AA)
        time_str = f"Approach: {state.time_to_impact_s:.1f}s"
        cv2.putText(view, time_str, (STAGE_W - 165, 29), FONT, 0.52, C_TEXT_WHITE, 1, cv2.LINE_AA)

    # Bottom Hold Progress Indicator on Stage
    hold_ratio = max(0.0, min(1.0, state.hold_elapsed_s / state.required_hold_s))
    cv2.rectangle(view, (16, STAGE_H - 28), (STAGE_W - 16, STAGE_H - 12), (30, 34, 44), -1)
    bar_fill = int((STAGE_W - 32) * hold_ratio)
    if bar_fill > 0:
        bar_col = C_PASS if hold_ratio >= 1.0 else C_CYAN
        cv2.rectangle(view, (16, STAGE_H - 28), (16 + bar_fill, STAGE_H - 12), bar_col, -1)
    hold_txt = f"Hold: {state.hold_elapsed_s:.1f}s / {state.required_hold_s:.1f}s"
    cv2.putText(view, hold_txt, (STAGE_W // 2 - 60, STAGE_H - 16), FONT, 0.38, C_TEXT_WHITE, 1, cv2.LINE_AA)

    return view
    bar_fill = int((STAGE_W - 32) * hold_ratio)
    if bar_fill > 0:
        bar_col = C_PASS if hold_ratio >= 1.0 else C_CYAN
        cv2.rectangle(view, (16, STAGE_H - 28), (16 + bar_fill, STAGE_H - 12), bar_col, -1)
    hold_txt = f"Hold: {state.hold_elapsed_s:.1f}s / {state.required_hold_s:.1f}s"
    cv2.putText(view, hold_txt, (STAGE_W // 2 - 60, STAGE_H - 16), FONT, 0.38, C_TEXT_WHITE, 1, cv2.LINE_AA)

    return view


def render_sidebar_panel(
    state: WallShapeTestState,
    match_res: PoseMatchResult,
    fps: float,
    pose_fps: float,
) -> np.ndarray:
    """Render metrics sidebar: Live Match Score, Hold Progress, Per-Wall Result, Joint Table."""
    panel = np.full((CANVAS_H, SIDEBAR_W, 3), C_PANEL_BG, dtype=np.uint8)
    tmpl = state.current_template()

    # Section 1: Header
    cv2.rectangle(panel, (0, 0), (SIDEBAR_W, 55), (15, 17, 24), -1)
    cv2.putText(panel, "WALL SHAPE: GAME B", (16, 28), FONT, 0.65, C_CYAN, 2, cv2.LINE_AA)
    cv2.putText(panel, "Pose Matcher & Hole-in-Wall Test", (16, 46), FONT, 0.38, C_TEXT_MUTED, 1, cv2.LINE_AA)

    y = 72

    # Section 2: Active Wall Card
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, y + 68), C_CARD_BG, -1)
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, y + 68), C_BORDER, 1)
    w_title = f"Wall [{state.template_idx + 1}/{len(state.templates)}]: {tmpl.name}"
    cv2.putText(panel, w_title, (22, y + 24), FONT, 0.52, C_TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.putText(panel, tmpl.description[:46], (22, y + 44), FONT, 0.36, C_TEXT_GREY, 1, cv2.LINE_AA)
    tol_txt = f"Tol: +/-{state.angle_tolerance_deg:.0f} deg | Req Hold: {state.required_hold_s:.1f}s"
    cv2.putText(panel, tol_txt, (22, y + 60), FONT, 0.34, C_TEXT_MUTED, 1, cv2.LINE_AA)

    y += 82

    # Section 3: Live Match Score (Rule 3)
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, y + 74), C_CARD_BG, -1)
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, y + 74), C_BORDER, 1)
    cv2.putText(panel, "LIVE MATCH SCORE", (22, y + 20), FONT, 0.44, C_TEXT_GREY, 1, cv2.LINE_AA)

    score_val = match_res.match_pct
    score_col = C_PASS if score_val >= 80.0 else (C_WARN if score_val >= 60.0 else C_FAIL)
    cv2.putText(panel, f"{score_val:4.1f}%", (22, y + 54), FONT, 0.95, score_col, 2, cv2.LINE_AA)

    match_status = "MATCH HOLD" if match_res.is_full_match else f"{match_res.matched_joints_count}/{match_res.total_joints_count} Joints OK"
    cv2.putText(panel, match_status, (150, y + 48), FONT, 0.42, score_col, 1, cv2.LINE_AA)

    # Score Gauge Bar
    bar_x = 22
    bar_w = SIDEBAR_W - 44
    cv2.rectangle(panel, (bar_x, y + 62), (bar_x + bar_w, y + 68), (40, 44, 56), -1)
    fill_w = int(bar_w * (score_val / 100.0))
    if fill_w > 0:
        cv2.rectangle(panel, (bar_x, y + 62), (bar_x + fill_w, y + 68), score_col, -1)

    y += 88

    # Section 4: Hold Progress & Wall Result (Rule 3)
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, y + 64), C_CARD_BG, -1)
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, y + 64), C_BORDER, 1)
    cv2.putText(panel, "HOLD PROGRESS", (22, y + 18), FONT, 0.44, C_TEXT_GREY, 1, cv2.LINE_AA)

    hold_ratio = max(0.0, min(1.0, state.hold_elapsed_s / state.required_hold_s))
    hold_col = C_PASS if hold_ratio >= 1.0 else C_CYAN
    cv2.putText(panel, f"{state.hold_elapsed_s:3.1f}s / {state.required_hold_s:3.1f}s", (22, y + 42), FONT, 0.65, hold_col, 2, cv2.LINE_AA)

    res_badge = state.last_wall_result if state.last_wall_result != "None" else "Active"
    badge_col = C_PASS if "PASSED" in res_badge else (C_FAIL if "MISSED" in res_badge else C_TEXT_MUTED)
    cv2.putText(panel, f"Last: {res_badge}", (170, y + 40), FONT, 0.38, badge_col, 1, cv2.LINE_AA)

    cv2.rectangle(panel, (bar_x, y + 52), (bar_x + bar_w, y + 58), (40, 44, 56), -1)
    h_fill = int(bar_w * hold_ratio)
    if h_fill > 0:
        cv2.rectangle(panel, (bar_x, y + 52), (bar_x + h_fill, y + 58), hold_col, -1)

    y += 76

    # Section 5: Joint Angles Breakdown Table
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, y + 180), C_CARD_BG, -1)
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, y + 180), C_BORDER, 1)
    cv2.putText(panel, "JOINT BREAKDOWN", (22, y + 18), FONT, 0.44, C_TEXT_GREY, 1, cv2.LINE_AA)

    table_hdr = f"{'JOINT':<14} {'ACTUAL':>6} {'TARGET':>6} {'DIFF':>6} {'STATUS':>8}"
    cv2.putText(panel, table_hdr, (20, y + 36), FONT, 0.33, C_TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.line(panel, (18, y + 42), (SIDEBAR_W - 18, y + 42), C_BORDER, 1)

    row_y = y + 57
    for j_name, j_res in match_res.joint_results.items():
        disp_name = j_name.replace("_", " ").title()[:13]
        act_s = f"{j_res.player_angle:5.0f} deg"
        tgt_s = f"{j_res.target_angle:5.0f} deg"
        diff_s = f"{j_res.diff_deg:+5.0f} deg"
        status_s = "PASS" if j_res.is_matched else "ADJ"
        status_col = C_PASS if j_res.is_matched else C_WARN

        row_str = f"{disp_name:<13} {act_s} {tgt_s} {diff_s}  {status_s}"
        cv2.putText(panel, row_str, (20, row_y), FONT, 0.32, status_col if not j_res.is_matched else C_TEXT_WHITE, 1, cv2.LINE_AA)
        row_y += 18

    y += 192

    # Section 6: Controls & Diagnostics Footer
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, CANVAS_H - 12), (18, 20, 26), -1)
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, CANVAS_H - 12), C_BORDER, 1)

    cv2.putText(panel, f"App FPS: {fps:.0f} | Pose FPS: {pose_fps:.0f}", (20, y + 18), FONT, 0.35, C_TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(panel, "[1-8]: Select Pose  [Space]: Next Wall", (20, y + 36), FONT, 0.34, C_TEXT_GREY, 1, cv2.LINE_AA)
    cv2.putText(panel, "[T]: Tol (+/-15 deg)  [H]: Hold Time", (20, y + 52), FONT, 0.34, C_TEXT_GREY, 1, cv2.LINE_AA)
    kb_str = "ON (Synthetic)" if state.keyboard_mode else "OFF (Webcam)"
    cv2.putText(panel, f"[K]: Keyboard Mode: {kb_str}", (20, y + 68), FONT, 0.34, C_CYAN if state.keyboard_mode else C_TEXT_GREY, 1, cv2.LINE_AA)

    return panel


# ==============================================================================
# Headless Step Function (for offline/automated test suites outside repo)
# ==============================================================================

def step_frame(
    state: WallShapeTestState,
    dt: float,
    source: Any | None = None,
    cam_frame: np.ndarray | None = None,
    player_pts: Sequence[Any] | None = None,
    mapper: FloorMapper | None = None,
    fps: float = 30.0,
    pose_fps: float = 30.0,
) -> tuple[np.ndarray, PoseMatchResult]:
    """Execute one simulation frame, update Wall Shape test state, and render canvas."""
    tmpl = state.current_template()

    # Determine player landmarks and position
    pts_to_evaluate: Sequence[Any] | None = None
    player_pos: tuple[float, float] | None = None
    pf: PoseFrame | None = None
    debug: dict = {}

    if state.keyboard_mode:
        # Generate synthetic posture for keyboard mode
        synthetic_pts = make_template_ref_posture(state.kb_posture_id)
        if state.kb_arm_offset_deg != 0.0:
            # Shift right wrist to simulate arm offset
            r_wrist = list(synthetic_pts[R_WRIST])
            rad = math.radians(state.kb_arm_offset_deg)
            r_wrist[1] += 0.15 * math.sin(rad)
            synthetic_pts[R_WRIST] = tuple(r_wrist)
        pts_to_evaluate = synthetic_pts
        player_pos = state.kb_pos_m
        pf = PoseFrame(tracked=True, floor_x_m=state.kb_pos_m[0], floor_y_m=state.kb_pos_m[1],
                       left_foot_m=(state.kb_pos_m[0] - 0.1, state.kb_pos_m[1]),
                       right_foot_m=(state.kb_pos_m[0] + 0.1, state.kb_pos_m[1]))
    elif source is not None:
        # Pull live frame and landmarks from CameraPoseSource
        if hasattr(source, "snapshot3d"):
            pf, skel, frame, debug = source.snapshot3d()
            if frame is not None:
                cam_frame = frame
            if skel and hasattr(skel, "points_m"):
                pts_to_evaluate = skel.points_m
            elif debug.get("pts"):
                pts_to_evaluate = debug["pts"]
            if pf and getattr(pf, "tracked", False):
                fx = getattr(pf, "floor_x_m", None)
                fy = getattr(pf, "floor_y_m", None)
                if fx is not None and fy is not None:
                    player_pos = (fx, fy)
    elif player_pts is not None:
        pts_to_evaluate = player_pts

    # Extract angles and evaluate against active template
    player_angles: dict[str, float] = {}
    if pts_to_evaluate is not None and len(pts_to_evaluate) >= 29:
        player_angles = extract_pose_angles(pts_to_evaluate)

    match_result = evaluate_pose(
        player_angles=player_angles,
        player_pos_m=player_pos,
        template=tmpl,
        angle_tolerance_override=state.angle_tolerance_deg,
    )

    # Update Wall Shape test engine
    state.update(dt, match_result)

    # Render stage and sidebar
    stage = render_stage_view(cam_frame, state, match_result, pts_to_evaluate, mapper=mapper, pf=pf, debug=debug)
    sidebar = render_sidebar_panel(state, match_result, fps, pose_fps)
    canvas = np.hstack([stage, sidebar])

    return canvas, match_result


# ==============================================================================
# Main Interactive CLI
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Wall Shape (Game B) Baseline Test Harness")
    parser.add_argument("--keyboard", action="store_true", help="run with synthetic poses, no camera needed")
    parser.add_argument("--camera", type=int, default=0, help="camera device index (default: 0)")
    parser.add_argument("--tolerance", type=float, default=20.0, help="angular tolerance in degrees (default: 20.0)")
    parser.add_argument("--hold", type=float, default=1.0, help="required hold time in seconds (default: 1.0)")
    parser.add_argument("--wall-time", type=float, default=7.0, help="wall approach duration in seconds (default: 7.0)")
    parser.add_argument("--template", type=str, default="", help="initial template id (e.g. t_pose, star, cactus)")
    parser.add_argument("--headless", action="store_true", help="run headlessly without opening GUI window")
    args = parser.parse_args()

    state = WallShapeTestState(
        angle_tolerance_deg=args.tolerance,
        required_hold_s=args.hold,
        wall_approach_s=args.wall_time,
        keyboard_mode=args.keyboard,
    )

    if args.template:
        t_ids = [t.id for t in state.templates]
        if args.template in t_ids:
            state.template_idx = t_ids.index(args.template)

    mapper = FloorMapper.load()
    source = None
    if not args.keyboard and not args.headless:
        from pose_core.capture import CameraPoseSource
        try:
            source = CameraPoseSource(camera_index=args.camera, mapper=mapper)
            source.start()
            atexit.register(source.stop)
            print(f"Wall Shape CameraPoseSource started on camera index {args.camera}.")
        except Exception as e:
            print(f"Warning: Could not start camera {args.camera}: {e}. Falling back to keyboard mode.")
            state.keyboard_mode = True

    print("==================================================================")
    print(" WALL SHAPE (Game B) - BASELINE TEST HARNESS")
    print(" Controls:")
    print("   [1-8]        : Select pose (1:T-Pose, 2:Star, 3:Cactus, 4:Tree...)")
    print("   [Space]      : Cycle to next wall template")
    print("   [T]          : Toggle angle tolerance (+/-10, +/-15, +/-20, +/-25 deg)")
    print("   [H]          : Toggle required hold time (0.5s, 1.0s, 1.5s)")
    print("   [K]          : Toggle Keyboard (Synthetic) mode")
    print("   [WASD]       : Move player position on floor (keyboard mode)")
    print("   [ [ / ] ]    : Perturb arm angle by +/-10 deg (keyboard mode)")
    print("   [Q / ESC]    : Quit")
    print("==================================================================")

    last_time = time.monotonic()
    fps = 30.0

    try:
        while True:
            now = time.monotonic()
            dt = min(now - last_time, 0.1)
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            pose_fps = source.pose_fps if source else (fps if state.keyboard_mode else 0.0)
            canvas, res = step_frame(state, dt, source=source, mapper=mapper, fps=fps, pose_fps=pose_fps)

            if args.headless:
                time.sleep(0.03)
                continue

            cv2.imshow("Wall Shape (Game B Baseline)", canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):  # Q or ESC
                break
            elif key == ord(" "):       # Space: Next wall
                state.next_wall()
            elif key in [ord(str(i)) for i in range(1, 9)]:
                idx = int(chr(key)) - 1
                state.select_template(idx)
                if state.keyboard_mode:
                    state.kb_posture_id = state.current_template().id
                    state.kb_arm_offset_deg = 0.0
            elif key in (ord("t"), ord("T")):
                tols = [10.0, 15.0, 20.0, 25.0]
                cur_i = tols.index(state.angle_tolerance_deg) if state.angle_tolerance_deg in tols else 1
                state.angle_tolerance_deg = tols[(cur_i + 1) % len(tols)]
                print(f"Tolerance set to +/-{state.angle_tolerance_deg} deg")
            elif key in (ord("h"), ord("H")):
                holds = [0.5, 1.0, 1.5, 2.0]
                cur_i = holds.index(state.required_hold_s) if state.required_hold_s in holds else 1
                state.required_hold_s = holds[(cur_i + 1) % len(holds)]
                print(f"Required hold set to {state.required_hold_s}s")
            elif key in (ord("k"), ord("K")):
                state.keyboard_mode = not state.keyboard_mode
                if state.keyboard_mode:
                    state.kb_posture_id = state.current_template().id
                print(f"Keyboard mode: {'ON' if state.keyboard_mode else 'OFF'}")
            elif key in (ord("w"), ord("W")):
                state.kb_pos_m = (state.kb_pos_m[0], state.kb_pos_m[1] + 0.05)
            elif key in (ord("s"), ord("S")):
                state.kb_pos_m = (state.kb_pos_m[0], max(0.0, state.kb_pos_m[1] - 0.05))
            elif key in (ord("a"), ord("A")):
                state.kb_pos_m = (max(0.0, state.kb_pos_m[0] - 0.05), state.kb_pos_m[1])
            elif key in (ord("d"), ord("D")):
                state.kb_pos_m = (state.kb_pos_m[0] + 0.05, state.kb_pos_m[1])
            elif key == ord("["):
                state.kb_arm_offset_deg -= 10.0
                print(f"Synthetic arm offset: {state.kb_arm_offset_deg} deg")
            elif key == ord("]"):
                state.kb_arm_offset_deg += 10.0
                print(f"Synthetic arm offset: {state.kb_arm_offset_deg} deg")

    finally:
        if source:
            source.stop()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
