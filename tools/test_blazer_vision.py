"""Interactive Vision & Pose Matching Debugger for Pose Blazer.

Features:
  - 7.0-second countdown timer with auto-advancing poses.
  - Hold-to-snap lock-in mechanism (0.30s at >= 75% match).
  - Speed bonus scoring calculation (+base, +speed bonus, +perfection bonus).
  - Real-time skeleton overlay, key-joint coloring, and directional hints.
  - Webcam mode (--camera 0) or Keyboard simulation mode (--keyboard).

Usage:
  python tools/test_blazer_vision.py --camera 0
  python tools/test_blazer_vision.py --keyboard
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pygame

# Insert project root into sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Insert site-packages if available
VENV_PKGS = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if VENV_PKGS.is_dir() and str(VENV_PKGS) not in sys.path:
    sys.path.insert(0, str(VENV_PKGS))

from pose_core.blazer_matcher import (
    C_ALERT_RED_BGR,
    C_AMBER_YELLOW_BGR,
    C_DARK_BG_BGR,
    C_NEON_CYAN_BGR,
    C_NEON_GREEN_BGR,
    load_blazer_catalog,
    render_reference_card,
)
from pose_core.capture import CameraPoseSource
from pose_core.floor_map import FloorMapper
from pose_core.pose_matcher import extract_pose_angles
from pose_core.skeleton import BONES
from shared.contracts import Phase
from station_b.blazer_logic import BlazerGameEngine
from station_b.keyboard_input import KeyboardWallPoseProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose Blazer Vision Debugger")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--keyboard", action="store_true", help="Keyboard simulation mode")
    parser.add_argument("--catalog", type=str, default="data/poses_index.json", help="Catalog path")
    parser.add_argument("--time", type=float, default=7.0, help="Round duration in seconds (default 7.0)")
    args = parser.parse_args()

    catalog = load_blazer_catalog(args.catalog)
    if not catalog:
        print(f"Catalog at '{args.catalog}' is empty. Run 'python tools/index_poses.py --init-defaults' first.")
        return

    pygame.init()
    win_w, win_h = 1180, 680
    screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
    pygame.display.set_caption("Pose Blazer - 7s Time-Attack Vision Engine")
    clock = pygame.time.Clock()

    # Pose provider setup
    if args.keyboard:
        provider = KeyboardWallPoseProvider(arena_w_m=1.8, arena_d_m=1.8)
    else:
        try:
            mapper = FloorMapper.load()
            if mapper is None:
                mapper = FloorMapper(
                    corners_px=[[100, 460], [540, 460], [460, 180], [180, 180]],
                    w_m=1.8,
                    h_m=1.8,
                )
            provider = CameraPoseSource(camera_index=args.camera, mapper=mapper)
            provider.start()
        except Exception as e:
            print(f"Warning: Could not open camera {args.camera} ({e}). Falling back to keyboard.")
            provider = KeyboardWallPoseProvider(arena_w_m=1.8, arena_d_m=1.8)
            args.keyboard = True

    # Initialize 7-Second Game State Engine
    engine = BlazerGameEngine(
        catalog=catalog,
        round_duration_s=args.time,
        total_rounds=len(catalog),
    )
    engine.start_game()

    running = True

    try:
        while running:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Manual skip to next pose
                        engine._advance_to_next_pose()

            # Snapshot pose and camera frame
            cam_frame = None
            debug_info = None
            player_angles: dict[str, float | None] = {}
            player_head_norm: tuple[float, float] | None = None

            if hasattr(provider, "snapshot"):
                pose, cam_frame, debug_info = provider.snapshot()
                if debug_info and "landmarks_px" in debug_info and len(debug_info["landmarks_px"]) > 0:
                    cw = cam_frame.shape[1] if cam_frame is not None else 640
                    ch = cam_frame.shape[0] if cam_frame is not None else 480
                    norm_lms = [(float(lm[0]) / cw, float(lm[1]) / ch) for lm in debug_info["landmarks_px"]]
                    player_angles = extract_pose_angles(norm_lms, mirrored=True, planar_2d=True)
                    lm0 = debug_info["landmarks_px"][0]
                    hx, hy = float(lm0[0]), float(lm0[1])
                    hvis = float(lm0[2]) if len(lm0) > 2 else 1.0
                    if cam_frame is not None and hvis > 0.35:
                        player_head_norm = (hx / cw, hy / ch)
            else:
                pose = provider.latest()

            if args.keyboard and isinstance(provider, KeyboardWallPoseProvider):
                keys = pygame.key.get_pressed()
                provider.handle_keys(keys, dt)
                player_angles = provider.get_angles()
                player_head_norm = provider.get_head()

            # Tick 7-Second Auto-Advancing Engine
            state, current_pose = engine.update(
                pose=pose,
                dt=dt,
                player_angles=player_angles,
                player_head_norm=player_head_norm,
            )

            if state.phase == Phase.RESULT:
                # Loop continuously in debugger
                engine.start_game()
                current_pose = engine.current_pose

            if current_pose is None:
                current_pose = catalog[0]

            match_res = engine.last_match_result

            # Build Composite Canvas: [Left: Reference Card | Right: Live Feed]
            card_w = 340
            live_w = win_w - card_w - 20
            live_h = win_h - 20

            # 1. Render Reference Card
            ref_card = render_reference_card(current_pose, width=card_w, height=live_h)

            # 2. Render Live Feed View
            if cam_frame is not None:
                live_view = cv2.resize(cam_frame, (live_w, live_h))
            else:
                live_view = np.zeros((live_h, live_w, 3), dtype=np.uint8)
                live_view[:] = (22, 26, 36)

            _render_live_hud(live_view, engine, match_res, debug_info)

            # Assemble into unified screen
            unified_bgr = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            unified_bgr[:] = (12, 14, 20)
            unified_bgr[10 : 10 + live_h, 10 : 10 + card_w] = ref_card
            unified_bgr[10 : 10 + live_h, 20 + card_w : 20 + card_w + live_w] = live_view

            # Blit to pygame surface without axis distortion
            rgb_frame = cv2.cvtColor(unified_bgr, cv2.COLOR_BGR2RGB)
            surf = pygame.image.frombuffer(rgb_frame.tobytes(), (win_w, win_h), "RGB")
            screen.blit(surf, (0, 0))
            pygame.display.flip()

    finally:
        if hasattr(provider, "stop"):
            provider.stop()
        pygame.quit()


def _render_live_hud(
    canvas: np.ndarray,
    engine: BlazerGameEngine,
    res: Any,
    debug_info: dict | None,
) -> None:
    """Render player skeleton, countdown timer bar, match gauge, and feedback."""
    h, w = canvas.shape[:2]

    # 1. Head Alignment Target Box
    tgt_x = w // 2
    tgt_y = int(h * 0.28)
    bw, bh = int(w * 0.18), int(h * 0.24)
    bx1, by1 = tgt_x - bw // 2, tgt_y - bh // 2
    bx2, by2 = tgt_x + bw // 2, tgt_y + bh // 2

    is_head_ok = res.head_aligned if res else False
    box_col = C_NEON_GREEN_BGR if is_head_ok else (0, 220, 255)
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), box_col, 2, cv2.LINE_AA)

    lbl = "HEAD ALIGNED" if is_head_ok else "ALIGN HEAD HERE"
    cv2.putText(canvas, lbl, (bx1 + 4, by1 - 8), cv2.FONT_HERSHEY_DUPLEX, 0.45, box_col, 1, cv2.LINE_AA)

    # 2. Player skeleton overlay with color-coded joints
    if debug_info and "landmarks_px" in debug_info and len(debug_info["landmarks_px"]) >= 29:
        lms = debug_info["landmarks_px"]
        cw = 640
        ch = 480
        pts = [(int(float(lm[0]) * w / cw), int(float(lm[1]) * h / ch)) for lm in lms]

        # Map landmarks to joint names in mirrored screen-space
        lm_to_joint = {
            12: "left_shoulder",
            14: "left_elbow",
            16: "left_elbow",
            11: "right_shoulder",
            13: "right_elbow",
            15: "right_elbow",
            24: "left_hip",
            26: "left_knee",
            28: "left_knee",
            23: "right_hip",
            25: "right_knee",
            27: "right_knee",
        }

        # Bone to joint mapping for color coding limbs
        bone_to_joint = {
            (12, 14): "left_shoulder",
            (14, 16): "left_elbow",
            (11, 13): "right_shoulder",
            (13, 15): "right_elbow",
            (24, 26): "left_hip",
            (26, 28): "left_knee",
            (23, 25): "right_hip",
            (25, 27): "right_knee",
        }

        joint_cols = res.joint_colors_bgr if res and hasattr(res, "joint_colors_bgr") else {}
        joint_statuses = res.joint_status if res and hasattr(res, "joint_status") else {}

        # Draw bones with dynamic joint colors
        for j1, j2 in BONES:
            if j1 < len(pts) and j2 < len(pts):
                p1, p2 = pts[j1], pts[j2]
                assoc_joint = bone_to_joint.get((j1, j2), bone_to_joint.get((j2, j1)))
                if assoc_joint and assoc_joint in joint_cols:
                    b_col = joint_cols[assoc_joint]
                    is_matched = joint_statuses.get(assoc_joint) == "matched"
                    thick = 4 if is_matched else 2
                    cv2.line(canvas, p1, p2, (20, 20, 20), thick + 2, cv2.LINE_AA)
                    cv2.line(canvas, p1, p2, b_col, thick, cv2.LINE_AA)
                else:
                    t_col = C_NEON_GREEN_BGR if is_head_ok else (120, 140, 160)
                    cv2.line(canvas, p1, p2, t_col, 2, cv2.LINE_AA)

        # Draw joint circles with halo feedback
        for idx, (px, py) in enumerate(pts[:29]):
            assoc_joint = lm_to_joint.get(idx)
            if assoc_joint and assoc_joint in joint_cols:
                col = joint_cols[assoc_joint]
                stat = joint_statuses.get(assoc_joint, "off")
                if stat == "matched":
                    # Glowing neon green halo for matched joints
                    cv2.circle(canvas, (px, py), 9, C_NEON_GREEN_BGR, 2, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)
                elif stat == "close":
                    cv2.circle(canvas, (px, py), 6, col, -1, cv2.LINE_AA)
                else:
                    cv2.circle(canvas, (px, py), 5, col, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, (px, py), 4, (0, 220, 255), -1, cv2.LINE_AA)

    # 3. Top Info Header: Round & Score Badges
    top_badge_y = 22
    round_str = f"ROUND {engine.round_number}/{engine.total_rounds}"
    score_str = f"SCORE: {engine.state.score}  |  STREAK: {engine.state.streak}"
    cv2.putText(canvas, round_str, (16, top_badge_y), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, score_str, (w - 290, top_badge_y), cv2.FONT_HERSHEY_DUPLEX, 0.55, C_NEON_GREEN_BGR, 1, cv2.LINE_AA)

    # 4. 7.0-Second Countdown Timer Bar
    t_ratio = max(0.0, min(1.0, engine.time_left_s / engine.round_duration_s))
    t_bar_w = int(w * 0.70)
    t_bar_h = 16
    tx = (w - t_bar_w) // 2
    ty = 36

    # Background
    cv2.rectangle(canvas, (tx, ty), (tx + t_bar_w, ty + t_bar_h), (25, 30, 42), -1)

    # Timer fill color: Cyan -> Amber (<4s) -> Flashing Red (<2s)
    if engine.time_left_s > 3.5:
        t_col = C_NEON_CYAN_BGR
    elif engine.time_left_s > 1.8:
        t_col = C_AMBER_YELLOW_BGR
    else:
        # Flashing alert red
        t_col = C_ALERT_RED_BGR if int(time.time() * 6) % 2 == 0 else (255, 100, 100)

    fill_tw = int(t_bar_w * t_ratio)
    cv2.rectangle(canvas, (tx, ty), (tx + fill_tw, ty + t_bar_h), t_col, -1)
    cv2.rectangle(canvas, (tx, ty), (tx + t_bar_w, ty + t_bar_h), (120, 140, 160), 1)

    timer_text = f"TIME: {engine.time_left_s:.1f}s"
    cv2.putText(canvas, timer_text, (tx + t_bar_w + 12, ty + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, t_col, 1, cv2.LINE_AA)

    # 5. Match Progress Gauge
    bar_w = int(w * 0.70)
    bar_h = 24
    bx = (w - bar_w) // 2
    by = 58

    cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), (30, 35, 48), -1)

    match_pct = res.match_pct if res else 0.0
    fill_w = int(bar_w * (match_pct / 100.0))
    is_ok = res.is_matched if res else False
    bar_col = C_NEON_GREEN_BGR if is_ok else (C_AMBER_YELLOW_BGR if match_pct >= 50 else C_ALERT_RED_BGR)
    cv2.rectangle(canvas, (bx, by), (bx + fill_w, by + bar_h), bar_col, -1)
    cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), (255, 255, 255), 2)

    # 75% threshold marker
    mark_x = int(bx + bar_w * 0.75)
    cv2.line(canvas, (mark_x, by - 4), (mark_x, by + bar_h + 4), (255, 255, 255), 2)

    # Hold progress text
    if is_ok:
        hold_pct = min(100, int((engine.hold_timer_s / 0.30) * 100))
        gauge_text = f"MATCH: {match_pct:.1f}%  (HOLDING: {hold_pct}%)"
    else:
        gauge_text = f"MATCH: {match_pct:.1f}%  (AIM >= 75%)"

    cv2.putText(canvas, gauge_text, (bx + 12, by + 17), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # 6. Directional Feedback Hint Banner at Bottom
    cv2.rectangle(canvas, (0, h - 50), (w, h), (18, 22, 30), -1)
    cv2.line(canvas, (0, h - 50), (w, h - 50), bar_col, 2)

    if engine.banner_msg:
        b_text = engine.banner_msg.upper()
        b_col = engine.banner_color
    elif res:
        b_text = f"HINT: {res.primary_hint.upper()}"
        b_col = bar_col
    else:
        b_text = "STRIKE THE POSE!"
        b_col = C_NEON_CYAN_BGR

    cv2.putText(canvas, b_text, (24, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.65, b_col, 1, cv2.LINE_AA)
