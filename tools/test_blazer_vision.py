"""Interactive Vision & Pose Matching Debugger for Pose Blazer.

Run to test pose extraction, matching, and visual feedback in real-time:
  # With webcam:
  python tools/test_blazer_vision.py --camera 0

  # With keyboard simulation (no camera needed):
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
    evaluate_blazer_pose,
    load_blazer_catalog,
    render_reference_card,
)
from pose_core.capture import CameraPoseSource
from pose_core.floor_map import FloorMapper
from pose_core.pose_matcher import extract_pose_angles
from pose_core.skeleton import BONES
from station_b.keyboard_input import KeyboardWallPoseProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose Blazer Vision Debugger")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--keyboard", action="store_true", help="Keyboard simulation mode")
    parser.add_argument("--catalog", type=str, default="data/poses_index.json", help="Catalog path")
    args = parser.parse_args()

    catalog = load_blazer_catalog(args.catalog)
    if not catalog:
        print(f"Catalog at '{args.catalog}' is empty. Run 'python tools/index_poses.py --init-defaults' first.")
        return

    pygame.init()
    win_w, win_h = 1180, 680
    screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
    pygame.display.set_caption("Pose Blazer - Vision & Pose Matching Debugger")
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

    pose_idx = 0
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
                        pose_idx = (pose_idx + 1) % len(catalog)

            current_pose = catalog[pose_idx]

            # Fetch live frame and landmarks
            cam_frame = None
            debug_info = None
            player_angles: dict[str, float | None] = {}
            player_head_norm: tuple[float, float] | None = None

            if hasattr(provider, "snapshot"):
                pose, cam_frame, debug_info = provider.snapshot()
                if debug_info and "landmarks_px" in debug_info and len(debug_info["landmarks_px"]) > 0:
                    player_angles = extract_pose_angles(debug_info["landmarks_px"])
                    lm0 = debug_info["landmarks_px"][0]
                    hx, hy = float(lm0[0]), float(lm0[1])
                    hvis = float(lm0[2]) if len(lm0) > 2 else 1.0
                    if cam_frame is not None and hvis > 0.35:
                        h, w = cam_frame.shape[:2]
                        player_head_norm = (hx / w, hy / h)
            else:
                pose = provider.latest()

            if args.keyboard and isinstance(provider, KeyboardWallPoseProvider):
                keys = pygame.key.get_pressed()
                provider.handle_keys(keys, dt)
                player_angles = provider.get_angles()
                player_head_norm = provider.get_head()

            # Evaluate pose similarity
            match_res = evaluate_blazer_pose(
                player_landmarks=player_angles,
                target_pose=current_pose,
                player_head_norm=player_head_norm,
                mirrored=True,
            )

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

            _render_live_hud(live_view, match_res, debug_info, player_head_norm)

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
    res: Any,
    debug_info: dict | None,
    player_head_norm: tuple[float, float] | None,
) -> None:
    """Render player skeleton, head alignment box, and match gauge onto live view."""
    h, w = canvas.shape[:2]

    # 1. Head Alignment Target Box
    tgt_x = w // 2
    tgt_y = int(h * 0.25)
    bw, bh = int(w * 0.18), int(h * 0.24)
    bx1, by1 = tgt_x - bw // 2, tgt_y - bh // 2
    bx2, by2 = tgt_x + bw // 2, tgt_y + bh // 2

    box_col = C_NEON_GREEN_BGR if res.head_aligned else (0, 220, 255)
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), box_col, 2, cv2.LINE_AA)

    lbl = "HEAD ALIGNED" if res.head_aligned else "ALIGN HEAD HERE"
    cv2.putText(canvas, lbl, (bx1 + 4, by1 - 8), cv2.FONT_HERSHEY_DUPLEX, 0.45, box_col, 1, cv2.LINE_AA)

    # 2. Player skeleton overlay with color-coded joints
    if debug_info and "landmarks_px" in debug_info and len(debug_info["landmarks_px"]) >= 29:
        lms = debug_info["landmarks_px"]
        pts = [(int(lm[0] * w / 640), int(lm[1] * h / 480)) for lm in lms]

        for j1, j2 in BONES:
            if j1 < len(pts) and j2 < len(pts):
                cv2.line(canvas, pts[j1], pts[j2], (180, 180, 180), 2, cv2.LINE_AA)

        for px, py in pts:
            cv2.circle(canvas, (px, py), 5, (255, 255, 0), -1, cv2.LINE_AA)

    # 3. Match Progress Gauge at Top
    bar_w = int(w * 0.70)
    bar_h = 24
    bx = (w - bar_w) // 2
    by = 20

    # Background
    cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), (30, 35, 48), -1)

    # Progress fill
    fill_w = int(bar_w * (res.match_pct / 100.0))
    bar_col = C_NEON_GREEN_BGR if res.is_matched else (C_AMBER_YELLOW_BGR if res.match_pct >= 50 else C_ALERT_RED_BGR)
    cv2.rectangle(canvas, (bx, by), (bx + fill_w, by + bar_h), bar_col, -1)
    cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), (255, 255, 255), 2)

    # 75% threshold marker
    mark_x = int(bx + bar_w * 0.75)
    cv2.line(canvas, (mark_x, by - 4), (mark_x, by + bar_h + 4), (255, 255, 255), 2)

    gauge_text = f"MATCH: {res.match_pct:.1f}%  ({'PASSED!' if res.is_matched else 'AIM >= 75%'})"
    cv2.putText(canvas, gauge_text, (bx + 12, by + 17), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Directional Feedback Hint Banner at Bottom
    cv2.rectangle(canvas, (0, h - 50), (w, h), (18, 22, 30), -1)
    cv2.line(canvas, (0, h - 50), (w, h - 50), bar_col, 2)
    hint_str = f"HINT: {res.primary_hint.upper()}"
    cv2.putText(canvas, hint_str, (24, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.65, bar_col, 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
