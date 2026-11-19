"""Station B (Pose Blazer Tournament) Pygame + OpenCV AR Renderer.
Draws:
- Split-screen HD layout (380x520 reference card on left, live player AR feed on right).
- 2-Stage tournament HUD (Precision Blitz vs Speed Rush).
- Neon color-coded bones for 6 core limbs (green <= 22 deg, yellow <= 42 deg, red > 42 deg).
- Real-time joint angle telemetry panel (YOU vs TGT).
- Match gauge, hold timer progress, and skip feedback.
- Tournament Result summary screen with score and streak statistics.
"""
from __future__ import annotations

import time
from typing import Any, Sequence

import cv2
import numpy as np
import pygame

from pose_core.blazer_matcher import (
    C_ALERT_RED_BGR,
    C_AMBER_YELLOW_BGR,
    C_DARK_BG_BGR,
    C_NEON_CYAN_BGR,
    C_NEON_GREEN_BGR,
    BlazerMatchResult,
    BlazerPoseTemplate,
    render_reference_card,
)
from shared.contracts import GameState, Phase, PoseFrame
from station_b.logic import BlazerGameEngine

# Canvas dimensions
HUD_TOP_H = 64
HUD_BOT_H = 46

# Colors
TEXT_GOLD = (255, 204, 0)
TEXT_WHITE = (245, 248, 255)
TEXT_MUTED = (140, 148, 165)
SAFE_GREEN = (0, 255, 120)


class BlazerArenaRenderer:
    """HD Tournament Renderer for Pose Blazer (Station B)."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fullscreen: bool = False,
        poses_dir: str = "data/poses",
    ) -> None:
        pygame.init()
        pygame.font.init()

        self.width = width
        self.height = height
        self.poses_dir = poses_dir
        self.flags = pygame.DOUBLEBUF | pygame.RESIZABLE
        if fullscreen:
            self.flags |= pygame.FULLSCREEN

        self.screen = pygame.display.set_mode((width, height), self.flags)
        pygame.display.set_caption("Station B: Pose Blazer Tournament")

        self.is_maximized = False
        if fullscreen:
            self._maximize_window()

        self.font_title = pygame.font.SysFont("Impact", 28)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_mid = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)
        self.font_tiny = pygame.font.SysFont("Consolas", 11)

        # Bone definitions for 6 limbs
        self.primary_limb_bones = [
            (11, 12),           # Shoulders
            (11, 13), (13, 15), # Right Arm (mirrored: right shoulder to wrist)
            (12, 14), (14, 16), # Left Arm (mirrored: left shoulder to wrist)
            (11, 23), (12, 24), # Torso sides
            (23, 24),           # Hips
            (23, 25), (25, 27), # Right Leg
            (24, 26), (26, 28), # Left Leg
        ]

        self.bone_to_joint = {
            (12, 14): "left_shoulder",
            (14, 16): "left_elbow",
            (11, 13): "right_shoulder",
            (13, 15): "right_elbow",
            (24, 26): "left_knee",
            (26, 28): "left_knee",
            (23, 25): "right_knee",
            (25, 27): "right_knee",
        }

    def _maximize_window(self) -> None:
        try:
            import ctypes
            wm = pygame.display.get_wm_info()
            hwnd = wm.get("window")
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 3)
                self.is_maximized = True
        except Exception:
            pass

    def toggle_fullscreen(self) -> None:
        self.is_maximized = not self.is_maximized
        try:
            import ctypes
            wm = pygame.display.get_wm_info()
            hwnd = wm.get("window")
            if hwnd:
                cmd = 3 if self.is_maximized else 9
                ctypes.windll.user32.ShowWindow(hwnd, cmd)
        except Exception:
            pass

    def draw(
        self,
        state: GameState,
        engine: BlazerGameEngine,
        cam_frame: np.ndarray | None = None,
        debug_info: dict | None = None,
        match_res: BlazerMatchResult | None = None,
        player_angles: dict[str, float | None] | None = None,
        show_skeleton_on_card: bool = False,
        fps: float = 60.0,
    ) -> None:
        win_w, win_h = self.screen.get_size()
        self.screen.fill((16, 18, 24))

        curr_pose = engine.current_pose

        # ----------------------------------------------------------------------
        # 1. Split Layout Calculations
        # ----------------------------------------------------------------------
        top_h = HUD_TOP_H
        bot_h = HUD_BOT_H
        play_h = win_h - top_h - bot_h

        # Left Column: Reference Card (360px wide)
        card_w = min(400, int(win_w * 0.32))
        card_h = play_h - 20
        card_x = 16
        card_y = top_h + 10

        # Right Column: Live Player Feed
        cam_x = card_x + card_w + 16
        cam_y = card_y
        cam_w = win_w - cam_x - 16
        cam_h = card_h

        # ----------------------------------------------------------------------
        # 2. Render Left Reference Pose Card
        # ----------------------------------------------------------------------
        if curr_pose:
            card_bgr = render_reference_card(
                curr_pose,
                width=card_w,
                height=card_h,
                poses_dir=self.poses_dir,
            )
            card_rgb = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2RGB)
            card_surf = pygame.image.frombuffer(card_rgb.tobytes(), (card_w, card_h), "RGB")
            self.screen.blit(card_surf, (card_x, card_y))

            # Description below card header
            if curr_pose.description:
                desc_bg = pygame.Surface((card_w - 4, 32), pygame.SRCALPHA)
                desc_bg.fill((10, 12, 16, 210))
                self.screen.blit(desc_bg, (card_x + 2, card_y + card_h - 60))
                d_txt = self.font_tiny.render(curr_pose.description[:48], True, (200, 215, 230))
                self.screen.blit(d_txt, (card_x + 10, card_y + card_h - 52))

        # ----------------------------------------------------------------------
        # 3. Render Right Camera Feed & Player Skeleton
        # ----------------------------------------------------------------------
        cam_surface = pygame.Surface((cam_w, cam_h))
        cam_surface.fill((22, 26, 36))

        if cam_frame is not None:
            # Draw player AR canvas in OpenCV
            ar_canvas = cv2.resize(cam_frame, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
            ch, cw = ar_canvas.shape[:2]

            # Head Alignment Target Box
            if engine.check_head:
                tgt_hx = cw // 2
                tgt_hy = int(ch * 0.26)
                bw, bh = int(cw * 0.18), int(ch * 0.24)
                bx1, by1 = tgt_hx - bw // 2, tgt_hy - bh // 2
                bx2, by2 = tgt_hx + bw // 2, tgt_hy + bh // 2
                is_head_ok = match_res.head_aligned if match_res else False
                b_col = C_NEON_GREEN_BGR if is_head_ok else (0, 220, 255)
                cv2.rectangle(ar_canvas, (bx1, by1), (bx2, by2), b_col, 2, cv2.LINE_AA)
                cv2.putText(
                    ar_canvas,
                    "HEAD ALIGNED" if is_head_ok else "ALIGN HEAD HERE",
                    (bx1 + 4, by1 - 8),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.42,
                    b_col,
                    1,
                    cv2.LINE_AA,
                )

            # Draw Skeleton Bones
            if debug_info and "landmarks_px" in debug_info and len(debug_info["landmarks_px"]) >= 29:
                lms = debug_info["landmarks_px"]
                orig_w = cam_frame.shape[1]
                orig_h = cam_frame.shape[0]
                pts = [(int(float(lm[0]) * cw / orig_w), int(float(lm[1]) * ch / orig_h)) for lm in lms]

                joint_cols = match_res.joint_colors_bgr if match_res else {}
                joint_stats = match_res.joint_status if match_res else {}

                for j1, j2 in self.primary_limb_bones:
                    if j1 < len(pts) and j2 < len(pts):
                        p1, p2 = pts[j1], pts[j2]
                        assoc_j = self.bone_to_joint.get((j1, j2), self.bone_to_joint.get((j2, j1)))
                        if assoc_j and assoc_j in joint_cols:
                            col = joint_cols[assoc_j]
                            thick = 4 if joint_stats.get(assoc_j) == "matched" else 2
                            cv2.line(ar_canvas, p1, p2, (20, 20, 20), thick + 2, cv2.LINE_AA)
                            cv2.line(ar_canvas, p1, p2, col, thick, cv2.LINE_AA)
                        else:
                            cv2.line(ar_canvas, p1, p2, (100, 120, 140), 2, cv2.LINE_AA)

                for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                    if idx < len(pts):
                        cv2.circle(ar_canvas, pts[idx], 5, (0, 220, 255), -1, cv2.LINE_AA)

            # Convert to Pygame
            rgb_cam = cv2.cvtColor(ar_canvas, cv2.COLOR_BGR2RGB)
            cam_feed_surf = pygame.image.frombuffer(rgb_cam.tobytes(), (cw, ch), "RGB")
            cam_surface.blit(cam_feed_surf, (0, 0))
        else:
            # Virtual stage grid
            for y_line in range(0, cam_h, 30):
                pygame.draw.line(cam_surface, (30, 36, 48), (0, y_line), (cam_w, y_line), 1)

        self.screen.blit(cam_surface, (cam_x, cam_y))
        pygame.draw.rect(self.screen, (55, 68, 90), (cam_x, cam_y, cam_w, cam_h), 2)

        # ----------------------------------------------------------------------
        # 4. Live Joint Angle Telemetry Panel (Top-Right inside Cam Feed)
        # ----------------------------------------------------------------------
        eval_joints = engine.active_joints
        target_angles = curr_pose.target_angles if curr_pose else {}
        angles_map = player_angles or {}

        panel_w = 230
        panel_h = 24 + len(eval_joints) * 20
        pan_x = cam_x + cam_w - panel_w - 12
        pan_y = cam_y + 12

        pan_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pan_surf.fill((16, 20, 28, 220))
        self.screen.blit(pan_surf, (pan_x, pan_y))
        pygame.draw.rect(self.screen, (60, 75, 100), (pan_x, pan_y, panel_w, panel_h), 1)

        hdr_txt = self.font_tiny.render("JOINT TELEMETRY (TGT vs YOU)", True, (0, 220, 255))
        self.screen.blit(hdr_txt, (pan_x + 8, pan_y + 5))

        for i, j_name in enumerate(eval_joints):
            tgt = target_angles.get(j_name)
            act = angles_map.get(j_name)
            row_y = pan_y + 22 + i * 19
            j_short = j_name.replace("left_", "L_").replace("right_", "R_").upper()

            if tgt is not None:
                tgt_str = f"{tgt:.0f}°"
                if act is None:
                    act_str = "--"
                    col = TEXT_MUTED
                else:
                    diff = abs(act - tgt)
                    act_str = f"{act:.0f}°"
                    if diff <= 22.0:
                        col = SAFE_GREEN
                    elif diff <= 42.0:
                        col = TEXT_GOLD
                    else:
                        col = (255, 60, 60)
                row_str = f"{j_short:<9} TGT:{tgt_str:<4} YOU:{act_str:<4}"
                r_txt = self.font_tiny.render(row_str, True, col)
                self.screen.blit(r_txt, (pan_x + 8, row_y))

        # ----------------------------------------------------------------------
        # 5. Top Tournament HUD
        # ----------------------------------------------------------------------
        hud_surface = pygame.Surface((win_w, top_h), pygame.SRCALPHA)
        hud_surface.fill((12, 15, 20, 235))
        self.screen.blit(hud_surface, (0, 0))
        pygame.draw.line(self.screen, (45, 55, 75), (0, top_h), (win_w, top_h), 2)

        # Stage Badge
        stage_num = engine.current_stage
        stage_title = engine.current_stage_name
        stage_col = (0, 220, 255) if stage_num == 1 else (255, 140, 0)
        stg_txt = self.font_title.render(f"STAGE {stage_num}: {stage_title}", True, stage_col)
        self.screen.blit(stg_txt, (24, 8))

        round_txt = self.font_small.render(f"ROUND {engine.round_number}/{engine.total_rounds} | 6-LIMB ARCADE", True, (180, 200, 220))
        self.screen.blit(round_txt, (24, 38))

        # Score & Streak
        score_x = max(360, win_w // 2 - 120)
        s_txt = self.font_large.render(f"SCORE: {state.score:04d}", True, TEXT_GOLD)
        self.screen.blit(s_txt, (score_x, 8))

        streak_col = SAFE_GREEN if state.streak > 0 else TEXT_MUTED
        streak_mult = min(2.5, 1.0 + (state.streak // 2) * 0.2)
        skips_left = max(0, 4 - engine.consecutive_skips)
        stk_str = f"STREAK: {state.streak} (x{streak_mult:.1f}) | PASSED: {state.dodges} | SKIPS LEFT: {skips_left}"
        stk_txt = self.font_small.render(stk_str, True, streak_col)
        self.screen.blit(stk_txt, (score_x, 38))

        # Round Countdown Timer Bar
        t_box_w = 180
        t_box_h = 16
        t_x = win_w - t_box_w - 24
        t_y = 20

        secs = max(0.0, engine.time_left_s)
        time_ratio = max(0.0, min(1.0, secs / max(0.1, engine.round_duration_s)))
        t_bar_col = (0, 220, 255) if secs > 3.0 else ((255, 160, 0) if secs > 1.5 else (255, 50, 50))

        pygame.draw.rect(self.screen, (25, 30, 42), (t_x, t_y, t_box_w, t_box_h), border_radius=3)
        pygame.draw.rect(self.screen, t_bar_col, (t_x, t_y, int(t_box_w * time_ratio), t_box_h), border_radius=3)
        pygame.draw.rect(self.screen, (100, 120, 140), (t_x, t_y, t_box_w, t_box_h), 1, border_radius=3)

        time_val = self.font_large.render(f"{secs:04.1f}s", True, t_bar_col)
        self.screen.blit(time_val, (t_x - 70, 14))

        # ----------------------------------------------------------------------
        # 6. Bottom Guidance Banner
        # ----------------------------------------------------------------------
        bot_surface = pygame.Surface((win_w, bot_h), pygame.SRCALPHA)
        bot_surface.fill((12, 15, 20, 235))
        self.screen.blit(bot_surface, (0, win_h - bot_h))
        pygame.draw.line(self.screen, (45, 55, 75), (0, win_h - bot_h), (win_w, win_h - bot_h), 2)

        if engine.banner_msg:
            b_surf = self.font_mid.render(engine.banner_msg.upper(), True, engine.banner_color)
            self.screen.blit(b_surf, b_surf.get_rect(center=(win_w // 2, win_h - bot_h // 2)))
        elif match_res:
            hint_txt = f"HINT: {match_res.primary_hint.upper()}"
            h_surf = self.font_mid.render(hint_txt, True, TEXT_GOLD)
            self.screen.blit(h_surf, h_surf.get_rect(center=(win_w // 2, win_h - bot_h // 2)))

        # ----------------------------------------------------------------------
        # 7. Match Progress Gauge inside Live View
        # ----------------------------------------------------------------------
        gauge_w = int(cam_w * 0.48)
        gauge_h = 24
        gx = cam_x + 16
        gy = cam_y + 16

        pygame.draw.rect(self.screen, (20, 24, 32), (gx, gy, gauge_w, gauge_h), border_radius=4)
        pct = match_res.match_pct if match_res else 0.0
        fill_gw = int(gauge_w * min(1.0, pct / 100.0))
        g_col = SAFE_GREEN if (match_res and match_res.is_matched) else (TEXT_GOLD if pct >= 50 else (240, 50, 50))
        pygame.draw.rect(self.screen, g_col, (gx, gy, fill_gw, gauge_h), border_radius=4)
        pygame.draw.rect(self.screen, (255, 255, 255), (gx, gy, gauge_w, gauge_h), 2, border_radius=4)

        pass_line_x = gx + int(gauge_w * 0.65)
        pygame.draw.line(self.screen, (255, 255, 255), (pass_line_x, gy - 3), (pass_line_x, gy + gauge_h + 3), 2)

        gauge_str = f"MATCH: {pct:.1f}% (PASS >= 65%)"
        if match_res and match_res.is_matched:
            req_h = 0.20 if engine.current_stage == 2 else 0.35
            h_pct = int(min(100.0, (engine.hold_timer_s / req_h) * 100.0))
            gauge_str = f"LOCKED: {pct:.1f}% [HOLD: {h_pct}%]"
        g_txt = self.font_small.render(gauge_str, True, (255, 255, 255))
        self.screen.blit(g_txt, (gx + 8, gy + 4))

        # ----------------------------------------------------------------------
        # 8. Tournament Result Phase Overlay
        # ----------------------------------------------------------------------
        if state.phase == Phase.RESULT:
            overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
            overlay.fill((10, 12, 18, 235))
            self.screen.blit(overlay, (0, 0))

            over_title = self.font_title.render("TOURNAMENT COMPLETE!", True, TEXT_GOLD)
            final_score = self.font_large.render(f"FINAL SCORE: {state.score}", True, TEXT_WHITE)

            total_att = state.dodges + state.hits
            acc = int(round((state.dodges / max(1, total_att)) * 100))
            stats = self.font_mid.render(
                f"POSES CLEARED: {state.dodges}/{engine.total_rounds}   ACCURACY: {acc}%   SKIPS: {engine.total_skips}",
                True,
                (0, 220, 255),
            )

            prompt = self.font_mid.render("PRESS [R] TO PLAY AGAIN   |   [ESC] TO EXIT", True, SAFE_GREEN)

            self.screen.blit(over_title, over_title.get_rect(center=(win_w // 2, win_h // 2 - 80)))
            self.screen.blit(final_score, final_score.get_rect(center=(win_w // 2, win_h // 2 - 20)))
            self.screen.blit(stats, stats.get_rect(center=(win_w // 2, win_h // 2 + 30)))
            self.screen.blit(prompt, prompt.get_rect(center=(win_w // 2, win_h // 2 + 90)))

        pygame.display.flip()


# Backward compatibility alias
WallArenaRenderer = BlazerArenaRenderer
