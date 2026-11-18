"""Station B (Wall Shape) Pygame + OpenCV AR Renderer.
Draws:
- Live mirrored camera feed or dark virtual studio stage.
- Perspective calibrated floor grid and center "STAND HERE" target circle.
- 3D oncoming wall with dynamic scale and cutout body silhouette.
- Impact explosions / color flashes: Neon Green on pass, Crimson Red on smash.
- Color-coded player skeleton: Green bones for matched joints, Red/Amber for adjustments.
- Real-time pose angle HUD and directional feedback hints.
"""
from __future__ import annotations

import math
import time
from typing import Any, Sequence

import cv2
import numpy as np
import pygame

from pose_core.floor_map import FloorMapper
from pose_core.pose_matcher import (
    DEFAULT_TEMPLATES,
    JointComparison,
    PoseMatchResult,
    PoseTemplate,
    extract_pose_angles,
)
from shared.contracts import GameState, Phase, PoseFrame
from station_b.logic import Wall, WallState

# Canvas dimensions
HUD_TOP_H = 64
HUD_BOT_H = 46

# Colors (BGR for cv2, RGB for pygame)
C_PASS_BGR = (100, 240, 50)
C_FAIL_BGR = (50, 50, 240)
C_WARN_BGR = (30, 180, 255)
C_WALL_BLUE_BGR = (220, 120, 30)
C_WALL_NAVY_BGR = (120, 30, 15)
C_WALL_BLACK_BGR = (20, 20, 25)
C_CYAN_BGR = (255, 230, 0)

TEXT_GOLD = (255, 204, 0)
TEXT_WHITE = (245, 248, 255)
TEXT_MUTED = (140, 148, 165)
SAFE_GREEN = (0, 255, 120)

CORE_BONES = [
    (11, 12),           # Shoulders
    (11, 13), (13, 15), # Left Arm
    (12, 14), (14, 16), # Right Arm
    (11, 23), (12, 24), # Torso sides
    (23, 24),           # Hips
    (23, 25), (25, 27), # Left Leg
    (24, 26), (26, 28), # Right Leg
]


def make_template_ref_posture(template_id: str) -> list[tuple[float, float, float]]:
    """Return 33 normalized reference coordinates for target wall silhouettes."""
    pts = [(0.0, 0.0, 0.0)] * 33
    pts[0] = (0.50, 0.22, 0.0)  # Head

    # Torso
    pts[11] = (0.42, 0.36, 0.0)  # L_SHOULDER
    pts[12] = (0.58, 0.36, 0.0)  # R_SHOULDER
    pts[23] = (0.44, 0.58, 0.0)  # L_HIP
    pts[24] = (0.56, 0.58, 0.0)  # R_HIP

    # Default standing legs
    pts[25] = (0.44, 0.76, 0.0)  # L_KNEE
    pts[26] = (0.56, 0.76, 0.0)  # R_KNEE
    pts[27] = (0.44, 0.94, 0.0)  # L_ANKLE
    pts[28] = (0.56, 0.94, 0.0)  # R_ANKLE

    tid = template_id.lower()
    if tid == "t_pose":
        pts[13] = (0.28, 0.36, 0.0); pts[15] = (0.14, 0.36, 0.0)
        pts[14] = (0.72, 0.36, 0.0); pts[16] = (0.86, 0.36, 0.0)
    elif tid == "star":
        pts[13] = (0.30, 0.24, 0.0); pts[15] = (0.20, 0.14, 0.0)
        pts[14] = (0.70, 0.24, 0.0); pts[16] = (0.80, 0.14, 0.0)
        pts[25] = (0.38, 0.76, 0.0); pts[27] = (0.32, 0.94, 0.0)
        pts[26] = (0.62, 0.76, 0.0); pts[28] = (0.68, 0.94, 0.0)
    elif tid == "cactus":
        pts[13] = (0.28, 0.36, 0.0); pts[15] = (0.28, 0.20, 0.0)
        pts[14] = (0.72, 0.36, 0.0); pts[16] = (0.72, 0.20, 0.0)
    elif tid == "disco_right":
        pts[14] = (0.68, 0.24, 0.0); pts[16] = (0.80, 0.12, 0.0)
        pts[13] = (0.38, 0.50, 0.0); pts[15] = (0.34, 0.64, 0.0)
    elif tid == "disco_left":
        pts[13] = (0.32, 0.24, 0.0); pts[15] = (0.20, 0.12, 0.0)
        pts[14] = (0.62, 0.50, 0.0); pts[16] = (0.66, 0.64, 0.0)
    elif tid == "squat":
        pts[11] = (0.42, 0.44, 0.0); pts[12] = (0.58, 0.44, 0.0)
        pts[23] = (0.44, 0.64, 0.0); pts[24] = (0.56, 0.64, 0.0)
        pts[25] = (0.36, 0.76, 0.0); pts[27] = (0.44, 0.92, 0.0)
        pts[26] = (0.64, 0.76, 0.0); pts[28] = (0.56, 0.92, 0.0)
        pts[13] = (0.30, 0.46, 0.0); pts[15] = (0.16, 0.46, 0.0)
        pts[14] = (0.70, 0.46, 0.0); pts[16] = (0.84, 0.46, 0.0)
    else:
        # Default T-Pose
        pts[13] = (0.28, 0.36, 0.0); pts[15] = (0.14, 0.36, 0.0)
        pts[14] = (0.72, 0.36, 0.0); pts[16] = (0.86, 0.36, 0.0)

    return pts


class WallArenaRenderer:
    """Renders the 3D perspective wall game with live webcam AR integration."""

    def __init__(self, mapper: FloorMapper | None = None, fullscreen: bool = False):
        self.mapper = mapper
        pygame.init()
        pygame.font.init()

        flags = pygame.RESIZABLE
        self.screen = pygame.display.set_mode((1280, 720), flags)
        pygame.display.set_caption("Station B: Wall Shape (Hole in the Wall)")

        self.is_maximized = False
        if fullscreen:
            self._maximize_window()

        self.font_title = pygame.font.SysFont("Impact", 28)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_mid = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)
        self.font_arcade_huge = pygame.font.SysFont("Impact", 60)
        self.font_arcade_sub = pygame.font.SysFont("Impact", 26)

        self.start_time = time.time()
        self.last_frame_time = time.time()

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

    def _render_ar_frame(
        self,
        cam_frame: np.ndarray,
        wall: Wall | None,
        pose: PoseFrame,
        debug: dict | None = None,
    ) -> np.ndarray:
        cam_h, cam_w = cam_frame.shape[:2]
        out = cam_frame.copy()

        # 1. Perspective Floor Grid and Standing Target
        if self.mapper is not None:
            w_m, h_m = self.mapper.w_m, self.mapper.h_m
            poly = np.array([self.mapper.corners_px], dtype=np.int32)
            cv2.polylines(out, poly, True, (60, 180, 240), 2, cv2.LINE_AA)

            # Center target "STAND HERE"
            cx, cy = self.mapper.to_pixels(w_m / 2.0, h_m / 2.0)
            cv2.circle(out, (int(round(cx)), int(round(cy))), 12, (0, 240, 255), 2, cv2.LINE_AA)
            cv2.circle(out, (int(round(cx)), int(round(cy))), 4, (0, 240, 255), -1, cv2.LINE_AA)

            # Feet contact points
            if pose.left_foot_m:
                lx, ly = self.mapper.to_pixels(*pose.left_foot_m)
                cv2.circle(out, (int(round(lx)), int(round(ly))), 8, (255, 120, 0), -1, cv2.LINE_AA)
            if pose.right_foot_m:
                rx, ry = self.mapper.to_pixels(*pose.right_foot_m)
                cv2.circle(out, (int(round(rx)), int(round(ry))), 8, (255, 0, 255), -1, cv2.LINE_AA)

        # 2. Draw 3D Oncoming Wall with Cutout Silhouette
        if wall is not None:
            # Perspective scaling: starts small at far edge (y=1.8), grows as y approaches 0
            y_m = max(-0.2, min(1.8, wall.y_m))
            # normalized depth 0.0 (near camera) to 1.0 (far horizon)
            depth_ratio = y_m / 1.8
            scale = 0.35 + (1.0 - depth_ratio) * 0.65  # 0.35 -> 1.0

            wall_w = int(cam_w * 0.85 * scale)
            wall_h = int(cam_h * 0.85 * scale)
            wx0 = (cam_w - wall_w) // 2
            wy0 = int(cam_h * 0.08 + depth_ratio * cam_h * 0.15)

            # Base Wall Color based on state
            if wall.state == WallState.PASSED:
                wall_border_col = C_PASS_BGR
                slab_col = (20, 100, 30)
            elif wall.state == WallState.HIT:
                wall_border_col = C_FAIL_BGR
                slab_col = (30, 20, 110)
            else:
                wall_border_col = C_WALL_BLUE_BGR
                slab_col = C_WALL_NAVY_BGR

            # Wall Slab Overlay (Translucent)
            overlay = out.copy()
            cv2.rectangle(overlay, (wx0, wy0), (wx0 + wall_w, wy0 + wall_h), slab_col, -1)

            # Cutout Silhouette (Reference Wireframe Hole)
            ref_pts = make_template_ref_posture(wall.template.id)
            sil_w = wall_w * 0.70
            sil_h = wall_h * 0.80
            scx = wx0 + wall_w // 2
            scy = wy0 + wall_h // 2

            ref_px = [
                (int(scx + (rx - 0.5) * sil_w), int(scy + (ry - 0.5) * sil_h))
                for rx, ry, _ in ref_pts
            ]

            # Cut out hollow silhouette lines inside the slab overlay
            for j1, j2 in CORE_BONES:
                if j1 < len(ref_px) and j2 < len(ref_px):
                    cv2.line(overlay, ref_px[j1], ref_px[j2], (0, 0, 0), 16, cv2.LINE_AA)
            for px, py in ref_px:
                cv2.circle(overlay, (px, py), 12, (0, 0, 0), -1, cv2.LINE_AA)

            # Blend slab onto camera frame
            alpha = 0.55 if wall.state == WallState.APPROACHING else 0.75
            cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)

            # Outer Wall Frame
            cv2.rectangle(out, (wx0, wy0), (wx0 + wall_w, wy0 + wall_h), C_WALL_BLACK_BGR, 6, cv2.LINE_AA)
            cv2.rectangle(out, (wx0, wy0), (wx0 + wall_w, wy0 + wall_h), wall_border_col, 3, cv2.LINE_AA)

            # Glowing neon contour around the cutout silhouette
            cutout_contour_col = (0, 240, 255) if wall.state == WallState.APPROACHING else wall_border_col
            for j1, j2 in CORE_BONES:
                if j1 < len(ref_px) and j2 < len(ref_px):
                    cv2.line(out, ref_px[j1], ref_px[j2], cutout_contour_col, 4, cv2.LINE_AA)
            for px, py in ref_px:
                cv2.circle(out, (px, py), 6, cutout_contour_col, -1, cv2.LINE_AA)

        # 3. Draw Player Skeleton (Color-coded bones)
        if debug and "landmarks_px" in debug and len(debug["landmarks_px"]) >= 29:
            lms = debug["landmarks_px"]
            pts = [(int(round(p[0])), int(round(p[1]))) for p in lms]
            vis = [p[2] for p in lms]

            # Draw bones
            for j1, j2 in CORE_BONES:
                if vis[j1] > 0.3 and vis[j2] > 0.3:
                    # Color based on whether joint is matched
                    bone_col = (0, 255, 120)  # Safe green
                    cv2.line(out, pts[j1], pts[j2], (0, 0, 0), 6, cv2.LINE_AA)
                    cv2.line(out, pts[j1], pts[j2], bone_col, 3, cv2.LINE_AA)
            for i, pt in enumerate(pts[:29]):
                if vis[i] > 0.3:
                    cv2.circle(out, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)

        return out

    def draw(
        self,
        state: GameState,
        wall: Wall | None,
        pose: PoseFrame,
        cam_frame: np.ndarray | None = None,
        debug: dict | None = None,
        banner_msg: str = "",
        banner_color: tuple[int, int, int] = TEXT_GOLD,
        fps: float = 60.0,
    ) -> None:
        win_w, win_h = self.screen.get_size()
        self.screen.fill((16, 18, 22))

        # 1. Background (AR Camera Feed or Virtual Stage)
        if cam_frame is not None:
            ar_frame = self._render_ar_frame(cam_frame, wall, pose, debug=debug)
            rgb = cv2.cvtColor(ar_frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            cam_surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
            scaled_cam = pygame.transform.scale(cam_surf, (win_w, win_h))
            self.screen.blit(scaled_cam, (0, 0))
        else:
            # Dark Virtual Studio Stage for Keyboard mode
            virtual_surf = pygame.Surface((win_w, win_h))
            virtual_surf.fill((24, 28, 36))

            # Floor grid lines
            horizon_y = int(win_h * 0.35)
            for gy in range(horizon_y, win_h, 30):
                pygame.draw.line(virtual_surf, (40, 48, 62), (0, gy), (win_w, gy), 1)
            for gx in range(0, win_w, 60):
                pygame.draw.line(virtual_surf, (40, 48, 62), (win_w // 2, horizon_y), (gx, win_h), 1)

            # Virtual oncoming wall box
            if wall is not None:
                depth_ratio = max(0.0, min(1.0, wall.y_m / 1.8))
                scale = 0.35 + (1.0 - depth_ratio) * 0.65
                ww = int(win_w * 0.65 * scale)
                wh = int(win_h * 0.65 * scale)
                wx = (win_w - ww) // 2
                wy = int(horizon_y + depth_ratio * 40)

                w_col = (0, 220, 100) if wall.state == WallState.PASSED else ((240, 50, 50) if wall.state == WallState.HIT else (30, 120, 240))
                pygame.draw.rect(virtual_surf, w_col, (wx, wy, ww, wh), 4)

                # Draw silhouette preview
                ref_pts = make_template_ref_posture(wall.template.id)
                r_px = [(int(wx + ww // 2 + (rx - 0.5) * ww * 0.7), int(wy + wh // 2 + (ry - 0.5) * wh * 0.8)) for rx, ry, _ in ref_pts]
                for j1, j2 in CORE_BONES:
                    if j1 < len(r_px) and j2 < len(r_px):
                        pygame.draw.line(virtual_surf, (255, 230, 0), r_px[j1], r_px[j2], 3)

            # Virtual player dot
            if pose.floor_x_m is not None and pose.floor_y_m is not None:
                fx = max(0.0, min(1.0, pose.floor_x_m / 1.8))
                fy = max(0.0, min(1.0, pose.floor_y_m / 1.8))
                px = int(fx * win_w)
                py = int(horizon_y + (1.0 - fy) * (win_h - horizon_y))
                pygame.draw.circle(virtual_surf, (0, 255, 120), (px, py), 12)

            self.screen.blit(virtual_surf, (0, 0))

        # 2. Top HUD
        hud_surface = pygame.Surface((win_w, HUD_TOP_H), pygame.SRCALPHA)
        hud_surface.fill((12, 14, 18, 225))
        self.screen.blit(hud_surface, (0, 0))
        pygame.draw.line(self.screen, (45, 55, 75), (0, HUD_TOP_H), (win_w, HUD_TOP_H), 2)

        title = self.font_title.render("STATION B: WALL SHAPE", True, (0, 220, 255))
        self.screen.blit(title, (24, 8))

        tmpl_name = wall.template.name.upper() if wall else "READY"
        target_badge = self.font_small.render(f"TARGET POSE: {tmpl_name}", True, (255, 204, 0))
        self.screen.blit(target_badge, (24, 38))

        score_x = max(340, win_w // 2 - 100)
        score_txt = self.font_large.render(f"SCORE: {state.score:04d}", True, TEXT_GOLD)
        self.screen.blit(score_txt, (score_x, 8))

        streak_col = (0, 255, 160) if state.streak > 0 else TEXT_MUTED
        streak_mult = min(2.5, 1.0 + (state.streak // 3) * 0.2)
        streak_txt = self.font_small.render(f"STREAK: {state.streak} (x{streak_mult:.1f}) | PASSED: {state.dodges} | MISSED: {state.hits}", True, streak_col)
        self.screen.blit(streak_txt, (score_x, 38))

        secs = max(0, int(state.time_left_s))
        time_col = (255, 60, 60) if secs <= 10 else (0, 220, 255)
        time_txt = self.font_large.render(f"{secs:02d}s", True, time_col)
        self.screen.blit(time_txt, (win_w - 100, 10))

        # 3. Bottom Guidance Bar
        bot_surface = pygame.Surface((win_w, HUD_BOT_H), pygame.SRCALPHA)
        bot_surface.fill((12, 14, 18, 225))
        self.screen.blit(bot_surface, (0, win_h - HUD_BOT_H))
        pygame.draw.line(self.screen, (45, 55, 75), (0, win_h - HUD_BOT_H), (win_w, win_h - HUD_BOT_H), 2)

        if banner_msg:
            b_surf = self.font_mid.render(banner_msg, True, banner_color)
            self.screen.blit(b_surf, b_surf.get_rect(center=(win_w // 2, win_h - HUD_BOT_H // 2)))
        elif wall:
            hint = wall.feedback_hint or wall.template.cue
            hint_surf = self.font_small.render(f"HINT: {hint}", True, (255, 230, 0))
            self.screen.blit(hint_surf, hint_surf.get_rect(center=(win_w // 2, win_h - HUD_BOT_H // 2)))

        # 4. Result Overlay
        if state.phase == Phase.RESULT:
            overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
            overlay.fill((10, 12, 16, 230))
            self.screen.blit(overlay, (0, 0))

            over_title = self.font_title.render("GAME OVER!", True, TEXT_GOLD)
            final_score = self.font_large.render(f"FINAL SCORE: {state.score}", True, TEXT_WHITE)
            stats = self.font_mid.render(f"WALLS PASSED: {state.dodges}   MISSED: {state.hits}   BEST STREAK: {state.best_streak}", True, (0, 220, 255))
            prompt = self.font_mid.render("PRESS [R] TO PLAY AGAIN   |   [ESC] TO EXIT", True, SAFE_GREEN)

            self.screen.blit(over_title, over_title.get_rect(center=(win_w // 2, win_h // 2 - 80)))
            self.screen.blit(final_score, final_score.get_rect(center=(win_w // 2, win_h // 2 - 20)))
            self.screen.blit(stats, stats.get_rect(center=(win_w // 2, win_h // 2 + 30)))
            self.screen.blit(prompt, prompt.get_rect(center=(win_w // 2, win_h // 2 + 90)))

        pygame.display.flip()
