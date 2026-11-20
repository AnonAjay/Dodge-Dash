"""Station B: Retro 8-Bit HD Arcade AR Renderer for Pose Blazer Tournament.
Features:
- Retro 8-bit pixel bitmap typography (Titans' Arcade sunset gradient & drop shadow).
- Split-screen HD layout (Target Reference Card on left, Live Player AR feed on right).
- Cyber-arcade glowing neon borders with corner brackets.
- 6-limb color-coded bone skeleton with glowing joint halos.
- Segmented arcade Match Gauge with pass threshold lock indicator and hold animation.
- Animated floating banners for Stage transitions, Snap locks, and Perfection bonuses.
- Folded-arms skip charging bar indicator.
- Fullscreen retro tournament result / victory screen with Titan Rank badges.
- Auto-maximized resizable window with [F11] toggle.
"""
from __future__ import annotations

import math
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

# Geometry
HUD_TOP_H = 78
HUD_BOT_H = 46

# Colors
C_BG = (14, 16, 22)
C_PANEL_BG = (18, 22, 32, 230)
C_BORDER = (42, 54, 76)
C_BORDER_NEON = (0, 220, 255)
C_NEON_GREEN = (0, 255, 128)
C_NEON_CYAN = (0, 230, 255)
C_GOLD = (255, 215, 0)
C_ORANGE = (255, 130, 20)
C_RED = (255, 55, 75)
C_WHITE = (245, 248, 255)
C_MUTED = (140, 155, 175)

# ==============================================================================
# RETRO 8-BIT PIXEL BITMAP FONT RENDERER (TITANS' ARCADE)
# ==============================================================================
PIXEL_GLYPHS: dict[str, list[str]] = {
    "A": [" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
    "B": ["#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "],
    "C": [" ####", "#    ", "#    ", "#    ", "#    ", "#    ", " ####"],
    "D": ["#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "],
    "E": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"],
    "F": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "],
    "G": [" ####", "#    ", "#    ", "# ###", "#   #", "#   #", " ####"],
    "H": ["#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
    "I": ["###", " # ", " # ", " # ", " # ", " # ", "###"],
    "J": ["  ###", "    #", "    #", "    #", "    #", "#   #", " ### "],
    "K": ["#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"],
    "L": ["#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"],
    "M": ["#   #", "## ##", "# # #", "#   #", "#   #", "#   #", "#   #"],
    "N": ["#   #", "##  #", "# # #", "#  ##", "#   #", "#   #", "#   #"],
    "O": [" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
    "P": ["#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "],
    "Q": [" ### ", "#   #", "#   #", "#   #", "# # #", "#  ##", " ####"],
    "R": ["#### ", "#   #", "#   #", "#### ", "#  # ", "#   #", "#   #"],
    "S": [" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "],
    "s": ["     ", "     ", "#### ", "#    ", "#### ", "    #", "#### "],
    "T": ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "],
    "U": ["#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
    "V": ["#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "],
    "W": ["#   #", "#   #", "#   #", "# # #", "# # #", "## ##", "#   #"],
    "X": ["#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"],
    "Y": ["#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "],
    "Z": ["#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"],
    "0": [" ### ", "#  ##", "# # #", "# # #", "##  #", "#   #", " ### "],
    "1": [" ## ", "  # ", "  # ", "  # ", "  # ", "  # ", " ###"],
    "2": [" ### ", "#   #", "    #", "  ## ", " #   ", "#    ", "#####"],
    "3": ["#### ", "    #", "    #", " ### ", "    #", "    #", "#### "],
    "4": ["#   #", "#   #", "#   #", "#####", "    #", "    #", "    #"],
    "5": ["#####", "#    ", "#    ", "#### ", "    #", "    #", "#### "],
    "6": [" ### ", "#    ", "#    ", "#### ", "#   #", "#   #", " ### "],
    "7": ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "],
    "8": [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "],
    "9": [" ### ", "#   #", "#   #", " ####", "    #", "    #", " ### "],
    ":": [" ", "#", " ", " ", "#", " ", " "],
    "!": ["#", "#", "#", "#", "#", " ", "#"],
    "?": ["### ", "   #", "  # ", " #  ", " #  ", "    ", " #  "],
    ".": [" ", " ", " ", " ", " ", "#", "#"],
    ",": [" ", " ", " ", " ", " ", " #", "# "],
    "-": ["   ", "   ", "   ", "###", "   ", "   ", "   "],
    "/": ["    #", "   # ", "   # ", "  #  ", " #   ", " #   ", "#    "],
    "(": ["  #", " # ", "#  ", "#  ", "#  ", " # ", "  #"],
    ")": ["#  ", " # ", "  #", "  #", "  #", " # ", "#  "],
    "[": ["###", "#  ", "#  ", "#  ", "#  ", "#  ", "###"],
    "]": ["###", "  #", "  #", "  #", "  #", "  #", "###"],
    "+": ["   ", " # ", " # ", "###", " # ", " # ", "   "],
    "|": ["#", "#", "#", "#", "#", "#", "#"],
    "%": ["#   #", "#  # ", "  #  ", " #  #", "#   #", "    #", "    #"],
    "'": ["#", "#", " ", " ", " ", " ", " "],
    "\"": ["# #", "# #", "   ", "   ", "   ", "   ", "   "],
    "&": [" ### ", "#   #", " # # ", "  #  ", "# # #", "#  # ", " ## #"],
    "<": ["   #", "  # ", " #  ", "#   ", " #  ", "  # ", "   #"],
    ">": ["#   ", " #  ", "  # ", "   #", "  # ", " #  ", "#   "],
    " ": ["   ", "   ", "   ", "   ", "   ", "   ", "   "],
}

_PIXEL_TEXT_CACHE: dict[tuple, pygame.Surface] = {}


def render_pixel_text(
    text: str,
    scale: int = 3,
    top_color: tuple[int, int, int] = (255, 224, 51),
    bottom_color: tuple[int, int, int] = (232, 56, 24),
    outline_color: tuple[int, int, int] | None = (18, 10, 32),
    shadow_offset: int = 3,
) -> pygame.Surface:
    """Render crisp retro arcade pixel text with signature two-tone gradient and drop shadow."""
    cache_key = (text, scale, top_color, bottom_color, outline_color, shadow_offset)
    if cache_key in _PIXEL_TEXT_CACHE:
        return _PIXEL_TEXT_CACHE[cache_key]

    cols = []
    for ch in text:
        if ch in PIXEL_GLYPHS:
            glyph = PIXEL_GLYPHS[ch]
        elif ch.upper() in PIXEL_GLYPHS:
            glyph = PIXEL_GLYPHS[ch.upper()]
        else:
            glyph = PIXEL_GLYPHS["?"]
        gw = len(glyph[0])
        cols.append((glyph, gw))

    spacing = 1
    total_w = sum(gw for _, gw in cols) + max(0, len(cols) - 1) * spacing
    if total_w <= 0:
        total_w = 1

    grid_w = total_w + 2
    grid_h = 7 + 2

    fg_map: dict[tuple[int, int], tuple[int, int, int]] = {}
    curr_x = 1
    for glyph, gw in cols:
        for r, row_str in enumerate(glyph):
            col_tone = top_color if r < 4 else bottom_color
            for c, ch in enumerate(row_str):
                if ch == "#":
                    fg_map[(curr_x + c, 1 + r)] = col_tone
        curr_x += gw + spacing

    outline_set: set[tuple[int, int]] = set()
    if outline_color is not None:
        for (fx, fy) in fg_map:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = fx + dx, fy + dy
                    if 0 <= nx < grid_w and 0 <= ny < grid_h and (nx, ny) not in fg_map:
                        outline_set.add((nx, ny))

    surf_1x = pygame.Surface((grid_w, grid_h), pygame.SRCALPHA)
    for (ox, oy) in outline_set:
        surf_1x.set_at((ox, oy), (*outline_color, 255))
    for (fx, fy), col in fg_map.items():
        surf_1x.set_at((fx, fy), (*col, 255))

    scaled_w = grid_w * scale
    scaled_h = grid_h * scale
    scaled_main = pygame.transform.scale(surf_1x, (scaled_w, scaled_h))

    if shadow_offset > 0:
        shadow_1x = pygame.Surface((grid_w, grid_h), pygame.SRCALPHA)
        all_pts = outline_set.union(fg_map.keys())
        shadow_col = (10, 5, 18, 220)
        for (sx, sy) in all_pts:
            shadow_1x.set_at((sx, sy), shadow_col)
        scaled_shadow = pygame.transform.scale(shadow_1x, (scaled_w, scaled_h))

        final_surf = pygame.Surface((scaled_w + shadow_offset, scaled_h + shadow_offset), pygame.SRCALPHA)
        final_surf.blit(scaled_shadow, (shadow_offset, shadow_offset))
        final_surf.blit(scaled_main, (0, 0))
    else:
        final_surf = scaled_main

    _PIXEL_TEXT_CACHE[cache_key] = final_surf
    return final_surf


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
        pygame.display.set_caption("Titans' Arcade — Station B: Pose Blazer Tournament")

        self.is_maximized = False
        if fullscreen:
            self._maximize_window()

        # Supplementary system fonts for telemetry & dense readouts
        self.font_data = pygame.font.SysFont("Consolas", 13, bold=True)
        self.font_mini = pygame.font.SysFont("Consolas", 11)

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

        # Animated Floating Banner state
        self.banner_timer: float = 0.0
        self.last_banner_text: str = ""
        self.last_banner_surf: pygame.Surface | None = None
        self.pulse_phase: float = 0.0

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

    def _draw_cyber_frame(self, surf: pygame.Surface, rect: tuple[int, int, int, int], color: tuple[int, int, int] = C_BORDER_NEON) -> None:
        """Draw an arcade cyber border with reinforced glowing corner brackets."""
        x, y, w, h = rect
        # Base frame
        pygame.draw.rect(surf, (30, 38, 54), (x, y, w, h), 2)
        # Corner brackets (12px length)
        c_len = 14
        t = 3
        # Top-left
        pygame.draw.line(surf, color, (x, y), (x + c_len, y), t)
        pygame.draw.line(surf, color, (x, y), (x, y + c_len), t)
        # Top-right
        pygame.draw.line(surf, color, (x + w - c_len, y), (x + w, y), t)
        pygame.draw.line(surf, color, (x + w, y), (x + w, y + c_len), t)
        # Bottom-left
        pygame.draw.line(surf, color, (x, y + h), (x + c_len, y + h), t)
        pygame.draw.line(surf, color, (x, y + h - c_len), (x, y + h), t)
        # Bottom-right
        pygame.draw.line(surf, color, (x + w - c_len, y + h), (x + w, y + h), t)
        pygame.draw.line(surf, color, (x + w, y + h - c_len), (x + w, y + h), t)

    def draw(
        self,
        state: GameState,
        engine: BlazerGameEngine,
        cam_frame: np.ndarray | None = None,
        debug_info: dict | None = None,
        match_res: BlazerMatchResult | None = None,
        player_angles: dict[str, float | None] | None = None,
        fps: float = 60.0,
        dt: float = 0.016,
        is_connected: bool = False,
        last_player_name: str = "",
        last_player_score: int | None = None,
    ) -> None:
        win_w, win_h = self.screen.get_size()
        self.screen.fill(C_BG)
        self.pulse_phase = (self.pulse_phase + dt * 4.0) % (2.0 * math.pi)

        curr_pose = engine.current_pose

        # ----------------------------------------------------------------------
        # 1. Split Layout Coordinates
        # ----------------------------------------------------------------------
        top_h = HUD_TOP_H
        bot_h = HUD_BOT_H
        play_h = win_h - top_h - bot_h

        # Left Column: Target Reference Card (33% width)
        card_w = min(420, max(320, int(win_w * 0.33)))
        card_h = play_h - 16
        card_x = 14
        card_y = top_h + 8

        # Right Column: AR Live Camera Feed
        cam_x = card_x + card_w + 14
        cam_y = card_y
        cam_w = win_w - cam_x - 14
        cam_h = card_h

        # ----------------------------------------------------------------------
        # 2. Render Left Reference Pose Card
        # ----------------------------------------------------------------------
        card_container = pygame.Surface((card_w, card_h))
        card_container.fill((16, 20, 28))

        if curr_pose:
            card_bgr = render_reference_card(
                curr_pose,
                width=card_w,
                height=card_h,
                poses_dir=self.poses_dir,
            )
            card_rgb = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2RGB)
            card_surf = pygame.image.frombuffer(card_rgb.tobytes(), (card_w, card_h), "RGB")
            card_container.blit(card_surf, (0, 0))

            # Header Badge: Difficulty & Target Pose Name in Pixel Font
            diff_str = curr_pose.difficulty.upper()
            diff_col_top = (0, 255, 128) if diff_str == "EASY" else ((255, 200, 0) if diff_str == "MEDIUM" else (255, 60, 60))
            diff_col_bot = (0, 160, 80) if diff_str == "EASY" else ((200, 100, 0) if diff_str == "MEDIUM" else (180, 20, 30))

            card_badge = render_pixel_text(
                f"[{diff_str}] {curr_pose.name.upper()[:16]}",
                scale=2,
                top_color=diff_col_top,
                bottom_color=diff_col_bot,
                shadow_offset=2,
            )
            card_container.blit(card_badge, (12, 10))

            # Subtle bottom hint pill if description available
            if curr_pose.description:
                desc_bg = pygame.Surface((card_w - 16, 26), pygame.SRCALPHA)
                desc_bg.fill((10, 12, 18, 225))
                card_container.blit(desc_bg, (8, card_h - 32))
                d_txt = self.font_mini.render(curr_pose.description[:46], True, (180, 200, 220))
                card_container.blit(d_txt, (14, card_h - 26))

        self.screen.blit(card_container, (card_x, card_y))
        self._draw_cyber_frame(self.screen, (card_x, card_y, card_w, card_h), C_NEON_CYAN)

        # ----------------------------------------------------------------------
        # 3. Render Right Camera Feed (Clean Live View)
        # ----------------------------------------------------------------------
        cam_surface = pygame.Surface((cam_w, cam_h))
        cam_surface.fill((20, 24, 34))

        if cam_frame is not None:
            ar_canvas = cv2.resize(cam_frame, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
            ch, cw = ar_canvas.shape[:2]
            rgb_cam = cv2.cvtColor(ar_canvas, cv2.COLOR_BGR2RGB)
            cam_feed_surf = pygame.image.frombuffer(rgb_cam.tobytes(), (cw, ch), "RGB")
            cam_surface.blit(cam_feed_surf, (0, 0))
        else:
            # Virtual Grid Background
            for y_line in range(0, cam_h, 32):
                pygame.draw.line(cam_surface, (28, 34, 46), (0, y_line), (cam_w, y_line), 1)

        self.screen.blit(cam_surface, (cam_x, cam_y))
        self._draw_cyber_frame(self.screen, (cam_x, cam_y, cam_w, cam_h), (80, 110, 150))

        # ----------------------------------------------------------------------
        # 4. Skip Gesture Feedback Bar (Hand on Head)
        # ----------------------------------------------------------------------
        if engine.skip_hold_s > 0.03:
            skip_w = int(cam_w * 0.44)
            skip_h = 24
            sx = cam_x + 16
            sy = cam_y + 16

            s_bg = pygame.Surface((skip_w, skip_h), pygame.SRCALPHA)
            s_bg.fill((20, 10, 12, 230))
            self.screen.blit(s_bg, (sx, sy))

            skip_req = 0.40 if getattr(engine, "current_round", 1) == 2 else 0.50
            pct_skip = min(1.0, engine.skip_hold_s / skip_req)
            fill_sw = int(skip_w * pct_skip)
            pygame.draw.rect(self.screen, (255, 140, 20), (sx, sy, fill_sw, skip_h))
            pygame.draw.rect(self.screen, (255, 80, 80), (sx, sy, skip_w, skip_h), 2)

            skip_txt = render_pixel_text(
                f"HAND ON HEAD: SKIPPING... {int(pct_skip * 100)}%",
                scale=1,
                top_color=(255, 255, 255),
                bottom_color=(255, 200, 200),
                shadow_offset=1,
            )
            self.screen.blit(skip_txt, (sx + 6, sy + 5))

        # ----------------------------------------------------------------------
        # 5. Top Tournament Arcade HUD (Sunset Gradient Pixel Fonts)
        # ----------------------------------------------------------------------
        hud_surface = pygame.Surface((win_w, top_h), pygame.SRCALPHA)
        hud_surface.fill((10, 12, 18, 240))
        self.screen.blit(hud_surface, (0, 0))
        pygame.draw.line(self.screen, (45, 58, 80), (0, top_h), (win_w, top_h), 2)

        # Stage Badge (Left)
        r_num = engine.current_round
        r_name = engine.current_stage_name
        if r_num == 1:
            stg_top, stg_bot = (0, 240, 255), (0, 130, 255)
        else:
            stg_top, stg_bot = (255, 170, 0), (255, 50, 0)

        stage_surf = render_pixel_text(f"ROUND {r_num}: {r_name}", scale=3, top_color=stg_top, bottom_color=stg_bot)
        self.screen.blit(stage_surf, (18, 10))

        if getattr(engine, "tournament_phase", None) == "intermission":
            cd_s = int(math.ceil(engine.intermission_timer_s))
            sub_str = f"INTERMISSION — STARTING IN {cd_s}..."
            sub_col = (255, 200, 50)
        elif r_num == 1:
            sub_str = f"POSE {engine.round_number}/10   |   PERFECTIONS: {engine.perfections_count}"
            sub_col = (200, 220, 240)
        else:
            sub_str = f"GLOBAL SPEEDRUN   |   CLEARED: {engine.r2_cleared}"
            sub_col = (255, 180, 50)

        round_surf = render_pixel_text(sub_str, scale=2, top_color=sub_col, bottom_color=(120, 140, 170))
        self.screen.blit(round_surf, (18, 48))

        # Score & Status (Center) — NO STREAK
        score_x = max(380, win_w // 2 - 140)
        score_surf = render_pixel_text(f"SCORE: {state.score:04d}", scale=4, top_color=(255, 224, 51), bottom_color=(232, 56, 24))
        self.screen.blit(score_surf, (score_x, 8))

        if r_num == 1:
            skips_left = max(0, 4 - engine.consecutive_skips)
            status_str = f"CLEARED: {state.dodges}/10   |   PERFECTION: {engine.perfections_count}   |   SKIPS: {skips_left} LEFT"
        else:
            status_str = f"CLEARED: {state.dodges}   |   GLOBAL PACE: FAST"

        status_surf = render_pixel_text(
            status_str,
            scale=2,
            top_color=(0, 255, 128),
            bottom_color=(0, 170, 70),
        )
        self.screen.blit(status_surf, (score_x, 48))

        # Countdown Timer (Right)
        secs = max(0.0, engine.time_left_s)
        time_ratio = max(0.0, min(1.0, secs / max(0.1, engine.round_duration_s)))

        if secs > 3.0:
            t_col_top, t_col_bot = (0, 240, 255), (0, 140, 255)
        elif secs > 1.5:
            t_col_top, t_col_bot = (255, 200, 0), (230, 100, 0)
        else:
            t_col_top, t_col_bot = (255, 60, 60), (180, 20, 20)

        t_box_w = 160
        t_box_h = 14
        t_x = win_w - t_box_w - 20
        t_y = 50

        # Progress bar
        pygame.draw.rect(self.screen, (25, 32, 45), (t_x, t_y, t_box_w, t_box_h), border_radius=2)
        pygame.draw.rect(self.screen, t_col_top, (t_x, t_y, int(t_box_w * time_ratio), t_box_h), border_radius=2)
        pygame.draw.rect(self.screen, (80, 100, 130), (t_x, t_y, t_box_w, t_box_h), 1, border_radius=2)

        time_val_surf = render_pixel_text(f"{secs:04.1f}S", scale=4, top_color=t_col_top, bottom_color=t_col_bot)
        self.screen.blit(time_val_surf, (t_x - 100, 12))

        # ----------------------------------------------------------------------
        # 8. Bottom Guidance Marquee Banner
        # ----------------------------------------------------------------------
        bot_surface = pygame.Surface((win_w, bot_h), pygame.SRCALPHA)
        bot_surface.fill((10, 12, 18, 240))
        self.screen.blit(bot_surface, (0, win_h - bot_h))
        pygame.draw.line(self.screen, (45, 58, 80), (0, win_h - bot_h), (win_w, win_h - bot_h), 2)

        bot_msg = ""
        bot_top, bot_bot = (255, 215, 0), (220, 120, 0)
        if engine.banner_msg:
            bot_msg = engine.banner_msg.upper()
            if "SNAP" in bot_msg or "PERFECTION" in bot_msg:
                bot_top, bot_bot = (0, 255, 128), (0, 180, 70)
            elif "SKIPPED" in bot_msg or "ELIMINATED" in bot_msg or "MISS" in bot_msg:
                bot_top, bot_bot = (255, 70, 70), (190, 20, 20)
        elif match_res:
            bot_msg = f"HINT: {match_res.primary_hint.upper()}"

        if bot_msg:
            b_surf = render_pixel_text(bot_msg[:54], scale=2, top_color=bot_top, bottom_color=bot_bot)
            self.screen.blit(b_surf, b_surf.get_rect(center=(win_w // 2, win_h - bot_h // 2)))

        # ----------------------------------------------------------------------
        # 9. Animated Floating Banner (Lock-in, Perfection, Stage Change)
        # ----------------------------------------------------------------------
        if engine.banner_msg != self.last_banner_text and engine.banner_msg:
            self.last_banner_text = engine.banner_msg
            self.banner_timer = 1.6

        if self.banner_timer > 0.0:
            self.banner_timer = max(0.0, self.banner_timer - dt)
            alpha_ratio = min(1.0, self.banner_timer / 0.3) if self.banner_timer < 0.3 else 1.0

            # Only float high-impact banners in center
            if any(k in self.last_banner_text for k in ("PERFECTION", "SNAP", "STAGE", "ELIMINATED")):
                float_top = (0, 255, 128) if "PERFECTION" in self.last_banner_text else (
                    (0, 230, 255) if "SNAP" in self.last_banner_text else (
                        (255, 50, 50) if "ELIMINATED" in self.last_banner_text else (255, 180, 0)
                    )
                )
                float_bot = (0, 160, 60) if "PERFECTION" in self.last_banner_text else (
                    (0, 120, 200) if "SNAP" in self.last_banner_text else (
                        (180, 10, 10) if "ELIMINATED" in self.last_banner_text else (220, 80, 0)
                    )
                )

                f_surf = render_pixel_text(
                    self.last_banner_text[:38],
                    scale=4,
                    top_color=float_top,
                    bottom_color=float_bot,
                    shadow_offset=4,
                )
                fw, fh = f_surf.get_size()
                bg_w, bg_h = fw + 40, fh + 24
                f_bg = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
                f_bg.fill((8, 10, 16, int(220 * alpha_ratio)))
                pygame.draw.rect(f_bg, float_top, (0, 0, bg_w, bg_h), 2)

                cy = win_h // 2 - 40
                self.screen.blit(f_bg, f_bg.get_rect(center=(win_w // 2, cy)))
                self.screen.blit(f_surf, f_surf.get_rect(center=(win_w // 2, cy)))

        # ----------------------------------------------------------------------
        # 10. Intermission Countdown Overlay
        # ----------------------------------------------------------------------
        if getattr(engine, "tournament_phase", None) == "intermission":
            inter_overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
            inter_overlay.fill((8, 10, 16, 215))
            self.screen.blit(inter_overlay, (0, 0))

            cd_s = int(math.ceil(engine.intermission_timer_s))
            r_num = engine.current_round
            r_name = engine.current_stage_name
            t_top = (0, 240, 255) if r_num == 1 else (255, 170, 0)
            t_bot = (0, 130, 255) if r_num == 1 else (255, 50, 0)

            inter_title = render_pixel_text(f"ROUND {r_num}: {r_name}", scale=5, top_color=t_top, bottom_color=t_bot, shadow_offset=5)
            self.screen.blit(inter_title, inter_title.get_rect(center=(win_w // 2, win_h // 2 - 50)))

            sub_msg = f"STARTING IN {cd_s}..." if cd_s > 0 else "STRIKE POSE!"
            inter_sub = render_pixel_text(sub_msg, scale=4, top_color=(255, 220, 50), bottom_color=(230, 100, 20), shadow_offset=3)
            self.screen.blit(inter_sub, inter_sub.get_rect(center=(win_w // 2, win_h // 2 + 40)))

        # ----------------------------------------------------------------------
        # 11. Tournament Result Phase Overlay
        # ----------------------------------------------------------------------
        if state.phase == Phase.RESULT:
            overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
            overlay.fill((8, 10, 16, 245))
            self.screen.blit(overlay, (0, 0))

            # Upper Big Heading (H1)
            over_title = render_pixel_text(
                "TOURNAMENT ENDED",
                scale=5,
                top_color=(255, 224, 51),
                bottom_color=(232, 56, 24),
                shadow_offset=5,
            )
            self.screen.blit(over_title, over_title.get_rect(center=(win_w // 2, int(win_h * 0.16))))

            # Center Hero: Final Score (H2)
            score_str = f"FINAL SCORE: {state.score:04d} PTS"
            final_score = render_pixel_text(
                score_str,
                scale=5,
                top_color=(0, 240, 255),
                bottom_color=(0, 140, 255),
                shadow_offset=4,
            )
            self.screen.blit(final_score, final_score.get_rect(center=(win_w // 2, int(win_h * 0.36))))

            # Center Hero: Titan Rank (H3)
            if state.score >= 1200:
                rank_str, r_top, r_bot = "TITAN RANK: S+ (MASTER POSE BLAZER)", (255, 220, 50), (255, 120, 0)
            elif state.score >= 800:
                rank_str, r_top, r_bot = "TITAN RANK: A (SUPERIOR STRIKER)", (0, 255, 128), (0, 160, 70)
            elif state.score >= 400:
                rank_str, r_top, r_bot = "TITAN RANK: B (AGILE CONTENDER)", (0, 220, 255), (0, 130, 220)
            else:
                rank_str, r_top, r_bot = "TITAN RANK: C (NOVICE DRIFTER)", (255, 140, 30), (200, 60, 20)

            rank_surf = render_pixel_text(rank_str, scale=3, top_color=r_top, bottom_color=r_bot, shadow_offset=3)
            self.screen.blit(rank_surf, rank_surf.get_rect(center=(win_w // 2, int(win_h * 0.48))))

            # Beneath Center: Performance Statistics
            total_att = state.dodges + state.hits
            acc = int(round((state.dodges / max(1, total_att)) * 100))
            stats_str = f"CLEARED: {state.dodges}/{engine.total_rounds}   ACCURACY: {acc}%   SKIPS: {engine.total_skips}"
            stats = render_pixel_text(
                stats_str,
                scale=2,
                top_color=(220, 230, 245),
                bottom_color=(140, 155, 180),
                shadow_offset=2,
            )
            self.screen.blit(stats, stats.get_rect(center=(win_w // 2, int(win_h * 0.59))))

            # Highlighted Buttons: Bottom-Left (Retry) and Bottom-Right (Exit)
            btn_w, btn_h = 240, 50
            bl_x, bl_y = 40, win_h - btn_h - 26
            bl_surf = pygame.Surface((btn_w, btn_h), pygame.SRCALPHA)
            bl_surf.fill((10, 36, 22, 230))
            self.screen.blit(bl_surf, (bl_x, bl_y))
            self._draw_cyber_frame(self.screen, (bl_x, bl_y, btn_w, btn_h), C_NEON_GREEN)
            r_prompt = render_pixel_text("[R] PLAY AGAIN", scale=2, top_color=(0, 255, 128), bottom_color=(0, 170, 70), shadow_offset=2)
            self.screen.blit(r_prompt, r_prompt.get_rect(center=(bl_x + btn_w // 2, bl_y + btn_h // 2)))

            br_x, br_y = win_w - btn_w - 40, win_h - btn_h - 26
            br_surf = pygame.Surface((btn_w, btn_h), pygame.SRCALPHA)
            br_surf.fill((38, 12, 16, 230))
            self.screen.blit(br_surf, (br_x, br_y))
            self._draw_cyber_frame(self.screen, (br_x, br_y, btn_w, btn_h), C_RED)
            esc_prompt = render_pixel_text("[ESC] EXIT", scale=2, top_color=(255, 70, 70), bottom_color=(190, 20, 20), shadow_offset=2)
            self.screen.blit(esc_prompt, esc_prompt.get_rect(center=(br_x + btn_w // 2, br_y + btn_h // 2)))

        # ----------------------------------------------------------------------
        # 12. Attract / Start Menu Overlay (Phase.IDLE)
        # ----------------------------------------------------------------------
        if state.phase == Phase.IDLE:
            overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
            overlay.fill((8, 10, 16, 220))
            self.screen.blit(overlay, (0, 0))

            # Header
            arcade_tag = render_pixel_text("TITANS' ARCADE — STATION B", scale=2, top_color=(0, 240, 255), bottom_color=(0, 140, 255), shadow_offset=2)
            self.screen.blit(arcade_tag, arcade_tag.get_rect(center=(win_w // 2, int(win_h * 0.12))))

            title_surf = render_pixel_text(
                "POSE BLAZER",
                scale=5 if win_w > 900 else 4,
                top_color=(255, 224, 51),
                bottom_color=(232, 56, 24),
                shadow_offset=4,
            )
            self.screen.blit(title_surf, title_surf.get_rect(center=(win_w // 2, int(win_h * 0.22))))

            sub_title = render_pixel_text("SPEED POSE MATCH TOURNAMENT (PRECISION & SPEED RUSH)", scale=2, top_color=(200, 210, 230), bottom_color=(120, 140, 160), shadow_offset=2)
            self.screen.blit(sub_title, sub_title.get_rect(center=(win_w // 2, int(win_h * 0.30))))

            # Network / Standalone Status Panel
            panel_w, panel_h = min(680, win_w - 60), 160
            p_x = (win_w - panel_w) // 2
            p_y = int(win_h * 0.36)

            p_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            p_surf.fill((16, 22, 34, 235))
            self.screen.blit(p_surf, (p_x, p_y))
            border_color = C_NEON_GREEN if is_connected else (255, 190, 0)
            self._draw_cyber_frame(self.screen, (p_x, p_y, panel_w, panel_h), border_color)

            if is_connected:
                badge = render_pixel_text("[ ONLINE — REGISTRATION DESK CONNECTED ]", scale=2, top_color=(0, 255, 180), bottom_color=(0, 180, 100), shadow_offset=2)
                self.screen.blit(badge, badge.get_rect(center=(win_w // 2, p_y + 35)))

                pulse_alpha = int(180 + 75 * math.sin(self.pulse_phase))
                main_prompt = render_pixel_text("WAITING FOR PLAYER DISPATCH...", scale=3, top_color=(255, 220, 50), bottom_color=(255, 140, 0), shadow_offset=3)
                main_prompt.set_alpha(pulse_alpha)
                self.screen.blit(main_prompt, main_prompt.get_rect(center=(win_w // 2, p_y + 80)))

                sub_prompt = render_pixel_text("Register player at Laptop 3 to launch station!", scale=2, top_color=(180, 200, 220), bottom_color=(100, 120, 140), shadow_offset=1)
                self.screen.blit(sub_prompt, sub_prompt.get_rect(center=(win_w // 2, p_y + 125)))
            else:
                badge = render_pixel_text("[ STANDALONE ARCADE MODE ]", scale=2, top_color=(255, 200, 0), bottom_color=(200, 120, 0), shadow_offset=2)
                self.screen.blit(badge, badge.get_rect(center=(win_w // 2, p_y + 35)))

                pulse_alpha = int(180 + 75 * math.sin(self.pulse_phase))
                main_prompt = render_pixel_text("PRESS [SPACE] TO START", scale=3, top_color=(0, 255, 180), bottom_color=(0, 180, 100), shadow_offset=3)
                main_prompt.set_alpha(pulse_alpha)
                self.screen.blit(main_prompt, main_prompt.get_rect(center=(win_w // 2, p_y + 80)))

                sub_prompt = render_pixel_text("Step into camera view and press SPACE to begin", scale=2, top_color=(180, 200, 220), bottom_color=(100, 120, 140), shadow_offset=1)
                self.screen.blit(sub_prompt, sub_prompt.get_rect(center=(win_w // 2, p_y + 125)))

            # Recent Run Banner
            if last_player_score is not None:
                rec_w, rec_h = min(560, win_w - 80), 45
                rx = (win_w - rec_w) // 2
                ry = int(win_h * 0.60)
                rec_bg = pygame.Surface((rec_w, rec_h), pygame.SRCALPHA)
                rec_bg.fill((24, 28, 40, 210))
                self.screen.blit(rec_bg, (rx, ry))
                self._draw_cyber_frame(self.screen, (rx, ry, rec_w, rec_h), (0, 200, 255))
                name_disp = last_player_name or "PLAYER"
                rec_txt = render_pixel_text(f"RECENT RUN: {name_disp.upper()} — {last_player_score:04d} PTS", scale=2, top_color=(0, 240, 255), bottom_color=(0, 160, 240), shadow_offset=2)
                self.screen.blit(rec_txt, rec_txt.get_rect(center=(win_w // 2, ry + rec_h // 2)))

            # Gesture Pro-Tip
            tip_y = int(win_h * 0.72)
            tip_surf = render_pixel_text("PRO-TIP: RIGHT HAND ON HEAD (0.5s) TO SKIP TO NEXT POSE", scale=2, top_color=(0, 255, 128), bottom_color=(0, 160, 70), shadow_offset=2)
            self.screen.blit(tip_surf, tip_surf.get_rect(center=(win_w // 2, tip_y)))

            # Controls footer
            foot_str = "[SPACE] START   |   [R] RESET   |   [F11] FULLSCREEN   |   [ESC] EXIT"
            foot_surf = render_pixel_text(foot_str, scale=2, top_color=(150, 165, 185), bottom_color=(90, 105, 125), shadow_offset=1)
            self.screen.blit(foot_surf, foot_surf.get_rect(center=(win_w // 2, win_h - 35)))

        pygame.display.flip()


# Backward compatibility alias
WallArenaRenderer = BlazerArenaRenderer
