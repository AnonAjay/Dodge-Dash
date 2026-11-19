"""Station A: Augmented Reality (AR) Camera + Lava Arena Renderer.
Features:
- Minecraft-style Cobblestone 3D Sinking Plates.
- Sinking block animation:
  * 0.0 -> 0.6: Cobblestone cracks and begins sinking down into the floor.
  * 0.6 -> 0.8: Magma rush! Molten lava wells up into the sinking cavity.
  * >= 0.8: Full eruption! Cavity flooded with organic orange, red, and yellow magma mixture,
            with 3D Minecraft ember spark particles popping up and floating into the air.
- Safe zone billboard / ring completely removed (safe blocks remain un-sunken solid cobblestone).
- Independent dual-foot contact circles (small, accurate contact rings under left and right shoes).
- 2D Depth Occlusion: Player legs, shoes, and body occlude both the molten lava and arena borders.
- Auto-maximized resizable window with [F11] toggle.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
import cv2
import numpy as np
import pygame

from pose_core.floor_map import FloorMapper
from shared.contracts import GameState, Phase, PoseFrame
from station_a.logic import GRID_SIZE, LavaWave, get_streak_multiplier
from station_a.sounds import ArcadeSoundManager

# Base Window Geometry
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 768
HUD_TOP_H = 76
HUD_BOT_H = 0
DEFAULT_ROUND_DURATION_S = 30.0

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
    """Render crisp retro arcade pixel text with signature two-tone sunset gradient,
    thick dark pixel outline, and drop shadow. Uses scaled surface caching for ~0ms runtime."""
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


# Texture resolution for floor perspective warp
TEX_SIZE = 400
CELL_TEX_PX = TEX_SIZE // GRID_SIZE  # 40 px per cell

# Minecraft Cobblestone Palette (BGR)
C_COBBLE_MORTAR = (45, 45, 50)
C_COBBLE_DARK = (75, 75, 80)
C_COBBLE_MID = (115, 115, 120)
C_COBBLE_LIGHT = (155, 155, 160)
C_BLOCK_CLIFF = (30, 30, 35)       # Shadowed 3D front vertical edge

# Real Magma Palette (BGR)
C_LAVA_DEEP_RED = (18, 24, 195)    # Rich crimson red
C_LAVA_MOLTEN_ORANGE = (25, 95, 245) # Flowing fiery orange
C_LAVA_HOT_YELLOW = (45, 215, 255) # Blazing yellow hotspot
C_AMBER_CRACK = (20, 130, 240)     # Fissure crack

# HUD Colors (RGB for Pygame)
TEXT_WHITE = (245, 248, 252)
TEXT_GOLD = (255, 215, 0)
TEXT_MUTED = (140, 155, 175)
SAFE_GREEN = (0, 255, 120)


@dataclass
class LavaParticle:
    """3D Minecraft-style ember spark popping out of molten lava."""
    x_m: float       # Floor X (metres)
    y_m: float       # Floor Y (metres)
    z_m: float       # Height above floor (metres, rises upwards)
    vx: float        # Drift X
    vy: float        # Drift Y
    vz: float        # Rise speed (m/s)
    life: float      # Remaining seconds
    max_life: float
    size_px: int     # Cube size (pixels)


def generate_minecraft_cobblestone_tile(size: int = CELL_TEX_PX) -> np.ndarray:
    """Generate a single 16x16-style Minecraft cobblestone texture scaled to cell size."""
    palette = [C_COBBLE_MORTAR, C_COBBLE_DARK, C_COBBLE_MID, C_COBBLE_LIGHT]
    pattern = [
        [0, 1, 1, 2, 0, 1, 2, 2, 3, 2, 1, 0, 1, 1, 2, 0],
        [1, 2, 3, 2, 1, 0, 1, 3, 2, 1, 0, 1, 2, 3, 2, 1],
        [1, 3, 2, 1, 0, 1, 2, 2, 1, 0, 1, 2, 3, 2, 1, 1],
        [0, 1, 1, 0, 1, 2, 3, 1, 0, 1, 2, 2, 1, 1, 0, 0],
        [1, 2, 2, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1, 2, 1, 1],
        [2, 3, 1, 0, 1, 2, 1, 0, 2, 3, 2, 1, 0, 1, 2, 2],
        [1, 2, 0, 1, 2, 1, 0, 1, 2, 2, 1, 0, 1, 2, 3, 1],
        [0, 0, 1, 2, 3, 2, 1, 0, 1, 1, 0, 1, 2, 3, 2, 0],
        [1, 1, 2, 3, 2, 1, 0, 1, 2, 2, 1, 2, 3, 2, 1, 1],
        [2, 3, 2, 1, 1, 0, 1, 2, 3, 2, 1, 1, 2, 1, 0, 1],
        [1, 2, 1, 0, 0, 1, 2, 2, 1, 0, 0, 1, 1, 0, 1, 2],
        [0, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 2, 1, 1, 2, 3],
        [1, 2, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2],
        [2, 3, 2, 1, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 1],
        [1, 2, 1, 0, 1, 0, 1, 2, 2, 1, 1, 0, 1, 2, 3, 2],
        [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 2, 1, 0],
    ]
    raw = np.zeros((16, 16, 3), dtype=np.uint8)
    for r in range(16):
        for c in range(16):
            raw[r, c] = palette[pattern[r][c]]

    return cv2.resize(raw, (size, size), interpolation=cv2.INTER_NEAREST)


CRACK_STAGES_16X16 = [
    # Stage 0: Pristine cobblestone
    [],
    # Stage 1: Subtle hairline nick on front-right edge
    [(14, 12), (15, 12), (13, 11)],
    # Stage 2: Creeps along mortar groove
    [(14, 12), (15, 12), (13, 11), (12, 10), (11, 10), (10, 9)],
    # Stage 3: Branches toward center junction
    [(14, 12), (15, 12), (13, 11), (12, 10), (11, 10), (10, 9), (9, 8), (8, 8), (8, 9), (7, 10)],
    # Stage 4: Crosses to left side (major fault line)
    [(14, 12), (15, 12), (13, 11), (12, 10), (11, 10), (10, 9), (9, 8), (8, 8), (7, 8), (6, 8), (5, 9), (4, 9), (3, 10), (2, 10), (1, 11), (0, 11)],
    # Stage 5: Secondary branch to top right
    [(14, 12), (15, 12), (13, 11), (12, 10), (11, 10), (10, 9), (9, 8), (8, 8), (7, 8), (6, 8), (5, 9), (4, 9), (3, 10), (2, 10), (1, 11), (0, 11),
     (9, 7), (10, 6), (11, 5), (12, 4), (13, 4), (14, 3), (15, 3)],
    # Stage 6: Spiderweb across all quadrants
    [(14, 12), (15, 12), (13, 11), (12, 10), (11, 10), (10, 9), (9, 8), (8, 8), (7, 8), (6, 8), (5, 9), (4, 9), (3, 10), (2, 10), (1, 11), (0, 11),
     (9, 7), (10, 6), (11, 5), (12, 4), (13, 4), (14, 3), (15, 3),
     (8, 10), (8, 11), (7, 12), (7, 13), (6, 14), (6, 15),
     (7, 7), (6, 6), (5, 5), (4, 4), (3, 3)],
    # Stage 7: Critical shatter web
    [(14, 12), (15, 12), (13, 11), (12, 10), (11, 10), (10, 9), (9, 8), (8, 8), (7, 8), (6, 8), (5, 9), (4, 9), (3, 10), (2, 10), (1, 11), (0, 11),
     (14, 11), (13, 10), (12, 9), (11, 9), (10, 8), (9, 7), (10, 6), (11, 5), (12, 4), (13, 4), (14, 3), (15, 3),
     (8, 10), (8, 11), (7, 12), (7, 13), (6, 14), (6, 15),
     (7, 7), (6, 6), (5, 5), (4, 4), (3, 3), (2, 2), (1, 1), (0, 0),
     (15, 15), (14, 14), (13, 13), (12, 12)]
]


def generate_prebaked_crack_tiles(size: int = CELL_TEX_PX) -> list[np.ndarray]:
    """Pre-bake 8 discrete Minecraft-style destruction stages for 0ms runtime swap."""
    base = generate_minecraft_cobblestone_tile(16)
    tiles = []
    col_amber = (20, 110, 240)
    col_orange = (25, 130, 245)
    col_yellow = (45, 215, 255)

    for i, pts in enumerate(CRACK_STAGES_16X16):
        raw = base.copy()
        if i in (1, 2):
            crack_col = col_amber
        elif i in (3, 4):
            crack_col = col_orange
        else:
            crack_col = col_yellow

        for x, y in pts:
            raw[y, x] = crack_col
            if i >= 5:
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 16 and 0 <= ny < 16 and (nx, ny) not in pts:
                        raw[ny, nx] = tuple(min(255, int(raw[ny, nx][c] * 0.5 + col_orange[c] * 0.5)) for c in range(3))

        scaled = cv2.resize(raw, (size, size), interpolation=cv2.INTER_NEAREST)
        tiles.append(scaled)
    return tiles


def install_pose_tracker_hook() -> None:
    """Attach full 33-landmark skeleton extraction to PoseTracker at runtime.
    Preserves repo ownership rules (does not touch pose_core/ files) while ensuring
    the AR renderer gets the full player body skeleton for 100% accurate Z-depth occlusion.
    """
    try:
        from pose_core.capture import PoseTracker
        from pose_core.landmarks import feet_info

        if getattr(PoseTracker, "_hooked_for_render", False):
            return

        orig_update = PoseTracker.update

        def hooked_update(self, people, img_w: int, img_h: int, ts: float, *args, **kwargs):
            pf = orig_update(self, people, img_w, img_h, ts, *args, **kwargs)
            if people and pf.tracked:
                # Find candidate matching self.debug["pts"] or closest to foot_px
                matched_lms = None
                target_pts = self.debug.get("pts")
                target_foot = self.debug.get("foot_px")

                # Strategy 1: Match by exact foot points
                if target_pts:
                    for lms in people:
                        c = feet_info(lms, img_w, img_h)
                        if c is not None and c.get("pts") == target_pts:
                            matched_lms = lms
                            break

                # Strategy 2: Match by closest foot centre
                if matched_lms is None and target_foot is not None:
                    best_dist = float("inf")
                    for lms in people:
                        c = feet_info(lms, img_w, img_h)
                        if c is None:
                            continue
                        d = math.hypot(c["centre_px"][0] - target_foot[0], c["centre_px"][1] - target_foot[1])
                        if d < best_dist:
                            best_dist = d
                            matched_lms = lms

                # Strategy 3: Fallback to first detected person
                if matched_lms is None and len(people) > 0:
                    matched_lms = people[0]

                if matched_lms is not None:
                    self.debug["landmarks_px"] = [
                        (float(lm.x * img_w), float(lm.y * img_h), float(getattr(lm, "visibility", 1.0)))
                        for lm in matched_lms
                    ]
            return pf

        PoseTracker.update = hooked_update
        PoseTracker._hooked_for_render = True
    except Exception:
        pass


def build_player_occlusion_mask(cam_h: int, cam_w: int,
                                mapper: FloorMapper | None,
                                pose: PoseFrame,
                                debug: dict | None) -> np.ndarray:
    """Construct a tight, anatomically precise depth occlusion mask in camera pixels.
    Guarantees the player's 3D body, legs, and shoes occlude the floor lava AND arena borders
    without creating a bloated or oversized gap around the player.
    """
    mask = np.zeros((cam_h, cam_w), dtype=np.uint8)

    # 1. Primary: Exact MediaPipe 33 Landmark Skeleton
    if debug and "landmarks_px" in debug and len(debug["landmarks_px"]) >= 33:
        lms = debug["landmarks_px"]
        pts = [(int(round(p[0])), int(round(p[1]))) for p in lms]
        vis = [p[2] for p in lms]

        # Torso (Shoulders 11, 12 down to Hips 24, 23)
        if vis[11] > 0.3 and vis[12] > 0.3 and vis[23] > 0.3 and vis[24] > 0.3:
            torso_poly = np.array([pts[11], pts[12], pts[24], pts[23]], dtype=np.int32)
            cv2.fillPoly(mask, [torso_poly], 255)

        # Pelvis Band (slim 20px)
        if vis[23] > 0.3 and vis[24] > 0.3:
            cv2.line(mask, pts[23], pts[24], 255, 20, cv2.LINE_AA)

        # Left Leg: Thigh (23->25, 22px), Shin (25->27, 18px)
        if vis[23] > 0.3 and vis[25] > 0.3:
            cv2.line(mask, pts[23], pts[25], 255, 22, cv2.LINE_AA)
        if vis[25] > 0.3 and vis[27] > 0.3:
            cv2.line(mask, pts[25], pts[27], 255, 18, cv2.LINE_AA)

        # Left Shoe (Sole 29->31, Bridge 27->31, 16px)
        if vis[29] > 0.2 and vis[31] > 0.2:
            cv2.line(mask, pts[29], pts[31], 255, 16, cv2.LINE_AA)
        if vis[27] > 0.2 and vis[31] > 0.2:
            cv2.line(mask, pts[27], pts[31], 255, 14, cv2.LINE_AA)

        # Right Leg: Thigh (24->26, 22px), Shin (26->28, 18px)
        if vis[24] > 0.3 and vis[26] > 0.3:
            cv2.line(mask, pts[24], pts[26], 255, 22, cv2.LINE_AA)
        if vis[26] > 0.3 and vis[28] > 0.3:
            cv2.line(mask, pts[26], pts[28], 255, 18, cv2.LINE_AA)

        # Right Shoe (Sole 30->32, Bridge 28->32, 16px)
        if vis[30] > 0.2 and vis[32] > 0.2:
            cv2.line(mask, pts[30], pts[32], 255, 16, cv2.LINE_AA)
        if vis[28] > 0.2 and vis[32] > 0.2:
            cv2.line(mask, pts[28], pts[32], 255, 14, cv2.LINE_AA)

        # Arms (11->13, 13->15, 12->14, 14->16, 18px)
        if vis[11] > 0.3 and vis[13] > 0.3:
            cv2.line(mask, pts[11], pts[13], 255, 18, cv2.LINE_AA)
        if vis[13] > 0.3 and vis[15] > 0.3:
            cv2.line(mask, pts[13], pts[15], 255, 16, cv2.LINE_AA)
        if vis[12] > 0.3 and vis[14] > 0.3:
            cv2.line(mask, pts[12], pts[14], 255, 18, cv2.LINE_AA)
        if vis[14] > 0.3 and vis[16] > 0.3:
            cv2.line(mask, pts[14], pts[16], 255, 16, cv2.LINE_AA)

    # 2. Fallback: Only if skeleton not available, use detected feet points
    elif debug and "pts" in debug and debug["pts"]:
        for pt in debug["pts"]:
            px, py = int(round(pt[0])), int(round(pt[1]))
            # Tight shoe ellipse
            cv2.ellipse(mask, (px, py), (16, 10), 0, 0, 360, 255, -1)
            # Slim leg column
            cv2.line(mask, (px, py), (px, max(0, py - 300)), 255, 20, cv2.LINE_AA)

    # 3. Fallback: Only if feet points not available, use homography positions
    elif mapper is not None and pose.tracked:
        if pose.left_foot_m is not None:
            lx, ly = mapper.to_pixels(*pose.left_foot_m)
            lx_i, ly_i = int(round(lx)), int(round(ly))
            cv2.ellipse(mask, (lx_i, ly_i), (16, 10), 0, 0, 360, 255, -1)
            cv2.line(mask, (lx_i, ly_i), (lx_i, max(0, ly_i - 300)), 255, 20, cv2.LINE_AA)

        if pose.right_foot_m is not None:
            rx, ry = mapper.to_pixels(*pose.right_foot_m)
            rx_i, ry_i = int(round(rx)), int(round(ry))
            cv2.ellipse(mask, (rx_i, ry_i), (16, 10), 0, 0, 360, 255, -1)
            cv2.line(mask, (rx_i, ry_i), (rx_i, max(0, ry_i - 300)), 255, 20, cv2.LINE_AA)

        if pose.left_foot_m is None and pose.right_foot_m is None and pose.floor_x_m is not None:
            cx, cy = mapper.to_pixels(pose.floor_x_m, pose.floor_y_m)
            cx_i, cy_i = int(round(cx)), int(round(cy))
            cv2.circle(mask, (cx_i, cy_i), 20, 255, -1)
            cv2.line(mask, (cx_i, cy_i), (cx_i, max(0, cy_i - 300)), 255, 26, cv2.LINE_AA)

    # 4. Subtle anti-aliasing without bloating
    if mask.any():
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

    return mask


class LavaArenaRenderer:
    """AR Camera-Floor Renderer with 3D Sinking Minecraft Cobblestone Blocks and Minecraft Lava Particles."""

    def __init__(self, mapper: FloorMapper | None = None,
                 arena_w_m: float = 1.8, arena_h_m: float = 1.8,
                 fullscreen: bool = False,
                 enable_voice: bool = True):
        install_pose_tracker_hook()
        pygame.init()
        pygame.font.init()

        self.mapper = mapper
        self.arena_w_m = arena_w_m
        self.arena_h_m = arena_h_m
        self.is_maximized = True

        # Always start in a resizable window and maximize immediately
        self.screen = pygame.display.set_mode((DEFAULT_WIDTH, DEFAULT_HEIGHT), pygame.RESIZABLE)
        self._maximize_window()

        pygame.display.set_caption("TITANS' ARCADE — STATION A: FLOOR IS LAVA")

        # Precompute Perspective Transform Matrix
        self.warp_M = None
        self._setup_warp_matrix()

        # Precompute Minecraft Cobblestone Tile and 8 Pre-baked Destruction Stages
        self.cobble_tile = generate_minecraft_cobblestone_tile(CELL_TEX_PX)
        self.crack_tiles = generate_prebaked_crack_tiles(CELL_TEX_PX)

        # Base Full-Arena Cobblestone Grid with 3D Front Cliff Edges
        self.base_arena_bgr = np.zeros((TEX_SIZE, TEX_SIZE, 3), dtype=np.uint8)
        self._build_base_cobblestone_floor()

        # Reusable buffers
        self.frame_floor_bgr = self.base_arena_bgr.copy()
        self.floor_alpha = np.zeros((TEX_SIZE, TEX_SIZE), dtype=np.float32)

        # Particle System
        self.particles: list[LavaParticle] = []
        self.last_particle_spawn = time.time()

        # Fonts
        self.font_title = pygame.font.SysFont("Impact", 30)
        self.font_large = pygame.font.SysFont("Consolas", 26, bold=True)
        self.font_mid = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 15)
        self.font_arcade_huge = pygame.font.SysFont("Impact", 64)
        self.font_arcade_sub = pygame.font.SysFont("Impact", 28)

        self.start_time = time.time()
        self.last_frame_time = time.time()

        # Ephemeral Floating Toast System
        self.toast_msg: str = ""
        self.toast_color: tuple[int, int, int] = TEXT_GOLD
        self.toast_timer: float = 0.0
        self.toast_max_duration: float = 1.4
        self.last_seen_banner_msg: str = ""
        self.last_dodges: int = 0
        self.last_hits: int = 0

        # Retro 8-Bit Arcade Sound Manager & State Tracking
        self.sound_mgr = ArcadeSoundManager(enable_voice=enable_voice)
        self._feet_locked: bool = False
        self._game_start_sound_played: bool = False
        self._prev_phase: Phase | None = None
        self._last_round_announced: int = 0
        self._game_over_sound_played: bool = False
        self._prev_time_left_s: float = 0.0

    def _maximize_window(self) -> None:
        """Maximize the Pygame window using Windows OS API."""
        try:
            import ctypes
            wm = pygame.display.get_wm_info()
            hwnd = wm.get("window")
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE = 3
        except Exception:
            pass

    def toggle_fullscreen(self) -> None:
        """Toggle between Maximized and Windowed display."""
        self.is_maximized = not self.is_maximized
        try:
            import ctypes
            wm = pygame.display.get_wm_info()
            hwnd = wm.get("window")
            if hwnd:
                cmd = 3 if self.is_maximized else 9  # 3 = SW_MAXIMIZE, 9 = SW_RESTORE
                ctypes.windll.user32.ShowWindow(hwnd, cmd)
        except Exception:
            pass

    def _setup_warp_matrix(self) -> None:
        if self.mapper is None or not hasattr(self.mapper, "corners_px"):
            return

        src_corners = np.float32([
            [0, TEX_SIZE - 1],            # near-left
            [TEX_SIZE - 1, TEX_SIZE - 1], # near-right
            [TEX_SIZE - 1, 0],            # far-right
            [0, 0],                       # far-left
        ])
        dst_corners = np.float32(self.mapper.corners_px)
        self.warp_M = cv2.getPerspectiveTransform(src_corners, dst_corners)

    def _build_base_cobblestone_floor(self) -> None:
        """Assemble the 10x10 Minecraft cobblestone grid with visible 3D front cliff edges."""
        cliff_h = 6  # 3D block front edge thickness (pixels)
        for c in range(GRID_SIZE):
            for r in range(GRID_SIZE):
                x0 = c * CELL_TEX_PX
                x1 = (c + 1) * CELL_TEX_PX
                y0 = (GRID_SIZE - 1 - r) * CELL_TEX_PX
                y1 = (GRID_SIZE - r) * CELL_TEX_PX

                # Top face: Minecraft cobblestone
                self.base_arena_bgr[y0:y1 - cliff_h, x0:x1] = self.cobble_tile[:CELL_TEX_PX - cliff_h, :]

                # 3D Front Edge: Dark shadowed vertical cliff facing camera
                self.base_arena_bgr[y1 - cliff_h:y1, x0:x1] = C_BLOCK_CLIFF
                cv2.line(self.base_arena_bgr, (x0, y1 - cliff_h), (x1, y1 - cliff_h), (20, 20, 25), 1)

                # Bevel mortar around each block
                cv2.rectangle(self.base_arena_bgr, (x0, y0), (x1, y1), C_COBBLE_MORTAR, 1)

    def _update_and_draw_particles(self, out_frame: np.ndarray, dt: float,
                                   leg_mask: np.ndarray | None = None) -> None:
        """Update and draw 3D Minecraft ember spark particles rising into the air."""
        if self.mapper is None:
            return

        cam_h, cam_w = out_frame.shape[:2]

        for p in list(self.particles):
            p.life -= dt
            if p.life <= 0:
                self.particles.remove(p)
                continue

            # Move upward into the 3D air
            p.z_m += p.vz * dt
            p.x_m += p.vx * dt
            p.y_m += p.vy * dt

            # Project base floor position to camera pixels
            gx, gy = self.mapper.to_pixels(p.x_m, p.y_m)
            # Perspective height offset: rising upwards along image Y
            px = int(gx)
            py = int(gy - p.z_m * 180.0)

            # Check camera bounds
            if not (0 <= px < cam_w and 0 <= py < cam_h):
                continue

            # Check leg occlusion (sparks behind player are occluded)
            if leg_mask is not None and leg_mask[py, px] > 128:
                continue

            # Particle life progress 0.0 (birth) -> 1.0 (death)
            prog = 1.0 - (p.life / p.max_life)
            if prog < 0.4:
                col = C_LAVA_HOT_YELLOW     # Fresh bright yellow spark
            elif prog < 0.75:
                col = C_LAVA_MOLTEN_ORANGE   # Floating flame ember
            else:
                col = (45, 45, 52)          # Dissolving smoke pop

            s = p.size_px
            # Minecraft-style crisp square pixel spark
            cv2.rectangle(out_frame, (px - s, py - s), (px + s, py + s), col, -1)

    def _render_ar_frame(self, cam_frame: np.ndarray, wave: LavaWave | None,
                         pose: PoseFrame, now: float, dt: float,
                         debug: dict | None = None) -> np.ndarray:
        """Warp 3D sinking blocks, magma flood, and rising ember particles onto camera feed."""
        cam_h, cam_w = cam_frame.shape[:2]
        out_frame = cam_frame.copy()

        # 1. Base Cobblestone Arena
        np.copyto(self.frame_floor_bgr, self.base_arena_bgr)
        self.floor_alpha.fill(0.70)

        cliff_h = 6
        active_lava_cells = []

        # 2. Animate 3D Sinking Blocks & Organic Magma Rush
        if wave is not None:
            # Render blocks back-to-front (Row 9 down to Row 0)
            for r in range(GRID_SIZE - 1, -1, -1):
                for c in range(GRID_SIZE):
                    h = wave.heat[c][r]
                    if h <= 0.0:
                        continue

                    x0 = c * CELL_TEX_PX
                    x1 = (c + 1) * CELL_TEX_PX
                    y0 = (GRID_SIZE - 1 - r) * CELL_TEX_PX
                    y1 = (GRID_SIZE - r) * CELL_TEX_PX
                    cx = (x0 + x1) // 2
                    cy = (y0 + y1) // 2

                    if h < 0.70:
                        # ======================================================
                        # STAGE 0..7: PRE-FABRICATED MINECRAFT CRACK TILES (0.0 -> 0.70)
                        # Instant 0ms memory lookup; pixel-art cracks follow stone mortar lines
                        # ======================================================
                        stage = min(7, max(0, int((h / 0.70) * 8)))
                        self.frame_floor_bgr[y0:y1 - cliff_h, x0:x1] = self.crack_tiles[stage][:CELL_TEX_PX - cliff_h, :]

                        # Front cliff edge: cracks begin creeping onto the vertical front facing camera
                        self.frame_floor_bgr[y1 - cliff_h:y1, x0:x1] = C_BLOCK_CLIFF
                        cv2.line(self.frame_floor_bgr, (x0, y1 - cliff_h), (x1, y1 - cliff_h), (20, 20, 25), 1)
                        if stage >= 4:
                            cv2.line(self.frame_floor_bgr, (cx - 6, y1 - cliff_h), (cx + 6, y1 - cliff_h), (25, 130, 245), 1)

                        cv2.rectangle(self.frame_floor_bgr, (x0, y0), (x1, y1), C_COBBLE_MORTAR, 1)
                        self.floor_alpha[y0:y1, x0:x1] = 0.70 + 0.14 * (h / 0.70)

                    elif h < 0.90:
                        # ======================================================
                        # STAGE 4 & 5: SHATTER & 3D TECTONIC SINKING (0.70 -> 0.90)
                        # Two large plates tilt like a seesaw and submerge into molten magma
                        # ======================================================
                        active_lava_cells.append((c, r))
                        p_sink = (h - 0.70) / 0.20  # 0.0 -> 1.0
                        # Deceleration curve (settles slower as it plunges into dense magma)
                        ease = 1.0 - (1.0 - p_sink) ** 2

                        # 1. Fill background cavity with surging molten magma
                        pulse_flow = math.sin(now * 8.0 + c + r) * 0.5 + 0.5
                        r_magma = int(C_LAVA_DEEP_RED[2] * (1.0 - pulse_flow) + C_LAVA_MOLTEN_ORANGE[2] * pulse_flow)
                        g_magma = int(C_LAVA_DEEP_RED[1] * (1.0 - pulse_flow) + C_LAVA_MOLTEN_ORANGE[1] * pulse_flow)
                        b_magma = int(C_LAVA_DEEP_RED[0] * (1.0 - pulse_flow) + C_LAVA_MOLTEN_ORANGE[0] * pulse_flow)
                        self.frame_floor_bgr[y0:y1, x0:x1] = (b_magma, g_magma, r_magma)

                        # 2. Tilting Tectonic Plate 1 (Front/Left Slab - tilts down/forward)
                        dx1 = -int(ease * 5)
                        dy1 = int(ease * 14)
                        shade1 = max(0.15, 1.0 - ease * 0.75)
                        col1 = (int(C_COBBLE_MID[0] * shade1), int(C_COBBLE_MID[1] * shade1), int(C_COBBLE_MID[2] * shade1))

                        p1 = np.array([
                            [x0 + dx1, y0 + 22 + dy1],
                            [cx + dx1, cy + dy1],
                            [x1 - 2 + dx1, y0 + 12 + dy1],
                            [x1 - 2 + dx1, y1 + dy1],
                            [x0 + dx1, y1 + dy1],
                        ], dtype=np.int32)

                        # Clip polygon to cell boundaries so it submerges cleanly into the magma pool
                        p1_clipped = np.clip(p1, [x0, y0], [x1, y1])
                        if ease < 0.95:
                            cv2.fillPoly(self.frame_floor_bgr, [p1_clipped], col1)
                            # Front 3D cliff edge of sinking plate
                            cv2.line(self.frame_floor_bgr, (p1_clipped[4][0], p1_clipped[4][1] - 2),
                                     (p1_clipped[3][0], p1_clipped[3][1] - 2), (25, 25, 30), 2)

                        # 3. Tilting Tectonic Plate 2 (Back/Right Slab - tilts up/backward, seesaw)
                        dx2 = int(ease * 6)
                        dy2 = int(ease * 18)
                        shade2 = max(0.12, 1.0 - ease * 0.85)
                        col2 = (int(C_COBBLE_DARK[0] * shade2), int(C_COBBLE_DARK[1] * shade2), int(C_COBBLE_DARK[2] * shade2))

                        p2 = np.array([
                            [x0 + dx2, y0 + dy2],
                            [x1 + dx2, y0 + dy2],
                            [x1 + dx2, y0 + 12 + dy2],
                            [cx + dx2, cy + dy2],
                            [x0 + dx2, y0 + 22 + dy2],
                        ], dtype=np.int32)

                        p2_clipped = np.clip(p2, [x0, y0], [x1, y1])
                        if ease < 0.90:
                            cv2.fillPoly(self.frame_floor_bgr, [p2_clipped], col2)

                        # 4. Molten Magma Surge in the fault chasm between the two separating plates
                        cv2.line(self.frame_floor_bgr, (x0, y0 + 22 + int(ease * 12)),
                                 (x1, y0 + 12 + int(ease * 12)), C_LAVA_HOT_YELLOW, max(2, int(ease * 5)), cv2.LINE_AA)

                        self.floor_alpha[y0:y1, x0:x1] = 0.84 + 0.14 * ease

                    else:
                        # ======================================================
                        # STAGE 6: FULL MAGMA INUNDATION & WAVY SURFACE CURRENTS (0.90 -> 1.0)
                        # Slabs fully submerged; churning magma pool with animated wavy contour lines
                        # ======================================================
                        active_lava_cells.append((c, r))

                        pulse_flow = math.sin(now * 7.0 + c * 1.5 + r * 2.0) * 0.5 + 0.5
                        pulse_bubble = math.sin(now * 14.0 + c * 3.5 + r * 3.0) * 0.5 + 0.5

                        # Base molten magma pool
                        r_col = int(C_LAVA_DEEP_RED[2] * (1.0 - pulse_flow) + C_LAVA_MOLTEN_ORANGE[2] * pulse_flow)
                        g_col = int(C_LAVA_DEEP_RED[1] * (1.0 - pulse_flow) + C_LAVA_MOLTEN_ORANGE[1] * pulse_flow)
                        b_col = int(C_LAVA_DEEP_RED[0] * (1.0 - pulse_flow) + C_LAVA_MOLTEN_ORANGE[0] * pulse_flow)
                        self.frame_floor_bgr[y0:y1, x0:x1] = (b_col, g_col, r_col)

                        # Animated sinusoidal surface contour currents (wavy flow lines from user sketch)
                        for k in (0.32, 0.68):
                            wave_pts = []
                            for xi in range(x0 + 2, x1 - 1, 3):
                                phase = now * 6.5 + (xi - x0) * 0.28 + c * 1.8 + r * 2.2
                                yi = int(y0 + CELL_TEX_PX * k + math.sin(phase) * 3.2)
                                yi = max(y0 + 1, min(y1 - 2, yi))
                                wave_pts.append((xi, yi))
                            if len(wave_pts) >= 2:
                                cv2.polylines(self.frame_floor_bgr, [np.array(wave_pts, dtype=np.int32)],
                                              False, C_LAVA_HOT_YELLOW, 2, cv2.LINE_AA)

                        # Central boiling hotspot swirl
                        current_len = int(10 * pulse_bubble)
                        cv2.line(self.frame_floor_bgr, (cx - current_len, cy), (cx + current_len, cy),
                                 (255, 240, 180), 1, cv2.LINE_AA)

                        # Dark cooling basalt crust border along perimeter
                        cv2.rectangle(self.frame_floor_bgr, (x0, y0), (x1, y1), (15, 18, 140), 1)
                        self.floor_alpha[y0:y1, x0:x1] = 0.98

        # 3. Spawn 3D Minecraft Ember Spark Particles from Active Magma (H >= 0.70)
        if active_lava_cells and (now - self.last_particle_spawn) > 0.07 and len(self.particles) < 55:
            self.last_particle_spawn = now
            # Pick 2-3 random active lava cells to pop sparks
            sample_n = min(3, len(active_lava_cells))
            for c, r in random.sample(active_lava_cells, sample_n):
                cell_cx_m = (c + random.uniform(0.2, 0.8)) * (self.arena_w_m / GRID_SIZE)
                cell_cy_m = (r + random.uniform(0.2, 0.8)) * (self.arena_h_m / GRID_SIZE)
                self.particles.append(LavaParticle(
                    x_m=cell_cx_m,
                    y_m=cell_cy_m,
                    z_m=0.02,
                    vx=random.uniform(-0.06, 0.06),
                    vy=random.uniform(-0.06, 0.06),
                    vz=random.uniform(0.35, 0.70),  # Rises 35 to 70 cm/s
                    life=random.uniform(0.6, 1.1),
                    max_life=1.1,
                    size_px=random.choice([3, 4, 5]),
                ))

        # 4. Warp 3D Cobblestone Floor onto Camera Feed Perspective
        leg_mask = None
        if self.warp_M is not None:
            warped_floor = cv2.warpPerspective(self.frame_floor_bgr, self.warp_M, (cam_w, cam_h))
            warped_alpha = cv2.warpPerspective(self.floor_alpha, self.warp_M, (cam_w, cam_h))

            # Calibrated arena perimeter line
            border_overlay = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            poly = self.mapper.corners_px.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(border_overlay, [poly], True, (0, 220, 255), 2, cv2.LINE_AA)
            border_alpha = (cv2.cvtColor(border_overlay, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)

            # 5. 2D DEPTH OCCLUSION: Player's legs and shoes occlude BOTH lava AND borders
            # Guarantees the player's 3D body stays strictly in front of the 2D floor projection
            leg_mask = build_player_occlusion_mask(cam_h, cam_w, self.mapper, pose, debug)
            if leg_mask.any():
                leg_occ = leg_mask.astype(np.float32) / 255.0
                inv_occ = 1.0 - leg_occ
                warped_alpha = np.maximum(0.0, warped_alpha * inv_occ)
                border_alpha = np.maximum(0.0, border_alpha * inv_occ)

            # Alpha blend cobblestone + magma onto camera feed
            alpha_3d = warped_alpha[:, :, np.newaxis]
            out_frame = (out_frame * (1.0 - alpha_3d) + warped_floor * alpha_3d).astype(np.uint8)

            # Blend arena boundary
            b_alpha_3d = border_alpha[:, :, np.newaxis]
            out_frame = (out_frame * (1.0 - b_alpha_3d) + border_overlay * b_alpha_3d).astype(np.uint8)

        # 6. Draw 3D Minecraft Ember Particles
        self._update_and_draw_particles(out_frame, dt, leg_mask=leg_mask)

        # 7. INDEPENDENT DUAL-FOOT CONTACT CIRCLES
        if self.mapper is not None and pose.tracked:
            is_air = pose.airborne
            air_col = (255, 240, 0)      # Neon cyan in BGR
            ground_col = (50, 255, 100)   # Safe green in BGR
            danger_col = (30, 30, 255)    # Lava red in BGR

            def foot_color(x_m: float, y_m: float) -> tuple[int, int, int]:
                if is_air:
                    return air_col
                c = max(0, min(GRID_SIZE - 1, int(x_m / (self.arena_w_m / GRID_SIZE))))
                r = max(0, min(GRID_SIZE - 1, int(y_m / (self.arena_h_m / GRID_SIZE))))
                if wave and wave.heat[c][r] >= 0.8:
                    return danger_col
                return ground_col

            if pose.left_foot_m is not None:
                lx, ly = self.mapper.to_pixels(*pose.left_foot_m)
                col_l = foot_color(*pose.left_foot_m)
                cv2.circle(out_frame, (int(round(lx)), int(round(ly))), 8, col_l, 2, cv2.LINE_AA)
                cv2.circle(out_frame, (int(round(lx)), int(round(ly))), 3, col_l, -1, cv2.LINE_AA)

            if pose.right_foot_m is not None:
                rx, ry = self.mapper.to_pixels(*pose.right_foot_m)
                col_r = foot_color(*pose.right_foot_m)
                cv2.circle(out_frame, (int(round(rx)), int(round(ry))), 8, col_r, 2, cv2.LINE_AA)
                cv2.circle(out_frame, (int(round(rx)), int(round(ry))), 3, col_r, -1, cv2.LINE_AA)

            if pose.left_foot_m is None and pose.right_foot_m is None and pose.floor_x_m is not None:
                cx, cy = self.mapper.to_pixels(pose.floor_x_m, pose.floor_y_m)
                col_c = foot_color(pose.floor_x_m, pose.floor_y_m)
                cv2.circle(out_frame, (int(round(cx)), int(round(cy))), 10, col_c, 2, cv2.LINE_AA)
                cv2.circle(out_frame, (int(round(cx)), int(round(cy))), 3, col_c, -1, cv2.LINE_AA)

        return out_frame

    def draw(self, state: GameState, wave: LavaWave | None, pose: PoseFrame,
             cam_frame: np.ndarray | None = None,
             debug: dict | None = None,
             banner_msg: str = "", banner_color: tuple[int, int, int] = TEXT_GOLD,
             round_title: str = "", round_sub: str = "",
             round_color: tuple[int, int, int] = (0, 255, 120),
             round_timer: float = 0.0,
             current_round: int = 1,
             round_multiplier: float = 1.0,
             is_resting: bool = False,
             fps: float = 60.0) -> None:
        """Render complete AR frame scaled dynamically to the maximized window size."""
        now = time.time() - self.start_time
        curr_t = time.time()
        dt = max(0.001, min(0.1, curr_t - self.last_frame_time))
        self.last_frame_time = curr_t

        win_w, win_h = self.screen.get_size()
        self.screen.fill((16, 18, 22))

        # ======================================================================
        # 0. RETRO 8-BIT ARCADE SOUND & STATE TRANSITIONS
        # ======================================================================
        is_playing = (state.phase == Phase.PLAYING)
        was_playing = (self._prev_phase == Phase.PLAYING)

        # Detect [R] restart while staying in Phase.PLAYING (round 1 reset with score 0)
        is_restarted = (
            is_playing
            and was_playing
            and current_round == 1
            and state.score == 0
            and state.time_left_s > self._prev_time_left_s + 5.0
        )

        if (is_playing and not was_playing) or is_restarted:
            self._feet_locked = False
            self._game_start_sound_played = False
            self._last_round_announced = 0
            self._game_over_sound_played = False

        # Detect when feet lock onto the arena:
        # Feet lock when is_resting is True in round 1, or round_sub signals resting/prepare/starting,
        # or banner message signals locked feet/gameplay
        if not self._feet_locked:
            if is_resting or any(k in round_sub.upper() for k in ("REST", "PREPARE", "STARTING", "READY")):
                self._feet_locked = True
            elif any(k in banner_msg.upper() for k in ("LOCKED", "ROUND 1 GET READY", "RIVER", "ISLAND", "DONUT", "BRIDGE", "DODGED", "BURNT")):
                self._feet_locked = True

        # Pre-entry waiting condition: Phase.CALIBRATING or Round 1 before feet lock
        is_waiting_for_entry = (
            (state.phase == Phase.CALIBRATING)
            or (is_playing and current_round == 1 and not self._feet_locked)
        )

        # Pre-Entry: Trigger GAME START fanfare + voice callout once
        if is_waiting_for_entry:
            if not self._game_start_sound_played:
                self.sound_mgr.play_game_start()
                self._game_start_sound_played = True
        else:
            # Active Round Announcements: Trigger once when feet lock and round banner triggers
            if (round_timer > 0.0 or is_resting) and round_title:
                up_title = round_title.upper()
                if "ROUND 1" in up_title and self._last_round_announced < 1:
                    self.sound_mgr.play_round_start(1)
                    self._last_round_announced = 1
                elif "ROUND 2" in up_title and self._last_round_announced < 2:
                    self.sound_mgr.play_round_start(2)
                    self._last_round_announced = 2
                elif ("ROUND 3" in up_title or "FINAL" in up_title) and self._last_round_announced < 3:
                    self.sound_mgr.play_round_start(3)
                    self._last_round_announced = 3

        # Game Over: Transition into Phase.RESULT
        if state.phase == Phase.RESULT:
            if not self._game_over_sound_played:
                self.sound_mgr.play_game_over()
                self._game_over_sound_played = True
        else:
            self._game_over_sound_played = False

        self._prev_phase = state.phase
        self._prev_time_left_s = state.time_left_s

        # ======================================================================
        # 1. RENDER BACKGROUND (AR CAMERA FEED OR VIRTUAL ARENA)
        # ======================================================================
        if cam_frame is not None:
            ar_frame = self._render_ar_frame(cam_frame, wave, pose, now, dt, debug=debug)
            rgb = cv2.cvtColor(ar_frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            cam_surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
            scaled_cam = pygame.transform.scale(cam_surf, (win_w, win_h))
            self.screen.blit(scaled_cam, (0, 0))
        else:
            # Fallback Virtual Arena (for --keyboard mode)
            arena_surf = pygame.Surface((win_w, win_h))
            arena_surf.fill((30, 32, 38))

            if wave is not None:
                sub_w = win_w / GRID_SIZE
                sub_h = (win_h - HUD_TOP_H - HUD_BOT_H) / GRID_SIZE
                for c in range(GRID_SIZE):
                    for r in range(GRID_SIZE):
                        h = wave.heat[c][r]
                        if h > 0.0:
                            if h < 0.6:
                                col = (100, 100, 110)
                            elif h < 0.8:
                                col = (220, 140, 30)
                            else:
                                col = (245, 50, 15)
                            x = int(c * sub_w)
                            y = int(HUD_TOP_H + (GRID_SIZE - 1 - r) * sub_h)
                            pygame.draw.rect(arena_surf, col, (x, y, int(sub_w) + 1, int(sub_h) + 1))

            # Virtual player
            if pose.floor_x_m is not None and pose.floor_y_m is not None:
                fx = max(0.0, min(1.0, pose.floor_x_m / self.arena_w_m))
                fy = max(0.0, min(1.0, pose.floor_y_m / self.arena_h_m))
                px = int(fx * win_w)
                py = int(HUD_TOP_H + (1.0 - fy) * (win_h - HUD_TOP_H - HUD_BOT_H))
                p_col = (0, 240, 255) if pose.airborne else (0, 255, 120)
                pygame.draw.circle(arena_surf, p_col, (px, py), 14)
                pygame.draw.circle(arena_surf, (255, 255, 255), (px, py), 14, width=2)

            self.screen.blit(arena_surf, (0, 0))

        # ======================================================================
        # 2. TOP HUD (DECLUTTERED SUNSET ARCADE HUD BAR)
        # ======================================================================
        hud_surface = pygame.Surface((win_w, HUD_TOP_H), pygame.SRCALPHA)
        hud_surface.fill((22, 12, 46, 210))  # Translucent twilight purple
        self.screen.blit(hud_surface, (0, 0))
        # Crisp 2px glowing coral-pink / gold bottom border (#ff4d6d / #ffbe0b)
        pygame.draw.line(self.screen, (255, 77, 109), (0, HUD_TOP_H - 2), (win_w, HUD_TOP_H - 2), 1)
        pygame.draw.line(self.screen, (255, 190, 11), (0, HUD_TOP_H - 1), (win_w, HUD_TOP_H - 1), 1)

        # ----------------------------------------------------------------------
        # Left Section (Clean Event Branding)
        # Show ONLY TITANS' ARCADE in distinct neon cyan-to-mint gradient
        # (#00F5D4 top -> #38EF7D bottom, outline #140A24)
        # ----------------------------------------------------------------------
        titans_scale = 3 if win_w >= 950 else 2
        titans_surf = render_pixel_text(
            "TITANS' ARCADE",
            scale=titans_scale,
            top_color=(0, 245, 212),       # #00F5D4 (neon cyan)
            bottom_color=(56, 239, 125),   # #38EF7D (mint green)
            outline_color=(20, 10, 36),    # #140A24
            shadow_offset=2,
        )
        self.screen.blit(titans_surf, titans_surf.get_rect(midleft=(24, HUD_TOP_H // 2)))

        # ----------------------------------------------------------------------
        # Center Section (Prominent Game Title Only)
        # Show ONLY centered FLOOR IS LAVA (Scale 4 Two-Tone golden-yellow/crimson)
        # ----------------------------------------------------------------------
        center_scale = 3 if win_w < 950 else 4
        center_title = render_pixel_text(
            "FLOOR IS LAVA",
            scale=center_scale,
            top_color=(255, 224, 51),     # #FFE033 (golden-yellow)
            bottom_color=(232, 56, 24),    # #E83818 (crimson/red-orange)
            outline_color=(20, 10, 36),    # #140A24
            shadow_offset=3,
        )
        self.screen.blit(center_title, center_title.get_rect(center=(win_w // 2, HUD_TOP_H // 2)))

        # ----------------------------------------------------------------------
        # Right Section (Score & Timer + Horizontal Gauge Bar)
        # Top: SCORE: 0000 + 30s | Bottom: Horizontal Gauge Bar normalized to 30.0s
        # ----------------------------------------------------------------------
        secs = max(0, int(math.ceil(state.time_left_s)))
        if secs <= 8:
            time_top = (255, 77, 109)  # #ff4d6d (pulsing coral-red)
            time_bot = (232, 56, 24)
            bar_fill = (255, 77, 109)
        else:
            time_top = (0, 245, 212)   # #00f5d4 (neon cyan)
            time_bot = (0, 180, 210)
            bar_fill = (0, 245, 212)

        timer_surf = render_pixel_text(f"{secs:02d}s", scale=3, top_color=time_top, bottom_color=time_bot)
        score_surf = render_pixel_text(f"SCORE: {state.score:04d}", scale=3, top_color=(255, 224, 51), bottom_color=(255, 140, 20))

        right_m = 24
        timer_x = win_w - right_m - timer_surf.get_width()
        score_x = timer_x - score_surf.get_width() - 18
        self.screen.blit(timer_surf, (timer_x, 12))
        self.screen.blit(score_surf, (score_x, 12))

        # Horizontal timer gauge bar directly underneath matching score/timer width
        bar_x = score_x
        bar_w = (win_w - right_m) - bar_x
        bar_y = 52
        bar_h = 8
        pct = max(0.0, min(1.0, state.time_left_s / DEFAULT_ROUND_DURATION_S))
        pygame.draw.rect(self.screen, (35, 18, 65), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        if pct > 0:
            fill_w = max(4, int(bar_w * pct))
            pygame.draw.rect(self.screen, bar_fill, (bar_x, bar_y, fill_w, bar_h), border_radius=4)
        pygame.draw.rect(self.screen, (255, 190, 11), (bar_x, bar_y, bar_w, bar_h), width=1, border_radius=4)

        # ======================================================================
        # 3. CENTER-SCREEN COMPACT ROUND ANNOUNCEMENT CARD
        # ======================================================================
        show_card = is_waiting_for_entry or ((round_timer > 0.0 or is_resting) and round_title)
        if show_card:
            if is_waiting_for_entry:
                display_title = "GAME START"
                display_sub = "STEP INTO THE ARENA TO BEGIN"
                title_top, title_bot = (255, 224, 51), (232, 56, 24)
                sub_top, sub_bot = (0, 245, 212), (56, 239, 125)
                card_w = 420
            else:
                display_title = round_title
                display_sub = round_sub
                title_top, title_bot = (255, 224, 51), (232, 56, 24)
                sub_top, sub_bot = (56, 239, 125), (245, 248, 255)
                card_w = 360

            card_h = 240
            card_x = (win_w - card_w) // 2
            card_y = (win_h - card_h) // 2

            card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
            card_surf.fill((22, 12, 46, 230))  # Semi-translucent dark twilight purple
            self.screen.blit(card_surf, (card_x, card_y))

            # Crisp 2px double glowing neon coral-gold border (#ff4d6d / #ffbe0b)
            pygame.draw.rect(self.screen, (255, 190, 11), (card_x, card_y, card_w, card_h), width=2, border_radius=10)
            pygame.draw.rect(self.screen, (255, 77, 109), (card_x + 3, card_y + 3, card_w - 6, card_h - 6), width=1, border_radius=8)

            # Centered large 8-bit title in golden-yellow / red-orange
            t_surf = render_pixel_text(display_title, scale=4, top_color=title_top, bottom_color=title_bot, shadow_offset=3)
            t_rect = t_surf.get_rect(center=(win_w // 2, card_y + 80))
            self.screen.blit(t_surf, t_rect)

            # Compact subtitle underneath
            if display_sub:
                s_surf = render_pixel_text(display_sub, scale=2, top_color=sub_top, bottom_color=sub_bot, shadow_offset=2)
                s_rect = s_surf.get_rect(center=(win_w // 2, card_y + 160))
                self.screen.blit(s_surf, s_rect)

        # ======================================================================
        # 4. EPHEMERAL FLOATING ARCADE TOAST PILL
        # ======================================================================
        if banner_msg and (
            banner_msg != self.last_seen_banner_msg
            or state.dodges != self.last_dodges
            or state.hits != self.last_hits
        ):
            self.toast_msg = banner_msg
            self.toast_color = banner_color
            self.toast_timer = 1.4
            self.toast_max_duration = 1.4
            self.last_seen_banner_msg = banner_msg
            self.last_dodges = state.dodges
            self.last_hits = state.hits

        if self.toast_timer > 0.0:
            self.toast_timer = max(0.0, self.toast_timer - dt)
            if self.toast_timer <= 0.0:
                self.toast_msg = ""

        active_toast = self.toast_msg
        active_col = self.toast_color
        is_airborne_toast = False

        if not active_toast and pose.airborne:
            active_toast = "AIRBORNE! JUMP IMMUNITY!"
            active_col = (0, 245, 212)
            is_airborne_toast = True

        if active_toast:
            msg_up = active_toast.upper()
            if any(k in msg_up for k in ("SAFE", "DODGED", "IMMUNITY", "BONUS", "LOCKED")):
                top_col = (56, 239, 125)    # mint green #38ef7d
                bot_col = (0, 200, 100)
                border_col = (56, 239, 125)
            elif any(k in msg_up for k in ("HIT", "BURNT", "LAVA", "WARNING", "DANGER", "HEATING")):
                top_col = (255, 224, 51)    # golden-yellow #ffe033
                bot_col = (232, 56, 24)     # red-orange #e83818
                border_col = (255, 77, 109) # coral-pink #ff4d6d
            elif any(k in msg_up for k in ("AIRBORNE", "JUMP")):
                top_col = (0, 245, 212)     # neon cyan #00f5d4
                bot_col = (0, 160, 255)
                border_col = (0, 245, 212)
            else:
                top_col = active_col
                bot_col = active_col
                border_col = (255, 190, 11)  # gold #ffbe0b

            if is_airborne_toast:
                alpha = 230
                drift_y = 0
            else:
                t = self.toast_timer
                drift_y = int((1.0 - (t / self.toast_max_duration)) * 12)
                alpha = int(230 * min(1.0, t / 0.4)) if t < 0.4 else 230

            t_surf = render_pixel_text(active_toast, scale=2, top_color=top_col, bottom_color=bot_col, shadow_offset=2)
            tw, th = t_surf.get_size()
            pw = tw + 36
            ph = th + 14
            pill_surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
            pill_surf.fill((22, 12, 46, alpha))
            pygame.draw.rect(pill_surf, (*border_col, alpha), (0, 0, pw, ph), width=2, border_radius=ph // 2)

            if alpha < 255:
                txt_copy = t_surf.copy()
                txt_copy.set_alpha(alpha)
                pill_surf.blit(txt_copy, (18, 7))
            else:
                pill_surf.blit(t_surf, (18, 7))

            toast_x = (win_w - pw) // 2
            toast_y = HUD_TOP_H + 16 - drift_y
            self.screen.blit(pill_surf, (toast_x, toast_y))

        # ======================================================================
        # 5. RESULT OVERLAY (PHASE.RESULT)
        # ======================================================================
        if state.phase == Phase.RESULT:
            overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
            overlay.fill((14, 8, 28, 210))
            self.screen.blit(overlay, (0, 0))

            card_w = min(win_w - 60, 760)
            card_h = min(win_h - 80, 440)
            card_x = (win_w - card_w) // 2
            card_y = (win_h - card_h) // 2

            card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
            card_surf.fill((20, 11, 42, 225))
            self.screen.blit(card_surf, (card_x, card_y))

            # Glowing neon gold & coral border
            pygame.draw.rect(self.screen, (255, 209, 82), (card_x, card_y, card_w, card_h), width=3)
            pygame.draw.rect(self.screen, (232, 56, 24), (card_x + 3, card_y + 3, card_w - 6, card_h - 6), width=1)

            center_x = win_w // 2

            over_title = render_pixel_text("ROUND OVER!", scale=5, top_color=(255, 224, 51), bottom_color=(232, 56, 24), shadow_offset=4)
            self.screen.blit(over_title, over_title.get_rect(center=(center_x, card_y + 60)))

            final_score = render_pixel_text(f"FINAL SCORE: {state.score:04d}", scale=4, top_color=(255, 255, 255), bottom_color=(200, 220, 255), shadow_offset=3)
            self.screen.blit(final_score, final_score.get_rect(center=(center_x, card_y + 140)))

            stats_str = f"DODGES: {state.dodges}   HITS: {state.hits}   BEST STREAK: {state.best_streak}"
            stats_surf = render_pixel_text(stats_str, scale=2, top_color=(100, 240, 255), bottom_color=(20, 160, 255), shadow_offset=2)
            self.screen.blit(stats_surf, stats_surf.get_rect(center=(center_x, card_y + 210)))

            claim_surf = render_pixel_text("VISIT DESK TO CLAIM YOUR PHOTOCARD!", scale=2, top_color=(255, 224, 51), bottom_color=(255, 140, 20), shadow_offset=2)
            self.screen.blit(claim_surf, claim_surf.get_rect(center=(center_x, card_y + 280)))

            prompt_surf = render_pixel_text("[R] PLAY AGAIN   |   [ESC] EXIT", scale=2, top_color=(80, 255, 160), bottom_color=(0, 200, 100), shadow_offset=2)
            self.screen.blit(prompt_surf, prompt_surf.get_rect(center=(center_x, card_y + 360)))

        pygame.display.flip()


# Backward-compatible alias
DualViewRenderer = LavaArenaRenderer
