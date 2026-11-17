"""Floor is Lava - baseline test game for vision and floor geometry.

Validates camera capture, floor homography, dual-foot tracking, and jumping
immunity on a hierarchical 3x3 macro / 9x9 micro floor grid.

GRID CONCEPT (Hierarchical 3x3 x 3x3 = 9x9):
  * Outer Floor: 3x3 Macro Zones (mega-tiles, cols 0..2, rows 0..2).
  * Inner Resolution: Each macro zone is 3x3 micro-tiles, creating a 9x9 grid
    (81 micro-tiles, cols 0..8, rows 0..8).
  * This allows arbitrary lava shapes (clusters, rings, rivers, or full macro blocks)
    while precisely tracking which tile each foot is on.

HOW TO PLAY:
  * The grid represents the calibrated arena floor (y grows away from the camera).
  * When a wave starts, warning tiles blink YELLOW / ORANGE.
  * After the countdown, they erupt into RED LAVA.
  * Safe tiles stay dark. Walk to safe tiles before the eruption!
  * JUMP IMMUNITY: If you JUMP (airborne=True) while on lava, you are SAFE in mid-air!
  * Step out of the floor box or step on grounded lava -> Lava Hit!

RUN FROM REPO ROOT:
    python -m tools.floor_in_lava_test --keyboard    # No camera, WASD to move, SPACE to jump
    python -m tools.floor_in_lava_test               # Real camera tracking (needs calibration.local.json)
    python -m tools.floor_in_lava_test --pattern cluster  # Test arbitrary micro-cluster lava shapes

CONTROLS:
    SPACE or Raise Both Hands : Start game / Restart after Game Over
    V                         : Switch view (Dual side-by-side / Map only / Camera overlay)
    P                         : Cycle lava shape pattern (random, macro, island, cluster, cross)
    K                         : Toggle keyboard movement mode (WASD + SPACE)
    R                         : Reset / restart game
    Q or ESC                  : Quit
"""
from __future__ import annotations

import argparse
import atexit
import math
import random
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from pose_core.capture import CameraPoseSource
from pose_core.floor_map import FloorMapper
from shared.contracts import PoseFrame
from tools.calibrate_floor import calibrate_floor

# ---- UI & Rendering Constants ------------------------------------------------
MAP_SIZE = 600           # Dimension of square floor map (pixels)
SIDEBAR_W = 320          # Width of stats sidebar (pixels)
PAD = 40                 # Margin around floor on map panel
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Colors in BGR
C_BG = (20, 20, 24)
C_GRID_MICRO = (48, 45, 42)
C_GRID_MACRO = (180, 160, 90)
C_SAFE_TILE = (34, 32, 38)
C_WARN_TILE = (0, 165, 255)       # Amber / Orange
C_LAVA_TILE = (0, 30, 230)        # Hot Red
C_LAVA_CORE = (0, 100, 255)       # Bright Lava Core
C_FOOT_L = (255, 140, 0)          # Left foot: Sky Blue
C_FOOT_R = (255, 0, 255)          # Right foot: Magenta
C_FOOT_BOTH = (200, 50, 200)      # Purple when feet together
C_AIRBORNE = (50, 230, 70)        # Green
C_TEXT_WHITE = (245, 245, 245)
C_TEXT_GREY = (160, 160, 165)
C_TEXT_WARN = (0, 200, 255)
C_OUT_OF_BOUNDS = (0, 0, 255)

# Game Timers
WARN_DURATION_S = 1.6
LAVA_DURATION_S = 2.0
COOLDOWN_S = 0.8


# ==============================================================================
# Pure Functions: Hierarchical Grid Geometry
# ==============================================================================

def tile_of_grid(
    x: float,
    y: float,
    w_m: float,
    h_m: float,
    cols: int = 9,
    rows: int = 9,
) -> tuple[int, int] | None:
    """Return (col, row) on a cols x rows grid for floor point (x, y).

    Returns None if (x, y) is outside [0, w_m] x [0, h_m].
    Convention: (0,0) near-left, x to right, y away from camera.
    """
    if x < 0.0 or x > w_m or y < 0.0 or y > h_m:
        return None
    if cols <= 0 or rows <= 0 or w_m <= 0.0 or h_m <= 0.0:
        return None
    col = min(cols - 1, int(x / (w_m / cols)))
    row = min(rows - 1, int(y / (h_m / rows)))
    return (col, row)


def micro_to_macro(col: int, row: int, sub_cols: int = 3, sub_rows: int = 3) -> tuple[int, int]:
    """Map a 9x9 micro-tile (col, row) to its containing 3x3 macro-zone (macro_col, macro_row)."""
    return (col // sub_cols, row // sub_rows)


def macro_to_micros(
    macro_c: int,
    macro_r: int,
    sub_cols: int = 3,
    sub_rows: int = 3,
) -> set[tuple[int, int]]:
    """Return all 9 micro-tiles (c, r) contained in a macro-zone (macro_c, macro_r)."""
    out = set()
    base_c = macro_c * sub_cols
    base_r = macro_r * sub_rows
    for c in range(base_c, base_c + sub_cols):
        for r in range(base_r, base_r + sub_rows):
            out.add((c, r))
    return out


def tile_corners_m(
    col: int,
    row: int,
    w_m: float,
    h_m: float,
    cols: int = 9,
    rows: int = 9,
) -> list[tuple[float, float]]:
    """Return 4 corner points [(x1, y1), (x2, y1), (x2, y2), (x1, y2)] in floor metres."""
    dx = w_m / cols
    dy = h_m / rows
    x1, x2 = col * dx, (col + 1) * dx
    y1, y2 = row * dy, (row + 1) * dy
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


# ==============================================================================
# Lava Pattern Generator (Arbitrary Shapes on 9x9 Grid)
# ==============================================================================

PATTERNS = ["random", "cluster", "macro", "island", "cross"]


def generate_lava_tiles(
    pattern: str,
    cols: int = 9,
    rows: int = 9,
    macro_cols: int = 3,
    macro_rows: int = 3,
    player_micro: tuple[int, int] | None = None,
) -> tuple[set[tuple[int, int]], str]:
    """Generate a set of micro-tiles for the next lava eruption wave.

    Supports macro zones, safe islands, rivers/crosses, and arbitrary micro-clusters.
    """
    mode = pattern
    if mode == "random":
        mode = random.choice(["macro", "island", "cluster", "cross"])

    lava: set[tuple[int, int]] = set()
    desc = mode.upper()

    if mode == "macro":
        # 1 to 3 macro-zones ignite
        n_zones = random.randint(1, 3)
        all_zones = [(c, r) for c in range(macro_cols) for r in range(macro_rows)]
        chosen = random.sample(all_zones, n_zones)
        for mc, mr in chosen:
            lava.update(macro_to_micros(mc, mr, cols // macro_cols, rows // macro_rows))
        desc = f"MACRO ZONES ({n_zones}/9 erupting)"

    elif mode == "island":
        # Only 1 safe macro island remains, all other 8 erupt!
        all_zones = [(c, r) for c in range(macro_cols) for r in range(macro_rows)]
        # Prefer the zone where the player is or adjacent
        if player_micro is not None:
            safe_mc, safe_mr = micro_to_macro(player_micro[0], player_micro[1])
        else:
            safe_mc, safe_mr = random.choice(all_zones)
        for mc, mr in all_zones:
            if (mc, mr) != (safe_mc, safe_mr):
                lava.update(macro_to_micros(mc, mr, cols // macro_cols, rows // macro_rows))
        desc = f"SAFE ISLAND: Zone ({safe_mc},{safe_mr})"

    elif mode == "cross":
        # Cross rivers / lines of lava
        axis = random.choice(["horizontal", "vertical", "cross"])
        if axis in ("horizontal", "cross"):
            r_line = random.randint(1, rows - 2)
            for c in range(cols):
                lava.add((c, r_line))
        if axis in ("vertical", "cross"):
            c_line = random.randint(1, cols - 2)
            for r in range(rows):
                lava.add((c_line, r))
        desc = f"LAVA RIVER ({axis})"

    else:  # "cluster": Arbitrary shape of connected micro-tiles
        # Pick 2-3 seed points and expand outward randomly
        n_seeds = random.randint(2, 4)
        cluster_size = random.randint(15, 30)
        seeds = [
            (random.randint(0, cols - 1), random.randint(0, rows - 1))
            for _ in range(n_seeds)
        ]
        lava.update(seeds)
        queue = list(seeds)
        while len(lava) < cluster_size and queue:
            qc, qr = queue.pop(random.randrange(len(queue)))
            neighbors = [
                (qc + 1, qr), (qc - 1, qr), (qc, qr + 1), (qc, qr - 1),
                (qc + 1, qr + 1), (qc - 1, qr - 1)
            ]
            random.shuffle(neighbors)
            for nc, nr in neighbors:
                if 0 <= nc < cols and 0 <= nr < rows and (nc, nr) not in lava:
                    lava.add((nc, nr))
                    queue.append((nc, nr))
                    if len(lava) >= cluster_size:
                        break
        desc = f"ARBITRARY CLUSTER ({len(lava)} micro-tiles)"

    return lava, desc


# ==============================================================================
# Player State Representation
# ==============================================================================

@dataclass
class PlayerState:
    tracked: bool = False
    in_box: bool = False
    x_m: float = 0.6225
    y_m: float = 0.6225
    left_m: tuple[float, float] | None = None
    right_m: tuple[float, float] | None = None
    airborne: bool = False
    arms_up: bool = False


# ==============================================================================
# Pure Game Engine: Floor-is-Lava State Machine
# ==============================================================================

class FloorLavaGame:
    """Pure game logic for Floor-is-Lava. No GUI or camera dependencies."""

    def __init__(
        self,
        w_m: float = 1.245,
        h_m: float = 1.245,
        cols: int = 9,
        rows: int = 9,
        macro_cols: int = 3,
        macro_rows: int = 3,
        pattern: str = "random",
    ):
        self.w_m = float(w_m)
        self.h_m = float(h_m)
        self.cols = int(cols)
        self.rows = int(rows)
        self.macro_cols = int(macro_cols)
        self.macro_rows = int(macro_rows)
        self.pattern_mode = pattern

        self.phase: str = "IDLE"  # "IDLE", "WARNING", "LAVA", "COOLDOWN", "GAME_OVER"
        self.score: int = 0
        self.waves_survived: int = 0
        self.dodges: int = 0
        self.hits: int = 0
        self.lava_hits: int = 0
        self.out_of_bounds_hits: int = 0

        self.timer: float = 0.0
        self.hazard_tiles: set[tuple[int, int]] = set()
        self.hazard_desc: str = "Ready"
        self.banner_msg: str = "RAISE HANDS OR PRESS SPACE TO START"
        self.banner_color: tuple[int, int, int] = C_TEXT_WARN
        self.game_over_reason: str = ""

    def reset(self) -> None:
        self.phase = "IDLE"
        self.score = 0
        self.waves_survived = 0
        self.dodges = 0
        self.hits = 0
        self.lava_hits = 0
        self.out_of_bounds_hits = 0
        self.timer = 0.0
        self.hazard_tiles = set()
        self.hazard_desc = "Ready"
        self.banner_msg = "RAISE HANDS OR PRESS SPACE TO START"
        self.banner_color = C_TEXT_WARN
        self.game_over_reason = ""

    def start_game(self) -> None:
        self.reset()
        self._spawn_wave()

    def _spawn_wave(self, player_micro: tuple[int, int] | None = None) -> None:
        self.phase = "WARNING"
        self.timer = WARN_DURATION_S
        self.hazard_tiles, self.hazard_desc = generate_lava_tiles(
            self.pattern_mode,
            self.cols,
            self.rows,
            self.macro_cols,
            self.macro_rows,
            player_micro,
        )
        self.banner_msg = f"WARNING! Lava erupting in {self.timer:.1f}s - MOVE!"
        self.banner_color = C_WARN_TILE

    def update(self, dt: float, player: PlayerState) -> None:
        if self.phase == "IDLE":
            if player.arms_up:
                self.start_game()
            return

        if self.phase == "GAME_OVER":
            if player.arms_up:
                self.start_game()
            return

        # Player presence check
        if not player.tracked:
            self.banner_msg = "PLAYER LOST - Step into the arena"
            self.banner_color = C_OUT_OF_BOUNDS
            return

        # Player tile lookup
        tile_l = tile_of_grid(player.left_m[0], player.left_m[1], self.w_m, self.h_m, self.cols, self.rows) if player.left_m else None
        tile_r = tile_of_grid(player.right_m[0], player.right_m[1], self.w_m, self.h_m, self.cols, self.rows) if player.right_m else None
        tile_c = tile_of_grid(player.x_m, player.y_m, self.w_m, self.h_m, self.cols, self.rows)

        # Check arena boundaries
        if not player.in_box:
            self.hits += 1
            self.out_of_bounds_hits += 1
            self.phase = "GAME_OVER"
            self.game_over_reason = "STEPPED OUT OF ARENA"
            self.banner_msg = "GAME OVER - Stepped out of the box!"
            self.banner_color = C_OUT_OF_BOUNDS
            return

        self.timer -= dt

        if self.phase == "WARNING":
            self.banner_msg = f"LAVA INCOMING: {max(0.0, self.timer):.1f}s"
            self.banner_color = C_WARN_TILE
            if self.timer <= 0.0:
                self.phase = "LAVA"
                self.timer = LAVA_DURATION_S
                self.banner_msg = "LAVA ACTIVE! DO NOT TOUCH RED TILES!"
                self.banner_color = C_LAVA_TILE

        elif self.phase == "LAVA":
            self.banner_msg = f"HOT LAVA ACTIVE! ({self.timer:.1f}s)"
            self.banner_color = C_LAVA_TILE

            # Check collision: if player is airborne (jumping), they are safe in mid-air!
            if player.airborne:
                self.banner_msg = "AIRBORNE! (Safe from lava in mid-air)"
                self.banner_color = C_AIRBORNE
            else:
                # Player is grounded: check whether any foot is on lava
                on_lava = False
                if tile_l is not None and tile_l in self.hazard_tiles:
                    on_lava = True
                if tile_r is not None and tile_r in self.hazard_tiles:
                    on_lava = True
                if tile_l is None and tile_r is None and tile_c is not None and tile_c in self.hazard_tiles:
                    on_lava = True

                if on_lava:
                    self.hits += 1
                    self.lava_hits += 1
                    self.phase = "GAME_OVER"
                    self.game_over_reason = "STEPPED IN LAVA!"
                    self.banner_msg = "GAME OVER - Stepped in Lava!"
                    self.banner_color = C_OUT_OF_BOUNDS
                    return

            if self.timer <= 0.0:
                # Successfully survived lava wave!
                self.waves_survived += 1
                self.dodges += 1
                self.score += 10
                self.phase = "COOLDOWN"
                self.timer = COOLDOWN_S
                self.hazard_tiles.clear()
                self.banner_msg = f"WAVE SURVIVED! (+10 pts) Get ready..."
                self.banner_color = C_AIRBORNE

        elif self.phase == "COOLDOWN":
            if self.timer <= 0.0:
                self._spawn_wave(tile_c)


# ==============================================================================
# Rendering: Top-Down Floor Map, Camera Overlay, & Sidebar
# ==============================================================================

def render_top_down_map(game: FloorLavaGame, pl: PlayerState) -> np.ndarray:
    """Render the 600x600 top-down floor grid with 3x3 macro and 9x9 micro tiles."""
    img = np.full((MAP_SIZE, MAP_SIZE, 3), C_BG, dtype=np.uint8)

    inner = MAP_SIZE - 2 * PAD
    dx_micro = inner / game.cols
    dy_micro = inner / game.rows

    # 1. Fill micro-tiles
    pulse = 0.75 + 0.25 * math.sin(time.monotonic() * 9.0)
    for c in range(game.cols):
        for r in range(game.rows):
            # Floor convention: row 0 nearest to camera (screen bottom), row (rows-1) farthest (screen top)
            x1 = int(PAD + c * dx_micro)
            x2 = int(PAD + (c + 1) * dx_micro)
            y2 = int(MAP_SIZE - PAD - r * dy_micro)
            y1 = int(MAP_SIZE - PAD - (r + 1) * dy_micro)

            tile_key = (c, r)
            if tile_key in game.hazard_tiles:
                if game.phase == "WARNING":
                    # Warning amber flash
                    flash = 0.5 + 0.5 * math.sin(time.monotonic() * 12.0)
                    col = (
                        int(C_WARN_TILE[0] * flash),
                        int(C_WARN_TILE[1] * flash),
                        int(C_WARN_TILE[2] * flash),
                    )
                else:  # LAVA
                    col = (
                        int(C_LAVA_TILE[0] * pulse),
                        int(C_LAVA_TILE[1] * pulse),
                        min(255, int(C_LAVA_TILE[2] * pulse + 30)),
                    )
            else:
                col = C_SAFE_TILE

            cv2.rectangle(img, (x1, y1), (x2, y2), col, -1)

    # 2. Draw 9x9 micro-grid lines
    for c in range(game.cols + 1):
        x = int(PAD + c * dx_micro)
        cv2.line(img, (x, PAD), (x, MAP_SIZE - PAD), C_GRID_MICRO, 1, cv2.LINE_AA)
    for r in range(game.rows + 1):
        y = int(MAP_SIZE - PAD - r * dy_micro)
        cv2.line(img, (PAD, y), (MAP_SIZE - PAD, y), C_GRID_MICRO, 1, cv2.LINE_AA)

    # 3. Draw 3x3 macro-grid lines (thicker lines)
    sub_c = game.cols // game.macro_cols
    sub_r = game.rows // game.macro_rows
    for mc in range(game.macro_cols + 1):
        x = int(PAD + mc * sub_c * dx_micro)
        cv2.line(img, (x, PAD), (x, MAP_SIZE - PAD), C_GRID_MACRO, 2, cv2.LINE_AA)
    for mr in range(game.macro_rows + 1):
        y = int(MAP_SIZE - PAD - mr * sub_r * dy_micro)
        cv2.line(img, (PAD, y), (MAP_SIZE - PAD, y), C_GRID_MACRO, 2, cv2.LINE_AA)

    # Outer boundary border
    border_col = C_AIRBORNE if pl.in_box else C_OUT_OF_BOUNDS
    cv2.rectangle(img, (PAD, PAD), (MAP_SIZE - PAD, MAP_SIZE - PAD), border_col, 3)

    # Labels for boundaries
    cv2.putText(img, "NEAR (CAMERA)", (MAP_SIZE // 2 - 60, MAP_SIZE - 12), FONT, 0.42, C_TEXT_GREY, 1, cv2.LINE_AA)
    cv2.putText(img, "FAR", (MAP_SIZE // 2 - 14, PAD - 14), FONT, 0.42, C_TEXT_GREY, 1, cv2.LINE_AA)
    cv2.putText(img, "LEFT", (8, MAP_SIZE // 2), FONT, 0.42, C_TEXT_GREY, 1, cv2.LINE_AA)
    cv2.putText(img, "RIGHT", (MAP_SIZE - 36, MAP_SIZE // 2), FONT, 0.42, C_TEXT_GREY, 1, cv2.LINE_AA)

    # 4. Draw Player Position & Feet
    def to_map_px(fx: float, fy: float) -> tuple[int, int]:
        mx = int(PAD + (fx / game.w_m) * inner)
        my = int(MAP_SIZE - PAD - (fy / game.h_m) * inner)
        return (mx, my)

    if pl.tracked:
        # Draw feet
        if pl.left_m and pl.right_m:
            l_px = to_map_px(*pl.left_m)
            r_px = to_map_px(*pl.right_m)

            # Left Foot
            cv2.circle(img, l_px, 8, C_FOOT_L, -1, cv2.LINE_AA)
            cv2.circle(img, l_px, 10, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "L", (l_px[0] - 4, l_px[1] + 4), FONT, 0.35, (0, 0, 0), 1)

            # Right Foot
            cv2.circle(img, r_px, 8, C_FOOT_R, -1, cv2.LINE_AA)
            cv2.circle(img, r_px, 10, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "R", (r_px[0] - 4, r_px[1] + 4), FONT, 0.35, (0, 0, 0), 1)

            # Body center connector
            c_px = to_map_px(pl.x_m, pl.y_m)
            cv2.line(img, l_px, c_px, (150, 150, 150), 1, cv2.LINE_AA)
            cv2.line(img, r_px, c_px, (150, 150, 150), 1, cv2.LINE_AA)
            cv2.circle(img, c_px, 5, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            c_px = to_map_px(pl.x_m, pl.y_m)
            cv2.circle(img, c_px, 10, C_AIRBORNE if pl.airborne else C_FOOT_BOTH, -1, cv2.LINE_AA)
            cv2.circle(img, c_px, 12, (255, 255, 255), 1, cv2.LINE_AA)

        # Airborne halo
        if pl.airborne:
            c_px = to_map_px(pl.x_m, pl.y_m)
            cv2.circle(img, c_px, 24, C_AIRBORNE, 2, cv2.LINE_AA)
            cv2.putText(img, "AIRBORNE", (c_px[0] - 32, c_px[1] - 28), FONT, 0.42, C_AIRBORNE, 1, cv2.LINE_AA)

    # 5. Top Banner Overlay on Map
    cv2.rectangle(img, (0, 0), (MAP_SIZE, 34), (16, 16, 20), -1)
    cv2.line(img, (0, 34), (MAP_SIZE, 34), (60, 60, 65), 1)
    cv2.putText(img, game.banner_msg, (12, 22), FONT, 0.45, game.banner_color, 1, cv2.LINE_AA)

    return img


def render_sidebar(
    game: FloorLavaGame,
    pl: PlayerState,
    fps: float,
    pose_fps: float,
    keyboard_mode: bool,
) -> np.ndarray:
    """Render the 320x600 information sidebar."""
    side = np.full((MAP_SIZE, SIDEBAR_W, 3), (16, 16, 20), dtype=np.uint8)
    cv2.line(side, (0, 0), (0, MAP_SIZE), (45, 45, 50), 1)

    y = 28
    # Title
    cv2.putText(side, "FLOOR IS LAVA", (18, y), FONT, 0.72, (0, 140, 255), 2, cv2.LINE_AA)
    cv2.putText(side, "3x3 Zone / 9x9 Tile Baseline", (18, y + 18), FONT, 0.38, C_TEXT_GREY, 1, cv2.LINE_AA)

    y += 46
    cv2.line(side, (18, y), (SIDEBAR_W - 18, y), (40, 40, 45), 1)

    # Score & Wave HUD
    y += 24
    cv2.putText(side, f"SCORE: {game.score}", (18, y), FONT, 0.65, (0, 230, 255), 2, cv2.LINE_AA)
    y += 22
    cv2.putText(side, f"Waves Survived: {game.waves_survived}", (18, y), FONT, 0.45, C_TEXT_WHITE, 1, cv2.LINE_AA)
    y += 20
    cv2.putText(side, f"Dodges: {game.dodges}   Hits: {game.hits}", (18, y), FONT, 0.45, C_TEXT_WHITE, 1, cv2.LINE_AA)

    y += 28
    cv2.line(side, (18, y), (SIDEBAR_W - 18, y), (40, 40, 45), 1)

    # Active Hazard Info
    y += 22
    cv2.putText(side, f"PATTERN: {game.pattern_mode.upper()} ([P] cycle)", (18, y), FONT, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
    y += 18
    cv2.putText(side, f"Shape: {game.hazard_desc}", (18, y), FONT, 0.38, C_TEXT_GREY, 1, cv2.LINE_AA)
    y += 18
    active_cnt = len(game.hazard_tiles)
    active_pct = (active_cnt / (game.cols * game.rows)) * 100.0
    cv2.putText(side, f"Lava Coverage: {active_cnt}/81 tiles ({active_pct:.0f}%)", (18, y), FONT, 0.38, C_TEXT_WHITE, 1, cv2.LINE_AA)

    y += 26
    cv2.line(side, (18, y), (SIDEBAR_W - 18, y), (40, 40, 45), 1)

    # Player Position Telemetry
    y += 22
    cv2.putText(side, "PLAYER TRACKING", (18, y), FONT, 0.46, (255, 200, 0), 1, cv2.LINE_AA)
    y += 20
    track_str = "TRACKED" if pl.tracked else "SEARCHING..."
    track_col = C_AIRBORNE if pl.tracked else C_OUT_OF_BOUNDS
    cv2.putText(side, f"Status: {track_str}", (18, y), FONT, 0.42, track_col, 1, cv2.LINE_AA)

    y += 18
    box_str = "INSIDE ARENA" if pl.in_box else "OUT OF BOUNDS"
    box_col = C_AIRBORNE if pl.in_box else C_OUT_OF_BOUNDS
    cv2.putText(side, f"Arena: {box_str}", (18, y), FONT, 0.42, box_col, 1, cv2.LINE_AA)

    # Tile Coordinates
    tile_c = tile_of_grid(pl.x_m, pl.y_m, game.w_m, game.h_m, game.cols, game.rows)
    macro_c = micro_to_macro(tile_c[0], tile_c[1]) if tile_c else None
    y += 20
    if tile_c and macro_c:
        cv2.putText(side, f"Micro Tile: ({tile_c[0]}, {tile_c[1]}) [0..8]", (18, y), FONT, 0.40, C_TEXT_WHITE, 1, cv2.LINE_AA)
        y += 18
        cv2.putText(side, f"Macro Zone: ({macro_c[0]}, {macro_c[1]}) [0..2]", (18, y), FONT, 0.40, C_GRID_MACRO, 1, cv2.LINE_AA)
    else:
        cv2.putText(side, "Tile: Out of Bounds", (18, y), FONT, 0.40, C_OUT_OF_BOUNDS, 1, cv2.LINE_AA)
        y += 18

    # Feet details
    y += 18
    tile_l = tile_of_grid(pl.left_m[0], pl.left_m[1], game.w_m, game.h_m, game.cols, game.rows) if pl.left_m else None
    tile_r = tile_of_grid(pl.right_m[0], pl.right_m[1], game.w_m, game.h_m, game.cols, game.rows) if pl.right_m else None
    l_str = f"L-Foot: ({tile_l[0]},{tile_l[1]})" if tile_l else "L-Foot: -"
    r_str = f"R-Foot: ({tile_r[0]},{tile_r[1]})" if tile_r else "R-Foot: -"
    cv2.putText(side, f"{l_str}  {r_str}", (18, y), FONT, 0.38, C_TEXT_GREY, 1, cv2.LINE_AA)

    # Airborne status
    y += 24
    if pl.airborne:
        cv2.rectangle(side, (18, y - 14), (SIDEBAR_W - 18, y + 10), (0, 80, 20), -1)
        cv2.putText(side, "AIRBORNE! LAVA IMMUNITY", (26, y + 3), FONT, 0.45, C_AIRBORNE, 1, cv2.LINE_AA)
    else:
        cv2.putText(side, "Ground Contact: Solid", (18, y), FONT, 0.38, C_TEXT_GREY, 1, cv2.LINE_AA)

    # Performance
    y += 40
    cv2.putText(side, f"Render: {fps:.0f} FPS | Pose: {pose_fps:.0f} FPS", (18, y), FONT, 0.36, C_TEXT_GREY, 1, cv2.LINE_AA)
    y += 18
    mode_desc = "KEYBOARD (WASD+SPACE)" if keyboard_mode else "CAMERA TRACKING"
    cv2.putText(side, f"Source: {mode_desc}", (18, y), FONT, 0.36, C_TEXT_GREY, 1, cv2.LINE_AA)

    # Controls footer
    cv2.line(side, (18, MAP_SIZE - 70), (SIDEBAR_W - 18, MAP_SIZE - 70), (40, 40, 45), 1)
    cv2.putText(side, "SPACE: Start/Jump  |  V: Switch View", (18, MAP_SIZE - 48), FONT, 0.36, C_TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(side, "P: Pattern  |  K: Keyboard  |  Q: Quit", (18, MAP_SIZE - 30), FONT, 0.36, C_TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(side, "Hands Up: Start/Restart", (18, MAP_SIZE - 12), FONT, 0.36, (0, 200, 255), 1, cv2.LINE_AA)

    return side


def render_camera_overlay(
    frame: np.ndarray,
    mapper: FloorMapper,
    game: FloorLavaGame,
    pl: PlayerState,
    pf: PoseFrame,
    debug: dict,
) -> np.ndarray:
    """Project the 9x9 floor grid and glowing lava tiles onto the live camera image."""
    img = frame.copy()
    h_vid, w_vid = img.shape[:2]

    # Create transparent overlay for lava tiles
    overlay = img.copy()

    pulse = 0.75 + 0.25 * math.sin(time.monotonic() * 9.0)
    for c, r in game.hazard_tiles:
        corners_m = tile_corners_m(c, r, game.w_m, game.h_m, game.cols, game.rows)
        corners_px = np.array([mapper.to_pixels(x, y) for x, y in corners_m], dtype=np.int32)
        if game.phase == "WARNING":
            flash = 0.5 + 0.5 * math.sin(time.monotonic() * 12.0)
            col = (int(C_WARN_TILE[0] * flash), int(C_WARN_TILE[1] * flash), int(C_WARN_TILE[2] * flash))
        else:
            col = (int(C_LAVA_TILE[0] * pulse), int(C_LAVA_TILE[1] * pulse), min(255, int(C_LAVA_TILE[2] * pulse + 30)))
        cv2.fillPoly(overlay, [corners_px], col)

    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # Draw 9x9 micro-grid lines on video
    dx_m = game.w_m / game.cols
    dy_m = game.h_m / game.rows
    for c in range(game.cols + 1):
        x = min(game.w_m, c * dx_m)
        p1 = mapper.to_pixels(x, 0.0)
        p2 = mapper.to_pixels(x, game.h_m)
        cv2.line(img, (int(round(p1[0])), int(round(p1[1]))),
                 (int(round(p2[0])), int(round(p2[1]))), (100, 90, 80), 1, cv2.LINE_AA)

    for r in range(game.rows + 1):
        y = min(game.h_m, r * dy_m)
        p1 = mapper.to_pixels(0.0, y)
        p2 = mapper.to_pixels(game.w_m, y)
        cv2.line(img, (int(round(p1[0])), int(round(p1[1]))),
                 (int(round(p2[0])), int(round(p2[1]))), (100, 90, 80), 1, cv2.LINE_AA)

    # Draw 3x3 macro-grid lines (thicker lines)
    sub_c = game.cols // game.macro_cols
    sub_r = game.rows // game.macro_rows
    for mc in range(game.macro_cols + 1):
        x = min(game.w_m, mc * sub_c * dx_m)
        p1 = mapper.to_pixels(x, 0.0)
        p2 = mapper.to_pixels(x, game.h_m)
        cv2.line(img, (int(round(p1[0])), int(round(p1[1]))),
                 (int(round(p2[0])), int(round(p2[1]))), C_GRID_MACRO, 2, cv2.LINE_AA)

    for mr in range(game.macro_rows + 1):
        y = min(game.h_m, mr * sub_r * dy_m)
        p1 = mapper.to_pixels(0.0, y)
        p2 = mapper.to_pixels(game.w_m, y)
        cv2.line(img, (int(round(p1[0])), int(round(p1[1]))),
                 (int(round(p2[0])), int(round(p2[1]))), C_GRID_MACRO, 2, cv2.LINE_AA)

    # Draw floor boundary box
    poly = mapper.corners_px.astype(np.int32).reshape(-1, 1, 2)
    border_col = (0, 255, 0) if pl.in_box else (0, 0, 255)
    cv2.polylines(img, [poly], True, border_col, 2, cv2.LINE_AA)

    # Draw player feet on camera feed
    if pl.left_m:
        lx_px, ly_px = mapper.to_pixels(*pl.left_m)
        cv2.circle(img, (int(round(lx_px)), int(round(ly_px))), 7, C_FOOT_L, -1, cv2.LINE_AA)
        cv2.circle(img, (int(round(lx_px)), int(round(ly_px))), 9, (255, 255, 255), 1, cv2.LINE_AA)

    if pl.right_m:
        rx_px, ry_px = mapper.to_pixels(*pl.right_m)
        cv2.circle(img, (int(round(rx_px)), int(round(ry_px))), 7, C_FOOT_R, -1, cv2.LINE_AA)
        cv2.circle(img, (int(round(rx_px)), int(round(ry_px))), 9, (255, 255, 255), 1, cv2.LINE_AA)

    if debug.get("foot_px") is not None:
        fx, fy = debug["foot_px"]
        cv2.circle(img, (int(round(fx)), int(round(fy))), 9, (255, 255, 255), 2, cv2.LINE_AA)

    # Top status strip on video
    cv2.rectangle(img, (0, 0), (w_vid, 36), (15, 15, 20), -1)
    cv2.putText(img, game.banner_msg, (12, 24), FONT, 0.55, game.banner_color, 2, cv2.LINE_AA)

    return img


# ==============================================================================
# Main Entry Point & Loop
# ==============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Floor is Lava baseline test game.")
    ap.add_argument("--keyboard", action="store_true", help="no camera, WASD + SPACE jump")
    ap.add_argument("--camera", type=int, default=0, help="camera device index (default: 0)")
    ap.add_argument("--floor-w", type=float, default=1.245, help="floor width in metres")
    ap.add_argument("--floor-h", type=float, default=1.245, help="floor height in metres")
    ap.add_argument("--cols", type=int, default=9, help="micro grid cols (default: 9)")
    ap.add_argument("--rows", type=int, default=9, help="micro grid rows (default: 9)")
    ap.add_argument("--pattern", choices=PATTERNS, default="random", help="lava pattern mode")
    ap.add_argument("--recalibrate", action="store_true", help="run floor calibration first")
    args = ap.parse_args()

    w_m, h_m = args.floor_w, args.floor_h
    source = None
    mapper = None
    keyboard_mode = bool(args.keyboard)

    if not keyboard_mode:
        mapper = None if args.recalibrate else FloorMapper.load()
        if mapper is None:
            print("No saved calibration found. Launching calibration...")
            try:
                temp_source = CameraPoseSource(args.camera)
                temp_source.start()
                atexit.register(temp_source.stop)
                t0 = time.monotonic()
                while temp_source.latest_frame() is None:
                    if time.monotonic() - t0 > 8.0:
                        sys.exit("Timeout waiting for camera frame during calibration.")
                    time.sleep(0.05)
                mapper = calibrate_floor(temp_source.latest_frame, w_m, h_m)
                temp_source.stop()
            except Exception as e:
                sys.exit(f"Camera initialization failed: {e}")

            if mapper is None:
                sys.exit("Calibration cancelled.")

        w_m, h_m = mapper.w_m, mapper.h_m
        try:
            source = CameraPoseSource(args.camera, mapper=mapper)
            source.start()
            atexit.register(source.stop)
        except Exception as e:
            sys.exit(f"Failed to start camera pose source: {e}")

        # Wait for camera to warm up
        t0 = time.monotonic()
        while source.latest_frame() is None:
            if time.monotonic() - t0 > 10.0:
                sys.exit("Timeout waiting for camera frames.")
            time.sleep(0.05)

    # Initialize Game Engine
    game = FloorLavaGame(
        w_m=w_m,
        h_m=h_m,
        cols=args.cols,
        rows=args.rows,
        macro_cols=3,
        macro_rows=3,
        pattern=args.pattern,
    )

    # Keyboard player state
    kb_x, kb_y = w_m / 2.0, h_m / 2.0
    jump_until = 0.0

    # View Mode: "dual" (map + cam overlay), "map" (map only), "camera" (cam overlay only)
    view_mode = "dual" if not keyboard_mode else "map"
    last_t = time.monotonic()
    fps = 30.0

    cv2.namedWindow("Floor is Lava (Baseline Test)", cv2.WINDOW_AUTOSIZE)

    while True:
        now = time.monotonic()
        dt = min(now - last_t, 0.1)
        last_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Retrieve player telemetry
        if keyboard_mode:
            is_airborne = now < jump_until
            pl = PlayerState(
                tracked=True,
                in_box=(0.0 <= kb_x <= w_m and 0.0 <= kb_y <= h_m),
                x_m=kb_x,
                y_m=kb_y,
                left_m=(max(0.0, kb_x - 0.08), kb_y),
                right_m=(min(w_m, kb_x + 0.08), kb_y),
                airborne=is_airborne,
                arms_up=False,
            )
            cam_frame = None
            debug_info = {}
            pf = PoseFrame(tracked=True, floor_x_m=kb_x, floor_y_m=kb_y, airborne=is_airborne)
        else:
            if source.error:
                sys.exit("Camera error: " + source.error)
            pf, cam_frame, debug_info = source.snapshot()
            in_box = bool(pf.tracked and pf.floor_x_m is not None and mapper.inside(pf.floor_x_m, pf.floor_y_m))
            pl = PlayerState(
                tracked=bool(pf.tracked),
                in_box=in_box,
                x_m=float(pf.floor_x_m or (w_m / 2.0)),
                y_m=float(pf.floor_y_m or (h_m / 2.0)),
                left_m=pf.left_foot_m,
                right_m=pf.right_foot_m,
                airborne=bool(pf.airborne),
                arms_up=bool(pf.arms_up),
            )

        # Update game rules
        game.update(dt, pl)

        # Render Top-Down Map & Sidebar
        map_panel = render_top_down_map(game, pl)
        pose_fps = 0.0 if source is None else source.pose_fps
        sidebar_panel = render_sidebar(game, pl, fps, pose_fps, keyboard_mode)

        # Compose display canvas based on view mode
        if view_mode == "dual" and cam_frame is not None and mapper is not None:
            cam_overlay = render_camera_overlay(cam_frame, mapper, game, pl, pf, debug_info)
            cam_resized = cv2.resize(cam_overlay, (MAP_SIZE, MAP_SIZE))
            canvas = np.hstack([cam_resized, map_panel, sidebar_panel])
        elif view_mode == "camera" and cam_frame is not None and mapper is not None:
            cam_overlay = render_camera_overlay(cam_frame, mapper, game, pl, pf, debug_info)
            cam_resized = cv2.resize(cam_overlay, (MAP_SIZE, MAP_SIZE))
            canvas = np.hstack([cam_resized, sidebar_panel])
        else:
            canvas = np.hstack([map_panel, sidebar_panel])

        cv2.imshow("Floor is Lava (Baseline Test)", canvas)

        # Handle user input
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):  # Q or ESC
            break
        elif k == ord("r"):      # R restart
            game.reset()
        elif k == ord("v") and not keyboard_mode:  # V switch view
            modes = ["dual", "map", "camera"]
            view_mode = modes[(modes.index(view_mode) + 1) % len(modes)]
        elif k == ord("p"):      # P cycle pattern
            p_idx = (PATTERNS.index(game.pattern_mode) + 1) % len(PATTERNS)
            game.pattern_mode = PATTERNS[p_idx]
        elif k == ord("k"):      # K toggle keyboard mode
            keyboard_mode = not keyboard_mode
            if keyboard_mode and view_mode == "dual":
                view_mode = "map"
        elif k == ord(" "):      # SPACE: Start game in IDLE/GAMEOVER, or Jump in keyboard mode
            if game.phase in ("IDLE", "GAME_OVER"):
                game.start_game()
            elif keyboard_mode:
                jump_until = time.monotonic() + 0.6  # 600ms jump window

        # Keyboard WASD motion
        if keyboard_mode:
            step = 0.05
            if k == ord("a"): kb_x = max(-0.1, kb_x - step)
            if k == ord("d"): kb_x = min(w_m + 0.1, kb_x + step)
            if k == ord("w"): kb_y = min(h_m + 0.1, kb_y + step)   # Away from camera
            if k == ord("s"): kb_y = max(-0.1, kb_y - step)

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
