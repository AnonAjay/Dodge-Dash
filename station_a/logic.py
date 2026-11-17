"""Game rules and 10x10 Continuous Heatmap Lava Engine.
PURE PYTHON: No cv2, no pygame, no network imports.

Implements:
- 10x10 micro-grid resolution (18 cm x 18 cm subtiles for a 1.8m x 1.8m arena).
- Continuous cell heat H in [0.0, 1.0] with quadratic ease-in acceleration.
- Safe Island & procedural hazard wave generator (Islands, Rivers, Donut, Clusters).
- Hit / Dodge / Jump-Immunity collision detection.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum

from shared.contracts import GameState, Phase, PoseFrame

GRID_SIZE = 10           # 10x10 subtiles
DEFAULT_ARENA_M = 1.8    # 1.8m x 1.8m physical floor
SUBTILE_M = DEFAULT_ARENA_M / GRID_SIZE  # 0.18 m (18 cm) per cell

BASE_WARN_TIME_S = 2.5   # Warning countdown duration in seconds
ERUPT_HOLD_TIME_S = 1.0  # How long lava stays fully erupted before cooling
COOLDOWN_TIME_S = 0.5    # Pause between hazard waves


class WavePattern(str, Enum):
    ISLAND = "island"      # One safe sanctuary island, rest of the floor turns to lava
    RIVER = "river"        # A band of lava cuts through, sides remain safe
    DONUT = "donut"        # Outer perimeter turns to lava, center is safe
    CLUSTER = "cluster"    # Random boiling lava puddles


@dataclass
class LavaWave:
    """Tracks the active hazard wave over the 10x10 micro-grid."""
    pattern: WavePattern = WavePattern.ISLAND
    timer: float = BASE_WARN_TIME_S        # Counts down from total_warn -> 0 -> -erupt_hold
    total_warn: float = BASE_WARN_TIME_S
    erupt_duration: float = ERUPT_HOLD_TIME_S
    cooldown: float = 0.0

    # Target mask: True means this cell will turn into lava
    target_mask: list[list[bool]] = field(default_factory=lambda: [[False]*GRID_SIZE for _ in range(GRID_SIZE)])
    # Heat matrix: float 0.0 (stone) to 1.0 (active lava)
    heat: list[list[float]] = field(default_factory=lambda: [[0.0]*GRID_SIZE for _ in range(GRID_SIZE)])

    safe_center_m: tuple[float, float] = (0.9, 0.9)
    safe_radius_m: float = 0.45
    judged: bool = False                   # Ensures hit/dodge is judged once per wave


def create_empty_heat() -> list[list[float]]:
    return [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]


def cell_bounds_m(col: int, row: int, subtile_m: float = SUBTILE_M) -> tuple[float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max) in floor meters for cell (col, row)."""
    return (col * subtile_m, (col + 1) * subtile_m, row * subtile_m, (row + 1) * subtile_m)


def cells_touched(x_m: float, y_m: float, radius_m: float = 0.22,
                  grid_size: int = GRID_SIZE, subtile_m: float = SUBTILE_M) -> set[tuple[int, int]]:
    """Return {(col, row)} of all micro-cells overlapping a circle footprint."""
    touched: set[tuple[int, int]] = set()
    r2 = radius_m * radius_m + 1e-9

    for c in range(grid_size):
        x_min = c * subtile_m
        x_max = (c + 1) * subtile_m
        cx = max(x_min, min(x_m, x_max))
        dx = x_m - cx

        for r in range(grid_size):
            y_min = r * subtile_m
            y_max = (r + 1) * subtile_m
            cy = max(y_min, min(y_m, y_max))
            dy = y_m - cy

            if (dx * dx + dy * dy) <= r2:
                touched.add((c, r))

    return touched


# Backward-compatibility alias for 3x3 spec tests
def tiles_touched(x_m: float, y_m: float, radius_m: float,
                  cols: int = 3, rows: int = 3, tile_m: float = 0.6) -> set[tuple[int, int]]:
    """3x3 tile overlap function matching original contract."""
    return cells_touched(x_m, y_m, radius_m=radius_m, grid_size=cols, subtile_m=tile_m)


def generate_wave(pattern: WavePattern, player_x_m: float, player_y_m: float,
                  arena_w: float = DEFAULT_ARENA_M, arena_h: float = DEFAULT_ARENA_M,
                  warn_time_s: float = BASE_WARN_TIME_S) -> LavaWave:
    """Generate a new lava wave mask with guaranteed solvability."""
    mask = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
    safe_radius = 0.42

    if pattern == WavePattern.ISLAND:
        # Spawn safe island AWAY from player's current position to force movement
        # If player is on left (x < 0.9), place island on right (x > 1.2), etc.
        target_x = random.uniform(1.1, 1.5) if player_x_m < (arena_w / 2) else random.uniform(0.3, 0.7)
        target_y = random.uniform(1.1, 1.5) if player_y_m < (arena_h / 2) else random.uniform(0.3, 0.7)
        safe_center = (round(target_x, 2), round(target_y, 2))

        for c in range(GRID_SIZE):
            for r in range(GRID_SIZE):
                cell_cx = (c + 0.5) * SUBTILE_M
                cell_cy = (r + 0.5) * SUBTILE_M
                dist = math.hypot(cell_cx - safe_center[0], cell_cy - safe_center[1])
                # All cells outside the safe island become lava
                if dist > safe_radius:
                    mask[c][r] = True

    elif pattern == WavePattern.DONUT:
        # Center is safe, perimeter is lava
        safe_center = (arena_w / 2.0, arena_h / 2.0)
        safe_radius = 0.55
        for c in range(GRID_SIZE):
            for r in range(GRID_SIZE):
                cell_cx = (c + 0.5) * SUBTILE_M
                cell_cy = (r + 0.5) * SUBTILE_M
                dist = math.hypot(cell_cx - safe_center[0], cell_cy - safe_center[1])
                if dist > safe_radius:
                    mask[c][r] = True

    elif pattern == WavePattern.RIVER:
        # A 3-cell band of lava runs across horizontal or vertical
        safe_center = (player_x_m, player_y_m)
        is_horizontal = random.choice([True, False])
        band_start = random.randint(3, 5)
        for c in range(GRID_SIZE):
            for r in range(GRID_SIZE):
                idx = r if is_horizontal else c
                if band_start <= idx <= band_start + 2:
                    mask[c][r] = True

    else:  # CLUSTER
        safe_center = (arena_w / 2.0, arena_h / 2.0)
        # Random boiling puddles (40-60% coverage)
        for c in range(GRID_SIZE):
            for r in range(GRID_SIZE):
                cell_cx = (c + 0.5) * SUBTILE_M
                cell_cy = (r + 0.5) * SUBTILE_M
                # Keep immediate sanctuary around player safe at start of wave
                dist = math.hypot(cell_cx - player_x_m, cell_cy - player_y_m)
                if dist > 0.35 and random.random() < 0.55:
                    mask[c][r] = True

    return LavaWave(
        pattern=pattern,
        timer=warn_time_s,
        total_warn=warn_time_s,
        erupt_duration=ERUPT_HOLD_TIME_S,
        cooldown=0.0,
        target_mask=mask,
        heat=[[0.0] * GRID_SIZE for _ in range(GRID_SIZE)],
        safe_center_m=safe_center,
        safe_radius_m=safe_radius,
        judged=False,
    )


class LavaEngine:
    """Manages continuous heatmap updates, hazard wave progression, and hit detection.
    Features 3-Stage Tournament Rounds:
    - Round 1 (0-20s): Slow and steady, safe circular islands, 3.0s warning.
    - Round 2 (20-40s): Chaotic geometric shapes (rivers, clusters), 2.0s warning, 1.5x multiplier.
    - Round 3 (40-60s): Final Round rapid fire (1.6s -> 1.1s warning), mixed shapes, 2.0x multiplier.
    """

    def __init__(self, duration_s: float = 60.0):
        self.duration_s = duration_s
        self.state = GameState(phase=Phase.PLAYING, time_left_s=duration_s, speed=1.0)
        self.wave: LavaWave | None = None
        self.last_banner_msg = "ROUND 1 ... BEGIN!"
        self.last_banner_color = (0, 255, 120)
        self.banner_timer = 2.0

        # 3-Stage Round System
        self.current_round: int = 1
        self.round_multiplier: float = 1.0
        self.round_banner_title: str = "ROUND 1"
        self.round_banner_subtitle: str = "EASY WARMUP ... BEGIN!"
        self.round_banner_color: tuple[int, int, int] = (0, 255, 120)
        self.round_banner_timer: float = 2.0

    def start_game(self, player_id: str = "P001") -> None:
        self.state = GameState(
            phase=Phase.PLAYING,
            player_id=player_id,
            time_left_s=self.duration_s,
            score=0,
            streak=0,
            best_streak=0,
            dodges=0,
            hits=0,
            speed=1.0,
        )
        self.wave = None
        self.current_round = 1
        self.round_multiplier = 1.0
        self.round_banner_title = "ROUND 1"
        self.round_banner_subtitle = "EASY WARMUP ... BEGIN!"
        self.round_banner_color = (0, 255, 120)
        self.round_banner_timer = 2.0
        self.last_banner_msg = "ROUND 1 ... BEGIN!"
        self.last_banner_color = (0, 255, 120)
        self.banner_timer = 2.0

    def update(self, pose: PoseFrame, dt: float) -> tuple[GameState, LavaWave | None]:
        dt = min(max(0.0, dt), 0.1)

        if self.state.phase != Phase.PLAYING:
            return self.state, self.wave

        # Tick round announcement banner timer
        if self.round_banner_timer > 0:
            self.round_banner_timer = max(0.0, self.round_banner_timer - dt)

        if not pose.tracked:
            # Player lost: hold game progress gracefully
            return self.state, self.wave

        # Round countdown
        self.state.time_left_s = max(0.0, self.state.time_left_s - dt)
        if self.state.time_left_s <= 0.0:
            self.state.phase = Phase.RESULT
            self.wave = None
            return self.state, None

        # ----------------------------------------------------------------------
        # 3-Stage Round Progression & Difficulty Escalation
        # ----------------------------------------------------------------------
        elapsed = max(0.0, self.duration_s - self.state.time_left_s)
        if elapsed < 20.0:
            target_round = 1
            round_mult = 1.0
            allowed_patterns = [WavePattern.ISLAND, WavePattern.DONUT]
            base_warn = 3.0  # Slow and steady 3.0s warning
        elif elapsed < 40.0:
            target_round = 2
            round_mult = 1.5
            allowed_patterns = [WavePattern.RIVER, WavePattern.CLUSTER]  # Random shapes, no circles
            base_warn = 2.0  # Faster breakdown 2.0s warning
        else:
            target_round = 3
            round_mult = 2.0
            allowed_patterns = list(WavePattern)  # Fast mix of all shapes
            # Rapid fire: warning decreases from 1.6s down to 1.1s
            r3_prog = min(1.0, max(0.0, (elapsed - 40.0) / 20.0))
            base_warn = max(1.1, 1.6 - r3_prog * 0.5)

        # Trigger Street Fighter style round transition
        if target_round != self.current_round:
            self.current_round = target_round
            self.round_multiplier = round_mult
            if target_round == 2:
                self.round_banner_title = "ROUND 2"
                self.round_banner_subtitle = "CHAOS SHAPES ... BEGIN!"
                self.round_banner_color = (255, 180, 0)
                self.round_banner_timer = 2.0
            elif target_round == 3:
                self.round_banner_title = "FINAL ROUND"
                self.round_banner_subtitle = "SUDDEN SURGE ... FIGHT!"
                self.round_banner_color = (255, 50, 50)
                self.round_banner_timer = 2.5

        # Determine player position in meters
        px = pose.floor_x_m if pose.floor_x_m is not None else 0.9
        py = pose.floor_y_m if pose.floor_y_m is not None else 0.9

        # Identify cells touched by independent dual feet (or center fallback)
        if pose.left_foot_m is not None and pose.right_foot_m is not None:
            left_touched = cells_touched(pose.left_foot_m[0], pose.left_foot_m[1], radius_m=0.10)
            right_touched = cells_touched(pose.right_foot_m[0], pose.right_foot_m[1], radius_m=0.10)
            touched = left_touched | right_touched
        elif pose.left_foot_m is not None:
            touched = cells_touched(pose.left_foot_m[0], pose.left_foot_m[1], radius_m=0.12)
        elif pose.right_foot_m is not None:
            touched = cells_touched(pose.right_foot_m[0], pose.right_foot_m[1], radius_m=0.12)
        else:
            touched = cells_touched(px, py, radius_m=0.18)

        self.state.occupied_tiles = touched

        # Wave Progression
        if self.wave is None:
            pattern = random.choice(allowed_patterns)
            self.wave = generate_wave(pattern, px, py, warn_time_s=base_warn)
            self.last_banner_msg = f"WARNING! {pattern.value.upper()} HEATING UP!"
            self.last_banner_color = (255, 180, 0)
            self.banner_timer = 1.5

        # Update active wave
        wave = self.wave

        if wave.cooldown > 0:
            wave.cooldown = max(0.0, wave.cooldown - dt)
            if wave.cooldown <= 0:
                self.wave = None
            return self.state, wave

        wave.timer -= dt

        if wave.timer > 0:
            # WARMING PHASE: Heat accelerates from 0.0 -> 1.0 via Quadratic Ease-In
            progress = max(0.0, min(1.0, (wave.total_warn - wave.timer) / wave.total_warn))
            current_heat = progress * progress  # ease-in curve

            for c in range(GRID_SIZE):
                for r in range(GRID_SIZE):
                    if wave.target_mask[c][r]:
                        wave.heat[c][r] = current_heat
                    else:
                        wave.heat[c][r] = 0.0

        elif wave.timer > -wave.erupt_duration:
            # ERUPTION PHASE: Heat is 1.0 (BOILING LAVA)
            for c in range(GRID_SIZE):
                for r in range(GRID_SIZE):
                    if wave.target_mask[c][r]:
                        wave.heat[c][r] = 1.0

            # Judge Hit vs Dodge at impact moment
            if not wave.judged:
                wave.judged = True
                is_on_lava = any(wave.target_mask[c][r] for (c, r) in touched)

                if is_on_lava and not pose.airborne:
                    # HIT: Caught in lava while grounded
                    self.state.hits += 1
                    self.state.streak = 0
                    self.last_banner_msg = "BURNT! HIT BY LAVA!"
                    self.last_banner_color = (255, 40, 40)
                    self.banner_timer = 1.2
                else:
                    # DODGE: Jumped over or stood on safe ground
                    self.state.dodges += 1
                    self.state.streak += 1
                    self.state.best_streak = max(self.state.best_streak, self.state.streak)

                    streak_mult = min(2.0, 1.0 + (self.state.streak // 5) * 0.1)
                    score_gain = int(10 * streak_mult * self.round_multiplier)
                    self.state.score += score_gain
                    self.state.speed = min(2.0, 1.0 + self.state.dodges * 0.04)

                    if is_on_lava and pose.airborne:
                        self.last_banner_msg = f"JUMP IMMUNITY! +{score_gain}"
                        self.last_banner_color = (0, 240, 255)
                    else:
                        self.last_banner_msg = f"SAFE! DODGED! +{score_gain}"
                        self.last_banner_color = (0, 255, 120)
                    self.banner_timer = 1.2

        else:
            # Wave over -> cooldown phase
            wave.cooldown = COOLDOWN_TIME_S
            for c in range(GRID_SIZE):
                for r in range(GRID_SIZE):
                    wave.heat[c][r] = 0.0

        return self.state, wave
