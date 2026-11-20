"""Game rules and 10x10 Continuous Heatmap Lava Engine for Station A.
PURE PYTHON: No cv2, no pygame, no network imports.

Implements:
- 10x10 micro-grid resolution (18 cm x 18 cm subtiles for 1.8m x 1.8m arena).
- Tournament Lifecycle: Foot Lock -> Resting (7.5s) -> Round 1 (30s) -> Resting (7.5s)
  -> Round 2 (30s) -> Resting (7.5s) -> Round 3 (30s) -> Result.
- 3 Unique Round Archetypes:
  * Round 1: Rivers Only (5.0s pacing, horizontal/vertical lava bands).
  * Round 2: Circles Only (4.0s pacing, random safe islands & donuts).
  * Round 3: Bridges & Stepping Stones (3.0s pacing, narrow walkways, NO rivers/circles).
- Anti-Camping Probability Mapping: 70% probability hazard targets player's current position.
- Scoring: +10 * streak_multiplier per dodge; -2 points penalty per hit (score >= 0).
  Streak multipliers: <3 (1.0x), 3-5 (1.2x), 6-8 (1.5x), 9+ (2.0x).
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

ROUND_DURATION_S = 60.0  # 60.0s active play per round
DEFAULT_ROUND_DURATION_S = 60.0  # Alias for backward-compatibility
REST_DURATION_S = 7.5    # 7.5s resting interval between rounds
TOTAL_ROUNDS = 3
FEET_LOCK_REQUIRED_S = 0.5
ERUPT_HOLD_TIME_S = 1.0  # How long lava stays fully erupted before cooling
COOLDOWN_TIME_S = 0.4    # Pause between hazard waves


class WavePattern(str, Enum):
    RIVER = "river"        # Round 1: Bands of lava cutting across, sides remain safe
    ISLAND = "island"      # Round 2: One safe circular island, rest turns to lava
    DONUT = "donut"        # Round 2: Circular donut ring
    BRIDGES = "bridges"    # Round 3: Narrow crossing walkways and stepping stones (NO circles/rivers)
    CLUSTER = "cluster"    # Random boiling puddles


class TournamentPhase(str, Enum):
    WAITING_FOR_PLAYER = "waiting" # Looking for feet in arena
    RESTING = "resting"            # 7.5s intermission (showing Round X, countdown, cool floor)
    ROUND_ACTIVE = "active"        # 30.0s active hazard dodging
    COMPLETED = "completed"        # Match finished


@dataclass
class LavaWave:
    """Tracks the active hazard wave over the 10x10 micro-grid."""
    pattern: WavePattern = WavePattern.RIVER
    timer: float = 4.0                     # Counts down from total_warn -> 0 -> -erupt_hold
    total_warn: float = 4.0
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


def get_streak_multiplier(streak: int) -> float:
    """Return score multiplier based on active dodge streak:
    - 0 to 2: 1.0x
    - 3 to 5: 1.2x
    - 6 to 8: 1.5x
    - 9+: 2.0x
    """
    if streak < 3:
        return 1.0
    elif streak <= 5:
        return 1.2
    elif streak <= 8:
        return 1.5
    else:
        return 2.0


def generate_wave(pattern: WavePattern, player_x_m: float, player_y_m: float,
                  arena_w: float = DEFAULT_ARENA_M, arena_h: float = DEFAULT_ARENA_M,
                  warn_time_s: float = 4.0) -> LavaWave:
    """Generate a hazard wave with anti-camping probability mapping:
    70% probability that hazard specifically targets the player's current coordinate.
    """
    mask = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
    safe_center = (arena_w / 2.0, arena_h / 2.0)
    safe_radius = 0.40

    col_p = max(0, min(GRID_SIZE - 1, int(player_x_m / SUBTILE_M)))
    row_p = max(0, min(GRID_SIZE - 1, int(player_y_m / SUBTILE_M)))

    target_player = random.random() < 0.70

    if pattern == WavePattern.RIVER:
        # Round 1: River of lava cutting across.
        # Anti-camping: 70% chance the river cuts directly across the player's position!
        is_horizontal = random.choice([True, False])
        if is_horizontal:
            band_center = row_p if target_player else random.randint(2, GRID_SIZE - 3)
            band_rows = {max(0, min(GRID_SIZE - 1, r)) for r in (band_center - 1, band_center, band_center + 1)}
            for c in range(GRID_SIZE):
                for r in band_rows:
                    mask[c][r] = True
            safe_center = (player_x_m, 0.2 if band_center > 4 else 1.6)
        else:
            band_center = col_p if target_player else random.randint(2, GRID_SIZE - 3)
            band_cols = {max(0, min(GRID_SIZE - 1, c)) for c in (band_center - 1, band_center, band_center + 1)}
            for c in band_cols:
                for r in range(GRID_SIZE):
                    mask[c][r] = True
            safe_center = (0.2 if band_center > 4 else 1.6, player_y_m)

    elif pattern == WavePattern.ISLAND:
        # Round 2: Circular safe island.
        # Anti-camping: 70% chance island spawns far away from the player's current quadrant!
        if target_player:
            target_x = random.uniform(1.2, 1.5) if player_x_m < (arena_w / 2) else random.uniform(0.3, 0.6)
            target_y = random.uniform(1.2, 1.5) if player_y_m < (arena_h / 2) else random.uniform(0.3, 0.6)
        else:
            target_x = random.uniform(0.5, arena_w - 0.5)
            target_y = random.uniform(0.5, arena_h - 0.5)

        safe_center = (round(target_x, 2), round(target_y, 2))
        safe_radius = 0.38

        for c in range(GRID_SIZE):
            for r in range(GRID_SIZE):
                cell_cx = (c + 0.5) * SUBTILE_M
                cell_cy = (r + 0.5) * SUBTILE_M
                dist = math.hypot(cell_cx - safe_center[0], cell_cy - safe_center[1])
                if dist > safe_radius:
                    mask[c][r] = True

    elif pattern == WavePattern.DONUT:
        # Round 2: Ring geometry.
        # If player is in center, center turns to lava and ring is safe.
        dist_to_center = math.hypot(player_x_m - (arena_w / 2), player_y_m - (arena_h / 2))
        center_is_lava = (dist_to_center < 0.45) if target_player else random.choice([True, False])

        safe_center = (arena_w / 2.0, arena_h / 2.0)
        safe_radius = 0.50

        for c in range(GRID_SIZE):
            for r in range(GRID_SIZE):
                cell_cx = (c + 0.5) * SUBTILE_M
                cell_cy = (r + 0.5) * SUBTILE_M
                dist = math.hypot(cell_cx - safe_center[0], cell_cy - safe_center[1])
                if center_is_lava:
                    # Center circle is lava; outer perimeter is safe
                    if dist <= safe_radius:
                        mask[c][r] = True
                else:
                    # Center circle is safe; outer perimeter is lava
                    if dist > safe_radius:
                        mask[c][r] = True

    elif pattern == WavePattern.BRIDGES:
        # Round 3: Bridges & Stepping Stones (NO circles, NO full rivers).
        # Mostly boiling lava (75-80% lava) with narrow 1-cell walkways.
        for c in range(GRID_SIZE):
            for r in range(GRID_SIZE):
                mask[c][r] = True

        # Create crossing bridges (1 horizontal bridge, 1 vertical bridge, and stepping stones)
        bridge_row = random.randint(2, 7)
        bridge_col = random.randint(2, 7)
        for c in range(GRID_SIZE):
            mask[c][bridge_row] = False
        for r in range(GRID_SIZE):
            mask[bridge_col][r] = False

        # Additional stepping stones
        for c in range(1, GRID_SIZE - 1, 2):
            for r in range(1, GRID_SIZE - 1, 2):
                mask[c][r] = False

        # Anti-camping: 70% probability player's exact tile is forced to be lava!
        if target_player:
            mask[col_p][row_p] = True

    else:  # CLUSTER
        for c in range(GRID_SIZE):
            for r in range(GRID_SIZE):
                dist = math.hypot((c + 0.5) * SUBTILE_M - player_x_m, (r + 0.5) * SUBTILE_M - player_y_m)
                if dist < 0.35:
                    mask[c][r] = target_player
                else:
                    mask[c][r] = (random.random() < 0.55)

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
    """Manages tournament lifecycle, 3 round archetypes, resting intermissions,
    anti-camping probability waves, and hit/dodge scoring.
    """

    def __init__(self, duration_s: float = ROUND_DURATION_S,
                 arena_w_m: float = DEFAULT_ARENA_M, arena_h_m: float = DEFAULT_ARENA_M):
        self.round_duration_s = duration_s
        self.duration_s = duration_s  # Alias for backward-compatibility
        self.arena_w_m = arena_w_m
        self.arena_h_m = arena_h_m
        self.state = GameState(phase=Phase.PLAYING, time_left_s=duration_s, speed=1.0)
        self.wave: LavaWave | None = None

        # Tournament State Machine
        self.tournament_phase: TournamentPhase = TournamentPhase.WAITING_FOR_PLAYER
        self.current_round: int = 1
        self.rest_timer: float = REST_DURATION_S
        self.round_timer: float = duration_s
        self.feet_lock_timer: float = 0.0

        # Presentation Banners
        self.round_banner_title: str = "ROUND 1"
        self.round_banner_subtitle: str = "STEP ONTO ARENA TO BEGIN"
        self.round_banner_color: tuple[int, int, int] = (0, 255, 120)
        self.round_banner_timer: float = 2.0
        self.last_banner_msg: str = "STEP ONTO ARENA TO BEGIN"
        self.last_banner_color: tuple[int, int, int] = (0, 255, 120)
        self.banner_timer: float = 2.0

        # Multipliers
        self.round_multiplier: float = 1.0
        self.player_name: str = ""

    def reset_to_idle(self) -> None:
        """Reset game engine to Phase.IDLE (Start Menu / Attract Mode)."""
        self.state.phase = Phase.IDLE
        self.wave = None
        self.tournament_phase = TournamentPhase.WAITING_FOR_PLAYER
        self.feet_lock_timer = 0.0

    def start_game(self, player_id: str = "P001", player_name: str = "") -> None:
        self.player_name = player_name or player_id
        self.state = GameState(
            phase=Phase.PLAYING,
            player_id=player_id,
            time_left_s=self.round_duration_s,
            score=0,
            streak=0,
            best_streak=0,
            dodges=0,
            hits=0,
            speed=1.0,
        )
        self.wave = None
        self.tournament_phase = TournamentPhase.WAITING_FOR_PLAYER
        self.current_round = 1
        self.rest_timer = REST_DURATION_S
        self.round_timer = self.round_duration_s
        self.feet_lock_timer = 0.0
        self.round_multiplier = 1.0

        self.round_banner_title = "ROUND 1"
        self.round_banner_subtitle = f"READY {self.player_name.upper()}!"
        self.round_banner_color = (0, 255, 120)
        self.round_banner_timer = 2.0
        self.last_banner_msg = "SEARCHING FOR FEET... STEP ONTO ARENA"
        self.last_banner_color = (255, 180, 0)
        self.banner_timer = 2.0

    def update(self, pose: PoseFrame, dt: float) -> tuple[GameState, LavaWave | None]:
        dt = min(max(0.0, dt), 0.1)

        if self.state.phase != Phase.PLAYING:
            return self.state, self.wave

        if self.round_banner_timer > 0:
            self.round_banner_timer = max(0.0, self.round_banner_timer - dt)
        if self.banner_timer > 0:
            self.banner_timer = max(0.0, self.banner_timer - dt)

        # Player floor position
        px = pose.floor_x_m if pose.floor_x_m is not None else 0.9
        py = pose.floor_y_m if pose.floor_y_m is not None else 0.9

        # Identify cells touched by independent feet
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

        # ======================================================================
        # 1. PHASE: WAITING FOR PLAYER LOCK
        # ======================================================================
        if self.tournament_phase == TournamentPhase.WAITING_FOR_PLAYER:
            in_arena = (0.0 <= px <= self.arena_w_m and 0.0 <= py <= self.arena_h_m)
            if pose.tracked and in_arena:
                self.feet_lock_timer += dt
                self.last_banner_msg = "FEET DETECTED! HOLD STEADY..."
                self.last_banner_color = (0, 240, 255)
                self.banner_timer = 0.5

                if self.feet_lock_timer >= FEET_LOCK_REQUIRED_S:
                    # Feet locked! Transition to Resting Phase 1
                    self.tournament_phase = TournamentPhase.RESTING
                    self.current_round = 1
                    self.rest_timer = REST_DURATION_S
                    self.round_banner_title = "ROUND 1"
                    self.round_banner_subtitle = "REST & PREPARE"
                    self.round_banner_color = (0, 255, 120)
                    self.round_banner_timer = 2.0
                    self.last_banner_msg = "FEET LOCKED! ROUND 1 GET READY"
                    self.last_banner_color = (0, 255, 120)
                    self.banner_timer = 1.5
            else:
                self.feet_lock_timer = 0.0
                self.last_banner_msg = "SEARCHING FOR FEET... STEP ONTO ARENA"
                self.last_banner_color = (255, 180, 0)
                self.banner_timer = 0.5

            self.state.time_left_s = self.round_duration_s
            return self.state, None

        # ======================================================================
        # 2. PHASE: RESTING INTERMISSION (7.5s)
        # ======================================================================
        if self.tournament_phase == TournamentPhase.RESTING:
            self.rest_timer -= dt
            self.state.time_left_s = self.round_duration_s
            self.wave = None  # 100% Cool stone floor during resting

            # Center broadcast countdown
            if self.rest_timer > 1.5:
                countdown_s = int(math.ceil(self.rest_timer - 1.5))
                round_name = f"ROUND {self.current_round}" if self.current_round < 3 else "FINAL ROUND"
                self.round_banner_title = round_name
                self.round_banner_subtitle = f"STARTING IN {countdown_s}..."
                self.round_banner_timer = 1.0
            else:
                self.round_banner_title = "START!"
                self.round_banner_subtitle = "DODGE THE LAVA!"
                self.round_banner_timer = 1.5

            if self.rest_timer <= 1e-4:
                # Intermission over -> start active round!
                self.tournament_phase = TournamentPhase.ROUND_ACTIVE
                self.round_timer = self.round_duration_s
                self.wave = None

            return self.state, None

        # ======================================================================
        # 3. PHASE: ACTIVE ROUND DODGING (30.0s)
        # ======================================================================
        if self.tournament_phase == TournamentPhase.ROUND_ACTIVE:
            if not pose.tracked:
                # Gracefully pause countdown if player lost
                return self.state, self.wave

            self.round_timer = max(0.0, self.round_timer - dt)
            self.state.time_left_s = self.round_timer

            # Round configuration
            if self.current_round == 1:
                # Round 1: Rivers Only (5.0s pacing)
                warn_time = 4.0
                allowed_patterns = [WavePattern.RIVER]
                self.round_multiplier = 1.0
            elif self.current_round == 2:
                # Round 2: Circles Only (4.0s pacing)
                warn_time = 3.0
                allowed_patterns = [WavePattern.ISLAND, WavePattern.DONUT]
                self.round_multiplier = 1.2
            else:
                # Round 3: Bridges & Stepping Stones (3.0s pacing, NO rivers, NO circles)
                warn_time = 2.0
                allowed_patterns = [WavePattern.BRIDGES]
                self.round_multiplier = 1.5

            # Wave Progression
            if self.wave is None:
                pattern = random.choice(allowed_patterns)
                self.wave = generate_wave(pattern, px, py, warn_time_s=warn_time)
                self.last_banner_msg = f"WARNING! {pattern.value.upper()} HEATING UP!"
                self.last_banner_color = (255, 180, 0)
                self.banner_timer = 1.2

            wave = self.wave

            # Wave Cooldown
            if wave.cooldown > 0:
                wave.cooldown = max(0.0, wave.cooldown - dt)
                if wave.cooldown <= 0:
                    self.wave = None
                return self.state, wave

            wave.timer -= dt

            if wave.timer > 0:
                # Warming Phase: Quadratic Ease-In
                progress = max(0.0, min(1.0, (wave.total_warn - wave.timer) / wave.total_warn))
                current_heat = progress * progress
                for c in range(GRID_SIZE):
                    for r in range(GRID_SIZE):
                        wave.heat[c][r] = current_heat if wave.target_mask[c][r] else 0.0

            elif wave.timer > -wave.erupt_duration:
                # Eruption Phase: Boiling Lava
                for c in range(GRID_SIZE):
                    for r in range(GRID_SIZE):
                        if wave.target_mask[c][r]:
                            wave.heat[c][r] = 1.0

                # Judge Hit vs Dodge at impact moment
                if not wave.judged:
                    wave.judged = True
                    is_on_lava = any(wave.target_mask[c][r] for (c, r) in touched)

                    if is_on_lava and not pose.airborne:
                        # HIT: -2 penalty, score cannot go below 0, streak reset
                        self.state.hits += 1
                        self.state.streak = 0
                        self.state.score = max(0, self.state.score - 2)
                        self.last_banner_msg = "BURNT! -2 PTS"
                        self.last_banner_color = (255, 40, 40)
                        self.banner_timer = 1.2
                    else:
                        # DODGE / JUMP IMMUNITY: +10 * streak multiplier
                        self.state.dodges += 1
                        self.state.streak += 1
                        self.state.best_streak = max(self.state.best_streak, self.state.streak)

                        mult = get_streak_multiplier(self.state.streak)
                        gain = int(round(10 * mult))
                        self.state.score += gain

                        if is_on_lava and pose.airborne:
                            self.last_banner_msg = f"JUMP IMMUNITY! +{gain} (x{mult:.1f})"
                            self.last_banner_color = (0, 240, 255)
                        else:
                            self.last_banner_msg = f"SAFE! DODGED! +{gain} (x{mult:.1f})"
                            self.last_banner_color = (0, 255, 120)
                        self.banner_timer = 1.2

            else:
                # Wave finished -> cooldown
                wave.cooldown = COOLDOWN_TIME_S
                for c in range(GRID_SIZE):
                    for r in range(GRID_SIZE):
                        wave.heat[c][r] = 0.0

            # Round Timer Finished Check
            if self.round_timer <= 0.0:
                if self.current_round < TOTAL_ROUNDS:
                    # Transition to next Resting Phase!
                    self.current_round += 1
                    self.tournament_phase = TournamentPhase.RESTING
                    self.rest_timer = REST_DURATION_S
                    self.wave = None
                    round_name = f"ROUND {self.current_round}" if self.current_round < 3 else "FINAL ROUND"
                    self.round_banner_title = round_name
                    self.round_banner_subtitle = "REST & RESET POSITION"
                    self.round_banner_color = (255, 180, 0) if self.current_round == 2 else (255, 50, 50)
                    self.round_banner_timer = 2.5
                else:
                    # All 3 rounds completed -> Result!
                    self.tournament_phase = TournamentPhase.COMPLETED
                    self.state.phase = Phase.RESULT
                    self.wave = None

            return self.state, self.wave

        return self.state, self.wave
