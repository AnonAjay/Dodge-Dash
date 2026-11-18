"""Station B (Wall Shape) Pure Game Logic.
PURE PYTHON: No cv2, no pygame, no network imports.

Implements:
- 3D oncoming Wall tracking along the floor depth plane (1.8m -> 0.0m).
- Collision slice judgment window at the player's ground depth position.
- Real-time pose matching against calibrated templates (T-Pose, Star, Cactus, etc.).
- Scoring, streak multipliers, and Phase lifecycle.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from pose_core.pose_matcher import (
    DEFAULT_TEMPLATES,
    JointComparison,
    PoseMatchResult,
    PoseTemplate,
    evaluate_pose,
)
from shared.contracts import GameState, Phase, PoseFrame

ARENA_DEPTH_M = 1.8         # Calibrated arena depth (far edge = 1.8m, near edge = 0.0m)
ARENA_WIDTH_M = 1.8
BASE_WALL_SPEED_M_S = 0.6   # 1.8m / 0.6 m/s = 3.0 seconds travel time
COLLISION_TOLERANCE_M = 0.15 # +/- 15 cm slice for wall pass-through
COOLDOWN_TIME_S = 0.8       # Pause before spawning next oncoming wall
PASS_THRESHOLD_PCT = 75.0   # 75% match required to pass cleanly


class WallState(str, Enum):
    APPROACHING = "approaching"   # Gliding from back to player
    COLLIDING = "colliding"       # Inside player collision window
    PASSED = "passed"             # Successfully matched pose
    HIT = "hit"                   # Failed pose match (smashed)
    CLEARED = "cleared"           # Past player and near edge


@dataclass
class Wall:
    """An oncoming 3D wall with a cutout pose silhouette."""
    template: PoseTemplate
    y_m: float = ARENA_DEPTH_M               # Far edge (meters)
    speed_m_s: float = BASE_WALL_SPEED_M_S
    state: WallState = WallState.APPROACHING
    judged: bool = False
    head_aligned: bool = False
    match_pct: float = 0.0
    score_earned: int = 0
    feedback_hint: str = ""
    joint_results: dict[str, JointComparison] = field(default_factory=dict)


class WallShapeEngine:
    """Manages oncoming wall simulation, collision detection, and scoring for Station B."""

    def __init__(self, duration_s: float = 60.0, templates: Sequence[PoseTemplate] | None = None):
        self.duration_s = duration_s
        self.templates = list(templates) if templates is not None else DEFAULT_TEMPLATES
        self.state = GameState(phase=Phase.PLAYING, time_left_s=duration_s, speed=1.0)
        self.active_wall: Wall | None = None
        self.cooldown_timer: float = 0.5
        self.last_banner_msg: str = "GET READY! MATCH THE WALL!"
        self.last_banner_color: tuple[int, int, int] = (0, 240, 255)
        self.banner_timer: float = 2.0
        self.walls_passed: int = 0
        self.walls_hit: int = 0
        self.wall_index: int = 0

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
        self.active_wall = None
        self.cooldown_timer = 0.5
        self.walls_passed = 0
        self.walls_hit = 0
        self.wall_index = 0
        self.last_banner_msg = "ROUND 1 ... BEGIN!"
        self.last_banner_color = (0, 255, 120)
        self.banner_timer = 2.0

    def spawn_wall(self) -> Wall:
        """Create a new oncoming wall with a selected pose template."""
        template = self.templates[self.wall_index % len(self.templates)]
        self.wall_index += 1
        speed = BASE_WALL_SPEED_M_S * self.state.speed
        wall = Wall(
            template=template,
            y_m=ARENA_DEPTH_M,
            speed_m_s=speed,
            feedback_hint=template.cue,
        )
        self.last_banner_msg = f"INCOMING: {template.name.upper()}!"
        self.last_banner_color = (0, 220, 255)
        self.banner_timer = 1.5
        return wall

    def update(
        self,
        pose: PoseFrame,
        dt: float,
        player_angles: dict[str, float | None] | None = None,
        player_head_norm: tuple[float, float] | None = None,
    ) -> tuple[GameState, Wall | None]:
        dt = min(max(0.0, dt), 0.1)

        if self.state.phase != Phase.PLAYING:
            return self.state, self.active_wall

        if self.banner_timer > 0:
            self.banner_timer = max(0.0, self.banner_timer - dt)

        # Countdown round timer
        self.state.time_left_s = max(0.0, self.state.time_left_s - dt)
        if self.state.time_left_s <= 0.0:
            self.state.phase = Phase.RESULT
            self.active_wall = None
            return self.state, None

        # Check head alignment (Target is center 0.50, +/- 0.09 lateral)
        head_aligned = True
        if player_head_norm is not None:
            hx, hy = player_head_norm
            head_aligned = abs(hx - 0.50) <= 0.09 and (0.10 <= hy <= 0.40)
        elif pose.floor_x_m is not None:
            # Fallback to floor center if head landmark not provided directly
            head_aligned = abs(pose.floor_x_m - (ARENA_WIDTH_M / 2.0)) <= 0.35

        # Player depth position on floor (default to center if untracked)
        py = pose.floor_y_m if pose.floor_y_m is not None else 0.9

        # Handle cooldown between walls
        if self.active_wall is None:
            self.cooldown_timer -= dt
            if self.cooldown_timer <= 0.0:
                self.active_wall = self.spawn_wall()
            return self.state, self.active_wall

        wall = self.active_wall
        wall.head_aligned = head_aligned

        # Move wall toward camera (depth decreases from 1.8m -> 0.0m)
        wall.y_m -= wall.speed_m_s * dt

        # Collision distance from wall plane to player
        dist_to_player = wall.y_m - py

        # ----------------------------------------------------------------------
        # Collision Slice & Judgment Window
        # ----------------------------------------------------------------------
        if abs(dist_to_player) <= COLLISION_TOLERANCE_M:
            wall.state = WallState.COLLIDING

            if not wall.judged:
                wall.judged = True
                px = pose.floor_x_m if pose.floor_x_m is not None else 0.9
                player_pos = (px, py) if pose.tracked else None

                # Evaluate pose similarity with head alignment
                match_res = evaluate_pose(
                    player_angles=player_angles or {},
                    player_pos_m=player_pos,
                    template=wall.template,
                    player_tracked=pose.tracked,
                    player_head_norm=player_head_norm,
                )

                wall.match_pct = match_res.match_pct
                wall.joint_results = match_res.joint_results
                wall.feedback_hint = match_res.primary_hint

                # Clean pass requires: pose match threshold + player tracked + head aligned with cutout
                if match_res.match_pct >= PASS_THRESHOLD_PCT and pose.tracked and head_aligned:
                    # SUCCESS: Player cleanly fit through the cutout
                    wall.state = WallState.PASSED
                    self.walls_passed += 1
                    self.state.dodges += 1
                    self.state.streak += 1
                    self.state.best_streak = max(self.state.best_streak, self.state.streak)

                    streak_mult = min(2.5, 1.0 + (self.state.streak // 3) * 0.2)
                    gain = int(100 * streak_mult)
                    wall.score_earned = gain
                    self.state.score += gain

                    # Speed escalation (up to 1.5x)
                    self.state.speed = min(1.5, 1.0 + self.state.dodges * 0.05)

                    self.last_banner_msg = f"PERFECT FIT! +{gain} (x{streak_mult:.1f})"
                    self.last_banner_color = (0, 255, 120)
                    self.banner_timer = 1.2
                else:
                    # FAIL: Smashed into the wall
                    wall.state = WallState.HIT
                    self.walls_hit += 1
                    self.state.hits += 1
                    self.state.streak = 0
                    wall.score_earned = 0

                    if not head_aligned and match_res.match_pct >= PASS_THRESHOLD_PCT:
                        hint = "ALIGN HEAD WITH CUTOUT"
                    else:
                        hint = match_res.primary_hint if pose.tracked else "STEP INTO ARENA"
                    self.last_banner_msg = f"SMASH! {hint.upper()}"
                    self.last_banner_color = (255, 40, 40)
                    self.banner_timer = 1.2

        elif dist_to_player < -COLLISION_TOLERANCE_M:
            if wall.state not in (WallState.PASSED, WallState.HIT):
                wall.state = WallState.CLEARED

        # Wall has passed the camera view entirely (< -0.3m) -> Despawn
        if wall.y_m < -0.3:
            self.active_wall = None
            self.cooldown_timer = COOLDOWN_TIME_S

        return self.state, wall
