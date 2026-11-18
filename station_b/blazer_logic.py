"""Pose Blazer: Game State Machine and 5-Round Auto-Advancing Engine.

Handles:
  - 5-round pool sampled from data/poses_index.json catalog.
  - 7.0-second countdown timer per pose.
  - Hold-to-snap lock-in window (0.30s) when match >= 75%.
  - Speed bonus calculation: faster snap -> higher points.
  - Automatic transition to the next pose on match or 7s timeout.
  - Integration with shared.contracts (Phase, GameState, PoseFrame).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from pose_core.blazer_matcher import (
    BlazerMatchResult,
    BlazerPoseTemplate,
    evaluate_blazer_pose,
    load_blazer_catalog,
    sample_game_poses,
)
from shared.contracts import GameState, Phase, PoseFrame

# Constants
DEFAULT_ROUND_DURATION_S = 7.0
DEFAULT_HOLD_REQUIRED_S = 0.30
DEFAULT_COOLDOWN_PAUSE_S = 1.0
DEFAULT_TOTAL_ROUNDS = 5
BASE_MATCH_SCORE = 100
PERFECTION_BONUS = 50
PERFECTION_THRESHOLD_PCT = 90.0


@dataclass
class BlazerRoundRecord:
    """Historical record for one completed pose round."""
    round_num: int
    pose_id: str
    pose_name: str
    matched: bool
    time_taken_s: float
    match_pct: float
    score_earned: int
    speed_bonus: int


class BlazerGameEngine:
    """State machine governing the 5-round auto-advancing Pose Blazer game."""

    def __init__(
        self,
        catalog: list[BlazerPoseTemplate] | None = None,
        catalog_path: str = "data/poses_index.json",
        round_duration_s: float = DEFAULT_ROUND_DURATION_S,
        total_rounds: int = DEFAULT_TOTAL_ROUNDS,
    ) -> None:
        self.round_duration_s = float(round_duration_s)
        self.total_rounds = int(total_rounds)

        if catalog is not None and len(catalog) > 0:
            self.full_catalog = list(catalog)
        else:
            self.full_catalog = load_blazer_catalog(catalog_path)

        if not self.full_catalog:
            raise ValueError(f"Blazer catalog at '{catalog_path}' is empty.")

        self.state = GameState(phase=Phase.IDLE)
        self.round_poses: list[BlazerPoseTemplate] = []
        self.current_round_idx: int = 0
        self.time_left_s: float = self.round_duration_s
        self.hold_timer_s: float = 0.0
        self.is_pose_locked: bool = False
        self.cooldown_timer_s: float = 0.0

        self.banner_msg: str = ""
        self.banner_color: tuple[int, int, int] = (255, 204, 0)
        self.round_history: list[BlazerRoundRecord] = []
        self.last_match_result: BlazerMatchResult | None = None

    @property
    def current_pose(self) -> BlazerPoseTemplate | None:
        if 0 <= self.current_round_idx < len(self.round_poses):
            return self.round_poses[self.current_round_idx]
        return None

    @property
    def round_number(self) -> int:
        return self.current_round_idx + 1

    def start_game(self, player_id: str = "P001", rng: random.Random | None = None) -> None:
        """Start a new 5-round run for a player with randomized poses."""
        self.round_poses = sample_game_poses(self.full_catalog, n=self.total_rounds, rng=rng)
        self.current_round_idx = 0
        self.time_left_s = self.round_duration_s
        self.hold_timer_s = 0.0
        self.is_pose_locked = False
        self.cooldown_timer_s = 0.0
        self.banner_msg = "STRIKE THE POSE!"
        self.banner_color = (0, 220, 255)
        self.round_history.clear()
        self.last_match_result = None

        self.state = GameState(
            player_id=player_id,
            phase=Phase.PLAYING,
            score=0,
            dodges=0,  # Used as 'poses passed' count
            hits=0,    # Used as 'poses missed' count
            streak=0,
            time_left_s=self.round_duration_s * self.total_rounds,
        )

    def update(
        self,
        pose: PoseFrame,
        dt: float,
        player_angles: dict[str, float | None] | None = None,
        player_head_norm: tuple[float, float] | None = None,
    ) -> tuple[GameState, BlazerPoseTemplate | None]:
        """Advance game physics and timers by dt seconds."""
        dt = min(max(0.0, dt), 0.1)

        if self.state.phase != Phase.PLAYING:
            return self.state, self.current_pose

        curr_pose = self.current_pose
        if curr_pose is None:
            self._finish_game()
            return self.state, None

        # ----------------------------------------------------------------------
        # 1. Cooldown Transition between Poses
        # ----------------------------------------------------------------------
        if self.cooldown_timer_s > 0.0:
            self.cooldown_timer_s -= dt
            if self.cooldown_timer_s <= 0.0:
                self._advance_to_next_pose()
            return self.state, self.current_pose

        # ----------------------------------------------------------------------
        # 2. Evaluate Live Player Pose
        # ----------------------------------------------------------------------
        match_res = evaluate_blazer_pose(
            player_landmarks=player_angles or {},
            target_pose=curr_pose,
            player_head_norm=player_head_norm,
            mirrored=True,
        )
        self.last_match_result = match_res

        # ----------------------------------------------------------------------
        # 3. 7-Second Countdown Timer & Hold Lock-in
        # ----------------------------------------------------------------------
        self.time_left_s = max(0.0, self.time_left_s - dt)
        time_taken_s = round(self.round_duration_s - self.time_left_s, 2)

        if match_res.is_matched and pose.tracked:
            # Player is mirroring the pose within tolerance
            self.hold_timer_s += dt

            # Lock-in after holding for 0.30s
            if self.hold_timer_s >= DEFAULT_HOLD_REQUIRED_S and not self.is_pose_locked:
                self.is_pose_locked = True
                self._record_success(time_taken_s, match_res.match_pct)
        else:
            # Drain hold timer quickly if player drops pose
            self.hold_timer_s = max(0.0, self.hold_timer_s - dt * 2.5)

        # ----------------------------------------------------------------------
        # 4. Check 7-Second Timeout (Automatic Miss)
        # ----------------------------------------------------------------------
        if self.time_left_s <= 0.0 and not self.is_pose_locked:
            self._record_timeout(match_res.match_pct)

        return self.state, self.current_pose

    def _record_success(self, time_taken_s: float, match_pct: float) -> None:
        """Handle a successful pose snap."""
        speed_bonus = max(0, int(round(self.time_left_s * 20.0)))
        perf_bonus = PERFECTION_BONUS if match_pct >= PERFECTION_THRESHOLD_PCT else 0
        earned = BASE_MATCH_SCORE + speed_bonus + perf_bonus

        self.state.score += earned
        self.state.dodges += 1  # Passed count
        self.state.streak += 1

        curr_pose = self.current_pose
        if curr_pose:
            self.round_history.append(
                BlazerRoundRecord(
                    round_num=self.round_number,
                    pose_id=curr_pose.id,
                    pose_name=curr_pose.name,
                    matched=True,
                    time_taken_s=time_taken_s,
                    match_pct=match_pct,
                    score_earned=earned,
                    speed_bonus=speed_bonus,
                )
            )

        self.banner_msg = f"SNAP! LOCKED IN {time_taken_s:.1f}s (+{earned} PTS)"
        self.banner_color = (50, 240, 100)
        self.cooldown_timer_s = DEFAULT_COOLDOWN_PAUSE_S

    def _record_timeout(self, match_pct: float) -> None:
        """Handle 7-second time limit expiration."""
        self.state.hits += 1  # Missed count
        self.state.streak = 0

        curr_pose = self.current_pose
        if curr_pose:
            self.round_history.append(
                BlazerRoundRecord(
                    round_num=self.round_number,
                    pose_id=curr_pose.id,
                    pose_name=curr_pose.name,
                    matched=False,
                    time_taken_s=self.round_duration_s,
                    match_pct=match_pct,
                    score_earned=0,
                    speed_bonus=0,
                )
            )

        self.banner_msg = "TIME OUT! MISS (0 PTS)"
        self.banner_color = (255, 40, 40)
        self.cooldown_timer_s = DEFAULT_COOLDOWN_PAUSE_S

    def _advance_to_next_pose(self) -> None:
        """Advance to the next pose or conclude game if 5 rounds finished."""
        self.current_round_idx += 1
        if self.current_round_idx >= self.total_rounds:
            self._finish_game()
        else:
            self.time_left_s = self.round_duration_s
            self.hold_timer_s = 0.0
            self.is_pose_locked = False
            self.banner_msg = f"ROUND {self.round_number}/{self.total_rounds}: STRIKE POSE!"
            self.banner_color = (0, 220, 255)

    def _finish_game(self) -> None:
        """Conclude the 5-round run and display result phase."""
        self.state.phase = Phase.RESULT
        self.banner_msg = f"GAME OVER! FINAL SCORE: {self.state.score}"
        self.banner_color = (255, 215, 0)
