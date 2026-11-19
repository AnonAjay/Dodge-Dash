"""Station B (Pose Blazer Tournament) Pure Game Logic.
PURE PYTHON: No cv2, no pygame, no network imports.

Handles:
  - 2-Stage Tournament Game State Machine:
      * Stage 1: Precision Blitz (Rounds 1-3, 10.0s countdown, adjust posture for 4+ greens -> +150 perfection bonus).
      * Stage 2: Speed Rush (Rounds 4-6, 5.0s countdown, instant snap on pass -> high speed bonus).
  - Non-blocking Folded Arms Skip Gesture: hold crossed arms for 1.0s to forfeit round.
  - Elimination Rule: 4 consecutive skips results in instant disqualification (Phase.RESULT).
  - Default 6-limb tracking (LIMB_6_JOINTS) ignoring unstable hip angles.
  - Integration with shared.contracts (Phase, GameState, PoseFrame).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from pose_core.blazer_matcher import (
    LIMB_6_JOINTS,
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
DEFAULT_TOTAL_ROUNDS = 6
BASE_MATCH_SCORE = 100
PERFECTION_THRESHOLD_PCT = 85.0  # 4 greens out of 6 limbs achieves >= 85%
PERFECTION_BONUS = 150

# Tournament 2-Stage Configuration
STAGE_1_ROUNDS = 3           # Rounds 1-3: Precision Blitz
STAGE_1_DURATION_S = 10.0    # 10s counter to optimize greens
STAGE_1_HOLD_S = 0.35        # 0.35s hold to confirm
STAGE_1_PERFECTION_PTS = 150 # Awarded if match >= 85%

STAGE_2_ROUNDS = 3           # Rounds 4-6: Speed Rush
STAGE_2_DURATION_S = 5.0     # 5s speedrun rush
STAGE_2_HOLD_S = 0.20        # 0.20s quick snap
STAGE_2_SPEED_MULT = 40.0    # pts per second remaining

# Skip & Elimination Constraints
MAX_CONSECUTIVE_SKIPS = 4    # 4 consecutive skips = instant game over
SKIP_HOLD_REQUIRED_S = 1.0   # Hold folded arms for 1.0s to skip pose


def is_arms_crossed(
    player_angles: dict[str, float | None] | None,
    player_landmarks_norm: Sequence[Sequence[float]] | None = None,
) -> bool:
    """Pure geometric check for crossed / folded arms skip gesture.

    Requires:
    1. Both elbows bent sharply across chest (50 to 105 deg).
    2. Both shoulders lowered/neutral (< 50 deg).
    3. If 33 normalized landmarks available, wrists must cross past each other at chest level.
    """
    if not player_angles:
        return False

    l_el = player_angles.get("left_elbow")
    r_el = player_angles.get("right_elbow")
    l_sh = player_angles.get("left_shoulder")
    r_sh = player_angles.get("right_shoulder")

    if l_el is None or r_el is None or l_sh is None or r_sh is None:
        return False

    # 1. Elbows bent sharply across chest
    if not (50.0 <= l_el <= 105.0 and 50.0 <= r_el <= 105.0):
        return False

    # 2. Shoulders neutral or low (not raised in bicep flex or salute)
    if not (l_sh < 50.0 and r_sh < 50.0):
        return False

    # 3. If normalized landmarks available (MediaPipe 33 landmarks), verify wrist cross
    if player_landmarks_norm and len(player_landmarks_norm) >= 17:
        lw = player_landmarks_norm[15]  # L_WRIST
        rw = player_landmarks_norm[16]  # R_WRIST
        lsh = player_landmarks_norm[11] # L_SHOULDER
        rsh = player_landmarks_norm[12] # R_SHOULDER

        crossed = (lw[0] < rw[0]) if lsh[0] > rsh[0] else (rw[0] < lw[0])
        mid_sh_y = (lsh[1] + rsh[1]) / 2.0
        chest_level = (lw[1] > mid_sh_y - 0.05) and (rw[1] > mid_sh_y - 0.05)
        if not (crossed and chest_level):
            return False

    return True


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
    skipped: bool = False


class BlazerGameEngine:
    """State machine governing the 2-Stage Tournament Pose Blazer game."""

    def __init__(
        self,
        catalog: list[BlazerPoseTemplate] | None = None,
        catalog_path: str = "data/poses_index.json",
        round_duration_s: float | None = None,
        total_rounds: int = DEFAULT_TOTAL_ROUNDS,
        active_joints: Sequence[str] | None = None,
        check_head: bool = True,
        timer_enabled: bool = True,
        tournament_mode: bool = True,
    ) -> None:
        self.tournament_mode = tournament_mode
        self.total_rounds = int(total_rounds)
        # Default to 6 limb joints (shoulders, elbows, knees) ignoring ambiguous hip
        self.active_joints = list(active_joints) if active_joints is not None else list(LIMB_6_JOINTS)
        self.check_head = check_head
        self.timer_enabled = timer_enabled

        # If custom duration provided, disable dynamic stage durations
        if round_duration_s is not None:
            self.round_duration_s = float(round_duration_s)
            self.tournament_mode = False
        else:
            self.round_duration_s = STAGE_1_DURATION_S if self.tournament_mode else DEFAULT_ROUND_DURATION_S

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
        self.skip_hold_s: float = 0.0
        self.consecutive_skips: int = 0
        self.total_skips: int = 0
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

    @property
    def current_stage(self) -> int:
        """Stage 1: Precision Blitz (Rounds 1-3), Stage 2: Speed Rush (Rounds 4-6)."""
        if self.tournament_mode and self.round_number > STAGE_1_ROUNDS:
            return 2
        return 1

    @property
    def current_stage_name(self) -> str:
        return "SPEED RUSH" if self.current_stage == 2 else "PRECISION BLITZ"

    def previous_pose(self) -> BlazerPoseTemplate | None:
        """Manually navigate to previous pose (practice/test mode)."""
        if not self.round_poses:
            return None
        self.current_round_idx = (self.current_round_idx - 1) % len(self.round_poses)
        self._reset_round_timers()
        self.banner_msg = f"POSE {self.round_number}/{len(self.round_poses)}: STRIKE POSE!"
        self.banner_color = (0, 220, 255)
        return self.current_pose

    def next_pose(self) -> BlazerPoseTemplate | None:
        """Manually navigate to next pose (practice/test mode)."""
        if not self.round_poses:
            return None
        self.current_round_idx = (self.current_round_idx + 1) % len(self.round_poses)
        self._reset_round_timers()
        self.banner_msg = f"POSE {self.round_number}/{len(self.round_poses)}: STRIKE POSE!"
        self.banner_color = (0, 220, 255)
        return self.current_pose

    def _reset_round_timers(self) -> None:
        """Apply stage-appropriate duration and reset round accumulators."""
        if self.tournament_mode:
            self.round_duration_s = STAGE_2_DURATION_S if self.current_stage == 2 else STAGE_1_DURATION_S
        self.time_left_s = self.round_duration_s
        self.hold_timer_s = 0.0
        self.skip_hold_s = 0.0
        self.is_pose_locked = False
        self.cooldown_timer_s = 0.0

    def start_game(self, player_id: str = "P001", rng: random.Random | None = None) -> None:
        """Start a new tournament run for a player with randomized poses."""
        self.round_poses = sample_game_poses(self.full_catalog, n=self.total_rounds, rng=rng)
        self.current_round_idx = 0
        self.consecutive_skips = 0
        self.total_skips = 0
        self._reset_round_timers()

        stage_name = self.current_stage_name
        self.banner_msg = f"STAGE {self.current_stage}: {stage_name} — STRIKE POSE!"
        self.banner_color = (0, 220, 255)
        self.round_history.clear()
        self.last_match_result = None

        self.state = GameState(
            player_id=player_id,
            phase=Phase.PLAYING,
            score=0,
            dodges=0,  # Passed count
            hits=0,    # Missed count
            streak=0,
            time_left_s=self.round_duration_s * self.total_rounds,
        )

    def update(
        self,
        pose: PoseFrame,
        dt: float,
        player_angles: dict[str, float | None] | None = None,
        player_head_norm: tuple[float, float] | None = None,
        player_landmarks_norm: Sequence[Sequence[float]] | None = None,
    ) -> tuple[GameState, BlazerPoseTemplate | None]:
        """Advance tournament game state and accumulators by dt seconds."""
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
        # 2. Check Folded Arms Skip Gesture (1.0s non-blocking hold)
        # ----------------------------------------------------------------------
        if is_arms_crossed(player_angles, player_landmarks_norm):
            self.skip_hold_s += dt
            pct_held = int(min(100.0, (self.skip_hold_s / SKIP_HOLD_REQUIRED_S) * 100.0))
            if self.skip_hold_s < SKIP_HOLD_REQUIRED_S:
                self.banner_msg = f"HOLDING TO SKIP... {pct_held}%"
                self.banner_color = (255, 160, 0)
            else:
                self.skip_hold_s = 0.0
                self._record_skip()
                return self.state, self.current_pose
        else:
            self.skip_hold_s = max(0.0, self.skip_hold_s - dt * 2.5)

        # ----------------------------------------------------------------------
        # 3. Evaluate Live Player Pose
        # ----------------------------------------------------------------------
        match_res = evaluate_blazer_pose(
            player_landmarks=player_angles or {},
            target_pose=curr_pose,
            player_head_norm=player_head_norm,
            mirrored=True,
            active_joints=self.active_joints,
            check_head=self.check_head,
        )
        self.last_match_result = match_res

        # ----------------------------------------------------------------------
        # 4. Countdown Timer & Stage-Aware Hold Lock-in
        # ----------------------------------------------------------------------
        if self.timer_enabled:
            self.time_left_s = max(0.0, self.time_left_s - dt)
            time_taken_s = round(self.round_duration_s - self.time_left_s, 2)
        else:
            self.time_left_s = self.round_duration_s
            time_taken_s = 0.0

        required_hold = STAGE_2_HOLD_S if self.current_stage == 2 else (
            STAGE_1_HOLD_S if self.tournament_mode else DEFAULT_HOLD_REQUIRED_S
        )

        if match_res.is_matched and pose.tracked:
            self.hold_timer_s += dt

            # Lock-in upon holding required duration
            if self.hold_timer_s >= required_hold and not self.is_pose_locked:
                self.is_pose_locked = True
                self._record_success(time_taken_s, match_res.match_pct)
        else:
            # Drain hold timer quickly if player drops pose
            self.hold_timer_s = max(0.0, self.hold_timer_s - dt * 2.5)

        # ----------------------------------------------------------------------
        # 5. Check Timeout
        # ----------------------------------------------------------------------
        if self.timer_enabled and self.time_left_s <= 0.0 and not self.is_pose_locked:
            self._record_timeout(match_res.match_pct)

        return self.state, self.current_pose

    def _record_success(self, time_taken_s: float, match_pct: float) -> None:
        """Handle a successful pose snap with stage-specific bonuses."""
        self.consecutive_skips = 0  # Successful clear resets consecutive skip counter

        if self.current_stage == 1:
            # Stage 1: Precision Blitz rewards perfection (>= 85%) heavily
            speed_bonus = max(0, int(round(self.time_left_s * 15.0))) if self.timer_enabled else 0
            perf_bonus = STAGE_1_PERFECTION_PTS if match_pct >= PERFECTION_THRESHOLD_PCT else 0
        else:
            # Stage 2: Speed Rush rewards fast reflex snaps heavily
            speed_bonus = max(0, int(round(self.time_left_s * STAGE_2_SPEED_MULT))) if self.timer_enabled else 0
            perf_bonus = 50 if match_pct >= PERFECTION_THRESHOLD_PCT else 0

        earned = BASE_MATCH_SCORE + speed_bonus + perf_bonus
        self.state.score += earned
        self.state.dodges += 1
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

        if self.timer_enabled:
            perf_text = " [PERFECTION!]" if perf_bonus >= 100 else ""
            self.banner_msg = f"SNAP! LOCKED IN {time_taken_s:.1f}s (+{earned} PTS{perf_text})"
            self.banner_color = (50, 240, 100)
            self.cooldown_timer_s = DEFAULT_COOLDOWN_PAUSE_S
        else:
            self.banner_msg = f"SNAP! MATCHED {int(match_pct)}% (PRESS [->] FOR NEXT)"
            self.banner_color = (50, 240, 100)
            self.cooldown_timer_s = 0.0

    def _record_skip(self) -> None:
        """Handle a deliberate pose skip (folded arms held for 1.0s)."""
        self.consecutive_skips += 1
        self.total_skips += 1
        self.state.hits += 1
        self.state.streak = 0

        curr_pose = self.current_pose
        if curr_pose:
            self.round_history.append(
                BlazerRoundRecord(
                    round_num=self.round_number,
                    pose_id=curr_pose.id,
                    pose_name=curr_pose.name,
                    matched=False,
                    time_taken_s=round(self.round_duration_s - self.time_left_s, 2),
                    match_pct=0.0,
                    score_earned=0,
                    speed_bonus=0,
                    skipped=True,
                )
            )

        if self.consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
            self.state.phase = Phase.RESULT
            self.banner_msg = "ELIMINATED! (4 CONSECUTIVE SKIPS)"
            self.banner_color = (255, 40, 40)
            self.cooldown_timer_s = 0.0
        else:
            remaining = MAX_CONSECUTIVE_SKIPS - self.consecutive_skips
            self.banner_msg = f"SKIPPED! (0 PTS | {remaining} SKIPS LEFT BEFORE ELIMINATION)"
            self.banner_color = (255, 140, 0)
            self.cooldown_timer_s = DEFAULT_COOLDOWN_PAUSE_S

    def _record_timeout(self, match_pct: float) -> None:
        """Handle round time expiration."""
        self.state.hits += 1
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
        """Advance to next pose card or finish tournament."""
        self.current_round_idx += 1
        if self.current_round_idx >= self.total_rounds:
            self._finish_game()
        else:
            self._reset_round_timers()
            stage_name = self.current_stage_name
            self.banner_msg = f"STAGE {self.current_stage}: {stage_name} (ROUND {self.round_number}/{self.total_rounds})"
            self.banner_color = (0, 220, 255)

    def _finish_game(self) -> None:
        """Conclude the tournament run and enter result phase."""
        self.state.phase = Phase.RESULT
        self.banner_msg = f"MATCH COMPLETE! FINAL SCORE: {self.state.score}"
        self.banner_color = (255, 215, 0)
