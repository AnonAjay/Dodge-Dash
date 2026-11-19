"""Station B (Pose Blazer Tournament) Pure Game Logic.
PURE PYTHON: No cv2, no pygame, no network imports.

Handles:
  - 2-Round Tournament Game Lifecycle:
      * Round 1: Precision Blitz (10 poses, 6.0s max per pose, +150 perfection bonus for form >= 85%).
      * Intermission (5.0s rest & countdown between rounds, "3, 2, 1, GO!").
      * Round 2: Speed Rush (30.0s global continuous countdown, rapid snap speedrun).
  - No Streak Multipliers: scoring is based strictly on precision form and reaction speed.
  - Non-blocking Folded Arms Skip Gesture (1.0s in Round 1, 0.6s in Round 2).
  - Elimination Constraint: 4 consecutive skips in Round 1 results in disqualification.
  - Full-body presence gating: pauses when player is occluded or out of frame.
"""
from __future__ import annotations

from enum import Enum
import math
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
ROUND_1_POSES_COUNT = 10
ROUND_1_POSE_DURATION_S = 6.0
ROUND_1_HOLD_S = 0.35
ROUND_1_PERFECTION_PTS = 150
ROUND_1_PERFECTION_THRESHOLD = 85.0
ROUND_1_SKIP_HOLD_S = 0.50  # Snappy 0.5s skip hold (Hand on Head)

ROUND_2_GLOBAL_DURATION_S = 30.0
ROUND_2_HOLD_S = 0.20
ROUND_2_SKIP_HOLD_S = 0.40  # 0.4s fast skip in Round 2
ROUND_2_SPEED_MULT = 45.0

INTERMISSION_DURATION_S = 5.0
BASE_MATCH_SCORE = 100
MAX_CONSECUTIVE_SKIPS = 4

# Backward-compatibility aliases
STAGE_1_DURATION_S = ROUND_1_POSE_DURATION_S
STAGE_2_DURATION_S = ROUND_2_GLOBAL_DURATION_S
DEFAULT_ROUND_DURATION_S = ROUND_1_POSE_DURATION_S
DEFAULT_HOLD_REQUIRED_S = ROUND_1_HOLD_S
DEFAULT_COOLDOWN_PAUSE_S = 1.0
DEFAULT_TOTAL_ROUNDS = ROUND_1_POSES_COUNT
PERFECTION_THRESHOLD_PCT = ROUND_1_PERFECTION_THRESHOLD
PERFECTION_BONUS = ROUND_1_PERFECTION_PTS


class BlazerTournamentPhase(str, Enum):
    WAITING_FOR_PLAYER = "waiting"
    INTERMISSION = "intermission"
    ROUND_ACTIVE = "active"
    COMPLETED = "completed"


def is_hand_on_head(
    player_angles: dict[str, float | None] | None,
    player_landmarks_norm: Sequence[Sequence[float]] | None = None,
) -> bool:
    """Detect skip gesture: Hand on Head (Right/Left hand on crown of head) OR Crossed Arms.
    
    Vision conditions with MediaPipe 33 normalized landmarks:
    - Right wrist horizontally aligned with head: abs(wrist.x - nose.x) < 0.15
    - Right wrist vertically above nose: wrist.y < nose.y
    - Right elbow bent to reach crown of head: right_elbow < 115.0
    (Same check supported for Left hand as an ergonomic bonus).
    """
    if not player_angles:
        return False

    r_el = player_angles.get("right_elbow")
    r_sh = player_angles.get("right_shoulder")
    l_el = player_angles.get("left_elbow")
    l_sh = player_angles.get("left_shoulder")

    # 1. Landmark checks (when 33 MediaPipe landmarks are available)
    if player_landmarks_norm and len(player_landmarks_norm) >= 17:
        nose = player_landmarks_norm[0]
        rw = player_landmarks_norm[16]
        lw = player_landmarks_norm[15]

        # Check A: Right hand on head
        r_bent = (r_el is not None and r_el < 115.0) or (r_sh is not None and r_sh > 40.0)
        r_on_head = (rw[1] < nose[1]) and (abs(rw[0] - nose[0]) < 0.15) and r_bent
        if r_on_head:
            return True

        # Check B: Left hand on head
        l_bent = (l_el is not None and l_el < 115.0) or (l_sh is not None and l_sh > 40.0)
        l_on_head = (lw[1] < nose[1]) and (abs(lw[0] - nose[0]) < 0.15) and l_bent
        if l_on_head:
            return True

        # Check C: Crossed Arms with landmarks
        lsh = player_landmarks_norm[11]
        rsh = player_landmarks_norm[12]
        crossed = (lw[0] < rw[0]) if lsh[0] > rsh[0] else (rw[0] < lw[0])
        mid_sh_y = (lsh[1] + rsh[1]) / 2.0
        chest_level = (lw[1] > mid_sh_y - 0.05) and (rw[1] > mid_sh_y - 0.05)
        elbows_bent = (l_el is not None and 50.0 <= l_el <= 105.0) and (r_el is not None and 50.0 <= r_el <= 105.0)
        if crossed and chest_level and elbows_bent:
            return True

        # Landmarks present, neither hand on head nor crossed arms
        return False

    # 2. Angle-only fallback (when landmarks are None / not provided)
    if r_el is not None and r_sh is not None:
        if r_el < 105.0 and r_sh > 65.0:
            return True
    if l_el is not None and r_el is not None and l_sh is not None and r_sh is not None:
        if (50.0 <= l_el <= 105.0 and 50.0 <= r_el <= 105.0) and (l_sh < 50.0 and r_sh < 50.0):
            return True

    return False


# Backward-compatibility alias
is_arms_crossed = is_hand_on_head


def check_body_in_frame(
    player_landmarks: Sequence[Sequence[float]] | None,
    tracked: bool = True,
) -> bool:
    """Verify player's core body limbs (shoulders, elbows, wrists, hips, knees) are in frame."""
    if not tracked or not player_landmarks or len(player_landmarks) < 27:
        return False
    for idx in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26):
        lm = player_landmarks[idx]
        x, y = float(lm[0]), float(lm[1])
        vis = float(lm[2]) if len(lm) > 2 else 1.0
        if vis < 0.35 or not (-0.05 <= x <= 1.05 and -0.05 <= y <= 1.05):
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
    """State machine governing the 2-Round Tournament Pose Blazer game."""

    def __init__(
        self,
        catalog: list[BlazerPoseTemplate] | None = None,
        catalog_path: str = "data/poses_index.json",
        round_duration_s: float | None = None,
        total_rounds: int = ROUND_1_POSES_COUNT,
        active_joints: Sequence[str] | None = None,
        check_head: bool = False,
        timer_enabled: bool = True,
        tournament_mode: bool = True,
    ) -> None:
        self.tournament_mode = tournament_mode
        self.total_rounds = int(total_rounds)
        self.active_joints = list(active_joints) if active_joints is not None else list(LIMB_6_JOINTS)
        self.check_head = check_head
        self.timer_enabled = timer_enabled

        if round_duration_s is not None:
            self.round_duration_s: float = float(round_duration_s)
            self.time_left_s: float = float(round_duration_s)
            self.tournament_mode = False
            self.tournament_phase = BlazerTournamentPhase.ROUND_ACTIVE
        else:
            self.round_duration_s: float = ROUND_1_POSE_DURATION_S
            self.time_left_s: float = ROUND_1_POSE_DURATION_S
            self.tournament_mode = tournament_mode
            self.tournament_phase = BlazerTournamentPhase.WAITING_FOR_PLAYER

        if catalog is not None and len(catalog) > 0:
            self.full_catalog = list(catalog)
        else:
            self.full_catalog = load_blazer_catalog(catalog_path)

        if not self.full_catalog:
            raise ValueError(f"Blazer catalog at '{catalog_path}' is empty.")

        self.state = GameState(phase=Phase.IDLE)

        # Rounds tracking
        self.current_round: int = 1
        self.round_1_poses: list[BlazerPoseTemplate] = []
        self.round_2_pool: list[BlazerPoseTemplate] = []
        self.current_round_idx: int = 0  # In Round 1: 0..9

        # Timers
        self.intermission_timer_s: float = INTERMISSION_DURATION_S
        self.round_2_timer_s: float = ROUND_2_GLOBAL_DURATION_S
        self.pose_elapsed_s: float = 0.0
        self.hold_timer_s: float = 0.0
        self.skip_hold_s: float = 0.0
        self.cooldown_timer_s: float = 0.0

        # Stats
        self.consecutive_skips: int = 0
        self.total_skips: int = 0
        self.perfections_count: int = 0
        self.r1_cleared: int = 0
        self.r2_cleared: int = 0
        self.is_pose_locked: bool = False

        self.banner_msg: str = ""
        self.banner_color: tuple[int, int, int] = (255, 204, 0)
        self.round_history: list[BlazerRoundRecord] = []
        self.last_match_result: BlazerMatchResult | None = None

    @property
    def current_pose(self) -> BlazerPoseTemplate | None:
        if self.current_round == 1:
            if 0 <= self.current_round_idx < len(self.round_1_poses):
                return self.round_1_poses[self.current_round_idx]
        else:
            if self.round_2_pool:
                idx = self.current_round_idx % len(self.round_2_pool)
                return self.round_2_pool[idx]
        return None

    @property
    def current_stage(self) -> int:
        return self.current_round

    @property
    def current_stage_name(self) -> str:
        return "SPEED RUSH" if self.current_round == 2 else "PRECISION BLITZ"

    @property
    def round_number(self) -> int:
        if self.current_round == 1:
            return self.current_round_idx + 1
        return self.r2_cleared + 1

    def start_game(self, player_id: str = "P001", rng: random.Random | None = None) -> None:
        """Initialize a new tournament run with randomized poses."""
        rand = rng or random.Random()
        # Sample 10 poses for Round 1
        self.round_1_poses = sample_game_poses(self.full_catalog, n=ROUND_1_POSES_COUNT, rng=rand)
        # Pool for Round 2
        r1_ids = {p.id for p in self.round_1_poses}
        rem = [p for p in self.full_catalog if p.id not in r1_ids]
        rand.shuffle(rem)
        self.round_2_pool = rem if len(rem) >= 10 else list(self.full_catalog)

        self.current_round = 1
        self.current_round_idx = 0
        if not self.tournament_mode:
            self.tournament_phase = BlazerTournamentPhase.ROUND_ACTIVE
            self.time_left_s = self.round_duration_s
        else:
            self.tournament_phase = BlazerTournamentPhase.WAITING_FOR_PLAYER
            self.time_left_s = ROUND_1_POSE_DURATION_S
        self.round_2_timer_s = ROUND_2_GLOBAL_DURATION_S
        self.pose_elapsed_s = 0.0

        self.consecutive_skips = 0
        self.total_skips = 0
        self.perfections_count = 0
        self.r1_cleared = 0
        self.r2_cleared = 0
        self.is_pose_locked = False
        self.cooldown_timer_s = 0.0

        self.banner_msg = "STEP INTO FRAME TO BEGIN"
        self.banner_color = (0, 220, 255)
        self.round_history.clear()
        self.last_match_result = None

        self.state = GameState(
            player_id=player_id,
            phase=Phase.PLAYING,
            score=0,
            dodges=0,
            hits=0,
            streak=0,
            time_left_s=self.time_left_s,
        )

    def _reset_round_timers(self) -> None:
        """Reset per-pose counters."""
        self.hold_timer_s = 0.0
        self.skip_hold_s = 0.0
        self.pose_elapsed_s = 0.0
        self.is_pose_locked = False
        self.cooldown_timer_s = 0.0
        if not self.tournament_mode:
            self.time_left_s = self.round_duration_s
        elif self.current_round == 1:
            self.round_duration_s = ROUND_1_POSE_DURATION_S
            self.time_left_s = ROUND_1_POSE_DURATION_S
        else:
            self.round_duration_s = ROUND_2_GLOBAL_DURATION_S
            self.time_left_s = self.round_2_timer_s

    def update(
        self,
        pose: PoseFrame,
        dt: float,
        player_angles: dict[str, float | None] | None = None,
        player_head_norm: tuple[float, float] | None = None,
        player_landmarks_norm: Sequence[Sequence[float]] | None = None,
    ) -> tuple[GameState, BlazerPoseTemplate | None]:
        """Advance tournament lifecycle, countdowns, and pose evaluation."""
        dt = min(max(0.0, dt), 0.1)

        if self.state.phase != Phase.PLAYING:
            return self.state, self.current_pose

        # ----------------------------------------------------------------------
        # 1. Full-Body Presence Check
        # ----------------------------------------------------------------------
        in_frame = True
        if player_landmarks_norm is not None:
            in_frame = check_body_in_frame(player_landmarks_norm, pose.tracked)

        if not in_frame:
            self.banner_msg = "STEP INTO FRAME"
            self.banner_color = (255, 180, 0)
            self.hold_timer_s = 0.0
            return self.state, self.current_pose

        # Transition from Waiting to First Intermission upon body acquisition
        if self.tournament_phase == BlazerTournamentPhase.WAITING_FOR_PLAYER:
            self.tournament_phase = BlazerTournamentPhase.INTERMISSION
            self.intermission_timer_s = INTERMISSION_DURATION_S
            self.banner_msg = "ROUND 1: PRECISION BLITZ"
            self.banner_color = (0, 230, 255)

        # ----------------------------------------------------------------------
        # 2. Intermission Phase (5.0s Rest & Ready Countdown)
        # ----------------------------------------------------------------------
        if self.tournament_phase == BlazerTournamentPhase.INTERMISSION:
            self.intermission_timer_s -= dt
            self.state.time_left_s = max(0.0, self.intermission_timer_s)
            cd_s = int(math.ceil(self.intermission_timer_s))

            if self.intermission_timer_s > 1.2:
                r_title = "ROUND 1: PRECISION BLITZ" if self.current_round == 1 else "ROUND 2: SPEED RUSH"
                self.banner_msg = f"{r_title} — STARTING IN {cd_s}..."
                self.banner_color = (0, 230, 255) if self.current_round == 1 else (255, 170, 0)
            else:
                self.banner_msg = "STRIKE POSE!"
                self.banner_color = (0, 255, 128)

            if self.intermission_timer_s <= 0.0:
                self.tournament_phase = BlazerTournamentPhase.ROUND_ACTIVE
                self._reset_round_timers()
                r_title = "ROUND 1: PRECISION BLITZ" if self.current_round == 1 else "ROUND 2: SPEED RUSH"
                self.banner_msg = f"{r_title} — GO!"
                self.banner_color = (0, 255, 128)

            return self.state, self.current_pose

        # ----------------------------------------------------------------------
        # 3. Active Round Play
        # ----------------------------------------------------------------------
        curr_pose = self.current_pose
        if curr_pose is None:
            self._finish_game()
            return self.state, None

        # Cooldown pause between poses
        if self.cooldown_timer_s > 0.0:
            self.cooldown_timer_s -= dt
            if self.cooldown_timer_s <= 0.0:
                self._advance_to_next_pose()
            return self.state, self.current_pose

        # Folded Arms Skip Gesture
        skip_req = ROUND_2_SKIP_HOLD_S if self.current_round == 2 else ROUND_1_SKIP_HOLD_S
        if is_arms_crossed(player_angles, player_landmarks_norm):
            self.skip_hold_s += dt
            pct_held = int(min(100.0, (self.skip_hold_s / skip_req) * 100.0))
            if self.skip_hold_s < skip_req:
                self.banner_msg = f"HOLDING TO SKIP... {pct_held}%"
                self.banner_color = (255, 160, 0)
            else:
                self.skip_hold_s = 0.0
                self._record_skip()
                return self.state, self.current_pose
        else:
            self.skip_hold_s = max(0.0, self.skip_hold_s - dt * 2.5)

        # Evaluate Live Pose
        match_res = evaluate_blazer_pose(
            player_landmarks=player_angles or {},
            target_pose=curr_pose,
            player_head_norm=player_head_norm,
            mirrored=True,
            active_joints=self.active_joints,
            check_head=self.check_head,
        )
        self.last_match_result = match_res
        self.pose_elapsed_s += dt

        # ----------------------------------------------------------------------
        # Round 1 Logic (Per-Pose 6.0s Timer, Quality Precision)
        # ----------------------------------------------------------------------
        if self.current_round == 1:
            if self.timer_enabled:
                if self.tournament_mode:
                    self.round_duration_s = ROUND_1_POSE_DURATION_S
                self.time_left_s = max(0.0, self.time_left_s - dt)
                self.state.time_left_s = self.time_left_s
            else:
                self.time_left_s = self.round_duration_s

            if match_res.is_matched and pose.tracked:
                self.hold_timer_s += dt
                if self.hold_timer_s >= ROUND_1_HOLD_S and not self.is_pose_locked:
                    self.is_pose_locked = True
                    self._record_success(self.pose_elapsed_s, match_res.match_pct)
            else:
                self.hold_timer_s = max(0.0, self.hold_timer_s - dt * 2.5)

            # Round 1 Pose Timeout
            if self.timer_enabled and self.time_left_s <= 0.0 and not self.is_pose_locked:
                self._record_timeout(match_res.match_pct)

        # ----------------------------------------------------------------------
        # Round 2 Logic (Global 30.0s Speedrun Clock)
        # ----------------------------------------------------------------------
        else:
            if self.timer_enabled:
                self.round_2_timer_s = max(0.0, self.round_2_timer_s - dt)
                self.round_duration_s = ROUND_2_GLOBAL_DURATION_S
                self.time_left_s = self.round_2_timer_s
                self.state.time_left_s = self.round_2_timer_s
            else:
                self.time_left_s = ROUND_2_GLOBAL_DURATION_S

            if match_res.is_matched and pose.tracked:
                self.hold_timer_s += dt
                if self.hold_timer_s >= ROUND_2_HOLD_S and not self.is_pose_locked:
                    self.is_pose_locked = True
                    self._record_success(self.pose_elapsed_s, match_res.match_pct)
            else:
                self.hold_timer_s = max(0.0, self.hold_timer_s - dt * 2.5)

            # Round 2 Global Time Expiration
            if self.timer_enabled and self.round_2_timer_s <= 0.0:
                self._finish_game()

        return self.state, self.current_pose

    def _record_success(self, time_taken_s: float, match_pct: float) -> None:
        """Handle successful snap with form or speed bonus (NO streak multiplier)."""
        self.consecutive_skips = 0

        if self.current_round == 1:
            # Round 1: Precision Blitz rewards posture >= 85% heavily
            speed_bonus = max(0, int(round(self.time_left_s * 15.0))) if self.timer_enabled else 0
            perf_bonus = ROUND_1_PERFECTION_PTS if match_pct >= ROUND_1_PERFECTION_THRESHOLD else 0
            if perf_bonus > 0:
                self.perfections_count += 1
            earned = BASE_MATCH_SCORE + speed_bonus + perf_bonus
            self.r1_cleared += 1
            perf_tag = " [PERFECTION!]" if perf_bonus > 0 else ""
            self.banner_msg = f"SNAP! LOCKED IN {time_taken_s:.1f}S (+{earned} PTS{perf_tag})"
            self.cooldown_timer_s = DEFAULT_COOLDOWN_PAUSE_S
        else:
            # Round 2: Speed Rush rewards fast reflex snap
            speed_bonus = max(20, int(round(max(0.0, 3.5 - time_taken_s) * ROUND_2_SPEED_MULT))) if self.timer_enabled else 30
            earned = BASE_MATCH_SCORE + speed_bonus
            self.r2_cleared += 1
            self.banner_msg = f"SNAP! FAST CLEAR (+{earned} PTS)"
            # Instant advance in speedrun
            self.cooldown_timer_s = 0.25

        self.state.score += earned
        self.state.dodges += 1
        self.banner_color = (0, 255, 128)

        curr_pose = self.current_pose
        if curr_pose:
            self.round_history.append(
                BlazerRoundRecord(
                    round_num=self.round_number,
                    pose_id=curr_pose.id,
                    pose_name=curr_pose.name,
                    matched=True,
                    time_taken_s=round(time_taken_s, 2),
                    match_pct=match_pct,
                    score_earned=earned,
                    speed_bonus=speed_bonus,
                )
            )

    def _record_skip(self) -> None:
        """Handle folded-arms skip gesture."""
        self.consecutive_skips += 1
        self.total_skips += 1
        self.state.hits += 1

        curr_pose = self.current_pose
        if curr_pose:
            self.round_history.append(
                BlazerRoundRecord(
                    round_num=self.round_number,
                    pose_id=curr_pose.id,
                    pose_name=curr_pose.name,
                    matched=False,
                    time_taken_s=round(self.pose_elapsed_s, 2),
                    match_pct=0.0,
                    score_earned=0,
                    speed_bonus=0,
                    skipped=True,
                )
            )

        if self.current_round == 1 and self.consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
            self.state.phase = Phase.RESULT
            self.banner_msg = "ELIMINATED! (4 CONSECUTIVE SKIPS)"
            self.banner_color = (255, 40, 40)
            self.cooldown_timer_s = 0.0
        else:
            remaining = MAX_CONSECUTIVE_SKIPS - self.consecutive_skips if self.current_round == 1 else 99
            self.banner_msg = f"SKIPPED! (0 PTS | {remaining} SKIPS LEFT)" if self.current_round == 1 else "SKIPPED! NEXT POSE"
            self.banner_color = (255, 140, 0)
            self.cooldown_timer_s = 0.4

    def _record_timeout(self, match_pct: float) -> None:
        """Handle Round 1 per-pose timeout."""
        self.state.hits += 1
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
        """Advance to next pose card or transition to Round 2 / Finish."""
        self.current_round_idx += 1

        if self.current_round == 1:
            if self.current_round_idx >= len(self.round_1_poses):
                # Transition to Round 2 Intermission!
                self.current_round = 2
                self.current_round_idx = 0
                self.tournament_phase = BlazerTournamentPhase.INTERMISSION
                self.intermission_timer_s = INTERMISSION_DURATION_S
                self.banner_msg = "ROUND 1 COMPLETE! REST & PREPARE FOR SPEED RUSH"
                self.banner_color = (255, 170, 0)
                self._reset_round_timers()
            else:
                self._reset_round_timers()
                self.banner_msg = f"POSE {self.round_number}/{len(self.round_1_poses)}: STRIKE POSE!"
                self.banner_color = (0, 220, 255)
        else:
            # Round 2: simply reset pose timers and present next pose
            self._reset_round_timers()
            self.banner_msg = f"SPEED RUSH (CLEARED: {self.r2_cleared})"
            self.banner_color = (255, 140, 0)

    def _finish_game(self) -> None:
        """End tournament and enter RESULT phase."""
        self.state.phase = Phase.RESULT
        self.tournament_phase = BlazerTournamentPhase.COMPLETED
        self.banner_msg = f"TOURNAMENT ENDED! FINAL SCORE: {self.state.score}"
        self.banner_color = (255, 215, 0)
