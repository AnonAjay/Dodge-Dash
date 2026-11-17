"""Wall Shape (Game B) baseline test harness and interactive pose matcher.

Follows Hard Rules:
  - NO game rules: NO total score, NO lives, NO game over.
  - Displays: Live match score (0..100%), Hold progress, and Per-wall result.
  - Mirrored camera view with oncoming wall cut-out silhouette and color-coded skeleton.
  - Real-time joint breakdown table with directional adjustment hints.
  - Interactive --keyboard mode for offline testing without a webcam.
"""
from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from pose_core.floor_map import FloorMapper
from pose_core.pose_matcher import (
    DEFAULT_TEMPLATES,
    JointComparison,
    PoseMatchResult,
    PoseTemplate,
    evaluate_pose,
    extract_angles_from_skeleton,
    extract_pose_angles,
    make_untracked_result,
    snapshot_pose_match,
)
from shared.contracts import PoseFrame

# ==============================================================================
# UI & Theme Constants
# ==============================================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX
STAGE_W, STAGE_H = 680, 560
SIDEBAR_W = 420
CANVAS_W, CANVAS_H = STAGE_W + SIDEBAR_W, STAGE_H

# Colors (BGR)
C_BG = (16, 17, 22)
C_PANEL_BG = (22, 24, 32)
C_CARD_BG = (28, 32, 42)
C_BORDER = (45, 52, 68)

C_TEXT_WHITE = (245, 248, 255)
C_TEXT_GREY = (160, 168, 185)
C_TEXT_MUTED = (110, 118, 135)

C_PASS = (0, 230, 115)         # Neon Green
C_WARN = (0, 170, 255)         # Bright Amber / Orange
C_FAIL = (60, 60, 240)         # Crimson Red
C_CYAN = (255, 225, 0)         # Cyan (BGR)
C_WALL_NAVY = (128, 28, 10)    # Bold Dark Navy Blue (BGR)
C_WALL_BLACK = (16, 16, 20)    # Deep Black outline (BGR)
C_WALL_ACCENT = C_WALL_NAVY     # Darker, bolder wall border
C_WALL_FILL = (20, 24, 35)     # Translucent Dark Wall Tint

# Standard MediaPipe Joint Indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

CORE_BONES: list[tuple[int, int]] = [
    (11, 12),           # Shoulders
    (11, 13), (13, 15), # Left Arm
    (12, 14), (14, 16), # Right Arm
    (11, 23), (12, 24), # Torso sides
    (23, 24),           # Hips
    (23, 25), (25, 27), # Left Leg
    (24, 26), (26, 28), # Right Leg
    (0, 11),  (0, 12),  # Neck / Head
]


class StageMap:
    """Uniform scaling and letterbox mapping from camera frame pixels to stage pixels."""

    def __init__(self, frame_w: int, frame_h: int, stage_w: int = STAGE_W, stage_h: int = STAGE_H):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.stage_w = stage_w
        self.stage_h = stage_h

        self.s = min(stage_w / frame_w, stage_h / frame_h)
        self.scaled_w = int(round(frame_w * self.s))
        self.scaled_h = int(round(frame_h * self.s))
        self.ox = (stage_w - self.scaled_w) / 2.0
        self.oy = (stage_h - self.scaled_h) / 2.0

    def to_stage(self, x: float, y: float) -> tuple[int, int]:
        sx = int(round(x * self.s + self.ox))
        sy = int(round(y * self.s + self.oy))
        return (sx, sy)

    def render_base_stage(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.scaled_w, self.scaled_h))
        stage = np.zeros((self.stage_h, self.stage_w, 3), dtype=np.uint8)
        x_start = int(round(self.ox))
        y_start = int(round(self.oy))
        stage[y_start:y_start + self.scaled_h, x_start:x_start + self.scaled_w] = resized
        return stage


# ==============================================================================
# Reference 2D Posture Generator (Target Wall Cut-Out Silhouettes)
# ==============================================================================

def make_template_ref_posture(template_id: str) -> list[tuple[float, float, float]]:
    """Return 33 normalized 2D/3D reference coordinates for target wall silhouettes.

    Mirrored display coordinates:
      - x = 0.0 is screen-left (player's left limbs)
      - x = 1.0 is screen-right (player's right limbs)
      - y = 0.0 is top, y = 1.0 is bottom
    """
    pts = [(0.0, 0.0, 0.0)] * 33
    pts[0] = (0.50, 0.25, 0.0)  # Nose/Head

    # Torso
    pts[11] = (0.43, 0.38, 0.0)  # L_SHOULDER
    pts[12] = (0.57, 0.38, 0.0)  # R_SHOULDER
    pts[23] = (0.45, 0.60, 0.0)  # L_HIP
    pts[24] = (0.55, 0.60, 0.0)  # R_HIP

    # Default standing legs
    pts[25] = (0.45, 0.76, 0.0)  # L_KNEE
    pts[26] = (0.55, 0.76, 0.0)  # R_KNEE
    pts[27] = (0.45, 0.92, 0.0)  # L_ANKLE
    pts[28] = (0.55, 0.92, 0.0)  # R_ANKLE

    tid = template_id.lower()
    if tid == "t_pose":
        pts[13] = (0.29, 0.38, 0.0); pts[15] = (0.15, 0.38, 0.0)
        pts[14] = (0.71, 0.38, 0.0); pts[16] = (0.85, 0.38, 0.0)
    elif tid == "star":
        pts[13] = (0.33, 0.28, 0.0); pts[15] = (0.23, 0.18, 0.0)
        pts[14] = (0.67, 0.28, 0.0); pts[16] = (0.77, 0.18, 0.0)
        pts[25] = (0.40, 0.76, 0.0); pts[27] = (0.35, 0.92, 0.0)
        pts[26] = (0.60, 0.76, 0.0); pts[28] = (0.65, 0.92, 0.0)
    elif tid == "cactus":
        pts[13] = (0.29, 0.38, 0.0); pts[15] = (0.29, 0.24, 0.0)
        pts[14] = (0.71, 0.38, 0.0); pts[16] = (0.71, 0.24, 0.0)
    elif tid == "disco_right":
        pts[14] = (0.67, 0.26, 0.0); pts[16] = (0.77, 0.14, 0.0)
        pts[13] = (0.39, 0.52, 0.0); pts[15] = (0.35, 0.66, 0.0)
    elif tid == "disco_left":
        pts[13] = (0.33, 0.26, 0.0); pts[15] = (0.23, 0.14, 0.0)
        pts[14] = (0.61, 0.52, 0.0); pts[16] = (0.65, 0.66, 0.0)
    elif tid == "tree_pose":
        pts[13] = (0.33, 0.28, 0.0); pts[15] = (0.23, 0.18, 0.0)
        pts[14] = (0.67, 0.28, 0.0); pts[16] = (0.77, 0.18, 0.0)
        pts[26] = (0.63, 0.70, 0.0); pts[28] = (0.48, 0.76, 0.0)
    elif tid == "l_pose":
        pts[13] = (0.43, 0.24, 0.0); pts[15] = (0.43, 0.10, 0.0)
        pts[14] = (0.71, 0.38, 0.0); pts[16] = (0.85, 0.38, 0.0)
    elif tid == "squat":
        pts[11] = (0.43, 0.44, 0.0); pts[12] = (0.57, 0.44, 0.0)
        pts[23] = (0.45, 0.64, 0.0); pts[24] = (0.55, 0.64, 0.0)
        pts[25] = (0.37, 0.75, 0.0); pts[27] = (0.45, 0.90, 0.0)
        pts[26] = (0.63, 0.75, 0.0); pts[28] = (0.55, 0.90, 0.0)
        pts[13] = (0.31, 0.45, 0.0); pts[15] = (0.18, 0.45, 0.0)
        pts[14] = (0.69, 0.45, 0.0); pts[16] = (0.82, 0.45, 0.0)
    else:
        # Default T-Pose fallback
        pts[13] = (0.29, 0.38, 0.0); pts[15] = (0.15, 0.38, 0.0)
        pts[14] = (0.71, 0.38, 0.0); pts[16] = (0.85, 0.38, 0.0)

    return pts


# ==============================================================================
# Hands-Free Gesture & Automated Test Run Utilities
# ==============================================================================

def _lm_xyv(lm: Any) -> tuple[float, float, float]:
    """Extract (x, y, visibility) safely from landmark coordinate."""
    if isinstance(lm, (tuple, list)):
        x = float(lm[0])
        y = float(lm[1])
        v = float(lm[2]) if len(lm) >= 3 else 1.0
        return (x, y, v)
    if hasattr(lm, "x") and hasattr(lm, "y"):
        return (float(lm.x), float(lm.y), float(getattr(lm, "visibility", 1.0)))
    if isinstance(lm, dict) and "x" in lm and "y" in lm:
        return (float(lm["x"]), float(lm["y"]), float(lm.get("visibility", 1.0)))
    return (0.0, 0.0, 0.0)


def is_right_hand_raised(
    debug: dict | None = None,
    player_pts: Sequence[Any] | None = None,
    keyboard_mode: bool = False,
) -> bool:
    """Detect if the player has raised their physical right hand above their right shoulder."""
    # 1. Camera mode: check debug["landmarks_px"] or debug["pts"]
    lms = None
    thresh = 25.0
    if debug and debug.get("landmarks_px") and len(debug["landmarks_px"]) >= 29:
        lms = debug["landmarks_px"]
        thresh = 25.0
    elif debug and debug.get("pts") and len(debug["pts"]) >= 29:
        lms = debug["pts"]
        thresh = 0.05

    if lms is not None:
        p11_x, _, _ = _lm_xyv(lms[L_SHOULDER])
        p12_x, _, _ = _lm_xyv(lms[R_SHOULDER])
        # In a mirrored camera view, index 12 is screen-left (smaller x),
        # so physical right hand is on screen-right (index 11 shoulder, 15 wrist).
        if p12_x < p11_x:
            sh_x, sh_y, v_sh = _lm_xyv(lms[L_SHOULDER])
            wr_x, wr_y, v_wr = _lm_xyv(lms[L_WRIST])
        else:
            sh_x, sh_y, v_sh = _lm_xyv(lms[R_SHOULDER])
            wr_x, wr_y, v_wr = _lm_xyv(lms[R_WRIST])

        if v_sh >= 0.4 and v_wr >= 0.4:
            # In image coordinates, y is downwards (0 at top).
            # Hand raised means wrist y is significantly smaller than shoulder y.
            return wr_y < (sh_y - thresh)

    # 2. Keyboard / synthetic mode: check normalized posture points
    if keyboard_mode and player_pts and len(player_pts) >= 29:
        p11_x, _, _ = _lm_xyv(player_pts[L_SHOULDER])
        p12_x, _, _ = _lm_xyv(player_pts[R_SHOULDER])
        if p12_x < p11_x:
            sh_x, sh_y, _ = _lm_xyv(player_pts[L_SHOULDER])
            wr_x, wr_y, _ = _lm_xyv(player_pts[L_WRIST])
        else:
            sh_x, sh_y, _ = _lm_xyv(player_pts[R_SHOULDER])
            wr_x, wr_y, _ = _lm_xyv(player_pts[R_WRIST])
        return wr_y < (sh_y - 0.05)

    return False


def generate_diagnostic_report(
    run_results: list[dict],
    run_id: str,
    data_dir: Path | None = None,
) -> Path:
    """Compile diagnostic run results into Markdown report and append to history JSON.

    Saves to:
      - data/wall_shape_diagnostic_report.md
      - data/wall_shape_history.json
    Both stay on local system only (gitignored by data/*).
    """
    if data_dir is None:
        data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    history_file = data_dir / "wall_shape_history.json"
    history: list[dict] = []
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    total_poses = len(run_results)
    passed_poses = sum(1 for r in run_results if r.get("result") == "PASSED")
    missed_poses = sum(1 for r in run_results if r.get("result") == "MISSED")
    pass_rate_pct = (passed_poses / max(1, total_poses)) * 100.0

    current_summary = {
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_poses": total_poses,
        "passed": passed_poses,
        "missed": missed_poses,
        "pass_rate_pct": round(pass_rate_pct, 1),
        "results": run_results,
    }
    history.append(current_summary)

    try:
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save run history to {history_file}: {e}")

    # Build Markdown Report
    lines: list[str] = [
        "# Wall Shape (Game B) Diagnostic Test Run Report",
        "",
        f"**Run ID**: `{run_id}`  ",
        f"**Date/Time**: `{current_summary['timestamp']}`  ",
        f"**Overall Result**: **{passed_poses}/{total_poses} Poses Passed ({pass_rate_pct:.1f}%)**  ",
        "",
        "---",
        "",
        "## 1. Test Run Parameters",
        "",
        "| Parameter | Value | Description |",
        "| :--- | :--- | :--- |",
        "| **Pacing** | 3.0s prep countdown + 5.0s approach per wall | Paced pose approach duration |",
        "| **Hold Requirement** | 1.0s continuous match | Duration player must sustain valid posture to PASS |",
        "| **Angular Tolerance** | ±20.0° base | Angular error threshold for individual joints |",
        "| **Hands-Free Trigger** | Right hand raised above shoulder (>=1.2s) | Arming gesture to trigger test run |",
        "",
        "---",
        "",
        "## 2. Per-Pose Breakdown & Evidence Capture",
        "",
    ]

    for item in run_results:
        idx = item.get("pose_index", 0)
        name = item.get("template_name", "Unknown")
        result = item.get("result", "UNKNOWN")
        match_pct = item.get("match_pct", 0.0)
        hold_s = item.get("hold_elapsed_s", 0.0)
        req_hold_s = item.get("required_hold_s", 1.0)
        snap_path = item.get("snapshot_path", "")
        joints = item.get("joints", {})

        status_badge = "✅ **PASSED**" if result == "PASSED" else "❌ **MISSED**"

        lines.append(f"### Pose {idx}: {name} — {status_badge}")
        lines.append("")
        lines.append(f"- **Outcome**: {result} ({'Match held for required duration' if result == 'PASSED' else 'Timed out during approach'})")
        lines.append(f"- **Live Match Score**: `{match_pct:.1f}%` (Hold: `{hold_s:.1f}s / {req_hold_s:.1f}s`)")
        if snap_path:
            norm_snap = snap_path.replace("\\", "/")
            lines.append(f"- **Full Canvas Evidence**: [`{Path(snap_path).name}`](file:///{norm_snap})")
        lines.append("")
        lines.append("| Joint | Actual Angle | Target Angle | Difference | Status | Evaluation Hint |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :--- |")

        failed_joints = []
        for j_name, j_data in joints.items():
            disp_j = j_name.replace("_", " ").title()
            p_ang = j_data.get("player_angle")
            t_ang = j_data.get("target_angle", 0.0)
            diff = j_data.get("diff_deg")
            is_ok = j_data.get("is_matched", False)
            hint = j_data.get("hint", "")

            p_str = f"{p_ang:.0f}°" if p_ang is not None else "--"
            t_str = f"{t_ang:.0f}°"
            diff_str = f"{diff:+.0f}°" if diff is not None else "--"
            status_str = "PASS" if is_ok else ("NOT VISIBLE" if p_ang is None else "ADJUST")

            lines.append(f"| {disp_j} | {p_str} | {t_str} | {diff_str} | {status_str} | {hint} |")
            if not is_ok:
                failed_joints.append((disp_j, p_ang, t_ang, diff, hint))

        lines.append("")
        lines.append("**Diagnostic Diagnosis:**")
        if result == "PASSED":
            lines.append("- All required joints were sustained within angular tolerance. Accurate execution.")
        else:
            if not failed_joints:
                lines.append(f"- Player was out of position or hold time `{hold_s:.1f}s` fell short of required `{req_hold_s:.1f}s` before impact.")
            else:
                for j_disp, p_ang, t_ang, diff, hint in failed_joints:
                    if p_ang is None:
                        lines.append(f"- **{j_disp}**: *Not Visible / Missing Data*. Landmark visibility was <0.5. Probable camera FOV cutoff or body occlusion.")
                    elif "Elbow" in j_disp:
                        lines.append(f"- **{j_disp}**: Deviation `{diff:+.0f}°` (Actual: {p_ang:.0f}°, Target: {t_ang:.0f}°). Hint: '{hint}'. Probable perspective foreshortening if arm was pointed forward/backward, or arm was slightly bent.")
                    elif "Knee" in j_disp or "Hip" in j_disp:
                        lines.append(f"- **{j_disp}**: Deviation `{diff:+.0f}°` (Actual: {p_ang:.0f}°, Target: {t_ang:.0f}°). Hint: '{hint}'. Probable loose clothing obscuring leg hinge, or incomplete stance.")
                    else:
                        lines.append(f"- **{j_disp}**: Deviation `{diff:+.0f}°` (Actual: {p_ang:.0f}°, Target: {t_ang:.0f}°). Hint: '{hint}'.")
        lines.append("")

    # Historical multi-run section
    lines.append("---")
    lines.append("")
    lines.append("## 3. Multi-Run Learning & Rule Calibration Recommendations")
    lines.append("")
    lines.append(f"Aggregated across **{len(history)}** recorded test run(s).")
    lines.append("")

    # Aggregate stats across runs
    pose_stats: dict[str, dict[str, int]] = {}
    joint_deviations: dict[str, list[float]] = {}
    joint_fail_counts: dict[str, int] = {}

    for run in history:
        for r in run.get("results", []):
            p_name = r.get("template_name", "Unknown")
            if p_name not in pose_stats:
                pose_stats[p_name] = {"tested": 0, "passed": 0, "missed": 0}
            pose_stats[p_name]["tested"] += 1
            if r.get("result") == "PASSED":
                pose_stats[p_name]["passed"] += 1
            else:
                pose_stats[p_name]["missed"] += 1

            for j_name, j_data in r.get("joints", {}).items():
                diff = j_data.get("diff_deg")
                is_matched = j_data.get("is_matched", False)
                if diff is not None:
                    joint_deviations.setdefault(j_name, []).append(abs(diff))
                if not is_matched:
                    joint_fail_counts[j_name] = joint_fail_counts.get(j_name, 0) + 1

    lines.append("### Cumulative Pose Success Rate")
    lines.append("")
    lines.append("| Pose Name | Runs Tested | Passed | Missed | Success Rate |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for p_name, st in pose_stats.items():
        rate = (st["passed"] / max(1, st["tested"])) * 100.0
        lines.append(f"| {p_name} | {st['tested']} | {st['passed']} | {st['missed']} | {rate:.1f}% |")
    lines.append("")

    lines.append("### Joint Deviation & Failure Analysis")
    lines.append("")
    lines.append("| Joint | Total Non-Matches | Mean Abs Deviation | Recommendation |")
    lines.append("| :--- | :---: | :---: | :--- |")
    for j_name, fails in sorted(joint_fail_counts.items(), key=lambda x: x[1], reverse=True):
        devs = joint_deviations.get(j_name, [])
        mean_dev = (sum(devs) / len(devs)) if devs else 0.0
        j_title = j_name.replace("_", " ").title()
        if "elbow" in j_name:
            rec = "Widen tolerance to ±25° to account for 2D perspective foreshortening."
        elif "knee" in j_name or "hip" in j_name:
            rec = "Verify camera angle captures full lower body; check standing distance >= 2.0m."
        else:
            rec = "Maintain ±20° tolerance; check player alignment."
        lines.append(f"| {j_title} | {fails} | {mean_dev:.1f}° | {rec} |")
    lines.append("")

    lines.append("### Actionable Calibration Recommendations for Game B & Engine Integration")
    lines.append("")
    lines.append("1. **Tolerance Tuning**: For upper arm/elbow joints, widen default angular tolerance from `±20°` to `±25°`. This accommodates 2D perspective compression when limbs point slightly out of the frontal camera plane without compromising pose distinctiveness.")
    lines.append("2. **Hold Time Pacing**: Keep required hold time at `1.0s` for standard testing, but support dynamic reduction to `0.8s` in rapid arcade gameplay to reduce player fatigue.")
    lines.append("3. **Camera Placement**: Ensure the camera is positioned at 0.9m–1.2m height with an unobstructed view of the floor grid, and the player stands centered on the floor target tile (`STAND HERE`) ~2.2m from the lens to guarantee full head-to-toe landmark visibility.")
    lines.append("4. **Missing Data Handling**: Never score unobserved joints as 0° (which causes false failures). Poses with partial visibility should prompt the player with *'Show your whole body'* rather than failing the pose outright.")
    lines.append("")

    report_path = data_dir / "wall_shape_diagnostic_report.md"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"Warning: could not write diagnostic report to {report_path}: {e}")

    return report_path


# ==============================================================================
# Wall Shape Test Engine State (Rule 3 Compliant: No Total Score, No Lives)
# ==============================================================================

@dataclass
class WallShapeTestState:
    templates: list[PoseTemplate] = field(default_factory=lambda: list(DEFAULT_TEMPLATES))
    template_idx: int = 0
    angle_tolerance_deg: float = 20.0
    required_hold_s: float = 1.0
    wall_approach_s: float = 5.0

    # Live approach loop
    time_to_impact_s: float = 5.0
    hold_elapsed_s: float = 0.0

    # Per-Wall Result (Transient banner, Rule 3)
    wall_state: str = "APPROACHING"  # APPROACHING | PASSED | MISSED
    result_banner_msg: str = ""
    result_banner_col: tuple[int, int, int] = C_PASS
    result_display_s: float = 0.0
    last_wall_result: str = "None"

    # Synthetic Keyboard Player State
    keyboard_mode: bool = False
    kb_posture_id: str = "t_pose"
    kb_arm_offset_deg: float = 0.0
    kb_pos_m: tuple[float, float] = (0.62, 0.62)

    # Automated Diagnostic Test Run State
    test_run_mode: bool = False
    test_run_state: str = "STANDBY"  # STANDBY | COUNTDOWN | RUNNING | TRANSITION | COMPLETED
    hand_raise_timer_s: float = 0.0
    countdown_timer_s: float = 3.0
    transition_timer_s: float = 0.0
    run_id: str = ""
    run_results: list[dict] = field(default_factory=list)
    temp_screenshots_dir: Path | None = None
    should_capture_snapshot: bool = False
    snapshot_path: str = ""

    def current_template(self) -> PoseTemplate:
        return self.templates[self.template_idx % len(self.templates)]

    def start_test_run(self) -> None:
        """Initiate an automated 8-pose diagnostic test run."""
        self.test_run_mode = True
        self.test_run_state = "COUNTDOWN"
        self.countdown_timer_s = 3.0
        self.template_idx = 0
        self.wall_approach_s = 5.0
        self.time_to_impact_s = 5.0
        self.hold_elapsed_s = 0.0
        self.wall_state = "APPROACHING"
        self.hand_raise_timer_s = 0.0
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.temp_screenshots_dir = Path(tempfile.gettempdir()) / "dodge_dash_wall_shape_runs" / f"run_{self.run_id}"
        self.temp_screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.run_results.clear()
        self.should_capture_snapshot = False
        self.snapshot_path = ""
        if self.keyboard_mode:
            self.kb_posture_id = self.current_template().id
            self.kb_arm_offset_deg = 0.0
        print(f"\n[Test Run Started] Run ID: {self.run_id}. Snapshots: {self.temp_screenshots_dir}")

    def next_wall(self) -> None:
        self.template_idx = (self.template_idx + 1) % len(self.templates)
        self.time_to_impact_s = self.wall_approach_s
        self.hold_elapsed_s = 0.0
        self.wall_state = "APPROACHING"
        if self.keyboard_mode:
            self.kb_posture_id = self.current_template().id
            self.kb_arm_offset_deg = 0.0

    def select_template(self, index: int) -> None:
        if 0 <= index < len(self.templates):
            self.template_idx = index
            self.time_to_impact_s = self.wall_approach_s
            self.hold_elapsed_s = 0.0
            self.wall_state = "APPROACHING"
            if self.keyboard_mode:
                self.kb_posture_id = self.current_template().id
                self.kb_arm_offset_deg = 0.0

    def _record_pose_result(self, result_str: str, match_result: PoseMatchResult | None) -> None:
        tmpl = self.current_template()
        joints_data = {}
        match_pct = 0.0
        is_full = False
        cov = 0.0
        rel = False
        hint = ""
        if match_result is not None:
            match_pct = match_result.match_pct
            is_full = match_result.is_full_match
            cov = match_result.coverage
            rel = match_result.reliable
            hint = match_result.primary_hint
            for j_name, j_res in match_result.joint_results.items():
                joints_data[j_name] = {
                    "player_angle": j_res.player_angle,
                    "target_angle": j_res.target_angle,
                    "diff_deg": j_res.diff_deg,
                    "is_matched": j_res.is_matched,
                    "hint": j_res.hint,
                }
        entry = {
            "pose_index": self.template_idx + 1,
            "template_id": tmpl.id,
            "template_name": tmpl.name,
            "description": tmpl.description,
            "result": result_str,
            "match_pct": match_pct,
            "is_full_match": is_full,
            "coverage": cov,
            "reliable": rel,
            "hold_elapsed_s": round(self.hold_elapsed_s, 2),
            "required_hold_s": self.required_hold_s,
            "angle_tolerance_deg": self.angle_tolerance_deg,
            "primary_hint": hint,
            "snapshot_path": self.snapshot_path,
            "joints": joints_data,
        }
        self.run_results.append(entry)

    def update(
        self,
        dt: float,
        match_result: PoseMatchResult | None = None,
        debug: dict | None = None,
        pf: PoseFrame | None = None,
        player_pts: Sequence[Any] | None = None,
    ) -> None:
        """Update wall approach countdown, hold timer, and test run state machine."""
        # Check right hand raise trigger when in STANDBY or COMPLETED
        if self.test_run_state in ("STANDBY", "COMPLETED"):
            if is_right_hand_raised(debug=debug, player_pts=player_pts, keyboard_mode=self.keyboard_mode):
                self.hand_raise_timer_s += dt
                if self.hand_raise_timer_s >= 1.2:
                    self.start_test_run()
                    return
            else:
                self.hand_raise_timer_s = max(0.0, self.hand_raise_timer_s - 2.0 * dt)

        # 1. Countdown before starting run
        if self.test_run_state == "COUNTDOWN":
            self.countdown_timer_s -= dt
            if self.countdown_timer_s <= 0.0:
                self.test_run_state = "RUNNING"
                self.wall_state = "APPROACHING"
                self.time_to_impact_s = self.wall_approach_s
                self.hold_elapsed_s = 0.0
            return

        # 2. Transition between poses during test run
        if self.test_run_state == "TRANSITION":
            self.transition_timer_s -= dt
            if self.transition_timer_s <= 0.0:
                if self.template_idx + 1 < len(self.templates):
                    self.template_idx += 1
                    self.time_to_impact_s = self.wall_approach_s
                    self.hold_elapsed_s = 0.0
                    self.wall_state = "APPROACHING"
                    self.test_run_state = "RUNNING"
                    if self.keyboard_mode:
                        self.kb_posture_id = self.current_template().id
                        self.kb_arm_offset_deg = 0.0
                else:
                    # Completed all 8 poses!
                    self.test_run_state = "COMPLETED"
                    self.test_run_mode = False
                    report_path = generate_diagnostic_report(self.run_results, self.run_id)
                    print(f"\n[Test Run Completed] 8/8 poses evaluated. Report saved to: {report_path}")
            return

        # 3. Running / Approaching wall
        if self.wall_state == "APPROACHING":
            self.time_to_impact_s = max(0.0, self.time_to_impact_s - dt)

            # Hold progress accumulates if pose matches
            if match_result is not None and match_result.is_full_match:
                self.hold_elapsed_s = min(self.required_hold_s, self.hold_elapsed_s + dt)
            else:
                # Slight decay on imperfect pose to reward sustained posture
                self.hold_elapsed_s = max(0.0, self.hold_elapsed_s - 1.2 * dt)

            # Wall Outcome Judgement (Rule 3: Per-wall result only, no game over)
            if self.hold_elapsed_s >= self.required_hold_s:
                self.wall_state = "PASSED"
                self.result_banner_msg = "WALL PASSED! (Match Held)"
                self.result_banner_col = C_PASS
                self.result_display_s = 1.8
                self.last_wall_result = f"PASSED ({self.current_template().name})"

                if self.test_run_mode:
                    if self.temp_screenshots_dir:
                        snap_name = f"pose_{self.template_idx + 1:02d}_{self.current_template().id}_passed.png"
                        self.snapshot_path = str(self.temp_screenshots_dir / snap_name)
                        self.should_capture_snapshot = True
                    self._record_pose_result("PASSED", match_result)
                    self.test_run_state = "TRANSITION"
                    self.transition_timer_s = 1.8

            elif self.time_to_impact_s <= 0.0:
                self.wall_state = "MISSED"
                self.result_banner_msg = "WALL MISSED! (Timed Out)"
                self.result_banner_col = C_FAIL
                self.result_display_s = 1.8
                self.last_wall_result = f"MISSED ({self.current_template().name})"

                if self.test_run_mode:
                    if self.temp_screenshots_dir:
                        snap_name = f"pose_{self.template_idx + 1:02d}_{self.current_template().id}_missed.png"
                        self.snapshot_path = str(self.temp_screenshots_dir / snap_name)
                        self.should_capture_snapshot = True
                    self._record_pose_result("MISSED", match_result)
                    self.test_run_state = "TRANSITION"
                    self.transition_timer_s = 1.8

        elif self.wall_state in ("PASSED", "MISSED") and not self.test_run_mode:
            self.result_display_s -= dt
            if self.result_display_s <= 0.0:
                self.next_wall()


# ==============================================================================
# Rendering Pipeline
# ==============================================================================

def render_stage_view(
    frame: np.ndarray | None,
    state: WallShapeTestState,
    match_res: PoseMatchResult,
    player_pts: Sequence[Any] | None,
    mapper: FloorMapper | None = None,
    pf: PoseFrame | None = None,
    debug: dict | None = None,
) -> np.ndarray:
    """Render mirrored camera feed, floor grid/standing zone, oncoming cut-out wall, and player skeleton."""
    stage_map = None
    if frame is not None:
        fh, fw = frame.shape[:2]
        stage_map = StageMap(fw, fh, STAGE_W, STAGE_H)
        view = stage_map.render_base_stage(frame)
    else:
        # Dark studio stage canvas for keyboard / offline mode
        view = np.full((STAGE_H, STAGE_W, 3), C_BG, dtype=np.uint8)
        # Perspective floor grid
        for gy in range(STAGE_H // 2, STAGE_H, 24):
            cv2.line(view, (0, gy), (STAGE_W, gy), (28, 30, 40), 1, cv2.LINE_AA)
        for gx in range(0, STAGE_W, 60):
            cv2.line(view, (STAGE_W // 2, STAGE_H // 2), (gx, STAGE_H), (28, 30, 40), 1, cv2.LINE_AA)

    # 1. Perspective Floor Grid & Standing Target Zone on Live Video
    if mapper is not None and stage_map is not None:
        w_m, h_m = mapper.w_m, mapper.h_m
        grid_overlay = view.copy()

        # 3x3 floor grid
        cols, rows = 3, 3
        dx, dy = w_m / cols, h_m / rows
        for c in range(cols + 1):
            x = min(w_m, c * dx)
            p1 = stage_map.to_stage(*mapper.to_pixels(x, 0.0))
            p2 = stage_map.to_stage(*mapper.to_pixels(x, h_m))
            cv2.line(grid_overlay, p1, p2, (140, 130, 60), 1, cv2.LINE_AA)

        for r in range(rows + 1):
            y = min(h_m, r * dy)
            p1 = stage_map.to_stage(*mapper.to_pixels(0.0, y))
            p2 = stage_map.to_stage(*mapper.to_pixels(w_m, y))
            cv2.line(grid_overlay, p1, p2, (140, 130, 60), 1, cv2.LINE_AA)

        # Center target standing zone (Center 1x1 tile)
        c_corners_m = [
            (dx, dy), (2 * dx, dy), (2 * dx, 2 * dy), (dx, 2 * dy)
        ]
        c_corners_px = np.array([stage_map.to_stage(*mapper.to_pixels(x, y)) for x, y in c_corners_m], dtype=np.int32)
        in_pos = match_res.pos_matched
        tile_tint = (0, 90, 0) if in_pos else (90, 70, 0)
        cv2.fillPoly(grid_overlay, [c_corners_px], tile_tint)

        # Floor arena boundary
        poly = np.array([stage_map.to_stage(pt[0], pt[1]) for pt in mapper.corners_px], dtype=np.int32).reshape(-1, 1, 2)
        border_col = C_PASS if in_pos else C_WARN
        cv2.polylines(grid_overlay, [poly], True, border_col, 2, cv2.LINE_AA)

        cv2.addWeighted(grid_overlay, 0.45, view, 0.55, 0, view)

        # "STAND HERE" Target Label on floor
        c_px = stage_map.to_stage(*mapper.to_pixels(w_m / 2.0, h_m / 2.0))
        tag_col = C_PASS if in_pos else C_CYAN
        cv2.circle(view, c_px, 6, tag_col, -1, cv2.LINE_AA)
        cv2.putText(view, "STAND HERE", (c_px[0] - 46, c_px[1] + 4),
                    FONT, 0.38, tag_col, 1, cv2.LINE_AA)

        # Player live feet markers on floor (L & R)
        if pf is not None:
            if getattr(pf, "left_foot_m", None):
                lx, ly = stage_map.to_stage(*mapper.to_pixels(*pf.left_foot_m))
                cv2.circle(view, (lx, ly), 6, (255, 120, 0), -1, cv2.LINE_AA)
                cv2.circle(view, (lx, ly), 8, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(view, "L", (lx - 3, ly + 3), FONT, 0.32, (0, 0, 0), 1)
            if getattr(pf, "right_foot_m", None):
                rx, ry = stage_map.to_stage(*mapper.to_pixels(*pf.right_foot_m))
                cv2.circle(view, (rx, ry), 6, (255, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(view, (rx, ry), 8, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(view, "R", (rx - 3, ry + 3), FONT, 0.32, (0, 0, 0), 1)

    # 2. Oncoming Wall Approaching Border
    tmpl = state.current_template()
    approach_ratio = max(0.0, min(1.0, 1.0 - (state.time_to_impact_s / state.wall_approach_s)))
    depth_pad_x = int(35 * (1.0 - approach_ratio))
    depth_pad_y = int(25 * (1.0 - approach_ratio))

    wall_box = (
        depth_pad_x + 10,
        depth_pad_y + 10,
        STAGE_W - 2 * (depth_pad_x + 10),
        STAGE_H - 2 * (depth_pad_y + 10),
    )
    # Bolder oncoming wall border (outer dark contour + navy blue frame)
    cv2.rectangle(
        view,
        (wall_box[0], wall_box[1]),
        (wall_box[0] + wall_box[2], wall_box[1] + wall_box[3]),
        C_WALL_BLACK,
        5,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        view,
        (wall_box[0], wall_box[1]),
        (wall_box[0] + wall_box[2], wall_box[1] + wall_box[3]),
        C_WALL_NAVY,
        3,
        cv2.LINE_AA,
    )

    # 3. Target Cut-Out Silhouette (Bold Dark Navy Blue & Black Reference Wireframe)
    ref_pts = make_template_ref_posture(tmpl.id)
    ref_px: list[tuple[int, int]] = []
    w_scale = 1.0 + 0.12 * approach_ratio

    # Center target silhouette above arena center if camera/mapper active, else screen center
    if mapper is not None and stage_map is not None:
        pts = mapper.corners_px
        w_near_px = math.hypot(pts[0][0] - pts[3][0], pts[0][1] - pts[3][1])
        w_far_px = math.hypot(pts[1][0] - pts[2][0], pts[1][1] - pts[2][1])
        scale_px_m = ((w_near_px + w_far_px) / (2.0 * max(0.1, mapper.w_m))) * stage_map.s
        c_px = stage_map.to_stage(*mapper.to_pixels(mapper.w_m / 2.0, mapper.h_m / 2.0))
        cx = c_px[0]
        cy = int(round(c_px[1] - 0.42 * scale_px_m))
        sil_w = scale_px_m * 1.05 * w_scale
        sil_h = scale_px_m * 1.55 * w_scale
    else:
        scale_px_m = 230.0
        cx, cy = STAGE_W // 2, int(STAGE_H * 0.44)
        sil_w = STAGE_W * 0.72 * w_scale
        sil_h = STAGE_H * 0.68 * w_scale

    for rx, ry, _ in ref_pts:
        px = int(cx + (rx - 0.5) * sil_w)
        py = int(cy + (ry - 0.5) * sil_h)
        ref_px.append((px, py))

    # Draw target cut-out bones (Crisp bold dark silhouette: black outline + navy blue core)
    for j1, j2 in CORE_BONES:
        if j1 < len(ref_px) and j2 < len(ref_px):
            cv2.line(view, ref_px[j1], ref_px[j2], C_WALL_BLACK, 10, cv2.LINE_AA)
            cv2.line(view, ref_px[j1], ref_px[j2], C_WALL_NAVY, 6, cv2.LINE_AA)
    for px, py in ref_px:
        cv2.circle(view, (px, py), 8, C_WALL_BLACK, -1, cv2.LINE_AA)
        cv2.circle(view, (px, py), 6, C_WALL_NAVY, -1, cv2.LINE_AA)

    # 4. Draw Player Skeleton (Color-coded: Green = within tolerance, Red/Amber = adjust)
    player_px: list[tuple[int, int]] = []
    vis_list: list[float] = []

    if debug and debug.get("landmarks_px") and len(debug["landmarks_px"]) >= 29 and stage_map is not None:
        # Live camera mode: 2D image landmarks in frame pixels
        for lm in debug["landmarks_px"]:
            lx, ly = lm[0], lm[1]
            lv = lm[2] if len(lm) >= 3 else 1.0
            player_px.append(stage_map.to_stage(lx, ly))
            vis_list.append(lv)
    elif state.keyboard_mode and player_pts and len(player_pts) >= 29:
        # Keyboard mode: synthetic 2D points centered in studio view
        for lm in player_pts:
            rx, ry = lm[0], lm[1]
            px = int(cx + (rx - 0.5) * (STAGE_W * 0.72) * w_scale)
            py = int(cy + (ry - 0.5) * (STAGE_H * 0.68) * w_scale)
            player_px.append((px, py))
            vis_list.append(1.0)

    # Detect if rendered skeleton is horizontally mirrored (e.g. cv2.flip camera view)
    is_mirrored = False
    if len(player_px) >= 25:
        is_mirrored = (player_px[12][0] < player_px[11][0])

    def is_joint_ok(j_idx: int) -> bool:
        """Unconstrained joints (Head, Hips, Wrists, Ankles) default to green unless requested."""
        if is_mirrored:
            # In mirrored view: index 12 is screen-left (physical left), index 11 is screen-right (physical right)
            name_map = {
                12: "left_shoulder",
                11: "right_shoulder",
                14: "left_elbow",
                13: "right_elbow",
                24: "left_hip",
                23: "right_hip",
                26: "left_knee",
                25: "right_knee",
            }
        else:
            name_map = {
                11: "left_shoulder",
                12: "right_shoulder",
                13: "left_elbow",
                14: "right_elbow",
                23: "left_hip",
                24: "right_hip",
                25: "left_knee",
                26: "right_knee",
            }
        j_name = name_map.get(j_idx)
        if j_name is None:
            return True
        if j_name in match_res.joint_results:
            return match_res.joint_results[j_name].is_matched
        return True

    if player_px:
        # Draw bones (visibility >= 0.5 for both joints)
        for j1, j2 in CORE_BONES:
            if j1 < len(player_px) and j2 < len(player_px):
                v1 = vis_list[j1] if j1 < len(vis_list) else 1.0
                v2 = vis_list[j2] if j2 < len(vis_list) else 1.0
                if v1 >= 0.5 and v2 >= 0.5:
                    p1, p2 = player_px[j1], player_px[j2]
                    if 0 <= p1[0] < STAGE_W and 0 <= p1[1] < STAGE_H and 0 <= p2[0] < STAGE_W and 0 <= p2[1] < STAGE_H:
                        m1 = is_joint_ok(j1)
                        m2 = is_joint_ok(j2)
                        bone_col = C_PASS if (m1 and m2) else C_WARN
                        cv2.line(view, p1, p2, bone_col, 3, cv2.LINE_AA)

        # Draw joints (visibility >= 0.5)
        for idx in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if idx < len(player_px):
                v = vis_list[idx] if idx < len(vis_list) else 1.0
                if v >= 0.5:
                    pt = player_px[idx]
                    if 0 <= pt[0] < STAGE_W and 0 <= pt[1] < STAGE_H:
                        is_ok = is_joint_ok(idx)
                        j_col = C_PASS if is_ok else C_FAIL
                        cv2.circle(view, pt, 6, j_col, -1, cv2.LINE_AA)
                        cv2.circle(view, pt, 8, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. Top Guidance Banner & Per-Wall Result (Rule 3)
    if state.test_run_state == "COUNTDOWN":
        # Centered countdown box before starting test run
        overlay = view.copy()
        cv2.rectangle(overlay, (STAGE_W // 2 - 200, STAGE_H // 2 - 95),
                      (STAGE_W // 2 + 200, STAGE_H // 2 + 95), (12, 14, 20), -1)
        cv2.addWeighted(overlay, 0.88, view, 0.12, 0, view)
        cv2.rectangle(view, (STAGE_W // 2 - 200, STAGE_H // 2 - 95),
                      (STAGE_W // 2 + 200, STAGE_H // 2 + 95), C_CYAN, 2)
        cv2.putText(view, "TEST RUN STARTING", (STAGE_W // 2 - 130, STAGE_H // 2 - 50),
                    FONT, 0.65, C_TEXT_WHITE, 2, cv2.LINE_AA)
        cv2.putText(view, f"Get Ready for Pose 1: {state.current_template().name}",
                    (STAGE_W // 2 - 145, STAGE_H // 2 - 20), FONT, 0.44, C_TEXT_GREY, 1, cv2.LINE_AA)
        cnt_num = str(int(math.ceil(state.countdown_timer_s)))
        cv2.putText(view, cnt_num, (STAGE_W // 2 - 20, STAGE_H // 2 + 65),
                    FONT, 2.2, C_PASS, 4, cv2.LINE_AA)

    elif state.test_run_state == "COMPLETED":
        # Centered completion summary card
        overlay = view.copy()
        cv2.rectangle(overlay, (STAGE_W // 2 - 230, STAGE_H // 2 - 105),
                      (STAGE_W // 2 + 230, STAGE_H // 2 + 105), (12, 14, 20), -1)
        cv2.addWeighted(overlay, 0.90, view, 0.10, 0, view)
        cv2.rectangle(view, (STAGE_W // 2 - 230, STAGE_H // 2 - 105),
                      (STAGE_W // 2 + 230, STAGE_H // 2 + 105), C_PASS, 2)
        cv2.putText(view, "DIAGNOSTIC RUN COMPLETED", (STAGE_W // 2 - 175, STAGE_H // 2 - 60),
                    FONT, 0.65, C_CYAN, 2, cv2.LINE_AA)
        passed_cnt = sum(1 for r in state.run_results if r.get("result") == "PASSED")
        tot_cnt = len(state.run_results)
        rate_val = (passed_cnt / max(1, tot_cnt)) * 100.0
        cv2.putText(view, f"Passed: {passed_cnt}/{tot_cnt} Poses ({rate_val:.0f}%)",
                    (STAGE_W // 2 - 110, STAGE_H // 2 - 20), FONT, 0.58, C_TEXT_WHITE, 2, cv2.LINE_AA)
        cv2.putText(view, "Report saved to data/wall_shape_diagnostic_report.md",
                    (STAGE_W // 2 - 200, STAGE_H // 2 + 20), FONT, 0.38, C_PASS, 1, cv2.LINE_AA)
        cv2.putText(view, "Raise Right Hand or Press [R] to Run Again",
                    (STAGE_W // 2 - 170, STAGE_H // 2 + 65), FONT, 0.44, C_WARN, 1, cv2.LINE_AA)

    elif state.wall_state in ("PASSED", "MISSED"):
        # Big Per-Wall Announcement Banner
        overlay = view.copy()
        cv2.rectangle(overlay, (0, 0), (STAGE_W, 70), (10, 10, 15), -1)
        cv2.addWeighted(overlay, 0.75, view, 0.25, 0, view)
        cv2.putText(view, state.result_banner_msg, (STAGE_W // 2 - 190, 45), FONT, 0.85, state.result_banner_col, 2, cv2.LINE_AA)

        # In test run mode, display upcoming pose teaser
        if state.test_run_mode and state.template_idx + 1 < len(state.templates):
            nxt_name = state.templates[state.template_idx + 1].name
            cv2.putText(view, f"UP NEXT: [{state.template_idx + 2}/8] {nxt_name}",
                        (STAGE_W // 2 - 120, STAGE_H - 45), FONT, 0.50, C_TEXT_WHITE, 1, cv2.LINE_AA)
    else:
        # Live directional cue
        cue_msg = match_res.primary_hint if match_res.primary_hint else "HOLD POSE!"
        cue_col = C_PASS if match_res.is_full_match else C_WARN
        cv2.rectangle(view, (0, 0), (STAGE_W, 44), (12, 14, 20), -1)
        cv2.putText(view, cue_msg, (16, 29), FONT, 0.60, cue_col, 2, cv2.LINE_AA)
        time_str = f"Approach: {state.time_to_impact_s:.1f}s"
        cv2.putText(view, time_str, (STAGE_W - 165, 29), FONT, 0.52, C_TEXT_WHITE, 1, cv2.LINE_AA)

    # Arming Progress Indicator when right hand is raised
    if state.test_run_state in ("STANDBY", "COMPLETED") and state.hand_raise_timer_s > 0.0:
        arm_pct = min(1.0, state.hand_raise_timer_s / 1.2)
        cv2.rectangle(view, (0, 0), (STAGE_W, 36), (15, 20, 28), -1)
        fill_w = int(STAGE_W * arm_pct)
        if fill_w > 0:
            cv2.rectangle(view, (0, 0), (fill_w, 36), (0, 160, 80), -1)
        cv2.putText(view, f"ARMING TEST RUN: Hold Right Hand Up... ({int(arm_pct * 100)}%)",
                    (16, 24), FONT, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

    # Bottom Hold Progress Indicator on Stage
    hold_ratio = max(0.0, min(1.0, state.hold_elapsed_s / state.required_hold_s))
    cv2.rectangle(view, (16, STAGE_H - 28), (STAGE_W - 16, STAGE_H - 12), (30, 34, 44), -1)
    bar_fill = int((STAGE_W - 32) * hold_ratio)
    if bar_fill > 0:
        bar_col = C_PASS if hold_ratio >= 1.0 else C_CYAN
        cv2.rectangle(view, (16, STAGE_H - 28), (16 + bar_fill, STAGE_H - 12), bar_col, -1)
    hold_txt = f"Hold: {state.hold_elapsed_s:.1f}s / {state.required_hold_s:.1f}s"
    cv2.putText(view, hold_txt, (STAGE_W // 2 - 60, STAGE_H - 16), FONT, 0.38, C_TEXT_WHITE, 1, cv2.LINE_AA)

    return view


def render_sidebar_panel(
    state: WallShapeTestState,
    match_res: PoseMatchResult,
    fps: float,
    pose_fps: float,
) -> np.ndarray:
    """Render metrics sidebar: Live Match Score, Hold Progress, Per-Wall Result, Joint Table."""
    panel = np.full((CANVAS_H, SIDEBAR_W, 3), C_PANEL_BG, dtype=np.uint8)
    tmpl = state.current_template()

    # Section 1: Header
    cv2.rectangle(panel, (0, 0), (SIDEBAR_W, 55), (15, 17, 24), -1)
    cv2.putText(panel, "WALL SHAPE: GAME B", (16, 28), FONT, 0.65, C_CYAN, 2, cv2.LINE_AA)
    cv2.putText(panel, "Pose Matcher & Hole-in-Wall Test", (16, 46), FONT, 0.38, C_TEXT_MUTED, 1, cv2.LINE_AA)

    y = 72

    # Section 2: Active Wall Card
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, y + 68), C_CARD_BG, -1)
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, y + 68), C_BORDER, 1)
    w_prefix = f"RUN [{state.template_idx + 1}/{len(state.templates)}]" if state.test_run_mode else f"Wall [{state.template_idx + 1}/{len(state.templates)}]"
    w_title = f"{w_prefix}: {tmpl.name}"
    cv2.putText(panel, w_title, (22, y + 24), FONT, 0.52, C_TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.putText(panel, tmpl.description[:46], (22, y + 44), FONT, 0.36, C_TEXT_GREY, 1, cv2.LINE_AA)
    tol_txt = f"Tol: +/-{state.angle_tolerance_deg:.0f} deg | Req Hold: {state.required_hold_s:.1f}s"
    cv2.putText(panel, tol_txt, (22, y + 60), FONT, 0.34, C_TEXT_MUTED, 1, cv2.LINE_AA)

    y += 82

    # Section 3: Live Match Score (Rule 3)
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, y + 74), C_CARD_BG, -1)
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, y + 74), C_BORDER, 1)
    cv2.putText(panel, "LIVE MATCH SCORE", (22, y + 20), FONT, 0.44, C_TEXT_GREY, 1, cv2.LINE_AA)

    if match_res.reliable:
        score_val = match_res.match_pct
        score_col = C_PASS if score_val >= 80.0 else (C_WARN if score_val >= 60.0 else C_FAIL)
        score_str = f"{score_val:4.1f}%"
        match_status = "MATCH HOLD" if match_res.is_full_match else f"{match_res.matched_joints_count}/{match_res.total_joints_count} Joints OK"
    else:
        score_val = 0.0
        score_col = C_TEXT_MUTED
        score_str = "--"
        match_status = "NOT TRACKED" if match_res.primary_hint == "Player not tracked" else "OUT OF VIEW"

    cv2.putText(panel, score_str, (22, y + 54), FONT, 0.95, score_col, 2, cv2.LINE_AA)
    cv2.putText(panel, match_status, (150, y + 48), FONT, 0.42, score_col, 1, cv2.LINE_AA)

    # Score Gauge Bar
    bar_x = 22
    bar_w = SIDEBAR_W - 44
    cv2.rectangle(panel, (bar_x, y + 62), (bar_x + bar_w, y + 68), (40, 44, 56), -1)
    fill_w = int(bar_w * (score_val / 100.0))
    if fill_w > 0:
        cv2.rectangle(panel, (bar_x, y + 62), (bar_x + fill_w, y + 68), score_col, -1)

    y += 88

    # Section 4: Hold Progress & Wall Result (Rule 3)
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, y + 64), C_CARD_BG, -1)
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, y + 64), C_BORDER, 1)
    cv2.putText(panel, "HOLD PROGRESS", (22, y + 18), FONT, 0.44, C_TEXT_GREY, 1, cv2.LINE_AA)

    hold_ratio = max(0.0, min(1.0, state.hold_elapsed_s / state.required_hold_s))
    hold_col = C_PASS if hold_ratio >= 1.0 else C_CYAN
    cv2.putText(panel, f"{state.hold_elapsed_s:3.1f}s / {state.required_hold_s:3.1f}s", (22, y + 42), FONT, 0.65, hold_col, 2, cv2.LINE_AA)

    res_badge = state.last_wall_result if state.last_wall_result != "None" else "Active"
    badge_col = C_PASS if "PASSED" in res_badge else (C_FAIL if "MISSED" in res_badge else C_TEXT_MUTED)
    cv2.putText(panel, f"Last: {res_badge}", (170, y + 40), FONT, 0.38, badge_col, 1, cv2.LINE_AA)

    cv2.rectangle(panel, (bar_x, y + 52), (bar_x + bar_w, y + 58), (40, 44, 56), -1)
    h_fill = int(bar_w * hold_ratio)
    if h_fill > 0:
        cv2.rectangle(panel, (bar_x, y + 52), (bar_x + h_fill, y + 58), hold_col, -1)

    y += 76

    # Section 5: Joint Angles Breakdown Table
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, y + 180), C_CARD_BG, -1)
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, y + 180), C_BORDER, 1)
    cv2.putText(panel, "JOINT BREAKDOWN", (22, y + 18), FONT, 0.44, C_TEXT_GREY, 1, cv2.LINE_AA)

    table_hdr = f"{'JOINT':<13} {'ACTUAL':>7} {'TARGET':>8} {'DIFF':>7}  {'STATUS':<11}"
    cv2.putText(panel, table_hdr, (20, y + 36), FONT, 0.33, C_TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.line(panel, (18, y + 42), (SIDEBAR_W - 18, y + 42), C_BORDER, 1)

    row_y = y + 57
    for j_name, j_res in match_res.joint_results.items():
        disp_name = j_name.replace("_", " ").title()[:13]
        tgt_s = f"{j_res.target_angle:5.0f} deg"
        if j_res.player_angle is None:
            act_s = "   --  "
            diff_s = "   -- "
            status_s = "NOT VISIBLE"
            row_col = C_TEXT_MUTED
        else:
            act_s = f"{j_res.player_angle:5.0f} deg"
            diff_s = f"{j_res.diff_deg:+5.0f} deg" if j_res.diff_deg is not None else "   -- "
            status_s = "PASS" if j_res.is_matched else "ADJ"
            status_col = C_PASS if j_res.is_matched else C_WARN
            row_col = status_col if not j_res.is_matched else C_TEXT_WHITE

        row_str = f"{disp_name:<13} {act_s} {tgt_s} {diff_s}  {status_s}"
        cv2.putText(panel, row_str, (20, row_y), FONT, 0.32, row_col, 1, cv2.LINE_AA)
        row_y += 18

    y += 192

    # Section 6: Controls & Diagnostics Footer
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, CANVAS_H - 12), (18, 20, 26), -1)
    cv2.rectangle(panel, (12, y), (SIDEBAR_W - 12, CANVAS_H - 12), C_BORDER, 1)

    cv2.putText(panel, f"App FPS: {fps:.0f} | Pose FPS: {pose_fps:.0f}", (20, y + 18), FONT, 0.35, C_TEXT_MUTED, 1, cv2.LINE_AA)
    run_txt = "[R]: Run Active (Paced)" if state.test_run_mode else "[R]: Start Test Run"
    cv2.putText(panel, f"{run_txt}  [Space]: Next", (20, y + 36), FONT, 0.34, C_PASS if state.test_run_mode else C_TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(panel, "[1-8]: Select Pose  [T]: Tol  [H]: Hold", (20, y + 52), FONT, 0.34, C_TEXT_GREY, 1, cv2.LINE_AA)
    kb_str = "ON (Synthetic)" if state.keyboard_mode else "OFF (Webcam)"
    cv2.putText(panel, f"[K]: Keyboard Mode: {kb_str}", (20, y + 68), FONT, 0.34, C_CYAN if state.keyboard_mode else C_TEXT_GREY, 1, cv2.LINE_AA)

    return panel


# ==============================================================================
# Headless Step Function (for offline/automated test suites outside repo)
# ==============================================================================

def step_frame(
    state: WallShapeTestState,
    dt: float,
    source: Any | None = None,
    cam_frame: np.ndarray | None = None,
    player_pts: Sequence[Any] | None = None,
    mapper: FloorMapper | None = None,
    fps: float = 30.0,
    pose_fps: float = 30.0,
    pf: PoseFrame | None = None,
    debug: dict | None = None,
) -> tuple[np.ndarray, PoseMatchResult]:
    """Execute one simulation frame, update Wall Shape test state, and render canvas."""
    tmpl = state.current_template()

    # Determine player landmarks and position
    pts_to_evaluate: Any | None = None
    player_pos: tuple[float, float] | None = None
    player_tracked = False
    if debug is None:
        debug = {}

    if state.keyboard_mode:
        # Generate synthetic posture for keyboard mode
        synthetic_pts = make_template_ref_posture(state.kb_posture_id)
        if state.kb_arm_offset_deg != 0.0:
            # Shift right wrist to simulate arm offset
            r_wrist = list(synthetic_pts[R_WRIST])
            rad = math.radians(state.kb_arm_offset_deg)
            r_wrist[1] += 0.15 * math.sin(rad)
            synthetic_pts[R_WRIST] = tuple(r_wrist)
        pts_to_evaluate = synthetic_pts
        player_pos = state.kb_pos_m
        pf = PoseFrame(tracked=True, floor_x_m=state.kb_pos_m[0], floor_y_m=state.kb_pos_m[1],
                       left_foot_m=(state.kb_pos_m[0] - 0.1, state.kb_pos_m[1]),
                       right_foot_m=(state.kb_pos_m[0] + 0.1, state.kb_pos_m[1]))
        player_tracked = True
    elif source is not None:
        # Pull live frame and landmarks from CameraPoseSource
        if hasattr(source, "snapshot3d"):
            pf, skel, frame, debug = source.snapshot3d()
            if frame is not None:
                cam_frame = frame
            if skel and hasattr(skel, "points_m"):
                pts_to_evaluate = skel
            elif debug.get("pts"):
                pts_to_evaluate = debug["pts"]
            if pf and getattr(pf, "tracked", False):
                player_tracked = True
                fx = getattr(pf, "floor_x_m", None)
                fy = getattr(pf, "floor_y_m", None)
                if fx is not None and fy is not None:
                    player_pos = (fx, fy)
    elif player_pts is not None:
        pts_to_evaluate = player_pts
        if pf is not None:
            player_tracked = bool(getattr(pf, "tracked", False))
            fx = getattr(pf, "floor_x_m", None)
            fy = getattr(pf, "floor_y_m", None)
            if fx is not None and fy is not None:
                player_pos = (fx, fy)
        else:
            player_tracked = True

    # If player is not tracked, return untracked result without evaluating pose
    if not player_tracked:
        match_result = make_untracked_result(tmpl, hint="Player not tracked")
    else:
        player_angles: dict[str, float | None] = {}
        if pts_to_evaluate is not None:
            if hasattr(pts_to_evaluate, "points_m"):
                player_angles = extract_angles_from_skeleton(pts_to_evaluate)
            elif len(pts_to_evaluate) >= 29:
                player_angles = extract_pose_angles(pts_to_evaluate)

        match_result = evaluate_pose(
            player_angles=player_angles,
            player_pos_m=player_pos,
            template=tmpl,
            angle_tolerance_override=state.angle_tolerance_deg,
            player_tracked=player_tracked,
        )

    # Update Wall Shape test engine
    state.update(dt, match_result, debug=debug, pf=pf, player_pts=pts_to_evaluate)

    # Render stage and sidebar
    stage = render_stage_view(cam_frame, state, match_result, pts_to_evaluate, mapper=mapper, pf=pf, debug=debug)
    sidebar = render_sidebar_panel(state, match_result, fps, pose_fps)
    canvas = np.hstack([stage, sidebar])

    # Save diagnostic snapshot if triggered
    if state.should_capture_snapshot and state.snapshot_path:
        try:
            os.makedirs(os.path.dirname(state.snapshot_path), exist_ok=True)
            cv2.imwrite(state.snapshot_path, canvas)
        except Exception as e:
            print(f"Warning: Failed to save snapshot {state.snapshot_path}: {e}")
        state.should_capture_snapshot = False

    return canvas, match_result


# ==============================================================================
# Main Interactive CLI
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Wall Shape (Game B) Baseline Test Harness")
    parser.add_argument("--keyboard", action="store_true", help="run with synthetic poses, no camera needed")
    parser.add_argument("--camera", type=int, default=0, help="camera device index (default: 0)")
    parser.add_argument("--tolerance", type=float, default=20.0, help="angular tolerance in degrees (default: 20.0)")
    parser.add_argument("--hold", type=float, default=1.0, help="required hold time in seconds (default: 1.0)")
    parser.add_argument("--wall-time", type=float, default=5.0, help="wall approach duration in seconds (default: 5.0)")
    parser.add_argument("--template", type=str, default="", help="initial template id (e.g. t_pose, star, cactus)")
    parser.add_argument("--headless", action="store_true", help="run headlessly without opening GUI window")
    args = parser.parse_args()

    state = WallShapeTestState(
        angle_tolerance_deg=args.tolerance,
        required_hold_s=args.hold,
        wall_approach_s=args.wall_time,
        keyboard_mode=args.keyboard,
    )

    if args.template:
        t_ids = [t.id for t in state.templates]
        if args.template in t_ids:
            state.template_idx = t_ids.index(args.template)

    mapper = FloorMapper.load()
    source = None
    if not args.keyboard and not args.headless:
        from pose_core.capture import CameraPoseSource
        try:
            source = CameraPoseSource(camera_index=args.camera, mapper=mapper)
            source.start()
            atexit.register(source.stop)
            print(f"Wall Shape CameraPoseSource started on camera index {args.camera}.")
        except Exception as e:
            print(f"Warning: Could not start camera {args.camera}: {e}. Falling back to keyboard mode.")
            state.keyboard_mode = True

    print("==================================================================")
    print(" WALL SHAPE (Game B) - BASELINE TEST HARNESS")
    print(" Controls:")
    print("   [R]          : Start Automated 8-Pose Test Run (or raise right hand)")
    print("   [1-8]        : Select pose (1:T-Pose, 2:Star, 3:Cactus, 4:Tree...)")
    print("   [Space]      : Cycle to next wall template")
    print("   [T]          : Toggle angle tolerance (+/-10, +/-15, +/-20, +/-25 deg)")
    print("   [H]          : Toggle required hold time (0.5s, 1.0s, 1.5s)")
    print("   [K]          : Toggle Keyboard (Synthetic) mode")
    print("   [WASD]       : Move player position on floor (keyboard mode)")
    print("   [ [ / ] ]    : Perturb arm angle by +/-10 deg (keyboard mode)")
    print("   [Q / ESC]    : Quit")
    print("==================================================================")

    last_time = time.monotonic()
    fps = 30.0
    calib_warned = False

    try:
        while True:
            now = time.monotonic()
            dt = min(now - last_time, 0.1)
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            pose_fps = source.pose_fps if source else (fps if state.keyboard_mode else 0.0)
            canvas, res = step_frame(state, dt, source=source, mapper=mapper, fps=fps, pose_fps=pose_fps)

            # Check calibration fit once when camera frame is available
            if not calib_warned and source is not None and mapper is not None:
                cam_f = getattr(source, "latest_frame", lambda: None)()
                if cam_f is not None:
                    calib_warned = True
                    fh, fw = cam_f.shape[:2]
                    print(f"Camera frame size: {fw}x{fh}")
                    corners = mapper.corners_px
                    out_of_bounds = any(
                        x < 0 or y < 0 or x >= fw or y >= fh
                        for x, y in corners
                    )
                    if out_of_bounds:
                        print("WARNING: calibration does not fit this camera frame (different resolution or camera moved). Run: python -m tools.calibrate_floor")

            if args.headless:
                time.sleep(0.03)
                continue

            cv2.imshow("Wall Shape (Game B Baseline)", canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):  # Q or ESC
                break
            elif key in (ord("r"), ord("R")):
                state.start_test_run()
            elif key == ord(" "):       # Space: Next wall
                state.next_wall()
            elif key in [ord(str(i)) for i in range(1, 9)]:
                idx = int(chr(key)) - 1
                state.select_template(idx)
                if state.keyboard_mode:
                    state.kb_posture_id = state.current_template().id
                    state.kb_arm_offset_deg = 0.0
            elif key in (ord("t"), ord("T")):
                tols = [10.0, 15.0, 20.0, 25.0]
                cur_i = tols.index(state.angle_tolerance_deg) if state.angle_tolerance_deg in tols else 1
                state.angle_tolerance_deg = tols[(cur_i + 1) % len(tols)]
                print(f"Tolerance set to +/-{state.angle_tolerance_deg} deg")
            elif key in (ord("h"), ord("H")):
                holds = [0.5, 1.0, 1.5, 2.0]
                cur_i = holds.index(state.required_hold_s) if state.required_hold_s in holds else 1
                state.required_hold_s = holds[(cur_i + 1) % len(holds)]
                print(f"Required hold set to {state.required_hold_s}s")
            elif key in (ord("k"), ord("K")):
                state.keyboard_mode = not state.keyboard_mode
                if state.keyboard_mode:
                    state.kb_posture_id = state.current_template().id
                print(f"Keyboard mode: {'ON' if state.keyboard_mode else 'OFF'}")
            elif key in (ord("w"), ord("W")):
                state.kb_pos_m = (state.kb_pos_m[0], state.kb_pos_m[1] + 0.05)
            elif key in (ord("s"), ord("S")):
                state.kb_pos_m = (state.kb_pos_m[0], max(0.0, state.kb_pos_m[1] - 0.05))
            elif key in (ord("a"), ord("A")):
                state.kb_pos_m = (max(0.0, state.kb_pos_m[0] - 0.05), state.kb_pos_m[1])
            elif key in (ord("d"), ord("D")):
                state.kb_pos_m = (state.kb_pos_m[0] + 0.05, state.kb_pos_m[1])
            elif key == ord("["):
                state.kb_arm_offset_deg -= 10.0
                print(f"Synthetic arm offset: {state.kb_arm_offset_deg} deg")
            elif key == ord("]"):
                state.kb_arm_offset_deg += 10.0
                print(f"Synthetic arm offset: {state.kb_arm_offset_deg} deg")

    finally:
        if source:
            source.stop()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
