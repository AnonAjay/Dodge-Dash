"""Pose Blazer: Pose-to-Pose Distance Metric, Scoring, and Catalog Management.

Provides:
  - BlazerPoseTemplate: Schema for indexed pose challenges.
  - BlazerMatchResult: Real-time match telemetry, joint scores, and directional hints.
  - evaluate_blazer_pose: Weighted joint comparison with key-limb prioritization.
  - Catalog management: Load, save, and sample poses for 5-round runs.
  - render_reference_card: High-tech arcade neon card renderer with image/vector fallback.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from pose_core.pose_matcher import (
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    angle_3pts,
    extract_pose_angles,
)
from pose_core.skeleton import BONES

# Default tolerances and thresholds
DEFAULT_PASS_THRESHOLD_PCT = 75.0
DEFAULT_ANGLE_TOLERANCE_DEG = 25.0
KEY_JOINT_WEIGHT_MULTIPLIER = 2.5
STANDARD_JOINT_WEIGHT = 1.0

# 8 Core Joints tracked in frontal poses
CORE_JOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
)

# Standard BGR Colors for arcade UI
C_NEON_GREEN_BGR = (50, 240, 100)
C_AMBER_YELLOW_BGR = (0, 215, 255)
C_ALERT_RED_BGR = (40, 40, 240)
C_NEON_CYAN_BGR = (255, 220, 0)
C_DARK_BG_BGR = (18, 20, 28)
C_PANEL_BG_BGR = (28, 32, 44)


@dataclass
class BlazerPoseTemplate:
    """A target pose challenge in Pose Blazer."""
    id: str
    name: str
    image_file: str | None = None
    target_angles: dict[str, float] = field(default_factory=dict)
    key_joints: list[str] = field(default_factory=list)
    landmarks_norm: list[list[float]] = field(default_factory=list)
    difficulty: str = "medium"  # "easy" | "medium" | "hard"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlazerPoseTemplate:
        return cls(
            id=data["id"],
            name=data["name"],
            image_file=data.get("image_file"),
            target_angles={k: float(v) for k, v in data.get("target_angles", {}).items() if v is not None},
            key_joints=list(data.get("key_joints", [])),
            landmarks_norm=list(data.get("landmarks_norm", [])),
            difficulty=data.get("difficulty", "medium"),
            description=data.get("description", ""),
        )


@dataclass
class BlazerMatchResult:
    """Real-time match telemetry between live player and target pose."""
    match_pct: float
    is_matched: bool
    joint_diffs: dict[str, float]
    joint_scores: dict[str, float]
    joint_status: dict[str, str]  # "matched" | "close" | "off"
    joint_colors_bgr: dict[str, tuple[int, int, int]]
    primary_hint: str
    head_aligned: bool
    key_joints: list[str]
    missing_joints: list[str] = field(default_factory=list)


def evaluate_blazer_pose(
    player_landmarks: Sequence[Any] | dict[str, float | None],
    target_pose: BlazerPoseTemplate | dict[str, Any],
    player_head_norm: tuple[float, float] | None = None,
    mirrored: bool | None = None,
    pass_threshold_pct: float = DEFAULT_PASS_THRESHOLD_PCT,
    angle_tolerance_deg: float = DEFAULT_ANGLE_TOLERANCE_DEG,
) -> BlazerMatchResult:
    """Compare a player's live pose against a target BlazerPoseTemplate.

    Args:
        player_landmarks: Either a 33-point MediaPipe landmark list, or a pre-extracted
                          angle dictionary {"left_shoulder": 90.0, ...}.
        target_pose: The target challenge (BlazerPoseTemplate instance or raw dict).
        player_head_norm: (x, y) normalized coordinate of head (0..1) for standing boundary.
        mirrored: Mirroring convention. If True, screen-left = player physical left.
        pass_threshold_pct: Minimum percentage (e.g. 75.0) required to consider matched.
        angle_tolerance_deg: Angular margin of error per joint (default +/- 25 degrees).

    Returns:
        BlazerMatchResult with scores, colors, pass/fail status, and directional hints.
    """
    if isinstance(target_pose, dict):
        target = BlazerPoseTemplate.from_dict(target_pose)
    else:
        target = target_pose

    # Extract or parse player angles
    if isinstance(player_landmarks, dict):
        player_angles = {k: float(v) for k, v in player_landmarks.items() if v is not None}
    else:
        player_angles = extract_pose_angles(player_landmarks, mirrored=mirrored, planar_2d=True)

    target_angles = target.target_angles
    key_joints_set = set(target.key_joints)

    # Check Head Alignment in the pre-defined target standing box
    # Lateral center = 0.50 +/- 0.09; Vertical range = 0.10 .. 0.40
    head_aligned = True
    if player_head_norm is not None:
        hx, hy = player_head_norm
        head_aligned = (abs(hx - 0.50) <= 0.09) and (0.10 <= hy <= 0.42)

    joint_diffs: dict[str, float] = {}
    joint_scores: dict[str, float] = {}
    joint_status: dict[str, str] = {}
    joint_colors: dict[str, tuple[int, int, int]] = {}
    missing_joints: list[str] = []

    total_weight = 0.0
    weighted_score_sum = 0.0

    worst_joint: str | None = None
    worst_diff = 0.0
    worst_is_key = False

    for joint in CORE_JOINTS:
        if joint not in target_angles:
            continue

        tgt_val = target_angles[joint]
        is_key = joint in key_joints_set
        weight = KEY_JOINT_WEIGHT_MULTIPLIER if is_key else STANDARD_JOINT_WEIGHT

        if joint not in player_angles or player_angles[joint] is None:
            # Occluded or untracked joint
            missing_joints.append(joint)
            joint_diffs[joint] = 999.0
            joint_scores[joint] = 0.0
            joint_status[joint] = "off"
            joint_colors[joint] = C_ALERT_RED_BGR
            total_weight += weight
            continue

        act_val = float(player_angles[joint])
        diff = abs(act_val - tgt_val)
        joint_diffs[joint] = round(diff, 1)

        # Linear decay score: 100% at diff=0 -> 0% at diff=tolerance
        score = max(0.0, 100.0 - (diff / angle_tolerance_deg) * 100.0)
        joint_scores[joint] = round(score, 1)

        # Color coding: Green within 18 deg, Amber within 32 deg, Red beyond
        if diff <= (angle_tolerance_deg * 0.72):
            status = "matched"
            col = C_NEON_GREEN_BGR
        elif diff <= (angle_tolerance_deg * 1.30):
            status = "close"
            col = C_AMBER_YELLOW_BGR
        else:
            status = "off"
            col = C_ALERT_RED_BGR

        joint_status[joint] = status
        joint_colors[joint] = col

        total_weight += weight
        weighted_score_sum += weight * score

        # Track worst joint for directional hint (prioritizing key joints)
        if diff > angle_tolerance_deg * 0.60:
            if is_key and not worst_is_key:
                worst_joint = joint
                worst_diff = diff
                worst_is_key = True
            elif is_key == worst_is_key and diff > worst_diff:
                worst_joint = joint
                worst_diff = diff

    # Compute overall match percentage
    if total_weight > 0.0:
        match_pct = round(weighted_score_sum / total_weight, 1)
    else:
        match_pct = 0.0

    # Clean match condition: match score >= threshold AND head aligned
    is_matched = (match_pct >= pass_threshold_pct) and head_aligned

    # Generate actionable directional hint
    primary_hint = _generate_hint(
        worst_joint,
        player_angles.get(worst_joint) if worst_joint else None,
        target_angles.get(worst_joint) if worst_joint else None,
        head_aligned,
        player_head_norm,
    )

    return BlazerMatchResult(
        match_pct=match_pct,
        is_matched=is_matched,
        joint_diffs=joint_diffs,
        joint_scores=joint_scores,
        joint_status=joint_status,
        joint_colors_bgr=joint_colors,
        primary_hint=primary_hint,
        head_aligned=head_aligned,
        key_joints=list(target.key_joints),
        missing_joints=missing_joints,
    )


def _generate_hint(
    worst_joint: str | None,
    actual_deg: float | None,
    target_deg: float | None,
    head_aligned: bool,
    head_norm: tuple[float, float] | None,
) -> str:
    """Generate human-readable directional feedback for the player."""
    if not head_aligned and head_norm is not None:
        hx, _ = head_norm
        if hx > 0.59:
            return "Step Left to center inside target box"
        if hx < 0.41:
            return "Step Right to center inside target box"
        return "Align Head inside target box"

    if worst_joint is None or actual_deg is None or target_deg is None:
        return "Mirror the target pose!"

    diff = actual_deg - target_deg
    j_name = worst_joint.replace("_", " ").title()

    if "Elbow" in j_name:
        if diff > 0:
            return f"Bend {j_name.split()[0]} Arm"
        return f"Straighten {j_name.split()[0]} Arm"

    if "Shoulder" in j_name:
        if diff < 0:
            return f"Raise {j_name.split()[0]} Arm Higher"
        return f"Lower {j_name.split()[0]} Arm"

    if "Knee" in j_name:
        if diff > 0:
            return f"Bend {j_name.split()[0]} Knee"
        return f"Straighten {j_name.split()[0]} Leg"

    if "Hip" in j_name:
        if diff > 0:
            return f"Squat / Bend {j_name.split()[0]} Hip"
        return f"Stand Taller on {j_name.split()[0]} Leg"

    return f"Adjust {j_name}"


# ==============================================================================
# Catalog Serialization & Sampling
# ==============================================================================

def load_blazer_catalog(path: str | Path = "data/poses_index.json") -> list[BlazerPoseTemplate]:
    """Load the pose catalog from JSON. Returns empty list if file not found."""
    p = Path(path)
    if not p.is_file():
        return []

    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return [BlazerPoseTemplate.from_dict(item) for item in raw]
        if isinstance(raw, dict) and "poses" in raw:
            return [BlazerPoseTemplate.from_dict(item) for item in raw["poses"]]
    except Exception as e:
        print(f"Warning: Failed to load blazer catalog from {p}: {e}")
    return []


def save_blazer_catalog(catalog: list[BlazerPoseTemplate], path: str | Path = "data/poses_index.json") -> None:
    """Save the pose catalog to JSON with indentation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = [p.to_dict() for p in catalog]
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def sample_game_poses(
    catalog: list[BlazerPoseTemplate],
    n: int = 5,
    rng: random.Random | None = None,
) -> list[BlazerPoseTemplate]:
    """Sample n distinct poses for a 5-round game run.

    If catalog has fewer than n poses, samples with replacement.
    """
    if not catalog:
        raise ValueError("Cannot sample poses from an empty catalog.")

    r = rng if rng is not None else random
    if len(catalog) >= n:
        return r.sample(catalog, n)
    return [r.choice(catalog) for _ in range(n)]


# ==============================================================================
# Visual Reference Card Renderer (Image + Vector Fallback)
# ==============================================================================

def render_reference_card(
    target_pose: BlazerPoseTemplate,
    width: int = 360,
    height: int = 480,
    poses_dir: str | Path = "data/poses",
) -> np.ndarray:
    """Render a high-tech arcade pose card.

    If target_pose.image_file exists in poses_dir, renders the photo with arcade
    borders and badges. If image is missing, renders a glowing neon wireframe
    mannequin from target_pose.landmarks_norm on a dark grid.
    """
    card = np.zeros((height, width, 3), dtype=np.uint8)
    card[:] = C_DARK_BG_BGR

    # Try loading reference image
    img_loaded = False
    if target_pose.image_file:
        img_p = Path(poses_dir) / target_pose.image_file
        if img_p.is_file():
            ref_img = cv2.imread(str(img_p))
            if ref_img is not None and ref_img.size > 0:
                card = _letterbox_image(ref_img, width, height)
                img_loaded = True

    if not img_loaded:
        # Fallback: Draw Neon Vector Mannequin
        _render_vector_skeleton(card, target_pose.landmarks_norm, target_pose.key_joints)

    # Outer Neon Frame & Header Badge
    cv2.rectangle(card, (0, 0), (width - 1, height - 1), (50, 60, 80), 2)
    cv2.rectangle(card, (2, 2), (width - 3, height - 3), C_NEON_CYAN_BGR, 1)

    # Top Header Banner
    banner_h = 44
    cv2.rectangle(card, (0, 0), (width, banner_h), C_PANEL_BG_BGR, -1)
    cv2.line(card, (0, banner_h), (width, banner_h), C_NEON_CYAN_BGR, 2)

    # Pose Title
    title = target_pose.name.upper()
    cv2.putText(card, title, (14, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # Difficulty Pill
    diff_text = target_pose.difficulty.upper()
    diff_col = C_NEON_GREEN_BGR if diff_text == "EASY" else (C_AMBER_YELLOW_BGR if diff_text == "MEDIUM" else C_ALERT_RED_BGR)
    cv2.putText(card, diff_text, (width - 78, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, diff_col, 1, cv2.LINE_AA)

    # Bottom Key Limbs Pill
    if target_pose.key_joints:
        key_str = "KEY: " + ", ".join(j.split("_")[0][0].upper() + " " + j.split("_")[1].title() for j in target_pose.key_joints[:3])
        cv2.rectangle(card, (0, height - 28), (width, height), (15, 18, 24), -1)
        cv2.putText(card, key_str, (12, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_NEON_CYAN_BGR, 1, cv2.LINE_AA)

    return card


def _letterbox_image(img: np.ndarray, tgt_w: int, tgt_h: int) -> np.ndarray:
    """Scale image to fit within target canvas, preserving aspect ratio."""
    canvas = np.zeros((tgt_h, tgt_w, 3), dtype=np.uint8)
    canvas[:] = C_DARK_BG_BGR

    ih, iw = img.shape[:2]
    scale = min((tgt_w - 20) / iw, (tgt_h - 70) / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    x0 = (tgt_w - nw) // 2
    y0 = 46 + ((tgt_h - 74) - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _render_vector_skeleton(
    canvas: np.ndarray,
    landmarks_norm: Sequence[Sequence[float]],
    key_joints: Sequence[str],
) -> None:
    """Render arcade wireframe skeleton from normalized (0..1) landmarks."""
    h, w = canvas.shape[:2]

    # Draw subtle background grid
    for gx in range(0, w, 32):
        cv2.line(canvas, (gx, 44), (gx, h - 28), (25, 30, 42), 1)
    for gy in range(44, h - 28, 32):
        cv2.line(canvas, (0, gy), (w, gy), (25, 30, 42), 1)

    if not landmarks_norm or len(landmarks_norm) < 29:
        cv2.putText(canvas, "NO SKELETON DATA", (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        return

    # Map landmarks to canvas coordinates
    draw_w = w * 0.76
    draw_h = (h - 76) * 0.85
    ox = (w - draw_w) / 2.0
    oy = 52.0

    pts_px: list[tuple[int, int]] = []
    for lm in landmarks_norm:
        lx = float(lm[0])
        ly = float(lm[1])
        px = int(ox + lx * draw_w)
        py = int(oy + ly * draw_h)
        pts_px.append((px, py))

    # Draw core bones
    for j1, j2 in BONES:
        if j1 < len(pts_px) and j2 < len(pts_px):
            p1, p2 = pts_px[j1], pts_px[j2]
            cv2.line(canvas, p1, p2, (0, 140, 220), 3, cv2.LINE_AA)
            cv2.line(canvas, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw joints (Key joints glow vibrant yellow/amber)
    key_indices = set()
    for kj in key_joints:
        if "shoulder" in kj:
            key_indices.update((L_SHOULDER, R_SHOULDER))
        if "elbow" in kj:
            key_indices.update((L_ELBOW, R_ELBOW))
        if "knee" in kj:
            key_indices.update((L_KNEE, R_KNEE))
        if "hip" in kj:
            key_indices.update((L_HIP, R_HIP))

    for idx, (px, py) in enumerate(pts_px):
        if idx in key_indices:
            cv2.circle(canvas, (px, py), 7, C_AMBER_YELLOW_BGR, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 10, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (px, py), 4, C_NEON_CYAN_BGR, -1, cv2.LINE_AA)
