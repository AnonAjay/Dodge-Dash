"""Pose matching and joint angle extraction engine for Game B: Wall Shape.

Provides scale- and position-invariant joint angle extraction from 2D/3D landmarks,
standard wall cut-out pose templates, and tolerance-based similarity scoring.

Conventions:
  - Mirrored display: what the player sees on screen-left corresponds to left limbs.
  - Interior joint angles (degrees 0..180) calculated via vector dot products.
  - Dual matching: angular similarity score (0..100%) + optional arena floor position.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


# ==============================================================================
# Vector Mathematics
# ==============================================================================

def angle_between_vectors(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Calculate the interior angle (in degrees, 0..180) between two 2D or 3D vectors."""
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-7 or norm_b < 1e-7:
        return 0.0

    dot = float(np.dot(a, b))
    cos_theta = dot / (norm_a * norm_b)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(math.degrees(math.acos(cos_theta)))


def angle_3pts(
    pt_a: Sequence[float],
    pt_vertex_b: Sequence[float],
    pt_c: Sequence[float],
) -> float:
    """Calculate interior angle at vertex B formed by line segments B->A and B->C."""
    va = [a - b for a, b in zip(pt_a, pt_vertex_b)]
    vc = [c - b for c, b in zip(pt_c, pt_vertex_b)]
    return angle_between_vectors(va, vc)


# ==============================================================================
# Landmark Extraction & Data Structures
# ==============================================================================

# Standard MediaPipe 33 Landmark Indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


def _get_coord(lm: Any) -> tuple[float, float, float]:
    """Extract (x, y, z) coordinate tuple from landmark object, dict, or tuple."""
    if isinstance(lm, (tuple, list)):
        if len(lm) >= 3:
            return (float(lm[0]), float(lm[1]), float(lm[2]))
        if len(lm) == 2:
            return (float(lm[0]), float(lm[1]), 0.0)
    if hasattr(lm, "x") and hasattr(lm, "y"):
        z = getattr(lm, "z", 0.0)
        return (float(lm.x), float(lm.y), float(z))
    if isinstance(lm, dict) and "x" in lm and "y" in lm:
        return (float(lm["x"]), float(lm["y"]), float(lm.get("z", 0.0)))
    return (0.0, 0.0, 0.0)


def extract_pose_angles(landmarks: Sequence[Any]) -> dict[str, float]:
    """Extract all frontal joint angles in degrees (0..180) from a 33-point landmark list.

    Works identically with MediaPipe normalized landmarks, pixel landmarks, or Skeleton3D points.
    """
    if len(landmarks) < 29:
        return {}

    # Extract coordinates
    ls = _get_coord(landmarks[L_SHOULDER])
    rs = _get_coord(landmarks[R_SHOULDER])
    le = _get_coord(landmarks[L_ELBOW])
    re = _get_coord(landmarks[R_ELBOW])
    lw = _get_coord(landmarks[L_WRIST])
    rw = _get_coord(landmarks[R_WRIST])
    lh = _get_coord(landmarks[L_HIP])
    rh = _get_coord(landmarks[R_HIP])
    lk = _get_coord(landmarks[L_KNEE])
    rk = _get_coord(landmarks[R_KNEE])
    la = _get_coord(landmarks[L_ANKLE])
    ra = _get_coord(landmarks[R_ANKLE])

    # 1. Elbow angles (forearm relative to upper arm: 180 = straight, 90 = bent)
    left_elbow = angle_3pts(ls, le, lw)
    right_elbow = angle_3pts(rs, re, rw)

    # 2. Shoulder angles (upper arm relative to torso spine: 0 = down, 90 = T-pose, 180 = overhead)
    left_shoulder = angle_3pts(lh, ls, le)
    right_shoulder = angle_3pts(rh, rs, re)

    # 3. Knee angles (lower leg relative to thigh: 180 = straight, 90 = bent)
    left_knee = angle_3pts(lh, lk, la)
    right_knee = angle_3pts(rh, rk, ra)

    # 4. Hip angles (thigh relative to torso: 180 = straight standing, 90 = squat/seated)
    left_hip = angle_3pts(ls, lh, lk)
    right_hip = angle_3pts(rs, rh, rk)

    # 5. Torso spine inclination (angle of spine vector relative to horizontal)
    # Hip midpoint to shoulder midpoint
    spine_v = (
        (ls[0] + rs[0]) / 2.0 - (lh[0] + rh[0]) / 2.0,
        (ls[1] + rs[1]) / 2.0 - (lh[1] + rh[1]) / 2.0,
    )
    # Relative to vertical (0, -1 in image space, or 0, 1 in metric Z)
    torso_tilt = angle_between_vectors(spine_v, (0.0, -1.0))

    return {
        "left_elbow": round(left_elbow, 1),
        "right_elbow": round(right_elbow, 1),
        "left_shoulder": round(left_shoulder, 1),
        "right_shoulder": round(right_shoulder, 1),
        "left_knee": round(left_knee, 1),
        "right_knee": round(right_knee, 1),
        "left_hip": round(left_hip, 1),
        "right_hip": round(right_hip, 1),
        "torso_tilt": round(torso_tilt, 1),
    }


# ==============================================================================
# Wall Shape Pose Templates & Evaluation
# ==============================================================================

@dataclass
class JointComparison:
    name: str
    player_angle: float
    target_angle: float
    diff_deg: float
    is_matched: bool
    hint: str


@dataclass
class PoseTemplate:
    id: str
    name: str
    description: str
    expected_angles: dict[str, float]
    target_pos_m: tuple[float, float] | None = None  # Floor coordinate target (x, y) or None
    angle_tolerance_deg: float = 15.0                # Baseline tolerance (+/- degrees)
    pos_tolerance_m: float = 0.25                    # Metric distance tolerance (+/- metres)


@dataclass
class PoseMatchResult:
    match_pct: float                                 # 0.0 to 100.0%
    is_full_match: bool                              # True if all requirements met
    angle_score_pct: float                           # Angular similarity component
    pos_score_pct: float                             # Spatial position component
    pos_matched: bool                                # True if within position tolerance
    pos_dist_m: float                                # Distance to target floor position in metres
    joint_results: dict[str, JointComparison] = field(default_factory=dict)
    matched_joints_count: int = 0
    total_joints_count: int = 0
    primary_hint: str = ""


# Default Wall Shape Catalog (Frontal, scale-invariant poses)
DEFAULT_TEMPLATES: list[PoseTemplate] = [
    PoseTemplate(
        id="t_pose",
        name="T-Pose",
        description="Arms straight out horizontally at 90 deg, legs straight",
        expected_angles={
            "left_shoulder": 90.0,
            "right_shoulder": 90.0,
            "left_elbow": 175.0,
            "right_elbow": 175.0,
            "left_knee": 175.0,
            "right_knee": 175.0,
        },
    ),
    PoseTemplate(
        id="star",
        name="Star / Y-Pose",
        description="Arms raised high at 135 deg in a Y shape, straight knees",
        expected_angles={
            "left_shoulder": 135.0,
            "right_shoulder": 135.0,
            "left_elbow": 175.0,
            "right_elbow": 175.0,
            "left_knee": 175.0,
            "right_knee": 175.0,
        },
    ),
    PoseTemplate(
        id="cactus",
        name="Cactus / Goalpost",
        description="Shoulders horizontal (90 deg), elbows bent up at 90 deg",
        expected_angles={
            "left_shoulder": 90.0,
            "right_shoulder": 90.0,
            "left_elbow": 90.0,
            "right_elbow": 90.0,
            "left_knee": 175.0,
            "right_knee": 175.0,
        },
    ),
    PoseTemplate(
        id="disco_right",
        name="Disco Right",
        description="Right arm high up-diagonal (145 deg), left arm down (35 deg)",
        expected_angles={
            "right_shoulder": 145.0,
            "right_elbow": 175.0,
            "left_shoulder": 35.0,
            "left_elbow": 175.0,
            "left_knee": 175.0,
            "right_knee": 175.0,
        },
    ),
    PoseTemplate(
        id="disco_left",
        name="Disco Left",
        description="Left arm high up-diagonal (145 deg), right arm down (35 deg)",
        expected_angles={
            "left_shoulder": 145.0,
            "left_elbow": 175.0,
            "right_shoulder": 35.0,
            "right_elbow": 175.0,
            "left_knee": 175.0,
            "right_knee": 175.0,
        },
    ),
    PoseTemplate(
        id="tree_pose",
        name="Tree Balance",
        description="Arms in a Y, left leg straight, right knee bent at 75 deg",
        expected_angles={
            "left_shoulder": 135.0,
            "right_shoulder": 135.0,
            "left_knee": 175.0,
            "right_knee": 75.0,
        },
    ),
    PoseTemplate(
        id="l_pose",
        name="L-Pose",
        description="Left arm straight up (175 deg), right arm straight out (90 deg)",
        expected_angles={
            "left_shoulder": 175.0,
            "left_elbow": 175.0,
            "right_shoulder": 90.0,
            "right_elbow": 175.0,
            "left_knee": 175.0,
            "right_knee": 175.0,
        },
    ),
    PoseTemplate(
        id="squat",
        name="Crouch / Squat",
        description="Both knees bent at 105 deg, arms forward at 90 deg",
        expected_angles={
            "left_knee": 105.0,
            "right_knee": 105.0,
            "left_shoulder": 85.0,
            "right_shoulder": 85.0,
            "left_elbow": 170.0,
            "right_elbow": 170.0,
        },
    ),
]


def evaluate_pose(
    player_angles: dict[str, float],
    player_pos_m: tuple[float, float] | None,
    template: PoseTemplate,
    angle_tolerance_override: float | None = None,
    pos_tolerance_override: float | None = None,
) -> PoseMatchResult:
    """Compare player joint angles and floor position against a Wall Shape template.

    Returns detailed JointComparison metrics, similarity scores (0..100%), and guidance hints.
    """
    angle_tol = angle_tolerance_override if angle_tolerance_override is not None else template.angle_tolerance_deg
    pos_tol = pos_tolerance_override if pos_tolerance_override is not None else template.pos_tolerance_m

    joint_results: dict[str, JointComparison] = {}
    joint_scores: list[float] = []
    hints: list[str] = []

    for j_name, tgt_angle in template.expected_angles.items():
        act_angle = player_angles.get(j_name, 0.0)
        diff = abs(act_angle - tgt_angle)
        is_matched = diff <= angle_tol

        # Hint generation
        hint = "OK"
        if not is_matched:
            if "elbow" in j_name:
                hint = "Straighten arm" if act_angle < tgt_angle else "Bend elbow"
            elif "shoulder" in j_name:
                hint = "Raise arm higher" if act_angle < tgt_angle else "Lower arm"
            elif "knee" in j_name:
                hint = "Straighten leg" if act_angle < tgt_angle else "Bend knee"
            elif "hip" in j_name:
                hint = "Stand upright" if act_angle < tgt_angle else "Squat lower"
            hints.append(f"{j_name.replace('_', ' ').title()}: {hint}")

        # Linear scoring falloff: 100% within 5 deg, down to 0% at 2 * tol
        if diff <= 5.0:
            score = 100.0
        elif diff <= angle_tol:
            score = 100.0 - 30.0 * ((diff - 5.0) / max(1.0, angle_tol - 5.0))
        else:
            score = max(0.0, 70.0 - 70.0 * ((diff - angle_tol) / max(1.0, angle_tol)))

        joint_scores.append(score)
        joint_results[j_name] = JointComparison(
            name=j_name,
            player_angle=act_angle,
            target_angle=tgt_angle,
            diff_deg=round(diff, 1),
            is_matched=is_matched,
            hint=hint,
        )

    matched_count = sum(1 for j in joint_results.values() if j.is_matched)
    total_count = len(joint_results)
    angle_score_pct = float(np.mean(joint_scores)) if joint_scores else 0.0

    # Position evaluation
    pos_dist = 0.0
    pos_matched = True
    pos_score_pct = 100.0

    if template.target_pos_m is not None:
        if player_pos_m is not None:
            pos_dist = math.hypot(
                player_pos_m[0] - template.target_pos_m[0],
                player_pos_m[1] - template.target_pos_m[1],
            )
            pos_matched = pos_dist <= pos_tol
            pos_score_pct = max(0.0, 100.0 - (pos_dist / (2.0 * pos_tol)) * 100.0)
            if not pos_matched:
                dx = template.target_pos_m[0] - player_pos_m[0]
                dy = template.target_pos_m[1] - player_pos_m[1]
                p_hint = "Step Left" if dx < 0 else "Step Right"
                if abs(dy) > abs(dx):
                    p_hint = "Step Forward" if dy > 0 else "Step Back"
                hints.insert(0, f"Position: {p_hint}")
        else:
            pos_matched = False
            pos_score_pct = 0.0
            hints.insert(0, "Position: Step inside arena")

    # Overall Combined Match Percentage
    if template.target_pos_m is not None:
        match_pct = round(0.75 * angle_score_pct + 0.25 * pos_score_pct, 1)
    else:
        match_pct = round(angle_score_pct, 1)

    # Full match threshold: average angle score >= 80% and all critical joints matched
    is_full_match = (matched_count == total_count) and pos_matched and (match_pct >= 80.0)
    primary_hint = hints[0] if hints else "PERFECT POSE!"

    return PoseMatchResult(
        match_pct=match_pct,
        is_full_match=is_full_match,
        angle_score_pct=round(angle_score_pct, 1),
        pos_score_pct=round(pos_score_pct, 1),
        pos_matched=pos_matched,
        pos_dist_m=round(pos_dist, 2),
        joint_results=joint_results,
        matched_joints_count=matched_count,
        total_joints_count=total_count,
        primary_hint=primary_hint,
    )


# ==============================================================================
# Non-Breaking Integration Helpers (for Station B and Test Tools)
# ==============================================================================

def extract_angles_from_skeleton(skeleton: Any) -> dict[str, float]:
    """Extract joint angles directly from a Skeleton3D or list of 33 landmarks.

    Scale- and position-invariant; works with metric 3D skeleton points,
    MediaPipe normalized landmarks, or image pixel landmarks.
    """
    if skeleton is None:
        return {}
    if hasattr(skeleton, "points_m"):
        return extract_pose_angles(skeleton.points_m)
    if isinstance(skeleton, (list, tuple)):
        return extract_pose_angles(skeleton)
    return {}


def snapshot_pose_match(
    source: Any,
    template: PoseTemplate,
    angle_tolerance_override: float | None = None,
    pos_tolerance_override: float | None = None,
) -> tuple[PoseMatchResult, Any, Any, dict]:
    """One-line helper for Station B and test tools to evaluate pose match.

    Extracts the latest skeleton, calculates real-time joint angles, evaluates
    against the target template, and returns:
        (PoseMatchResult, PoseFrame, camera_frame, debug_dict)

    Avoids any breaking changes to shared/contracts.py or pose_core/capture.py.
    """
    frame = None
    debug: dict = {}
    skel = None
    pf = None

    if hasattr(source, "snapshot3d"):
        pf, skel, frame, debug = source.snapshot3d()
        angles = extract_angles_from_skeleton(skel)
    elif hasattr(source, "snapshot"):
        pf, frame, debug = source.snapshot()
        pts = debug.get("pts")
        angles = extract_pose_angles(pts) if pts else {}
    else:
        pf_fn = getattr(source, "latest", None)
        pf = pf_fn() if callable(pf_fn) else None
        skel_fn = getattr(source, "latest_skeleton", None)
        skel = skel_fn() if callable(skel_fn) else None
        angles = extract_angles_from_skeleton(skel)

    player_pos = None
    if pf and getattr(pf, "tracked", False):
        fx = getattr(pf, "floor_x_m", None)
        fy = getattr(pf, "floor_y_m", None)
        if fx is not None and fy is not None:
            player_pos = (fx, fy)

    result = evaluate_pose(
        player_angles=angles,
        player_pos_m=player_pos,
        template=template,
        angle_tolerance_override=angle_tolerance_override,
        pos_tolerance_override=pos_tolerance_override,
    )
    return result, pf, frame, debug

