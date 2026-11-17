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

def angle_between_vectors(v1: Sequence[float], v2: Sequence[float]) -> float | None:
    """Calculate the interior angle (in degrees, 0..180) between two 2D or 3D vectors.

    Returns None for (near) zero-length vectors.
    """
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-7 or norm_b < 1e-7:
        return None

    dot = float(np.dot(a, b))
    cos_theta = dot / (norm_a * norm_b)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(math.degrees(math.acos(cos_theta)))


def angle_3pts(
    pt_a: Sequence[float] | None,
    pt_vertex_b: Sequence[float] | None,
    pt_c: Sequence[float] | None,
) -> float | None:
    """Calculate interior angle at vertex B formed by line segments B->A and B->C."""
    if pt_a is None or pt_vertex_b is None or pt_c is None:
        return None
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


def _get_vis(lm: Any, idx: int, visibility: Sequence[float] | None) -> float:
    """Extract visibility score (0..1) for a landmark."""
    if visibility is not None:
        if idx < len(visibility):
            return float(visibility[idx])
        return 0.0
    if hasattr(lm, "visibility"):
        return float(getattr(lm, "visibility"))
    if isinstance(lm, dict) and "visibility" in lm:
        return float(lm["visibility"])
    if isinstance(lm, (tuple, list)) and len(lm) >= 3:
        # Check if 3rd element could be visibility (0..1)
        # However, for 3D coordinates z is typically metric, so only use if explicit visibility or hasattr
        pass
    return 1.0


def extract_pose_angles(
    landmarks: Sequence[Any],
    visibility: Sequence[float] | None = None,
    min_vis: float = 0.5,
    mirrored: bool | None = None,
) -> dict[str, float | None]:
    """Extract all frontal joint angles in degrees (0..180) from a 33-point landmark list.

    Works with mirrored live camera landmarks, unmirrored landmarks, or Skeleton3D points.
    If mirrored is None, automatically detects whether landmarks are mirrored (e.g. from cv2.flip):
    in a mirrored view, landmark 12 (MediaPipe's right shoulder) is on the screen-left (smaller x)
    which corresponds to the player's physical left arm.
    """
    if len(landmarks) < 29:
        return {}

    # Check mirror orientation if not explicitly provided
    if mirrored is None:
        p11 = _get_coord(landmarks[L_SHOULDER])
        p12 = _get_coord(landmarks[R_SHOULDER])
        # In a mirrored camera view, landmark 12 is on screen-left (smaller x),
        # meaning MediaPipe's 'right' (12) is the player's physical left.
        mirrored = (p12[0] < p11[0])

    if mirrored:
        # Player's Physical Left (screen-left) uses MediaPipe indices (12, 14, 16, 24, 26, 28)
        # Player's Physical Right (screen-right) uses MediaPipe indices (11, 13, 15, 23, 25, 27)
        idx_l_sh, idx_r_sh = R_SHOULDER, L_SHOULDER
        idx_l_el, idx_r_el = R_ELBOW, L_ELBOW
        idx_l_wr, idx_r_wr = R_WRIST, L_WRIST
        idx_l_hip, idx_r_hip = R_HIP, L_HIP
        idx_l_kn, idx_r_kn = R_KNEE, L_KNEE
        idx_l_ank, idx_r_ank = R_ANKLE, L_ANKLE
    else:
        idx_l_sh, idx_r_sh = L_SHOULDER, R_SHOULDER
        idx_l_el, idx_r_el = L_ELBOW, R_ELBOW
        idx_l_wr, idx_r_wr = L_WRIST, R_WRIST
        idx_l_hip, idx_r_hip = L_HIP, R_HIP
        idx_l_kn, idx_r_kn = L_KNEE, R_KNEE
        idx_l_ank, idx_r_ank = L_ANKLE, R_ANKLE

    def joint_ang(p1_idx: int, vertex_idx: int, p2_idx: int) -> float | None:
        v1 = _get_vis(landmarks[p1_idx], p1_idx, visibility)
        v2 = _get_vis(landmarks[vertex_idx], vertex_idx, visibility)
        v3 = _get_vis(landmarks[p2_idx], p2_idx, visibility)
        if v1 < min_vis or v2 < min_vis or v3 < min_vis:
            return None
        p1 = _get_coord(landmarks[p1_idx])
        pv = _get_coord(landmarks[vertex_idx])
        p2 = _get_coord(landmarks[p2_idx])
        ang = angle_3pts(p1, pv, p2)
        return round(ang, 1) if ang is not None else None

    # 1. Elbow angles (forearm relative to upper arm: 180 = straight, 90 = bent)
    left_elbow = joint_ang(idx_l_sh, idx_l_el, idx_l_wr)
    right_elbow = joint_ang(idx_r_sh, idx_r_el, idx_r_wr)

    # 2. Shoulder angles (upper arm relative to torso spine: 0 = down, 90 = T-pose, 180 = overhead)
    left_shoulder = joint_ang(idx_l_hip, idx_l_sh, idx_l_el)
    right_shoulder = joint_ang(idx_r_hip, idx_r_sh, idx_r_el)

    # 3. Knee angles (lower leg relative to thigh: 180 = straight, 90 = bent)
    left_knee = joint_ang(idx_l_hip, idx_l_kn, idx_l_ank)
    right_knee = joint_ang(idx_r_hip, idx_r_kn, idx_r_ank)

    # 4. Hip angles (thigh relative to torso: 180 = straight standing, 90 = squat/seated)
    left_hip = joint_ang(idx_l_sh, idx_l_hip, idx_l_kn)
    right_hip = joint_ang(idx_r_sh, idx_r_hip, idx_r_kn)

    # 5. Torso spine inclination (angle of spine vector relative to vertical)
    ls_vis = _get_vis(landmarks[L_SHOULDER], L_SHOULDER, visibility)
    rs_vis = _get_vis(landmarks[R_SHOULDER], R_SHOULDER, visibility)
    lh_vis = _get_vis(landmarks[L_HIP], L_HIP, visibility)
    rh_vis = _get_vis(landmarks[R_HIP], R_HIP, visibility)
    if ls_vis < min_vis or rs_vis < min_vis or lh_vis < min_vis or rh_vis < min_vis:
        torso_tilt = None
    else:
        ls = _get_coord(landmarks[L_SHOULDER])
        rs = _get_coord(landmarks[R_SHOULDER])
        lh = _get_coord(landmarks[L_HIP])
        rh = _get_coord(landmarks[R_HIP])
        dx = (ls[0] + rs[0]) / 2.0 - (lh[0] + rh[0]) / 2.0
        if abs(ls[2]) > 0.05 or abs(lh[2]) > 0.05:
            # 3D floor coordinates: Z is vertical height (upward is positive)
            dz = (ls[2] + rs[2]) / 2.0 - (lh[2] + rh[2]) / 2.0
            ang = angle_between_vectors((dx, dz), (0.0, 1.0))
        else:
            # 2D image coordinates: Y is vertical downwards (upward is negative)
            dy = (ls[1] + rs[1]) / 2.0 - (lh[1] + rh[1]) / 2.0
            ang = angle_between_vectors((dx, dy), (0.0, -1.0))
        torso_tilt = round(ang, 1) if ang is not None else None

    return {
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_knee": left_knee,
        "right_knee": right_knee,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "torso_tilt": torso_tilt,
    }


# ==============================================================================
# Wall Shape Pose Templates & Evaluation
# ==============================================================================

@dataclass
class JointComparison:
    name: str
    player_angle: float | None
    target_angle: float
    diff_deg: float | None
    is_matched: bool
    hint: str


@dataclass
class PoseTemplate:
    id: str
    name: str
    description: str
    expected_angles: dict[str, float]
    target_pos_m: tuple[float, float] | None = None  # Floor coordinate target (x, y) or None
    angle_tolerance_deg: float = 20.0                # Baseline tolerance (+/- degrees)
    pos_tolerance_m: float = 0.40                    # Metric distance tolerance (+/- metres)
    joint_tolerances: dict[str, float] = field(default_factory=dict)  # Per-joint custom tolerances
    cue: str = ""                                    # Clear human guidance cue


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
    coverage: float = 1.0                            # visible joints / expected joints, 0.0..1.0
    reliable: bool = True                            # coverage >= 0.7 and player_tracked


# Default Wall Shape Catalog (Frontal, scale-invariant poses)
DEFAULT_TEMPLATES: list[PoseTemplate] = [
    PoseTemplate(
        id="t_pose",
        name="T-Pose",
        description="Arms straight out horizontally at 95 deg, legs straight",
        expected_angles={
            "left_shoulder": 95.0,
            "right_shoulder": 95.0,
            "left_elbow": 170.0,
            "right_elbow": 170.0,
            "left_knee": 165.0,
            "right_knee": 165.0,
        },
        target_pos_m=(0.62, 0.62),
        angle_tolerance_deg=20.0,
        pos_tolerance_m=0.40,
        joint_tolerances={"left_elbow": 25.0, "right_elbow": 25.0},
        cue="Arms straight out horizontally, legs straight",
    ),
    PoseTemplate(
        id="star",
        name="Star / Y-Pose",
        description="Arms raised high at 135 deg in a Y shape, straight knees",
        expected_angles={
            "left_shoulder": 135.0,
            "right_shoulder": 135.0,
            "left_elbow": 170.0,
            "right_elbow": 170.0,
            "left_knee": 165.0,
            "right_knee": 165.0,
        },
        target_pos_m=(0.62, 0.62),
        angle_tolerance_deg=20.0,
        pos_tolerance_m=0.40,
        joint_tolerances={"left_elbow": 25.0, "right_elbow": 25.0, "left_shoulder": 22.0, "right_shoulder": 22.0},
        cue="Arms high in a Y shape, legs straight",
    ),
    PoseTemplate(
        id="cactus",
        name="Cactus / Goalpost",
        description="Shoulders horizontal (95 deg), elbows bent up at 90 deg",
        expected_angles={
            "left_shoulder": 95.0,
            "right_shoulder": 95.0,
            "left_elbow": 90.0,
            "right_elbow": 90.0,
            "left_knee": 165.0,
            "right_knee": 165.0,
        },
        target_pos_m=(0.62, 0.62),
        angle_tolerance_deg=20.0,
        pos_tolerance_m=0.40,
        joint_tolerances={"left_elbow": 25.0, "right_elbow": 25.0},
        cue="Shoulders level, elbows bent UP at 90 deg",
    ),
    PoseTemplate(
        id="disco_right",
        name="Disco Right",
        description="Right arm high up-diagonal (145 deg), left arm down (35 deg)",
        expected_angles={
            "right_shoulder": 145.0,
            "right_elbow": 170.0,
            "left_shoulder": 35.0,
            "left_elbow": 170.0,
            "left_knee": 165.0,
            "right_knee": 165.0,
        },
        target_pos_m=(0.62, 0.62),
        angle_tolerance_deg=20.0,
        pos_tolerance_m=0.40,
        joint_tolerances={"left_elbow": 25.0, "right_elbow": 25.0, "left_knee": 25.0, "right_knee": 25.0},
        cue="Raise RIGHT arm UP, keep LEFT arm DOWN",
    ),
    PoseTemplate(
        id="disco_left",
        name="Disco Left",
        description="Left arm high up-diagonal (145 deg), right arm down (35 deg)",
        expected_angles={
            "left_shoulder": 145.0,
            "left_elbow": 170.0,
            "right_shoulder": 35.0,
            "right_elbow": 170.0,
            "left_knee": 165.0,
            "right_knee": 165.0,
        },
        target_pos_m=(0.62, 0.62),
        angle_tolerance_deg=20.0,
        pos_tolerance_m=0.40,
        joint_tolerances={"left_elbow": 25.0, "right_elbow": 25.0, "left_knee": 25.0, "right_knee": 25.0},
        cue="Raise LEFT arm UP, keep RIGHT arm DOWN",
    ),
    PoseTemplate(
        id="tree_pose",
        name="Tree Balance",
        description="Arms in a Y, left leg straight, right knee bent at 75 deg",
        expected_angles={
            "left_shoulder": 135.0,
            "right_shoulder": 135.0,
            "left_knee": 165.0,
            "right_knee": 75.0,
        },
        target_pos_m=(0.62, 0.62),
        angle_tolerance_deg=20.0,
        pos_tolerance_m=0.40,
        joint_tolerances={"right_knee": 30.0, "left_knee": 25.0, "left_shoulder": 22.0, "right_shoulder": 22.0},
        cue="Arms high, stand on LEFT leg, bend RIGHT knee",
    ),
    PoseTemplate(
        id="l_pose",
        name="L-Pose",
        description="Left arm straight up (170 deg), right arm straight out (95 deg)",
        expected_angles={
            "left_shoulder": 170.0,
            "left_elbow": 170.0,
            "right_shoulder": 95.0,
            "right_elbow": 170.0,
            "left_knee": 165.0,
            "right_knee": 165.0,
        },
        target_pos_m=(0.62, 0.62),
        angle_tolerance_deg=20.0,
        pos_tolerance_m=0.40,
        joint_tolerances={"left_elbow": 25.0, "right_elbow": 25.0},
        cue="LEFT arm straight UP, RIGHT arm horizontal",
    ),
    PoseTemplate(
        id="squat",
        name="Crouch / Squat",
        description="Both knees bent at 105 deg, arms forward at 85 deg",
        expected_angles={
            "left_knee": 105.0,
            "right_knee": 105.0,
            "left_shoulder": 85.0,
            "right_shoulder": 85.0,
            "left_elbow": 170.0,
            "right_elbow": 170.0,
        },
        target_pos_m=(0.62, 0.62),
        angle_tolerance_deg=20.0,
        pos_tolerance_m=0.40,
        joint_tolerances={"left_knee": 25.0, "right_knee": 25.0, "left_elbow": 25.0, "right_elbow": 25.0},
        cue="Bend both knees into a squat, arms forward",
    ),
]


def make_untracked_result(template: PoseTemplate, hint: str = "Player not tracked") -> PoseMatchResult:
    """Return an unreliable result with coverage=0.0 and all joint actuals None."""
    joint_results: dict[str, JointComparison] = {}
    for j_name, tgt_angle in template.expected_angles.items():
        joint_results[j_name] = JointComparison(
            name=j_name,
            player_angle=None,
            target_angle=tgt_angle,
            diff_deg=None,
            is_matched=False,
            hint="not visible",
        )
    return PoseMatchResult(
        match_pct=0.0,
        is_full_match=False,
        angle_score_pct=0.0,
        pos_score_pct=0.0,
        pos_matched=False,
        pos_dist_m=0.0,
        joint_results=joint_results,
        matched_joints_count=0,
        total_joints_count=len(template.expected_angles),
        primary_hint=hint,
        coverage=0.0,
        reliable=False,
    )


def evaluate_pose(
    player_angles: dict[str, float | None],
    player_pos_m: tuple[float, float] | None,
    template: PoseTemplate,
    angle_tolerance_override: float | None = None,
    pos_tolerance_override: float | None = None,
    player_tracked: bool = True,
) -> PoseMatchResult:
    """Compare player joint angles and floor position against a Wall Shape template.

    Returns detailed JointComparison metrics, similarity scores (0..100%), coverage, and guidance hints.
    """
    angle_tol = angle_tolerance_override if angle_tolerance_override is not None else template.angle_tolerance_deg
    pos_tol = pos_tolerance_override if pos_tolerance_override is not None else template.pos_tolerance_m

    joint_results: dict[str, JointComparison] = {}
    joint_scores: list[float] = []
    hints: list[str] = []

    total_expected = len(template.expected_angles)
    visible_count = 0

    for j_name, tgt_angle in template.expected_angles.items():
        act_angle = player_angles.get(j_name)
        if act_angle is None:
            joint_results[j_name] = JointComparison(
                name=j_name,
                player_angle=None,
                target_angle=tgt_angle,
                diff_deg=None,
                is_matched=False,
                hint="not visible",
            )
            continue

        visible_count += 1
        diff = abs(act_angle - tgt_angle)

        # Determine joint tolerance: apply joint-specific custom tolerance
        if angle_tolerance_override is not None:
            base_offset = template.joint_tolerances.get(j_name, template.angle_tolerance_deg) - template.angle_tolerance_deg
            j_tol = max(5.0, angle_tolerance_override + base_offset)
        else:
            j_tol = template.joint_tolerances.get(j_name, angle_tol)

        is_matched = diff <= j_tol

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

        # Linear scoring falloff: 100% within 5 deg, down to 0% at 2 * j_tol
        if diff <= 5.0:
            score = 100.0
        elif diff <= j_tol:
            score = 100.0 - 30.0 * ((diff - 5.0) / max(1.0, j_tol - 5.0))
        else:
            score = max(0.0, 70.0 - 70.0 * ((diff - j_tol) / max(1.0, j_tol)))

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
    total_count = total_expected
    angle_score_pct = float(np.mean(joint_scores)) if joint_scores else 0.0

    # Coverage and reliability
    coverage = round(visible_count / total_count, 3) if total_count > 0 else 0.0
    reliable = (coverage >= 0.7) and bool(player_tracked)

    # Smart mirror-inversion check for asymmetrical poses
    l_sh_act = player_angles.get("left_shoulder")
    r_sh_act = player_angles.get("right_shoulder")
    l_sh_tgt = template.expected_angles.get("left_shoulder")
    r_sh_tgt = template.expected_angles.get("right_shoulder")
    if (
        l_sh_act is not None and r_sh_act is not None and
        l_sh_tgt is not None and r_sh_tgt is not None
    ):
        if r_sh_tgt - l_sh_tgt >= 50.0 and l_sh_act - r_sh_act >= 40.0:
            hints.insert(0, "Swap arms! (Raise RIGHT arm, lower LEFT arm)")
        elif l_sh_tgt - r_sh_tgt >= 50.0 and r_sh_act - l_sh_act >= 40.0:
            hints.insert(0, "Swap arms! (Raise LEFT arm, lower RIGHT arm)")

    l_kn_act = player_angles.get("left_knee")
    r_kn_act = player_angles.get("right_knee")
    l_kn_tgt = template.expected_angles.get("left_knee")
    r_kn_tgt = template.expected_angles.get("right_knee")
    if (
        l_kn_act is not None and r_kn_act is not None and
        l_kn_tgt is not None and r_kn_tgt is not None
    ):
        if l_kn_tgt - r_kn_tgt >= 50.0 and r_kn_act - l_kn_act >= 40.0:
            hints.insert(0, "Swap legs! (Bend RIGHT knee, stand on LEFT leg)")
        elif r_kn_tgt - l_kn_tgt >= 50.0 and l_kn_act - r_kn_act >= 40.0:
            hints.insert(0, "Swap legs! (Bend LEFT knee, stand on RIGHT leg)")

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

    if not reliable:
        is_full_match = False
        # match_pct computed over visible joints only (or 0.0 if none)
        match_pct = round(angle_score_pct, 1) if joint_scores else 0.0
        primary_hint = "Player not tracked" if not player_tracked else "Show your whole body (head to feet)"
    else:
        # Overall Combined Match Percentage
        if template.target_pos_m is not None:
            match_pct = round(0.75 * angle_score_pct + 0.25 * pos_score_pct, 1)
        else:
            match_pct = round(angle_score_pct, 1)

        is_full_match = pos_matched and (
            (matched_count == total_count and match_pct >= 75.0) or
            (matched_count >= max(1, total_count - 1) and match_pct >= 80.0)
        )
        if hints:
            primary_hint = hints[0]
        elif template.cue and not is_full_match:
            primary_hint = template.cue
        else:
            primary_hint = "PERFECT POSE!"

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
        coverage=coverage,
        reliable=reliable,
    )


# ==============================================================================
# Non-Breaking Integration Helpers (for Station B and Test Tools)
# ==============================================================================

def extract_angles_from_skeleton(skeleton: Any) -> dict[str, float | None]:
    """Extract joint angles directly from a Skeleton3D or list of 33 landmarks.

    Scale- and position-invariant; works with metric 3D skeleton points,
    MediaPipe normalized landmarks, or image pixel landmarks.
    """
    if skeleton is None:
        return {}
    if hasattr(skeleton, "points_m"):
        vis = getattr(skeleton, "visibility", None)
        return extract_pose_angles(skeleton.points_m, visibility=vis)
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

    player_tracked = bool(pf and getattr(pf, "tracked", False))
    if not player_tracked:
        result = make_untracked_result(template, hint="Player not tracked")
        return result, pf, frame, debug

    player_pos = None
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
        player_tracked=player_tracked,
    )
    return result, pf, frame, debug

