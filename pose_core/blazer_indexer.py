"""Pose Blazer: Image Ingestion, Landmark Extraction, and Catalog Indexer.

Uses MediaPipe Tasks API PoseLandmarker in static IMAGE mode to extract 33 landmarks,
compute joint angles, detect key distinguishing limbs, compress reference cards,
and maintain the data/poses_index.json catalog.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from pose_core.blazer_matcher import (
    CORE_JOINTS,
    BlazerPoseTemplate,
    load_blazer_catalog,
    save_blazer_catalog,
)
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
    extract_pose_angles,
)
from pose_core.skeleton import BONES

# Neutral standing reference angles
NEUTRAL_STANDING_ANGLES: dict[str, float] = {
    "left_shoulder": 15.0,
    "right_shoulder": 15.0,
    "left_elbow": 170.0,
    "right_elbow": 170.0,
    "left_hip": 175.0,
    "right_hip": 175.0,
    "left_knee": 175.0,
    "right_knee": 175.0,
}


class BlazerIndexer:
    """Extracts landmarks and angles from static reference images to build the game catalog."""

    def __init__(self, model_path: str | Path = "models/pose_landmarker_lite.task") -> None:
        self.model_path = Path(model_path)
        self._landmarker = None

    def _get_landmarker(self):
        """Lazy load MediaPipe PoseLandmarker in static image mode."""
        if self._landmarker is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"PoseLandmarker model not found at {self.model_path.resolve()}."
                )

            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.25,
                min_tracking_confidence=0.25,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        return self._landmarker

    def extract_from_image(
        self,
        img_bgr: np.ndarray,
    ) -> tuple[dict[str, float], list[list[float]], list[str]]:
        """Extract angles, 33 normalized landmarks, and key joints from an image.

        Includes multi-pass border padding fallback for stylized or tight crop images.

        Returns:
            (target_angles, landmarks_norm, key_joints)
        """
        import mediapipe as mp

        landmarker = self._get_landmarker()
        h, w = img_bgr.shape[:2]

        # Pass 1: Direct detection
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_img)

        # Pass 2: Fallback with padding (for tight crops / anime drawings)
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            pad = 60
            padded = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[240, 240, 240])
            padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            mp_padded = mp.Image(image_format=mp.ImageFormat.SRGB, data=padded_rgb)
            result = landmarker.detect(mp_padded)
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                ph, pw = padded.shape[:2]
                raw_lms = result.pose_landmarks[0]
                # Unpad normalized landmarks back to original image space
                unpadded_lms = []
                for lm in raw_lms:
                    orig_x = (lm.x * pw - pad) / w
                    orig_y = (lm.y * ph - pad) / h
                    unpadded_lms.append(
                        type("Landmark", (), {
                            "x": max(0.0, min(1.0, orig_x)),
                            "y": max(0.0, min(1.0, orig_y)),
                            "z": getattr(lm, "z", 0.0),
                            "visibility": getattr(lm, "visibility", 1.0),
                        })()
                    )
                lms = unpadded_lms
            else:
                raise ValueError("No person detected in reference image. Please use a clear, full-body photo.")
        else:
            lms = result.pose_landmarks[0]

        # 33 landmarks in normalized [0, 1] space
        landmarks_norm = [[round(float(lm.x), 4), round(float(lm.y), 4)] for lm in lms]

        # Extract 2D angles in mirrored screen-space so screen-left matches screen-left
        angles_raw = extract_pose_angles(landmarks_norm, mirrored=True)
        target_angles = {k: float(v) for k, v in angles_raw.items() if v is not None}

        # Detect Key Joints (limbs that define this pose vs neutral standing)
        key_joints = detect_key_joints(target_angles)

        return target_angles, landmarks_norm, key_joints

    def index_image_file(
        self,
        img_path: str | Path,
        pose_id: str | None = None,
        pose_name: str | None = None,
        difficulty: str = "medium",
        description: str = "",
        output_dir: str | Path = "data/poses",
    ) -> BlazerPoseTemplate:
        """Process an image file, compress to a game card, and return a BlazerPoseTemplate."""
        src_p = Path(img_path)
        if not src_p.is_file():
            raise FileNotFoundError(f"Image not found at {src_p}")

        img = cv2.imread(str(src_p))
        if img is None:
            raise ValueError(f"Could not decode image at {src_p}")

        if pose_id is None:
            pose_id = src_p.stem.lower().replace(" ", "_").replace("-", "_")
        if pose_name is None:
            pose_name = src_p.stem.replace("_", " ").replace("-", " ").title()

        # Extract angles & landmarks
        target_angles, landmarks_norm, key_joints = self.extract_from_image(img)

        # Save compressed 400x500 JPEG/WebP card
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        card_filename = f"{pose_id}.jpg"
        card_path = out_p / card_filename

        card_img = cv2.resize(img, (400, 500), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(card_path), card_img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        return BlazerPoseTemplate(
            id=pose_id,
            name=pose_name,
            image_file=card_filename,
            target_angles=target_angles,
            key_joints=key_joints,
            landmarks_norm=landmarks_norm,
            difficulty=difficulty,
            description=description,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


def detect_key_joints(angles: dict[str, float]) -> list[str]:
    """Detect limbs that deviate significantly from a neutral standing pose.

    A limb deviating by > 25 degrees is flagged as a key joint.
    """
    key_joints: list[str] = []
    for joint in CORE_JOINTS:
        if joint not in angles or angles[joint] is None:
            continue
        actual = angles[joint]
        ref = NEUTRAL_STANDING_ANGLES.get(joint, 175.0)
        if abs(actual - ref) >= 25.0:
            key_joints.append(joint)

    # Fallback: if all joints are neutral, mark shoulders as key
    if not key_joints:
        key_joints = ["left_shoulder", "right_shoulder"]
    return key_joints


# ==============================================================================
# Starter Poses Synthesizer
# ==============================================================================

def generate_starter_catalog(
    catalog_path: str | Path = "data/poses_index.json",
    poses_dir: str | Path = "data/poses",
) -> list[BlazerPoseTemplate]:
    """Generate 6 high-quality starter poses with clean synthetic vector cards.

    Allows immediate gameplay testing out-of-the-box before custom photos are added.
    """
    starter_templates: list[BlazerPoseTemplate] = [
        BlazerPoseTemplate(
            id="t_pose",
            name="T-Pose",
            image_file="t_pose.jpg",
            target_angles={
                "left_shoulder": 90.0,
                "right_shoulder": 90.0,
                "left_elbow": 175.0,
                "right_elbow": 175.0,
                "left_hip": 175.0,
                "right_hip": 175.0,
                "left_knee": 175.0,
                "right_knee": 175.0,
            },
            key_joints=["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"],
            landmarks_norm=_synth_landmarks(sh_l=90, sh_r=90, el_l=175, el_r=175, kn_l=175, kn_r=175),
            difficulty="easy",
            description="Stand tall with arms stretched horizontally at shoulder height.",
        ),
        BlazerPoseTemplate(
            id="victory_y",
            name="Victory Y",
            image_file="victory_y.jpg",
            target_angles={
                "left_shoulder": 145.0,
                "right_shoulder": 145.0,
                "left_elbow": 170.0,
                "right_elbow": 170.0,
                "left_hip": 175.0,
                "right_hip": 175.0,
                "left_knee": 175.0,
                "right_knee": 175.0,
            },
            key_joints=["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"],
            landmarks_norm=_synth_landmarks(sh_l=145, sh_r=145, el_l=170, el_r=170, kn_l=175, kn_r=175),
            difficulty="easy",
            description="Raise both arms overhead forming a bold victory 'Y' shape.",
        ),
        BlazerPoseTemplate(
            id="disco_fever",
            name="Disco Fever",
            image_file="disco_fever.jpg",
            target_angles={
                "left_shoulder": 140.0,
                "right_shoulder": 25.0,
                "left_elbow": 170.0,
                "right_elbow": 165.0,
                "left_hip": 175.0,
                "right_hip": 165.0,
                "left_knee": 175.0,
                "right_knee": 150.0,
            },
            key_joints=["left_shoulder", "right_shoulder", "right_knee"],
            landmarks_norm=_synth_landmarks(sh_l=140, sh_r=25, el_l=170, el_r=165, kn_l=175, kn_r=150),
            difficulty="medium",
            description="Point your left arm high to the sky with right arm down to hip.",
        ),
        BlazerPoseTemplate(
            id="double_biceps",
            name="Double Biceps",
            image_file="double_biceps.jpg",
            target_angles={
                "left_shoulder": 90.0,
                "right_shoulder": 90.0,
                "left_elbow": 75.0,
                "right_elbow": 75.0,
                "left_hip": 175.0,
                "right_hip": 175.0,
                "left_knee": 175.0,
                "right_knee": 175.0,
            },
            key_joints=["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"],
            landmarks_norm=_synth_landmarks(sh_l=90, sh_r=90, el_l=75, el_r=75, kn_l=175, kn_r=175),
            difficulty="medium",
            description="Flex both biceps with elbows bent at 90 degrees.",
        ),
        BlazerPoseTemplate(
            id="warrior_pose",
            name="Warrior Pose",
            image_file="warrior_pose.jpg",
            target_angles={
                "left_shoulder": 90.0,
                "right_shoulder": 90.0,
                "left_elbow": 175.0,
                "right_elbow": 175.0,
                "left_hip": 115.0,
                "right_hip": 155.0,
                "left_knee": 105.0,
                "right_knee": 175.0,
            },
            key_joints=["left_knee", "left_hip", "left_shoulder", "right_shoulder"],
            landmarks_norm=_synth_landmarks(sh_l=90, sh_r=90, el_l=175, el_r=175, kn_l=105, kn_r=175),
            difficulty="hard",
            description="Lunge deep on your left leg with arms horizontal.",
        ),
        BlazerPoseTemplate(
            id="tree_pose",
            name="Tree Pose",
            image_file="tree_pose.jpg",
            target_angles={
                "left_shoulder": 155.0,
                "right_shoulder": 155.0,
                "left_elbow": 160.0,
                "right_elbow": 160.0,
                "left_hip": 175.0,
                "right_hip": 120.0,
                "left_knee": 175.0,
                "right_knee": 65.0,
            },
            key_joints=["right_knee", "right_hip", "left_shoulder", "right_shoulder"],
            landmarks_norm=_synth_landmarks(sh_l=155, sh_r=155, el_l=160, el_r=160, kn_l=175, kn_r=65),
            difficulty="hard",
            description="Balance on your left leg, bend right knee outward, hands overhead.",
        ),
    ]

    # Generate starter card graphics in data/poses/
    poses_p = Path(poses_dir)
    poses_p.mkdir(parents=True, exist_ok=True)

    for tmpl in starter_templates:
        if tmpl.image_file:
            card_p = poses_p / tmpl.image_file
            if not card_p.is_file():
                _generate_synthetic_card(card_p, tmpl)

    # Save to catalog
    save_blazer_catalog(starter_templates, catalog_path)
    return starter_templates


def _generate_synthetic_card(card_path: Path, tmpl: BlazerPoseTemplate) -> None:
    """Generate a high-tech synthetic visual card for a starter pose."""
    w, h = 400, 500
    card = np.zeros((h, w, 3), dtype=np.uint8)
    card[:] = (20, 24, 32)

    # Cyber grid lines
    for x in range(0, w, 25):
        cv2.line(card, (x, 0), (x, h), (30, 36, 48), 1)
    for y in range(0, h, 25):
        cv2.line(card, (0, y), (w, y), (30, 36, 48), 1)

    # Draw wireframe body
    pts_px: list[tuple[int, int]] = []
    dw, dh = w * 0.72, h * 0.72
    ox, oy = (w - dw) / 2.0, (h - dh) / 2.0

    for lm in tmpl.landmarks_norm:
        px = int(ox + lm[0] * dw)
        py = int(oy + lm[1] * dh)
        pts_px.append((px, py))

    # Glow effect bones
    for j1, j2 in BONES:
        if j1 < len(pts_px) and j2 < len(pts_px):
            cv2.line(card, pts_px[j1], pts_px[j2], (0, 180, 255), 4, cv2.LINE_AA)
            cv2.line(card, pts_px[j1], pts_px[j2], (255, 255, 255), 2, cv2.LINE_AA)

    for px, py in pts_px:
        cv2.circle(card, (px, py), 5, (0, 240, 100), -1, cv2.LINE_AA)

    # Card border
    cv2.rectangle(card, (2, 2), (w - 3, h - 3), (0, 220, 255), 2)
    cv2.imwrite(str(card_path), card, [cv2.IMWRITE_JPEG_QUALITY, 90])


def _synth_landmarks(
    sh_l: float,
    sh_r: float,
    el_l: float,
    el_r: float,
    kn_l: float,
    kn_r: float,
) -> list[list[float]]:
    """Synthesize normalized 33 MediaPipe landmarks for standard body angles."""
    lms = [[0.50, 0.50] for _ in range(33)]

    # Head (0: Nose)
    lms[0] = [0.50, 0.16]

    # Shoulders (11: Left, 12: Right in image space)
    lms[11] = [0.42, 0.28]
    lms[12] = [0.58, 0.28]

    # Hips (23: Left, 24: Right)
    lms[23] = [0.45, 0.55]
    lms[24] = [0.55, 0.55]

    # Arms - Left
    # sh_l = 15 -> down, 90 -> horizontal, 145 -> up
    rad_sh_l = math.radians(sh_l)
    el_lx = 0.42 - 0.15 * math.sin(rad_sh_l)
    el_ly = 0.28 + 0.15 * math.cos(rad_sh_l) if sh_l < 90 else 0.28 - 0.15 * math.cos(math.pi - rad_sh_l)
    lms[13] = [round(el_lx, 4), round(el_ly, 4)]

    rad_el_l = math.radians(el_l)
    wr_lx = el_lx - 0.14 * math.sin(rad_el_l)
    wr_ly = el_ly - 0.14 * math.cos(rad_el_l)
    lms[15] = [round(wr_lx, 4), round(wr_ly, 4)]

    # Arms - Right
    rad_sh_r = math.radians(sh_r)
    el_rx = 0.58 + 0.15 * math.sin(rad_sh_r)
    el_ry = 0.28 + 0.15 * math.cos(rad_sh_r) if sh_r < 90 else 0.28 - 0.15 * math.cos(math.pi - rad_sh_r)
    lms[14] = [round(el_rx, 4), round(el_ry, 4)]

    rad_el_r = math.radians(el_r)
    wr_rx = el_rx + 0.14 * math.sin(rad_el_r)
    wr_ry = el_ry - 0.14 * math.cos(rad_el_r)
    lms[16] = [round(wr_rx, 4), round(wr_ry, 4)]

    # Legs - Left
    lms[25] = [0.44, 0.74]
    lms[27] = [0.44, 0.94]

    # Legs - Right
    if kn_r < 90:
        lms[26] = [0.65, 0.68]  # Bent outward
        lms[28] = [0.55, 0.72]
    else:
        lms[26] = [0.56, 0.74]
        lms[28] = [0.56, 0.94]

    return lms
