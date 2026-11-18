"""Unit tests for Pose Blazer Vision, Key-Joint Weighting, and Catalog Management."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pose_core.blazer_indexer import detect_key_joints, generate_starter_catalog
from pose_core.blazer_matcher import (
    BlazerPoseTemplate,
    evaluate_blazer_pose,
    load_blazer_catalog,
    render_reference_card,
    sample_game_poses,
    save_blazer_catalog,
)


def test_starter_catalog_generation_and_loading():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cat_file = Path(tmp_dir) / "test_poses.json"
        poses_dir = Path(tmp_dir) / "poses"

        # Generate starter catalog
        templates = generate_starter_catalog(catalog_path=cat_file, poses_dir=poses_dir)
        assert len(templates) >= 6

        # Reload from JSON
        reloaded = load_blazer_catalog(cat_file)
        assert len(reloaded) == len(templates)
        assert reloaded[0].id == "t_pose"
        assert "left_shoulder" in reloaded[0].target_angles
        assert len(reloaded[0].key_joints) > 0


def test_sample_game_poses():
    catalog = [
        BlazerPoseTemplate(id=f"pose_{i}", name=f"Pose {i}")
        for i in range(8)
    ]
    sampled = sample_game_poses(catalog, n=5)
    assert len(sampled) == 5
    # Unique check
    assert len({p.id for p in sampled}) == 5


def test_perfect_pose_match():
    target = BlazerPoseTemplate(
        id="t_pose",
        name="T-Pose",
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
    )

    # Identical live angles
    live_angles = dict(target.target_angles)
    res = evaluate_blazer_pose(
        player_landmarks=live_angles,
        target_pose=target,
        player_head_norm=(0.50, 0.25),
        pass_threshold_pct=75.0,
    )

    assert res.match_pct >= 98.0
    assert res.is_matched is True
    assert res.head_aligned is True
    assert res.joint_status["left_shoulder"] == "matched"


def test_key_joint_weighting():
    """Verify that failing key joints causes a failure even if non-key joints match."""
    target = BlazerPoseTemplate(
        id="t_pose",
        name="T-Pose",
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
    )

    # Case A: Wrong Arms (arms hanging at 20 deg), but legs match (175 deg)
    wrong_arms = {
        "left_shoulder": 20.0,
        "right_shoulder": 20.0,
        "left_elbow": 175.0,
        "right_elbow": 175.0,
        "left_hip": 175.0,
        "right_hip": 175.0,
        "left_knee": 175.0,
        "right_knee": 175.0,
    }
    res_a = evaluate_blazer_pose(
        player_landmarks=wrong_arms,
        target_pose=target,
        player_head_norm=(0.50, 0.25),
    )
    # Should fail because shoulders carry 2.5x weight
    assert res_a.match_pct < 75.0
    assert res_a.is_matched is False
    assert "Arm" in res_a.primary_hint or "Raise" in res_a.primary_hint

    # Case B: Correct Arms, but minor leg difference (knees slightly bent at 160 vs 175)
    good_arms_slight_legs = {
        "left_shoulder": 90.0,
        "right_shoulder": 90.0,
        "left_elbow": 175.0,
        "right_elbow": 175.0,
        "left_hip": 175.0,
        "right_hip": 175.0,
        "left_knee": 160.0,
        "right_knee": 160.0,
    }
    res_b = evaluate_blazer_pose(
        player_landmarks=good_arms_slight_legs,
        target_pose=target,
        player_head_norm=(0.50, 0.25),
    )
    # Should pass because key limbs (arms) are spot on
    assert res_b.match_pct >= 85.0
    assert res_b.is_matched is True


def test_head_alignment_boundary():
    target = BlazerPoseTemplate(
        id="t_pose",
        name="T-Pose",
        target_angles={"left_shoulder": 90.0, "right_shoulder": 90.0},
        key_joints=["left_shoulder", "right_shoulder"],
    )
    perfect_angles = {"left_shoulder": 90.0, "right_shoulder": 90.0}

    # Case A: Centered head (x=0.50, y=0.25) -> Matched
    res_aligned = evaluate_blazer_pose(
        player_landmarks=perfect_angles,
        target_pose=target,
        player_head_norm=(0.50, 0.25),
    )
    assert res_aligned.head_aligned is True
    assert res_aligned.is_matched is True

    # Case B: Off-center head (x=0.15, y=0.25) -> Head not aligned, fail despite matching angles
    res_off = evaluate_blazer_pose(
        player_landmarks=perfect_angles,
        target_pose=target,
        player_head_norm=(0.15, 0.25),
    )
    assert res_off.head_aligned is False
    assert res_off.is_matched is False
    assert "Step Right" in res_off.primary_hint


def test_key_joint_detector():
    # Disco fever angles: left shoulder is high (140 vs 15), right knee is bent (150 vs 175)
    disco_angles = {
        "left_shoulder": 140.0,
        "right_shoulder": 18.0,
        "left_elbow": 170.0,
        "right_elbow": 165.0,
        "left_hip": 175.0,
        "right_hip": 165.0,
        "left_knee": 175.0,
        "right_knee": 140.0,
    }
    keys = detect_key_joints(disco_angles)
    assert "left_shoulder" in keys
    assert "right_knee" in keys
    assert "left_elbow" not in keys


def test_reference_card_renderer():
    tmpl = BlazerPoseTemplate(
        id="test_pose",
        name="Test Pose",
        image_file="non_existent_image.jpg",
        key_joints=["left_shoulder", "right_shoulder"],
        landmarks_norm=[[0.5, 0.2] for _ in range(33)],
    )

    # Render with missing image -> should gracefully fall back to vector mannequin
    card = render_reference_card(tmpl, width=320, height=440)
    assert isinstance(card, np.ndarray)
    assert card.shape == (440, 320, 3)


def test_user_poses_catalog_and_images():
    """Verify user's added poses load from data/poses_index.json and render cards."""
    catalog = load_blazer_catalog("data/poses_index.json")
    assert len(catalog) >= 5

    expected_ids = {
        "ash_ketchum",
        "gojo_funnypose",
        "mario",
        "naruto_sittingpose",
        "spiderman_pose",
    }
    actual_ids = {p.id for p in catalog}
    assert expected_ids.issubset(actual_ids)

    for pose in catalog:
        assert len(pose.target_angles) >= 6
        assert len(pose.key_joints) >= 1
        assert len(pose.landmarks_norm) == 33

        # Test evaluating pose against its own target angles (should yield 100%)
        res = evaluate_blazer_pose(
            player_landmarks=pose.target_angles,
            target_pose=pose,
            player_head_norm=(0.50, 0.25),
        )
        assert res.match_pct >= 95.0
        assert res.is_matched is True

        # Test card rendering with actual image files
        card = render_reference_card(pose, width=340, height=460, poses_dir="data/poses")
        assert card.shape == (460, 340, 3)


def test_blazer_engine_auto_advance_on_7s_timeout():
    from shared.contracts import Phase, PoseFrame
    from station_b.blazer_logic import BlazerGameEngine

    catalog = load_blazer_catalog("data/poses_index.json")
    engine = BlazerGameEngine(catalog=catalog, round_duration_s=7.0, total_rounds=5)
    engine.start_game("P001")

    assert engine.state.phase == Phase.PLAYING
    assert engine.round_number == 1
    pose1_id = engine.current_pose.id

    pose_frame = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9)
    wrong_angles = {"left_shoulder": 0.0, "right_shoulder": 0.0}

    # Advance 7.1 seconds (exceeding 7.0s timeout limit)
    for _ in range(71):
        engine.update(pose_frame, dt=0.1, player_angles=wrong_angles)

    # Must register miss
    assert engine.state.hits == 1
    assert engine.state.score == 0
    assert "TIME OUT" in engine.banner_msg

    # Advance 1.1s through the cooldown pause -> must auto-advance to round 2!
    for _ in range(11):
        engine.update(pose_frame, dt=0.1)

    assert engine.round_number == 2
    assert engine.time_left_s == 7.0


def test_blazer_engine_hold_and_speed_bonus():
    from shared.contracts import Phase, PoseFrame
    from station_b.blazer_logic import BlazerGameEngine

    catalog = load_blazer_catalog("data/poses_index.json")
    engine = BlazerGameEngine(catalog=catalog, round_duration_s=7.0, total_rounds=5)
    engine.start_game("P001")

    pose_frame = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9)
    current_target = engine.current_pose
    matching_angles = dict(current_target.target_angles)

    # 1. Player does not match pose at t = 0.5s
    for _ in range(5):
        engine.update(pose_frame, dt=0.1, player_angles={})

    # 2. Player holds matching pose for 0.4s (exceeding 0.3s hold threshold)
    for _ in range(4):
        engine.update(
            pose_frame,
            dt=0.1,
            player_angles=matching_angles,
            player_head_norm=(0.50, 0.25),
        )

    # Should lock in with speed bonus (> 100 pts)
    assert engine.is_pose_locked is True
    assert engine.state.dodges == 1
    assert engine.state.score >= 200  # Base 100 + ~120 speed bonus
    assert "SNAP" in engine.banner_msg

    # Advance 1.1s cooldown -> auto-advances to round 2
    for _ in range(11):
        engine.update(pose_frame, dt=0.1)

    assert engine.round_number == 2
