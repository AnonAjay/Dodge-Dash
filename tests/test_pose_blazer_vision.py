"""Unit tests for Pose Blazer Vision, Key-Joint Weighting, and Catalog Management."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pose_core.blazer_indexer import detect_key_joints, generate_starter_catalog
from pose_core.blazer_matcher import (
    CORE_JOINTS,
    LIMB_6_JOINTS,
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


def test_traffic_light_scoring_thresholds():
    """Verify that 3 green + 3 yellow passes, all yellow passes, but >=2 reds fail."""
    target = BlazerPoseTemplate(
        id="t_pose",
        name="T-Pose",
        target_angles={
            "left_shoulder": 90.0,
            "right_shoulder": 90.0,
            "left_elbow": 175.0,
            "right_elbow": 175.0,
            "left_knee": 175.0,
            "right_knee": 175.0,
        },
        key_joints=["left_shoulder", "right_shoulder"],
    )

    # 1. 3 Green joints (diff <= 15 deg) + 3 Yellow joints (diff ~30 deg)
    player_3g_3y = {
        "left_shoulder": 95.0,   # diff 5 (green)
        "right_shoulder": 85.0,  # diff 5 (green)
        "left_elbow": 165.0,     # diff 10 (green)
        "right_elbow": 145.0,    # diff 30 (yellow)
        "left_knee": 145.0,      # diff 30 (yellow)
        "right_knee": 145.0,     # diff 30 (yellow)
    }
    res = evaluate_blazer_pose(
        player_landmarks=player_3g_3y,
        target_pose=target,
        active_joints=LIMB_6_JOINTS,
        check_head=False,
    )
    assert res.is_matched is True
    assert res.match_pct >= 65.0
    red_count = sum(1 for s in res.joint_status.values() if s == "off")
    assert red_count == 0

    # 2. All 6 joints in yellow region (~32 deg diff) -> must still pass
    player_all_yellow = {
        "left_shoulder": 122.0,  # diff 32 (yellow)
        "right_shoulder": 58.0,  # diff 32 (yellow)
        "left_elbow": 143.0,     # diff 32 (yellow)
        "right_elbow": 143.0,    # diff 32 (yellow)
        "left_knee": 143.0,      # diff 32 (yellow)
        "right_knee": 143.0,     # diff 32 (yellow)
    }
    res_y = evaluate_blazer_pose(
        player_landmarks=player_all_yellow,
        target_pose=target,
        active_joints=LIMB_6_JOINTS,
        check_head=False,
    )
    assert res_y.is_matched is True
    assert res_y.match_pct >= 65.0

    # 3. Two joints in red region (> 45 deg diff) -> must fail
    player_2_red = {
        "left_shoulder": 140.0,  # diff 50 (red)
        "right_shoulder": 40.0,  # diff 50 (red)
        "left_elbow": 175.0,     # diff 0 (green)
        "right_elbow": 175.0,    # diff 0 (green)
        "left_knee": 175.0,      # diff 0 (green)
        "right_knee": 175.0,     # diff 0 (green)
    }
    res_red = evaluate_blazer_pose(
        player_landmarks=player_2_red,
        target_pose=target,
        active_joints=LIMB_6_JOINTS,
        check_head=False,
    )
    assert res_red.is_matched is False


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
        "t_pose",
        "double_bicep",
        "superman",
        "the_dab",
        "usain_bolt",
    }
    actual_ids = {p.id for p in catalog}
    assert expected_ids.issubset(actual_ids)

    # Lightweight test: verify first 6 sample poses for speed
    for pose in catalog[:6]:
        assert len(pose.target_angles) >= 4
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


def test_limb_6_joints_ignores_bad_hip():
    """Verify that 6-limb mode (shoulders, elbows, knees) ignores obscure hip angles."""
    target = BlazerPoseTemplate(
        id="t_pose",
        name="T-Pose",
        target_angles={
            "left_shoulder": 90.0,
            "right_shoulder": 90.0,
            "left_elbow": 175.0,
            "right_elbow": 175.0,
            "left_knee": 175.0,
            "right_knee": 175.0,
            "left_hip": 175.0,
            "right_hip": 175.0,
        },
        key_joints=["left_shoulder", "right_shoulder"],
    )

    # Player has perfect arms and knees, but completely wrong hips (e.g. 30 deg)
    player_angles = {
        "left_shoulder": 90.0,
        "right_shoulder": 90.0,
        "left_elbow": 175.0,
        "right_elbow": 175.0,
        "left_knee": 175.0,
        "right_knee": 175.0,
        "left_hip": 30.0,
        "right_hip": 30.0,
    }

    # In 6-limb mode: Hips are not in the evaluated set -> Match is near 100%!
    res_6 = evaluate_blazer_pose(
        player_landmarks=player_angles,
        target_pose=target,
        active_joints=LIMB_6_JOINTS,
    )
    assert res_6.match_pct >= 98.0
    assert res_6.is_matched is True
    assert "left_hip" not in res_6.joint_scores

    # In 8-joint mode: Bad hips drag the score down
    res_8 = evaluate_blazer_pose(
        player_landmarks=player_angles,
        target_pose=target,
        active_joints=CORE_JOINTS,
    )
    assert res_8.match_pct < res_6.match_pct


def test_check_head_disabled_allows_off_center():
    """Verify check_head=False allows player to match even when standing off-center."""
    target = BlazerPoseTemplate(
        id="t_pose",
        name="T-Pose",
        target_angles={"left_shoulder": 90.0, "right_shoulder": 90.0},
        key_joints=["left_shoulder", "right_shoulder"],
    )
    perfect_angles = {"left_shoulder": 90.0, "right_shoulder": 90.0}

    # Head is off-center (x=0.10)
    res_disabled = evaluate_blazer_pose(
        player_landmarks=perfect_angles,
        target_pose=target,
        player_head_norm=(0.10, 0.25),
        check_head=False,
    )
    assert res_disabled.is_matched is True
    assert "Step Right" not in res_disabled.primary_hint

    # When check_head=True, head is enforced
    res_enabled = evaluate_blazer_pose(
        player_landmarks=perfect_angles,
        target_pose=target,
        player_head_norm=(0.10, 0.25),
        check_head=True,
    )
    assert res_enabled.is_matched is False


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


def test_is_arms_crossed_detection():
    from station_b.blazer_logic import is_arms_crossed

    # 1. Normal standing pose (not crossed)
    standing = {
        "left_shoulder": 20.0,
        "right_shoulder": 20.0,
        "left_elbow": 170.0,
        "right_elbow": 170.0,
    }
    assert is_arms_crossed(standing) is False

    # 2. Folded arms signature (sharp elbows, low shoulders)
    folded = {
        "left_shoulder": 22.0,
        "right_shoulder": 20.0,
        "left_elbow": 75.0,
        "right_elbow": 75.0,
    }
    assert is_arms_crossed(folded) is True

    # 3. With landmark coordinates: Left wrist crossed past right wrist
    # Synthetic landmarks with 33 points
    lms = [[0.5, 0.5]] * 33
    lms[11] = [0.65, 0.22]  # L_SHOULDER
    lms[12] = [0.35, 0.22]  # R_SHOULDER
    lms[15] = [0.45, 0.35]  # L_WRIST (crossed towards right side)
    lms[16] = [0.55, 0.35]  # R_WRIST (crossed towards left side)
    assert is_arms_crossed(folded, lms) is True

    # Hands at hips (not crossed): lw.x > rw.x
    lms_hips = list(lms)
    lms_hips[15] = [0.70, 0.50]  # uncrossed
    lms_hips[16] = [0.30, 0.50]
    assert is_arms_crossed(folded, lms_hips) is False


def test_blazer_skip_and_consecutive_elimination():
    from shared.contracts import Phase, PoseFrame
    from station_b.blazer_logic import BlazerGameEngine

    catalog = load_blazer_catalog("data/poses_index.json")
    engine = BlazerGameEngine(catalog=catalog, total_rounds=6, tournament_mode=True)
    engine.start_game("P001")

    pose_frame = PoseFrame(tracked=True)
    folded_angles = {
        "left_shoulder": 20.0,
        "right_shoulder": 20.0,
        "left_elbow": 75.0,
        "right_elbow": 75.0,
    }

    # 1. Holding folded arms for 0.5s does not skip prematurely
    for _ in range(5):
        engine.update(pose_frame, dt=0.1, player_angles=folded_angles)
    assert engine.round_number == 1
    assert engine.consecutive_skips == 0
    assert "HOLDING TO SKIP" in engine.banner_msg

    # 2. Advance another 0.6s (total > 1.0s) -> triggers skip!
    for _ in range(6):
        engine.update(pose_frame, dt=0.1, player_angles=folded_angles)
    assert engine.consecutive_skips == 1
    assert "SKIPPED" in engine.banner_msg

    # Advance 1.1s cooldown -> moves to round 2
    for _ in range(11):
        engine.update(pose_frame, dt=0.1)
    assert engine.round_number == 2

    # 3. Skip 3 more times consecutively -> 4 consecutive skips must ELIMINATE player!
    for expected_skip in [2, 3, 4]:
        for _ in range(11):  # 1.1s folded arms
            engine.update(pose_frame, dt=0.1, player_angles=folded_angles)
        assert engine.consecutive_skips == expected_skip
        if expected_skip < 4:
            for _ in range(11):  # cooldown
                engine.update(pose_frame, dt=0.1)

    # 4. Instant disqualification on 4th consecutive skip
    assert engine.state.phase == Phase.RESULT
    assert "ELIMINATED" in engine.banner_msg


def test_blazer_2_stage_tournament_progression():
    from shared.contracts import Phase, PoseFrame
    from station_b.blazer_logic import BlazerGameEngine

    catalog = load_blazer_catalog("data/poses_index.json")
    engine = BlazerGameEngine(catalog=catalog, total_rounds=6, tournament_mode=True)
    engine.start_game("P001")

    # Round 1 is Stage 1: Precision Blitz (10.0s)
    assert engine.round_number == 1
    assert engine.current_stage == 1
    assert engine.current_stage_name == "PRECISION BLITZ"
    assert engine.time_left_s == 10.0

    pose_frame = PoseFrame(tracked=True)
    # Perfect match in Stage 1 -> awards +150 perfection bonus
    match_angles = dict(engine.current_pose.target_angles)
    for _ in range(5):
        engine.update(pose_frame, dt=0.1, player_angles=match_angles)
    assert engine.state.score >= 250  # Base 100 + 150 perfection

    # Fast forward through rounds 1..3 to reach Stage 2 (round 4)
    for r in range(1, 4):
        engine.current_round_idx = r
        engine._reset_round_timers()

    assert engine.round_number == 4
    assert engine.current_stage == 2
    assert engine.current_stage_name == "SPEED RUSH"
    assert engine.time_left_s == 5.0  # Stage 2 uses 5.0s rush timer!
