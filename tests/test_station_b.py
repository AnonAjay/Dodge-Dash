"""Unit tests for Station B (Pose Blazer Tournament) game logic."""
import time
from shared.contracts import Phase, PoseFrame, ScoreEvent
from station_b.logic import (
    BlazerGameEngine,
    BlazerRoundRecord,
    is_arms_crossed,
    STAGE_1_DURATION_S,
    STAGE_2_DURATION_S,
    MAX_CONSECUTIVE_SKIPS,
)
from pose_core.blazer_matcher import load_blazer_catalog


def test_station_b_tournament_initialization():
    catalog = load_blazer_catalog("data/poses_index.json")
    engine = BlazerGameEngine(catalog=catalog, total_rounds=10, tournament_mode=True)
    engine.start_game("P001")

    assert engine.state.phase == Phase.PLAYING
    assert engine.state.score == 0
    assert engine.state.dodges == 0
    assert engine.state.hits == 0
    assert engine.current_round == 1
    assert engine.current_stage == 1
    assert engine.current_stage_name == "PRECISION BLITZ"
    assert engine.round_duration_s == STAGE_1_DURATION_S


def test_station_b_precision_blitz_perfection_bonus():
    catalog = load_blazer_catalog("data/poses_index.json")
    engine = BlazerGameEngine(catalog=catalog, total_rounds=10, tournament_mode=True)
    engine.start_game("P001")
    engine.tournament_phase = "active"

    pose = PoseFrame(tracked=True)
    matching_angles = dict(engine.current_pose.target_angles)

    # Hold matching pose in Stage 1 for 0.4s (>= 0.35s required)
    for _ in range(4):
        engine.update(pose, dt=0.1, player_angles=matching_angles)

    # 100 base + 150 perfection (>= 85%) + speed bonus
    assert engine.is_pose_locked is True
    assert engine.state.dodges == 1
    assert engine.state.score >= 250
    assert "PERFECTION" in engine.banner_msg


def test_station_b_speed_rush_progression():
    catalog = load_blazer_catalog("data/poses_index.json")
    engine = BlazerGameEngine(catalog=catalog, total_rounds=10, tournament_mode=True)
    engine.start_game("P001")

    # Transition to Round 2 (Speed Rush)
    engine.current_round = 2
    engine.tournament_phase = "active"
    engine._reset_round_timers()

    assert engine.current_stage == 2
    assert engine.current_stage_name == "SPEED RUSH"
    assert engine.time_left_s == STAGE_2_DURATION_S

    # Rapid snap (0.2s hold in Speed Rush)
    pose = PoseFrame(tracked=True)
    matching_angles = dict(engine.current_pose.target_angles)
    for _ in range(3):
        engine.update(pose, dt=0.1, player_angles=matching_angles)

    assert engine.is_pose_locked is True
    assert engine.state.score >= 120


def test_station_b_folded_arms_skip():
    catalog = load_blazer_catalog("data/poses_index.json")
    engine = BlazerGameEngine(catalog=catalog, total_rounds=10, tournament_mode=True)
    engine.start_game("P001")
    engine.tournament_phase = "active"

    pose = PoseFrame(tracked=True)
    folded_angles = {
        "left_shoulder": 20.0,
        "right_shoulder": 20.0,
        "left_elbow": 75.0,
        "right_elbow": 75.0,
    }

    # Hold folded arms for 1.1s (exceeding 1.0s requirement)
    for _ in range(11):
        engine.update(pose, dt=0.1, player_angles=folded_angles)

    assert engine.consecutive_skips == 1
    assert engine.state.hits == 1
    assert "SKIPPED" in engine.banner_msg


def test_station_b_elimination_after_4_skips():
    catalog = load_blazer_catalog("data/poses_index.json")
    engine = BlazerGameEngine(catalog=catalog, total_rounds=10, tournament_mode=True)
    engine.start_game("P001")
    engine.tournament_phase = "active"


    pose = PoseFrame(tracked=True)
    folded_angles = {
        "left_shoulder": 20.0,
        "right_shoulder": 20.0,
        "left_elbow": 75.0,
        "right_elbow": 75.0,
    }

    for skip_idx in range(1, 5):
        for _ in range(11):  # 1.1s folded arms
            engine.update(pose, dt=0.1, player_angles=folded_angles)
        assert engine.consecutive_skips == skip_idx
        if skip_idx < 4:
            for _ in range(11):  # cooldown between poses
                engine.update(pose, dt=0.1)

    # 4 consecutive skips triggers instant disqualification
    assert engine.state.phase == Phase.RESULT
    assert "ELIMINATED" in engine.banner_msg


def test_station_b_score_event_contract():
    catalog = load_blazer_catalog("data/poses_index.json")
    engine = BlazerGameEngine(catalog=catalog, total_rounds=6, tournament_mode=True)
    engine.start_game("P001")

    # Complete tournament
    engine.state.score = 850
    engine.state.dodges = 5
    engine.state.hits = 1
    engine._finish_game()

    score_event = ScoreEvent(
        event_id=f"B-BLAZER-P001-{int(time.time())}",
        player_id="P001",
        station="B",
        game="blazer",
        score=engine.state.score,
        accuracy=0.83,
        best_streak=engine.state.streak,
        duration_s=engine.round_duration_s * engine.total_rounds,
        finished_at="2026-11-20T00:00:00Z",
    )
    assert score_event.station == "B"
    assert score_event.game == "blazer"
    assert score_event.score == 850


def test_station_b_arena_renderer_headless():
    """Verify BlazerArenaRenderer renders HUD, AR canvas, and Result overlays safely."""
    import os
    import numpy as np
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    from station_b.render import BlazerArenaRenderer, render_pixel_text
    renderer = BlazerArenaRenderer(width=800, height=600, fullscreen=False)

    catalog = load_blazer_catalog("data/poses_index.json")
    engine = BlazerGameEngine(catalog=catalog, total_rounds=6, tournament_mode=True)
    engine.start_game("P001")

    # Render PLAYING frame with dummy camera
    dummy_cam = np.zeros((480, 640, 3), dtype=np.uint8)
    renderer.draw(
        state=engine.state,
        engine=engine,
        cam_frame=dummy_cam,
        fps=60.0,
        dt=0.016,
    )

    # Render RESULT frame
    engine._finish_game()
    renderer.draw(
        state=engine.state,
        engine=engine,
        cam_frame=None,
        fps=60.0,
        dt=0.016,
    )


def test_station_b_body_visibility_gate():
    """Verify game pauses and displays 'STEP INTO FRAME' when player is partially out of bounds."""
    from station_b.logic import check_body_in_frame
    catalog = load_blazer_catalog("data/poses_index.json")
    engine = BlazerGameEngine(catalog=catalog, total_rounds=6, tournament_mode=True)
    engine.start_game("P001")

    # Incomplete / occluded landmarks (only 10 points)
    incomplete_lms = [[0.5, 0.5, 0.9]] * 10
    assert check_body_in_frame(incomplete_lms, tracked=True) is False

    # Low visibility on knees
    valid_lms = [[0.5, 0.5, 0.9]] * 33
    valid_lms[25] = [0.5, 0.5, 0.1]  # Left knee occluded/out of frame
    assert check_body_in_frame(valid_lms, tracked=True) is False

    # Fully visible landmarks
    all_good_lms = [[0.5, 0.5, 0.95]] * 33
    assert check_body_in_frame(all_good_lms, tracked=True) is True

    # When update receives occluded body, it pauses countdown and sets banner
    initial_time = engine.time_left_s
    pose = PoseFrame(tracked=True)
    engine.update(pose, dt=0.5, player_landmarks_norm=valid_lms)

    assert "STEP INTO FRAME" in engine.banner_msg
    assert engine.time_left_s == initial_time  # Clock was paused!


