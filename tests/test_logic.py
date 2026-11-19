"""Spec-by-test for game logic."""
from shared.contracts import PoseFrame
from station_a.logic import (
    GRID_SIZE,
    SUBTILE_M,
    LavaEngine,
    LavaWave,
    WavePattern,
    cells_touched,
    tiles_touched,
)


def test_mid_tile_touches_one_tile():
    assert tiles_touched(0.3, 0.3, 0.25) == {(0, 0)}


def test_corner_touches_four_tiles():
    assert tiles_touched(0.6, 0.6, 0.25) == {(0, 0), (1, 0), (0, 1), (1, 1)}


def test_edge_touches_two_tiles():
    assert tiles_touched(0.6, 0.3, 0.25) == {(0, 0), (1, 0)}


def test_outside_grid_touches_nothing():
    assert tiles_touched(5.0, 5.0, 0.25) == set()


def test_cells_touched_10x10():
    # Center of cell (0, 0) is (0.09, 0.09) m
    touched = cells_touched(0.09, 0.09, radius_m=0.05)
    assert touched == {(0, 0)}


def test_lava_engine_warming_quadratic_ease_in():
    engine = LavaEngine(duration_s=60.0)
    engine.start_game()
    # Force a deterministic wave
    mask = [[True] * GRID_SIZE for _ in range(GRID_SIZE)]
    mask[0][0] = False  # (0, 0) safe
    wave = LavaWave(
        pattern=WavePattern.ISLAND,
        timer=2.0,
        total_warn=2.0,
        target_mask=mask,
        heat=[[0.0] * GRID_SIZE for _ in range(GRID_SIZE)],
    )
    engine.wave = wave

    # Step 10 ticks of 0.1s (total 1.0s) because dt is capped at 0.1s to prevent teleportation
    pose = PoseFrame(tracked=True, floor_x_m=0.09, floor_y_m=0.09, airborne=False)
    for _ in range(10):
        state, active_wave = engine.update(pose, dt=0.1)

    assert active_wave is not None
    assert abs(active_wave.heat[5][5] - 0.25) < 1e-4
    assert active_wave.heat[0][0] == 0.0  # Safe cell remains stone


def test_lava_engine_grounded_hit():
    engine = LavaEngine(duration_s=60.0)
    engine.start_game()
    # Lava on all cells
    mask = [[True] * GRID_SIZE for _ in range(GRID_SIZE)]
    wave = LavaWave(
        pattern=WavePattern.ISLAND,
        timer=0.05,
        total_warn=2.0,
        target_mask=mask,
        heat=[[0.0] * GRID_SIZE for _ in range(GRID_SIZE)],
    )
    engine.wave = wave

    # Player standing grounded on cell (5, 5)
    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9, airborne=False)
    # Advance past 0.0 to trigger eruption judgement
    state, active_wave = engine.update(pose, dt=0.1)

    assert state.hits == 1
    assert state.dodges == 0
    assert state.streak == 0
    assert "BURNT" in engine.last_banner_msg


def test_lava_engine_jump_immunity():
    engine = LavaEngine(duration_s=60.0)
    engine.start_game()
    mask = [[True] * GRID_SIZE for _ in range(GRID_SIZE)]
    wave = LavaWave(
        pattern=WavePattern.ISLAND,
        timer=0.05,
        total_warn=2.0,
        target_mask=mask,
        heat=[[0.0] * GRID_SIZE for _ in range(GRID_SIZE)],
    )
    engine.wave = wave

    # Player jumping airborne over cell (5, 5)
    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9, airborne=True)
    state, active_wave = engine.update(pose, dt=0.1)

    assert state.hits == 0
    assert state.dodges == 1
    assert state.streak == 1
    assert "JUMP IMMUNITY" in engine.last_banner_msg


def test_lava_engine_safe_island_dodge():
    engine = LavaEngine(duration_s=60.0)
    engine.start_game()
    # Everything is lava except central 4x4 block (cells 3..6)
    mask = [[True] * GRID_SIZE for _ in range(GRID_SIZE)]
    for c in range(3, 7):
        for r in range(3, 7):
            mask[c][r] = False

    wave = LavaWave(
        pattern=WavePattern.ISLAND,
        timer=0.05,
        total_warn=2.0,
        target_mask=mask,
        heat=[[0.0] * GRID_SIZE for _ in range(GRID_SIZE)],
    )
    engine.wave = wave

    # Player standing safely in the center (0.9, 0.9)
    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9, airborne=False)
    state, active_wave = engine.update(pose, dt=0.1)

    assert state.hits == 0
    assert state.dodges == 1
    assert state.streak == 1
    assert "SAFE" in engine.last_banner_msg


def test_build_player_occlusion_mask_skeleton_coverage():
    from station_a.render import build_player_occlusion_mask
    cam_w, cam_h = 640, 480
    lms = [(0.0, 0.0, 0.1)] * 33
    lms[11] = (310.0, 120.0, 0.95)
    lms[12] = (360.0, 120.0, 0.95)
    lms[23] = (320.0, 220.0, 0.95)
    lms[24] = (350.0, 220.0, 0.95)
    lms[25] = (310.0, 310.0, 0.95)
    lms[26] = (350.0, 310.0, 0.95)
    lms[27] = (300.0, 380.0, 0.95)
    lms[28] = (360.0, 380.0, 0.95)
    lms[29] = (295.0, 390.0, 0.95)
    lms[30] = (355.0, 390.0, 0.95)
    lms[31] = (305.0, 395.0, 0.95)
    lms[32] = (365.0, 395.0, 0.95)

    debug = {"landmarks_px": lms, "pts": [(300.0, 380.0)], "foot_px": (330.0, 380.0)}
    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9)
    mask = build_player_occlusion_mask(cam_h, cam_w, mapper=None, pose=pose, debug=debug)

    assert mask is not None
    # Key anatomical checkpoints must have full occlusion (255)
    assert mask[390, 300] == 255  # Left shoe
    assert mask[390, 360] == 255  # Right shoe
    assert mask[350, 305] == 255  # Left shin
    assert mask[350, 355] == 255  # Right shin
    assert mask[310, 310] == 255  # Left knee
    assert mask[310, 350] == 255  # Right knee
    assert mask[260, 315] == 255  # Left thigh
    assert mask[260, 345] == 255  # Right thigh
    assert mask[220, 335] == 255  # Pelvis


def test_build_player_occlusion_mask_feet_fallback():
    from station_a.render import build_player_occlusion_mask
    cam_w, cam_h = 640, 480
    debug = {"pts": [(320.0, 400.0)], "foot_px": (320.0, 400.0)}
    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9)
    mask = build_player_occlusion_mask(cam_h, cam_w, mapper=None, pose=pose, debug=debug)

    assert mask is not None
    assert mask[400, 320] == 255  # Contact shoe
    assert mask[250, 320] == 255  # Rising leg column past arena far edge


def test_round_1_initialization_and_patterns():
    engine = LavaEngine(duration_s=60.0)
    engine.start_game("P001")
    assert engine.current_round == 1
    assert engine.round_multiplier == 1.0
    assert engine.round_banner_title == "ROUND 1"
    assert engine.round_banner_timer > 0

    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9, airborne=False)
    state, wave = engine.update(pose, dt=0.01)
    assert wave is not None
    assert wave.pattern in (WavePattern.ISLAND, WavePattern.DONUT)
    assert wave.total_warn == 3.0


def test_round_2_transition_and_patterns():
    engine = LavaEngine(duration_s=60.0)
    engine.start_game("P001")
    # Fast forward to 20.5s elapsed (time_left = 39.5s)
    engine.state.time_left_s = 39.5
    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9, airborne=False)
    engine.update(pose, dt=0.01)

    assert engine.current_round == 2
    assert engine.round_multiplier == 1.5
    assert engine.round_banner_title == "ROUND 2"

    # Spawn new wave in Round 2
    engine.wave = None
    state, wave = engine.update(pose, dt=0.01)
    assert wave is not None
    # Round 2 only allows random geometric shapes (no circles)
    assert wave.pattern in (WavePattern.RIVER, WavePattern.CLUSTER)
    assert wave.total_warn == 2.0


def test_round_3_final_round_transition_and_scoring():
    engine = LavaEngine(duration_s=60.0)
    engine.start_game("P001")
    # Fast forward to 40.5s elapsed (time_left = 19.5s)
    engine.state.time_left_s = 19.5
    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9, airborne=False)
    engine.update(pose, dt=0.01)

    assert engine.current_round == 3
    assert engine.round_multiplier == 2.0
    assert engine.round_banner_title == "FINAL ROUND"

    # Verify score scaling by round multiplier
    mask = [[True] * GRID_SIZE for _ in range(GRID_SIZE)]
    # Make center safe
    for c in range(3, 7):
        for r in range(3, 7):
            mask[c][r] = False

    wave = LavaWave(
        pattern=WavePattern.ISLAND,
        timer=0.05,
        total_warn=1.5,
        target_mask=mask,
        heat=[[0.0] * GRID_SIZE for _ in range(GRID_SIZE)],
    )
    engine.wave = wave

    # Standing safe in center during eruption
    state, active_wave = engine.update(pose, dt=0.1)
    assert state.dodges == 1
    # Base score gain 10 * streak_mult (1.0) * round_mult (2.0) = 20
    assert state.score == 20


def test_round_default_duration_and_stage_splits():
    from shared.contracts import Phase

    # 1. Default initialization
    engine = LavaEngine()
    assert engine.duration_s == 60.0
    assert engine.state.time_left_s == 60.0

    # 2. Start / restart resets to 60.0s
    engine.state.time_left_s = 15.0
    engine.start_game("P001")
    assert engine.state.time_left_s == 60.0
    assert engine.current_round == 1
    assert engine.round_multiplier == 1.0

    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9, airborne=False)

    # 3. Round 1 Warmup: 60s -> 40s (e.g. at 50s left)
    engine.state.time_left_s = 50.0
    state, _ = engine.update(pose, dt=0.01)
    assert engine.current_round == 1
    assert engine.round_multiplier == 1.0
    assert state.phase == Phase.PLAYING

    # 4. Round 2 Chaos Shapes: 40s -> 20s (e.g. at 30s left)
    engine.state.time_left_s = 30.0
    state, _ = engine.update(pose, dt=0.01)
    assert engine.current_round == 2
    assert engine.round_multiplier == 1.5
    assert state.phase == Phase.PLAYING

    # 5. Final Round Sudden Surge: 20s -> 0s (e.g. at 10s left)
    engine.state.time_left_s = 10.0
    state, _ = engine.update(pose, dt=0.01)
    assert engine.current_round == 3
    assert engine.round_multiplier == 2.0
    assert state.phase == Phase.PLAYING

    # 6. Expiry at 0s transitions to Phase.RESULT
    engine.state.time_left_s = 0.05
    state, wave = engine.update(pose, dt=0.1)
    assert state.phase == Phase.RESULT
    assert state.time_left_s == 0.0
    assert wave is None




