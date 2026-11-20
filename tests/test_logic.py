"""Spec-by-test for game logic."""
from shared.contracts import PoseFrame
from station_a.logic import (
    GRID_SIZE,
    SUBTILE_M,
    LavaEngine,
    LavaWave,
    TournamentPhase,
    WavePattern,
    cells_touched,
    get_streak_multiplier,
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
    engine.tournament_phase = TournamentPhase.ROUND_ACTIVE
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
    engine.tournament_phase = TournamentPhase.ROUND_ACTIVE
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
    engine.tournament_phase = TournamentPhase.ROUND_ACTIVE
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
    engine.tournament_phase = TournamentPhase.ROUND_ACTIVE
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


def test_feet_lock_transition_to_resting():
    engine = LavaEngine()
    engine.start_game("P001")
    assert engine.tournament_phase == TournamentPhase.WAITING_FOR_PLAYER

    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9)
    # Tick 0.2s (2 ticks of 0.1s): not locked yet
    engine.update(pose, dt=0.1)
    engine.update(pose, dt=0.1)
    assert engine.tournament_phase == TournamentPhase.WAITING_FOR_PLAYER

    # Tick past 0.5s requirement (4 more ticks of 0.1s -> total 0.6s >= 0.5s)
    for _ in range(4):
        engine.update(pose, dt=0.1)
    assert engine.tournament_phase == TournamentPhase.RESTING
    assert engine.current_round == 1
    assert 7.0 <= engine.rest_timer <= 7.5


def test_resting_countdown_to_round_1_active():
    engine = LavaEngine()
    engine.start_game("P001")
    engine.tournament_phase = TournamentPhase.RESTING
    engine.rest_timer = 7.5

    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9)
    # Step through 7.5s resting phase (76 ticks of 0.1s = 7.6s)
    for _ in range(76):
        engine.update(pose, dt=0.1)

    assert engine.tournament_phase == TournamentPhase.ROUND_ACTIVE
    assert engine.current_round == 1
    assert 29.0 <= engine.round_timer <= 30.0


def test_round_1_rivers_only_and_5s_pacing():
    engine = LavaEngine()
    engine.start_game("P001")
    engine.tournament_phase = TournamentPhase.ROUND_ACTIVE
    engine.current_round = 1

    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9)
    state, wave = engine.update(pose, dt=0.01)

    assert wave is not None
    assert wave.pattern == WavePattern.RIVER
    assert wave.total_warn == 4.0  # 4.0s warn + 1.0s erupt = 5.0s per wave


def test_round_2_circles_only_and_4s_pacing():
    engine = LavaEngine()
    engine.start_game("P001")
    engine.tournament_phase = TournamentPhase.ROUND_ACTIVE
    engine.current_round = 2

    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9)
    state, wave = engine.update(pose, dt=0.01)

    assert wave is not None
    assert wave.pattern in (WavePattern.ISLAND, WavePattern.DONUT)
    assert wave.total_warn == 3.0  # 3.0s warn + 1.0s erupt = 4.0s per wave


def test_round_3_bridges_only_and_3s_pacing():
    engine = LavaEngine()
    engine.start_game("P001")
    engine.tournament_phase = TournamentPhase.ROUND_ACTIVE
    engine.current_round = 3

    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9)
    state, wave = engine.update(pose, dt=0.01)

    assert wave is not None
    assert wave.pattern == WavePattern.BRIDGES
    assert wave.total_warn == 2.0  # 2.0s warn + 1.0s erupt = 3.0s per wave


def test_streak_multipliers_and_hit_penalties():
    # Multiplier tiers
    assert get_streak_multiplier(0) == 1.0
    assert get_streak_multiplier(2) == 1.0
    assert get_streak_multiplier(3) == 1.2
    assert get_streak_multiplier(5) == 1.2
    assert get_streak_multiplier(6) == 1.5
    assert get_streak_multiplier(8) == 1.5
    assert get_streak_multiplier(9) == 2.0
    assert get_streak_multiplier(12) == 2.0

    engine = LavaEngine()
    engine.start_game("P001")
    engine.tournament_phase = TournamentPhase.ROUND_ACTIVE
    engine.state.score = 20
    engine.state.streak = 5

    # Simulate Hit (Burnt): streak reset, hits incremented, NO score penalty
    mask = [[True] * GRID_SIZE for _ in range(GRID_SIZE)]
    wave = LavaWave(
        pattern=WavePattern.RIVER,
        timer=0.05,
        total_warn=4.0,
        target_mask=mask,
        heat=[[0.0] * GRID_SIZE for _ in range(GRID_SIZE)],
    )
    engine.wave = wave

    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9, airborne=False)
    engine.update(pose, dt=0.1)

    assert engine.state.hits == 1
    assert engine.state.score == 20  # No negative deduction on hit
    assert engine.state.streak == 0
    assert "BURNT" in engine.last_banner_msg


def test_intermission_transitions_between_rounds():
    engine = LavaEngine()
    engine.start_game("P001")
    engine.tournament_phase = TournamentPhase.ROUND_ACTIVE
    engine.current_round = 1
    engine.round_timer = 0.05

    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9)
    # Tick past end of Round 1
    engine.update(pose, dt=0.1)

    assert engine.current_round == 2
    assert engine.tournament_phase == TournamentPhase.RESTING
    assert engine.rest_timer == 7.5



def test_round_default_duration_and_stage_splits():
    from shared.contracts import Phase

    # 1. Default initialization (30.0s active play per round)
    engine = LavaEngine()
    assert engine.duration_s == 30.0
    assert engine.round_duration_s == 30.0
    assert engine.state.time_left_s == 30.0

    # 2. Start / restart resets to 30.0s
    engine.state.time_left_s = 15.0
    engine.start_game("P001")
    assert engine.state.time_left_s == 30.0
    assert engine.current_round == 1
    assert engine.round_multiplier == 1.0
    assert engine.tournament_phase == TournamentPhase.WAITING_FOR_PLAYER

    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9, airborne=False)

    # 3. Round 1 Active countdown starts from 30.0s
    engine.tournament_phase = TournamentPhase.ROUND_ACTIVE
    engine.round_timer = 30.0
    state, _ = engine.update(pose, dt=0.1)
    assert 29.8 <= state.time_left_s <= 30.0
    assert engine.current_round == 1
    assert engine.round_multiplier == 1.0
    assert state.phase == Phase.PLAYING

    # 4. Round 1 expiration transitions to Round 2 Resting
    engine.round_timer = 0.05
    state, _ = engine.update(pose, dt=0.1)
    assert engine.current_round == 2
    assert engine.tournament_phase == TournamentPhase.RESTING
    assert state.phase == Phase.PLAYING

    # 5. Round 2 expiration transitions to Round 3 Resting
    engine.tournament_phase = TournamentPhase.ROUND_ACTIVE
    engine.round_timer = 0.05
    state, _ = engine.update(pose, dt=0.1)
    assert engine.current_round == 3
    assert engine.tournament_phase == TournamentPhase.RESTING
    assert state.phase == Phase.PLAYING

    # 6. Final Round expiration transitions to TournamentPhase.COMPLETED and Phase.RESULT
    engine.tournament_phase = TournamentPhase.ROUND_ACTIVE
    engine.round_timer = 0.05
    state, wave = engine.update(pose, dt=0.1)
    assert engine.tournament_phase == TournamentPhase.COMPLETED
    assert state.phase == Phase.RESULT
    assert state.time_left_s == 0.0
    assert wave is None


def test_station_a_idle_and_reset_lifecycle():
    from shared.contracts import Phase
    engine = LavaEngine()
    engine.reset_to_idle()
    assert engine.state.phase == Phase.IDLE
    assert engine.wave is None

    # In IDLE, update() should not tick rounds or waves
    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9)
    state, wave = engine.update(pose, dt=0.5)
    assert state.phase == Phase.IDLE
    assert wave is None

    # Starting game sets PLAYING
    engine.start_game(player_id="P999", player_name="Test Champion")
    assert engine.state.phase == Phase.PLAYING
    assert engine.player_name == "Test Champion"
    assert engine.state.player_id == "P999"

    # Resetting to idle restores IDLE
    engine.reset_to_idle()
    assert engine.state.phase == Phase.IDLE





