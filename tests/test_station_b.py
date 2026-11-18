"""Unit tests for Station B (Wall Shape) game logic."""
from pose_core.pose_matcher import DEFAULT_TEMPLATES
from shared.contracts import Phase, PoseFrame
from station_b.logic import (
    ARENA_DEPTH_M,
    COLLISION_TOLERANCE_M,
    WallShapeEngine,
    WallState,
)


def test_wall_shape_initialization():
    engine = WallShapeEngine(duration_s=60.0)
    engine.start_game("P001")
    assert engine.state.phase == Phase.PLAYING
    assert engine.state.score == 0
    assert engine.state.dodges == 0
    assert engine.state.hits == 0
    assert engine.state.streak == 0


def test_wall_spawns_and_moves_with_dt():
    engine = WallShapeEngine(duration_s=60.0)
    engine.start_game("P001")
    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9)

    # Cooldown is 0.5s at start; tick 0.6s
    for _ in range(6):
        engine.update(pose, dt=0.1)

    assert engine.active_wall is not None
    initial_y = engine.active_wall.y_m
    assert abs(initial_y - ARENA_DEPTH_M) < 0.1

    # Advance 0.5s -> wall must glide forward (smaller y)
    for _ in range(5):
        engine.update(pose, dt=0.1)

    assert engine.active_wall.y_m < initial_y


def test_wall_collision_pass_with_matching_pose():
    engine = WallShapeEngine(duration_s=60.0)
    engine.start_game("P001")
    wall = engine.spawn_wall()
    engine.active_wall = wall

    # Player standing at depth y = 0.9m
    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9)

    # Position wall just in front of player
    wall.y_m = 0.9 + COLLISION_TOLERANCE_M - 0.01

    # Matching T-Pose angles (template 0 is T-Pose)
    matching_angles = {
        "left_shoulder": 95.0,
        "right_shoulder": 95.0,
        "left_elbow": 170.0,
        "right_elbow": 170.0,
        "left_knee": 165.0,
        "right_knee": 165.0,
    }

    state, active_wall = engine.update(pose, dt=0.01, player_angles=matching_angles)
    assert active_wall is not None
    assert active_wall.judged is True
    assert active_wall.state == WallState.PASSED
    assert active_wall.match_pct >= 75.0
    assert state.dodges == 1
    assert state.hits == 0
    assert state.streak == 1
    assert state.score >= 100


def test_wall_collision_fail_with_wrong_pose():
    engine = WallShapeEngine(duration_s=60.0)
    engine.start_game("P001")
    wall = engine.spawn_wall()
    engine.active_wall = wall

    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9)
    wall.y_m = 0.9

    # Wrong pose: arms down (shoulder 20 deg, elbows bent)
    wrong_angles = {
        "left_shoulder": 20.0,
        "right_shoulder": 20.0,
        "left_elbow": 90.0,
        "right_elbow": 90.0,
        "left_knee": 165.0,
        "right_knee": 165.0,
    }

    state, active_wall = engine.update(pose, dt=0.01, player_angles=wrong_angles)
    assert active_wall is not None
    assert active_wall.judged is True
    assert active_wall.state == WallState.HIT
    assert active_wall.match_pct < 75.0
    assert state.dodges == 0
    assert state.hits == 1
    assert state.streak == 0
    assert state.score == 0


def test_wall_despawns_and_cooldown_triggers_next():
    engine = WallShapeEngine(duration_s=60.0)
    engine.start_game("P001")
    wall = engine.spawn_wall()
    engine.active_wall = wall

    # Teleport wall past near edge
    wall.y_m = -0.35
    pose = PoseFrame(tracked=True, floor_x_m=0.9, floor_y_m=0.9)

    # Update should despawn active wall
    state, active_wall = engine.update(pose, dt=0.05)
    assert engine.active_wall is None
    assert engine.cooldown_timer > 0.0

    # Advance through cooldown
    for _ in range(10):
        engine.update(pose, dt=0.1)

    # Next wall spawned with next template
    assert engine.active_wall is not None
    assert engine.active_wall.template.id == DEFAULT_TEMPLATES[1].id
