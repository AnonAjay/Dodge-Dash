"""Entry point for Station B: Wall Shape (Hole in the Wall).
Runs oncoming 3D perspective walls with real-time body pose matching.

Run from repo root:
    python -m station_b               # Live Camera Mode
    python -m station_b --keyboard    # Keyboard Simulation Mode (WASD to move, 1-8 to switch pose)
"""
from __future__ import annotations

import argparse
import datetime
import time
import pygame

from pose_core.capture import CameraPoseSource, PoseTracker
from pose_core.floor_map import FloorMapper
from pose_core.pose_matcher import extract_pose_angles
from shared.contracts import Phase, PoseProvider, ScoreEvent
from station_b.keyboard_input import KeyboardWallPoseProvider
from station_b.logic import WallShapeEngine
from station_b.outbox import enqueue_score
from station_b.render import WallArenaRenderer


def install_pose_tracker_hook() -> None:
    """Attach full 33-landmark skeleton extraction to PoseTracker at runtime."""
    try:
        if getattr(PoseTracker, "_hooked_for_station_b", False):
            return

        orig_update = PoseTracker.update

        def hooked_update(self, people, img_w: int, img_h: int, ts: float, *args, **kwargs):
            pf = orig_update(self, people, img_w, img_h, ts, *args, **kwargs)
            if people and pf.tracked:
                matched_lms = people[0]
                self.debug["landmarks_px"] = [
                    (float(lm.x * img_w), float(lm.y * img_h), float(getattr(lm, "visibility", 1.0)))
                    for lm in matched_lms
                ]
            return pf

        PoseTracker.update = hooked_update
        PoseTracker._hooked_for_station_b = True
    except Exception:
        pass


def main() -> None:
    install_pose_tracker_hook()
    parser = argparse.ArgumentParser(description="Station B: Wall Shape (Hole in the Wall)")
    parser.add_argument("--keyboard", action="store_true", help="Keyboard mode (WASD to move, 1-8 to switch pose)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--duration", type=float, default=60.0, help="Game duration in seconds (default: 60)")
    parser.add_argument("--player", type=str, default="P001", help="Player ID (default: P001)")
    parser.add_argument("--fullscreen", action="store_true", help="Launch in fullscreen mode")
    args = parser.parse_args()

    # 1. Floor Calibration
    mapper = FloorMapper.load()
    if mapper is not None:
        print(f"[Station B] Loaded calibrated arena: {mapper.w_m:.2f}m x {mapper.h_m:.2f}m")
    else:
        print("[Station B] Warning: calibration.local.json not found. Running with default geometry.")

    # 2. Select Pose Provider
    provider: PoseProvider
    if args.keyboard:
        provider = KeyboardWallPoseProvider(arena_w_m=1.8, arena_d_m=1.8)
        print("[Station B] Running in KEYBOARD mode: WASD to move, 1-8 to select pose.")
    else:
        try:
            provider = CameraPoseSource(args.camera, mapper=mapper)
            print(f"[Station B] Opening camera {args.camera} with body pose tracking...")
        except Exception as e:
            print(f"[Station B] Could not open camera ({e}). Falling back to keyboard.")
            provider = KeyboardWallPoseProvider(arena_w_m=1.8, arena_d_m=1.8)
            args.keyboard = True

    provider.start()

    # 3. Game Engine & Renderer
    engine = WallShapeEngine(duration_s=args.duration)
    engine.start_game(player_id=args.player)

    renderer = WallArenaRenderer(mapper=mapper, fullscreen=args.fullscreen)
    clock = pygame.time.Clock()
    score_saved = False

    print("[Station B] Wall Shape running at 60 FPS. Match the poses!")

    try:
        running = True
        while running:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        engine.start_game(player_id=args.player)
                        score_saved = False
                    elif event.key in (pygame.K_F11, pygame.K_f):
                        renderer.toggle_fullscreen()

            # Handle keyboard input
            player_angles: dict[str, float | None] = {}
            if args.keyboard and isinstance(provider, KeyboardWallPoseProvider):
                keys = pygame.key.get_pressed()
                provider.handle_keys(keys, dt)
                player_angles = provider.get_angles()

            # Snapshot pose and camera frame
            cam_frame = None
            debug_info = None
            if hasattr(provider, "snapshot"):
                pose, cam_frame, debug_info = provider.snapshot()
                if debug_info and "landmarks_px" in debug_info:
                    player_angles = extract_pose_angles(debug_info["landmarks_px"])
            else:
                pose = provider.latest()

            # Tick game physics
            state, wall = engine.update(pose, dt, player_angles=player_angles)

            # Record score upon game completion
            if state.phase == Phase.RESULT and not score_saved:
                total_attempts = engine.walls_passed + engine.walls_hit
                accuracy = round(engine.walls_passed / max(1, total_attempts), 2)
                event_id = f"B-{state.player_id}-{int(time.time())}"
                score_event = ScoreEvent(
                    event_id=event_id,
                    player_id=state.player_id or "P001",
                    station="B",
                    game="wall",
                    score=state.score,
                    accuracy=accuracy,
                    best_streak=state.best_streak,
                    duration_s=args.duration,
                    finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
                enqueue_score(score_event)
                score_saved = True

            # Render frame
            renderer.draw(
                state=state,
                wall=wall,
                pose=pose,
                cam_frame=cam_frame,
                debug=debug_info,
                banner_msg=engine.last_banner_msg,
                banner_color=engine.last_banner_color,
                fps=clock.get_fps(),
            )

    finally:
        provider.stop()
        pygame.quit()
        print("[Station B] Closed cleanly.")


if __name__ == "__main__":
    main()
