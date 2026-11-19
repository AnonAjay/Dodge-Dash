"""Entry point for Station A: Floor is Lava Game.
Runs the continuous thermal lava simulation with safe island generator,
jump immunity, and single full-arena display.

Run from repo root:
    python -m station_a               # Live Camera Mode
    python -m station_a --keyboard    # Keyboard Simulation Mode (WASD to walk, SPACE to jump)
"""
from __future__ import annotations

import argparse
import datetime
import time
import pygame

from pose_core.capture import CameraPoseSource
from pose_core.floor_map import FloorMapper
from shared.contracts import Phase, PoseProvider, ScoreEvent
from shared.station_client import StationNetworkClient
from station_a.keyboard_input import KeyboardPoseProvider
from station_a.logic import LavaEngine, ROUND_DURATION_S, TournamentPhase
from station_a.outbox import enqueue_score
from station_a.render import LavaArenaRenderer, install_pose_tracker_hook


def main() -> None:
    install_pose_tracker_hook()
    parser = argparse.ArgumentParser(description="Titans' Arcade — Station A: Floor is Lava")
    parser.add_argument("--keyboard", action="store_true", help="Keyboard controls (WASD to walk, SPACE to jump)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--duration", type=float, default=ROUND_DURATION_S, help="Round duration in seconds (default: 30)")
    parser.add_argument("--player", type=str, default="P001", help="Player ID (default: P001)")
    parser.add_argument("--server", type=str, default=None, help="Laptop 3 server URL (e.g. http://192.168.43.10:8000)")
    parser.add_argument("--fullscreen", action="store_true", help="Launch in full-screen display mode")
    args = parser.parse_args()

    # 1. Load Floor Calibration
    mapper = FloorMapper.load()
    if mapper is not None:
        arena_w = mapper.w_m
        arena_h = mapper.h_m
        print(f"[Station A] Loaded calibrated arena: {arena_w:.2f}m x {arena_h:.2f}m")
    else:
        arena_w = 1.8
        arena_h = 1.8
        if not args.keyboard:
            print("[Station A] Warning: calibration.local.json not found. Using default 1.8m x 1.8m arena.")
            default_corners = [[25.0, 440.0], [544.0, 449.0], [494.0, 241.0], [198.0, 236.0]]
            mapper = FloorMapper(default_corners, w_m=arena_w, h_m=arena_h)

    # 2. Select Pose Provider
    provider: PoseProvider
    if args.keyboard:
        provider = KeyboardPoseProvider(grid_w_m=arena_w, grid_h_m=arena_h)
        print("[Station A] Running in KEYBOARD mode: WASD to walk, SPACE to jump.")
        provider.start()
    else:
        try:
            provider = CameraPoseSource(args.camera, mapper=mapper)
            print(f"[Station A] Opening camera {args.camera} with calibrated floor tracking...")
            provider.start()
        except Exception as e:
            print(f"[Station A] Could not open camera ({e}). Falling back to keyboard.")
            provider = KeyboardPoseProvider(grid_w_m=arena_w, grid_h_m=arena_h)
            provider.start()
            args.keyboard = True

    # 3. Initialize Game Engine & Renderer
    engine = LavaEngine(duration_s=args.duration, arena_w_m=arena_w, arena_h_m=arena_h)
    engine.start_game(player_id=args.player)

    # 4. Background Network Dispatch Client
    net_client = StationNetworkClient(station="A", server_url=args.server)
    net_client.start()
    if net_client.server_url:
        print(f"[Station A] Connected to Laptop 3 Server: {net_client.server_url}")

    renderer = LavaArenaRenderer(mapper=mapper, arena_w_m=arena_w, arena_h_m=arena_h, fullscreen=args.fullscreen)
    clock = pygame.time.Clock()

    score_saved = False

    print("[Station A] Game running at 60 FPS. Survive the lava!")

    try:
        running = True
        while running:
            dt = clock.tick(60) / 1000.0

            # Auto-launch dispatched player from Laptop 3 queue if idle / round ended
            if engine.state.phase in (Phase.IDLE, Phase.RESULT) or score_saved:
                dispatched = net_client.pop_next_player()
                if dispatched:
                    new_pid = dispatched["player_id"]
                    new_name = dispatched.get("name", "")
                    print(f"[Station A] Auto-dispatching next player from queue: {new_name} ({new_pid})")
                    engine.start_game(player_id=new_pid)
                    score_saved = False

            # Update live station heartbeat
            net_client.update_status(
                status=engine.state.phase.value,
                player_id=engine.state.player_id,
                score=engine.state.score,
            )

            # Handle window and keyboard events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    # Maximized / resized window: surfaces automatically scale in draw()
                    pass
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        # Reset / Restart round
                        engine.start_game(player_id=args.player)
                        score_saved = False
                    elif event.key in (pygame.K_F11, pygame.K_f):
                        renderer.toggle_fullscreen()

            # Update keyboard input provider if applicable
            if args.keyboard and isinstance(provider, KeyboardPoseProvider):
                keys = pygame.key.get_pressed()
                provider.handle_keys(keys, dt)

            # Fetch latest pose, camera frame, and tracking debug (for AR depth occlusion)
            cam_frame = None
            debug_info = None
            if hasattr(provider, "snapshot"):
                pose, cam_frame, debug_info = provider.snapshot()
            else:
                pose = provider.latest()

            # Tick game physics & continuous thermal simulation
            state, wave = engine.update(pose, dt)

            # Record score when round finishes
            if state.phase == Phase.RESULT and not score_saved:
                total_actions = state.dodges + state.hits
                accuracy = round(state.dodges / total_actions, 2) if total_actions > 0 else 0.0
                event_id = f"A-{state.player_id}-{int(time.time())}"
                score_event = ScoreEvent(
                    event_id=event_id,
                    player_id=state.player_id or "P001",
                    station="A",
                    game="dodge",
                    score=state.score,
                    accuracy=accuracy,
                    best_streak=state.best_streak,
                    duration_s=args.duration,
                    finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
                enqueue_score(score_event)
                score_saved = True

            # Render frame (AR Camera View with Depth Occlusion or Virtual Arena)
            renderer.draw(
                state=state,
                wave=wave,
                pose=pose,
                cam_frame=cam_frame,
                debug=debug_info,
                banner_msg=engine.last_banner_msg,
                banner_color=engine.last_banner_color,
                round_title=engine.round_banner_title,
                round_sub=engine.round_banner_subtitle,
                round_color=engine.round_banner_color,
                round_timer=engine.round_banner_timer,
                current_round=engine.current_round,
                round_multiplier=engine.round_multiplier,
                is_resting=(engine.tournament_phase == TournamentPhase.RESTING),
                fps=clock.get_fps(),
            )

    finally:
        net_client.stop()
        provider.stop()
        pygame.quit()
        print("[Station A] Closed cleanly.")


if __name__ == "__main__":
    main()
