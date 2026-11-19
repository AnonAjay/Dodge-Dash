"""Entry point for Station B: Pose Blazer Tournament.
Runs 2-Stage Tournament pose matching with real-time MediaPipe body tracking.

Run from repo root:
    python -m station_b                    # Pose Blazer Tournament (Live Camera)
    python -m station_b --keyboard         # Keyboard simulation mode
    python -m station_b --fullscreen       # Fullscreen mode
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
from station_b.logic import BlazerGameEngine
from station_b.outbox import enqueue_score
from station_b.render import BlazerArenaRenderer


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
    parser = argparse.ArgumentParser(description="Station B: Pose Blazer Tournament")
    parser.add_argument("--keyboard", action="store_true", help="Keyboard mode (WASD / 1-8 key simulation)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--player", type=str, default="P001", help="Player ID (default: P001)")
    parser.add_argument("--fullscreen", action="store_true", help="Launch in fullscreen mode")
    parser.add_argument("--rounds", type=int, default=6, help="Total rounds (default: 6)")
    args = parser.parse_args()

    # 1. Floor Calibration (used by mapper and camera thread)
    mapper = FloorMapper.load()
    if mapper is not None:
        print(f"[Station B] Loaded calibrated arena: {mapper.w_m:.2f}m x {mapper.h_m:.2f}m")
    else:
        print("[Station B] Warning: calibration.local.json not found. Running with default geometry.")

    # 2. Select Pose Provider
    provider: PoseProvider
    if args.keyboard:
        provider = KeyboardWallPoseProvider(arena_w_m=1.8, arena_d_m=1.8)
        print("[Station B] Running in KEYBOARD mode.")
    else:
        try:
            provider = CameraPoseSource(args.camera, mapper=mapper)
            print(f"[Station B] Opening camera {args.camera} with MediaPipe pose tracking...")
        except Exception as e:
            print(f"[Station B] Could not open camera ({e}). Falling back to keyboard.")
            provider = KeyboardWallPoseProvider(arena_w_m=1.8, arena_d_m=1.8)
            args.keyboard = True

    provider.start()

    clock = pygame.time.Clock()
    score_saved = False

    print(f"[Station B] Launching POSE BLAZER TOURNAMENT ({args.rounds} Rounds)...")
    engine = BlazerGameEngine(total_rounds=args.rounds, tournament_mode=True)
    engine.start_game(player_id=args.player)
    renderer = BlazerArenaRenderer(fullscreen=args.fullscreen)

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
                    elif event.key == pygame.K_h:
                        engine.check_head = not engine.check_head
                        engine.banner_msg = f"HEAD CHECK: {'ON' if engine.check_head else 'OFF'}"

            cam_frame = None
            debug_info = None
            player_angles: dict[str, float | None] = {}
            player_head_norm: tuple[float, float] | None = None
            player_lms_norm: list[list[float]] | None = None

            if hasattr(provider, "snapshot"):
                pose, cam_frame, debug_info = provider.snapshot()
                if debug_info and "landmarks_px" in debug_info and len(debug_info["landmarks_px"]) > 0:
                    player_angles = extract_pose_angles(debug_info["landmarks_px"])
                    if cam_frame is not None:
                        ch, cw = cam_frame.shape[:2]
                        player_lms_norm = [
                            [float(lm[0]) / cw, float(lm[1]) / ch]
                            for lm in debug_info["landmarks_px"]
                        ]
                        lm0 = debug_info["landmarks_px"][0]
                        hx, hy = float(lm0[0]), float(lm0[1])
                        hvis = float(lm0[2]) if len(lm0) > 2 else 1.0
                        if hvis > 0.35:
                            player_head_norm = (hx / cw, hy / ch)
            else:
                pose = provider.latest()

            if args.keyboard and isinstance(provider, KeyboardWallPoseProvider):
                keys = pygame.key.get_pressed()
                provider.handle_keys(keys, dt)
                player_angles = provider.get_angles()
                player_head_norm = provider.get_head()

            # Update Game Physics & Timers
            state, _ = engine.update(
                pose,
                dt,
                player_angles=player_angles,
                player_head_norm=player_head_norm,
                player_landmarks_norm=player_lms_norm,
            )

            # Save score when tournament ends
            if state.phase == Phase.RESULT and not score_saved:
                total_att = state.dodges + state.hits
                acc = round(state.dodges / max(1, total_att), 2)
                event_id = f"B-BLAZER-{state.player_id}-{int(time.time())}"
                score_event = ScoreEvent(
                    event_id=event_id,
                    player_id=state.player_id or "P001",
                    station="B",
                    game="blazer",
                    score=state.score,
                    accuracy=acc,
                    best_streak=state.streak,
                    duration_s=engine.round_duration_s * engine.total_rounds,
                    finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
                enqueue_score(score_event)
                score_saved = True
                print(f"[Station B] Tournament score saved to outbox: {state.score} pts")

            # Render Split HD UI
            renderer.draw(
                state=state,
                engine=engine,
                cam_frame=cam_frame,
                debug_info=debug_info,
                match_res=engine.last_match_result,
                player_angles=player_angles,
                fps=clock.get_fps(),
            )

    finally:
        provider.stop()
        pygame.quit()
        print("[Station B] Closed cleanly.")


if __name__ == "__main__":
    main()
