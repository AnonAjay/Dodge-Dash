"""Entry point for Station B: Pose Blazer Tournament.
Runs 2-Stage Tournament pose matching with real-time MediaPipe body tracking,
retro 8-bit arcade visual effects, and in-memory synthesized chiptune audio feedback.

Run from repo root:
    python -m station_b                    # Pose Blazer Tournament (Live Camera)
    python -m station_b --keyboard         # Keyboard simulation mode
    python -m station_b --fullscreen       # Fullscreen mode
"""
from __future__ import annotations

import argparse
import datetime
import math
import time
import pygame

from pose_core.capture import CameraPoseSource, PoseTracker
from pose_core.floor_map import FloorMapper
from pose_core.pose_matcher import extract_pose_angles
from shared.contracts import Phase, PoseProvider, ScoreEvent
from shared.station_client import StationNetworkClient
from station_b.keyboard_input import KeyboardWallPoseProvider
from station_b.logic import BlazerGameEngine
from station_b.outbox import enqueue_score
from station_b.render import BlazerArenaRenderer
from station_b.sounds import BlazerSoundManager


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
    parser.add_argument("--server", type=str, default=None, help="Laptop 3 server URL (e.g. http://192.168.43.10:8000)")
    parser.add_argument("--no-voice", action="store_true", help="Disable text-to-speech announcer")
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

    # 3. Audio & Announcer System
    sounds = BlazerSoundManager(enable_voice=not args.no_voice)

    clock = pygame.time.Clock()
    score_saved = False
    result_timer_s = 0.0
    last_player_name = ""
    last_player_score: int | None = None

    print(f"[Station B] Initializing POSE BLAZER TOURNAMENT ({args.rounds} Rounds)...")
    engine = BlazerGameEngine(total_rounds=args.rounds, tournament_mode=True)
    engine.reset_to_idle()
    renderer = BlazerArenaRenderer(fullscreen=args.fullscreen)

    # 4. Background Network Dispatch Client
    net_client = StationNetworkClient(station="B", server_url=args.server)
    net_client.start()
    if net_client.server_url:
        print(f"[Station B] Connected to Laptop 3 Server: {net_client.server_url}")

    # Audio State Trackers
    prev_locked = False
    prev_skips = 0
    prev_stage = 1
    prev_phase = None
    last_tick_sec = -1

    try:
        running = True
        while running:
            dt = clock.tick(60) / 1000.0

            # Auto-dispatch next player from Laptop 3 queue if idle, waiting, or tournament ended
            curr_tp_str = str(getattr(engine, "tournament_phase", ""))
            if engine.state.phase in (Phase.IDLE, Phase.RESULT) or curr_tp_str in ("waiting", "completed") or score_saved:
                dispatched = net_client.pop_next_player()
                if dispatched:
                    new_pid = dispatched["player_id"]
                    new_name = dispatched.get("name", new_pid)
                    print(f"[Station B] Auto-dispatching next player from queue: {new_name} ({new_pid})")
                    engine.start_game(player_id=new_pid, player_name=new_name)
                    score_saved = False
                    result_timer_s = 0.0
                    prev_locked = False
                    prev_skips = 0
                    prev_stage = 1
                    last_tick_sec = -1
                    sounds.play_game_start()
                    sounds.play_stage_start(1)
                    if new_name:
                        sounds.speak(f"Welcome {new_name}. Step into frame.")

            # Update live station heartbeat
            net_client.update_status(
                status=engine.state.phase.value if engine.state.phase == Phase.IDLE else curr_tp_str,
                player_id=engine.state.player_id,
                score=engine.state.score,
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        if engine.state.phase == Phase.IDLE:
                            print(f"[Station B] Starting tournament for {args.player} via keyboard trigger.")
                            engine.start_game(player_id=args.player, player_name="Player 1")
                            score_saved = False
                            prev_locked = False
                            prev_skips = 0
                            prev_stage = 1
                            last_tick_sec = -1
                            sounds.play_game_start()
                            sounds.play_stage_start(1)
                            sounds.speak("Get ready. Match the poses.")
                        elif engine.state.phase == Phase.RESULT:
                            print("[Station B] Skipping results screen back to Start Menu.")
                            engine.reset_to_idle()
                            score_saved = False
                    elif event.key == pygame.K_r:
                        engine.start_game(player_id=args.player, player_name="Player 1")
                        score_saved = False
                        result_timer_s = 0.0
                        prev_locked = False
                        prev_skips = 0
                        prev_stage = 1
                        last_tick_sec = -1
                        sounds.play_game_start()
                        sounds.play_stage_start(1)
                    elif event.key in (pygame.K_F11, pygame.K_f):
                        renderer.toggle_fullscreen()

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
                            [float(lm[0]) / cw, float(lm[1]) / ch, float(lm[2]) if len(lm) > 2 else 1.0]
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

            # Audio Events Dispatch
            curr_tp = getattr(engine, "tournament_phase", None)

            # 1. Intermission start announcement
            if curr_tp == "intermission" and prev_phase != "intermission":
                sounds.play_stage_start(engine.current_round)

            # 2. Intermission countdown ticks (3, 2, 1)
            if curr_tp == "intermission":
                cd_s = int(math.ceil(engine.intermission_timer_s))
                if 1 <= cd_s <= 3 and cd_s != last_tick_sec:
                    sounds.play_tick()
                    last_tick_sec = cd_s
                elif cd_s > 3:
                    last_tick_sec = -1

            # 3. Pose Locked In
            if engine.is_pose_locked and not prev_locked:
                pct = engine.last_match_result.match_pct if engine.last_match_result else 0.0
                if pct >= 85.0:
                    sounds.play_perfection()
                else:
                    sounds.play_snap()
            prev_locked = engine.is_pose_locked

            # 4. Skip or Elimination
            if engine.consecutive_skips > prev_skips:
                if engine.consecutive_skips >= 4:
                    sounds.play_disqualified()
                else:
                    sounds.play_skip()
            prev_skips = engine.consecutive_skips

            # 5. Round 2 Global Countdown final 5-second Ticks
            if curr_tp == "active" and engine.current_round == 2:
                r2_s = int(math.ceil(engine.round_2_timer_s))
                if 1 <= r2_s <= 5 and r2_s != last_tick_sec:
                    sounds.play_tick()
                    last_tick_sec = r2_s
                elif r2_s > 5:
                    last_tick_sec = -1

            prev_phase = curr_tp

            # Save score when tournament ends and handle 8s auto-return to IDLE
            if state.phase == Phase.RESULT:
                if not score_saved:
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
                    sounds.play_game_over()
                    result_timer_s = 8.0  # Hold for 8 seconds
                    last_player_name = getattr(engine, "player_name", state.player_id or "Player 1")
                    last_player_score = state.score
                    print(f"[Station B] Tournament score saved to outbox: {state.score} pts. Holding 8s...")
                else:
                    result_timer_s -= dt
                    if result_timer_s <= 0.0:
                        print("[Station B] Result screen timeout elapsed. Returning to Start Menu (Phase.IDLE).")
                        engine.reset_to_idle()
                        score_saved = False

            # Render Split HD UI
            renderer.draw(
                state=state,
                engine=engine,
                cam_frame=cam_frame,
                debug_info=debug_info,
                match_res=engine.last_match_result,
                player_angles=player_angles,
                fps=clock.get_fps(),
                dt=dt,
                is_connected=bool(net_client.server_url),
                last_player_name=last_player_name,
                last_player_score=last_player_score,
            )

    finally:
        net_client.stop()
        sounds.stop_all()
        provider.stop()
        pygame.quit()
        print("[Station B] Closed cleanly.")


if __name__ == "__main__":
    main()
