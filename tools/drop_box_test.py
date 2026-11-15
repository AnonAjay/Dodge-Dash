"""
drop_box_test.py - the simplest possible game to test floor mapping (a test harness, not Station A).

HOW TO PLAY
  * The big square is the FLOOR of your taped box, seen from above.
  * The green circle "YOU" is where your feet are on that floor.
  * A box falls onto a spot: the orange square is where it will land.
    Walk OUT of the orange square before the countdown hits 0 (or jump!).
  * Half the boxes aim at wherever you are standing right now.

Run from the repo root:
    python -m tools.calibrate_floor --floor-w 1.245 --floor-h 1.245   # once per venue
    python -m tools.drop_box_test --keyboard      # no camera, WASD + SPACE
    python -m tools.drop_box_test                 # camera mode (uses calibration.local.json)

Keys:  V switch view | C recalibrate | R restart | Q quit
"""
from __future__ import annotations

# keep pose_core imports above cv2 (pose_core/__init__ sets a Windows camera env var)
import argparse
import atexit
import random
import sys
import time
from dataclasses import dataclass

from pose_core.capture import CameraPoseSource
from pose_core.floor_map import FloorMapper
from shared.contracts import PoseFrame
from tools.calibrate_floor import calibrate_floor

import cv2
import numpy as np

S = 600                # map panel size (pixels)
SIDE_W = 290           # sidebar width
PAD = 40               # border around the floor on the map
WARN_S = 0.5           # warning time before a box lands (bigger = slower)
IMPACT_S = 0.2        # how long the red "landed" square is shown
SPAWN_EVERY_S = 0.45     # seconds between new boxes (bigger = calmer)
BOX_FRACTION = 0.32     # danger square = this fraction of the floor size (49 in box -> ~0.4 m)
CHASE_PROB = 0.5       # chance a box aims at the player's current position

FONT = cv2.FONT_HERSHEY_SIMPLEX
ORANGE, RED, GREEN, CYAN, WHITE, GREY = (0, 165, 255), (0, 0, 230), (0, 220, 0), (255, 200, 0), (255, 255, 255), (170, 170, 170)


@dataclass
class Player:
    """What this test game needs, derived from a PoseFrame. y grows AWAY from the camera."""
    tracked: bool
    in_box: bool
    x: float
    y: float
    airborne: bool


def player_from(pf: PoseFrame, mapper: FloorMapper) -> Player:
    if not pf.tracked or pf.floor_x_m is None or pf.floor_y_m is None:
        return Player(False, False, mapper.w_m / 2, mapper.h_m / 2, False)
    return Player(True, mapper.inside(pf.floor_x_m, pf.floor_y_m),
                  pf.floor_x_m, pf.floor_y_m, pf.airborne)


@dataclass
class Hazard:
    cx: float
    cy: float
    size: float
    age: float = 0.0
    resolved: bool = False


class DropBoxGame:
    """Pure game logic: no drawing, no camera. Easy to unit test."""

    def __init__(self, w_m: float, h_m: float):
        self.w_m, self.h_m = w_m, h_m
        self.box_m = round(BOX_FRACTION * min(w_m, h_m), 2)
        self.reset()

    def reset(self):
        self.hazards: list[Hazard] = []
        self.score = 0
        self.hits = 0
        self.dodges = 0
        self.spawn_timer = 0.5
        self.message = ""
        self.message_t = 0.0

    def _spawn(self, pl: Player):
        half = self.box_m / 2
        if pl.in_box and random.random() < CHASE_PROB:
            cx, cy = pl.x, pl.y
        else:
            cx = random.uniform(half, self.w_m - half)
            cy = random.uniform(half, self.h_m - half)
        cx = min(max(cx, half), self.w_m - half)
        cy = min(max(cy, half), self.h_m - half)
        self.hazards.append(Hazard(cx, cy, self.box_m))

    def update(self, dt: float, pl: Player):
        self.message_t = max(0.0, self.message_t - dt)
        self.spawn_timer -= dt
        if self.spawn_timer <= 0 and pl.tracked and pl.in_box:
            self._spawn(pl)
            self.spawn_timer = SPAWN_EVERY_S

        for hz in self.hazards:
            hz.age += dt
            if hz.age >= WARN_S and not hz.resolved:
                hz.resolved = True
                if pl.tracked and pl.in_box:
                    if self.inside(hz, pl) and not pl.airborne:
                        self.hits += 1
                        self.score = max(0, self.score - 5)
                        self.message, self.message_t = "HIT!  -5", 0.8
                    else:
                        self.dodges += 1
                        self.score += 10
                        self.message = "JUMPED!  +10" if self.inside(hz, pl) else "DODGED!  +10"
                        self.message_t = 0.8
        self.hazards = [h for h in self.hazards if h.age < WARN_S + IMPACT_S]

    @staticmethod
    def inside(hz: Hazard, pl: Player) -> bool:
        half = hz.size / 2
        return abs(pl.x - hz.cx) <= half and abs(pl.y - hz.cy) <= half


# ---- drawing helpers ---------------------------------------------------------
def text(img, s, org, scale=0.55, color=WHITE, thick=1, center=False):
    x, y = org
    if center:
        (tw, th), _ = cv2.getTextSize(s, FONT, scale, thick)
        x, y = x - tw // 2, y + th // 2
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, color, thick, cv2.LINE_AA)


def draw_map(game: DropBoxGame, pl: Player) -> np.ndarray:
    img = np.full((S, S, 3), 28, np.uint8)
    k = (S - 2 * PAD) / max(game.w_m, game.h_m)

    def m2p(x, y):
        # y grows AWAY from the camera (contract), so the FAR edge is drawn at the TOP
        return int(PAD + x * k), int(PAD + (game.h_m - y) * k)

    # the floor, with a 0.5 m grid
    cv2.rectangle(img, m2p(0, game.h_m), m2p(game.w_m, 0), (72, 72, 72), -1)
    g = 0.0
    while g <= max(game.w_m, game.h_m) + 1e-6:
        if g <= game.w_m + 1e-6:
            cv2.line(img, m2p(g, 0), m2p(g, game.h_m), (95, 95, 95), 1)
        if g <= game.h_m + 1e-6:
            cv2.line(img, m2p(0, g), m2p(game.w_m, g), (95, 95, 95), 1)
        g += 0.5
    cv2.rectangle(img, m2p(0, game.h_m), m2p(game.w_m, 0), (210, 210, 210), 2)
    text(img, "FAR from camera", (S // 2, 20), 0.5, GREY, 1, True)
    text(img, "NEAR (camera side)", (S // 2, S - 16), 0.5, GREY, 1, True)

    in_danger = False
    for hz in game.hazards:
        half = hz.size / 2
        p1, p2 = m2p(hz.cx - half, hz.cy + half), m2p(hz.cx + half, hz.cy - half)  # top-left, bottom-right
        c = m2p(hz.cx, hz.cy)
        if hz.age < WARN_S:
            f = hz.age / WARN_S
            # landing spot (gets more orange as time runs out)
            cv2.rectangle(img, p1, p2, (0, int(40 + 100 * f), int(80 + 175 * f)), -1)
            cv2.rectangle(img, p1, p2, ORANGE, 3)
            text(img, f"{max(0.0, WARN_S - hz.age):.1f}", (c[0], p2[1] - 16), 0.8, WHITE, 2, True)
            if game.inside(hz, pl) and not pl.airborne:
                in_danger = True
        else:
            cv2.rectangle(img, p1, p2, RED, -1)
            cv2.rectangle(img, p1, p2, WHITE, 3)
            text(img, "BOOM", c, 0.8, WHITE, 2, True)

    if pl.tracked:
        c = m2p(pl.x, pl.y)
        if pl.airborne:
            cv2.circle(img, c, 24, CYAN, 3)
        cv2.circle(img, c, 14, RED if in_danger else (GREEN if pl.in_box else (0, 0, 255)), -1)
        cv2.circle(img, c, 14, WHITE, 2)
        text(img, "YOU", (c[0], c[1] - 28), 0.6, WHITE, 2, True)

    if not pl.tracked:
        text(img, "TRACKING LOST", (S // 2, S // 2 - 16), 0.9, (0, 0, 255), 2, True)
        text(img, "show your whole body (head to feet)", (S // 2, S // 2 + 20), 0.6, (0, 0, 255), 2, True)
    elif not pl.in_box:
        text(img, "STEP INTO THE BOX", (S // 2, S // 2), 0.9, ORANGE, 2, True)
    elif in_danger:
        text(img, "MOVE!", (S // 2, 60), 1.4, RED, 4, True)
    if game.message_t > 0:
        col = RED if game.message.startswith("HIT") else GREEN
        text(img, game.message, (S // 2, 110), 1.0, col, 3, True)
    return img


def draw_sidebar(cam_view, game: DropBoxGame, fps: float, pose_fps: float, keyboard: bool) -> np.ndarray:
    side = np.full((S, SIDE_W, 3), 42, np.uint8)
    y = 12
    if cam_view is not None:
        w = SIDE_W - 20
        h = int(w * cam_view.shape[0] / cam_view.shape[1])
        side[y:y + h, 10:10 + w] = cv2.resize(cam_view, (w, h))
        y += h + 16
        text(side, "camera: what the computer sees", (10, y), 0.45, GREY)
        y += 26
    elif keyboard:
        text(side, "KEYBOARD MODE", (10, y + 10), 0.6, ORANGE, 2)
        text(side, "WASD = walk, SPACE = jump", (10, y + 36), 0.5, WHITE)
        y += 64
    else:
        text(side, "waiting for camera...", (10, y + 10), 0.6, ORANGE, 2)
        y += 64

    text(side, "WHAT YOU SEE", (10, y), 0.55, WHITE, 2)
    y += 28
    rows = [
        ("circle", GREEN, "YOU (your feet, from above)"),
        ("square", ORANGE, "DANGER spot - leave it!"),
        ("square", RED, "BOOM - hit if you're in it"),
        ("ring", CYAN, "ring = jumping = safe"),
    ]
    for shape, col, label in rows:
        if shape == "circle":
            cv2.circle(side, (22, y - 4), 8, col, -1)
        elif shape == "square":
            cv2.rectangle(side, (13, y - 13), (31, y + 5), col, -1)
        else:
            cv2.circle(side, (22, y - 4), 10, col, 2)
        text(side, label, (42, y), 0.46)
        y += 28

    y += 14
    text(side, f"Score  {game.score}", (10, y), 0.8, WHITE, 2)
    text(side, f"Dodged {game.dodges}    Hits {game.hits}", (10, y + 30), 0.55, GREY)
    text(side, f"game {fps:.0f} FPS" + ("" if keyboard else f"   pose {pose_fps:.0f} FPS"),
         (10, y + 54), 0.45, GREY)
    text(side, "V view  C calibrate", (10, S - 38), 0.45, GREY)
    text(side, "R restart   Q quit", (10, S - 16), 0.45, GREY)
    return side


def draw_debug(frame, mapper: FloorMapper, pf: PoseFrame, debug: dict, in_box: bool):
    """Camera picture with the floor box, foot points and numbers drawn on it."""
    ok_col = (0, 255, 0) if (pf.tracked and in_box) else (0, 0, 255)
    poly = mapper.corners_px.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [poly], True, ok_col, 2)
    for p in debug.get("pts", []):
        cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0, 255, 255), -1)
    if debug.get("foot_px") is not None:
        fx, fy = debug["foot_px"]
        cv2.circle(frame, (int(fx), int(fy)), 9, (255, 0, 255), 2)
    xs = "-" if pf.floor_x_m is None else f"{pf.floor_x_m:.2f}"
    ys = "-" if pf.floor_y_m is None else f"{pf.floor_y_m:.2f}"
    lines = [
        f"floor: ({xs}, {ys}) m",
        f"airborne: {pf.airborne}   lift: {debug.get('lift', 0):.2f}",
        f"tracked: {pf.tracked}   arms_up: {pf.arms_up}",
        f"torso seen (needed for jump): {debug.get('body_ok', False)}",
    ]
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (8, 20 + i * 20), FONT, 0.55, (255, 255, 255), 2)
    return frame


def debug_cam_panel(cam_view) -> np.ndarray:
    panel = np.zeros((S, S, 3), np.uint8)
    h = S * 3 // 4
    panel[(S - h) // 2:(S - h) // 2 + h] = cv2.resize(cam_view, (S, h))
    return panel


# ---- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyboard", action="store_true", help="no camera, WASD + SPACE")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--floor-w", type=float, default=1.245, help="taped box width (m)")
    ap.add_argument("--floor-h", type=float, default=1.245, help="taped box depth (m)")
    args = ap.parse_args()

    source = mapper = None
    w_m, h_m = args.floor_w, args.floor_h

    if not args.keyboard:
        mapper = None if args.recalibrate else FloorMapper.load()
        try:
            source = CameraPoseSource(args.camera, mapper=mapper)
            source.start()
        except (RuntimeError, FileNotFoundError) as e:
            sys.exit(str(e))
        atexit.register(source.stop)  # release the camera however the script exits
        t0 = time.monotonic()
        while source.latest_frame() is None:
            if time.monotonic() - t0 > 5:
                sys.exit("No frames from the camera.")
            time.sleep(0.05)
        if mapper is None:
            mapper = calibrate_floor(source.latest_frame, w_m, h_m)
            if mapper is None:
                return
            try:
                source.set_mapper(mapper)
            except FileNotFoundError as e:
                sys.exit(str(e))
        w_m, h_m = mapper.w_m, mapper.h_m

    game = DropBoxGame(w_m, h_m)
    kb_x, kb_y, jump_until = w_m / 2, h_m / 2, 0.0
    last_t, fps = time.monotonic(), 0.0
    cam_view = None
    debug_view = False

    while True:
        now = time.monotonic()
        dt = min(now - last_t, 0.1)
        last_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        if args.keyboard:
            pl = Player(True, True, kb_x, kb_y, now < jump_until)
        else:
            if source.error:
                sys.exit("Tracking error: " + source.error)
            pf, frame, debug = source.snapshot()
            pl = player_from(pf, mapper)
            cam_view = draw_debug(frame, mapper, pf, debug, pl.in_box) if frame is not None else None

        game.update(dt, pl)
        panel = draw_map(game, pl)
        pose_fps = 0.0 if source is None else source.pose_fps
        if debug_view and cam_view is not None:
            canvas = np.hstack([debug_cam_panel(cam_view), panel])
        else:
            canvas = np.hstack([panel, draw_sidebar(cam_view, game, fps, pose_fps, args.keyboard)])
        cv2.imshow("Drop Box (baseline)", canvas)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("r"):
            game.reset()
        elif k == ord("v"):
            debug_view = not debug_view
        elif k == ord("c") and source is not None:
            m = calibrate_floor(source.latest_frame, w_m, h_m)
            if m is not None:
                mapper = m
                source.set_mapper(m)
        if args.keyboard:
            step = 0.06
            if k == ord("a"): kb_x = max(-0.1, kb_x - step)
            if k == ord("d"): kb_x = min(w_m + 0.1, kb_x + step)
            if k == ord("w"): kb_y = min(h_m + 0.1, kb_y + step)   # W = walk AWAY from the camera (y grows)
            if k == ord("s"): kb_y = max(-0.1, kb_y - step)
            if k == ord(" "): jump_until = time.monotonic() + 0.5

    if source is not None:
        source.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass  # Ctrl+C: atexit releases the camera
    finally:
        cv2.destroyAllWindows()
