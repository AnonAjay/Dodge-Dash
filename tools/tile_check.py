"""Tile accuracy check tool for the Floor-is-Lava game.

Measures how reliably floor tracking reports which floor tile the player's feet
are on, so tile dimensions can be chosen based on empirical data.

Orientation convention (confirmed for 3x3 game engine):
  - (0, 0) = near-left corner of the floor box
  - x grows to the right -> col 0 is leftmost (bottom-left), col (cols - 1) is rightmost
  - y grows AWAY from the camera -> row 0 is NEAREST the camera, row (rows - 1) is farthest
  - Points outside [0, w] x [0, h] return None.

Run from repo root:
    python -m tools.tile_check [--cols 3] [--rows 3] [--camera N]
                               [--targets quick|corners|all] [--note "text"]
"""
from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pose_core.capture import CameraPoseSource
from pose_core.floor_map import FloorMapper
from shared.contracts import PoseFrame


# ==============================================================================
# Pure Functions: Tile Geometry & Target Creation
# ==============================================================================

def tile_of(
    x: float,
    y: float,
    w: float,
    h: float,
    cols: int,
    rows: int,
) -> tuple[int, int] | None:
    """Return (col, row) for a point (x, y) on a cols x rows floor grid of size (w, h).

    Orientation convention:
      - (0, 0) = near-left corner of the floor box
      - x grows to the right -> col 0 is leftmost, col (cols - 1) is rightmost
      - y grows AWAY from the camera -> row 0 is NEAREST the camera, row (rows - 1) is farthest
      - Points outside [0, w] x [0, h] return None.
      - Points on outer boundaries: x == w returns cols - 1, y == h returns rows - 1.

    Orientation convention (confirmed for 3x3 game engine): row 0 nearest camera, col 0 bottom-left.
    """
    if x < 0.0 or x > w or y < 0.0 or y > h:
        return None
    if cols <= 0 or rows <= 0 or w <= 0.0 or h <= 0.0:
        return None

    col = min(cols - 1, int(x / (w / cols)))
    row = min(rows - 1, int(y / (h / rows)))
    return (col, row)


def tile_rect_corners(
    col: int,
    row: int,
    w: float,
    h: float,
    cols: int,
    rows: int,
) -> list[tuple[float, float]]:
    """Return 4 corners [(x1, y1), (x2, y1), (x2, y2), (x1, y2)] in floor metres."""
    dx = w / cols
    dy = h / rows
    x1 = col * dx
    x2 = (col + 1) * dx
    y1 = row * dy
    y2 = (row + 1) * dy
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def make_targets(
    cols: int,
    rows: int,
    w: float,
    h: float,
    mode: str = "quick",
) -> list[dict]:
    """Generate target positions for accuracy testing.

    Each target is a dict:
      {"target_idx": int, "col": int, "row": int, "mark_x": float, "mark_y": float, "name": str}
    """
    targets_grid: list[tuple[int, int, str]] = []
    if mode == "corners":
        raw = [
            (0, 0, "Near-Left Corner"),
            (cols - 1, 0, "Near-Right Corner"),
            (0, rows - 1, "Far-Left Corner"),
            (cols - 1, rows - 1, "Far-Right Corner"),
        ]
        seen = set()
        for c, r, name in raw:
            if (c, r) not in seen:
                seen.add((c, r))
                targets_grid.append((c, r, name))
    elif mode == "all":
        for r in range(rows):
            for c in range(cols):
                targets_grid.append((c, r, f"Tile ({c}, {r})"))
    else:  # quick (default): 4 corners + centre
        mid_c = cols // 2
        mid_r = rows // 2
        raw = [
            (0, 0, "Near-Left Corner"),
            (cols - 1, 0, "Near-Right Corner"),
            (mid_c, mid_r, "Center Tile"),
            (0, rows - 1, "Far-Left Corner"),
            (cols - 1, rows - 1, "Far-Right Corner"),
        ]
        seen = set()
        for c, r, name in raw:
            if (c, r) not in seen:
                seen.add((c, r))
                targets_grid.append((c, r, name))

    dx = w / cols
    dy = h / rows
    out = []
    for idx, (c, r, name) in enumerate(targets_grid):
        mx = (c + 0.5) * dx
        my = (r + 0.5) * dy
        out.append({
            "target_idx": idx,
            "col": c,
            "row": r,
            "mark_x": float(round(mx, 4)),
            "mark_y": float(round(my, 4)),
            "name": name,
        })
    return out


# ==============================================================================
# Pure Functions: Statistics & Modeling
# ==============================================================================

def target_stats(
    target: dict,
    samples: list[dict],
    w: float,
    h: float,
    cols: int,
    rows: int,
) -> dict:
    """Compute per-target accuracy, error vectors, and jitter statistics."""
    mark_x = target["mark_x"]
    mark_y = target["mark_y"]
    t_col = target["col"]
    t_row = target["row"]

    n_samples = len(samples)
    if n_samples == 0:
        return {
            "target_idx": target["target_idx"],
            "col": t_col,
            "row": t_row,
            "name": target["name"],
            "mark_x": mark_x,
            "mark_y": mark_y,
            "valid_frames": 0,
            "mean_centre_x": mark_x,
            "mean_centre_y": mark_y,
            "dx_cm": 0.0,
            "dy_cm": 0.0,
            "distance_cm": 0.0,
            "jitter_x_cm": 0.0,
            "jitter_y_cm": 0.0,
            "centre_hit_pct": 0.0,
            "left_hit_pct": 0.0,
            "right_hit_pct": 0.0,
            "errors": [],
        }

    c_xs = [s["centre_x"] for s in samples]
    c_ys = [s["centre_y"] for s in samples]

    mean_cx = float(np.mean(c_xs))
    mean_cy = float(np.mean(c_ys))
    dx_m = mean_cx - mark_x
    dy_m = mean_cy - mark_y
    dist_cm = float(math.hypot(dx_m, dy_m) * 100.0)

    jitter_x_cm = float(np.std(c_xs) * 100.0)
    jitter_y_cm = float(np.std(c_ys) * 100.0)

    centre_hits = sum(
        1 for s in samples
        if tile_of(s["centre_x"], s["centre_y"], w, h, cols, rows) == (t_col, t_row)
    )
    centre_hit_pct = float(centre_hits / n_samples * 100.0)

    left_hits = sum(
        1 for s in samples
        if s.get("left_x") is not None and s.get("left_y") is not None
        and tile_of(s["left_x"], s["left_y"], w, h, cols, rows) == (t_col, t_row)
    )
    left_hit_pct = float(left_hits / n_samples * 100.0)

    right_hits = sum(
        1 for s in samples
        if s.get("right_x") is not None and s.get("right_y") is not None
        and tile_of(s["right_x"], s["right_y"], w, h, cols, rows) == (t_col, t_row)
    )
    right_hit_pct = float(right_hits / n_samples * 100.0)

    errors = [(float(s["centre_x"] - mark_x), float(s["centre_y"] - mark_y)) for s in samples]

    return {
        "target_idx": target["target_idx"],
        "col": t_col,
        "row": t_row,
        "name": target["name"],
        "mark_x": mark_x,
        "mark_y": mark_y,
        "valid_frames": n_samples,
        "mean_centre_x": round(mean_cx, 4),
        "mean_centre_y": round(mean_cy, 4),
        "dx_cm": round(dx_m * 100.0, 2),
        "dy_cm": round(dy_m * 100.0, 2),
        "distance_cm": round(dist_cm, 2),
        "jitter_x_cm": round(jitter_x_cm, 2),
        "jitter_y_cm": round(jitter_y_cm, 2),
        "centre_hit_pct": round(centre_hit_pct, 1),
        "left_hit_pct": round(left_hit_pct, 1),
        "right_hit_pct": round(right_hit_pct, 1),
        "errors": errors,
    }


def p_correct_table(
    target_errors: list[list[tuple[float, float]]],
    w: float,
    h: float,
    n_range: range = range(2, 9),
) -> list[dict]:
    """Compute expected probability P(N) that a player standing at a random spot gets the correct tile.

    P(N) = mean over samples of max(0, 1 - |ex|/sx) * max(0, 1 - |ey|/sy)
    with each target weighted equally.
    """
    results = []
    active_targets = [errs for errs in target_errors if len(errs) > 0]
    if not active_targets:
        for N in n_range:
            sx = w / N
            sy = h / N
            results.append({
                "N": N,
                "sx_cm": round(sx * 100.0, 1),
                "sy_cm": round(sy * 100.0, 1),
                "P": 0.0,
            })
        return results

    for N in n_range:
        sx = w / N
        sy = h / N
        target_probs = []
        for errs in active_targets:
            p_samples = [
                max(0.0, 1.0 - abs(ex) / sx) * max(0.0, 1.0 - abs(ey) / sy)
                for ex, ey in errs
            ]
            target_probs.append(float(np.mean(p_samples)))
        overall_p = float(np.mean(target_probs))
        results.append({
            "N": N,
            "sx_cm": round(sx * 100.0, 1),
            "sy_cm": round(sy * 100.0, 1),
            "P": round(overall_p, 4),
        })
    return results


def compute_summary(per_target: list[dict], all_errors: list[tuple[float, float]]) -> dict:
    """Compute overall test summary metrics across all targets and frames."""
    if not all_errors:
        return {
            "total_frames": 0,
            "mean_error_cm": 0.0,
            "max_error_cm": 0.0,
            "p95_error_cm": 0.0,
            "overall_centre_hit_pct": 0.0,
        }

    dists_cm = [math.hypot(ex, ey) * 100.0 for ex, ey in all_errors]
    mean_err = float(np.mean(dists_cm))
    max_err = float(np.max(dists_cm))
    p95_err = float(np.percentile(dists_cm, 95))

    total_valid = sum(t["valid_frames"] for t in per_target)
    weighted_hit_sum = sum(t["centre_hit_pct"] * t["valid_frames"] for t in per_target)
    overall_hit_pct = float(weighted_hit_sum / total_valid) if total_valid > 0 else 0.0

    return {
        "total_frames": total_valid,
        "mean_error_cm": round(mean_err, 2),
        "max_error_cm": round(max_err, 2),
        "p95_error_cm": round(p95_err, 2),
        "overall_centre_hit_pct": round(overall_hit_pct, 1),
    }


# ==============================================================================
# Helper Extraction & Rendering Functions
# ==============================================================================

def extract_feet(pf: PoseFrame | None) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    if pf is None:
        return None, None
    return pf.left_foot_m, pf.right_foot_m


def compute_centre(pf: PoseFrame | None) -> tuple[float, float] | None:
    if pf is None:
        return None
    if pf.left_foot_m is not None and pf.right_foot_m is not None:
        return ((pf.left_foot_m[0] + pf.right_foot_m[0]) / 2.0,
                (pf.left_foot_m[1] + pf.right_foot_m[1]) / 2.0)
    if pf.left_foot_m is not None:
        return pf.left_foot_m
    if pf.right_foot_m is not None:
        return pf.right_foot_m
    if pf.floor_x_m is not None and pf.floor_y_m is not None:
        return (pf.floor_x_m, pf.floor_y_m)
    return None


def render_live(
    frame: np.ndarray | None,
    mapper: FloorMapper,
    pf: PoseFrame | None,
    cols: int,
    rows: int,
    fps: float = 0.0,
    target_active: dict | None = None,
    panel_title: str = "FLOOR IS LAVA",
) -> np.ndarray:
    """Render live video view with projected floor grid and a 300px side panel."""
    w_m, h_m = mapper.w_m, mapper.h_m

    if frame is not None:
        video = frame.copy()
        h_vid, w_vid = video.shape[:2]
    else:
        h_vid, w_vid = 480, 640
        video = np.full((h_vid, w_vid, 3), (30, 30, 35), dtype=np.uint8)

    overlay = video.copy()

    # 1. Tile highlights under feet
    left_m, right_m = extract_feet(pf)
    tile_l = tile_of(left_m[0], left_m[1], w_m, h_m, cols, rows) if left_m else None
    tile_r = tile_of(right_m[0], right_m[1], w_m, h_m, cols, rows) if right_m else None

    # Highlight active target tile in test mode if provided
    if target_active is not None:
        t_col, t_row = target_active["col"], target_active["row"]
        corners_m = tile_rect_corners(t_col, t_row, w_m, h_m, cols, rows)
        corners_px = np.array([mapper.to_pixels(x, y) for x, y in corners_m], dtype=np.int32)
        cv2.fillPoly(overlay, [corners_px], (0, 100, 0))

    if tile_l is not None and tile_r is not None and tile_l == tile_r:
        corners_m = tile_rect_corners(tile_l[0], tile_l[1], w_m, h_m, cols, rows)
        corners_px = np.array([mapper.to_pixels(x, y) for x, y in corners_m], dtype=np.int32)
        cv2.fillPoly(overlay, [corners_px], (200, 50, 200))  # Purple for both
    else:
        if tile_l is not None:
            corners_m = tile_rect_corners(tile_l[0], tile_l[1], w_m, h_m, cols, rows)
            corners_px = np.array([mapper.to_pixels(x, y) for x, y in corners_m], dtype=np.int32)
            cv2.fillPoly(overlay, [corners_px], (255, 120, 0))  # Blue (BGR) for left
        if tile_r is not None:
            corners_m = tile_rect_corners(tile_r[0], tile_r[1], w_m, h_m, cols, rows)
            corners_px = np.array([mapper.to_pixels(x, y) for x, y in corners_m], dtype=np.int32)
            cv2.fillPoly(overlay, [corners_px], (255, 0, 255))  # Magenta for right

    cv2.addWeighted(overlay, 0.40, video, 0.60, 0, video)

    # 2. Draw grid lines over floor
    dx = w_m / cols
    dy = h_m / rows
    grid_color = (255, 230, 0)  # Cyan in BGR

    for c in range(cols + 1):
        x = min(w_m, c * dx)
        p1 = mapper.to_pixels(x, 0.0)
        p2 = mapper.to_pixels(x, h_m)
        cv2.line(video, (int(round(p1[0])), int(round(p1[1]))),
                 (int(round(p2[0])), int(round(p2[1]))), grid_color, 1, cv2.LINE_AA)

    for r in range(rows + 1):
        y = min(h_m, r * dy)
        p1 = mapper.to_pixels(0.0, y)
        p2 = mapper.to_pixels(w_m, y)
        cv2.line(video, (int(round(p1[0])), int(round(p1[1]))),
                 (int(round(p2[0])), int(round(p2[1]))), grid_color, 1, cv2.LINE_AA)

    # 3. Target outline & crosshair in test mode
    if target_active is not None:
        t_col, t_row = target_active["col"], target_active["row"]
        corners_m = tile_rect_corners(t_col, t_row, w_m, h_m, cols, rows)
        corners_px = [mapper.to_pixels(x, y) for x, y in corners_m]
        for i in range(4):
            pt1 = (int(round(corners_px[i][0])), int(round(corners_px[i][1])))
            pt2 = (int(round(corners_px[(i + 1) % 4][0])), int(round(corners_px[(i + 1) % 4][1])))
            cv2.line(video, pt1, pt2, (0, 255, 0), 3, cv2.LINE_AA)

        mx, my = target_active["mark_x"], target_active["mark_y"]
        c_px = mapper.to_pixels(mx, my)
        cx_i, cy_i = int(round(c_px[0])), int(round(c_px[1]))
        cv2.circle(video, (cx_i, cy_i), 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.line(video, (cx_i - 14, cy_i), (cx_i + 14, cy_i), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(video, (cx_i, cy_i - 14), (cx_i, cy_i + 14), (0, 255, 0), 2, cv2.LINE_AA)

    # 4. Small dots for feet
    if left_m is not None:
        lx_px, ly_px = mapper.to_pixels(left_m[0], left_m[1])
        cv2.circle(video, (int(round(lx_px)), int(round(ly_px))), 6, (255, 120, 0), -1, cv2.LINE_AA)
        cv2.circle(video, (int(round(lx_px)), int(round(ly_px))), 8, (255, 255, 255), 1, cv2.LINE_AA)

    if right_m is not None:
        rx_px, ry_px = mapper.to_pixels(right_m[0], right_m[1])
        cv2.circle(video, (int(round(rx_px)), int(round(ry_px))), 6, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(video, (int(round(rx_px)), int(round(ry_px))), 8, (255, 255, 255), 1, cv2.LINE_AA)

    centre_m = compute_centre(pf)

    # 5. Build Side Panel (300px wide)
    panel_w = 300
    panel = np.full((h_vid, panel_w, 3), (22, 22, 26), dtype=np.uint8)
    cv2.line(panel, (0, 0), (0, h_vid), (60, 60, 70), 1)

    # Mini-map
    map_size = 200
    map_x0 = (panel_w - map_size) // 2
    map_y0 = 35
    map_x1 = map_x0 + map_size
    map_y1 = map_y0 + map_size

    cv2.rectangle(panel, (map_x0, map_y0), (map_x1, map_y1), (35, 35, 42), -1)

    # Mini-map tiles
    for c in range(cols):
        for r in range(rows):
            tx1 = map_x0 + int(c * (map_size / cols))
            tx2 = map_x0 + int((c + 1) * (map_size / cols))
            ty1 = map_y1 - int((r + 1) * (map_size / rows))
            ty2 = map_y1 - int(r * (map_size / rows))

            tile_col = (c, r)
            if target_active is not None and (t_col, t_row) == tile_col:
                cv2.rectangle(panel, (tx1, ty1), (tx2, ty2), (0, 100, 0), -1)

            if tile_l == tile_col and tile_r == tile_col:
                cv2.rectangle(panel, (tx1, ty1), (tx2, ty2), (180, 50, 180), -1)
            elif tile_l == tile_col:
                cv2.rectangle(panel, (tx1, ty1), (tx2, ty2), (200, 100, 0), -1)
            elif tile_r == tile_col:
                cv2.rectangle(panel, (tx1, ty1), (tx2, ty2), (180, 0, 180), -1)

            cv2.rectangle(panel, (tx1, ty1), (tx2, ty2), (65, 65, 75), 1)

    cv2.rectangle(panel, (map_x0, map_y0), (map_x1, map_y1), (140, 120, 0), 2)
    cv2.putText(panel, "FAR", (map_x0 + map_size // 2 - 12, map_y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 160), 1, cv2.LINE_AA)
    cv2.putText(panel, "NEAR", (map_x0 + map_size // 2 - 16, map_y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 160), 1, cv2.LINE_AA)

    # Mini-map foot dots
    if left_m is not None and 0.0 <= left_m[0] <= w_m and 0.0 <= left_m[1] <= h_m:
        lx = map_x0 + int((left_m[0] / w_m) * map_size)
        ly = map_y1 - int((left_m[1] / h_m) * map_size)
        cv2.circle(panel, (lx, ly), 4, (255, 120, 0), -1, cv2.LINE_AA)

    if right_m is not None and 0.0 <= right_m[0] <= w_m and 0.0 <= right_m[1] <= h_m:
        rx = map_x0 + int((right_m[0] / w_m) * map_size)
        ry = map_y1 - int((right_m[1] / h_m) * map_size)
        cv2.circle(panel, (rx, ry), 4, (255, 0, 255), -1, cv2.LINE_AA)

    # Mini-map target crosshair
    if target_active is not None:
        tmx = map_x0 + int((target_active["mark_x"] / w_m) * map_size)
        tmy = map_y1 - int((target_active["mark_y"] / h_m) * map_size)
        cv2.circle(panel, (tmx, tmy), 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.line(panel, (tmx - 6, tmy), (tmx + 6, tmy), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(panel, (tmx, tmy - 6), (tmx, tmy + 6), (0, 255, 0), 1, cv2.LINE_AA)

    # Side panel text
    cv2.putText(panel, panel_title, (map_x0, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 2, cv2.LINE_AA)

    ty = map_y1 + 36
    tracked = bool(pf.tracked) if pf is not None else False
    airborne = bool(pf.airborne) if pf is not None else False

    tr_col = (0, 255, 0) if tracked else (0, 0, 255)
    tr_txt = "YES" if tracked else "NO"
    cv2.putText(panel, f"Tracked: {tr_txt}", (18, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, tr_col, 1, cv2.LINE_AA)

    ty += 20
    ab_col = (0, 165, 255) if airborne else (180, 180, 180)
    ab_txt = "AIRBORNE" if airborne else "GROUNDED"
    cv2.putText(panel, f"State: {ab_txt}", (18, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, ab_col, 1, cv2.LINE_AA)

    ty += 20
    c_txt = f"({centre_m[0]:.2f}, {centre_m[1]:.2f}) m" if centre_m else "--"
    cv2.putText(panel, f"Centre: {c_txt}", (18, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    ty += 20
    if tile_l == tile_r and tile_l is not None:
        tile_str = f"L & R: ({tile_l[0]}, {tile_l[1]})"
    else:
        l_str = f"({tile_l[0]}, {tile_l[1]})" if tile_l else "--"
        r_str = f"({tile_r[0]}, {tile_r[1]})" if tile_r else "--"
        tile_str = f"L: {l_str} | R: {r_str}"
    cv2.putText(panel, f"Tiles: {tile_str}", (18, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    ty += 20
    tile_w_cm = (w_m / cols) * 100.0
    tile_h_cm = (h_m / rows) * 100.0
    cv2.putText(panel, f"Tile size: {tile_w_cm:.1f} x {tile_h_cm:.1f} cm", (18, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    ty += 20
    cv2.putText(panel, f"Pose FPS: {fps:.0f}", (18, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 255), 1, cv2.LINE_AA)

    ty += 30
    cv2.line(panel, (18, ty), (panel_w - 18, ty), (50, 50, 60), 1)
    ty += 18
    cv2.putText(panel, "[+/-]: Grid (Cols/Rows)", (18, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
    ty += 16
    cv2.putText(panel, "[T]: Start Accuracy Test", (18, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 128), 1, cv2.LINE_AA)
    ty += 16
    cv2.putText(panel, "[Q]: Quit", (18, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    # Combine video and side panel
    canvas = np.hstack((video, panel))
    return canvas


def render_results(
    summary: dict,
    per_target: list[dict],
    p_table: list[dict],
    note: str,
    w_m: float,
    h_m: float,
    cols: int,
    rows: int,
    size: tuple[int, int] = (940, 560),
) -> np.ndarray:
    """Render full-screen summary card with P(N) modeling table and error metrics."""
    W, H = size
    img = np.full((H, W, 3), (20, 20, 25), dtype=np.uint8)

    # Header
    cv2.putText(img, "TILE ACCURACY TEST REPORT", (25, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2, cv2.LINE_AA)
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    cond_txt = f"Floor: {w_m:.3f} x {h_m:.3f} m | Grid: {cols}x{rows} | {len(per_target)} marks | {date_str}"
    if note:
        cond_txt += f" | Note: {note}"
    cv2.putText(img, cond_txt, (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.line(img, (25, 75), (W - 25, 75), (60, 60, 70), 1)

    # Left Column: Summary metrics card & Target breakdown
    cv2.putText(img, "SUMMARY METRICS", (25, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 128), 1, cv2.LINE_AA)

    metrics = [
        f"Valid frames recorded:  {summary['total_frames']}",
        f"Mean Error:             {summary['mean_error_cm']:.2f} cm",
        f"Maximum Error:          {summary['max_error_cm']:.2f} cm",
        f"95th Percentile Error:  {summary['p95_error_cm']:.2f} cm",
        f"Centre Tile Hit Rate:   {summary['overall_centre_hit_pct']:.1f}%",
    ]
    my = 126
    for m in metrics:
        cv2.putText(img, m, (35, my), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        my += 22

    # Target Breakdown table
    my += 15
    cv2.putText(img, "TARGET BREAKDOWN", (25, my), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 128), 1, cv2.LINE_AA)
    my += 22
    th_hdr = f"{'#':<3} {'Target':<17} {'Mark (x, y) m':<16} {'Err (cm)':<9} {'Jitter':<12} {'Hit%'}"
    cv2.putText(img, th_hdr, (30, my), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 170), 1, cv2.LINE_AA)
    my += 16

    for t in per_target[:6]:  # show up to 6 targets
        mark_str = f"({t['mark_x']:.2f}, {t['mark_y']:.2f})"
        jit_str = f"{t['jitter_x_cm']:.1f}, {t['jitter_y_cm']:.1f}"
        t_line = f"{t['target_idx']:<3} {t['name'][:16]:<17} {mark_str:<16} {t['distance_cm']:<9.1f} {jit_str:<12} {t['centre_hit_pct']:.0f}%"
        cv2.putText(img, t_line, (30, my), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1, cv2.LINE_AA)
        my += 18

    # Right Column: P(N) Tile-Size Estimation Table
    rx0 = 530
    cv2.putText(img, "EXPECTED CORRECT-TILE PROBABILITY P(N)", (rx0, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 1, cv2.LINE_AA)

    py = 128
    p_hdr = f"{'N':<4} {'Tile Size (cm)':<18} {'P(Correct Tile)'}"
    cv2.putText(img, p_hdr, (rx0 + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 170), 1, cv2.LINE_AA)
    py += 18

    rec_n = None
    for row in p_table:
        n_val = row["N"]
        p_val = row["P"]
        if p_val >= 0.95:
            rec_n = n_val
        p_pct = f"{p_val * 100.0:.1f}%"
        size_str = f"{row['sx_cm']:.1f} x {row['sy_cm']:.1f}"
        p_line = f"{n_val:<4} {size_str:<18} {p_pct}"
        color = (0, 255, 128) if p_val >= 0.95 else (180, 180, 180)
        cv2.putText(img, p_line, (rx0 + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
        py += 22

    py += 10
    rec_str = f"Largest grid with P >= 95%: {rec_n} x {rec_n}" if rec_n else "No grid achieves P >= 95%"
    cv2.putText(img, rec_str, (rx0 + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 128), 2, cv2.LINE_AA)

    py += 22
    disclaim = "* Estimate from this test only (weighted equally across marks)."
    cv2.putText(img, disclaim, (rx0 + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 150), 1, cv2.LINE_AA)

    # Footer
    cv2.line(img, (25, H - 50), (W - 25, H - 50), (60, 60, 70), 1)
    foot_txt = "Press [SPACE], [ESC], or [Q] to return to live mode"
    cv2.putText(img, foot_txt, (25, H - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)

    return img


# ==============================================================================
# State Machine for Interactive & Headless Accuracy Testing
# ==============================================================================

class TileCheckStateMachine:
    """Manages transitions between live view, target countdown, recording, and results."""

    def __init__(
        self,
        w_m: float,
        h_m: float,
        cols: int = 3,
        rows: int = 3,
        targets_mode: str = "quick",
        note: str = "",
        recordings_dir: Path | str = "recordings",
    ):
        self.w_m = float(w_m)
        self.h_m = float(h_m)
        self.cols = int(cols)
        self.rows = int(rows)
        self.targets_mode = str(targets_mode)
        self.note = str(note)
        self.recordings_dir = Path(recordings_dir)

        self.state = "live"
        self.targets: list[dict] = []
        self.target_idx: int = 0
        self.timer_start: float = 0.0

        # Data collection
        self.target_samples: list[dict] = []
        self.all_samples: list[list[dict]] = []
        self.rejected_frames: int = 0

        # Final computed results
        self.per_target_stats: list[dict] = []
        self.summary_stats: dict = {}
        self.p_table: list[dict] = []
        self.saved_paths: dict[str, str] = {}
        self.results_image: np.ndarray | None = None

    def start_test(self, now: float) -> None:
        """Initialize accuracy test targets and transition to test instructions."""
        self.targets = make_targets(self.cols, self.rows, self.w_m, self.h_m, mode=self.targets_mode)
        self.target_idx = 0
        self.all_samples = [[] for _ in self.targets]
        self.target_samples = []
        self.rejected_frames = 0
        self.state = "test_instructions"

        print("\n--- STARTING TILE ACCURACY TEST ---")
        print("Real-world coordinates in metres from near-left corner (0,0):")
        print("  x: along near edge to the right")
        print("  y: away from camera along left edge\n")
        for t in self.targets:
            print(f"  Target {t['target_idx'] + 1}/{len(self.targets)}: {t['name']} at x = {t['mark_x']:.3f} m, y = {t['mark_y']:.3f} m")
        print("\nPlace tape marks at these positions. Press SPACE to begin Target 1.\n")

    def current_target(self) -> dict | None:
        if self.state.startswith("test_") and 0 <= self.target_idx < len(self.targets):
            return self.targets[self.target_idx]
        return None

    def step(self, pf: PoseFrame | None, now: float, key: int | None = None) -> str:
        """Step state machine given current PoseFrame, monotonic timestamp, and optional keycode."""
        if key == 27:  # ESC aborts to live
            self.state = "live"
            return self.state

        if self.state == "live":
            if key in (ord("t"), ord("T")):
                self.start_test(now)
            elif key in (ord("+"), ord("=")):
                self.cols = min(10, self.cols + 1)
                self.rows = min(10, self.rows + 1)
            elif key in (ord("-"), ord("_")):
                self.cols = max(2, self.cols - 1)
                self.rows = max(2, self.rows - 1)

        elif self.state == "test_instructions":
            if key == 32:  # SPACE
                self.state = "test_wait_space"

        elif self.state == "test_wait_space":
            if key == 32:  # SPACE starts countdown
                self.state = "test_countdown"
                self.timer_start = now
            elif key in (ord("n"), ord("N")):  # Skip target
                self.all_samples[self.target_idx] = []
                self._advance_target(now)
            elif key in (ord("r"), ord("R")):  # Redo previous target
                if self.target_idx > 0:
                    self.target_idx -= 1
                    self.all_samples[self.target_idx] = []
                    self.target_samples = []

        elif self.state == "test_countdown":
            if now - self.timer_start >= 1.0:
                self.state = "test_recording"
                self.timer_start = now
                self.target_samples = []
                self.rejected_frames = 0

        elif self.state == "test_recording":
            centre = compute_centre(pf)
            left_m, right_m = extract_feet(pf)

            is_valid = (
                pf is not None
                and bool(pf.tracked)
                and (not bool(pf.airborne))
                and centre is not None
            )

            if is_valid:
                self.target_samples.append({
                    "target_idx": self.target_idx,
                    "t": now - self.timer_start,
                    "centre_x": centre[0],
                    "centre_y": centre[1],
                    "left_x": left_m[0] if left_m else centre[0],
                    "left_y": left_m[1] if left_m else centre[1],
                    "right_x": right_m[0] if right_m else centre[0],
                    "right_y": right_m[1] if right_m else centre[1],
                })
            else:
                self.rejected_frames += 1

            if now - self.timer_start >= 1.5:
                self.all_samples[self.target_idx] = list(self.target_samples)
                if len(self.target_samples) < 15:
                    self.state = "test_target_done"
                else:
                    self._advance_target(now)

        elif self.state == "test_target_done":
            if key in (ord("r"), ord("R")):
                self.target_samples = []
                self.all_samples[self.target_idx] = []
                self.state = "test_wait_space"
            elif key in (32, ord("n"), ord("N")):
                self._advance_target(now)

        elif self.state == "results":
            if key in (32, 27, ord("q"), ord("Q")):
                self.state = "live"

        return self.state

    def _advance_target(self, now: float) -> None:
        self.target_idx += 1
        if self.target_idx < len(self.targets):
            self.state = "test_wait_space"
            self.target_samples = []
        else:
            self._finalize_and_save()

    def _finalize_and_save(self) -> None:
        """Compute final statistics, print results, render results image, and persist files."""
        self.state = "results"

        # Compute per-target statistics
        self.per_target_stats = [
            target_stats(target, samples, self.w_m, self.h_m, self.cols, self.rows)
            for target, samples in zip(self.targets, self.all_samples)
        ]

        all_errs = []
        for t_stat in self.per_target_stats:
            all_errs.extend(t_stat["errors"])

        self.summary_stats = compute_summary(self.per_target_stats, all_errs)
        target_error_lists = [t["errors"] for t in self.per_target_stats]
        self.p_table = p_correct_table(target_error_lists, self.w_m, self.h_m, n_range=range(2, 9))

        # Print formatted summary to console
        print("\n" + "=" * 65)
        print("               ACCURACY TEST RESULTS")
        print("=" * 65)
        print(f"Condition: {self.note if self.note else 'default'} | Marks: {len(self.targets)} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"Total valid frames: {self.summary_stats['total_frames']}")
        print(f"Mean distance error:    {self.summary_stats['mean_error_cm']:.2f} cm")
        print(f"Maximum distance error: {self.summary_stats['max_error_cm']:.2f} cm")
        print(f"95th percentile error:  {self.summary_stats['p95_error_cm']:.2f} cm")
        print(f"Overall hit rate:       {self.summary_stats['overall_centre_hit_pct']:.1f}%")
        print("\nExpected Probability P(N) that random foot gets correct tile:")
        print(f"{'N':<4} {'Tile Size (cm)':<18} {'P(Correct)'}")
        rec_n = None
        for row in self.p_table:
            if row["P"] >= 0.95:
                rec_n = row["N"]
            print(f"{row['N']:<4} {row['sx_cm']:.1f} x {row['sy_cm']:.1f} cm{'':<5} {row['P'] * 100.0:.1f}%")
        rec_msg = f"Largest grid with P >= 95%: {rec_n} x {rec_n}" if rec_n else "No grid achieves P >= 95%"
        print(f"\n{rec_msg}")
        print("* Note: Estimate from this test only (equal target weighting).\n")

        # Render results image
        self.results_image = render_results(
            self.summary_stats,
            self.per_target_stats,
            self.p_table,
            self.note,
            self.w_m,
            self.h_m,
            self.cols,
            self.rows,
        )

        # Save files to recordings/
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path = self.recordings_dir / f"tile_check_{ts_str}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["target_idx", "t", "centre_x", "centre_y", "left_x", "left_y", "right_x", "right_y"])
            for target_samples in self.all_samples:
                for s in target_samples:
                    writer.writerow([
                        s["target_idx"],
                        f"{s['t']:.3f}",
                        f"{s['centre_x']:.4f}",
                        f"{s['centre_y']:.4f}",
                        f"{s['left_x']:.4f}",
                        f"{s['left_y']:.4f}",
                        f"{s['right_x']:.4f}",
                        f"{s['right_y']:.4f}",
                    ])

        json_path = self.recordings_dir / f"tile_check_{ts_str}.json"
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "note": self.note,
            "cols": self.cols,
            "rows": self.rows,
            "w_m": self.w_m,
            "h_m": self.h_m,
            "targets_mode": self.targets_mode,
            "summary": self.summary_stats,
            "per_target": [
                {k: v for k, v in t.items() if k != "errors"}
                for t in self.per_target_stats
            ],
            "p_table": self.p_table,
            "recommended_n": rec_n,
        }
        json_path.write_text(json.dumps(json_data, indent=2))

        png_path = self.recordings_dir / f"tile_check_{ts_str}.png"
        cv2.imwrite(str(png_path), self.results_image)

        self.saved_paths = {
            "csv": str(csv_path),
            "json": str(json_path),
            "png": str(png_path),
        }
        print(f"Saved test recordings:")
        print(f"  CSV:  {csv_path}")
        print(f"  JSON: {json_path}")
        print(f"  PNG:  {png_path}")
        print("=" * 65 + "\n")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Tile accuracy check for Floor-is-Lava.")
    ap.add_argument("--cols", type=int, default=3, help="initial grid columns (default: 3)")
    ap.add_argument("--rows", type=int, default=3, help="initial grid rows (default: 3)")
    ap.add_argument("--camera", type=int, default=0, help="camera device index (default: 0)")
    ap.add_argument("--targets", choices=["quick", "corners", "all"], default="quick",
                    help="accuracy test targets (default: quick)")
    ap.add_argument("--note", type=str, default="", help="custom label/conditions note")
    args = ap.parse_args()

    mapper = FloorMapper.load()
    if mapper is None:
        print("Run: python -m tools.calibrate_floor --floor-w 1.245 --floor-h 1.245")
        sys.exit(1)

    if not Path("models/pose_landmarker_lite.task").exists():
        print("Run: python -m tools.get_models")
        sys.exit(1)

    source = None
    try:
        source = CameraPoseSource(args.camera, mapper=mapper)
        source.start()
    except FileNotFoundError:
        print("Run: python -m tools.get_models")
        sys.exit(1)
    except RuntimeError as e:
        sys.exit(str(e))
    atexit.register(source.stop)

    state_machine = TileCheckStateMachine(
        w_m=mapper.w_m,
        h_m=mapper.h_m,
        cols=args.cols,
        rows=args.rows,
        targets_mode=args.targets,
        note=args.note,
        recordings_dir=Path("recordings"),
    )

    print("Tile Check live mode running. Keys: +/- grid, T test, Q quit.")
    cv2.namedWindow("Floor-is-Lava Tile Check", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            now = time.monotonic()
            pf, frame, debug = source.snapshot()

            # Process state machine step with keycode from previous frame
            # (cv2.waitKey called below)
            st = state_machine.state

            if st == "results" and state_machine.results_image is not None:
                display_img = state_machine.results_image
            else:
                curr_target = state_machine.current_target()
                title = "FLOOR IS LAVA"
                if st == "test_instructions":
                    title = "TEST INSTRUCTIONS"
                elif st == "test_wait_space":
                    title = f"TARGET {state_machine.target_idx + 1}/{len(state_machine.targets)}"
                elif st == "test_countdown":
                    rem = max(0.0, 1.0 - (now - state_machine.timer_start))
                    title = f"GET READY: {rem:.1f}s"
                elif st == "test_recording":
                    rem = max(0.0, 1.5 - (now - state_machine.timer_start))
                    title = f"RECORDING: {rem:.1f}s"
                elif st == "test_target_done":
                    title = "LOW VALID FRAMES"

                display_img = render_live(
                    frame=frame,
                    mapper=mapper,
                    pf=pf,
                    cols=state_machine.cols,
                    rows=state_machine.rows,
                    fps=getattr(source, "pose_fps", 0.0),
                    target_active=curr_target,
                    panel_title=title,
                )

                # Overlay banner text on video in test modes
                if st == "test_instructions":
                    cv2.rectangle(display_img, (15, 15), (620, 65), (20, 20, 25), -1)
                    cv2.putText(display_img, "Stand on tape mark. Press SPACE to begin.", (25, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2, cv2.LINE_AA)
                elif st == "test_wait_space" and curr_target is not None:
                    cv2.rectangle(display_img, (15, 15), (620, 70), (20, 20, 25), -1)
                    cv2.putText(display_img, f"Target {curr_target['target_idx'] + 1}/{len(state_machine.targets)}: {curr_target['name']}",
                                (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 2, cv2.LINE_AA)
                    cv2.putText(display_img, f"Stand at mark ({curr_target['mark_x']:.2f}, {curr_target['mark_y']:.2f}) m. SPACE when ready.",
                                (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                elif st == "test_countdown":
                    rem = max(0.0, 1.0 - (now - state_machine.timer_start))
                    cv2.rectangle(display_img, (15, 15), (620, 65), (20, 20, 25), -1)
                    cv2.putText(display_img, f"GET READY... {rem:.1f}s", (25, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2, cv2.LINE_AA)
                elif st == "test_recording":
                    rem = max(0.0, 1.5 - (now - state_machine.timer_start))
                    cv2.rectangle(display_img, (15, 15), (620, 65), (20, 20, 25), -1)
                    cv2.putText(display_img, f"RECORDING: {rem:.1f}s  (valid: {len(state_machine.target_samples)})",
                                (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                elif st == "test_target_done":
                    cv2.rectangle(display_img, (15, 15), (620, 70), (20, 20, 25), -1)
                    cv2.putText(display_img, f"Only {len(state_machine.target_samples)} valid frames (<15).",
                                (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(display_img, "[R] Redo target | [SPACE] or [N] Continue",
                                (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Floor-is-Lava Tile Check", display_img)
            k = cv2.waitKey(20) & 0xFF

            if k == ord("q") and state_machine.state == "live":
                break

            state_machine.step(pf, now, key=k)

    finally:
        if source is not None:
            source.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
