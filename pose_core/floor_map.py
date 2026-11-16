"""Camera pixels -> floor metres using a homography. See floor-homography skill.

Conventions (same as shared/contracts.py):
  (0, 0) = NEAR-LEFT corner of the floor box, x grows to the right,
  y grows AWAY from the camera.
  Corners are always given in this order: near-left, near-right, far-right, far-left.
A homography is only valid for points ON the floor (feet), not for a foot in the air.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CALIB_PATH = Path("calibration.local.json")      # git-ignored
CORNER_NAMES = ("near-left", "near-right", "far-right", "far-left")
BOX_MARGIN_M = 0.35    # how far outside the box still counts as "in the box"


def floor_corners(w_m: float, h_m: float) -> np.ndarray:
    """Real-world corner coordinates matching CORNER_NAMES."""
    return np.array([[0, 0], [w_m, 0], [w_m, h_m], [0, h_m]], dtype=np.float32)


def compute_homography(image_pts, floor_pts_m) -> np.ndarray:
    """3x3 matrix mapping image pixels to floor metres (4 or more point pairs)."""
    H, _ = cv2.findHomography(np.asarray(image_pts, np.float32),
                              np.asarray(floor_pts_m, np.float32))
    if H is None:
        raise ValueError("Could not compute a homography (are 3 corners in a line?)")
    return H


def to_floor(H, points_px) -> np.ndarray:
    """Pixels (N,2) -> metres (N,2)."""
    pts = np.asarray(points_px, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def save_calibration(H, path: Path = CALIB_PATH, corners_px=None, w_m=None, h_m=None):
    data = {
        "H": np.asarray(H).tolist(),
        "corners_px": None if corners_px is None else np.asarray(corners_px).tolist(),
        "w_m": w_m,
        "h_m": h_m,
        "order": list(CORNER_NAMES),
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_calibration(path: Path = CALIB_PATH):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


class FloorMapper:
    """The calibration, ready to use."""

    def __init__(self, corners_px, w_m: float, h_m: float):
        self.corners_px = np.array(corners_px, dtype=np.float32)
        self.w_m, self.h_m = float(w_m), float(h_m)
        self.H = compute_homography(self.corners_px, floor_corners(w_m, h_m))
        self.H_inv = np.linalg.inv(self.H)

    def to_floor(self, px: float, py: float):
        x, y = to_floor(self.H, [[px, py]])[0]
        return float(x), float(y)

    def to_pixels(self, x_m: float, y_m: float):
        """Floor metres -> image pixels (used to draw the grid over the live feed)."""
        px, py = to_floor(self.H_inv, [[x_m, y_m]])[0]
        return float(px), float(py)

    def inside(self, x: float, y: float, margin: float = BOX_MARGIN_M) -> bool:
        return (-margin <= x <= self.w_m + margin) and (-margin <= y <= self.h_m + margin)

    def save(self, path: Path = CALIB_PATH):
        save_calibration(self.H, path, self.corners_px, self.w_m, self.h_m)

    @classmethod
    def load(cls, path: Path = CALIB_PATH):
        d = load_calibration(path)
        if not d or not d.get("corners_px"):
            return None
        return cls(d["corners_px"], d["w_m"], d["h_m"])


def setup_warnings(clicks, img_w: int, img_h: int):
    """Sanity checks on the clicked box, so a bad camera setup is caught early."""
    pts = np.array(clicks, dtype=np.float32)  # near-left, near-right, far-right, far-left
    warns = []
    if (pts[:, 0].min() < 0.05 * img_w or pts[:, 0].max() > 0.95 * img_w
            or pts[:, 1].min() < 0.05 * img_h or pts[:, 1].max() > 0.95 * img_h):
        warns.append("Corner too close to the image edge: feet will leave the view.")
    if (pts[:, 1].max() - pts[:, 1].min()) < 0.20 * img_h:
        warns.append("Floor looks squashed: raise the camera / tilt it down.")
    near_w, far_w = abs(pts[1][0] - pts[0][0]), abs(pts[2][0] - pts[3][0])
    if near_w > 0 and far_w / near_w < 0.5:
        warns.append("Very strong perspective: camera too low, depth will be inaccurate.")
    return warns
