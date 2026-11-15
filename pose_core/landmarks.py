"""Pure helpers on landmark lists (no camera needed, easy to unit test).

A landmark list is MediaPipe's 33 points. Each has .x, .y (fractions of the image, 0..1)
and .visibility (0..1). y grows DOWNWARD in the image, so "higher" means a smaller y.
Indices: shoulders 11/12, wrists 15/16, hips 23/24, ankles 27/28, heels 29/30, toes 31/32.
"""
from __future__ import annotations

import math

import numpy as np

# ---- tunables ----------------------------------------------------------------
MIN_VIS = 0.5          # ignore landmarks the model is unsure about
JUMP_LIFT = 0.20       # feet must rise this fraction of torso length...
JUMP_FRAMES = 2        # ...for this many frames in a row
GROUND_ALPHA = 0.1     # how fast the "standing height" baseline follows you
HIP_RISE_RATIO = 0.6   # a real jump lifts the HIPS at least this fraction as much as the feet.
                       # Walking away from the camera lifts the feet in the image but barely the hips.
                       # Set to 0 to switch this check off.

L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32


def _vis_pts(lms, ids, w, h):
    """[(x_px, y_px, visibility)] for the listed landmarks that are visible enough."""
    out = []
    for i in ids:
        lm = lms[i]
        if lm.visibility >= MIN_VIS:
            out.append((lm.x * w, lm.y * h, lm.visibility))
    return out


def _mean_xy(pts):
    return (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))


def foot_points(lms, w, h):
    """Pixel points of the feet: heels + toes of both feet (ankles if those are missing)."""
    pts = _vis_pts(lms, (L_HEEL, R_HEEL, L_TOE, R_TOE), w, h)
    if len(pts) < 2:
        pts = _vis_pts(lms, (L_ANKLE, R_ANKLE), w, h)
    return [(x, y) for x, y, _ in pts]


def torso_length_px(lms, w, h):
    """Shoulder-midpoint to hip-midpoint distance in pixels, or None if not all 4 are visible."""
    ids = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if any(lms[i].visibility < MIN_VIS for i in ids):
        return None
    sx = (lms[L_SHOULDER].x + lms[R_SHOULDER].x) / 2 * w
    sy = (lms[L_SHOULDER].y + lms[R_SHOULDER].y) / 2 * h
    hx = (lms[L_HIP].x + lms[R_HIP].x) / 2 * w
    hy = (lms[L_HIP].y + lms[R_HIP].y) / 2 * h
    return math.hypot(sx - hx, sy - hy)


def arms_up(lms) -> bool:
    """Both wrists above their shoulders (needs all four visible)."""
    ids = (L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST)
    if any(lms[i].visibility < MIN_VIS for i in ids):
        return False
    return lms[L_WRIST].y < lms[L_SHOULDER].y and lms[R_WRIST].y < lms[R_SHOULDER].y


def feet_info(lms, w, h):
    """Everything the tracker needs about one person, or None if no feet are visible."""
    pts = _vis_pts(lms, (L_HEEL, R_HEEL, L_TOE, R_TOE), w, h)
    if len(pts) < 2:  # fall back to ankles
        pts = _vis_pts(lms, (L_ANKLE, R_ANKLE), w, h)
    if not pts:
        return None
    left = _vis_pts(lms, (L_ANKLE, L_HEEL, L_TOE), w, h)
    right = _vis_pts(lms, (R_ANKLE, R_HEEL, R_TOE), w, h)
    return {
        "centre_px": _mean_xy(pts),
        "left_px": _mean_xy(left) if left else None,
        "right_px": _mean_xy(right) if right else None,
        "conf": float(np.mean([p[2] for p in pts])),
        "yl": float(np.mean([p[1] for p in left])) if left else None,
        "yr": float(np.mean([p[1] for p in right])) if right else None,
        "torso": torso_length_px(lms, w, h),
        "y_hip": (lms[L_HIP].y + lms[R_HIP].y) / 2 * h if min(lms[L_HIP].visibility, lms[R_HIP].visibility) >= MIN_VIS else None,
        "pts": [(p[0], p[1]) for p in pts],
        "arms_up": arms_up(lms),
    }


class JumpDetector:
    """Both feet clearly above their standing height AND the hips rising with them,
    for JUMP_FRAMES frames in a row.

    lift = (standing_y - y) / torso_length. Dividing by torso length makes the threshold work
    at any distance and for any height. The hip check stops walking AWAY from the camera
    (feet move up in the image) from being mistaken for a jump.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.ground_l = None
        self.ground_r = None
        self.ground_h = None
        self.frames = 0

    @property
    def rising(self) -> bool:
        """True from the first frame that looks like a jump (before it is confirmed)."""
        return self.frames > 0

    def update(self, yl, yr, torso, y_hip=None):
        """Returns (airborne, lift). Resets itself if the needed points are missing."""
        if yl is None or yr is None or not torso:
            self.reset()
            return False, 0.0
        if self.ground_l is None:
            self.ground_l, self.ground_r = yl, yr
        if self.ground_h is None and y_hip is not None:
            self.ground_h = y_hip
        lift_l = (self.ground_l - yl) / torso   # positive = foot is higher
        lift_r = (self.ground_r - yr) / torso
        feet_up = min(lift_l, lift_r)
        up = feet_up > JUMP_LIFT
        if up and y_hip is not None and HIP_RISE_RATIO > 0:
            hip_lift = (self.ground_h - y_hip) / torso
            up = hip_lift >= HIP_RISE_RATIO * feet_up
        self.frames = self.frames + 1 if up else 0
        airborne = self.frames >= JUMP_FRAMES
        if not up:  # follow slow changes (walking closer / further)
            self.ground_l += GROUND_ALPHA * (yl - self.ground_l)
            self.ground_r += GROUND_ALPHA * (yr - self.ground_r)
            if y_hip is not None and self.ground_h is not None:
                self.ground_h += GROUND_ALPHA * (y_hip - self.ground_h)
        return airborne, feet_up
