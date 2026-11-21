"""Tests for pose_core ankle tracking and velocity-gated anchor locking."""
import pytest
import numpy as np
from types import SimpleNamespace

from pose_core.landmarks import (
    feet_info,
    foot_points,
    L_ANKLE,
    R_ANKLE,
    L_HEEL,
    R_HEEL,
    L_TOE,
    R_TOE,
    L_SHOULDER,
    R_SHOULDER,
    L_HIP,
    R_HIP,
)
from pose_core.capture import PoseTracker
from pose_core.floor_map import FloorMapper


def _make_dummy_landmarks(visible_ids: dict[int, tuple[float, float, float]]):
    """Create a 33-element list of landmarks with specific x, y, visibility."""
    lms = []
    for i in range(33):
        if i in visible_ids:
            x, y, v = visible_ids[i]
            lms.append(SimpleNamespace(x=x, y=y, visibility=v))
        else:
            lms.append(SimpleNamespace(x=0.0, y=0.0, visibility=0.0))
    return lms


def test_feet_info_prioritizes_ankles_when_heels_and_toes_occluded():
    """On budget webcams, heels/toes often drop below visibility. Ankles must lock tracking."""
    # Ankles visible with high confidence, heels/toes occluded (0.1 visibility)
    lms = _make_dummy_landmarks({
        L_ANKLE: (0.45, 0.85, 0.95),
        R_ANKLE: (0.55, 0.85, 0.92),
        L_HEEL: (0.44, 0.88, 0.10),
        R_HEEL: (0.54, 0.88, 0.15),
        L_TOE: (0.46, 0.92, 0.05),
        R_TOE: (0.56, 0.92, 0.08),
        L_SHOULDER: (0.46, 0.20, 0.90),
        R_SHOULDER: (0.54, 0.20, 0.90),
        L_HIP: (0.47, 0.50, 0.90),
        R_HIP: (0.53, 0.50, 0.90),
    })

    info = feet_info(lms, w=640, h=480)
    assert info is not None
    # Centre should be mean of ankles: 0.50 * 640 = 320, 0.85 * 480 = 408
    assert abs(info["centre_px"][0] - 320.0) < 1.0
    assert abs(info["centre_px"][1] - 408.0) < 1.0
    # Left and right foot points should match the ankles directly
    assert abs(info["left_px"][0] - 0.45 * 640) < 1.0
    assert abs(info["right_px"][0] - 0.55 * 640) < 1.0


def test_foot_points_uses_ankles_as_primary():
    lms = _make_dummy_landmarks({
        L_ANKLE: (0.45, 0.85, 0.95),
        R_ANKLE: (0.55, 0.85, 0.92),
        L_HEEL: (0.44, 0.88, 0.70),
        R_HEEL: (0.54, 0.88, 0.70),
    })
    pts = foot_points(lms, w=640, h=480)
    assert len(pts) == 2  # Ankles are returned
    assert abs(pts[0][0] - 0.45 * 640) < 1.0


class DummyMapper:
    """Mock FloorMapper converting pixel x,y directly to floor meters (scale = 0.005)."""
    w_m = 2.0
    h_m = 2.0

    def to_floor(self, px, py):
        return (px * 0.005, py * 0.005)

    def inside(self, fx, fy):
        return 0.0 <= fx <= 2.0 and 0.0 <= fy <= 2.0


def test_velocity_gated_anchor_lock_eliminates_stationary_jitter():
    tracker = PoseTracker(DummyMapper())

    # Frame 1: Person stands at center (px=200, py=300 -> floor = (1.00m, 1.50m))
    lms1 = _make_dummy_landmarks({
        L_ANKLE: (200 / 640, 300 / 480, 0.95),
        R_ANKLE: (200 / 640, 300 / 480, 0.95),
        L_SHOULDER: (0.3, 0.2, 0.9), R_SHOULDER: (0.4, 0.2, 0.9),
        L_HIP: (0.3, 0.4, 0.9), R_HIP: (0.4, 0.4, 0.9),
    })
    pf1 = tracker.update([lms1], 640, 480, ts=1.0)
    assert pf1.tracked is True
    initial_x = pf1.floor_x_m
    initial_y = pf1.floor_y_m

    # Frame 2: Webcam noise shakes landmarks by +- 4px (~2cm floor drift)
    lms2 = _make_dummy_landmarks({
        L_ANKLE: (204 / 640, 303 / 480, 0.95),
        R_ANKLE: (204 / 640, 303 / 480, 0.95),
        L_SHOULDER: (0.3, 0.2, 0.9), R_SHOULDER: (0.4, 0.2, 0.9),
        L_HIP: (0.3, 0.4, 0.9), R_HIP: (0.4, 0.4, 0.9),
    })
    pf2 = tracker.update([lms2], 640, 480, ts=1.033)

    # Anchor lock must hold the exact same coordinates (0.00 cm drift)
    assert pf2.floor_x_m == initial_x
    assert pf2.floor_y_m == initial_y


def test_anchor_lock_breaks_out_instantly_on_intentional_step():
    tracker = PoseTracker(DummyMapper())

    lms1 = _make_dummy_landmarks({
        L_ANKLE: (200 / 640, 300 / 480, 0.95),
        R_ANKLE: (200 / 640, 300 / 480, 0.95),
        L_SHOULDER: (0.3, 0.2, 0.9), R_SHOULDER: (0.4, 0.2, 0.9),
        L_HIP: (0.3, 0.4, 0.9), R_HIP: (0.4, 0.4, 0.9),
    })
    pf1 = tracker.update([lms1], 640, 480, ts=1.0)
    initial_x = pf1.floor_x_m

    # Player takes a definite step to the right (+60 px = +0.30m displacement)
    lms2 = _make_dummy_landmarks({
        L_ANKLE: (260 / 640, 300 / 480, 0.95),
        R_ANKLE: (260 / 640, 300 / 480, 0.95),
        L_SHOULDER: (0.4, 0.2, 0.9), R_SHOULDER: (0.5, 0.2, 0.9),
        L_HIP: (0.4, 0.4, 0.9), R_HIP: (0.5, 0.4, 0.9),
    })
    pf2 = tracker.update([lms2], 640, 480, ts=1.033)

    # The anchor must break immediately! Position must update
    assert pf2.floor_x_m > initial_x + 0.10
