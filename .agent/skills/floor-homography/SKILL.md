---
name: floor-homography
description: Use when mapping the player's feet from camera pixels to floor coordinates and tiles (pose_core/floor_map.py, tools/calibrate_floor.py, tile occupancy).
---
# Floor homography rules
- Calibration: 4 tape markers on the floor at the corners of the play area. Real distances come
  from config (grid cols/rows x tile_m). Click them in a fixed order, compute H with
  `cv2.getPerspectiveTransform` or `cv2.findHomography`, save to `calibration.local.json` (git-ignored).
- Apply H only to points ON the floor: heels and foot-index points. Never head, hands or hips.
- The camera must not move after calibration. If it is bumped, recalibrate.
- Camera placement matters more than code: mount high (about 1.5-2 m) and tilt down, so far tiles
  are not at a grazing angle. Verify with the accuracy test below.
- Jumping breaks the plane assumption: when both ankles rise fast, set `airborne=True` and hold
  the last floor position until landing.
- Footprint: a circle (radius from config) around the feet midpoint. A tile is occupied if the circle
  overlaps it. Judge at impact time, with a forgiveness margin (ignore overlaps under ~20-30%),
  so a few cm of jitter never causes an unfair hit.
- Accuracy test: stand on each tile centre; computed position should be within about 15 cm.
  If not, fix camera height/angle first, then fall back to a single row (3 lanes).
- Always provide a debug overlay: tile grid on the live feed + a top-down view with the player dot.
