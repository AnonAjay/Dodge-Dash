"""Pure helpers on landmark lists (no camera needed, easy to unit test).

TODO: foot_points(landmarks) -> list of (x_px, y_px) from heel + foot-index of both feet
      arms_up(landmarks) -> bool   (both wrists above the shoulders)
      is_airborne(prev, curr) -> bool   (ankles rising fast = jumping)
Landmark indices: ankles 27/28, heels 29/30, foot index 31/32. Check visibility.
"""
