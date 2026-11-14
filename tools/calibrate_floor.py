"""Floor calibration. Run once per venue, again if the camera moves.
TODO: show the camera, click the 4 floor markers in order (near-left, near-right, far-right, far-left),
      compute the homography with the real marker distances from config, save calibration.local.json,
      then draw the tile grid over the live feed so everyone can check it looks right.
See .agent/skills/floor-homography/SKILL.md
"""
