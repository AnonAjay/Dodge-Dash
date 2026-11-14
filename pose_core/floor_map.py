"""Camera pixels -> floor metres using a homography. See floor-homography skill.

HINT: a homography maps points on ONE flat plane. Which body points lie on the floor plane?

TODO:
  compute_homography(image_pts, floor_pts_m) -> 3x3 matrix   (cv2.getPerspectiveTransform / findHomography)
  to_floor(H, points_px) -> points in metres                  (cv2.perspectiveTransform)
  save_calibration(H, path) / load_calibration(path)          (JSON, file is git-ignored)
"""
