---
name: pose-tracking
description: Use when working in pose_core/ on camera capture, MediaPipe pose inference, smoothing, or producing PoseFrame values.
---
# Pose tracking rules
- Use the MediaPipe **Tasks API**: `vision.PoseLandmarker`, `RunningMode.VIDEO`. `mp.solutions` does not exist.
- Model files live in `models/` and are downloaded by `python -m tools.get_models`. Never commit them.
- Convert BGR -> RGB before `mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)`.
- `detect_for_video` timestamps (ms) must strictly increase. Use `time.monotonic()`.
- Camera + inference run in ONE daemon thread. Publish the newest PoseFrame under a lock.
  `latest()` returns immediately. It never waits for the camera.
- Request MJPG, 640x480, to keep camera FPS up. Log ms per stage so slowdowns are visible.
- Smoothing: exponential moving average on floor coordinates; median over ~5 frames for feet.
- Feet landmarks: ankles 27/28, heels 29/30, foot index 31/32. If visibility is low, set `tracked=False`.
- Decide mirroring ONCE, here in pose_core, and document it. Moving right must increase floor_x.
- Target 25-30 FPS tracking. If slower, lower resolution or the lite model before anything else.
