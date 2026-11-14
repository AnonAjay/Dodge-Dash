"""Camera + pose inference in ONE background thread. The game only calls latest().

HINT (guess before you code): the game loop runs at 60 FPS and pose at ~30 FPS.
What single piece of shared data lets the reader ALWAYS get an answer instantly,
without waiting for the camera?  (see .agent/skills/pose-tracking/SKILL.md)

Implements shared.contracts.PoseProvider.
"""
from __future__ import annotations

from shared.contracts import PoseFrame


class CameraPoseSource:
    def __init__(self, camera_index: int = 0,
                 model_path: str = "models/pose_landmarker_lite.task"):
        self.camera_index = camera_index
        self.model_path = model_path

    def start(self) -> None:
        raise NotImplementedError("TODO: start the capture+inference thread")

    def latest(self) -> PoseFrame:
        raise NotImplementedError("TODO: return the newest PoseFrame (never block)")

    def stop(self) -> None:
        raise NotImplementedError("TODO: stop thread, release camera")
