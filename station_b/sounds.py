"""Station B: Retro 8-Bit Chiptune Sound Manager & Pose Announcer.
Synthesizes lightweight, in-memory retro arcade audio waveforms using NumPy & Pygame,
and manages non-blocking voice announcer callouts via Windows SAPI.
Falls back to silent operation in headless test environments.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pygame

from station_a.sounds import ArcadeSoundManager


class BlazerSoundManager(ArcadeSoundManager):
    """Arcade sound generator and announcer tailored for Station B (Pose Blazer)."""

    def __init__(
        self,
        sample_rate: int = 44100,
        custom_sounds_dir: str | Path | None = None,
        enable_voice: bool = True,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            custom_sounds_dir=custom_sounds_dir,
            enable_voice=enable_voice,
        )
        if self.enabled:
            self._load_or_synthesize_blazer_sounds()

    def _load_or_synthesize_blazer_sounds(self) -> None:
        """Synthesize specialized sound effects for pose snapping, perfection, and skipping."""
        try:
            # 1. Pose Snap: Crisp double chime (987 Hz -> 1318 Hz)
            self.sounds["pose_snap"] = self._get_sound(
                "pose_snap",
                [(987.77, 0.05, 0.35), (1318.51, 0.12, 0.40)],
                duty=0.5,
            )

            # 2. Perfection Fanfare: Sparkling pentatonic arpeggio (523 -> 659 -> 783 -> 1046 -> 1318 Hz)
            self.sounds["perfection"] = self._get_sound(
                "perfection",
                [
                    (523.25, 0.045, 0.35),
                    (659.25, 0.045, 0.35),
                    (783.99, 0.045, 0.35),
                    (1046.50, 0.06, 0.40),
                    (1318.51, 0.22, 0.45),
                ],
                duty=0.45,
                vib_last=True,
            )

            # 3. Pose Skip: Flat descending buzz (180 Hz -> 130 Hz)
            self.sounds["pose_skip"] = self._get_sound(
                "pose_skip",
                [(180.0, 0.09, 0.30), (130.0, 0.16, 0.30)],
                duty=0.25,
            )

            # 4. Countdown Tick: Short high-contrast click (880 Hz, 30 ms)
            self.sounds["tick"] = self._get_sound(
                "tick",
                [(880.0, 0.03, 0.28)],
                duty=0.5,
            )

            # 5. Disqualified / Eliminated: Descending mournful tone (330 Hz -> 165 Hz)
            self.sounds["disqualified"] = self._get_sound(
                "disqualified",
                [(329.63, 0.14, 0.35), (220.00, 0.16, 0.35), (164.81, 0.35, 0.35)],
                duty=0.3,
                vib_last=True,
            )
        except Exception:
            pass

    def play_snap(self) -> None:
        """Play crisp chime when a pose is successfully held and locked."""
        self.play("pose_snap")

    def play_perfection(self) -> None:
        """Play triumphant fanfare when a player scores >= 85% perfection."""
        self.play("perfection")
        self.speak("Perfection!")

    def play_skip(self) -> None:
        """Play buzz when player triggers the folded-arms skip gesture."""
        self.play("pose_skip")
        self.speak("Skipped!")

    def play_tick(self) -> None:
        """Play sharp countdown tick during the final 3 seconds of a round."""
        self.play("tick")

    def play_disqualified(self) -> None:
        """Play elimination buzzer when player triggers 4 consecutive skips."""
        self.play("disqualified")
        self.speak("Disqualified!")

    def play_stage_start(self, stage_num: int) -> None:
        """Announce transition to Stage 1 (Precision Blitz) or Stage 2 (Speed Rush)."""
        if stage_num == 1:
            self.play("round_1")
            self.speak("Stage 1! Precision Blitz!")
        else:
            self.play("round_2")
            self.speak("Stage 2! Speed Rush!")
