"""Station A: Retro 8-Bit Chiptune Arcade Sound Manager & Offline Voice Announcer.
Provides lightweight, offline sound effects synthesized in memory using NumPy and Pygame,
plus asynchronous, non-blocking spoken arcade callouts using Windows speech synthesis.
Safely falls back to silent operation in headless environments or systems without audio devices.
Checks assets/sounds/*.wav first for custom audio files before synthesizing.
"""
from __future__ import annotations

import os
from pathlib import Path
import platform
import queue
import subprocess
import sys
import threading
import numpy as np
import pygame


class ArcadeSoundManager:
    """Lightweight chiptune audio generator, sound player, and offline voice announcer for Titans' Arcade."""

    def __init__(
        self,
        sample_rate: int = 44100,
        custom_sounds_dir: str | Path | None = None,
        enable_voice: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.enabled = False
        self.enable_voice = enable_voice
        self.sounds: dict[str, pygame.mixer.Sound] = {}

        if custom_sounds_dir is None:
            # Default to repo root / assets / sounds
            repo_root = Path(__file__).resolve().parent.parent
            self.custom_dir = repo_root / "assets" / "sounds"
        else:
            self.custom_dir = Path(custom_sounds_dir)

        # 1. Initialize Pygame Mixer safely
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=2, buffer=512)
            self.enabled = True
        except Exception:
            self.enabled = False

        # 2. Pre-generate or load all sound effects
        if self.enabled:
            self._load_or_synthesize_all()

        # 3. Non-blocking voice announcer queue & background worker
        self._voice_queue: queue.Queue[str | None] = queue.Queue(maxsize=8)
        self._voice_thread: threading.Thread | None = None
        self._sapi_voice = None

        if self.enable_voice:
            self._init_voice_system()

    def _init_voice_system(self) -> None:
        """Initialize speech synthesizer and start non-blocking worker thread."""
        try:
            import win32com.client
            self._sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
        except Exception:
            self._sapi_voice = None

        try:
            self._voice_thread = threading.Thread(
                target=self._voice_worker, daemon=True, name="ArcadeVoiceWorker"
            )
            self._voice_thread.start()
        except Exception:
            self._voice_thread = None

    def _voice_worker(self) -> None:
        """Background thread worker to speak phrases sequentially without blocking the 60 FPS Pygame loop."""
        while True:
            try:
                phrase = self._voice_queue.get()
                if phrase is None:
                    break
                self._speak_sync(phrase)
                self._voice_queue.task_done()
            except Exception:
                pass

    def _speak_sync(self, phrase: str) -> None:
        """Execute speech synchronously inside the worker thread."""
        if not self.enable_voice:
            return

        clean_phrase = "".join(c for c in phrase if c.isalnum() or c in " !?.,'-").strip()
        if not clean_phrase:
            return

        # 1. Try Windows SAPI via COM dispatch if available
        if self._sapi_voice is not None:
            try:
                self._sapi_voice.Speak(clean_phrase)
                return
            except Exception:
                self._sapi_voice = None

        # 2. Fall back to background PowerShell System.Speech or SAPI on Windows
        if sys.platform == "win32" or platform.system() == "Windows":
            try:
                # Set Rate = 1 for energetic arcade announcer cadence
                ps_script = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$s.Rate = 1; "
                    f"$s.Speak('{clean_phrase}')"
                )
                creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    creationflags=creation_flags,
                )
            except Exception:
                pass

    def speak(self, phrase: str) -> None:
        """Enqueue an announcer voice callout asynchronously."""
        if not self.enable_voice:
            return
        try:
            # Drain queue if full to prevent stale voice lag
            while self._voice_queue.full():
                try:
                    self._voice_queue.get_nowait()
                except queue.Empty:
                    break
            self._voice_queue.put_nowait(phrase)
        except Exception:
            pass

    def _load_or_synthesize_all(self) -> None:
        """Load from assets/sounds/ if present, or synthesize in-memory 8-bit waveforms."""
        if not self.enabled:
            return

        try:
            # Game Start: Upbeat rising 4-note arcade fanfare (C5 -> E5 -> G5 -> C6)
            self.sounds["game_start"] = self._get_sound(
                "game_start",
                [(523.25, 0.09, 0.35), (659.25, 0.09, 0.35), (783.99, 0.09, 0.35), (1046.50, 0.20, 0.40)],
                duty=0.5,
            )

            # Round 1: Ascending power-up arpeggio (C4 -> G4 -> C5 -> E5)
            self.sounds["round_1"] = self._get_sound(
                "round_1",
                [(261.63, 0.07, 0.35), (392.00, 0.07, 0.35), (523.25, 0.07, 0.35), (659.25, 0.16, 0.40)],
                duty=0.5,
            )

            # Round 2: Escalating tempo & pitch arpeggio (E4 -> B4 -> E5 -> G#5 -> B5)
            self.sounds["round_2"] = self._get_sound(
                "round_2",
                [
                    (329.63, 0.055, 0.35),
                    (493.88, 0.055, 0.35),
                    (659.25, 0.055, 0.35),
                    (830.61, 0.055, 0.35),
                    (987.77, 0.16, 0.40),
                ],
                duty=0.45,
            )

            # Round 3 / Final Round: Triumphant high-speed burst (G4 -> D5 -> G5 -> B5 -> D6 -> G6)
            self.sounds["round_3"] = self._get_sound(
                "round_3",
                [
                    (392.00, 0.045, 0.35),
                    (587.33, 0.045, 0.35),
                    (783.99, 0.045, 0.35),
                    (987.77, 0.045, 0.35),
                    (1174.66, 0.045, 0.35),
                    (1567.98, 0.22, 0.42),
                ],
                duty=0.4,
            )

            # Game Over: Descending 8-bit cadence (G4 -> E4 -> C4 -> G3 with vibrato)
            self.sounds["game_over"] = self._get_sound(
                "game_over",
                [(392.00, 0.15, 0.35), (329.63, 0.15, 0.35), (261.63, 0.18, 0.35), (196.00, 0.40, 0.35)],
                duty=0.5,
                vib_last=True,
            )
        except Exception:
            # Never crash during initialization
            pass

    def _get_sound(
        self,
        name: str,
        notes: list[tuple[float, float, float]],
        duty: float = 0.5,
        vib_last: bool = False,
    ) -> pygame.mixer.Sound:
        """Check for custom wav file; otherwise synthesize square wave notes in memory."""
        # 1. Check custom wav file
        if self.custom_dir.is_dir():
            for candidate in (f"{name}.wav", f"{name}.ogg"):
                p = self.custom_dir / candidate
                if p.is_file():
                    try:
                        return pygame.mixer.Sound(str(p))
                    except Exception:
                        pass

        # 2. Synthesize in-memory chiptune waveform
        return self._synthesize_sound(notes, duty=duty, vib_last=vib_last)

    def _synthesize_sound(
        self,
        notes: list[tuple[float, float, float]],
        duty: float = 0.5,
        vib_last: bool = False,
    ) -> pygame.mixer.Sound:
        """Synthesize a sequence of chiptune square-wave notes into a Pygame Sound."""
        chunks: list[np.ndarray] = []
        for i, (freq, dur, vol) in enumerate(notes):
            is_last = (i == len(notes) - 1)
            vr = 6.0 if (vib_last and is_last) else 0.0
            vd = 0.035 if (vib_last and is_last) else 0.0
            chunk = self._synthesize_note(
                freq, dur, volume=vol, duty=duty, vibrato_rate=vr, vibrato_depth=vd
            )
            chunks.append(chunk)

        full_pcm = np.concatenate(chunks, axis=0)
        return pygame.sndarray.make_sound(full_pcm)

    def _synthesize_note(
        self,
        freq: float,
        duration_s: float,
        volume: float = 0.35,
        duty: float = 0.5,
        vibrato_rate: float = 0.0,
        vibrato_depth: float = 0.0,
    ) -> np.ndarray:
        """Generate stereo 16-bit PCM square wave chunk with attack/decay envelope."""
        n_samples = max(1, int(self.sample_rate * duration_s))
        t = np.linspace(0, duration_s, n_samples, endpoint=False)

        if vibrato_rate > 0.0 and vibrato_depth > 0.0:
            f = freq * (1.0 + vibrato_depth * np.sin(2.0 * np.pi * vibrato_rate * t))
            phase = 2.0 * np.pi * np.cumsum(f) / self.sample_rate
        else:
            phase = 2.0 * np.pi * freq * t

        wave = np.where((phase / (2.0 * np.pi)) % 1.0 < duty, 1.0, -1.0)

        # Anti-click attack and exponential decay envelope
        attack = min(int(self.sample_rate * 0.004), n_samples // 4)
        decay = np.linspace(1.0, 0.15, n_samples - attack)
        env = np.ones(n_samples, dtype=np.float32)
        if attack > 0:
            env[:attack] = np.linspace(0.0, 1.0, attack)
        env[attack:] = decay

        mono_pcm = (wave * env * volume * 32767).astype(np.int16)
        return np.column_stack((mono_pcm, mono_pcm))

    def play(self, sound_key: str) -> None:
        """Play a sound effect by name safely."""
        if not self.enabled or sound_key not in self.sounds:
            return
        try:
            self.sounds[sound_key].play()
        except Exception:
            pass

    def play_game_start(self) -> None:
        """Play rising 4-note fanfare when game begins or restarts (music only, no speech clash)."""
        self.play("game_start")

    def play_round_start(self, round_num: int = 1) -> None:
        """Play energetic chiptune jingle and speak arcade callout for Round 1, 2, or 3."""
        key = f"round_{min(3, max(1, round_num))}"
        self.play(key)

        if round_num == 1:
            self.speak("Round 1!")
        elif round_num == 2:
            self.speak("Round 2!")
        else:
            self.speak("Final Round!")

    def play_game_over(self) -> None:
        """Play classic descending 8-bit cadence and speak 'Game Over!' on game over."""
        self.play("game_over")
        self.speak("Game Over!")

    def stop_all(self) -> None:
        """Stop all currently playing sound channels and drain queued speech."""
        if self.enabled:
            try:
                pygame.mixer.stop()
            except Exception:
                pass
        while not self._voice_queue.empty():
            try:
                self._voice_queue.get_nowait()
            except queue.Empty:
                break
