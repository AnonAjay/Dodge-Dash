"""Station A: Retro 8-Bit Chiptune Arcade Sound Manager & Offline Announcer.
Provides lightweight, offline sound effects synthesized in memory using NumPy and Pygame,
plus punchy 90s retro arcade cabinet announcer voice callouts with pitch-shift,
slapback echo, stereo chorus, and soft-clip saturation.
Safely falls back to silent operation in headless environments or systems without audio devices.
Checks assets/sounds/*.wav first for custom audio files before synthesizing.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
import platform
import queue
import subprocess
import sys
import threading
import wave
import numpy as np
import pygame


class ArcadeSoundManager:
    """Lightweight chiptune audio generator, sound player, and offline retro arcade voice announcer."""

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
        self._voice_cache: dict[str, pygame.mixer.Sound] = {}

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

        # 2. Pre-generate or load all chiptune sound effects
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
        """Background worker thread to synthesize & play voice callouts without blocking the 60 FPS Pygame loop."""
        while True:
            try:
                phrase = self._voice_queue.get()
                if phrase is None:
                    break

                # 1. If already cached as DSP-processed Sound, play directly
                if phrase in self._voice_cache:
                    try:
                        self._voice_cache[phrase].play()
                    except Exception:
                        pass
                else:
                    # 2. Synthesize phrase into memory stream and apply arcade DSP
                    sound = self._synthesize_voice_clip(phrase)
                    if sound is not None:
                        self._voice_cache[phrase] = sound
                        try:
                            sound.play()
                        except Exception:
                            pass
                    else:
                        # 3. Safe fallback to direct speech if MemoryStream audio unavailable
                        self._speak_fallback(phrase)

                self._voice_queue.task_done()
            except Exception:
                pass

    def _synthesize_voice_clip(self, phrase: str) -> pygame.mixer.Sound | None:
        """Synthesize speech into in-memory WAV buffer and apply punchy retro arcade DSP."""
        clean_phrase = "".join(c for c in phrase if c.isalnum() or c in " !?.,'-").strip()
        if not clean_phrase:
            return None

        # 1. Check custom wav file in assets/sounds/
        slug = clean_phrase.lower().replace(" ", "_").replace("!", "").replace(".", "").replace(",", "")
        if self.custom_dir.is_dir():
            for candidate in (f"voice_{slug}.wav", f"{slug}.wav"):
                p = self.custom_dir / candidate
                if p.is_file():
                    try:
                        return pygame.mixer.Sound(str(p))
                    except Exception:
                        pass

        # 2. Synthesize using Windows SpeechSynthesizer into memory stream
        if sys.platform == "win32" or platform.system() == "Windows":
            try:
                # Rate = 2 gives fast tempo that becomes thick & authoritative when resampled/pitched down
                ps_script = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$ms = New-Object System.IO.MemoryStream; "
                    "$s.SetOutputToWaveStream($ms); "
                    "$s.Rate = 2; "
                    f"$s.Speak('{clean_phrase}'); "
                    "$bytes = $ms.ToArray(); "
                    "[System.Console]::OpenStandardOutput().Write($bytes, 0, $bytes.Length)"
                )
                creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                proc = subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                    capture_output=True,
                    timeout=5,
                    creationflags=creation_flags,
                )
                if proc.returncode == 0 and len(proc.stdout) > 44:
                    sound = self._apply_retro_arcade_dsp(proc.stdout)
                    if sound is not None:
                        return sound
            except Exception:
                pass

        return None

    def _apply_retro_arcade_dsp(self, wav_bytes: bytes) -> pygame.mixer.Sound | None:
        """Apply fast NumPy retro arcade audio DSP:
        - Pitch/formant shift (resample to lower/authoritative register ~0.90x)
        - Resample to target sample rate (44100 Hz)
        - 35ms and 75ms slapback cabinet echo (at 30% and 15% gain)
        - Dual-voice stereo chorus / Haas micro-delay
        - Soft-clip saturation via np.tanh for arcade speaker warmth
        """
        try:
            if len(wav_bytes) <= 44:
                return None
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                raw_frames = wf.readframes(wf.getnframes())
                audio_int16 = np.frombuffer(raw_frames, dtype=np.int16)
                if n_channels == 2:
                    audio_mono = audio_int16[::2].astype(np.float32) / 32768.0
                else:
                    audio_mono = audio_int16.astype(np.float32) / 32768.0

            if len(audio_mono) == 0:
                return None

            # 1. Pitch / Speed Resampling (0.90x speed = lower, authoritative arcade announcer register)
            pitch_factor = 0.90
            old_x = np.arange(len(audio_mono))
            new_x = np.linspace(0, len(audio_mono) - 1, int(len(audio_mono) / pitch_factor))
            resampled = np.interp(new_x, old_x, audio_mono)

            # 2. Resample to target sample rate if needed
            target_sr = self.sample_rate
            if sr != target_sr:
                old_sr_x = np.arange(len(resampled))
                new_sr_x = np.linspace(0, len(resampled) - 1, int(len(resampled) * (target_sr / sr)))
                resampled = np.interp(new_sr_x, old_sr_x, resampled)

            # 3. Slapback Echo (35ms at 30%, 75ms at 15%)
            d1 = int(target_sr * 0.035)
            d2 = int(target_sr * 0.075)
            base_sig = np.pad(resampled, (0, d2))
            echo1 = np.pad(resampled * 0.30, (d1, d2 - d1))
            echo2 = np.pad(resampled * 0.15, (d2, 0))
            mixed = base_sig + echo1 + echo2

            # 4. Stereo Chorus / Dual-voice detune (Haas effect 8ms delay between L and R)
            delay_lr = int(target_sr * 0.008)
            left_chan = mixed.copy()
            right_chan = np.pad(mixed[:-delay_lr], (delay_lr, 0))

            # 5. Soft-clip saturation (np.tanh) for arcade cabinet drive
            left_sat = np.tanh(left_chan * 1.35)
            right_sat = np.tanh(right_chan * 1.35)

            # Normalize peak to 0.92
            peak = max(float(np.max(np.abs(left_sat))), float(np.max(np.abs(right_sat))))
            if peak > 0:
                left_sat = (left_sat / peak) * 0.92
                right_sat = (right_sat / peak) * 0.92

            # Convert to 16-bit signed stereo PCM
            left_pcm = (left_sat * 32767).astype(np.int16)
            right_pcm = (right_sat * 32767).astype(np.int16)
            stereo = np.column_stack((left_pcm, right_pcm))

            return pygame.sndarray.make_sound(stereo)
        except Exception:
            return None

    def _speak_fallback(self, phrase: str) -> None:
        """Fallback direct speech if DSP synthesis into memory stream is unavailable."""
        clean_phrase = "".join(c for c in phrase if c.isalnum() or c in " !?.,'-").strip()
        if not clean_phrase:
            return

        if self._sapi_voice is not None:
            try:
                self._sapi_voice.Speak(clean_phrase)
                return
            except Exception:
                self._sapi_voice = None

        if sys.platform == "win32" or platform.system() == "Windows":
            try:
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
        # If already cached as a Sound, play directly
        if phrase in self._voice_cache:
            try:
                self._voice_cache[phrase].play()
                return
            except Exception:
                pass

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
        """Play rising 4-note fanfare and announce 'Game Start! Step into the arena!'."""
        self.play("game_start")
        self.speak("Game Start! Step into the arena!")

    def play_round_start(self, round_num: int = 1) -> None:
        """Play energetic chiptune jingle and speak punchy arcade callout for Round 1, 2, or 3."""
        key = f"round_{min(3, max(1, round_num))}"
        self.play(key)

        if round_num == 1:
            self.speak("Round One! Ready... Go!")
        elif round_num == 2:
            self.speak("Round Two!")
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
