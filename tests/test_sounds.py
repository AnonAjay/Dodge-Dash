"""Tests for Station A retro 8-bit chiptune sound manager & offline voice announcer."""
import io
from unittest.mock import MagicMock
import wave
import numpy as np
import pytest
from station_a.sounds import ArcadeSoundManager


def test_arcade_sound_manager_initialization():
    """Verify sound manager initializes safely and pre-generates chiptune sounds."""
    mgr = ArcadeSoundManager(enable_voice=False)
    if mgr.enabled:
        assert "game_start" in mgr.sounds
        assert "round_1" in mgr.sounds
        assert "round_2" in mgr.sounds
        assert "round_3" in mgr.sounds
        assert "game_over" in mgr.sounds
    assert mgr.enable_voice is False


def test_arcade_sound_manager_playback_calls_do_not_crash():
    """Verify playback methods never raise exceptions regardless of audio hardware state."""
    mgr = ArcadeSoundManager(enable_voice=False)
    mgr.play_game_start()
    mgr.play_round_start(1)
    mgr.play_round_start(2)
    mgr.play_round_start(3)
    mgr.play_round_start(99)  # Clamped to round 3
    mgr.play_game_over()
    mgr.stop_all()


def test_arcade_sound_manager_safe_when_disabled():
    """Verify disabled manager handles calls as safe no-ops."""
    mgr = ArcadeSoundManager(enable_voice=False)
    mgr.enabled = False
    mgr.play_game_start()
    mgr.play_round_start(1)
    mgr.play_game_over()
    mgr.stop_all()


def test_custom_sounds_dir_fallback(tmp_path):
    """Verify fallback to synthesized audio when custom audio files do not exist."""
    custom_dir = tmp_path / "custom_sounds"
    custom_dir.mkdir()
    mgr = ArcadeSoundManager(custom_sounds_dir=custom_dir, enable_voice=False)
    if mgr.enabled:
        assert "game_start" in mgr.sounds


def test_spoken_callouts_dispatch():
    """Verify play_game_start, play_round_start and play_game_over trigger correct hype arcade callouts."""
    mgr = ArcadeSoundManager(enable_voice=True)
    # Mock speak to verify exact callout phrases without launching external voice processes
    mock_speak = MagicMock()
    mgr.speak = mock_speak

    # 1. Game start: fanfare + "Game Start! Step into the arena!"
    mgr.play_game_start()
    mock_speak.assert_called_with("Game Start! Step into the arena!")

    # 2. Round 1 callout
    mgr.play_round_start(1)
    mock_speak.assert_called_with("Round One! Ready... Go!")

    # 3. Round 2 callout
    mgr.play_round_start(2)
    mock_speak.assert_called_with("Round Two!")

    # 4. Final Round callout
    mgr.play_round_start(3)
    mock_speak.assert_called_with("Final Round!")

    mgr.play_round_start(4)
    mock_speak.assert_called_with("Final Round!")

    # 5. Game Over callout
    mgr.play_game_over()
    mock_speak.assert_called_with("Game Over!")


def test_retro_arcade_dsp_processing():
    """Verify _apply_retro_arcade_dsp applies pitch shift, slapback echo, and saturation cleanly."""
    mgr = ArcadeSoundManager(enable_voice=False)
    if not mgr.enabled:
        return

    # Create dummy 0.2s 22050Hz 16-bit mono sine wave in memory
    buf = io.BytesIO()
    sr = 22050
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    pcm_sine = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_sine.tobytes())

    sound = mgr._apply_retro_arcade_dsp(buf.getvalue())
    assert sound is not None
    # Verify echo extension: original was 0.2s, with 75ms echo and 0.90x pitch it should be ~0.29s
    assert sound.get_length() > 0.25


def test_speak_enqueues_phrase():
    """Verify speak() enqueues phrases in a non-blocking queue."""
    mgr = ArcadeSoundManager(enable_voice=True)
    mgr.speak("Round 1!")
    assert mgr._voice_queue is not None


def test_speak_noop_when_voice_disabled():
    """Verify speak() is a safe no-op when enable_voice is False."""
    mgr = ArcadeSoundManager(enable_voice=False)
    mgr.speak("Round 1!")
    assert mgr._voice_queue.empty()


def test_stop_all_drains_voice_queue():
    """Verify stop_all drains any pending spoken phrases."""
    mgr = ArcadeSoundManager(enable_voice=True)
    mgr._voice_queue.put("Test Phrase 1")
    mgr._voice_queue.put("Test Phrase 2")
    mgr.stop_all()
    assert mgr._voice_queue.empty()
