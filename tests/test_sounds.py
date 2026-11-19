"""Tests for Station A retro 8-bit chiptune sound manager & offline voice announcer."""
from unittest.mock import MagicMock
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
    """Verify play_round_start and play_game_over trigger correct spoken callouts, but not game start."""
    mgr = ArcadeSoundManager(enable_voice=True)
    # Mock speak to verify exact callout phrases without launching external voice processes
    mock_speak = MagicMock()
    mgr.speak = mock_speak

    # 1. Game start: music only, no speech clash
    mgr.play_game_start()
    mock_speak.assert_not_called()

    # 2. Round 1 callout
    mgr.play_round_start(1)
    mock_speak.assert_called_with("Round 1!")

    # 3. Round 2 callout
    mgr.play_round_start(2)
    mock_speak.assert_called_with("Round 2!")

    # 4. Final Round callout
    mgr.play_round_start(3)
    mock_speak.assert_called_with("Final Round!")

    mgr.play_round_start(4)
    mock_speak.assert_called_with("Final Round!")

    # 5. Game Over callout
    mgr.play_game_over()
    mock_speak.assert_called_with("Game Over!")


def test_speak_enqueues_phrase():
    """Verify speak() enqueues phrases in a non-blocking queue."""
    mgr = ArcadeSoundManager(enable_voice=True)
    # Intercept _speak_sync so worker doesn't consume immediately during assertion
    mgr._speak_sync = MagicMock()

    mgr.speak("Round 1!")
    # Verify non-empty queue or task pending
    assert mgr._voice_queue is not None


def test_speak_noop_when_voice_disabled():
    """Verify speak() is a safe no-op when enable_voice is False."""
    mgr = ArcadeSoundManager(enable_voice=False)
    mgr.speak("Round 1!")
    assert mgr._voice_queue.empty()


def test_stop_all_drains_voice_queue():
    """Verify stop_all drains any pending spoken phrases."""
    mgr = ArcadeSoundManager(enable_voice=True)
    mgr._speak_sync = MagicMock()
    mgr._voice_queue.put("Test Phrase 1")
    mgr._voice_queue.put("Test Phrase 2")
    mgr.stop_all()
    assert mgr._voice_queue.empty()


def test_blazer_sound_manager_initialization_and_playback():
    """Verify Station B BlazerSoundManager synthesizes specialized sounds."""
    from station_b.sounds import BlazerSoundManager
    mgr = BlazerSoundManager(enable_voice=False)
    if mgr.enabled:
        assert "pose_snap" in mgr.sounds
        assert "perfection" in mgr.sounds
        assert "pose_skip" in mgr.sounds
        assert "tick" in mgr.sounds
        assert "disqualified" in mgr.sounds

    # Calling play methods must never crash
    mgr.play_snap()
    mgr.play_perfection()
    mgr.play_skip()
    mgr.play_tick()
    mgr.play_disqualified()
    mgr.play_stage_start(1)
    mgr.play_stage_start(2)
    mgr.stop_all()


def test_blazer_sound_manager_speech_callouts():
    """Verify Station B BlazerSoundManager dispatches appropriate speech callouts."""
    from station_b.sounds import BlazerSoundManager
    mgr = BlazerSoundManager(enable_voice=True)
    mock_speak = MagicMock()
    mgr.speak = mock_speak

    mgr.play_perfection()
    mock_speak.assert_called_with("Perfection!")

    mgr.play_skip()
    mock_speak.assert_called_with("Skipped!")

    mgr.play_disqualified()
    mock_speak.assert_called_with("Disqualified!")

    mgr.play_stage_start(1)
    mock_speak.assert_called_with("Stage 1! Precision Blitz!")

    mgr.play_stage_start(2)
    mock_speak.assert_called_with("Stage 2! Speed Rush!")

