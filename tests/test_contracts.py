import json

from shared.contracts import GameState, Phase, PoseFrame, ScoreEvent


def test_pose_frame_defaults_to_untracked():
    assert PoseFrame().tracked is False


def test_game_state_starts_idle():
    assert GameState().phase == Phase.IDLE


def test_score_event_is_json_serialisable():
    e = ScoreEvent(event_id="A-P001-1", player_id="P001", station="A", game="dodge", score=120)
    assert json.loads(json.dumps(e.to_dict()))["event_id"] == "A-P001-1"
