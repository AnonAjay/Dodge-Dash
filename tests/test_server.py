import os

from fastapi.testclient import TestClient


def test_score_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib
    import server.app as app_module
    importlib.reload(app_module)
    client = TestClient(app_module.app)
    body = {"event_id": "A-P001-1", "player_id": "P001", "station": "A", "game": "dodge", "score": 50}
    assert client.post("/api/score", json=body).json()["duplicate"] is False
    assert client.post("/api/score", json=body).json()["duplicate"] is True
    assert len(client.get("/api/leaderboard").json()["dodge"]) == 1
