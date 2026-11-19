import base64
import importlib
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

import server.app as app_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    importlib.reload(app_module)
    return TestClient(app_module.app)


def test_score_is_idempotent(client):
    body = {"event_id": "A-P001-1", "player_id": "P001", "station": "A", "game": "dodge", "score": 50}
    res1 = client.post("/api/score", json=body)
    assert res1.status_code == 200
    assert res1.json()["accepted"] is True
    assert res1.json()["duplicate"] is False

    res2 = client.post("/api/score", json=body)
    assert res2.status_code == 200
    assert res2.json()["duplicate"] is True

    leaderboard = client.get("/api/leaderboard").json()
    assert len(leaderboard["dodge"]) == 1


def test_player_registration_sequential_ids_and_roll_no(client):
    # Register first player with roll_no
    p1 = client.post("/api/players", json={
        "name": "Alice",
        "roll_no": "24303107",
        "height_cm": 172.5
    }).json()
    assert p1["player_id"] == "P001"
    assert p1["name"] == "Alice"
    assert p1["roll_no"] == "24303107"
    assert p1["height_cm"] == 172.5
    assert p1["photo_url"] is None

    # Register second player without roll_no
    p2 = client.post("/api/players", json={"name": "Bob"}).json()
    assert p2["player_id"] == "P002"
    assert p2["name"] == "Bob"
    assert p2["roll_no"] is None
    assert p2["height_cm"] is None

    # Validate empty name error
    bad_res = client.post("/api/players", json={"name": "   "})
    assert bad_res.status_code == 422


def test_player_registration_with_photo(client):
    dummy_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb"
    photo_b64 = "data:image/jpeg;base64," + base64.b64encode(dummy_jpeg).decode()

    res = client.post("/api/players", json={
        "name": "Charlie",
        "roll_no": "24303199",
        "photo_base64": photo_b64
    }).json()
    assert res["player_id"] == "P001"
    assert res["roll_no"] == "24303199"
    assert res["photo_url"] == "/api/photos/P001.jpg"

    # Verify file saved on disk
    photo_path = Path("data/photos/P001.jpg")
    assert photo_path.exists()
    assert photo_path.read_bytes() == dummy_jpeg

    # Fetch photo through endpoint
    img_res = client.get("/api/photos/P001.jpg")
    assert img_res.status_code == 200
    assert img_res.content == dummy_jpeg


def test_get_players_lookup(client):
    client.post("/api/players", json={"name": "Dana", "roll_no": "24303108", "height_cm": 165.0})
    client.post("/api/players", json={"name": "Evan"})

    all_players = client.get("/api/players").json()
    assert len(all_players) == 2
    assert all_players[0]["player_id"] == "P001"
    assert all_players[0]["roll_no"] == "24303108"
    assert all_players[1]["player_id"] == "P002"
    assert all_players[1]["roll_no"] is None

    dana = client.get("/api/players/P001").json()
    assert dana["name"] == "Dana"
    assert dana["roll_no"] == "24303108"

    unknown = client.get("/api/players/P999")
    assert unknown.status_code == 404


def test_leaderboard_personal_best_and_tie_breaking(client):
    client.post("/api/players", json={"name": "Alice", "roll_no": "24303107"})
    client.post("/api/players", json={"name": "Bob", "roll_no": "24303108"})

    # P001 (Alice) plays dodge multiple times
    client.post("/api/score", json={
        "event_id": "A-P001-1", "player_id": "P001", "station": "A", "game": "dodge",
        "score": 50, "accuracy": 0.6, "best_streak": 5
    })
    client.post("/api/score", json={
        "event_id": "A-P001-2", "player_id": "P001", "station": "A", "game": "dodge",
        "score": 100, "accuracy": 0.8, "best_streak": 10
    })
    client.post("/api/score", json={
        "event_id": "A-P001-3", "player_id": "P001", "station": "A", "game": "dodge",
        "score": 80, "accuracy": 0.95, "best_streak": 12
    })

    # P001 should only have ONE personal-best entry in dodge leaderboard (score 100)
    lb = client.get("/api/leaderboard").json()
    assert len(lb["dodge"]) == 1
    assert lb["dodge"][0]["player_id"] == "P001"
    assert lb["dodge"][0]["score"] == 100
    assert lb["dodge"][0]["name"] == "Alice"

    # P002 (Bob) plays dodge: same score (100) but lower accuracy (0.75)
    client.post("/api/score", json={
        "event_id": "A-P002-1", "player_id": "P002", "station": "A", "game": "dodge",
        "score": 100, "accuracy": 0.75, "best_streak": 14
    })
    lb = client.get("/api/leaderboard").json()
    assert len(lb["dodge"]) == 2
    # Alice (accuracy 0.8) should beat Bob (accuracy 0.75)
    assert lb["dodge"][0]["player_id"] == "P001"
    assert lb["dodge"][1]["player_id"] == "P002"

    # Now Bob gets same score (100) and same accuracy (0.8), but higher streak (15 vs 10)
    client.post("/api/score", json={
        "event_id": "A-P002-2", "player_id": "P002", "station": "A", "game": "dodge",
        "score": 100, "accuracy": 0.8, "best_streak": 15
    })
    lb = client.get("/api/leaderboard").json()
    assert len(lb["dodge"]) == 2
    # Bob should now rank #1 on best_streak tie-break
    assert lb["dodge"][0]["player_id"] == "P002"
    assert lb["dodge"][0]["name"] == "Bob"
    assert lb["dodge"][1]["player_id"] == "P001"


def test_big_screen_display_offline(client):
    res = client.get("/")
    assert res.status_code == 200
    html = res.text
    assert "TITANS' ARCADE" in html or "Titans' Arcade" in html
    assert "FLOOR IS LAVA" in html
    assert "REGISTRATION PANEL" in html
    assert "LEADERBOARD" in html
    assert "LIVE RESULTS" in html
    assert "#58c038" in html
    assert "/static/bg-sunset.png" in html
    assert "/register" in html
    assert "pixel-panel" in html or "leaderboard-container" in html
    # D logo box, old status, badges, showcase box, and footer should be removed
    assert "brand-logo" not in html
    assert "Live Festival Network" not in html
    assert "Spectator Arena View" not in html
    assert "Personal Best Records Only" not in html
    assert "TECH-FEST 2026" not in html
    assert "Latest Finisher Card" not in html
    assert "showcase-panel" not in html
    # Spectator display should not have player registration form input
    assert 'id="player-name"' not in html
    # Ensure 100% offline (no external CDN links)
    assert "https://" not in html
    assert "http://" not in html


def test_operator_desk_offline(client):
    res = client.get("/register")
    assert res.status_code == 200
    html = res.text
    assert "TITANS' ARCADE" in html or "Titans' Arcade" in html
    assert "LEADERBOARD" in html
    assert "Player Registration" in html
    assert "URN" in html
    assert "Registered Participants" in html
    assert "Snap &amp; Register Player" in html or "Snap & Register Player" in html
    assert "aspect-ratio: 4 / 3" in html or "aspect-ratio: 4/3" in html
    assert "max-height: 44vh" in html
    assert "font-size: 14px" in html
    assert "font-size: 16px" in html
    assert "font-size: 13px" in html
    assert "font-size: 15px" in html
    assert "padding: 4px 10px" in html
    assert "navigator.mediaDevices.getUserMedia" in html
    # Check interactive drag-to-resize divider elements & features
    assert "drag-divider" in html
    assert "col-resize" in html
    assert "localStorage" in html
    assert "dblclick" in html
    assert "href=\"/\"" in html or "href='/'" in html
    assert "reg-warning" in html
    # Ensure removed D logo, old badge, and footer
    assert "brand-logo" not in html
    assert "Operator Check-In Desk" not in html
    assert "OPERATOR DESK" not in html
    assert "Quick Player Check-In" not in html
    # Ensure 100% offline (no external CDN links)
    assert "https://" not in html
    assert "http://" not in html


def test_duplicate_urn_rejection(client):
    # Register first player with URN
    p1 = client.post("/api/players", json={"name": "Drishti", "roll_no": "24303192"}).json()
    assert p1["player_id"] == "P001"
    assert p1["name"] == "Drishti"

    # Attempt to register another player with the same URN (case-insensitive & trimmed)
    res_dup = client.post("/api/players", json={"name": "Duplicate User", "roll_no": "  24303192  "})
    assert res_dup.status_code == 409
    dup_data = res_dup.json()
    assert dup_data["existing_player_id"] == "P001"
    assert dup_data["existing_name"] == "Drishti"
    assert "24303192" in dup_data["detail"]

    # Verify no duplicate player was created
    players = client.get("/api/players").json()
    assert len(players) == 1


def test_blank_urn_allowance(client):
    # Register multiple players with None or blank/whitespace URNs
    p1 = client.post("/api/players", json={"name": "Player One", "roll_no": None})
    assert p1.status_code == 200
    assert p1.json()["player_id"] == "P001"

    p2 = client.post("/api/players", json={"name": "Player Two", "roll_no": "   "})
    assert p2.status_code == 200
    assert p2.json()["player_id"] == "P002"

    p3 = client.post("/api/players", json={"name": "Player Three"})
    assert p3.status_code == 200
    assert p3.json()["player_id"] == "P003"

    players = client.get("/api/players").json()
    assert len(players) == 3


def test_station_queue_join_and_pop(client):
    # Register player
    p1 = client.post("/api/players", json={"name": "Aarav", "height_cm": 175.0}).json()
    pid = p1["player_id"]

    # Join queue for Station A
    res_join = client.post("/api/queue/join", json={"player_id": pid, "station": "A"})
    assert res_join.status_code == 200
    assert res_join.json()["ok"] is True
    assert res_join.json()["position"] == 1

    # Check station status
    stations = client.get("/api/stations").json()
    assert stations["A"]["queue_len"] == 1

    # Station A pops next player
    next_res = client.get("/api/queue/A/next").json()
    assert next_res["player"] is not None
    assert next_res["player"]["player_id"] == pid
    assert next_res["player"]["name"] == "Aarav"

    # Next pop is empty
    empty_res = client.get("/api/queue/A/next").json()
    assert empty_res["player"] is None
    assert empty_res["remaining"] == 0


def test_station_heartbeat_and_live_status(client):
    # Send heartbeat from Station B
    hb_res = client.post("/api/queue/B/heartbeat", json={
        "status": "playing",
        "player_id": "P005",
        "score": 1250,
    })
    assert hb_res.status_code == 200
    assert hb_res.json()["ok"] is True

    # Verify /api/stations reflects Station B online and playing
    stations = client.get("/api/stations").json()
    assert stations["B"]["online"] is True
    assert stations["B"]["status"] == "playing"
    assert stations["B"]["current_player"] == "P005"
    assert stations["B"]["current_score"] == 1250


def test_player_registration_with_station_dispatch(client):
    # Register with station parameter
    p = client.post("/api/players", json={
        "name": "Karan",
        "station": "B"
    }).json()
    assert p["player_id"] == "P001"
    assert p["queued_station"] == "B"

    # Verify automatically in Station B queue
    next_b = client.get("/api/queue/B/next").json()
    assert next_b["player"]["player_id"] == "P001"
    assert next_b["player"]["name"] == "Karan"

