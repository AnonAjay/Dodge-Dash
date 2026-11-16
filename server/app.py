"""Laptop 3: score server. Run:  uvicorn server.app:app --host 0.0.0.0 --port 8000
Minimal on purpose: health, accept score (idempotent), leaderboard, player registration.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

DATA = Path("data")
SCORES_FILE = DATA / "scores.jsonl"
PLAYERS_FILE = DATA / "players.json"
PHOTOS_DIR = DATA / "photos"

app = FastAPI(title="Dodge & Drip server")

_lock = threading.Lock()
_seen_event_ids: set[str] = set()


def _load_seen_event_ids() -> None:
    global _seen_event_ids
    _seen_event_ids = set()
    if SCORES_FILE.exists():
        try:
            for line in SCORES_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if "event_id" in record:
                            _seen_event_ids.add(record["event_id"])
                    except Exception:
                        pass
        except Exception:
            pass


_load_seen_event_ids()


class ScoreIn(BaseModel):
    event_id: str
    player_id: str
    station: str
    game: str
    score: int
    accuracy: float = 0.0
    best_streak: int = 0
    hype_avg: float | None = None
    duration_s: float = 0.0
    finished_at: str = ""


class PlayerCreate(BaseModel):
    name: str
    roll_no: str | None = None
    height_cm: float | None = None
    photo_base64: str | None = None


class PlayerOut(BaseModel):
    player_id: str
    name: str
    roll_no: str | None = None
    height_cm: float | None = None
    photo_url: str | None = None


def _read_scores_locked() -> list[dict]:
    if not SCORES_FILE.exists():
        return []
    try:
        return [json.loads(line) for line in SCORES_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []


def _read_players_locked() -> list[dict]:
    if not PLAYERS_FILE.exists():
        return []
    try:
        data = json.loads(PLAYERS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def _write_players_locked(players: list[dict]) -> None:
    DATA.mkdir(exist_ok=True)
    PLAYERS_FILE.write_text(json.dumps(players, indent=2), encoding="utf-8")


def _generate_next_player_id(players: list[dict]) -> str:
    max_id = 0
    for p in players:
        pid = p.get("player_id", "")
        match = re.match(r"^P(\d+)$", pid)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return f"P{max_id + 1:03d}"


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/players", response_model=PlayerOut)
def register_player(player_in: PlayerCreate) -> Any:
    clean_name = player_in.name.strip()
    if not clean_name:
        raise HTTPException(status_code=422, detail="Name cannot be empty")

    clean_roll = player_in.roll_no.strip() if player_in.roll_no and player_in.roll_no.strip() else None

    with _lock:
        players = _read_players_locked()

        if clean_roll:
            norm_roll = clean_roll.lower()
            for p in players:
                existing_roll = (p.get("roll_no") or "").strip().lower()
                if existing_roll and existing_roll == norm_roll:
                    existing_id = p.get("player_id", "")
                    existing_name = p.get("name", "")
                    return JSONResponse(
                        status_code=409,
                        content={
                            "detail": f"URN {clean_roll} is already registered to {existing_name} ({existing_id})",
                            "existing_player_id": existing_id,
                            "existing_name": existing_name,
                        },
                    )

        player_id = _generate_next_player_id(players)

        photo_url: str | None = None
        if player_in.photo_base64:
            raw_b64 = player_in.photo_base64.strip()
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(raw_b64)
                PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
                photo_file = PHOTOS_DIR / f"{player_id}.jpg"
                photo_file.write_bytes(img_bytes)
                photo_url = f"/api/photos/{player_id}.jpg"
            except Exception:
                photo_url = None

        profile = {
            "player_id": player_id,
            "name": clean_name,
            "roll_no": clean_roll,
            "height_cm": player_in.height_cm,
            "photo_url": photo_url,
        }
        players.append(profile)
        _write_players_locked(players)
        return profile


@app.get("/api/players", response_model=list[PlayerOut])
def list_players() -> list[dict]:
    with _lock:
        return _read_players_locked()


@app.get("/api/players/{player_id}", response_model=PlayerOut)
def get_player(player_id: str) -> dict:
    with _lock:
        players = _read_players_locked()
        for p in players:
            if p.get("player_id") == player_id:
                return p
    raise HTTPException(status_code=404, detail="Player not found")


@app.get("/api/photos/{filename}")
def get_photo(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    file_path = PHOTOS_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Photo not found")
    return FileResponse(file_path)


@app.post("/api/score")
def post_score(event: ScoreIn) -> dict:
    with _lock:
        if event.event_id in _seen_event_ids:
            return {"accepted": True, "duplicate": True}
        DATA.mkdir(exist_ok=True)
        with SCORES_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.model_dump()) + "\n")
        _seen_event_ids.add(event.event_id)
    return {"accepted": True, "duplicate": False}


@app.get("/api/leaderboard")
def leaderboard() -> dict[str, list[dict]]:
    with _lock:
        scores = _read_scores_locked()
        players = _read_players_locked()

    player_names = {p["player_id"]: p.get("name", p["player_id"]) for p in players}

    by_game: dict[str, dict[str, dict]] = {"dodge": {}, "wall": {}, "dance": {}}

    for score_record in scores:
        game = score_record.get("game", "")
        pid = score_record.get("player_id", "")
        if not game or not pid:
            continue

        if game not in by_game:
            by_game[game] = {}

        candidate_key = (
            score_record.get("score", 0),
            score_record.get("accuracy", 0.0),
            score_record.get("best_streak", 0),
        )

        existing = by_game[game].get(pid)
        if existing is None:
            by_game[game][pid] = score_record
        else:
            existing_key = (
                existing.get("score", 0),
                existing.get("accuracy", 0.0),
                existing.get("best_streak", 0),
            )
            if candidate_key > existing_key:
                by_game[game][pid] = score_record

    result: dict[str, list[dict]] = {}
    for game, player_bests in by_game.items():
        rows = []
        for pid, record in player_bests.items():
            row = dict(record)
            row["name"] = player_names.get(pid, pid)
            rows.append(row)

        rows.sort(
            key=lambda r: (
                r.get("score", 0),
                r.get("accuracy", 0.0),
                r.get("best_streak", 0),
            ),
            reverse=True,
        )
        result[game] = rows[:10]

    return result


@app.get("/", response_class=HTMLResponse)
def big_screen_display() -> str:
    """Public Big-Screen Spectator Display for Projector / TV (F11 fullscreen)."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dodge &amp; Drip // Live Arena Leaderboard</title>
<style>
  :root {
    --bg-dark: #07090e;
    --card-bg: #0f1523;
    --card-border: #1e293b;
    --accent-cyan: #00f2fe;
    --accent-purple: #9d4edd;
    --accent-pink: #ff007f;
    --accent-green: #10b981;
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
    --gold: #facc15;
    --silver: #e2e8f0;
    --bronze: #f97316;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  body {
    background-color: var(--bg-dark);
    color: var(--text-primary);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    user-select: none;
  }

  /* Header */
  header {
    height: 64px;
    background: #0b1120;
    border-bottom: 2px solid var(--card-border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    flex-shrink: 0;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .brand-text {
    font-size: 22px;
    font-weight: 900;
    letter-spacing: 1px;
    background: linear-gradient(90deg, #ffffff, var(--accent-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .header-right {
    display: flex;
    align-items: center;
    gap: 20px;
  }
  .status-tag {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: var(--text-muted);
  }
  .status-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #ef4444;
    box-shadow: 0 0 8px rgba(239, 68, 68, 0.7);
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse { 0% { opacity: 0.6; } 50% { opacity: 1; } 100% { opacity: 0.6; } }
  .nav-link {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    text-decoration: none;
    color: var(--text-muted);
    background: #1e293b;
    padding: 6px 14px;
    border-radius: 6px;
    border: 1px solid var(--card-border);
    transition: all 0.2s;
  }
  .nav-link:hover {
    color: #fff;
    border-color: var(--accent-cyan);
    box-shadow: 0 0 10px rgba(0, 242, 254, 0.25);
  }

  /* Main Grid */
  main {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 20px 32px;
    overflow: hidden;
  }

  /* Big Full-Width Leaderboard */
  .leaderboard-container {
    flex: 1;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 12px 30px rgba(0,0,0,0.5);
  }
  .leaderboard-top {
    padding: 14px 28px;
    background: rgba(11, 17, 32, 0.7);
    border-bottom: 1px solid var(--card-border);
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
  }
  .leaderboard-title {
    font-size: 1.35rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 800;
    color: #fff;
    text-align: center;
  }
  .tabs {
    display: flex;
    gap: 6px;
    justify-self: end;
  }
  .tab-btn {
    cursor: pointer;
    background: #090d16;
    color: var(--text-muted);
    border: 1px solid var(--card-border);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    transition: all 0.2s;
  }
  .tab-btn:hover { color: #fff; border-color: rgba(0, 242, 254, 0.5); }
  .tab-btn.active {
    background: linear-gradient(135deg, var(--accent-cyan), #0099ff);
    color: #030712;
    border-color: var(--accent-cyan);
    box-shadow: 0 0 14px rgba(0, 242, 254, 0.45);
  }

  .table-wrap {
    flex: 1;
    padding: 16px 28px;
    overflow-y: auto;
  }
  .lb-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0 8px;
  }
  .lb-table th {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 12px 18px;
    text-align: left;
    border-bottom: 1px solid var(--card-border);
  }
  .lb-table td {
    background: #090e18;
    padding: 15px 18px;
    font-size: 16px;
    font-weight: 600;
    border-top: 1px solid rgba(255, 255, 255, 0.04);
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
  }
  .lb-table tr td:first-child {
    border-top-left-radius: 10px;
    border-bottom-left-radius: 10px;
    border-left: 1px solid rgba(255, 255, 255, 0.04);
    font-weight: 900;
    font-size: 18px;
    width: 90px;
    text-align: center;
  }
  .lb-table tr td:last-child {
    border-top-right-radius: 10px;
    border-bottom-right-radius: 10px;
    border-right: 1px solid rgba(255, 255, 255, 0.04);
  }
  .rank-1 td { background: linear-gradient(90deg, rgba(250, 204, 21, 0.12), #090e18 50%); }
  .rank-2 td { background: linear-gradient(90deg, rgba(226, 232, 240, 0.09), #090e18 50%); }
  .rank-3 td { background: linear-gradient(90deg, rgba(249, 115, 22, 0.09), #090e18 50%); }
  .rank-1-badge { color: var(--gold); text-shadow: 0 0 12px rgba(250, 204, 21, 0.5); }
  .rank-2-badge { color: var(--silver); }
  .rank-3-badge { color: var(--bronze); }
  .score-val {
    font-size: 22px;
    font-weight: 900;
    color: var(--accent-cyan);
    font-variant-numeric: tabular-nums;
  }
  .pid-tag {
    background: #1e293b;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 13px;
    color: #cbd5e1;
    font-family: monospace;
    margin-right: 8px;
  }
  .empty-board {
    text-align: center;
    padding: 120px 20px;
    color: var(--text-muted);
    font-size: 18px;
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="brand-text">DODGE &amp; DRIP</div>
  </div>
  <div class="header-right">
    <div class="status-tag">
      <span class="status-dot"></span> LIVE RESULTS
    </div>
    <span style="color: var(--card-border);">|</span>
    <a href="/register" class="nav-link">REGISTRATION PANEL</a>
  </div>
</header>

<main>
  <!-- Full-Width Massive Leaderboard -->
  <section class="leaderboard-container">
    <div class="leaderboard-top">
      <div></div>
      <div class="leaderboard-title">LEADERBOARD</div>
      <div class="tabs">
        <button class="tab-btn active" data-game="dodge" onclick="switchTab('dodge')">Dodge</button>
        <button class="tab-btn" data-game="wall" onclick="switchTab('wall')">Wall Shape</button>
        <button class="tab-btn" data-game="dance" onclick="switchTab('dance')">Dance</button>
      </div>
    </div>
    <div class="table-wrap">
      <table class="lb-table" id="lb-table">
        <thead>
          <tr>
            <th style="width: 90px; text-align: center;">Rank</th>
            <th style="width: 140px;">Player ID</th>
            <th>Player Name</th>
            <th style="width: 160px;">Score</th>
            <th style="width: 140px;">Accuracy</th>
            <th style="width: 140px;">Best Streak</th>
          </tr>
        </thead>
        <tbody id="lb-body">
          <!-- Dynamically populated -->
        </tbody>
      </table>
      <div id="lb-empty" class="empty-board" style="display:none;">
        No scores recorded yet for this event.<br>Step on the arena tiles and claim the top rank!
      </div>
    </div>
  </section>
</main>

<script>
  let activeGame = 'dodge';
  let lbData = {};

  function switchTab(game) {
    activeGame = game;
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.game === game);
    });
    renderBoard();
  }

  async function fetchLeaderboard() {
    try {
      const res = await fetch('/api/leaderboard');
      if (res.ok) {
        lbData = await res.json();
        renderBoard();
      }
    } catch (e) {
      // offline polling
    }
  }

  function renderBoard() {
    const tbody = document.getElementById('lb-body');
    const table = document.getElementById('lb-table');
    const empty = document.getElementById('lb-empty');
    const rows = lbData[activeGame] || [];

    if (rows.length === 0) {
      tbody.innerHTML = '';
      table.style.display = 'none';
      empty.style.display = 'block';
      return;
    }

    table.style.display = 'table';
    empty.style.display = 'none';

    tbody.innerHTML = rows.map((r, i) => {
      const rank = i + 1;
      let rankText = '#' + rank;
      let rowClass = '';
      if (rank === 1) { rankText = '🥇 1'; rowClass = 'rank-1 rank-1-badge'; }
      else if (rank === 2) { rankText = '🥈 2'; rowClass = 'rank-2 rank-2-badge'; }
      else if (rank === 3) { rankText = '🥉 3'; rowClass = 'rank-3 rank-3-badge'; }

      const acc = r.accuracy != null ? (r.accuracy * 100).toFixed(0) + '%' : '-';
      const streak = r.best_streak ?? '-';

      return `<tr class="${rowClass}">
        <td>${rankText}</td>
        <td><span class="pid-tag">${r.player_id}</span></td>
        <td style="font-weight: 700; font-size: 16px;">${r.name || r.player_id}</td>
        <td class="score-val">${r.score}</td>
        <td>${acc}</td>
        <td>${streak}</td>
      </tr>`;
    }).join('');
  }

  fetchLeaderboard();
  setInterval(fetchLeaderboard, 2000);
</script>
</body>
</html>"""


@app.get("/register", response_class=HTMLResponse)
def operator_desk() -> str:
    """Operator Check-In & Roster Desk for Laptop 3 operator."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dodge &amp; Drip // Operator Registration Desk</title>
<style>
  :root {
    --bg-dark: #090d16;
    --card-bg: #121826;
    --card-border: #1e293b;
    --accent-cyan: #00f2fe;
    --accent-purple: #9d4edd;
    --accent-green: #10b981;
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  body {
    background-color: var(--bg-dark);
    color: var(--text-primary);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    user-select: none;
  }

  /* Header */
  header {
    height: 56px;
    background: #0f172a;
    border-bottom: 1px solid var(--card-border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    flex-shrink: 0;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand-text {
    font-size: 18px;
    font-weight: 800;
    color: #fff;
  }
  .nav-link {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    text-decoration: none;
    color: var(--text-muted);
    background: #1e293b;
    padding: 6px 14px;
    border-radius: 6px;
    border: 1px solid var(--card-border);
    transition: all 0.2s;
  }
  .nav-link:hover { color: #fff; border-color: var(--accent-cyan); }

  /* Main Split */
  main {
    flex: 1;
    display: flex;
    flex-direction: row;
    gap: 0;
    padding: 16px;
    overflow: hidden;
    position: relative;
  }

  /* Interactive Drag-to-Resize Divider */
  .drag-divider {
    width: 14px;
    margin: 0 4px;
    cursor: col-resize;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    user-select: none;
    z-index: 10;
    flex-shrink: 0;
    touch-action: none;
  }
  .drag-divider::before {
    content: "";
    position: absolute;
    top: 8px;
    bottom: 8px;
    width: 2px;
    background: var(--card-border);
    border-radius: 2px;
    transition: background-color 0.2s, box-shadow 0.2s, width 0.2s;
  }
  .drag-divider:hover::before,
  .drag-divider.dragging::before {
    background: var(--accent-cyan);
    box-shadow: 0 0 10px var(--accent-cyan);
    width: 4px;
  }
  .drag-handle {
    width: 6px;
    height: 36px;
    border-radius: 3px;
    background: #1e293b;
    border: 1px solid var(--card-border);
    position: absolute;
    transition: all 0.2s;
  }
  .drag-divider:hover .drag-handle,
  .drag-divider.dragging .drag-handle {
    background: var(--accent-cyan);
    border-color: var(--accent-cyan);
    box-shadow: 0 0 8px rgba(0, 242, 254, 0.7);
  }

  .panel {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .panel-header {
    padding: 12px 18px;
    background: rgba(15, 23, 42, 0.6);
    border-bottom: 1px solid var(--card-border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .panel-title {
    font-size: 14px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .panel-content {
    flex: 1;
    padding: 16px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  /* Form Elements */
  .form-group {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .form-group label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--text-muted);
    letter-spacing: 0.5px;
  }
  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  input[type="text"], input[type="number"] {
    background: #090d16;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 9px 12px;
    color: #fff;
    font-size: 13px;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  input[type="text"]:focus, input[type="number"]:focus {
    border-color: var(--accent-cyan);
    box-shadow: 0 0 0 2px rgba(0, 242, 254, 0.2);
  }

  /* Camera preview */
  .camera-container {
    position: relative;
    width: 100%;
    aspect-ratio: 4 / 3;
    max-height: 46vh;
    background: #06090e;
    border: 1px solid var(--card-border);
    border-radius: 10px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  video {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 10px;
    transform: scaleX(-1);
  }
  .camera-badge {
    position: absolute;
    top: 6px;
    left: 6px;
    background: rgba(0, 0, 0, 0.7);
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
    color: var(--accent-cyan);
  }
  .camera-placeholder {
    position: absolute;
    color: var(--text-muted);
    font-size: 12px;
    text-align: center;
    padding: 10px;
  }

  /* Buttons */
  .btn-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: 4px;
  }
  button {
    cursor: pointer;
    border: none;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    font-weight: 700;
  }
  .btn-primary {
    background: linear-gradient(135deg, var(--accent-cyan), #00aaff);
    color: #030712;
    box-shadow: 0 4px 12px rgba(0, 242, 254, 0.3);
  }
  .btn-secondary {
    background: #1e293b;
    color: var(--text-muted);
    border: 1px solid var(--card-border);
  }
  .btn-secondary:hover { color: #fff; }

  /* Success Card */
  .registered-card {
    display: none;
    background: linear-gradient(135deg, rgba(0, 242, 254, 0.08), rgba(157, 78, 221, 0.08));
    border: 1px solid rgba(0, 242, 254, 0.4);
    border-radius: 10px;
    padding: 12px;
    margin-top: 4px;
  }
  .reg-tag {
    font-size: 28px;
    font-weight: 900;
    color: var(--accent-cyan);
    letter-spacing: 1px;
    text-shadow: 0 0 14px rgba(0, 242, 254, 0.4);
  }

  /* Right Side: Roster */
  .search-bar {
    width: 100%;
    margin-bottom: 10px;
    background: #090d16;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 9px 12px;
    color: #fff;
    font-size: 13px;
  }
  .roster-table {
    width: 100%;
    border-collapse: collapse;
  }
  .roster-table th {
    background: #090d16;
    padding: 9px 12px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--text-muted);
    text-align: left;
    border-bottom: 1px solid var(--card-border);
  }
  .roster-table td {
    padding: 8px 12px;
    font-size: 13px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
  }
  .thumb-img {
    width: 32px;
    height: 32px;
    border-radius: 6px;
    object-fit: cover;
    background: #1e293b;
  }
  .thumb-placeholder {
    width: 32px;
    height: 32px;
    border-radius: 6px;
    background: #1e293b;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    color: var(--text-muted);
  }
  .pid-badge {
    background: #1e293b;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 12px;
    color: var(--accent-cyan);
  }
  .roll-badge {
    background: rgba(157, 78, 221, 0.15);
    border: 1px solid rgba(157, 78, 221, 0.3);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 11px;
    color: #d8b4fe;
  }

</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="brand-text">DODGE &amp; DRIP</div>
  </div>
  <div>
    <a href="/" class="nav-link">LEADERBOARD</a>
  </div>
</header>

<main>
  <!-- Left Side: Registration Form -->
  <section class="panel" id="left-panel" style="width: 420px; flex-shrink: 0; min-width: 280px;">
    <div class="panel-header">
      <div class="panel-title">Player Registration</div>
    </div>
    <div class="panel-content">
      <div class="form-group">
        <label for="player-name">PLAYER NAME *</label>
        <input type="text" id="player-name" placeholder="e.g. Ravi" autocomplete="off" autofocus />
      </div>

      <div class="form-row">
        <div class="form-group">
          <label for="player-roll">URN</label>
          <input type="text" id="player-roll" placeholder="e.g. 24303192" autocomplete="off" />
        </div>
        <div class="form-group">
          <label for="player-height">HEIGHT (CM)</label>
          <input type="number" id="player-height" placeholder="e.g. 153" min="100" max="230" />
        </div>
      </div>

      <div class="form-group">
        <label>Live Webcam Photo</label>
        <div class="camera-container">
          <div class="camera-badge">Live Webcam</div>
          <video id="webcam" autoplay playsinline muted></video>
          <div id="cam-placeholder" class="camera-placeholder" style="display:none;">
            Webcam access disabled or unavailable.<br>Player can still be registered without a photo.
          </div>
        </div>
      </div>

      <div class="btn-group">
        <button id="btn-snap-register" class="btn-primary" type="button">Snap &amp; Register Player</button>
        <button id="btn-skip-photo" class="btn-secondary" type="button">Register Without Photo</button>
      </div>

      <div id="reg-success" class="registered-card">
        <div style="font-size: 11px; font-weight: 700; color: var(--accent-green);">✓ ASSIGNED PLAYER ID</div>
        <div class="reg-tag" id="reg-pid">P001</div>
        <div style="font-size: 13px; color: #fff;" id="reg-name">Name: Ravi</div>
        <div style="font-size: 12px; color: var(--text-muted);" id="reg-roll">URN: 24303192</div>
        <div style="font-size: 11px; color: var(--accent-cyan); margin-top: 6px;">
          Send player to Station A or Station B with this ID!
        </div>
      </div>

      <div id="reg-warning" class="registered-card" style="display:none; border-color: rgba(239, 68, 68, 0.6); background: rgba(239, 68, 68, 0.1);">
        <div style="font-size: 11px; font-weight: 800; color: #ef4444; letter-spacing: 0.5px;">⚠️ ALREADY REGISTERED</div>
        <div class="reg-tag" id="warn-pid" style="color: #ef4444; text-shadow: 0 0 14px rgba(239, 68, 68, 0.4);">P001</div>
        <div style="font-size: 13px; color: #fff; font-weight: 700;" id="warn-name">Name: Drishti</div>
        <div style="font-size: 12px; color: #fca5a5; margin-top: 2px;" id="warn-detail">URN 24303192 already registered!</div>
        <div style="font-size: 11px; color: #fde047; margin-top: 6px;">
          Give player their existing Player ID.
        </div>
      </div>
    </div>
  </section>

  <!-- Interactive Drag-to-Resize Divider -->
  <div id="drag-divider" class="drag-divider" title="Drag to resize / Double-click to reset">
    <div class="drag-handle"></div>
  </div>

  <!-- Right Side: Roster Table -->
  <section class="panel" id="right-panel" style="flex: 1; min-width: 280px;">
    <div class="panel-header">
      <div class="panel-title">
        <span>Registered Participants</span>
      </div>
      <span id="player-count" style="font-size: 11px; color: var(--accent-cyan); font-weight: 700;">Total: 0</span>
    </div>
    <div class="panel-content" style="padding: 14px;">
      <input type="text" id="roster-search" class="search-bar" placeholder="Search by Name, URN, or Player ID (e.g. P001, 24303192)..." oninput="filterRoster()" />
      <div style="flex: 1; overflow-y: auto;">
        <table class="roster-table">
          <thead>
            <tr>
              <th style="width: 44px;">Photo</th>
              <th>Player ID</th>
              <th>Name</th>
              <th>URN</th>
              <th>Height</th>
            </tr>
          </thead>
          <tbody id="roster-body">
            <!-- Dynamically populated -->
          </tbody>
        </table>
        <div id="roster-empty" style="text-align: center; padding: 40px; color: var(--text-muted); font-size: 13px; display: none;">
          No participants match your query.
        </div>
      </div>
    </div>
  </section>
</main>

<canvas id="snapshot-canvas" style="display:none;"></canvas>

<script>
  let videoStream = null;
  const videoEl = document.getElementById('webcam');
  const canvasEl = document.getElementById('snapshot-canvas');
  const camPlaceholder = document.getElementById('cam-placeholder');
  let allPlayers = [];

  async function initCamera() {
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        videoStream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
          audio: false
        });
        videoEl.srcObject = videoStream;
        camPlaceholder.style.display = 'none';
      } else {
        camPlaceholder.style.display = 'block';
      }
    } catch (err) {
      camPlaceholder.style.display = 'block';
    }
  }

  function capturePhotoBase64() {
    if (!videoStream || !videoEl.videoWidth) return null;
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
    const ctx = canvasEl.getContext('2d');
    ctx.translate(canvasEl.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
    return canvasEl.toDataURL('image/jpeg', 0.85);
  }

  async function handleRegister(withPhoto) {
    const nameInput = document.getElementById('player-name');
    const rollInput = document.getElementById('player-roll');
    const heightInput = document.getElementById('player-height');

    const name = nameInput.value.trim();
    if (!name) {
      alert("Please enter a player name!");
      nameInput.focus();
      return;
    }

    const roll_no = rollInput.value.trim() || null;
    const height_cm = heightInput.value ? parseFloat(heightInput.value) : null;
    const photo_base64 = withPhoto ? capturePhotoBase64() : null;

    try {
      const res = await fetch('/api/players', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, roll_no, height_cm, photo_base64 })
      });

      if (res.status === 409) {
        const errData = await res.json();
        const successCard = document.getElementById('reg-success');
        if (successCard) successCard.style.display = 'none';

        const warnCard = document.getElementById('reg-warning');
        if (warnCard) {
          document.getElementById('warn-pid').textContent = errData.existing_player_id || '';
          document.getElementById('warn-name').textContent = 'Name: ' + (errData.existing_name || '');
          document.getElementById('warn-detail').textContent = errData.detail || 'URN already registered';
          warnCard.style.display = 'block';
        }

        const searchInput = document.getElementById('roster-search');
        if (searchInput) {
          searchInput.value = errData.existing_player_id || roll_no || '';
          filterRoster();
        }
        return;
      }

      if (!res.ok) throw new Error("Registration failed: " + res.statusText);
      const data = await res.json();

      const warnCard = document.getElementById('reg-warning');
      if (warnCard) warnCard.style.display = 'none';

      document.getElementById('reg-pid').textContent = data.player_id;
      document.getElementById('reg-name').textContent = 'Name: ' + data.name;
      document.getElementById('reg-roll').textContent = 'URN: ' + (data.roll_no || 'None');
      document.getElementById('reg-success').style.display = 'block';

      nameInput.value = '';
      rollInput.value = '';
      heightInput.value = '';
      nameInput.focus();

      loadRoster();
    } catch (err) {
      alert("Error registering player: " + err.message);
    }
  }

  document.getElementById('btn-snap-register').addEventListener('click', () => handleRegister(true));
  document.getElementById('btn-skip-photo').addEventListener('click', () => handleRegister(false));

  async function loadRoster() {
    try {
      const res = await fetch('/api/players');
      if (res.ok) {
        allPlayers = await res.json();
        document.getElementById('player-count').textContent = 'Total: ' + allPlayers.length;
        filterRoster();
      }
    } catch (e) {
      // offline
    }
  }

  function filterRoster() {
    const query = document.getElementById('roster-search').value.toLowerCase().trim();
    const tbody = document.getElementById('roster-body');
    const empty = document.getElementById('roster-empty');

    const filtered = allPlayers.filter(p => {
      const name = (p.name || '').toLowerCase();
      const pid = (p.player_id || '').toLowerCase();
      const roll = (p.roll_no || '').toLowerCase();
      return name.includes(query) || pid.includes(query) || roll.includes(query);
    });

    if (filtered.length === 0) {
      tbody.innerHTML = '';
      empty.style.display = 'block';
      return;
    }

    empty.style.display = 'none';
    tbody.innerHTML = filtered.map(p => {
      const photoHtml = p.photo_url
        ? `<img src="${p.photo_url}" class="thumb-img" alt="" />`
        : `<div class="thumb-placeholder">👤</div>`;
      const rollHtml = p.roll_no
        ? `<span class="roll-badge">${p.roll_no}</span>`
        : `<span style="color: var(--text-muted);">-</span>`;
      const heightHtml = p.height_cm ? `${p.height_cm} cm` : `<span style="color: var(--text-muted);">-</span>`;

      return `<tr>
        <td>${photoHtml}</td>
        <td><span class="pid-badge">${p.player_id}</span></td>
        <td style="font-weight: 600;">${p.name}</td>
        <td>${rollHtml}</td>
        <td>${heightHtml}</td>
      </tr>`;
    }).join('');
  }

  initCamera();
  loadRoster();
  setInterval(loadRoster, 4000);

  // Interactive Drag-to-Resize Divider Logic
  const divider = document.getElementById('drag-divider');
  const leftPanel = document.getElementById('left-panel');
  const mainContainer = document.querySelector('main');
  const DEFAULT_WIDTH = 420;
  const MIN_PANEL_WIDTH = 280;
  const STORAGE_KEY = 'dodge_register_split_width';

  function setLeftWidth(widthPx) {
    if (!mainContainer || !leftPanel) return;
    const mainRect = mainContainer.getBoundingClientRect();
    const maxAllowed = mainRect.width - MIN_PANEL_WIDTH - 24;
    const clamped = Math.max(MIN_PANEL_WIDTH, Math.min(widthPx, maxAllowed));
    leftPanel.style.width = clamped + 'px';
    try {
      localStorage.setItem(STORAGE_KEY, clamped.toString());
    } catch (e) {}
  }

  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const parsed = parseInt(saved, 10);
      if (!isNaN(parsed)) setLeftWidth(parsed);
    }
  } catch (e) {}

  let isDragging = false;

  divider.addEventListener('pointerdown', (e) => {
    isDragging = true;
    divider.classList.add('dragging');
    divider.setPointerCapture(e.pointerId);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });

  divider.addEventListener('pointermove', (e) => {
    if (!isDragging) return;
    const mainRect = mainContainer.getBoundingClientRect();
    const newWidth = e.clientX - mainRect.left;
    setLeftWidth(newWidth);
  });

  function stopDragging(e) {
    if (isDragging) {
      isDragging = false;
      divider.classList.remove('dragging');
      try {
        if (e && e.pointerId) divider.releasePointerCapture(e.pointerId);
      } catch (err) {}
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }
  }

  divider.addEventListener('pointerup', stopDragging);
  divider.addEventListener('pointercancel', stopDragging);
  window.addEventListener('pointerup', stopDragging);

  // Double-click to reset to default
  divider.addEventListener('dblclick', () => {
    setLeftWidth(DEFAULT_WIDTH);
  });
</script>
</body>
</html>"""
