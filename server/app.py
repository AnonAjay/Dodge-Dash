"""Laptop 3: score server. Run:  uvicorn server.app:app --host 0.0.0.0 --port 8000
Minimal on purpose: health, accept score (idempotent), leaderboard, player registration.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
import socket
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cards.generator import generate_card
from cards.qr import generate_qr_png_bytes
from cards.uploader import (
    get_cached_card_url,
    get_or_upload_public_card_url,
    invalidate_card_url,
    upload_card_if_online,
)

DATA = Path("data")
SCORES_FILE = DATA / "scores.jsonl"
PLAYERS_FILE = DATA / "players.json"
PHOTOS_DIR = DATA / "photos"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Titans' Arcade server")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_lock = threading.Lock()
_seen_event_ids: set[str] = set()

OPERATOR_PIN = os.environ.get("OPERATOR_PIN", "2026")
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def _is_operator_authorized(request: Request) -> bool:
    client_host = request.client.host if request.client else ""
    if client_host in LOCAL_HOSTS:
        return True
    pin = request.query_params.get("pin") or request.headers.get("x-operator-pin")
    if pin and pin.strip() == os.environ.get("OPERATOR_PIN", "2026"):
        return True
    return False


def _check_operator_auth(request: Request) -> None:
    if not _is_operator_authorized(request):
        raise HTTPException(status_code=403, detail="Forbidden: Operator PIN required")


def _bg_process_card_upload(player_id: str) -> None:
    if os.environ.get("DISABLE_PUBLIC_UPLOAD") == "1":
        return
    try:
        invalidate_card_url(player_id, data_dir=DATA)
        card_path = generate_card(player_id, data_dir=DATA, output_dir=DATA / "cards")
        get_or_upload_public_card_url(card_path, player_id, data_dir=DATA)
    except Exception:
        pass


def _trigger_bg_upload(player_id: str) -> None:
    if os.environ.get("DISABLE_PUBLIC_UPLOAD") == "1":
        return
    threading.Thread(target=_bg_process_card_upload, args=(player_id,), daemon=True).start()


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
def register_player(player_in: PlayerCreate, request: Request) -> Any:
    _check_operator_auth(request)
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
    _trigger_bg_upload(player_id)
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
    _trigger_bg_upload(event.player_id)
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


def _get_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def _resolve_base_url(request: Request) -> str:
    host = request.headers.get("host", "").strip()
    if host == "testserver":
        return "http://testserver"
    if host and not host.startswith("localhost") and not host.startswith("127.0.0.1"):
        return f"http://{host}"
    lan_ip = _get_lan_ip()
    port = ""
    if ":" in host:
        port = ":" + host.split(":", 1)[1]
    else:
        port = ":8000"
    return f"http://{lan_ip}{port}"


@app.get("/api/cards/{player_id}")
def get_player_card(player_id: str, request: Request) -> FileResponse:
    with _lock:
        players = _read_players_locked()
        found = any(p.get("player_id") == player_id for p in players)
    if not found:
        raise HTTPException(status_code=404, detail="Player not found")

    base_url = _resolve_base_url(request)
    try:
        card_path = generate_card(player_id, data_dir=DATA, output_dir=DATA / "cards", base_url=base_url)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return FileResponse(
        card_path,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="TitansArcade_{player_id}.png"'},
    )


@app.get("/api/cards/{player_id}/download")
def download_player_card(player_id: str, request: Request) -> FileResponse:
    with _lock:
        players = _read_players_locked()
        found = any(p.get("player_id") == player_id for p in players)
    if not found:
        raise HTTPException(status_code=404, detail="Player not found")

    base_url = _resolve_base_url(request)
    try:
        card_path = generate_card(player_id, data_dir=DATA, output_dir=DATA / "cards", base_url=base_url)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return FileResponse(
        card_path,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="TitansArcade_{player_id}.png"'},
    )


def _resolve_card_share_data(player_id: str, request: Request | None = None) -> tuple[str, bool]:
    """Ensure data/cards/{player_id}.png is generated and resolve the direct download URL.
    Returns (url, is_public).
    """
    base_url = _resolve_base_url(request) if request is not None else f"http://{_get_lan_ip()}:8000"
    card_path = generate_card(player_id, data_dir=DATA, output_dir=DATA / "cards", base_url=base_url)

    public_base = os.environ.get("PUBLIC_CARD_BASE_URL", "").strip()
    if public_base:
        return f"{public_base.rstrip('/')}/{player_id}.png", True

    uploaded_url = get_or_upload_public_card_url(card_path, player_id, data_dir=DATA)
    if uploaded_url and (uploaded_url.startswith("https://") or uploaded_url.startswith("http://")):
        return uploaded_url, True

    return f"{base_url.rstrip('/')}/api/cards/{player_id}/download", False


@app.get("/api/qr/{player_id}")
def get_player_qr(player_id: str, request: Request) -> Response:
    with _lock:
        players = _read_players_locked()
        found = any(p.get("player_id") == player_id for p in players)
    if not found:
        raise HTTPException(status_code=404, detail="Player not found")

    card_url, _ = _resolve_card_share_data(player_id, request=request)
    png_bytes = generate_qr_png_bytes(card_url, box_size=8, border=3)
    return Response(content=png_bytes, media_type="image/png")


@app.get("/api/cards/{player_id}/share-info")
def get_card_share_info(player_id: str, request: Request) -> dict[str, Any]:
    with _lock:
        players = _read_players_locked()
        found = any(p.get("player_id") == player_id for p in players)
    if not found:
        raise HTTPException(status_code=404, detail="Player not found")

    card_url, is_public = _resolve_card_share_data(player_id, request=request)
    mode = "PUBLIC CLOUD MODE" if is_public else "LOCAL LAN OFFLINE MODE"
    return {
        "player_id": player_id,
        "mode": mode,
        "is_public": is_public,
        "url": card_url,
        "qr_url": f"/api/qr/{player_id}",
        "card_url": f"/api/cards/{player_id}",
        "download_url": f"/api/cards/{player_id}/download",
    }


@app.get("/card/{player_id}", response_class=HTMLResponse)
def souvenir_card_page(player_id: str) -> HTMLResponse:
    with _lock:
        players = _read_players_locked()
        player = next((p for p in players if p.get("player_id") == player_id), None)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    name = player.get("name", "Challenger")
    roll_no = player.get("roll_no")
    roll_display = f" | URN: {roll_no}" if roll_no else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Titans' Arcade - Souvenir Card - {player_id}</title>
<style>
  :root {{
    --twilight-dark: #35235d;
    --twilight-purple: #48327a;
    --sunset-pink: #ea7d8b;
    --sunset-gold: #ffd152;
    --retro-green: #58c038;
    --neon-cyan: #00f0ff;
    --font-pixel: "Consolas", "Courier New", monospace;
    --font-retro: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    color: #f8fafc;
    font-family: var(--font-retro);
    background-color: #2c1b4e;
    background-image:
      url('/static/bg-sunset.png'),
      linear-gradient(180deg, #2b1e4a 0%, #5c3787 42%, #d97398 72%, #fca289 84%, #4ea832 84%, #5c3a21 90%);
    background-size: cover, cover;
    background-position: center bottom, center bottom;
    background-attachment: fixed, fixed;
    padding: 16px 12px 32px 12px;
  }}
  .scanlines {{
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 50;
    background: linear-gradient(
      rgba(72, 50, 122, 0) 50%,
      rgba(35, 18, 65, 0.22) 50%
    );
    background-size: 100% 4px;
  }}
  .card-container {{
    width: 100%;
    max-width: 520px;
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
    z-index: 10;
  }}
  header {{
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 4px 16px 4px;
  }}
  .brand-title {{
    font-family: var(--font-pixel);
    font-size: 18px;
    font-weight: 900;
    letter-spacing: 2px;
    color: var(--sunset-gold);
    text-shadow: 0 0 10px rgba(255, 209, 82, 0.85);
  }}
  .nav-btn {{
    font-family: var(--font-pixel);
    font-size: 11px;
    font-weight: 700;
    color: #fbcfe8;
    background: rgba(53, 35, 93, 0.9);
    border: 1px solid var(--sunset-pink);
    padding: 6px 12px;
    border-radius: 4px;
    text-decoration: none;
    letter-spacing: 1px;
  }}
  .player-tag {{
    font-family: var(--font-pixel);
    font-size: 13px;
    font-weight: 700;
    color: var(--neon-cyan);
    margin-bottom: 12px;
    text-align: center;
  }}
  .card-preview {{
    width: 100%;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 12px 36px rgba(18, 9, 32, 0.8), 0 0 20px rgba(234, 125, 139, 0.35);
    border: 2px solid var(--sunset-pink);
    background: #1e1038;
  }}
  .card-preview img {{
    width: 100%;
    height: auto;
    display: block;
  }}
  .action-box {{
    width: 100%;
    margin-top: 18px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }}
  .btn-download {{
    display: block;
    width: 100%;
    padding: 14px 20px;
    background: linear-gradient(135deg, #ea7d8b 0%, #fca289 100%);
    color: #1f0b2e;
    font-family: var(--font-pixel);
    font-size: 15px;
    font-weight: 900;
    text-align: center;
    text-decoration: none;
    letter-spacing: 1px;
    border-radius: 6px;
    border: none;
    box-shadow: 0 4px 14px rgba(234, 125, 139, 0.6);
  }}
  .btn-download:active {{
    transform: translateY(2px);
  }}
  .card-footer-info {{
    font-family: var(--font-pixel);
    font-size: 11px;
    color: #d8b4fe;
    text-align: center;
    margin-top: 8px;
    line-height: 1.5;
  }}
</style>
</head>
<body>
<div class="scanlines"></div>
<div class="card-container">
  <header>
    <div class="brand-title">TITANS' ARCADE</div>
    <a href="/" class="nav-btn">LEADERBOARD</a>
  </header>
  <div class="player-tag">★ SOUVENIR CARD: {name} ({player_id}){roll_display} ★</div>
  <div class="card-preview">
    <img src="/api/cards/{player_id}" alt="Titans' Arcade Souvenir Card - {player_id}" />
  </div>
  <div class="action-box">
    <a href="/api/cards/{player_id}" download="TitansArcade_{player_id}.png" class="btn-download">
      ⬇ DOWNLOAD SOUVENIR CARD
    </a>
    <div class="card-footer-info">
      Full 1080×1350 High Resolution PNG • 100% Offline Local Network<br>
      Saved directly to your device • Share with friends!
    </div>
  </div>
</div>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/", response_class=HTMLResponse)
def big_screen_display() -> str:
    """Public Big-Screen Spectator Display for Projector / TV (F11 fullscreen)."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Titans' Arcade - Leaderboard</title>
<style>
  :root {
    --twilight-dark: #35235d;
    --twilight-purple: #48327a;
    --twilight-lilac: #9b74b8;
    --sunset-pink: #ea7d8b;
    --sunset-peach: #fca289;
    --sunset-gold: #ffd152;
    --retro-green: #58c038;
    --neon-cyan: #00f0ff;
    --font-pixel: "Consolas", "Courier New", monospace;
    --font-retro: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    min-height: 100vh;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    user-select: none;
    color: #f8fafc;
    font-family: var(--font-retro);
    background-color: #2c1b4e;
    background-image:
      url('/static/bg-sunset.png'),
      linear-gradient(180deg, #2b1e4a 0%, #5c3787 42%, #d97398 72%, #fca289 84%, #4ea832 84%, #5c3a21 90%);
    background-size: cover, cover;
    background-position: center bottom, center bottom;
    background-repeat: no-repeat, no-repeat;
    background-attachment: fixed, fixed;
    image-rendering: pixelated;
  }

  /* CRT Scanline Overlay */
  .scanlines {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 50;
    background: linear-gradient(
      rgba(72, 50, 122, 0) 50%,
      rgba(35, 18, 65, 0.24) 50%
    );
    background-size: 100% 4px;
  }

  /* Header */
  header {
    width: 100%;
    height: 64px;
    border-bottom: 1px solid rgba(216, 180, 254, 0.3);
    padding: 0 28px;
    z-index: 10;
    flex-shrink: 0;
    background-color: rgba(53, 35, 93, 0.84);
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(35, 18, 65, 0.45);
    display: flex;
    align-items: center;
  }
  .header-inner {
    width: 100%;
    max-width: 1400px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .brand-title {
    font-family: var(--font-pixel);
    font-size: 22px;
    font-weight: 900;
    letter-spacing: 2px;
    color: var(--sunset-gold);
    text-transform: uppercase;
    text-shadow: 0 0 12px rgba(255, 209, 82, 0.85), 2px 2px 0 #241442;
  }
  .header-right {
    display: flex;
    align-items: center;
    gap: 20px;
  }
  .live-status {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .live-dot {
    width: 10px;
    height: 10px;
    background: var(--retro-green);
    box-shadow: 0 0 8px var(--retro-green);
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse { 0% { opacity: 0.55; } 50% { opacity: 1; } 100% { opacity: 0.55; } }
  .live-label {
    font-family: var(--font-pixel);
    font-size: 12px;
    font-weight: 700;
    color: #fbcfe8;
    letter-spacing: 1.5px;
  }
  .v-sep {
    height: 18px;
    width: 1px;
    background: rgba(192, 132, 252, 0.4);
  }
  .nav-btn {
    text-decoration: none;
    background: rgba(72, 50, 122, 0.92);
    color: #fef9c3;
    font-family: var(--font-pixel);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    padding: 8px 14px;
    text-transform: uppercase;
    box-shadow:
      inset 0 0 0 2px rgba(45, 25, 75, 0.9),
      0 0 0 2px var(--twilight-lilac);
    transition: all 0.12s;
  }
  .nav-btn:hover {
    background: #5c3e9b;
    color: #fff;
    box-shadow:
      inset 0 0 0 2px rgba(45, 25, 75, 0.9),
      0 0 0 2px var(--sunset-pink),
      0 0 10px rgba(234, 125, 139, 0.45);
  }

  /* Main Container */
  main {
    flex: 1;
    width: 100%;
    max-width: 1400px;
    margin: 0 auto;
    padding: 22px 28px 32px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    z-index: 10;
  }
  .pixel-panel {
    flex: 1;
    width: 100%;
    border-radius: 6px;
    padding: 22px 28px;
    background-color: rgba(53, 35, 93, 0.86);
    backdrop-filter: blur(10px);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow:
      0 0 0 2px var(--sunset-pink),
      0 0 0 4px var(--twilight-purple),
      0 0 0 6px #241442,
      0 15px 35px rgba(25, 10, 48, 0.65),
      0 0 30px rgba(234, 125, 139, 0.28);
  }

  /* Panel Top Bar */
  .panel-top {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    padding-bottom: 20px;
    gap: 16px;
    flex-shrink: 0;
  }
  .panel-title {
    font-family: var(--font-pixel);
    font-size: 26px;
    font-weight: 900;
    color: #ffffff;
    letter-spacing: 0.25em;
    text-align: center;
    text-shadow: 0 0 12px rgba(234, 125, 139, 0.85), 2px 2px 0 #241442;
  }
  .game-modes {
    display: flex;
    align-items: center;
    gap: 10px;
    justify-self: end;
  }
  .tab-btn {
    cursor: pointer;
    border: none;
    background: rgba(72, 50, 122, 0.85);
    color: #e9d5ff;
    font-family: var(--font-pixel);
    font-size: 12px;
    font-weight: 700;
    padding: 9px 14px;
    letter-spacing: 1px;
    text-transform: uppercase;
    box-shadow:
      inset 0 0 0 2px rgba(45, 25, 75, 0.9),
      0 0 0 2px var(--twilight-lilac);
    transition: all 0.12s;
  }
  .tab-btn:hover {
    color: #fff;
    box-shadow:
      inset 0 0 0 2px rgba(45, 25, 75, 0.9),
      0 0 0 2px var(--sunset-pink),
      0 0 10px rgba(234, 125, 139, 0.45);
  }
  .tab-btn.active {
    background: var(--retro-green);
    color: #0f172a;
    font-weight: 900;
    box-shadow:
      inset 0 0 0 2px #fff,
      0 0 0 2px var(--retro-green),
      0 0 14px rgba(88, 192, 56, 0.7);
  }

  /* Table Container */
  .table-wrap {
    flex: 1;
    width: 100%;
    overflow-y: auto;
    padding-right: 4px;
  }
  .lb-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0 10px;
  }
  .lb-table th {
    font-family: var(--font-pixel);
    font-size: 13px;
    font-weight: 800;
    color: rgba(251, 207, 232, 0.95);
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 12px 18px;
    text-align: left;
    background: rgba(61, 39, 107, 0.8);
    border-bottom: 1px solid rgba(249, 168, 212, 0.35);
  }
  .lb-table td {
    padding: 15px 18px;
    font-size: 16px;
    font-weight: 700;
    background: rgba(72, 50, 122, 0.68);
    border-top: 1px solid rgba(234, 125, 139, 0.22);
    border-bottom: 1px solid rgba(234, 125, 139, 0.22);
    transition: filter 0.12s;
  }
  .lb-table tbody tr:hover td {
    filter: brightness(1.12);
  }
  .lb-table tr td:first-child {
    border-left: 4px solid var(--twilight-lilac);
    border-top-left-radius: 3px;
    border-bottom-left-radius: 3px;
    text-align: center;
    width: 100px;
    font-family: var(--font-pixel);
    font-size: 15px;
    color: #e9d5ff;
  }
  .lb-table tr td:last-child {
    border-right: 1px solid rgba(234, 125, 139, 0.22);
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
  }

  /* Gold / Silver / Bronze Podium Rows */
  .rank-1 td {
    background: linear-gradient(90deg, rgba(255, 209, 82, 0.32) 0%, rgba(72, 50, 122, 0.78) 45%, rgba(53, 35, 93, 0.88) 100%);
    border-top-color: rgba(255, 209, 82, 0.45);
    border-bottom-color: rgba(255, 209, 82, 0.45);
  }
  .rank-1 td:first-child {
    border-left: 4px solid var(--sunset-gold);
    color: var(--sunset-gold);
    text-shadow: 0 0 8px rgba(255, 209, 82, 0.9);
  }
  .rank-1 .name-cell {
    color: var(--sunset-gold);
    text-shadow: 0 0 6px rgba(255, 209, 82, 0.6);
  }
  .rank-1 .stat-cell { color: var(--sunset-gold); }

  .rank-2 td {
    background: linear-gradient(90deg, rgba(226, 232, 240, 0.25) 0%, rgba(72, 50, 122, 0.78) 45%, rgba(53, 35, 93, 0.88) 100%);
    border-top-color: rgba(226, 232, 240, 0.35);
    border-bottom-color: rgba(226, 232, 240, 0.35);
  }
  .rank-2 td:first-child {
    border-left: 4px solid #e2e8f0;
    color: #ffffff;
    text-shadow: 0 0 8px rgba(255, 255, 255, 0.8);
  }
  .rank-2 .name-cell { color: #ffffff; }
  .rank-2 .stat-cell { color: #f3e8ff; }

  .rank-3 td {
    background: linear-gradient(90deg, rgba(252, 162, 137, 0.28) 0%, rgba(72, 50, 122, 0.78) 45%, rgba(53, 35, 93, 0.88) 100%);
    border-top-color: rgba(252, 162, 137, 0.4);
    border-bottom-color: rgba(252, 162, 137, 0.4);
  }
  .rank-3 td:first-child {
    border-left: 4px solid var(--sunset-peach);
    color: var(--sunset-peach);
    text-shadow: 0 0 8px rgba(252, 162, 137, 0.8);
  }
  .rank-3 .name-cell { color: var(--sunset-peach); }
  .rank-3 .stat-cell { color: var(--sunset-peach); }

  .pid-tag {
    display: inline-block;
    font-family: var(--font-pixel);
    font-size: 12px;
    background: var(--twilight-dark);
    color: #fef08a;
    padding: 4px 9px;
    border: 1px solid rgba(255, 209, 82, 0.55);
  }
  .name-cell {
    font-family: var(--font-pixel);
    font-size: 15px;
    color: #f3e8ff;
    letter-spacing: 0.5px;
  }
  .score-val {
    font-family: var(--font-pixel);
    font-size: 18px;
    font-weight: 900;
    color: var(--neon-cyan);
    letter-spacing: 1px;
    text-shadow: 0 0 8px rgba(0, 240, 255, 0.75);
  }
  .stat-cell {
    font-family: var(--font-pixel);
    font-size: 14px;
    color: #e9d5ff;
    text-align: center;
  }
  .empty-board {
    text-align: center;
    padding: 110px 20px;
    font-family: var(--font-pixel);
    font-size: 14px;
    line-height: 1.9;
    color: #fbcfe8;
  }
</style>
</head>
<body>
<div class="scanlines"></div>

<header>
  <div class="header-inner">
    <div class="brand-title">TITANS' ARCADE</div>
    <div class="header-right">
      <div class="live-status">
        <span class="live-dot"></span>
        <span class="live-label">LIVE RESULTS</span>
      </div>
      <div class="v-sep"></div>
      <a href="/register" class="nav-btn">REGISTRATION PANEL</a>
    </div>
  </div>
</header>

<main>
  <section class="pixel-panel">
    <div class="panel-top">
      <div></div>
      <h2 class="panel-title">LEADERBOARD</h2>
      <div class="game-modes">
        <button class="tab-btn active" data-game="dodge" onclick="switchTab('dodge')">FLOOR IS LAVA</button>
        <button class="tab-btn" data-game="wall" onclick="switchTab('wall')">WALL SHAPE</button>
        <button class="tab-btn" data-game="dance" onclick="switchTab('dance')">DANCE</button>
      </div>
    </div>

    <div class="table-wrap">
      <table class="lb-table" id="lb-table">
        <thead>
          <tr>
            <th style="width: 100px; text-align: center;">RANK</th>
            <th style="width: 140px; text-align: center;">PLAYER ID</th>
            <th>PLAYER NAME</th>
            <th style="width: 170px;">SCORE</th>
            <th style="width: 150px; text-align: center;">ACCURACY</th>
            <th style="width: 160px; text-align: center;">BEST STREAK</th>
          </tr>
        </thead>
        <tbody id="lb-body">
          <!-- Dynamically populated -->
        </tbody>
      </table>
      <div id="lb-empty" class="empty-board" style="display:none;">
        NO SCORES RECORDED YET FOR THIS EVENT.<br>STEP ON THE ARENA TILES AND CLAIM THE TOP RANK!
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
      let rankHtml = '#' + rank;
      let rowClass = '';
      if (rank === 1) { rankHtml = '★ 1'; rowClass = 'rank-1'; }
      else if (rank === 2) { rankHtml = '★ 2'; rowClass = 'rank-2'; }
      else if (rank === 3) { rankHtml = '★ 3'; rowClass = 'rank-3'; }

      const acc = r.accuracy != null ? (r.accuracy * 100).toFixed(0) + '%' : '-';
      const streak = r.best_streak ?? '-';

      return `<tr class="${rowClass}">
        <td>${rankHtml}</td>
        <td style="text-align: center;"><span class="pid-tag">${r.player_id}</span></td>
        <td class="name-cell">${r.name || r.player_id}</td>
        <td class="score-val">${r.score}</td>
        <td class="stat-cell">${acc}</td>
        <td class="stat-cell">${streak}</td>
      </tr>`;
    }).join('');
  }

  fetchLeaderboard();
  setInterval(fetchLeaderboard, 2000);
</script>
</body>
</html>"""


@app.get("/register", response_class=HTMLResponse)
def operator_desk(request: Request) -> str:
    """Operator Check-In & Roster Desk for Laptop 3 operator."""
    _check_operator_auth(request)
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Titans' Arcade - Player Registration</title>
<style>
  :root {
    --twilight-dark: #35235d;
    --twilight-purple: #48327a;
    --twilight-lilac: #9b74b8;
    --sunset-pink: #ea7d8b;
    --sunset-peach: #fca289;
    --sunset-gold: #ffd152;
    --retro-green: #58c038;
    --neon-cyan: #00f0ff;
    --input-bg: #241645;
    --font-pixel: "Consolas", "Courier New", monospace;
    --font-retro: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    user-select: none;
    color: #f8fafc;
    font-family: var(--font-retro);
    background-color: #2c1b4e;
    background-image:
      url('/static/bg-sunset.png'),
      linear-gradient(180deg, #2b1e4a 0%, #5c3787 42%, #d97398 72%, #fca289 84%, #4ea832 84%, #5c3a21 90%);
    background-size: cover, cover;
    background-position: center bottom, center bottom;
    background-repeat: no-repeat, no-repeat;
    background-attachment: fixed, fixed;
    image-rendering: pixelated;
  }

  /* CRT Scanline Overlay */
  .scanlines {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 50;
    background: linear-gradient(
      rgba(72, 50, 122, 0) 50%,
      rgba(35, 18, 65, 0.22) 50%
    );
    background-size: 100% 4px;
  }

  /* Custom Scrollbar */
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: #251642; }
  ::-webkit-scrollbar-thumb { background: #48327a; border: 1px solid #9b74b8; }
  ::-webkit-scrollbar-thumb:hover { background: #ea7d8b; }

  /* Header */
  header {
    width: 100%;
    height: 58px;
    border-bottom: 1px solid rgba(216, 180, 254, 0.3);
    padding: 0 24px;
    z-index: 10;
    flex-shrink: 0;
    background-color: rgba(53, 35, 93, 0.86);
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(35, 18, 65, 0.5);
    display: flex;
    align-items: center;
  }
  .header-inner {
    width: 100%;
    max-width: 1700px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .brand-title {
    font-family: var(--font-pixel);
    font-size: 20px;
    font-weight: 900;
    letter-spacing: 2px;
    color: var(--sunset-gold);
    text-transform: uppercase;
    text-shadow: 0 0 10px rgba(255, 209, 82, 0.85), 2px 2px 0 #241442;
  }
  .header-right {
    display: flex;
    align-items: center;
    gap: 18px;
  }
  .live-status {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .live-dot {
    width: 9px;
    height: 9px;
    background: var(--retro-green);
    box-shadow: 0 0 8px var(--retro-green);
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse { 0% { opacity: 0.55; } 50% { opacity: 1; } 100% { opacity: 0.55; } }
  .live-label {
    font-family: var(--font-pixel);
    font-size: 11px;
    font-weight: 700;
    color: #fbcfe8;
    letter-spacing: 1.2px;
  }
  .v-sep {
    height: 16px;
    width: 1px;
    background: rgba(192, 132, 252, 0.4);
  }
  .nav-btn {
    text-decoration: none;
    background: rgba(72, 50, 122, 0.92);
    color: #fef9c3;
    font-family: var(--font-pixel);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 7px 14px;
    text-transform: uppercase;
    box-shadow:
      inset 0 0 0 2px rgba(45, 25, 75, 0.9),
      0 0 0 2px var(--twilight-lilac);
    transition: all 0.12s;
  }
  .nav-btn:hover {
    background: #5c3e9b;
    color: #fff;
    box-shadow:
      inset 0 0 0 2px rgba(45, 25, 75, 0.9),
      0 0 0 2px var(--sunset-pink),
      0 0 10px rgba(234, 125, 139, 0.45);
  }

  /* Main Split */
  main {
    flex: 1;
    display: flex;
    flex-direction: row;
    gap: 0;
    padding: 18px 22px;
    max-width: 1700px;
    width: 100%;
    margin: 0 auto;
    overflow: hidden;
    position: relative;
    z-index: 10;
  }

  /* Interactive Drag-to-Resize Divider */
  .drag-divider {
    width: 16px;
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
    background: rgba(155, 116, 184, 0.5);
    transition: background-color 0.2s, box-shadow 0.2s, width 0.2s;
  }
  .drag-divider:hover::before,
  .drag-divider.dragging::before {
    background: var(--sunset-pink);
    box-shadow: 0 0 10px var(--sunset-pink);
    width: 4px;
  }
  .drag-handle {
    width: 6px;
    height: 36px;
    background: var(--twilight-purple);
    border: 1px solid var(--sunset-pink);
    position: absolute;
    transition: all 0.2s;
  }
  .drag-divider:hover .drag-handle,
  .drag-divider.dragging .drag-handle {
    background: var(--sunset-gold);
    border-color: #fff;
    box-shadow: 0 0 8px rgba(255, 209, 82, 0.8);
  }

  /* Sunset Arcade Panels */
  .pixel-panel {
    background-color: rgba(53, 35, 93, 0.88);
    backdrop-filter: blur(10px);
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow:
      0 0 0 2px var(--sunset-pink),
      0 0 0 4px var(--twilight-purple),
      0 0 0 6px #241442,
      0 15px 35px rgba(25, 10, 48, 0.65),
      0 0 25px rgba(234, 125, 139, 0.22);
  }
  .panel-header {
    padding: 14px 20px;
    border-bottom: 1px solid rgba(249, 168, 212, 0.3);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }
  .panel-title {
    font-family: var(--font-pixel);
    font-size: 16px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.8px;
    text-shadow: 0 0 8px rgba(234, 125, 139, 0.65);
  }
  .panel-content {
    flex: 1;
    padding: 16px 20px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 13px;
  }

  /* Form Elements */
  .form-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .form-group label {
    font-family: var(--font-pixel);
    font-size: 14px;
    font-weight: 800;
    text-transform: uppercase;
    color: #fbcfe8;
    letter-spacing: 1px;
  }
  .req-star { color: var(--neon-cyan); }
  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  .pixel-input {
    width: 100%;
    background-color: var(--input-bg);
    border: 2px solid #5a3c8c;
    box-shadow: inset 2px 2px 0 #140b29;
    border-radius: 4px;
    padding: 10px 13px;
    color: #fef9c3;
    font-family: var(--font-pixel);
    font-size: 14px;
    letter-spacing: 0.5px;
    outline: none;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  .pixel-input::placeholder { color: #a855f7; opacity: 0.75; }
  .pixel-input:focus {
    border-color: var(--sunset-pink);
    box-shadow: inset 2px 2px 0 #140b29, 0 0 10px rgba(234, 125, 139, 0.55);
  }

  /* Camera Viewfinder */
  .camera-container {
    position: relative;
    width: 100%;
    aspect-ratio: 4 / 3;
    max-height: 44vh;
    background: #1a0e30;
    border: 2px solid #5a3c8c;
    border-radius: 4px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: inset 0 0 18px rgba(0, 0, 0, 0.7);
  }
  video {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transform: scaleX(-1);
  }
  .vf-corner {
    position: absolute;
    width: 14px;
    height: 14px;
    pointer-events: none;
    z-index: 10;
    filter: drop-shadow(0 0 4px var(--neon-cyan));
  }
  .vf-tl { top: 8px; left: 8px; border-top: 2px solid var(--neon-cyan); border-left: 2px solid var(--neon-cyan); }
  .vf-tr { top: 8px; right: 8px; border-top: 2px solid var(--neon-cyan); border-right: 2px solid var(--neon-cyan); }
  .vf-bl { bottom: 8px; left: 8px; border-bottom: 2px solid var(--neon-cyan); border-left: 2px solid var(--neon-cyan); }
  .vf-br { bottom: 8px; right: 8px; border-bottom: 2px solid var(--neon-cyan); border-right: 2px solid var(--neon-cyan); }
  .vf-rec {
    position: absolute;
    top: 10px;
    left: 26px;
    z-index: 10;
    font-family: var(--font-pixel);
    font-size: 9px;
    color: #fbcfe8;
    background: rgba(43, 24, 74, 0.9);
    padding: 2px 7px;
    border: 1px solid rgba(244, 114, 182, 0.35);
    display: flex;
    align-items: center;
    gap: 5px;
    pointer-events: none;
  }
  .vf-rec-dot {
    width: 6px;
    height: 6px;
    background: var(--retro-green);
    box-shadow: 0 0 5px var(--retro-green);
    animation: pulse 1.2s infinite;
  }
  .vf-sensor {
    position: absolute;
    top: 10px;
    right: 26px;
    z-index: 10;
    font-family: var(--font-pixel);
    font-size: 9px;
    color: var(--neon-cyan);
    background: rgba(43, 24, 74, 0.9);
    padding: 2px 7px;
    border: 1px solid rgba(0, 240, 255, 0.45);
    text-shadow: 0 0 4px var(--neon-cyan);
    pointer-events: none;
  }
  .vf-guide {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: none;
    z-index: 10;
  }
  .vf-guide-box {
    width: 136px;
    height: 164px;
    border: 1px dashed rgba(0, 240, 255, 0.45);
    border-radius: 4px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    padding: 5px;
  }
  .vf-guide-top {
    font-family: var(--font-pixel);
    font-size: 8px;
    color: rgba(0, 240, 255, 0.85);
    background: rgba(26, 14, 48, 0.85);
    padding: 1px 5px;
    letter-spacing: 1px;
  }
  .vf-guide-bot {
    font-family: var(--font-pixel);
    font-size: 7px;
    color: rgba(0, 240, 255, 0.65);
  }
  .camera-placeholder {
    position: absolute;
    color: #fbcfe8;
    font-family: var(--font-pixel);
    font-size: 11px;
    line-height: 1.6;
    text-align: center;
    padding: 14px;
    background: rgba(26, 14, 48, 0.85);
    z-index: 12;
  }

  /* Retro 3D Arcade Buttons */
  .btn-group {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 2px;
  }
  button {
    cursor: pointer;
    border: none;
    border-radius: 4px;
    font-family: var(--font-pixel);
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.1s;
  }
  .btn-primary {
    background: var(--retro-green);
    color: #14260d;
    padding: 13px 16px;
    font-size: 16px;
    font-weight: 900;
    box-shadow:
      inset 0 3px 0 #a7f3d0,
      inset 0 -4px 0 #15803d,
      0 4px 0 #000,
      0 0 14px rgba(88, 192, 56, 0.55);
  }
  .btn-primary:hover { background: #68d644; }
  .btn-primary:active {
    transform: translateY(2px);
    box-shadow:
      inset 0 2px 0 #a7f3d0,
      inset 0 -2px 0 #15803d,
      0 2px 0 #000;
  }
  .btn-secondary {
    background: rgba(72, 50, 122, 0.92);
    color: #fce7f3;
    padding: 11px 16px;
    font-size: 14px;
    font-weight: 700;
    border: 1px solid rgba(192, 132, 252, 0.45);
    box-shadow:
      inset 0 2px 0 #6d4a96,
      inset 0 -3px 0 #1e0f36,
      0 3px 0 #000;
  }
  .btn-secondary:hover {
    background: #583d94;
    color: #fff;
    box-shadow:
      inset 0 2px 0 #8b5fb8,
      inset 0 -3px 0 #1e0f36,
      0 3px 0 var(--sunset-pink);
  }

  /* Assigned / Warning Status Cards */
  .registered-card {
    display: none;
    background: rgba(36, 22, 69, 0.92);
    border: 2px solid var(--neon-cyan);
    border-radius: 4px;
    padding: 12px 14px;
    margin-top: 2px;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.25);
  }
  .reg-tag {
    font-family: var(--font-pixel);
    font-size: 28px;
    font-weight: 900;
    color: var(--neon-cyan);
    letter-spacing: 2px;
    margin: 4px 0;
    text-shadow: 0 0 14px rgba(0, 240, 255, 0.6);
  }

  /* Right Side: Roster Table */
  .roster-table {
    width: 100%;
    border-collapse: collapse;
  }
  .roster-table th {
    background: rgba(61, 39, 107, 0.85);
    padding: 11px 12px;
    font-family: var(--font-pixel);
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #fbcfe8;
    text-align: left;
    border-bottom: 1px solid rgba(249, 168, 212, 0.35);
    position: sticky;
    top: 0;
    z-index: 2;
  }
  .roster-table td {
    padding: 10px 12px;
    font-size: 14px;
    border-bottom: 1px solid rgba(168, 85, 247, 0.22);
    transition: background-color 0.2s, box-shadow 0.2s;
  }
  .roster-table tbody tr:hover td {
    background: rgba(72, 50, 122, 0.55);
  }
  .roster-table tbody tr.row-highlight td {
    background: rgba(255, 209, 82, 0.22) !important;
    border-top: 1px solid var(--sunset-gold);
    border-bottom: 1px solid var(--sunset-gold);
  }
  .thumb-img {
    width: 34px;
    height: 34px;
    border-radius: 4px;
    object-fit: cover;
    background: #2b184a;
    border: 1px solid rgba(192, 132, 252, 0.45);
  }
  .thumb-placeholder {
    width: 34px;
    height: 34px;
    border-radius: 4px;
    background: #2b184a;
    border: 1px solid rgba(192, 132, 252, 0.45);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    color: #d8b4fe;
  }
  .pid-badge {
    font-family: var(--font-pixel);
    font-size: 15px;
    font-weight: 800;
    color: var(--neon-cyan);
    text-shadow: 0 0 6px rgba(0, 240, 255, 0.7);
  }
  .roll-badge {
    display: inline-block;
    background: var(--twilight-dark);
    border: 1px solid rgba(234, 125, 139, 0.5);
    padding: 4px 10px;
    border-radius: 4px;
    font-family: var(--font-pixel);
    font-size: 14px;
    color: #fbcfe8;
  }
  .height-cell {
    font-family: var(--font-pixel);
    font-size: 13px;
    color: var(--sunset-peach);
  }
  .btn-action-qr {
    background: rgba(53, 35, 93, 0.85);
    border: 1px solid var(--neon-cyan);
    color: var(--neon-cyan);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 6px;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.15s, color 0.15s, box-shadow 0.15s, transform 0.1s;
  }
  .btn-action-qr:hover {
    background: var(--neon-cyan);
    color: #1b0e33;
    box-shadow: 0 0 10px rgba(0, 240, 255, 0.6);
  }
  .btn-action-qr:active {
    transform: scale(0.95);
  }

  /* Modal Backdrop & Dialog */
  .modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(18, 9, 32, 0.85);
    backdrop-filter: blur(6px);
    z-index: 100;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px;
  }
  .modal-card {
    background: #241442;
    border: 2px solid var(--sunset-gold);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8), 0 0 20px rgba(255, 209, 82, 0.4);
    border-radius: 8px;
    width: 100%;
    max-width: 420px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .modal-header {
    background: #35235d;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255, 209, 82, 0.3);
  }
  .modal-title {
    font-family: var(--font-pixel);
    font-size: 13px;
    font-weight: 800;
    color: var(--sunset-gold);
    letter-spacing: 1px;
  }
  .modal-close-btn {
    background: none;
    border: none;
    color: #fbcfe8;
    font-size: 18px;
    font-weight: 800;
    cursor: pointer;
    padding: 0 4px;
    line-height: 1;
  }
  .modal-close-btn:hover { color: var(--sunset-pink); }
  .modal-body {
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 14px;
  }
  .modal-qr-img {
    width: 200px;
    height: 200px;
    border-radius: 6px;
    border: 2px solid var(--sunset-gold);
    background: #fff;
    box-shadow: 0 0 16px rgba(255, 209, 82, 0.3);
  }
  .modal-info {
    font-family: var(--font-pixel);
    font-size: 11px;
    color: #fbcfe8;
    text-align: center;
    line-height: 1.5;
  }
  .modal-actions {
    display: flex;
    gap: 10px;
    width: 100%;
    margin-top: 4px;
  }
  .modal-btn-primary {
    flex: 1;
    background: linear-gradient(135deg, #ea7d8b, #fca289);
    color: #1f0b2e;
    font-family: var(--font-pixel);
    font-size: 12px;
    font-weight: 800;
    padding: 10px 12px;
    border-radius: 4px;
    text-align: center;
    text-decoration: none;
    border: none;
  }
  .modal-btn-secondary {
    flex: 1;
    background: #35235d;
    border: 1px solid var(--neon-cyan);
    color: var(--neon-cyan);
    font-family: var(--font-pixel);
    font-size: 12px;
    font-weight: 800;
    padding: 10px 12px;
    border-radius: 4px;
    text-align: center;
    text-decoration: none;
  }
</style>
</head>
<body>
<div class="scanlines"></div>

<header>
  <div class="header-inner">
    <div class="brand-title">TITANS' ARCADE</div>
    <div class="header-right">
      <div class="live-status">
        <span class="live-dot"></span>
        <span class="live-label">LIVE REGISTRATION</span>
      </div>
      <div class="v-sep"></div>
      <a href="/" class="nav-btn">LEADERBOARD</a>
    </div>
  </div>
</header>

<main>
  <!-- Left Side: Registration Form -->
  <section class="pixel-panel" id="left-panel" style="width: 440px; flex-shrink: 0; min-width: 300px;">
    <div class="panel-header">
      <div class="panel-title">Player Registration</div>
    </div>
    <div class="panel-content">
      <div class="form-group">
        <label for="player-name">PLAYER NAME <span class="req-star">*</span></label>
        <input type="text" id="player-name" class="pixel-input" placeholder="e.g. Ravi" autocomplete="off" autofocus />
      </div>

      <div class="form-row">
        <div class="form-group">
          <label for="player-roll">URN</label>
          <input type="text" id="player-roll" class="pixel-input" placeholder="e.g. 24303192" autocomplete="off" />
        </div>
        <div class="form-group">
          <label for="player-height">HEIGHT (CM)</label>
          <input type="number" id="player-height" class="pixel-input" placeholder="e.g. 153" min="100" max="230" />
        </div>
      </div>

      <div class="form-group">
        <label>LIVE WEBCAM PHOTO</label>
        <div class="camera-container">
          <div class="vf-corner vf-tl"></div>
          <div class="vf-corner vf-tr"></div>
          <div class="vf-corner vf-bl"></div>
          <div class="vf-corner vf-br"></div>
          <div class="vf-rec"><span class="vf-rec-dot"></span><span>REC 720P</span></div>
          <div class="vf-sensor">[ SENSOR ACTIVE ]</div>
          <div class="vf-guide">
            <div class="vf-guide-box">
              <span class="vf-guide-top">FACE ALIGN</span>
              <span class="vf-guide-bot">[ READY ]</span>
            </div>
          </div>
          <video id="webcam" autoplay playsinline muted></video>
          <div id="cam-placeholder" class="camera-placeholder" style="display:none;">
            WEBCAM UNAVAILABLE.<br>PLAYER CAN STILL REGISTER WITHOUT A PHOTO.
          </div>
        </div>
      </div>

      <div class="btn-group">
        <button id="btn-snap-register" class="btn-primary" type="button">Snap &amp; Register Player</button>
        <button id="btn-skip-photo" class="btn-secondary" type="button">Register Without Photo</button>
      </div>

      <div id="reg-success" class="registered-card">
        <div style="font-family: var(--font-pixel); font-size: 11px; font-weight: 800; color: var(--retro-green);">✓ ASSIGNED PLAYER ID</div>
        <div class="reg-tag" id="reg-pid">P001</div>
        <div style="font-size: 13px; font-weight: 700; color: #fff;" id="reg-name">Name: Ravi</div>
        <div style="font-size: 12px; color: #fbcfe8;" id="reg-roll">URN: 24303192</div>
        <div style="font-size: 11px; color: var(--neon-cyan); margin-top: 6px;">
          Send player to Station A or Station B with this ID!
        </div>
        <div id="reg-qr-section" style="margin-top: 12px; padding: 10px; background: #201138; border: 1px solid #5a3c8c; border-radius: 6px; text-align: center;">
          <div style="font-family: var(--font-pixel); font-size: 10px; font-weight: 800; color: var(--sunset-gold); margin-bottom: 6px; letter-spacing: 0.5px;">
            SCAN WITH PHONE FOR SOUVENIR CARD
          </div>
          <img id="reg-qr-img" src="" alt="QR Code" style="width: 120px; height: 120px; border-radius: 4px; border: 1px solid var(--sunset-gold); background: #fff;" />
          <div style="margin-top: 6px;">
            <a id="reg-card-link" href="#" target="_blank" style="font-family: var(--font-pixel); font-size: 10px; color: var(--neon-cyan); text-decoration: underline;">
              Open Souvenir Card Page ↗
            </a>
          </div>
        </div>
      </div>

      <div id="reg-warning" class="registered-card" style="display:none; border-color: var(--sunset-pink); box-shadow: 0 0 15px rgba(234, 125, 139, 0.35);">
        <div style="font-family: var(--font-pixel); font-size: 11px; font-weight: 800; color: var(--sunset-pink);">⚠️ ALREADY REGISTERED</div>
        <div class="reg-tag" id="warn-pid" style="color: var(--sunset-pink); text-shadow: 0 0 14px rgba(234, 125, 139, 0.5);">P001</div>
        <div style="font-size: 13px; color: #fff; font-weight: 700;" id="warn-name">Name: Drishti</div>
        <div style="font-size: 12px; color: #fbcfe8; margin-top: 2px;" id="warn-detail">URN 24303192 already registered!</div>
        <div style="font-size: 11px; color: var(--sunset-gold); margin-top: 6px;">
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
  <section class="pixel-panel" id="right-panel" style="flex: 1; min-width: 300px;">
    <div class="panel-header">
      <div class="panel-title">Registered Participants</div>
      <span id="player-count" style="font-family: var(--font-pixel); font-size: 13px; color: var(--sunset-gold); font-weight: 800; text-shadow: 0 0 6px rgba(255, 209, 82, 0.6);">Total: 0</span>
    </div>
    <div class="panel-content" style="padding: 16px 20px;">
      <input type="text" id="roster-search" class="pixel-input" placeholder="Search by Name, URN, or Player ID (e.g. P001, 24303192)..." oninput="filterRoster()" />
      <div style="flex: 1; overflow-y: auto;">
        <table class="roster-table">
          <thead>
            <tr>
              <th style="width: 54px;">PHOTO</th>
              <th>PLAYER ID</th>
              <th>NAME</th>
              <th>URN</th>
              <th>HEIGHT</th>
              <th style="width: 100px; text-align: center;">PHOTOCARD</th>
            </tr>
          </thead>
          <tbody id="roster-body">
            <!-- Dynamically populated -->
          </tbody>
        </table>
        <div id="roster-empty" style="text-align: center; padding: 48px 20px; font-family: var(--font-pixel); color: #fbcfe8; font-size: 12px; display: none;">
          NO PARTICIPANTS MATCH YOUR QUERY.
        </div>
      </div>
    </div>
  </section>
</main>

<canvas id="snapshot-canvas" style="display:none;"></canvas>

<!-- Modal for QR & Souvenir Card -->
<div id="qr-modal" class="modal-backdrop" style="display: none;" onclick="if(event.target===this)closeQrModal()">
  <div class="modal-card">
    <div class="modal-header">
      <div class="modal-title" id="modal-player-title">PLAYER PHOTO CARD</div>
      <button class="modal-close-btn" type="button" onclick="closeQrModal()">✕</button>
    </div>
    <div class="modal-body">
      <div id="modal-network-mode" style="font-family: var(--font-pixel); font-size: 11px; font-weight: 800; padding: 4px 10px; border-radius: 4px; display: inline-block; margin-bottom: 8px; letter-spacing: 0.5px; background: rgba(88, 192, 56, 0.15); color: var(--retro-green); border: 1px solid rgba(88, 192, 56, 0.4);">CHECKING NETWORK...</div>
      <img id="modal-qr-img" src="" alt="QR" class="modal-qr-img" />
      <div class="modal-info" id="modal-player-info">Scan with phone camera to download high-res souvenir photo card</div>
      <div class="modal-actions">
        <a id="modal-view-card" href="#" target="_blank" class="modal-btn-primary">View Souvenir Page</a>
        <a id="modal-download-card" href="#" download class="modal-btn-secondary">Download PNG</a>
      </div>
    </div>
  </div>
</div>

<script>
  let videoStream = null;
  const videoEl = document.getElementById('webcam');
  const canvasEl = document.getElementById('snapshot-canvas');
  const camPlaceholder = document.getElementById('cam-placeholder');
  let allPlayers = [];
  let highlightedPid = null;

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

    const urlPin = new URLSearchParams(window.location.search).get('pin');
    const reqHeaders = { 'Content-Type': 'application/json' };
    if (urlPin) {
      reqHeaders['X-Operator-Pin'] = urlPin;
    }

    try {
      const res = await fetch('/api/players', {
        method: 'POST',
        headers: reqHeaders,
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

        // Keep all participants visible; highlight and scroll to the existing row
        const searchInput = document.getElementById('roster-search');
        if (searchInput && searchInput.value) {
          searchInput.value = '';
        }
        highlightedPid = errData.existing_player_id || null;
        filterRoster();
        if (highlightedPid) {
          const rowEl = document.getElementById('row-' + highlightedPid);
          if (rowEl) rowEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
          setTimeout(() => {
            highlightedPid = null;
            filterRoster();
          }, 4000);
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
      document.getElementById('reg-qr-img').src = '/api/qr/' + data.player_id + '?t=' + Date.now();
      document.getElementById('reg-card-link').href = '/api/cards/' + data.player_id + '/download';
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
        : `<span style="color: #c084fc;">-</span>`;
      const heightHtml = p.height_cm ? `<span class="height-cell">${p.height_cm} cm</span>` : `<span style="color: #c084fc;">-</span>`;
      const hlClass = (highlightedPid && p.player_id === highlightedPid) ? 'row-highlight' : '';

      return `<tr id="row-${p.player_id}" class="${hlClass}">
        <td>${photoHtml}</td>
        <td><span class="pid-badge">${p.player_id}</span></td>
        <td style="font-weight: 700; color: #fff;">${p.name}</td>
        <td>${rollHtml}</td>
        <td>${heightHtml}</td>
        <td style="text-align: center;">
          <button class="btn-action-qr" type="button" title="Open Photocard QR &amp; Download" onclick="openQrModal('${p.player_id}', '${p.name.replace(/'/g, "\\'")}')">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="display: block;">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
          </button>
        </td>
      </tr>`;
    }).join('');
  }

  function openQrModal(pid, name) {
    document.getElementById('modal-player-title').textContent = 'Photo Card: ' + name + ' (' + pid + ')';
    document.getElementById('modal-qr-img').src = '/api/qr/' + pid + '?t=' + Date.now();
    document.getElementById('modal-view-card').href = '/api/cards/' + pid;
    document.getElementById('modal-download-card').href = '/api/cards/' + pid + '/download';
    document.getElementById('modal-download-card').download = 'TitansArcade_' + pid + '.png';

    const modeEl = document.getElementById('modal-network-mode');
    if (modeEl) {
      modeEl.textContent = 'CHECKING NETWORK...';
      modeEl.style.background = 'rgba(255, 209, 82, 0.15)';
      modeEl.style.color = '#ffd152';
      modeEl.style.borderColor = 'rgba(255, 209, 82, 0.4)';
    }

    fetch('/api/cards/' + pid + '/share-info')
      .then(res => res.json())
      .then(info => {
        if (modeEl && info.mode) {
          modeEl.textContent = info.mode;
          if (info.is_public) {
            modeEl.style.background = 'rgba(88, 192, 56, 0.2)';
            modeEl.style.color = '#58c038';
            modeEl.style.borderColor = '#58c038';
          } else {
            modeEl.style.background = 'rgba(234, 125, 139, 0.2)';
            modeEl.style.color = '#ea7d8b';
            modeEl.style.borderColor = '#ea7d8b';
          }
        }
      })
      .catch(() => {});

    document.getElementById('qr-modal').style.display = 'flex';
  }

  function closeQrModal() {
    document.getElementById('qr-modal').style.display = 'none';
  }

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeQrModal();
  });

  initCamera();
  loadRoster();
  setInterval(loadRoster, 4000);

  // Interactive Drag-to-Resize Divider Logic
  const divider = document.getElementById('drag-divider');
  const leftPanel = document.getElementById('left-panel');
  const mainContainer = document.querySelector('main');
  const DEFAULT_WIDTH = 440;
  const MIN_PANEL_WIDTH = 300;
  const STORAGE_KEY = 'dodge_register_split_width';

  function setLeftWidth(widthPx) {
    if (!mainContainer || !leftPanel) return;
    const mainRect = mainContainer.getBoundingClientRect();
    const maxAllowed = mainRect.width - MIN_PANEL_WIDTH - 28;
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

  divider.addEventListener('dblclick', () => {
    setLeftWidth(DEFAULT_WIDTH);
  });
</script>
</body>
</html>"""