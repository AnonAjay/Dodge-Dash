"""Laptop 3: score server. Run:  uvicorn server.app:app --host 0.0.0.0 --port 8000
Minimal on purpose: health, accept score (idempotent), leaderboard. Extend from here.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

DATA = Path("data")
SCORES_FILE = DATA / "scores.jsonl"

app = FastAPI(title="Dodge & Drip server")


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


def _read_all() -> list[dict]:
    if not SCORES_FILE.exists():
        return []
    return [json.loads(line) for line in SCORES_FILE.read_text().splitlines() if line.strip()]


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/score")
def post_score(event: ScoreIn) -> dict:
    existing = _read_all()
    if any(e["event_id"] == event.event_id for e in existing):
        return {"accepted": True, "duplicate": True}   # safe to retry
    DATA.mkdir(exist_ok=True)
    with SCORES_FILE.open("a") as f:
        f.write(json.dumps(event.model_dump()) + "\n")
    return {"accepted": True, "duplicate": False}


@app.get("/api/leaderboard")
def leaderboard() -> dict:
    by_game: dict[str, list[dict]] = {}
    for e in _read_all():
        by_game.setdefault(e["game"], []).append(e)
    return {g: sorted(rows, key=lambda r: r["score"], reverse=True)[:10]
            for g, rows in by_game.items()}
