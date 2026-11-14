"""Load config.local.json (per laptop, git-ignored) or fall back to config.example.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    local = ROOT / "config.local.json"
    path = local if local.exists() else ROOT / "config.example.json"
    return json.loads(path.read_text())
