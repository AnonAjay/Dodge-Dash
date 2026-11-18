"""Score delivery for Station A.
Writes ScoreEvent to outbox.jsonl immediately upon game completion.
"""
from __future__ import annotations

import json
from pathlib import Path

from shared.contracts import ScoreEvent

OUTBOX_PATH = Path("outbox.jsonl")


def enqueue_score(event: ScoreEvent, path: Path = OUTBOX_PATH) -> None:
    """Append a ScoreEvent to the local outbox file."""
    line = json.dumps(event.to_dict())
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(f"[Outbox] Score saved: {event.event_id} (Score: {event.score})")
