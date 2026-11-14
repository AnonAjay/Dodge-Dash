---
name: score-contract
description: Use when sending or receiving scores: ScoreEvent, station_a/outbox.py, server/app.py, or any change to shared/contracts.py.
---
# Score contract rules
- `ScoreEvent` fields are defined in `shared/contracts.py`. Do not edit that file without the lead's approval
  and an announcement to the team.
- `event_id` format: `<station>-<player_id>-<unix seconds>`. The server ignores duplicates, so retries are safe.
- Write to `outbox.jsonl` BEFORE any network call. Remove an entry only after the server accepted it.
- Never block the game loop on the network. Send from a background thread or between rounds, with a short timeout.
- Endpoints on Laptop 3: `GET /api/health`, `POST /api/score`, `GET /api/leaderboard`.
- Server URL comes from config, never hard-coded.
