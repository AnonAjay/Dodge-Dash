---
name: game-logic
description: Use when working in station_a/ on game rules, the state machine, scoring, difficulty, or the keyboard test input.
---
# Game logic rules
- `station_a/logic.py` is pure Python. No cv2, pygame, or network imports. This is what makes
  keyboard play and pytest possible.
- Phase transitions (IDLE, CALIBRATING, PLAYING, RESULT) happen in ONE function only.
  A reset key always returns to IDLE.
- Use real elapsed `dt` in seconds, capped (for example 0.1 s). Never move "per frame".
- Telegraph every block: warn on the target tile at least ~1.2 s before impact. Judge the hit at impact.
- Difficulty: speed rises with dodges but is capped; rounds last 60-90 s so nobody hogs the station.
- Handle `pose.tracked == False`: pause or freeze, never crash, never count it as a hit.
- Scoring: +10 per dodge, streak multiplier +0.1 per 5 dodges (max 2.0), a hit resets the streak.
- Every rule gets a pytest in `tests/`. `tiles_touched` has spec tests already.
