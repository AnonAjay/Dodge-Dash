# AGENTS.md: Dodge & Drip

## Goal
A body-controlled camera game for a college tech-fest stall. Crowd retention is the win
condition: it must be understood in 10 seconds and be fun to watch from 10 feet away.
- Station A (Laptop 1): Floor is Lava (dodge on 10x10 floor-tile grid, 3-round tournament lifecycle).
- Station B (Laptop 2): Speed Pose Match (match target cutout poses via joint angles as fast as possible).
- Laptop 3: score server, big-screen leaderboard, photo cards.

## Hard constraints (never break these)
- No sensors, no extra hardware. Laptops, webcams, one speaker, one mic only.
- Runs fully offline on a local network. No cloud calls, no internet at runtime.
  * Target pose cutouts from the internet and reference angles MUST be preprocessed and
    stored locally offline. Never scrape, download, or run MediaPipe on reference images at runtime.
- A game station must keep working if the server is down: write the score to the local
  outbox first, send later.
- The camera/pose work must never block the game loop.
  * Target pose comparison must be fast pure-Python angle math in logic.py.
- Only the player's FEET go through the floor homography (it is valid for the floor plane only).
- Joint angle matching in Station B must be invariant to scale, height, and camera distance.

## Repo map and ownership
| Folder | Owner | Rule |
|---|---|---|
| shared/ | lead only | contracts; any change must be announced to the whole team |
| pose_core/ | pose team | camera thread, MediaPipe, smoothing, floor mapping |
| station_a/ | logic + render team | rules in logic.py (pure), drawing in render.py |
| station_b/ | Shagun (logic) / team | Speed Pose Match: cutout poses, joint angle extraction & matching |
| server/ | server team | FastAPI, leaderboard |
| cards/ | cards team | photo card + filter, runs on Laptop 3 only |
| tools/ | anyone | scripts, never imported by the game |
Only edit the folder you were asked to work in. If another folder must change, say so and stop.

## Commands
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m tools.get_models                              # downloads the pose model
python -m tools.timing_test                             # camera + pose ms per frame
python -m station_a --keyboard                          # play Station A without a camera
python -m station_b --keyboard                          # play Station B without a camera
uvicorn server.app:app --host 0.0.0.0 --port 8000       # Laptop 3
pytest
```

## Code rules
- Python 3.12 on every laptop. Type hints. Small functions. No clever one-liners.
- MediaPipe: Tasks API (`PoseLandmarker`) only. `mp.solutions` does not exist in the installed version.
- Do not import cv2, pygame, or any network library in `station_a/logic.py`.
- Do not install `opencv-python` (it conflicts with `opencv-contrib-python`).
- No new dependency without lead approval. Do not edit requirements.txt yourself.
- Never commit: models/, recordings, config.local.json, calibration.local.json, outbox.jsonl.
- Every game rule gets a pytest. Run `pytest` before saying a task is done.
- Skills for specific areas are in `.agent/skills/`. Read the matching one before starting.

## How to work with me (teaching mode)
I am learning while building and want to become a strong engineer, not just get code.
- Before a non-trivial module, ask me to guess the approach in 2-3 sentences. Then tell me
  how your approach differs and why.
- After you build it, name the 2-3 ideas it depends on and ask me to sketch the data flow.
- Be blunt. No flattery, no padding. If I am wrong, give a hint first, then the correction.
- If my plan conflicts with a hard constraint above, say so BEFORE writing code.
- Keep explanations short. Point me to the matching section of learning.md.
