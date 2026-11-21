# learning.md: how to learn this project while building it

Loop for every stage: **guess -> build -> draw from memory -> check**.
Do the guess and the drawing on paper BEFORE you read the hints or answers.
Add your own notes at the bottom as you go.

---

## Stage 0: the whole pipeline
**Guess:** draw the path from a person's body to a number on the leaderboard.
Mark where each of these lives: camera thread, game loop, outbox, server.
**Hint:** five boxes in a row: sensor, numbers, state, rules, picture. One box is a fork.
**Check:** can you point at one box and say what breaks if it fails?

## Stage 1: the game loop and delta time
**Guess:** a laptop runs the game at 30 FPS, another at 60 FPS. How do you make a block take
the same real time to fall on both?
**Hint:** do not move by "pixels per frame".
**Check:** what should happen if one frame takes 2 seconds (laptop hiccup)?

## Stage 2: camera thread and "latest pose"
**Guess:** pose takes ~30 ms, drawing takes ~16 ms. Why does one serial loop feel jerky?
**Hint:** add the times up, then ask who is waiting for whom.
**Check:** what must the reader of `latest()` never do?

## Stage 3: floor homography
**Guess:** why does a hitbox measured in camera pixels favour tall players or players far away?
**Hint:** think about what happens to size in the image as someone steps back.
**Guess 2:** the transform is right for feet but wrong for head and hands. What assumption does it make?
**Check:** the camera is bumped mid-event. What do you see, and what do you do?

## Stage 4: the state machine
**Guess:** a player walks away mid-round. Which part of the system decides what happens?
**Hint:** it is not "an if inside the draw code".
**Check:** list every phase and every arrow between phases.

## Stage 5: contracts and the outbox
**Guess:** the network is down when a round ends. Where does the score live? What if the
send succeeds but the reply is lost?
**Hint:** this is why `event_id` exists.
**Check:** why is `shared/` owned by one person?

## Stage 6: Git for three people
**Guess:** when exactly does Git produce a merge conflict?
**Hint:** it is not "when two people push".
**Check:** which two habits remove most conflicts?

## Stage 7: Defending the Project (Judge Q&A & Technical Defense)
**Goal:** How to defend this project against "issue-solving" apps without falling into buzzword traps.

### A. What to Say vs. What NOT to Say
| Do NOT Say (Traps) | SAY THIS INSTEAD (Strong Engineering) | Why? |
|---|---|---|
| "It's just a fun webcam game." | "It is **Real-Time Edge Spatial Computing on Commodity Hardware**." | Shows technical depth; you're replacing \$1,000+ VR/mocap gear with standard laptops. |
| "It's just a 2D dot on a screen." | "We project warped camera pixels into **Euclidean metric floor coordinates** via Planar Homography ($H \in \mathbb{R}^{3 \times 3}$) and handle depth occlusion." | Explains the actual math and computer vision behind the immersion. |
| "We use Reinforcement Learning to control a 33-joint robot." | "The system acts as a **Zero-Cost Edge Optical MoCap pipeline** for Humanoid Inverse Kinematics (IK) Retargeting and Imitation Learning." | Real robots have motor actuator limits and physics (gravity, mass, torque), not MediaPipe visual dots. |
| "We check angles every 1 millisecond." | "We evaluate scale- and rotation-invariant joint angles once per camera frame (30 FPS) inside a decoupled 60 FPS physics loop." | Shows you understand sensor sampling rates and threading architecture. |

### B. The Robotics / MoCap Pitch
* **The Problem Solved:** Human motion capture suits (Xsens, Vicon) cost \$15,000–\$50,000.
* **Your Solution:** Zero-cost optical motion capture extracting scale-invariant joint angles ($A \to B \to C$ vectors) on standard webcams.
* **The ML Connection:** These trajectories serve as human reference demonstrations for **Imitation Learning** and **Adversarial Motion Priors (AMP)** in physics engines (Isaac Sim / MuJoCo) before transferring to physical humanoids.

### C. The Core Engineering Pillars
1. **Zero Cloud Latency / Offline-First:** Outbox pattern with idempotent UUIDs guarantees the game never freezes if network drops.
2. **Decoupled Architecture:** Pose inference runs in a separate thread so heavy neural net inference never drops 60 FPS game loop frames.
3. **Planar Homography:** Only feet touch the floor plane; joint angles handle upper body articulation independently of distance.

---

## Stage 8: Dynamic Difficulty Adjustment (DDA) & Edge MoCap Telemetry
**Guess:** A beginner struggles with joint angles and feels frustrated, while a dancer gets 100% on every pose. How can a real-time game dynamically adapt to player skill without making the game feel "rigged" or breaking tournament fairness?
**Hint:** Scale the angular tolerance window ($\pm \Delta \theta$) and hold duration based on a trailing performance window, while keeping raw scores normalized.
**The ML Connection:** How do recorded live joint angles transform into reference datasets for humanoid robot Imitation Learning (MuJoCo / Isaac Sim)?

---

## Stage 9: Post-Event Post-Mortem & Real-World Stall Lessons
**What happened under live crowd conditions, what broke, and what we learned:**

### 1. The Postcard Lifecycle Flaw (Timing & Score Updation)
- **The Issue:** Photo cards were generated immediately upon player registration at the kiosk. When players scanned the QR code at registration, their score was empty (`0 PTS` / unplayed).
- **The Root Cause:** In `server/app.py`, `register_player()` immediately invoked `_trigger_bg_upload(player_id)`. The kiosk UI presented the download link before the match started. While `post_score()` regenerated the card upon match completion, players had already walked away with the initial empty card.
- **The Production Fix:** Decouple card creation into a state-driven lifecycle:
  1. **Registration:** Print/display a "Pass Activated" ticket (profile, QR linking to dynamic live card endpoint).
  2. **Match Complete (`Phase.RESULT`):** Stations A & B display a full-screen completion banner with the player's final score, tournament rank, and a large dynamically generated Souvenir Card QR code directly on the 55" station display.
  3. **Server:** The `/card/{player_id}` endpoint serves a dynamic SVG/HTML preview or redirects to the finalized PNG once the score event is confirmed.

### 2. Foot & Leg Tracking Loss in Station A (Floor is Lava)
- **The Issue:** Multiple players experienced intermittent foot tracking dropouts, jitter, or phantom lava burns while standing still.
- **The Root Causes:**
  1. *Foreshortening & Camera Pitch:* Webcams mounted at 45° pitch create extreme non-linear perspective compression near the bottom edge of the frame.
  2. *Planar Homography Constraint:* Homography $H \in \mathbb{R}^{3 \times 3}$ assumes points strictly reside on the $Z=0$ ground plane. When a player lifts a foot, bends a knee forward, or wears dark shoes on dark floor mats, the 2D keypoint projection deviates dramatically from the physical contact patch.
  3. *MediaPipe BlazePose Keypoint Gating:* BlazePose is a top-down single-person detector trained primarily on upright fitness poses. Under downward camera angles, baggy pants or occlusion from the other leg cause ankle/heel visibility confidence to fall below threshold ($< 0.5$).
- **The Optimization Pipeline & Research Papers:** See [Stage 10](#stage-10-computer-vision--kinematics-research-defense).

### 3. Station B Poses & Angular Tolerance Issues
- **The Issue:** Certain poses in the 36-pose catalog felt "physically impossible" or failed to register even when players mimicked the cutout image accurately.
- **The Root Causes:**
  1. *2D Projection Ambiguity:* An arm extended forward towards the camera has a foreshortened 2D projection, causing the calculated 2D joint angle $\angle(A, B, C)$ to collapse toward $180^\circ$ or $0^\circ$ despite being $90^\circ$ in 3D physical space.
  2. *Static Catalog Angles:* Catalog angles generated from 2D web images without manual ground-truth calibration contain artifacts from camera lens distortion and actor tilt.
  3. *Lack of Internal Studio Tooling:* Authors could not inspect or tune angles interactively with visual feedback before deploying to production.
- **The Production Fix:** Build `tools/pose_studio.py` (an interactive visual pose editor) allowing developers to tweak joint angles, adjust angular tolerance windows ($\pm \Delta \theta$), set key-limb masks, and test matching live with webcam feedback.

### 4. Skip Gesture Ergonomics
- **The Issue:** The initial 0.5s "Hand on Head" gesture was too strict for players of varying reach, while an unconstrained "Hand Raised High" trigger caused severe false-positive skips during natural poses (Superman, Cheer, Disco). Furthermore, holding a hand raised caused consecutive double-skips.
- **The Production Fix:**
  1. **Pinocchio Touch-Nose Gesture:** Replaced overhead hands with a deliberate finger/hand-to-nose touch (index, wrist, or pinky within $\Delta \le 0.075$ normalized distance and tight vertical gate $|y_{\text{hand}} - y_{\text{nose}}| < 0.045$). This completely prevents overhead poses, boxing guards, and military salutes from triggering accidental skips.
  2. **Edge-Triggered Latch ("Release-to-Rearm"):** Player must physically lower their hand away from their nose before another skip can charge, eliminating consecutive double-skips.
  3. **Visual Progress Ring:** On-screen 0.5s visual charge ring providing clear continuous visual feedback.

### 5. Stationary Jitter vs. Deadband Stutter in Station A (Floor is Lava)
- **The Issue:** Cheap laptop webcams produce $\pm 3\text{--}4\text{px}$ sensor noise. When projected through floor homography $H$, this causes $5\text{--}15\text{cm}$ floor drift, which can falsely burn a player standing near a lava boundary.
- **The Deadband Anti-Pattern (What Failed):** Clamping coordinates to an anchor point if displacement is $< 8\text{cm}$ stopped drift at rest, but when stepping, the coordinates snapped forward in $8\text{cm}$ chunks. This spatial quantization created the visual perception of a severe 5 FPS frame rate drop / stutter.
- **The Production Fix (1-Euro Adaptive Filter):** Implemented Géry Casiez's 1-Euro filter ($f_c = f_{\text{min}} + \beta |\hat{v}|$). When stationary, cutoff $f_c \to 0.8\text{ Hz}$ ($\alpha \approx 0.14$), heavily absorbing 85%+ of sensor noise. During fast steps ($|\hat{v}| \approx 1000\text{ px/s}$), cutoff jumps to $20+\text{ Hz}$ ($\alpha \to 0.81$), reducing lag to $< 15\text{ ms}$. Because it is $C^\infty$ continuous, motion remains silky 60 FPS without snap-to-grid artifacts.

---

## Stage 10: Computer Vision & Kinematics Research Defense
**How to answer technical interview questions regarding CV limitations and optimizations:**

### A. The 3 Academic Papers to Cite in Interviews
1. **MediaPipe BlazePose (Bazrev et al., CVPR Workshop 2020)**
   - *Title:* "BlazePose: On-device Real-time Body Pose Tracking"
   - *Architecture:* Two-stage detector-tracker pipeline. A lightweight detector predicts body bounding box + rotation; a regression network predicts 33 3D keypoints + visibility flags at 30+ FPS on CPU.
   - *Limitation in our setting:* Optimized for upright fitness poses in mobile/selfie orientations. Suffers from keypoint jitter under steep downward pitch angles ($>35^\circ$) and lacks explicit physical ground contact awareness.
2. **RTMPose / YOLO-Pose (Jiang et al., 2023 / Wang et al., 2022)**
   - *Title:* "RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose"
   - *Architecture:* Uses SimCC (Simple Coordinate Classification) treating keypoint localization as two 1D horizontal/vertical classification tasks rather than direct coordinate regression or high-resolution heatmap generation.
   - *Advantage:* Achieves sub-pixel precision on limbs/ankles, runs at 90+ FPS on edge hardware, and demonstrates robust tracking under low-contrast lighting and baggy clothing.
3. **ContactPose & Physical Ground Contact Modeling (Brahmbhatt et al., CVPR / Hassan et al.)**
   - *Title:* "ContactPose: A Dataset of Hand-Object and Foot-Surface Contact"
   - *Core Concept:* Explicitly classifies contact state $c \in \{0, 1\}$ alongside kinematic position.
   - *Application in Dodge & Drip:* When foot velocity $\|v_{\text{foot}}\| < \epsilon$, lock the ground contact anchor point to prevent visual jitter from artificially triggering lava boundaries.

### B. The Mathematical Formulation of Ground Contact & Homography
Given camera pixel coordinates $\mathbf{x} = [u, v, 1]^T$, the Planar Homography matrix $H \in \mathbb{R}^{3 \times 3}$ maps pixels to floor metric coordinates $\mathbf{X} = [X_w, Y_w, 1]^T$:

$$\mathbf{X} \sim H \mathbf{x} \implies \begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix} \sim \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

- **Why only feet?** Homography assumes all points lie on the plane $Z = 0$. For any point with elevation $Z > 0$ (such as head or hand), the perspective ray intersects the floor plane at a virtual point displaced by parallax error $\Delta \mathbf{X} \propto \frac{Z}{d_{\text{cam}}}$.

### C. Scale- and Distance-Invariant Joint Angle Formulation
To make pose matching invariant to player height, body proportion, and distance from the camera, we calculate interior joint angles using vector dot products rather than raw coordinate distances:

$$\mathbf{v}_1 = \mathbf{A} - \mathbf{B}, \quad \mathbf{v}_2 = \mathbf{C} - \mathbf{B}$$

$$\theta = \arccos\left( \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\| \|\mathbf{v}_2\| + \epsilon} \right) \times \frac{180^\circ}{\pi}$$

- **Rotation Invariance:** $\theta$ remains identical regardless of global camera rotation around the optical axis.
- **Scale Invariance:** Multiplying coordinates by any scalar $s > 0$ cancels out in the numerator and denominator: $\frac{(s\mathbf{v}_1) \cdot (s\mathbf{v}_2)}{\|s\mathbf{v}_1\| \|s\mathbf{v}_2\|} = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\| \|\mathbf{v}_2\|}$.

---

## Stage 11: What We Used, Why We Used It, How We Should Do It, How We Shouldn't
**The Comprehensive Technical Interview Cheat Sheet:**

| Subsystem | What We Used | Why We Used It (Strengths) | What We Shouldn't Do (Anti-Patterns) | How We Should Do It Next (Production / Web) |
|---|---|---|---|---|
| **Pose Estimation** | MediaPipe Tasks API (`PoseLandmarker`) | Pure CPU real-time inference (25–35ms), no GPU required, lightweight deployment on standard laptops. | Do not use deprecated `mp.solutions.pose`. Do not run heavy multi-stage models (e.g. OpenPose) synchronously on the main thread. | Export an optimized ONNX / TensorRT runtime of RTMPose-m with SIMCC head for foot stability; for Web, run `@mediapipe/tasks-vision` via WebAssembly + WebGL. |
| **Game Engine & Rendering** | Pygame CE (Python) | Low-overhead 2D arcade rendering, direct pixel/surface manipulation, sub-millisecond audio playback, rapid prototyping. | Do not write game logic inside drawing loops. Do not build custom physics from scratch without fixed delta time (`dt`). | Port to lightweight HTML5 Canvas + WebAssembly (or Pygame with Pygbag) for zero-install browser deployment on Render / Vercel. |
| **Networking & Backend** | FastAPI + Uvicorn (ASGI) | Async event handling, automatic OpenAPI/Pydantic validation, low memory footprint, sub-millisecond local network latency. | Do not use synchronous blocking Flask with global state. Do not rely on cloud endpoints at runtime. | Deploy FastAPI backend to Render / Fly.io with a PostgreSQL / Supabase store and WebSocket streaming for live multi-client updates. |
| **Fault Tolerance** | Local Outbox Pattern (`outbox.jsonl`) | Zero game freezing when Wi-Fi drops; idempotent writes with client-side UUIDs (`event_id`). | Do not fire-and-forget raw HTTP requests inside the 60 FPS loop. Do not crash the game if the server returns 500. | SQLite WAL (Write-Ahead Logging) queue with exponential backoff daemon worker for high concurrency. |
| **Pose Matching** | Scale-Invariant 6-Limb Joint Angles | Invariant to player distance, camera zoom, height, and body mass index. | Do not compare raw Euclidean pixel distances or 2D bounding boxes across different players. | Hybrid 3D kinematic vector matching with Dynamic Time Warping (DTW) for continuous dance/gesture sequences. |
| **Souvenir Postcards** | Pillow (`PIL`) Server-side Renderer | High-resolution 1080x1350 raster output with custom retro fonts and procedural sunset gradients, 100% offline. | Do not render cards before the player has completed the game. Do not depend on external CDN fonts or internet image APIs. | Render cards on demand via SVG template engine or client-side Canvas; generate post-match QR codes directly on the station display. |

---

## Stage 12: Deployment Topologies: 3-Laptop vs 2-Laptop vs 1-Laptop Web

```
                    ┌──────────────────────────────────────────────┐
                    │               DEPLOYMENT MODES               │
                    └──────────────────────┬───────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
   [ 3-Laptop Stall ]              [ 2-Laptop Setup ]             [ 1-Laptop / Web Demo ]
• Laptop 1: Station A (Lava)     • Laptop 1: Server + STA-A     • Single PC / Browser
• Laptop 2: Station B (Blazer)   • Laptop 2: Station B (Blazer) • Embedded local server
• Laptop 3: Server + Leaderboard • Leaderboard on Laptop 1      • Hosted on Render/Vercel
```

1. **Stall Mode (3 Physical Laptops):**
   - High-throughput arcade setup: dedicated registration kiosk + leaderboard TV, Station A, Station B.
2. **Showcase Mode (2 Laptops):**
   - Laptop 1 runs `uvicorn server.app:app` in background + Station A game on primary display.
   - Laptop 2 runs Station B game, targeting `http://<laptop1-ip>:8000`.
   - Registration and Leaderboard accessible via browser tab on Laptop 1 or via phone/tablet over Wi-Fi.
3. **Web / Portfolio Standalone Mode (1 Laptop or Cloud):**
   - Game stations run with `--standalone` flag (in-memory mock server, local score saving, zero network dependencies).
   - Web version compiled via WebAssembly / Pygbag or rewritten in React + MediaPipe Web JS for 1-click browser testing on Vercel / Render.

---

## Stage 13: Prioritized Engineering To-Do List

### Phase 1: High-Priority Fixes (Completed)
- [x] **Fix Postcard Timing:**
  - Removed premature card generation in `server/app.py:register_player()`.
  - Updated `post_score()` to finalize card generation upon game end.
  - Displayed post-match Souvenir Card QR code directly on Station A & Station B `Phase.RESULT` screens.
- [x] **Build Pose Studio (`tools/pose_studio.py`):**
  - Interactive tool to load poses, visualize target joint angle vectors on cutout images, tweak angles and tolerances with keyboard/sliders, and save directly to `data/poses_index.json`.
  - Live webcam split-screen to test player match ergonomics in real time with live skeleton overlay.
- [x] **Station B Skip & Match Engine Polish:**
  - Implemented Pinocchio Touch-Nose gesture with tight vertical alignment gate.
  - Implemented edge-triggered release-to-rearm latch preventing consecutive double-skips.
  - Added clean 3-zone retro arcade attract screen UI.
- [x] **Station A Foot Ground-Contact Gating & Smoothing:**
  - Replaced fixed EMA and deadband clamping with Géry Casiez's 1-Euro adaptive filter ($f_c = f_{\text{min}} + \beta |\hat{v}|$) in `pose_core/capture.py`.
  - Preserved continuous 60 FPS fluid motion while eliminating stationary sensor jitter.

### Phase 2: Station A UI/UX Revamp
- [ ] Add retro cyberpunk neon lava shader / procedural heatwave visual fx.
- [ ] Dynamic warning tiles: pulsing glowing borders, volcanic fissure cracks, particle embers.
- [ ] Screen shake on hit + dynamic combo multiplier banners.
- [ ] High-contrast tournament HUD with animated timer bar and stage transitions.

### Phase 3: Documentation & Portfolio Showcase
- [x] Overhaul `README.md` with high-resolution architecture diagrams, clear quickstart guides for 1-laptop, 2-laptop, and 3-laptop modes, and developer tools documentation.
- [ ] Record high-framerate gameplay demo clips for GitHub showcase and video portfolio.
- [ ] Package standalone web demo (MediaPipe JS / Pygbag) for zero-friction browser play.

---

## My notes
(write what you learned, in your own words)

## Mistakes I made
| Date | What I got wrong | What was actually true |
|---|---|---|
| 2026-11-20 | Premature card creation at registration | Postcard must be generated post-match to include final scores and tournament rank |
| 2026-11-20 | Assuming planar homography works for whole body | Homography is mathematically valid only for ground contact points ($Z=0$); upper body requires vector angle math |
| 2026-11-20 | Hand on head skip gesture was too strict | Real-world crowd variation in reach and camera angle requires generous spatial envelopes ($>20\%$ tolerance) |
| 2026-11-21 | Overhead hand skip gesture | Hands raised in poses like Superman and Cheer triggered false skips. Replaced with Touch-Nose + edge-triggered release latch to eliminate double skips |
| 2026-11-21 | Deadband coordinate clamping for stationary noise | Clamping coordinates within $<8\text{cm}$ stopped drift but created an optical 5 FPS frame-drop illusion due to snap-to-grid jumps. Continuous 1-Euro adaptive low-pass filter ($f_c = f_{\text{min}} + \beta |\hat{v}|$) eliminated jitter while keeping silky 60 FPS motion |
| 2026-11-21 | Forcing ankle-only tracking with floor-facing camera | BlazePose confidence drops when the body is cropped by downward camera tilt; multi-point foot centroiding (heels + toes) provides a more realistic ground contact patch on carpet |

---

## Answers (do not read until you have tried the guess)

0. Camera thread -> latest PoseFrame -> game loop (logic -> GameState -> render). The fork:
   logic also emits a ScoreEvent -> outbox -> server. Camera thread, game loop and server are
   separate so a slow or dead one does not freeze the others.
1. Move by `speed * dt`, where `dt` is real elapsed seconds. Cap `dt` (for example 0.1 s)
   so a freeze does not teleport everything.
2. 30 + 16 = 46 ms per loop is about 22 FPS and uneven. With a thread, drawing runs at its own
   rate and reads the newest pose. `latest()` must never wait for the camera.
3. A far player looks smaller in the image, so a pixel-space box is easier or harder to hit
   by distance and height. Feet are always on the floor, so floor coordinates are fair.
   The homography assumes every point lies on one flat plane (the floor); head and hands
   are not on it. If the camera moves, tiles drift from the tape on the floor: recalibrate.
4. The state machine in `logic.update`, with one place for phase transitions. Phases:
   IDLE -> CALIBRATING -> PLAYING -> RESULT -> IDLE, plus reset -> IDLE from anywhere.
5. In `outbox.jsonl` on disk, sent later. If the reply is lost, the game resends and the
   server ignores the duplicate `event_id`. One owner for `shared/` because every module
   depends on it.
6. A conflict only happens when two people change the same lines of the same file. Work in
   your own folder, and commit and merge small and often.
7. Never claim you are running reinforcement learning directly on a real robot with MediaPipe's
   33 visual points. Real robots have rigid actuator limits, torques, and gravity balance constraints.
   Frame it accurately as an edge optical MoCap pipeline capturing human reference kinematics for
   Inverse Kinematics (IK) retargeting and imitation learning in simulators like Isaac Sim or MuJoCo.
8. Dynamic Difficulty Adjustment (DDA) tracks a trailing 3-pose accuracy window. If accuracy < 60%,
   grace tolerance expands by up to +4 deg (preventing early crowd churn); if player achieves
   perfection (>85%), snap hold requirement tightens from 0.35s to 0.20s (testing rapid reflexive
   stability). High-fidelity 30 Hz joint angle streams logged to local JSONL act as zero-cost human
   demonstration datasets for offline adversarial imitation learning (AMP) in physics engines.
