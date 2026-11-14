"""The contracts every module builds against. LEAD ONLY: change fields here
and everyone breaks. Announce any change to the whole team first.

Flow:  PoseProvider -> PoseFrame -> game logic -> GameState -> renderer
       game logic also emits: ScoreEvent -> outbox -> server
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Protocol


# ---------- input side ----------
@dataclass
class PoseFrame:
    """What the game knows about the player this instant. No pixels, no camera."""
    timestamp: float = field(default_factory=time.monotonic)
    tracked: bool = False                # False = player lost / feet not visible
    floor_x_m: float | None = None       # player footprint centre on the floor, metres
    floor_y_m: float | None = None       # (0,0) = corner of the grid, y grows away from the camera
    left_foot_m: tuple[float, float] | None = None
    right_foot_m: tuple[float, float] | None = None
    arms_up: bool = False
    airborne: bool = False               # jumping: floor position is frozen


class PoseProvider(Protocol):
    """Anything that can supply PoseFrames: the real camera or the keyboard."""
    def start(self) -> None: ...
    def latest(self) -> PoseFrame: ...
    def stop(self) -> None: ...


# ---------- game side ----------
class Phase(str, Enum):
    IDLE = "idle"
    CALIBRATING = "calibrating"
    PLAYING = "playing"
    RESULT = "result"


@dataclass
class Block:
    col: int
    row: int
    time_to_impact_s: float              # counts down; hit is judged at 0
    warned: bool = False


@dataclass
class GameState:
    phase: Phase = Phase.IDLE
    player_id: str | None = None
    time_left_s: float = 0.0
    score: int = 0
    streak: int = 0
    best_streak: int = 0
    dodges: int = 0
    hits: int = 0
    speed: float = 1.0
    blocks: list[Block] = field(default_factory=list)
    occupied_tiles: set[tuple[int, int]] = field(default_factory=set)


# ---------- output side ----------
@dataclass
class ScoreEvent:
    event_id: str                        # e.g. "A-P017-<unix seconds>" -> server ignores duplicates
    player_id: str
    station: str                         # "A" or "B"
    game: str                            # "dodge" | "wall" | "dance"
    score: int
    accuracy: float = 0.0
    best_streak: int = 0
    hype_avg: float | None = None        # Station B only, 0..100
    duration_s: float = 0.0
    finished_at: str = ""                # ISO 8601

    def to_dict(self) -> dict:
        return asdict(self)
