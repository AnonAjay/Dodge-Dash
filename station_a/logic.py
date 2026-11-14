"""Game rules. PURE PYTHON: no cv2, no pygame, no network imports here.
That is what lets you test the whole game with the keyboard and with pytest.
"""
from __future__ import annotations

from shared.contracts import GameState, PoseFrame


def tiles_touched(x_m: float, y_m: float, radius_m: float,
                  cols: int = 3, rows: int = 3, tile_m: float = 0.6) -> set[tuple[int, int]]:
    """Return {(col, row)} of every tile the footprint circle overlaps.

    Standing mid-tile -> 1 tile. On an edge -> 2. On a corner -> 4.
    Outside the grid -> those tiles are simply not included.
    HINT: for each tile, find the point of the tile nearest to the circle centre;
    if that point is within radius_m, the tile is touched.
    """
    raise NotImplementedError


def update(state: GameState, pose: PoseFrame, dt: float) -> GameState:
    """Advance the game by dt seconds. Phase transitions happen ONLY here.

    TODO: IDLE -> CALIBRATING -> PLAYING -> RESULT -> IDLE
          spawn blocks, count down time_to_impact_s, judge hits at 0,
          ramp speed (capped), handle pose.tracked == False.
    """
    raise NotImplementedError
