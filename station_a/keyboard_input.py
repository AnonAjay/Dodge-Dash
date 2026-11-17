"""PoseProvider that moves a dot on the grid with arrow keys (and space = jump/airborne).
Lets everyone build + test the game with no camera. Implements shared.contracts.PoseProvider.
"""
from __future__ import annotations

import threading
import time
import pygame

from shared.contracts import PoseFrame


class KeyboardPoseProvider:
    """Simulates a player on the floor grid using keyboard controls."""

    def __init__(self, grid_w_m: float = 1.8, grid_h_m: float = 1.8, speed_m_s: float = 1.2):
        self.grid_w_m = grid_w_m
        self.grid_h_m = grid_h_m
        self.speed = speed_m_s

        self._lock = threading.Lock()
        self._running = False
        self.x = grid_w_m / 2.0
        self.y = grid_h_m / 2.0
        self._airborne_until = 0.0
        self._arms_up = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def handle_keys(self, keys: pygame.key.ScancodeWrapper, dt: float) -> None:
        """Call each game loop tick with pygame.key.get_pressed() and dt in seconds."""
        dx, dy = 0.0, 0.0

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= self.speed * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += self.speed * dt
        # In floor coordinates: y grows away from the camera (W / Up walks deeper into grid)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy += self.speed * dt
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy -= self.speed * dt

        now = time.monotonic()
        if keys[pygame.K_SPACE]:
            self._airborne_until = now + 0.45

        self._arms_up = bool(keys[pygame.K_u] or keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])

        with self._lock:
            self.x = max(0.0, min(self.grid_w_m, self.x + dx))
            self.y = max(0.0, min(self.grid_h_m, self.y + dy))

    def latest(self) -> PoseFrame:
        now = time.monotonic()
        with self._lock:
            x, y = self.x, self.y
            airborne = now < self._airborne_until
            arms_up = self._arms_up

        foot_offset = 0.15
        left_foot = (max(0.0, x - foot_offset), y)
        right_foot = (min(self.grid_w_m, x + foot_offset), y)

        return PoseFrame(
            timestamp=now,
            tracked=True,
            floor_x_m=x,
            floor_y_m=y,
            left_foot_m=left_foot,
            right_foot_m=right_foot,
            arms_up=arms_up,
            airborne=airborne,
        )
