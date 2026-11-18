"""Keyboard simulator for Station B: Wall Shape.
Allows moving the player with WASD / arrows and cycling simulated poses with 1-8.
"""
from __future__ import annotations

import threading
import time
import pygame

from pose_core.pose_matcher import DEFAULT_TEMPLATES
from shared.contracts import PoseFrame


class KeyboardWallPoseProvider:
    """Simulates a player on the floor and their body pose angles using keyboard controls."""

    def __init__(self, arena_w_m: float = 1.8, arena_d_m: float = 1.8, speed_m_s: float = 1.2):
        self.arena_w_m = arena_w_m
        self.arena_d_m = arena_d_m
        self.speed = speed_m_s

        self._lock = threading.Lock()
        self._running = False
        self.x = arena_w_m / 2.0
        self.y = arena_d_m / 2.0
        self.template_index = 0
        self.simulated_angles: dict[str, float | None] = dict(DEFAULT_TEMPLATES[0].expected_angles)

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def handle_keys(self, keys: pygame.key.ScancodeWrapper, dt: float) -> None:
        dx, dy = 0.0, 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= self.speed * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += self.speed * dt
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy += self.speed * dt
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy -= self.speed * dt

        with self._lock:
            self.x = max(0.2, min(self.arena_w_m - 0.2, self.x + dx))
            self.y = max(0.2, min(self.arena_d_m - 0.2, self.y + dy))

            # Number keys 1..8 select corresponding preset pose
            num_keys = [
                pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8,
            ]
            for idx, k in enumerate(num_keys):
                if keys[k] and idx < len(DEFAULT_TEMPLATES):
                    self.template_index = idx
                    self.simulated_angles = dict(DEFAULT_TEMPLATES[idx].expected_angles)

    def match_current_wall(self, template_expected_angles: dict[str, float]) -> None:
        """Helper to match whatever target pose is oncoming."""
        with self._lock:
            self.simulated_angles = dict(template_expected_angles)

    def latest(self) -> PoseFrame:
        with self._lock:
            return PoseFrame(
                timestamp=time.monotonic(),
                tracked=True,
                floor_x_m=round(self.x, 3),
                floor_y_m=round(self.y, 3),
                left_foot_m=(round(self.x - 0.15, 3), round(self.y, 3)),
                right_foot_m=(round(self.x + 0.15, 3), round(self.y, 3)),
                arms_up=False,
                airborne=False,
            )

    def get_angles(self) -> dict[str, float | None]:
        with self._lock:
            return dict(self.simulated_angles)
