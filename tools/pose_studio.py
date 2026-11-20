"""Pose Studio: Interactive Developer Tool for Custom Pose Creation, Angle Tuning & Live Calibration.

Launch:
  python -m tools.pose_studio
  python -m tools.pose_studio --no-cam
  python -m tools.pose_studio --catalog data/poses_index.json --camera 0

Controls:
  [LEFT] / [RIGHT] or [A] / [D] : Navigate Previous / Next Pose in catalog
  [UP] / [DOWN] or [1..8]        : Select Joint to tune
  [+] / [-] or [W] / [S]         : Adjust selected joint target angle (+/- 5 deg)
  [K]                            : Toggle Key Joint status (signature limb)
  [C]                            : CAPTURE live webcam angles as new target!
  [R]                            : Revert pose angles to catalog file values
  [CTRL+S] or [S]                : SAVE catalog to data/poses_index.json
  [ESC]                          : Exit Pose Studio
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pygame

from pose_core.blazer_matcher import (
    BlazerMatchResult,
    BlazerPoseTemplate,
    CORE_JOINTS,
    LIMB_6_JOINTS,
    evaluate_blazer_pose,
    load_blazer_catalog,
    save_blazer_catalog,
)
from pose_core.capture import CameraPoseSource
from pose_core.pose_matcher import extract_pose_angles
from pose_core.skeleton import BONES
from shared.contracts import PoseFrame

# Colors (Sunset Cyberpunk Arcade Theme)
C_DARK_BG = (16, 12, 28)
C_PANEL_BG = (28, 20, 48)
C_PANEL_BORDER = (90, 60, 140)
C_SUNSET_GOLD = (255, 209, 82)
C_SUNSET_PINK = (234, 125, 139)
C_NEON_CYAN = (0, 240, 255)
C_NEON_GREEN = (50, 240, 100)
C_AMBER_YELLOW = (255, 200, 40)
C_ALERT_RED = (240, 60, 60)
C_TEXT_WHITE = (248, 250, 252)
C_TEXT_MUTED = (160, 150, 190)

JOINT_NAMES = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
]


class PoseStudioApp:
    def __init__(
        self,
        catalog_path: str = "data/poses_index.json",
        poses_dir: str = "data/poses",
        cam_index: int = 0,
        no_cam: bool = False,
    ) -> None:
        pygame.init()
        pygame.font.init()

        self.catalog_path = Path(catalog_path)
        self.poses_dir = Path(poses_dir)
        self.cam_index = cam_index
        self.no_cam = no_cam

        self.width = 1280
        self.height = 720
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Titans' Arcade — Internal Pose Studio & Angle Tuner")

        # Fonts
        self.font_h1 = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_h2 = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_body = pygame.font.SysFont("Consolas", 14, bold=False)
        self.font_bold = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_mini = pygame.font.SysFont("Consolas", 11)

        # Load Catalog
        self.catalog: list[BlazerPoseTemplate] = load_blazer_catalog(self.catalog_path)
        if not self.catalog:
            print(f"[Pose Studio] Catalog at {self.catalog_path} is empty or missing! Loading starter fallback.")
            self.catalog = [
                BlazerPoseTemplate(
                    id="t_pose",
                    name="T-Pose",
                    image_file="t_pose.jpg",
                    target_angles={
                        "left_shoulder": 90.0,
                        "right_shoulder": 90.0,
                        "left_elbow": 180.0,
                        "right_elbow": 180.0,
                        "left_knee": 180.0,
                        "right_knee": 180.0,
                    },
                    key_joints=["left_shoulder", "right_shoulder"],
                )
            ]

        # Working state
        self.current_idx = 0
        self.selected_joint_idx = 0
        self.dirty = False
        self.banner_msg = "WELCOME TO POSE STUDIO: [LEFT/RIGHT] BROWSE  |  [+/-] TUNE  |  [C] CAPTURE  |  [S] SAVE"
        self.banner_timer = 5.0
        self.banner_color = C_SUNSET_GOLD

        # Backup original catalog values for revert
        self.original_catalog: list[BlazerPoseTemplate] = [
            BlazerPoseTemplate.from_dict(p.to_dict()) for p in self.catalog
        ]

        # Start Camera if enabled
        self.cam_source: CameraPoseSource | None = None
        if not self.no_cam:
            try:
                print(f"[Pose Studio] Initializing Camera {self.cam_index} for live angle calibration...")
                self.cam_source = CameraPoseSource(self.cam_index)
                self.cam_source.start()
            except Exception as e:
                print(f"[Pose Studio] Could not open camera {self.cam_index}: {e}. Running in camera-free mode.")
                self.cam_source = None

        self.clock = pygame.time.Clock()

    @property
    def current_pose(self) -> BlazerPoseTemplate:
        return self.catalog[self.current_idx]

    def save(self) -> None:
        """Save catalog to disk with safety backup."""
        try:
            if self.catalog_path.exists():
                bak_path = self.catalog_path.with_suffix(".json.bak")
                shutil.copy2(self.catalog_path, bak_path)

            save_blazer_catalog(self.catalog, self.catalog_path)
            self.dirty = False
            self.banner_msg = f"SUCCESS: Saved {len(self.catalog)} poses to {self.catalog_path} (backup created)!"
            self.banner_color = C_NEON_GREEN
            self.banner_timer = 4.0
            print(f"[Pose Studio] Saved catalog to {self.catalog_path}")
        except Exception as e:
            self.banner_msg = f"ERROR saving catalog: {e}"
            self.banner_color = C_ALERT_RED
            self.banner_timer = 5.0

    def revert(self) -> None:
        """Revert current pose angles to last saved state."""
        orig = self.original_catalog[self.current_idx]
        self.catalog[self.current_idx] = BlazerPoseTemplate.from_dict(orig.to_dict())
        self.banner_msg = f"Reverted '{self.current_pose.name}' to original angles."
        self.banner_color = C_AMBER_YELLOW
        self.banner_timer = 3.0

    def capture_live(self, live_angles: dict[str, float | None]) -> None:
        """Capture live webcam angles directly into current pose target."""
        valid_captured = {k: round(v, 1) for k, v in live_angles.items() if v is not None and k in JOINT_NAMES}
        if not valid_captured:
            self.banner_msg = "CANNOT CAPTURE: No full body tracked on webcam!"
            self.banner_color = C_ALERT_RED
            self.banner_timer = 3.0
            return

        for k, v in valid_captured.items():
            self.current_pose.target_angles[k] = v

        self.dirty = True
        self.banner_msg = f"CAPTURED! Updated {len(valid_captured)} angles from live camera pose."
        self.banner_color = C_NEON_CYAN
        self.banner_timer = 3.5

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(30) / 1000.0
            if self.banner_timer > 0:
                self.banner_timer -= dt

            # Fetch live camera frame and landmarks
            cam_frame = None
            player_angles: dict[str, float | None] = {}
            player_landmarks_norm = None
            player_head = None
            tracked = False

            if self.cam_source:
                pose, raw_frame, _ = self.cam_source.snapshot()
                tracked = pose.tracked
                if raw_frame is not None:
                    cam_frame = raw_frame
                if pose.landmarks_3d:
                    player_landmarks_norm = pose.landmarks_3d
                    player_angles = extract_pose_angles(pose.landmarks_3d, mirrored=True, planar_2d=True)
                    if len(pose.landmarks_3d) > 0:
                        player_head = (pose.landmarks_3d[0][0], pose.landmarks_3d[0][1])

            # Keyboard & Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    # Save
                    elif event.key == pygame.K_s and (pygame.key.get_mods() & pygame.KMOD_CTRL or pygame.key.get_mods() & pygame.KMOD_META):
                        self.save()
                    elif event.key == pygame.K_s and not (pygame.key.get_mods() & (pygame.KMOD_SHIFT | pygame.KMOD_CTRL)):
                        self.save()

                    # Revert
                    elif event.key == pygame.K_r:
                        self.revert()

                    # Navigate Poses
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        self.current_idx = (self.current_idx + 1) % len(self.catalog)
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        self.current_idx = (self.current_idx - 1) % len(self.catalog)

                    # Select Joint
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_TAB:
                        self.selected_joint_idx = (self.selected_joint_idx + 1) % len(JOINT_NAMES)
                    elif event.key == pygame.K_UP:
                        self.selected_joint_idx = (self.selected_joint_idx - 1) % len(JOINT_NAMES)
                    elif pygame.K_1 <= event.key <= pygame.K_8:
                        self.selected_joint_idx = event.key - pygame.K_1

                    # Tune Angle (+/-)
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_w):
                        j_name = JOINT_NAMES[self.selected_joint_idx]
                        curr_v = self.current_pose.target_angles.get(j_name, 120.0)
                        self.current_pose.target_angles[j_name] = round(min(180.0, curr_v + 5.0), 1)
                        self.dirty = True
                    elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_x):
                        j_name = JOINT_NAMES[self.selected_joint_idx]
                        curr_v = self.current_pose.target_angles.get(j_name, 120.0)
                        self.current_pose.target_angles[j_name] = round(max(0.0, curr_v - 5.0), 1)
                        self.dirty = True

                    # Toggle Key Joint
                    elif event.key == pygame.K_k:
                        j_name = JOINT_NAMES[self.selected_joint_idx]
                        if j_name in self.current_pose.key_joints:
                            self.current_pose.key_joints.remove(j_name)
                            self.banner_msg = f"Key Joint REMOVED: {j_name}"
                        else:
                            self.current_pose.key_joints.append(j_name)
                            self.banner_msg = f"Key Joint ADDED: {j_name}"
                        self.dirty = True
                        self.banner_color = C_SUNSET_GOLD
                        self.banner_timer = 2.0

                    # Capture Live Webcam Angles
                    elif event.key == pygame.K_c:
                        self.capture_live(player_angles)

            # Evaluate real-time match against current pose
            match_res: BlazerMatchResult | None = None
            if tracked and player_angles:
                match_res = evaluate_blazer_pose(
                    player_landmarks=player_angles,
                    target_pose=self.current_pose,
                    player_head_norm=player_head,
                    mirrored=True,
                    active_joints=LIMB_6_JOINTS,
                    check_head=False,
                )

            self._render(cam_frame, player_angles, match_res)

        if self.cam_source:
            self.cam_source.stop()
        pygame.quit()

    def _render(
        self,
        cam_frame: np.ndarray | None,
        player_angles: dict[str, float | None],
        match_res: BlazerMatchResult | None,
    ) -> None:
        self.screen.fill(C_DARK_BG)
        win_w, win_h = self.width, self.height

        # Header Bar
        header_h = 56
        pygame.draw.rect(self.screen, (24, 16, 42), (0, 0, win_w, header_h))
        pygame.draw.line(self.screen, C_PANEL_BORDER, (0, header_h), (win_w, header_h), 2)

        title_s = self.font_h1.render("TITANS' ARCADE : POSE STUDIO & ANGLE CALIBRATOR", True, C_SUNSET_GOLD)
        self.screen.blit(title_s, (18, 14))

        dirty_txt = "[UNSAVED CHANGES]" if self.dirty else "[SAVED]"
        dirty_col = C_ALERT_RED if self.dirty else C_NEON_GREEN
        dirty_s = self.font_bold.render(dirty_txt, True, dirty_col)
        self.screen.blit(dirty_s, (win_w - dirty_s.get_width() - 18, 18))

        # Bottom Banner Bar
        banner_h = 36
        banner_y = win_h - banner_h
        pygame.draw.rect(self.screen, (20, 14, 36), (0, banner_y, win_w, banner_h))
        pygame.draw.line(self.screen, C_PANEL_BORDER, (0, banner_y), (win_w, banner_y), 1)

        msg = self.banner_msg if self.banner_timer > 0 else "[LEFT/RIGHT] Browse  |  [UP/DOWN] Joint  |  [+/-] Tune  |  [K] Toggle Key  |  [C] Capture  |  [S] Save"
        col = self.banner_color if self.banner_timer > 0 else C_TEXT_MUTED
        b_surf = self.font_bold.render(msg, True, col)
        self.screen.blit(b_surf, (20, banner_y + 9))

        # Main 3-Column Layout:
        # Col 1: Reference Pose Photo & Skeleton (Width ~340)
        # Col 2: Interactive Angle Editor Panel (Width ~400)
        # Col 3: Live Camera Feed & Match Telemetry (Width ~Rest)
        content_y = header_h + 12
        content_h = win_h - header_h - banner_h - 24

        col1_w = int(win_w * 0.28)
        col2_w = int(win_w * 0.34)
        col3_w = win_w - col1_w - col2_w - 48

        x1 = 16
        x2 = x1 + col1_w + 12
        x3 = x2 + col2_w + 12

        self._render_col1_target_pose(x1, content_y, col1_w, content_h)
        self._render_col2_angle_editor(x2, content_y, col2_w, content_h, player_angles, match_res)
        self._render_col3_live_camera(x3, content_y, col3_w, content_h, cam_frame, match_res)

        pygame.display.flip()

    def _render_col1_target_pose(self, x: int, y: int, w: int, h: int) -> None:
        """Render reference pose cutout, name, difficulty and index."""
        pygame.draw.rect(self.screen, C_PANEL_BG, (x, y, w, h), border_radius=8)
        pygame.draw.rect(self.screen, C_PANEL_BORDER, (x, y, w, h), 2, border_radius=8)

        # Pose Heading
        p = self.current_pose
        idx_str = f"POSE {self.current_idx + 1} / {len(self.catalog)}"
        idx_s = self.font_mini.render(idx_str, True, C_NEON_CYAN)
        self.screen.blit(idx_s, (x + 14, y + 12))

        name_s = self.font_h2.render(p.name[:18].upper(), True, C_SUNSET_GOLD)
        self.screen.blit(name_s, (x + 14, y + 28))

        diff_col = C_NEON_GREEN if p.difficulty == "easy" else (C_AMBER_YELLOW if p.difficulty == "medium" else C_ALERT_RED)
        diff_s = self.font_bold.render(f"[{p.difficulty.upper()}]", True, diff_col)
        self.screen.blit(diff_s, (x + w - diff_s.get_width() - 14, y + 14))

        id_s = self.font_mini.render(f"ID: {p.id}", True, C_TEXT_MUTED)
        self.screen.blit(id_s, (x + 14, y + 50))

        # Photo Box
        img_box_y = y + 74
        img_box_h = h - 90
        img_box_w = w - 28
        pygame.draw.rect(self.screen, (14, 10, 22), (x + 14, img_box_y, img_box_w, img_box_h), border_radius=6)

        # Load reference photo if available
        loaded_img = None
        if p.image_file:
            for cand in [self.poses_dir / p.image_file, PROJECT_ROOT / "data" / "raw_poses" / p.image_file]:
                if cand.exists():
                    try:
                        raw = pygame.image.load(str(cand))
                        scale = min(img_box_w / raw.get_width(), img_box_h / raw.get_height())
                        nw, nh = int(raw.get_width() * scale), int(raw.get_height() * scale)
                        loaded_img = pygame.transform.smoothscale(raw, (nw, nh))
                        break
                    except Exception:
                        pass

        if loaded_img:
            ix = x + 14 + (img_box_w - loaded_img.get_width()) // 2
            iy = img_box_y + (img_box_h - loaded_img.get_height()) // 2
            self.screen.blit(loaded_img, (ix, iy))
        else:
            no_img_s = self.font_bold.render("NO PHOTO PREVIEW", True, C_TEXT_MUTED)
            self.screen.blit(no_img_s, no_img_s.get_rect(center=(x + 14 + img_box_w // 2, img_box_y + img_box_h // 2)))

        pygame.draw.rect(self.screen, (70, 50, 110), (x + 14, img_box_y, img_box_w, img_box_h), 1, border_radius=6)

    def _render_col2_angle_editor(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        player_angles: dict[str, float | None],
        match_res: BlazerMatchResult | None,
    ) -> None:
        """Render list of joints, target angles, adjustment sliders, and key limb toggles."""
        pygame.draw.rect(self.screen, C_PANEL_BG, (x, y, w, h), border_radius=8)
        pygame.draw.rect(self.screen, C_PANEL_BORDER, (x, y, w, h), 2, border_radius=8)

        # Section Header
        hdr = self.font_h2.render("TARGET ANGLES & CALIBRATION", True, C_NEON_CYAN)
        self.screen.blit(hdr, (x + 16, y + 14))

        sub = self.font_mini.render("[W/S] or [+/-] Tune 5 deg  |  [K] Toggle Key Joint", True, C_TEXT_MUTED)
        self.screen.blit(sub, (x + 16, y + 36))

        p = self.current_pose
        row_y = y + 64
        row_h = (h - 80) // len(JOINT_NAMES)

        for i, j_name in enumerate(JOINT_NAMES):
            is_selected = i == self.selected_joint_idx
            is_key = j_name in p.key_joints
            tgt_val = p.target_angles.get(j_name, 120.0)
            live_val = player_angles.get(j_name)

            # Row background highlight
            rx = x + 12
            rw = w - 24
            ry = row_y + i * row_h
            if is_selected:
                pygame.draw.rect(self.screen, (52, 34, 86), (rx, ry, rw, row_h - 4), border_radius=6)
                pygame.draw.rect(self.screen, C_SUNSET_GOLD, (rx, ry, rw, row_h - 4), 2, border_radius=6)
            else:
                pygame.draw.rect(self.screen, (22, 16, 38), (rx, ry, rw, row_h - 4), border_radius=6)

            # Number badge
            num_s = self.font_bold.render(f"[{i + 1}]", True, C_SUNSET_GOLD if is_selected else C_TEXT_MUTED)
            self.screen.blit(num_s, (rx + 8, ry + 8))

            # Joint Name
            disp_name = j_name.replace("_", " ").title()
            j_col = C_TEXT_WHITE if is_selected else (220, 215, 235)
            name_s = self.font_bold.render(disp_name, True, j_col)
            self.screen.blit(name_s, (rx + 42, ry + 8))

            # Key Joint Tag
            if is_key:
                key_s = self.font_mini.render("KEY LIMB", True, (20, 10, 30))
                tag_w = key_s.get_width() + 10
                pygame.draw.rect(self.screen, C_SUNSET_GOLD, (rx + 175, ry + 8, tag_w, 18), border_radius=3)
                self.screen.blit(key_s, (rx + 180, ry + 10))

            # Angle Value
            val_str = f"{tgt_val:5.1f} deg"
            val_s = self.font_bold.render(val_str, True, C_NEON_CYAN if is_selected else C_TEXT_WHITE)
            self.screen.blit(val_s, (rx + rw - 110, ry + 8))

            # Visual bar representing 0..180 degrees
            bar_w = rw - 20
            bar_h = 6
            bar_x = rx + 10
            bar_y = ry + row_h - 14
            pygame.draw.rect(self.screen, (40, 30, 60), (bar_x, bar_y, bar_w, bar_h), border_radius=3)

            fill_w = int(bar_w * (tgt_val / 180.0))
            pygame.draw.rect(self.screen, C_NEON_CYAN, (bar_x, bar_y, fill_w, bar_h), border_radius=3)

            # Mark live angle on the bar if available
            if live_val is not None:
                live_x = bar_x + int(bar_w * (live_val / 180.0))
                pygame.draw.circle(self.screen, C_SUNSET_PINK, (live_x, bar_y + bar_h // 2), 5)

    def _render_col3_live_camera(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        cam_frame: np.ndarray | None,
        match_res: BlazerMatchResult | None,
    ) -> None:
        """Render live camera preview, real-time match percentage, and capture hotkey."""
        pygame.draw.rect(self.screen, C_PANEL_BG, (x, y, w, h), border_radius=8)
        pygame.draw.rect(self.screen, C_PANEL_BORDER, (x, y, w, h), 2, border_radius=8)

        # Header
        hdr = self.font_h2.render("LIVE CALIBRATION & WEBCAM", True, C_SUNSET_PINK)
        self.screen.blit(hdr, (x + 16, y + 14))

        sub = self.font_mini.render("Press [C] to SNAP live pose into target!", True, C_SUNSET_GOLD)
        self.screen.blit(sub, (x + 16, y + 36))

        cam_box_y = y + 60
        cam_box_h = int(h * 0.52)
        cam_box_w = w - 32
        pygame.draw.rect(self.screen, (10, 8, 18), (x + 16, cam_box_y, cam_box_w, cam_box_h), border_radius=6)

        if cam_frame is not None:
            # Convert BGR to RGB for Pygame
            rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
            c_surf = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
            c_scaled = pygame.transform.smoothscale(c_surf, (cam_box_w, cam_box_h))
            self.screen.blit(c_scaled, (x + 16, cam_box_y))
        else:
            no_cam_s = self.font_bold.render("CAMERA INACTIVE / NO-CAM MODE", True, C_TEXT_MUTED)
            self.screen.blit(no_cam_s, no_cam_s.get_rect(center=(x + 16 + cam_box_w // 2, cam_box_y + cam_box_h // 2)))

        pygame.draw.rect(self.screen, C_SUNSET_PINK, (x + 16, cam_box_y, cam_box_w, cam_box_h), 2, border_radius=6)

        # Real-time Match Score Box
        match_y = cam_box_y + cam_box_h + 12
        match_h = h - (match_y - y) - 16
        pygame.draw.rect(self.screen, (20, 14, 34), (x + 16, match_y, cam_box_w, match_h), border_radius=6)
        pygame.draw.rect(self.screen, (60, 42, 90), (x + 16, match_y, cam_box_w, match_h), 1, border_radius=6)

        if match_res is not None:
            pct = match_res.match_pct
            status_txt = "MATCH PASS" if match_res.is_matched else "TRYING..."
            status_col = C_NEON_GREEN if match_res.is_matched else C_AMBER_YELLOW

            pct_s = self.font_h1.render(f"SIMILARITY: {pct:.1f}%", True, status_col)
            self.screen.blit(pct_s, (x + 28, match_y + 12))

            pass_s = self.font_bold.render(f"STATUS: {status_txt}", True, status_col)
            self.screen.blit(pass_s, (x + 28, match_y + 40))

            hint_s = self.font_bold.render(f"HINT: {match_res.primary_hint}", True, C_SUNSET_GOLD)
            self.screen.blit(hint_s, (x + 28, match_y + 64))
        else:
            prompt_s = self.font_bold.render("Stand in front of webcam to test live match", True, C_TEXT_MUTED)
            self.screen.blit(prompt_s, (x + 28, match_y + 24))

        # Big "CAPTURE [C]" Button Callout
        btn_y = match_y + match_h - 40
        pygame.draw.rect(self.screen, (40, 24, 60), (x + 26, btn_y, cam_box_w - 20, 30), border_radius=4)
        pygame.draw.rect(self.screen, C_SUNSET_GOLD, (x + 26, btn_y, cam_box_w - 20, 30), 1, border_radius=4)
        cap_s = self.font_bold.render("PRESS [C] TO CAPTURE CURRENT WEBCAM AS TARGET", True, C_SUNSET_GOLD)
        self.screen.blit(cap_s, cap_s.get_rect(center=(x + 26 + (cam_box_w - 20) // 2, btn_y + 15)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Titans' Arcade: Internal Pose Studio & Angle Tuner")
    parser.add_argument("--catalog", type=str, default="data/poses_index.json", help="Path to catalog JSON")
    parser.add_argument("--poses-dir", type=str, default="data/poses", help="Path to pose images")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--no-cam", action="store_true", help="Launch without webcam capture")
    args = parser.parse_args()

    app = PoseStudioApp(
        catalog_path=args.catalog,
        poses_dir=args.poses_dir,
        cam_index=args.camera,
        no_cam=args.no_cam,
    )
    app.run()


if __name__ == "__main__":
    main()
