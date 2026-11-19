"""Souvenir photo card generator for Titans' Arcade.
Generates high-resolution portrait PNG souvenir cards (1080x1350) in Sunset Arcade theme.
100% offline, zero external CDN / network calls.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _get_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.ImageFont:
    """Safely load a TrueType font with fallbacks for Windows, Linux, and headless environments."""
    static_dir = Path(__file__).resolve().parent.parent / "server" / "static"
    silkscreen = str(static_dir / "silkscreen.ttf")

    font_candidates: list[str] = []
    if mono:
        font_candidates = [
            silkscreen,
            "consola.ttf",
            "consolab.ttf" if bold else "consola.ttf",
            "Courier New.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    elif bold:
        font_candidates = [
            "segoeuib.ttf",
            "ariblk.ttf",
            "arialbd.ttf",
            "arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    else:
        font_candidates = [
            "segoeui.ttf",
            "arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]

    for name in font_candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_sunset_gradient(width: int, height: int) -> Image.Image:
    """Generate a procedural Sunset Arcade gradient background."""
    base = Image.new("RGBA", (width, height), (32, 17, 56, 255))
    draw = ImageDraw.Draw(base)

    # Gradient color stops: twilight -> magenta -> peach -> gold
    stops = [
        (0.0, (35, 18, 65)),
        (0.35, (72, 34, 98)),
        (0.65, (168, 62, 108)),
        (0.85, (234, 125, 139)),
        (1.0, (255, 209, 82)),
    ]

    for y in range(height):
        pos = y / float(height - 1)
        # Find enclosing stops
        c1, c2 = stops[0][1], stops[-1][1]
        t = 0.0
        for i in range(len(stops) - 1):
            if stops[i][0] <= pos <= stops[i + 1][0]:
                c1 = stops[i][1]
                c2 = stops[i + 1][1]
                seg_span = stops[i + 1][0] - stops[i][0]
                t = (pos - stops[i][0]) / seg_span if seg_span > 0 else 0.0
                break
        r = int(c1[0] + (c2[0] - c1[0]) * t)
        g = int(c1[1] + (c2[1] - c1[1]) * t)
        b = int(c1[2] + (c2[2] - c1[2]) * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

    return base


def _load_player_and_scores(
    player_id: str, data_dir: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load player info and highest scores per game for `player_id`."""
    players_file = data_dir / "players.json"
    if not players_file.exists():
        raise ValueError(f"Player {player_id} not found: players.json missing")

    try:
        players = json.loads(players_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Could not read players.json: {e}")

    player = None
    for p in players:
        if p.get("player_id") == player_id:
            player = p
            break

    if not player:
        raise ValueError(f"Player {player_id} not found in players.json")

    # Load scores from data/scores.jsonl
    scores_file = data_dir / "scores.jsonl"
    best_scores: dict[str, dict[str, Any]] = {}
    if scores_file.exists():
        try:
            for line in scores_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("player_id") == player_id:
                        game = record.get("game", "")
                        if not game:
                            continue
                        candidate_key = (
                            record.get("score", 0),
                            record.get("accuracy", 0.0),
                            record.get("best_streak", 0),
                        )
                        existing = best_scores.get(game)
                        if existing is None:
                            best_scores[game] = record
                        else:
                            existing_key = (
                                existing.get("score", 0),
                                existing.get("accuracy", 0.0),
                                existing.get("best_streak", 0),
                            )
                            if candidate_key > existing_key:
                                best_scores[game] = record
                except Exception:
                    continue
        except Exception:
            pass

    return player, best_scores


def generate_card(
    player_id: str,
    data_dir: Path | str = "data",
    output_dir: Path | str = "data/cards",
    base_url: str = "",
) -> Path:
    """Generate high-resolution portrait souvenir card (1080x1350) for `player_id`.

    Args:
        player_id: e.g. "P001"
        data_dir: Directory containing players.json, photos/, and scores.jsonl
        output_dir: Directory where the card PNG will be saved
        base_url: Base URL (e.g. "http://192.168.1.10:8000") for the QR code link

    Returns:
        Path to the generated PNG file.
    """
    data_path = Path(data_dir)
    out_path_dir = Path(output_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    player, best_scores = _load_player_and_scores(player_id, data_path)

    width = 1080
    height = 1350

    # 1. Base Canvas & Background
    bg_static = Path(__file__).resolve().parent.parent / "server" / "static" / "bg-sunset.png"
    if bg_static.exists():
        try:
            raw_bg = Image.open(bg_static).convert("RGBA")
            # Scale & crop to 1080x1350
            scale = max(width / raw_bg.width, height / raw_bg.height)
            new_w, new_h = int(raw_bg.width * scale), int(raw_bg.height * scale)
            bg_scaled = raw_bg.resize((new_w, new_h), Image.Resampling.LANCZOS)
            x_crop = (new_w - width) // 2
            y_crop = (new_h - height) // 2
            canvas = bg_scaled.crop((x_crop, y_crop, x_crop + width, y_crop + height))

            # Darken twilight overlay for high text contrast
            overlay = Image.new("RGBA", (width, height), (28, 14, 48, 180))
            canvas = Image.alpha_composite(canvas, overlay)
        except Exception:
            canvas = _draw_sunset_gradient(width, height)
    else:
        canvas = _draw_sunset_gradient(width, height)

    draw = ImageDraw.Draw(canvas)

    # 2. CRT Twilight Frame Borders
    # Outer Border (Neon Pink #ea7d8b)
    draw.rounded_rectangle([36, 36, width - 36, height - 36], radius=22, outline="#ea7d8b", width=4)
    # Inner Border (Sunset Gold #ffd152)
    draw.rounded_rectangle([44, 44, width - 44, height - 44], radius=16, outline="#ffd152", width=2)

    # Corner CRT Accents (Cyan #00f0ff)
    corner_len = 24
    cyan = "#00f0ff"
    # TL
    draw.line([(32, 32), (32 + corner_len, 32)], fill=cyan, width=4)
    draw.line([(32, 32), (32, 32 + corner_len)], fill=cyan, width=4)
    # TR
    draw.line([(width - 32, 32), (width - 32 - corner_len, 32)], fill=cyan, width=4)
    draw.line([(width - 32, 32), (width - 32, 32 + corner_len)], fill=cyan, width=4)
    # BL
    draw.line([(32, height - 32), (32 + corner_len, height - 32)], fill=cyan, width=4)
    draw.line([(32, height - 32), (32, height - 32 - corner_len)], fill=cyan, width=4)
    # BR
    draw.line([(width - 32, height - 32), (width - 32 - corner_len, height - 32)], fill=cyan, width=4)
    draw.line([(width - 32, height - 32), (width - 32, height - 32 - corner_len)], fill=cyan, width=4)

    # 3. Header Section (y = 56 to 160)
    font_sub_top = _get_font(18, bold=True, mono=True)
    top_badge = "★ TEQFEST 2K26 // TITANS' ARCADE ★"
    draw.text((width // 2, 72), top_badge, fill="#ffd152", font=font_sub_top, anchor="mt")

    font_title = _get_font(52, bold=True)
    # Drop shadow
    draw.text((width // 2 + 3, 102 + 3), "TITANS' ARCADE", fill="#180a2b", font=font_title, anchor="mt")
    draw.text((width // 2, 102), "TITANS' ARCADE", fill="#ea7d8b", font=font_title, anchor="mt")

    # Header accent bar
    bar_y = 168
    draw.line([(80, bar_y), (width - 80, bar_y)], fill="#48327a", width=2)
    draw.line([(width // 2 - 120, bar_y), (width // 2 + 120, bar_y)], fill="#ffd152", width=4)

    # 4. Player Photo Box (y = 186 to 636, centered, w=480, h=440)
    pw, ph = 480, 440
    px1 = (width - pw) // 2
    py1 = 188
    px2 = px1 + pw
    py2 = py1 + ph

    photo_rendered = False
    photo_file = data_path / "photos" / f"{player_id}.jpg"
    if photo_file.exists():
        try:
            raw_photo = Image.open(photo_file)
            photo = ImageOps.exif_transpose(raw_photo).convert("RGBA")
            scale = max(pw / photo.width, ph / photo.height)
            scaled_w, scaled_h = int(photo.width * scale), int(photo.height * scale)
            photo_scaled = photo.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
            x_crop = (scaled_w - pw) // 2
            y_crop = (scaled_h - ph) // 2
            photo_cropped = photo_scaled.crop((x_crop, y_crop, x_crop + pw, y_crop + ph))
            canvas.paste(photo_cropped, (px1, py1), photo_cropped)
            photo_rendered = True
        except Exception:
            photo_rendered = False

    if not photo_rendered:
        # Retro Avatar Placeholder
        draw.rectangle([px1, py1, px2, py2], fill="#201138")
        # Stylized Silhouette
        center_x = (px1 + px2) // 2
        # Head
        draw.ellipse([center_x - 60, py1 + 100, center_x + 60, py1 + 220], fill="#48327a", outline="#9b74b8", width=3)
        # Shoulders
        draw.chord([center_x - 130, py1 + 200, center_x + 130, py1 + 440], 180, 360, fill="#35235d", outline="#ea7d8b", width=3)
        font_no_photo = _get_font(20, bold=True, mono=True)
        draw.text((center_x, py1 + 370), "[ NO PHOTO RECORDED ]", fill="#ffd152", font=font_no_photo, anchor="mt")

    # Photo Viewfinder Framing & Badges
    draw.rectangle([px1, py1, px2, py2], outline="#00f0ff", width=3)
    # Corner viewfinder clips
    vf_len = 20
    draw.line([(px1, py1), (px1 + vf_len, py1)], fill="#ffd152", width=5)
    draw.line([(px1, py1), (px1, py1 + vf_len)], fill="#ffd152", width=5)
    draw.line([(px2, py1), (px2 - vf_len, py1)], fill="#ffd152", width=5)
    draw.line([(px2, py1), (px2, py1 + vf_len)], fill="#ffd152", width=5)
    draw.line([(px1, py2), (px1 + vf_len, py2)], fill="#ffd152", width=5)
    draw.line([(px1, py2), (px1, py2 - vf_len)], fill="#ffd152", width=5)
    draw.line([(px2, py2), (px2 - vf_len, py2)], fill="#ffd152", width=5)
    draw.line([(px2, py2), (px2, py2 - vf_len)], fill="#ffd152", width=5)

    # Viewfinder Tag Badges
    font_vf = _get_font(12, bold=True, mono=True)
    draw.rectangle([px1 + 12, py1 + 12, px1 + 96, py1 + 32], fill="#241442")
    draw.ellipse([px1 + 20, py1 + 18, px1 + 28, py1 + 26], fill="#58c038")
    draw.text((px1 + 34, py1 + 16), "REC 720P", fill="#fbcfe8", font=font_vf)

    draw.rectangle([px2 - 130, py1 + 12, px2 - 12, py1 + 32], fill="#241442")
    draw.text((px2 - 71, py1 + 16), "SENSOR [OK]", fill="#00f0ff", font=font_vf, anchor="mt")

    # 5. Player Identity Box (y = 648 to 766)
    font_pid = _get_font(34, bold=True, mono=True)
    pid_str = player.get("player_id", player_id)
    draw.text((width // 2, 652), f"[ {pid_str} ]", fill="#00f0ff", font=font_pid, anchor="mt")

    font_name = _get_font(46, bold=True)
    name_str = (player.get("name") or "CHALLENGER").upper()
    draw.text((width // 2 + 2, 694 + 2), name_str, fill="#180a2b", font=font_name, anchor="mt")
    draw.text((width // 2, 694), name_str, fill="#ffffff", font=font_name, anchor="mt")

    # URN pill badge (Height removed per spec)
    roll_no = player.get("roll_no")
    if roll_no:
        font_pill = _get_font(18, bold=True, mono=True)
        draw.text((width // 2, 750), f"URN: {roll_no}", fill="#ffd152", font=font_pill, anchor="mt")

    # 6. Tournament Performance Section (2 Games: Floor is Lava & Pose Blazer)
    font_sec = _get_font(18, bold=True, mono=True)
    draw.text((width // 2, 788), "-- ARCADE TOURNAMENT STATS --", fill="#ea7d8b", font=font_sec, anchor="mt")

    card_y = 818
    card_h = 280
    card_w = 444
    card_margin_x = 76
    card_gap = 40

    games_info = [
        ("dodge", "FLOOR IS LAVA", "Station A"),
        ("blazer", "POSE BLAZER", "Station B"),
    ]

    font_gh = _get_font(22, bold=True)
    font_gs = _get_font(13, mono=True)
    font_score = _get_font(48, bold=True)
    font_pts = _get_font(13, bold=True, mono=True)
    font_stat = _get_font(16, bold=True, mono=True)

    for i, (g_key, g_title, g_station) in enumerate(games_info):
        cx1 = card_margin_x + i * (card_w + card_gap)
        cx2 = cx1 + card_w
        cy1 = card_y
        cy2 = cy1 + card_h

        rec = best_scores.get(g_key)
        if not rec and g_key == "blazer":
            rec = best_scores.get("wall") or best_scores.get("pose_blazer")
        has_score = rec is not None

        # Panel Background
        draw.rounded_rectangle([cx1, cy1, cx2, cy2], radius=12, fill="#201138", outline="#5a3c8c", width=2)
        draw.rounded_rectangle([cx1 + 2, cy1 + 2, cx2 - 2, cy2 - 2], radius=10, outline="#35235d", width=1)

        # Header tab banner
        header_fill = "#35235d" if not has_score else "#48327a"
        draw.rounded_rectangle([cx1, cy1, cx2, cy1 + 46], radius=12, fill=header_fill)
        draw.rectangle([cx1, cy1 + 22, cx2, cy1 + 46], fill=header_fill)
        draw.text(((cx1 + cx2) // 2, cy1 + 10), g_title, fill="#ffd152", font=font_gh, anchor="mt")
        draw.text(((cx1 + cx2) // 2, cy1 + 34), g_station.upper(), fill="#d8b4fe", font=font_gs, anchor="mt")

        # Divider under header
        draw.line([(cx1, cy1 + 52), (cx2, cy1 + 52)], fill="#5a3c8c", width=1)

        if has_score and rec:
            score_val = rec.get("score", 0)
            acc_val = rec.get("accuracy", 0.0)
            if acc_val <= 1.0:
                acc_str = f"{acc_val * 100:.1f}%"
            else:
                acc_str = f"{acc_val:.1f}%"
            streak_val = rec.get("best_streak", 0)

            # Score Number
            score_color = "#58c038" if score_val > 0 else "#ffffff"
            draw.text(((cx1 + cx2) // 2, cy1 + 72), f"{score_val:,}", fill=score_color, font=font_score, anchor="mt")
            draw.text(((cx1 + cx2) // 2, cy1 + 126), "POINTS", fill="#d8b4fe", font=font_pts, anchor="mt")

            # Metrics
            draw.line([(cx1 + 28, cy1 + 148), (cx2 - 28, cy1 + 148)], fill="#48327a", width=1)
            draw.text(((cx1 + cx2) // 2, cy1 + 168), f"ACCURACY: {acc_str}", fill="#00f0ff", font=font_stat, anchor="mt")
            draw.text(((cx1 + cx2) // 2, cy1 + 202), f"MAX STREAK: {streak_val}", fill="#ea7d8b", font=font_stat, anchor="mt")
            draw.text(((cx1 + cx2) // 2, cy1 + 242), "RANKED RECORD", fill="#ffd152", font=font_gs, anchor="mt")
        else:
            draw.text(((cx1 + cx2) // 2, cy1 + 84), "--", fill="#6b5b95", font=font_score, anchor="mt")
            draw.text(((cx1 + cx2) // 2, cy1 + 134), "NOT PLAYED YET", fill="#8570ab", font=font_pts, anchor="mt")
            draw.line([(cx1 + 28, cy1 + 158), (cx2 - 28, cy1 + 158)], fill="#48327a", width=1)
            draw.text(((cx1 + cx2) // 2, cy1 + 180), "ACCURACY: --", fill="#6a548e", font=font_stat, anchor="mt")
            draw.text(((cx1 + cx2) // 2, cy1 + 214), "MAX STREAK: --", fill="#6a548e", font=font_stat, anchor="mt")
            draw.text(((cx1 + cx2) // 2, cy1 + 248), "STANDBY", fill="#6a548e", font=font_gs, anchor="mt")

    # 7. Bottom Verification & Event Archive Bar (y = 1130 to 1290, QR removed per spec)
    bot_y1 = 1130
    bot_y2 = 1290
    draw.rounded_rectangle([76, bot_y1, width - 76, bot_y2], radius=14, fill="#1c0e30", outline="#ffd152", width=2)
    draw.rounded_rectangle([80, bot_y1 + 4, width - 80, bot_y2 - 4], radius=10, outline="#48327a", width=1)

    font_b1 = _get_font(22, bold=True)
    font_b2 = _get_font(15, mono=True, bold=True)
    font_b3 = _get_font(14, mono=True)
    font_b4 = _get_font(13, mono=True)

    mid_x = width // 2
    draw.text((mid_x, bot_y1 + 22), "TITANS' ARCADE // TEQFEST 2K26", fill="#ffffff", font=font_b1, anchor="mt")
    draw.text((mid_x, bot_y1 + 58), "100% OFFLINE LOCAL ARENA ARCHIVE", fill="#58c038", font=font_b2, anchor="mt")
    draw.text((mid_x, bot_y1 + 88), f"VERIFIED PLAYER PASS: #TA-2026-{player_id}", fill="#00f0ff", font=font_b3, anchor="mt")
    draw.text((mid_x, bot_y1 + 116), "OFFICIAL TOURNAMENT RECORD // LAPTOP 3 SCORE SERVER", fill="#d8b4fe", font=font_b4, anchor="mt")

    # 8. CRT Scanline Overlay
    scanline_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(scanline_overlay)
    for y in range(0, height, 5):
        s_draw.line([(0, y), (width, y)], fill=(0, 0, 0, 24))
    canvas = Image.alpha_composite(canvas, scanline_overlay)

    # 9. Save as RGB PNG
    final_card = canvas.convert("RGB")
    dest_path = out_path_dir / f"{player_id}.png"
    final_card.save(dest_path, format="PNG", optimize=True)
    return dest_path
