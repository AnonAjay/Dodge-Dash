"""QR Code generator for Titans' Arcade souvenir cards.
100% offline-capable, zero external CDN / JS dependencies. Uses OpenCV QR encoder and Pillow.
Supports encoding direct public cloud image URLs when online, or local LAN download endpoints when offline.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def resolve_card_qr_url(
    player_id: str,
    card_path: Path | None = None,
    base_url: str = "",
    data_dir: Path | str = "data",
) -> str:
    """Resolve direct card PNG download URL for QR code encoding.

    Prefers PUBLIC_CARD_BASE_URL if set; then cached or uploaded public direct PNG URL;
    falls back to local LAN server endpoint when offline.
    """
    public_base = os.environ.get("PUBLIC_CARD_BASE_URL", "").strip()
    if public_base:
        return f"{public_base.rstrip('/')}/{player_id}.png"

    if card_path is not None and Path(card_path).exists():
        from cards.uploader import get_or_upload_public_card_url

        uploaded = get_or_upload_public_card_url(Path(card_path), player_id, data_dir=data_dir)
        if uploaded:
            return uploaded
    else:
        from cards.uploader import get_cached_card_url

        cached = get_cached_card_url(player_id, data_dir=data_dir)
        if cached:
            return cached

    if base_url:
        return f"{base_url.rstrip('/')}/api/cards/{player_id}/download"
    return f"http://127.0.0.1:8000/api/cards/{player_id}/download"


def generate_qr_image(
    data: str,
    box_size: int = 8,
    border: int = 4,
    fill_color: tuple[int, int, int] = (0, 0, 0),
    back_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Generate a PIL Image containing a QR code for `data`.

    Args:
        data: The URL or text string to encode.
        box_size: Multiplier for pixel size per module (default 8).
        border: Quiet zone module width around the QR code (default 4).
        fill_color: RGB tuple for dark modules (default black).
        back_color: RGB tuple for light modules (default white).

    Returns:
        PIL.Image in RGB mode.
    """
    if not data or not isinstance(data, str) or not data.strip():
        raise ValueError("QR data string cannot be empty")

    encoder = cv2.QRCodeEncoder_create()
    matrix = encoder.encode(data.strip())
    if matrix is None:
        raise ValueError(f"OpenCV QRCodeEncoder failed to encode data: {data}")

    # matrix has values: 0 for dark module, 255 for light module
    # Add quiet-zone border (white / 255)
    padded = cv2.copyMakeBorder(
        matrix,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    if box_size > 1:
        padded = cv2.resize(
            padded,
            (0, 0),
            fx=box_size,
            fy=box_size,
            interpolation=cv2.INTER_NEAREST,
        )

    # Convert to RGB image with custom colors if requested
    if fill_color == (0, 0, 0) and back_color == (255, 255, 255):
        return Image.fromarray(padded, mode="L").convert("RGB")

    h, w = padded.shape
    rgb_arr = np.zeros((h, w, 3), dtype=np.uint8)
    is_dark = padded == 0
    rgb_arr[is_dark] = fill_color
    rgb_arr[~is_dark] = back_color
    return Image.fromarray(rgb_arr, mode="RGB")


def generate_qr_png_bytes(
    data: str,
    box_size: int = 8,
    border: int = 4,
    fill_color: tuple[int, int, int] = (0, 0, 0),
    back_color: tuple[int, int, int] = (255, 255, 255),
) -> bytes:
    """Generate PNG bytes of the QR code for `data`."""
    image = generate_qr_image(
        data,
        box_size=box_size,
        border=border,
        fill_color=fill_color,
        back_color=back_color,
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
