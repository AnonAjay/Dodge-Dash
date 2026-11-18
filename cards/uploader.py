"""Automatic public card uploader for Titans' Arcade.
Uploads generated souvenir PNG cards to public hosts prioritizing pure direct-image links:
1. Primary: Litterbox (72h direct PNG: https://litter.catbox.moe/<id>.png)
2. Secondary fallback: Catbox (permanent direct PNG: https://files.catbox.moe/<id>.png)
3. Tertiary last-resort fallback: Tmpfiles.org (https://tmpfiles.org/dl/<id>/<name>)

Ensures 5G mobile scans open the raw .png image directly in the phone browser
instead of a third-party wrapper webpage.
Fails silently when offline so 100% offline local LAN operation is never blocked.
"""
from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
import threading
from typing import Any
import urllib.error
import urllib.request
import uuid

LITTERBOX_API_URL = "https://litterbox.catbox.moe/resources/internals/api.php"
CATBOX_API_URL = "https://catbox.moe/user/api.php"
TMPFILES_API_URL = "https://tmpfiles.org/api/v1/upload"
USER_AGENT = "TitansArcade/1.0"

_url_lock = threading.Lock()


def _get_urls_file(data_dir: Path | str) -> Path:
    return Path(data_dir) / "card_urls.json"


def get_cached_card_entry(player_id: str, data_dir: Path | str = "data") -> dict[str, Any] | None:
    """Return cached dictionary entry for player_id if present."""
    urls_file = _get_urls_file(data_dir)
    with _url_lock:
        if not urls_file.exists():
            return None
        try:
            data = json.loads(urls_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                val = data.get(player_id)
                if isinstance(val, dict):
                    return val
                if isinstance(val, str):
                    return {"url": val, "mtime": 0.0}
        except Exception:
            return None
    return None


def get_cached_card_url(player_id: str, data_dir: Path | str = "data") -> str | None:
    """Return cached public URL for player_id if present."""
    entry = get_cached_card_entry(player_id, data_dir=data_dir)
    if entry:
        return entry.get("url")
    return None


def set_cached_card_url(
    player_id: str,
    url: str,
    mtime: float | None = None,
    data_dir: Path | str = "data",
) -> None:
    """Cache public URL and file mtime for player_id in data/card_urls.json."""
    urls_file = _get_urls_file(data_dir)
    with _url_lock:
        try:
            urls: dict[str, Any] = {}
            if urls_file.exists():
                try:
                    loaded = json.loads(urls_file.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        urls = loaded
                except Exception:
                    urls = {}
            if mtime is None:
                card_p = Path(data_dir) / "cards" / f"{player_id}.png"
                if card_p.exists():
                    mtime = card_p.stat().st_mtime
                else:
                    mtime = 0.0

            urls[player_id] = {
                "url": url,
                "mtime": mtime,
            }
            urls_file.parent.mkdir(parents=True, exist_ok=True)
            urls_file.write_text(json.dumps(urls, indent=2), encoding="utf-8")
        except Exception:
            pass


def invalidate_card_url(player_id: str, data_dir: Path | str = "data") -> None:
    """Invalidate cached public URL when a card is regenerated with new scores."""
    urls_file = _get_urls_file(data_dir)
    with _url_lock:
        if not urls_file.exists():
            return
        try:
            data = json.loads(urls_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and player_id in data:
                del data[player_id]
                urls_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass


def _build_multipart_payload(
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
) -> tuple[bytes, str]:
    """Construct multipart/form-data payload using standard library only."""
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{value}\r\n".encode("utf-8"))

    filename = file_path.name
    content_type = mimetypes.guess_type(filename)[0] or "image/png"
    file_bytes = file_path.read_bytes()

    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode(
            "utf-8"
        )
    )
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.extend(file_bytes)
    body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    content_type_header = f"multipart/form-data; boundary={boundary}"
    return bytes(body), content_type_header


def _upload_to_litterbox(card_path: Path, timeout: float = 12.0) -> str | None:
    payload_bytes, content_type_hdr = _build_multipart_payload(
        fields={"reqtype": "fileupload", "time": "72h"},
        file_field="fileToUpload",
        file_path=card_path,
    )
    req = urllib.request.Request(
        LITTERBOX_API_URL,
        data=payload_bytes,
        headers={
            "Content-Type": content_type_hdr,
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        resp_text = response.read().decode("utf-8", errors="replace").strip()
        if resp_text.startswith("http://") or resp_text.startswith("https://"):
            return resp_text
    return None


def _upload_to_catbox(card_path: Path, timeout: float = 12.0) -> str | None:
    payload_bytes, content_type_hdr = _build_multipart_payload(
        fields={"reqtype": "fileupload"},
        file_field="fileToUpload",
        file_path=card_path,
    )
    req = urllib.request.Request(
        CATBOX_API_URL,
        data=payload_bytes,
        headers={
            "Content-Type": content_type_hdr,
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        resp_text = response.read().decode("utf-8", errors="replace").strip()
        if resp_text.startswith("http://") or resp_text.startswith("https://"):
            return resp_text
    return None


def _upload_to_tmpfiles(card_path: Path, timeout: float = 12.0) -> str | None:
    payload_bytes, content_type_hdr = _build_multipart_payload(
        fields={},
        file_field="file",
        file_path=card_path,
    )
    req = urllib.request.Request(
        TMPFILES_API_URL,
        data=payload_bytes,
        headers={
            "Content-Type": content_type_hdr,
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        resp_text = response.read().decode("utf-8", errors="replace").strip()
        try:
            data = json.loads(resp_text)
            if isinstance(data, dict):
                inner = data.get("data")
                raw_url = inner.get("url") if isinstance(inner, dict) else None
                if not raw_url and isinstance(inner, str):
                    raw_url = inner
                if raw_url and (raw_url.startswith("http://") or raw_url.startswith("https://")):
                    if "://tmpfiles.org/" in raw_url and "://tmpfiles.org/dl/" not in raw_url:
                        return raw_url.replace("://tmpfiles.org/", "://tmpfiles.org/dl/", 1)
                    return raw_url
        except Exception:
            return None
    return None


def get_or_upload_public_card_url(
    card_path: Path,
    player_id: str,
    data_dir: Path | str = "data",
    timeout: float = 12.0,
) -> str | None:
    """Upload card PNG to temporary public host, prioritizing pure direct-image hosts:
    1. Litterbox (72h direct PNG)
    2. Catbox (permanent direct PNG fallback)
    3. Tmpfiles.org (last resort)

    Automatically upgrades/invalidates stale tmpfiles.org URLs if Litterbox or Catbox is reachable.
    Checks cached mtime to avoid re-uploading identical files.
    Fails silently and returns None when offline or disabled.
    """
    if os.environ.get("DISABLE_PUBLIC_UPLOAD") == "1":
        return None

    card_p = Path(card_path)
    if not card_p.exists():
        return None

    card_mtime = card_p.stat().st_mtime

    cached_entry = get_cached_card_entry(player_id, data_dir=data_dir)
    stale_tmpfiles_url: str | None = None
    if cached_entry and isinstance(cached_entry, dict):
        cached_url = cached_entry.get("url")
        cached_mtime = cached_entry.get("mtime")
        if cached_url and cached_mtime is not None:
            if abs(float(cached_mtime) - float(card_mtime)) < 0.001:
                # If cached URL is a pure direct PNG (Litterbox or Catbox), return it immediately
                if "tmpfiles.org" not in cached_url:
                    return cached_url
                # If cached URL was tmpfiles.org, remember it but try upgrading to direct-PNG first
                stale_tmpfiles_url = cached_url

    # 1. Primary host: Litterbox (72h direct PNG)
    url: str | None = None
    try:
        url = _upload_to_litterbox(card_p, timeout=timeout)
    except Exception:
        url = None

    # 2. Secondary host: Catbox (permanent direct PNG fallback)
    if not url:
        try:
            url = _upload_to_catbox(card_p, timeout=timeout)
        except Exception:
            url = None

    # 3. Tertiary fallback: Tmpfiles (only if both Litterbox and Catbox fail)
    if not url:
        if stale_tmpfiles_url:
            url = stale_tmpfiles_url
        else:
            try:
                url = _upload_to_tmpfiles(card_p, timeout=timeout)
            except Exception:
                url = None

    if url:
        set_cached_card_url(player_id, url, mtime=card_mtime, data_dir=data_dir)
        return url

    return None


def upload_card_if_online(
    card_path: Path,
    player_id: str,
    data_dir: Path | str = "data",
    timeout: float = 12.0,
) -> str | None:
    """Backward-compatible alias for get_or_upload_public_card_url."""
    return get_or_upload_public_card_url(
        card_path=card_path,
        player_id=player_id,
        data_dir=data_dir,
        timeout=timeout,
    )
