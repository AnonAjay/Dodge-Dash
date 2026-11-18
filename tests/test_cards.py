"""Tests for Titans' Arcade souvenir photo cards and QR system.
Verifies direct PNG image downloads, public cloud vs offline LAN fallback QR URLs,
automatic public Litterbox uploader with Tmpfiles fallback, data/card_urls.json mtime caching,
share-info network mode reporting, operator PIN lockdown, and 100% offline compliance.
"""
from __future__ import annotations

import base64
import importlib
import io
import json
import os
from pathlib import Path
import time
import urllib.error
import cv2
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from cards.generator import generate_card
from cards.qr import generate_qr_image, generate_qr_png_bytes, resolve_card_qr_url
from cards.uploader import (
    get_cached_card_entry,
    get_cached_card_url,
    get_or_upload_public_card_url,
    invalidate_card_url,
    set_cached_card_url,
    upload_card_if_online,
)
import server.app as app_module


class FakeResponse:
    def __init__(self, data: str | bytes):
        if isinstance(data, str):
            self._data = data.encode("utf-8")
        else:
            self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DISABLE_PUBLIC_UPLOAD", "1")
    monkeypatch.delenv("PUBLIC_CARD_BASE_URL", raising=False)
    importlib.reload(app_module)
    return TestClient(app_module.app)


def test_qr_generation():
    test_url = "http://192.168.1.10:8000/api/cards/P001/download"
    img = generate_qr_image(test_url, box_size=6, border=4)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size[0] > 100

    det = cv2.QRCodeDetectorAruco()
    decoded, _, _ = det.detectAndDecode(np.array(img))
    assert decoded == test_url

    png_bytes = generate_qr_png_bytes(test_url, box_size=6, border=4)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    # Invalid empty data
    with pytest.raises(ValueError):
        generate_qr_image("")
    with pytest.raises(ValueError):
        generate_qr_png_bytes("   ")


def test_card_generation_with_photo_and_scores(tmp_path, monkeypatch):
    monkeypatch.setenv("DISABLE_PUBLIC_UPLOAD", "1")
    monkeypatch.delenv("PUBLIC_CARD_BASE_URL", raising=False)
    data_dir = tmp_path / "data"
    cards_dir = data_dir / "cards"
    photos_dir = data_dir / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy photo
    photo_img = Image.new("RGB", (640, 480), (120, 40, 180))
    photo_file = photos_dir / "P001.jpg"
    photo_img.save(photo_file, format="JPEG")

    # Create players.json
    players = [
        {
            "player_id": "P001",
            "name": "Drishti",
            "roll_no": "24303107",
            "height_cm": 153.0,
            "photo_url": "/api/photos/P001.jpg",
        }
    ]
    (data_dir / "players.json").write_text(json.dumps(players), encoding="utf-8")

    # Create scores.jsonl with scores for dodge, wall, dance
    scores = [
        {"event_id": "e1", "player_id": "P001", "game": "dodge", "score": 1250, "accuracy": 0.98, "best_streak": 22},
        {"event_id": "e2", "player_id": "P001", "game": "wall", "score": 900, "accuracy": 0.91, "best_streak": 14},
        {"event_id": "e3", "player_id": "P001", "game": "dance", "score": 1420, "accuracy": 0.99, "best_streak": 28},
    ]
    with (data_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(s) + "\n")

    card_path = generate_card(
        "P001",
        data_dir=data_dir,
        output_dir=cards_dir,
        base_url="http://testserver",
    )
    assert card_path.exists()
    assert card_path.name == "P001.png"

    card_img = Image.open(card_path)
    assert card_img.size == (1080, 1350)
    assert card_img.mode == "RGB"

    import cards.generator as gen_mod
    gen_src = Path(gen_mod.__file__).read_text(encoding="utf-8")
    assert "TITANS' ARCADE" in gen_src
    assert "FLOOR IS LAVA" in gen_src
    assert "#TA-2026-" in gen_src

    # Decode embedded direct-download QR from card (bottom-right region: x: 840-1030, y: 1120-1300)
    qr_crop = card_img.crop((840, 1120, 1030, 1300))
    det = cv2.QRCodeDetectorAruco()
    decoded_url, _, _ = det.detectAndDecode(np.array(qr_crop))
    assert decoded_url == "http://testserver/api/cards/P001/download"

    # Test with PUBLIC_CARD_BASE_URL set
    monkeypatch.setenv("PUBLIC_CARD_BASE_URL", "https://cdn.fest.org/cards")
    card_path_pub = generate_card(
        "P001",
        data_dir=data_dir,
        output_dir=cards_dir,
    )
    card_img_pub = Image.open(card_path_pub)
    qr_crop_pub = card_img_pub.crop((840, 1120, 1030, 1300))
    decoded_pub, _, _ = det.detectAndDecode(np.array(qr_crop_pub))
    assert decoded_pub == "https://cdn.fest.org/cards/P001.png"


def test_card_generation_without_photo_and_unplayed(tmp_path, monkeypatch):
    monkeypatch.setenv("DISABLE_PUBLIC_UPLOAD", "1")
    data_dir = tmp_path / "data"
    cards_dir = data_dir / "cards"
    data_dir.mkdir(parents=True, exist_ok=True)

    players = [
        {
            "player_id": "P002",
            "name": "Alex",
            "roll_no": None,
            "height_cm": None,
            "photo_url": None,
        }
    ]
    (data_dir / "players.json").write_text(json.dumps(players), encoding="utf-8")

    card_path = generate_card("P002", data_dir=data_dir, output_dir=cards_dir)
    assert card_path.exists()

    card_img = Image.open(card_path)
    assert card_img.size == (1080, 1350)

    # Test unknown player raises ValueError
    with pytest.raises(ValueError, match="not found"):
        generate_card("P999", data_dir=data_dir, output_dir=cards_dir)


def test_uploader_litterbox_success_and_cache(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    cards_dir = data_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    dummy_card = cards_dir / "P001.png"
    dummy_card.write_bytes(b"\x89PNG\r\n\x1a\nfakecard_v1")

    call_count = {"calls": 0}

    def fake_urlopen(req, timeout=12.0):
        call_count["calls"] += 1
        assert "litterbox.catbox.moe" in req.full_url
        assert req.get_method() == "POST"
        user_agent = req.headers.get("User-agent") or req.headers.get("User-Agent")
        assert user_agent == "TitansArcade/1.0"
        return FakeResponse("https://litter.catbox.moe/abc12345.png")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.delenv("DISABLE_PUBLIC_UPLOAD", raising=False)

    uploaded_url = get_or_upload_public_card_url(dummy_card, "P001", data_dir=data_dir)
    assert uploaded_url == "https://litter.catbox.moe/abc12345.png"
    assert call_count["calls"] == 1

    # Check cache structure in data/card_urls.json
    entry = get_cached_card_entry("P001", data_dir=data_dir)
    assert entry is not None
    assert entry["url"] == "https://litter.catbox.moe/abc12345.png"
    assert abs(entry["mtime"] - dummy_card.stat().st_mtime) < 0.001

    # Calling again without modifying file should use cache and NOT invoke urlopen
    cached_url = get_or_upload_public_card_url(dummy_card, "P001", data_dir=data_dir)
    assert cached_url == "https://litter.catbox.moe/abc12345.png"
    assert call_count["calls"] == 1

    # Modifying the card PNG (changing mtime) triggers re-upload
    dummy_card.write_bytes(b"\x89PNG\r\n\x1a\nfakecard_v2_updated")
    new_mtime = dummy_card.stat().st_mtime + 5.0
    os.utime(dummy_card, (new_mtime, new_mtime))

    reuploaded_url = get_or_upload_public_card_url(dummy_card, "P001", data_dir=data_dir)
    assert reuploaded_url == "https://litter.catbox.moe/abc12345.png"
    assert call_count["calls"] == 2

    # Invalidate
    invalidate_card_url("P001", data_dir=data_dir)
    assert get_cached_card_url("P001", data_dir=data_dir) is None


def test_uploader_catbox_fallback(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    cards_dir = data_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    dummy_card = cards_dir / "P001.png"
    dummy_card.write_bytes(b"\x89PNG\r\n\x1a\nfakecard")

    called_urls = []

    def fake_urlopen(req, timeout=12.0):
        called_urls.append(req.full_url)
        if "litterbox.catbox.moe" in req.full_url:
            raise urllib.error.URLError("Litterbox service offline")
        elif "catbox.moe" in req.full_url:
            assert req.get_method() == "POST"
            return FakeResponse("https://files.catbox.moe/xyz987.png")
        raise ValueError(f"Unexpected URL: {req.full_url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.delenv("DISABLE_PUBLIC_UPLOAD", raising=False)

    uploaded_url = get_or_upload_public_card_url(dummy_card, "P001", data_dir=data_dir)
    assert any("litterbox" in u for u in called_urls)
    assert any("catbox.moe/user/api.php" in u for u in called_urls)
    assert uploaded_url == "https://files.catbox.moe/xyz987.png"

    # Cached in card_urls.json
    entry = get_cached_card_entry("P001", data_dir=data_dir)
    assert entry is not None
    assert entry["url"] == "https://files.catbox.moe/xyz987.png"


def test_uploader_tmpfiles_fallback(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    cards_dir = data_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    dummy_card = cards_dir / "P001.png"
    dummy_card.write_bytes(b"\x89PNG\r\n\x1a\nfakecard")

    called_urls = []

    def fake_urlopen(req, timeout=12.0):
        called_urls.append(req.full_url)
        if "litterbox.catbox.moe" in req.full_url:
            raise urllib.error.URLError("Litterbox service unavailable")
        elif "catbox.moe" in req.full_url:
            raise urllib.error.URLError("Catbox service unavailable")
        elif "tmpfiles.org" in req.full_url:
            assert req.get_method() == "POST"
            resp_json = {
                "status": "success",
                "data": {
                    "url": "https://tmpfiles.org/887766/TitansArcade_P001.png"
                }
            }
            return FakeResponse(json.dumps(resp_json))
        raise ValueError(f"Unexpected URL: {req.full_url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.delenv("DISABLE_PUBLIC_UPLOAD", raising=False)

    uploaded_url = get_or_upload_public_card_url(dummy_card, "P001", data_dir=data_dir)
    assert any("litterbox" in u for u in called_urls)
    assert any("catbox" in u for u in called_urls)
    assert any("tmpfiles" in u for u in called_urls)
    # Verifies conversion to direct download URL (/dl/)
    assert uploaded_url == "https://tmpfiles.org/dl/887766/TitansArcade_P001.png"

    # Verifies cached in data/card_urls.json
    entry = get_cached_card_entry("P001", data_dir=data_dir)
    assert entry is not None
    assert entry["url"] == "https://tmpfiles.org/dl/887766/TitansArcade_P001.png"


def test_uploader_upgrades_stale_tmpfiles_cache(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    cards_dir = data_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    dummy_card = cards_dir / "P001.png"
    dummy_card.write_bytes(b"\x89PNG\r\n\x1a\nfakecard")

    # Pre-populate cache with a stale tmpfiles.org URL
    set_cached_card_url(
        "P001",
        "https://tmpfiles.org/dl/old123/p001.png",
        mtime=dummy_card.stat().st_mtime,
        data_dir=data_dir,
    )

    def fake_urlopen(req, timeout=12.0):
        if "litterbox.catbox.moe" in req.full_url:
            return FakeResponse("https://litter.catbox.moe/upgraded.png")
        raise ValueError(f"Unexpected URL: {req.full_url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.delenv("DISABLE_PUBLIC_UPLOAD", raising=False)

    # Should upgrade from tmpfiles.org to Litterbox direct PNG URL
    upgraded_url = get_or_upload_public_card_url(dummy_card, "P001", data_dir=data_dir)
    assert upgraded_url == "https://litter.catbox.moe/upgraded.png"

    # Cache must now reflect the upgraded URL
    entry = get_cached_card_entry("P001", data_dir=data_dir)
    assert entry is not None
    assert entry["url"] == "https://litter.catbox.moe/upgraded.png"


def test_uploader_offline_and_disabled(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    cards_dir = data_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    dummy_card = cards_dir / "P001.png"
    dummy_card.write_bytes(b"\x89PNG\r\n\x1a\nfakecard")

    # 1. Disabled via env var
    monkeypatch.setenv("DISABLE_PUBLIC_UPLOAD", "1")
    assert get_or_upload_public_card_url(dummy_card, "P001", data_dir=data_dir) is None
    assert upload_card_if_online(dummy_card, "P001", data_dir=data_dir) is None

    # 2. Network error / all hosts fail
    monkeypatch.delenv("DISABLE_PUBLIC_UPLOAD", raising=False)

    def error_urlopen(req, timeout=12.0):
        raise urllib.error.URLError("Network unreachable")

    monkeypatch.setattr("urllib.request.urlopen", error_urlopen)
    assert get_or_upload_public_card_url(dummy_card, "P001", data_dir=data_dir) is None
    assert upload_card_if_online(dummy_card, "P001", data_dir=data_dir) is None


def test_api_cards_and_download_endpoints(client):
    # Register player P001
    reg_res = client.post("/api/players", json={"name": "Ravi", "roll_no": "24303192"})
    assert reg_res.status_code == 200
    assert reg_res.json()["player_id"] == "P001"

    # Request souvenir card (inline disposition)
    card_res = client.get("/api/cards/P001")
    assert card_res.status_code == 200
    assert card_res.headers["content-type"] == "image/png"
    assert 'inline; filename="TitansArcade_P001.png"' in card_res.headers.get("content-disposition", "")
    assert card_res.content.startswith(b"\x89PNG\r\n\x1a\n")

    # Request direct download endpoint (attachment disposition)
    dl_res = client.get("/api/cards/P001/download")
    assert dl_res.status_code == 200
    assert dl_res.headers["content-type"] == "image/png"
    assert 'attachment; filename="TitansArcade_P001.png"' in dl_res.headers.get("content-disposition", "")
    assert dl_res.content.startswith(b"\x89PNG\r\n\x1a\n")

    # Verify PNG dimensions
    img = Image.open(io.BytesIO(card_res.content))
    assert img.size == (1080, 1350)

    # Unknown player returns 404
    assert client.get("/api/cards/P999").status_code == 404
    assert client.get("/api/cards/P999/download").status_code == 404


def test_api_qr_endpoint_offline_and_public(client, monkeypatch):
    reg_res = client.post("/api/players", json={"name": "Kiran"})
    assert reg_res.status_code == 200

    # 1. Offline fallback: QR points to direct download endpoint
    qr_res = client.get("/api/qr/P001")
    assert qr_res.status_code == 200
    assert qr_res.headers["content-type"] == "image/png"

    det = cv2.QRCodeDetectorAruco()
    img = Image.open(io.BytesIO(qr_res.content))
    decoded, _, _ = det.detectAndDecode(np.array(img))
    assert decoded == "http://testserver/api/cards/P001/download"

    # 2. Public Direct-Image URL Support (hotspot / mobile data)
    monkeypatch.setenv("PUBLIC_CARD_BASE_URL", "https://cdn.fest.org/cards")
    qr_pub_res = client.get("/api/qr/P001")
    assert qr_pub_res.status_code == 200

    img_pub = Image.open(io.BytesIO(qr_pub_res.content))
    decoded_pub, _, _ = det.detectAndDecode(np.array(img_pub))
    assert decoded_pub == "https://cdn.fest.org/cards/P001.png"

    # Unknown player returns 404
    assert client.get("/api/qr/P999").status_code == 404


def test_api_cards_share_info_and_qr_modes(client, monkeypatch):
    reg_res = client.post("/api/players", json={"name": "Pooja"})
    assert reg_res.status_code == 200
    assert reg_res.json()["player_id"] == "P001"

    # 1. Offline Mode (DISABLE_PUBLIC_UPLOAD=1)
    info_res = client.get("/api/cards/P001/share-info")
    assert info_res.status_code == 200
    info_data = info_res.json()
    assert info_data["mode"] == "LOCAL LAN OFFLINE MODE"
    assert info_data["is_public"] is False
    assert info_data["url"] == "http://testserver/api/cards/P001/download"
    assert info_data["player_id"] == "P001"

    qr_offline = client.get("/api/qr/P001")
    assert qr_offline.status_code == 200
    det = cv2.QRCodeDetectorAruco()
    img_offline = Image.open(io.BytesIO(qr_offline.content))
    decoded_off, _, _ = det.detectAndDecode(np.array(img_offline))
    assert decoded_off == "http://testserver/api/cards/P001/download"

    # 2. 5G / Public Cloud Mode (mocked online upload)
    def fake_urlopen(req, timeout=6.0):
        return FakeResponse("https://litter.catbox.moe/live_pooja_card.png\n")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.delenv("DISABLE_PUBLIC_UPLOAD", raising=False)
    invalidate_card_url("P001", data_dir=Path("data"))

    info_pub_res = client.get("/api/cards/P001/share-info")
    assert info_pub_res.status_code == 200
    info_pub_data = info_pub_res.json()
    assert info_pub_data["mode"] == "PUBLIC CLOUD MODE"
    assert info_pub_data["is_public"] is True
    assert info_pub_data["url"] == "https://litter.catbox.moe/live_pooja_card.png"

    # QR in public mode must encode that direct https:// URL
    qr_pub = client.get("/api/qr/P001")
    assert qr_pub.status_code == 200
    img_pub = Image.open(io.BytesIO(qr_pub.content))
    decoded_pub, _, _ = det.detectAndDecode(np.array(img_pub))
    assert decoded_pub == "https://litter.catbox.moe/live_pooja_card.png"

    # Now simulate network failure: invalidate cache and make urlopen fail
    invalidate_card_url("P001", data_dir=Path("data"))

    def error_urlopen(req, timeout=6.0):
        raise urllib.error.URLError("Hotspot offline")

    monkeypatch.setattr("urllib.request.urlopen", error_urlopen)

    qr_fallback_res = client.get("/api/qr/P001")
    assert qr_fallback_res.status_code == 200
    img_fallback = Image.open(io.BytesIO(qr_fallback_res.content))
    decoded_fallback, _, _ = det.detectAndDecode(np.array(img_fallback))
    assert decoded_fallback == "http://testserver/api/cards/P001/download"

    # 3. 404 for unknown player
    assert client.get("/api/cards/P999/share-info").status_code == 404


def test_operator_lockdown_pin_required(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DISABLE_PUBLIC_UPLOAD", "1")
    importlib.reload(app_module)

    # Simulated remote hotspot client (e.g. mobile phone at 192.168.43.5)
    remote_client = TestClient(app_module.app, client=("192.168.43.5", 50000))

    # Without PIN: access forbidden
    assert remote_client.get("/register").status_code == 403
    assert remote_client.post("/api/players", json={"name": "Attacker"}).status_code == 403

    # With invalid PIN: access forbidden
    assert remote_client.get("/register?pin=wrong").status_code == 403
    assert remote_client.post("/api/players", headers={"X-Operator-Pin": "wrong"}, json={"name": "Attacker"}).status_code == 403

    # With valid PIN via query parameter: access granted
    res_get_query = remote_client.get("/register?pin=2026")
    assert res_get_query.status_code == 200

    res_post_query = remote_client.post("/api/players?pin=2026", json={"name": "Alice"})
    assert res_post_query.status_code == 200
    assert res_post_query.json()["player_id"] == "P001"

    # With valid PIN via header: access granted
    res_get_header = remote_client.get("/register", headers={"X-Operator-Pin": "2026"})
    assert res_get_header.status_code == 200

    res_post_header = remote_client.post("/api/players", headers={"X-Operator-Pin": "2026"}, json={"name": "Bob"})
    assert res_post_header.status_code == 200
    assert res_post_header.json()["player_id"] == "P002"

    # Localhost client (testclient): access granted without PIN
    local_client = TestClient(app_module.app)
    assert local_client.get("/register").status_code == 200
    res_local = local_client.post("/api/players", json={"name": "Charlie"})
    assert res_local.status_code == 200
    assert res_local.json()["player_id"] == "P003"


def test_souvenir_card_html_page_offline(client):
    reg_res = client.post("/api/players", json={"name": "Drishti", "roll_no": "24303107"})
    assert reg_res.status_code == 200

    page_res = client.get("/card/P001")
    assert page_res.status_code == 200
    html = page_res.text

    assert "TITANS' ARCADE" in html or "Titans' Arcade" in html
    assert "Drishti" in html
    assert "P001" in html
    assert "24303107" in html
    assert "/api/cards/P001" in html
    assert 'download="TitansArcade_P001.png"' in html
    assert 'href="/"' in html or "href='/'" in html

    # Strict 100% offline verification: no external CDN / web links
    assert "https://" not in html
    assert "http://" not in html

    # Unknown player returns 404
    assert client.get("/card/P999").status_code == 404


def test_operator_desk_card_qr_elements(client):
    res = client.get("/register")
    assert res.status_code == 200
    html = res.text

    # Immediate QR container on registration
    assert "reg-qr-section" in html
    assert "reg-qr-img" in html
    assert "reg-card-link" in html

    # Table Action Header & Button
    assert "PHOTOCARD" in html
    assert "btn-action-qr" in html
    assert "Open Photocard QR &amp; Download" in html or "Open Photocard QR" in html
    assert "openQrModal" in html

    # Modal dialog
    assert "qr-modal" in html
    assert "modal-qr-img" in html
    assert "modal-network-mode" in html
    assert "modal-card-url" not in html
    assert "modal-view-card" in html
    assert "modal-download-card" in html

    # Strict offline verification
    assert "https://" not in html
    assert "http://" not in html
