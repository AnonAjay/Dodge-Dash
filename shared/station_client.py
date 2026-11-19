"""Non-blocking background client for Station A and Station B.
Maintains continuous, decoupled communication with the Laptop 3 server.

Guarantees:
- Runs in a background daemon thread; zero network calls inside the 60 FPS Pygame loop.
- Thread-safe atomic player queue polling and status reporting.
- Outbox Auto-Drainer: flushes local outbox.jsonl scores to Laptop 3 whenever server is online.
- Seamless Offline Fallback: if server drops, sets is_online=False without throwing or stalling.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from shared.config import load_config

DEFAULT_TIMEOUT_S = 1.5


class StationNetworkClient:
    """Daemon worker connecting a game station to Laptop 3."""

    def __init__(
        self,
        station: str,
        server_url: str | None = None,
        outbox_path: str | Path = "outbox.jsonl",
        poll_interval_s: float = 1.0,
    ) -> None:
        self.station = station.strip().upper()
        self.outbox_path = Path(outbox_path)
        self.poll_interval_s = float(poll_interval_s)

        if server_url is not None and server_url.strip():
            self.server_url: str | None = server_url.strip().rstrip("/")
        else:
            # Only connect if config.local.json explicitly configures server_url (never default to example)
            local_cfg = Path("config.local.json")
            if local_cfg.exists():
                try:
                    cfg = json.loads(local_cfg.read_text(encoding="utf-8"))
                    cfg_url = cfg.get("server_url")
                    self.server_url = cfg_url.strip().rstrip("/") if cfg_url else None
                except Exception:
                    self.server_url = None
            else:
                self.server_url = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._last_contact_ts: float = 0.0
        self._is_online: bool = False
        self._pending_player: dict[str, Any] | None = None

        self._current_status: str = "idle"
        self._current_player_id: str | None = None
        self._current_score: int | None = None

    @property
    def is_online(self) -> bool:
        """Return True if successfully contacted server within last 5 seconds."""
        if not self.server_url:
            return False
        with self._lock:
            return (time.time() - self._last_contact_ts) <= 5.0

    def start(self) -> None:
        """Start background polling thread if server_url is configured."""
        if not self.server_url:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name=f"StationClient-{self.station}", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Signal worker to stop and wait briefly."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_s)

    def update_status(self, status: str, player_id: str | None = None, score: int | None = None) -> None:
        """Update live status reported in heartbeats (called from game loop)."""
        with self._lock:
            self._current_status = status.lower()
            self._current_player_id = player_id
            self._current_score = score

    def pop_next_player(self) -> dict[str, Any] | None:
        """Atomically retrieve and consume the next player dispatched by the server."""
        with self._lock:
            p = self._pending_player
            self._pending_player = None
            return p

    def _http_request(self, method: str, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Make a short-timeout HTTP request without throwing exceptions."""
        if not self.server_url:
            return None
        url = f"{self.server_url}{endpoint}"
        req_data = None
        headers = {"Accept": "application/json"}
        if data is not None:
            req_data = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_S) as resp:
                if 200 <= resp.status < 300:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw) if raw.strip() else {}
        except Exception:
            return None
        return None

    def _sync_outbox_locked(self) -> None:
        """Flush unsent ScoreEvents from outbox.jsonl to Laptop 3."""
        if not self.outbox_path.exists():
            return

        try:
            lines = self.outbox_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return

        if not lines:
            return

        remaining_lines: list[str] = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
            except Exception:
                continue

            resp = self._http_request("POST", "/api/score", data=record)
            if resp and resp.get("accepted"):
                # Score successfully recorded on server (or marked duplicate)
                continue
            else:
                # Keep line to retry later
                remaining_lines.append(line_str)

        try:
            if remaining_lines:
                self.outbox_path.write_text("\n".join(remaining_lines) + "\n", encoding="utf-8")
            else:
                # Fully drained
                self.outbox_path.write_text("", encoding="utf-8")
        except Exception:
            pass

    def _run_loop(self) -> None:
        """Background heartbeat, queue polling, and outbox drain loop."""
        while not self._stop_event.is_set():
            if self.server_url:
                with self._lock:
                    st_status = self._current_status
                    st_pid = self._current_player_id
                    st_score = self._current_score
                    needs_player = (self._pending_player is None) and (st_status in ("idle", "result"))

                # 1. Send Heartbeat
                hb_payload = {
                    "status": st_status,
                    "player_id": st_pid,
                    "score": st_score,
                }
                hb_resp = self._http_request("POST", f"/api/queue/{self.station}/heartbeat", data=hb_payload)
                now = time.time()
                if hb_resp is not None:
                    with self._lock:
                        self._last_contact_ts = now
                        self._is_online = True

                    # 2. Drain Outbox if server responded
                    self._sync_outbox_locked()

                    # 3. Pull Next Player if station is waiting
                    if needs_player:
                        next_resp = self._http_request("GET", f"/api/queue/{self.station}/next")
                        if next_resp and next_resp.get("player"):
                            with self._lock:
                                self._pending_player = next_resp["player"]
                else:
                    with self._lock:
                        # Contact failed; if stale > 5s, mark offline
                        if (now - self._last_contact_ts) > 5.0:
                            self._is_online = False

            # Sleep poll interval
            self._stop_event.wait(self.poll_interval_s)
