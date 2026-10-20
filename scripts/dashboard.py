"""Mini dashboard fleet Waddle : courbes de PnL des 5 bots, lecture seule des DB.

Usage : DASH_BIND=<ip tailscale> uv run scripts/dashboard.py
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
PAGE = (Path(__file__).resolve().parent / "dashboard.html").read_bytes()

# Même source de vérité que telegram_push.py : 4 combos fixes + le bot retuné.
BOTS = [
    {"id": "C1", "db": "data/fleet_combo1.db", "params": "config/fleet/combo1_params.json", "strategy": "sans retune"},
    {"id": "C2", "db": "data/fleet_combo2.db", "params": "config/fleet/combo2_params.json", "strategy": "sans retune"},
    {"id": "C3", "db": "data/fleet_combo3.db", "params": "config/fleet/combo3_params.json", "strategy": "sans retune"},
    {"id": "C4", "db": "data/fleet_combo4.db", "params": "config/fleet/combo4_params.json", "strategy": "sans retune"},
    {"id": "R", "db": "data/waddle_bot.db", "params": "config/active_params.json", "strategy": "retune"},
]


def _label(bot: dict) -> str:
    # Les params de R changent à chaque retune : on les relit à chaque appel.
    try:
        p = json.loads((ROOT / bot["params"]).read_text())
        return f"{bot['id']} {p['entry_z']}/{p['exit_z']}"
    except (OSError, KeyError, ValueError):
        return bot["id"]


def meta() -> list[dict]:
    return [{"id": b["id"], "label": _label(b), "strategy": b["strategy"]} for b in BOTS]


def series(bot_ids: list[str], start_ms: int, end_ms: int, bucket_ms: int) -> dict:
    by_bot: dict[str, dict[int, float]] = {}
    errors: dict[str, str] = {}
    for bot in BOTS:
        if bot["id"] not in bot_ids:
            continue
        try:
            with closing(sqlite3.connect(f"file:{ROOT / bot['db']}?mode=ro", uri=True, timeout=5)) as con:
                # Baseline = 1er snapshot brut de la fenêtre, pas la fin du 1er bucket (sinon du PnL disparaît).
                first = con.execute(
                    "SELECT nav FROM bot_equity_snapshots WHERE ts >= ? AND ts < ? ORDER BY ts LIMIT 1",
                    (start_ms, end_ms),
                ).fetchone()
                rows = con.execute(
                    # La colonne nue `nav` prend la valeur de la ligne du MAX(ts) → dernier point du bucket.
                    "SELECT MAX(ts), nav FROM bot_equity_snapshots "
                    "WHERE ts >= ? AND ts < ? GROUP BY ts / ? ORDER BY 1",
                    (start_ms, end_ms, bucket_ms),
                ).fetchall()
        except sqlite3.Error as e:  # DB corrompue ou absente : on isole le bot, pas de 500 global
            errors[bot["id"]] = str(e)
            continue
        if first:
            by_bot[bot["id"]] = {ts // bucket_ms * bucket_ms: nav - first[0] for ts, nav in rows}
    xs = sorted({t for d in by_bot.values() for t in d})
    return {
        "x": [t // 1000 for t in xs],
        "series": {b: [d.get(t) for t in xs] for b, d in by_bot.items()},
        "errors": errors,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/":
            self._send(PAGE, "text/html; charset=utf-8")
        elif url.path == "/api/meta":
            self._send(json.dumps(meta()).encode(), "application/json")
        elif url.path == "/api/series":
            q = parse_qs(url.query)
            try:
                data = series(
                    q.get("bots", [""])[0].split(","),
                    int(q["start"][0]),
                    int(q["end"][0]),
                    max(int(q["bucket"][0]), 60_000),
                )
            except (KeyError, ValueError, OverflowError):
                self._send(b'{"error":"bad params"}', "application/json", 400)
                return
            self._send(json.dumps(data).encode(), "application/json")
        else:
            self._send(b"not found", "text/plain", 404)

    def log_message(self, *_) -> None:
        pass


if __name__ == "__main__":
    # Défaut loopback : on n'expose rien tant que DASH_BIND (IP Tailscale du N100) n'est pas posé.
    host = os.environ.get("DASH_BIND", "127.0.0.1")
    port = int(os.environ.get("DASH_PORT", "8765"))
    print(f"[dashboard] http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
