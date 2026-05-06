"""Push Waddle fleet status to a dedicated Telegram bot.

Reads each fleet combo DB, builds a compact status message, and sends it
via the Telegram Bot API. Designed to be called on a schedule (e.g. every 6h).

Configuration (via environment variables or .env):
    WADDLE_TELEGRAM_BOT_TOKEN   Bot token from @BotFather
    WADDLE_TELEGRAM_CHAT_ID     Chat ID of the target conversation
    PUSH_INTERVAL_H             Push interval in hours (default: 6, unused here — used by scheduler)

Usage:
    uv run scripts/telegram_push.py                   # push once now
    uv run scripts/telegram_push.py --dry-run         # print message, don't send
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

import pandas as pd

from waddle.bot.state import StateStore

FLEET_COMBOS = {
    1: {"label": "C1 entry=2.0/exit=0.5", "db": "data/fleet_combo1.db"},
    2: {"label": "C2 entry=2.2/exit=1.0", "db": "data/fleet_combo2.db"},
    3: {"label": "C3 entry=2.5/exit=0.5", "db": "data/fleet_combo3.db"},
    4: {"label": "C4 entry=2.7/exit=1.0", "db": "data/fleet_combo4.db"},
}

ALERT_DRAWDOWN_PCT = 5.0
ALERT_INACTIVE_H = 2.0


def _fmt_pnl(v: float) -> str:
    return f"+${v:.2f}" if v >= 0 else f"-${abs(v):.2f}"


def _combo_block(combo_id: int, combo: dict) -> tuple[str, list[str]]:
    """Return (status_line, alerts) for one combo."""
    path = Path(combo["db"])
    label = combo["label"]

    if not path.exists():
        return f"❓ {label} — DB not found", []

    store = StateStore(path)
    alerts: list[str] = []

    # Position
    positions = store.list_open_positions()
    if positions:
        pos = positions[0]
        now = pd.Timestamp.now(tz="UTC")
        entry_ts = pos.entry_ts
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        bars_held = int((now - entry_ts).total_seconds() // 3600)
        pos_str = f"{pos.direction} {bars_held}h"
    else:
        pos_str = "flat"

    # Equity
    eq = store.latest_equity()
    all_trades = store.list_trades(limit=10_000)
    realized = sum(t.pnl_net for t in all_trades)
    unrealized = eq.unrealized_pnl if eq else 0.0
    total_pnl = realized + unrealized
    n_trades = len(all_trades)

    # Drawdown alert (simple: total_pnl < -threshold * nav_initial)
    nav = eq.nav if eq else 100.0
    nav_initial = 100.0  # paper trading starts at $100 notional
    dd_pct = (total_pnl / nav_initial) * 100
    if dd_pct < -ALERT_DRAWDOWN_PCT:
        alerts.append(f"⚠️ {label}: drawdown {dd_pct:.1f}%")

    # Inactivity alert
    events = store.recent_events(limit=1)
    if events:
        last_event_ts = events[0]["ts"]
        if last_event_ts.tzinfo is None:
            last_event_ts = last_event_ts.tz_localize("UTC")
        inactive_h = (pd.Timestamp.now(tz="UTC") - last_event_ts).total_seconds() / 3600
        if inactive_h > ALERT_INACTIVE_H:
            alerts.append(f"🔇 {label}: inactive {inactive_h:.1f}h")

    # Failed tick alert
    recent_events = store.recent_events(limit=20)
    tick_fails = sum(1 for e in recent_events if e["event_type"] == "TICK_FAILED")
    if tick_fails > 0:
        alerts.append(f"💥 {label}: {tick_fails} TICK_FAILED (last 20 events)")

    pnl_str = _fmt_pnl(total_pnl)
    line = f"{'✅' if total_pnl >= 0 else '🔴'} {label}\n   {pos_str} | PnL {pnl_str} | {n_trades} trades"
    return line, alerts


def build_message() -> str:
    now_str = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"🦆 *Waddle Fleet Status* — {now_str}\n"]

    all_alerts: list[str] = []

    for combo_id, combo in FLEET_COMBOS.items():
        block, alerts = _combo_block(combo_id, combo)
        lines.append(block)
        all_alerts.extend(alerts)

    if all_alerts:
        lines.append("\n*Alertes :*")
        lines.extend(all_alerts)

    return "\n".join(lines)


def send_telegram(message: str, bot_token: str, chat_id: str) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read())
    if not result.get("ok"):
        raise RuntimeError(f"Telegram API error: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push Waddle fleet status to Telegram")
    parser.add_argument("--dry-run", action="store_true", help="Print message without sending")
    args = parser.parse_args()

    # Load .env if present (simple key=value parser, no dependency)
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    message = build_message()

    if args.dry_run:
        print(message)
        return

    bot_token = os.environ.get("WADDLE_TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("WADDLE_TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id:
        raise SystemExit(
            "ERROR: WADDLE_TELEGRAM_BOT_TOKEN and WADDLE_TELEGRAM_CHAT_ID must be set in .env"
        )

    send_telegram(message, bot_token, chat_id)
    print(f"✅ Pushed to Telegram chat {chat_id}")


if __name__ == "__main__":
    main()
