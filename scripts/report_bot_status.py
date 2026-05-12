"""Report the current status of a Waddle bot instance.

Reads a bot state DB and prints a human-readable status report:
current position, recent trades, equity summary, recent events, uptime.

Usage:
    uv run scripts/report_bot_status.py                          # reports all fleet_combo*.db
    uv run scripts/report_bot_status.py --db data/fleet_combo1.db
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from waddle.bot.state import StateStore


def _fmt_ts(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def _fmt_usd(v: float) -> str:
    return f"${v:+.2f}" if v != 0 else "$0.00"


def report(db_path: str) -> None:
    path = Path(db_path)
    if not path.exists():
        print(f"ERROR: database not found: {path}")
        return

    store = StateStore(path)
    print()
    print("=" * 64)
    print(f"  WADDLE BOT STATUS -- {path}")
    print("=" * 64)

    # --- Current position ---
    print()
    print("--- Current Position ---")
    positions = store.list_open_positions()
    if not positions:
        print("  No open position")
    else:
        for pos in positions:
            now = pd.Timestamp.now(tz="UTC")
            entry_ts = pos.entry_ts
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize("UTC")
            bars_held = max(0, int((now - entry_ts).total_seconds() // 3600))
            print(f"  Direction:   {pos.direction}")
            print(f"  Entry time:  {_fmt_ts(pos.entry_ts)}")
            print(f"  Entry z:     {pos.entry_z:.3f}")
            print(f"  Bars held:   {bars_held}h")
            print(f"  Entry spread:{pos.entry_spread:.6f}")
            print(f"  Pair:        {pos.symbol_a} / {pos.symbol_b}")
            print()

    # --- Last 5 trades ---
    print("--- Last 5 Trades ---")
    trades = store.list_trades(limit=5)
    if not trades:
        print("  No closed trades yet")
    else:
        print(f"  {'Exit time':<22} {'Dir':<14} {'Bars':>5} {'PnL net':>10} {'Reason':<16}")
        print(f"  {'-'*22} {'-'*14} {'-'*5} {'-'*10} {'-'*16}")
        for t in trades:
            print(
                f"  {_fmt_ts(t.exit_ts):<22} "
                f"{t.exit_reason:<14} "
                f"{t.bars_held:>5} "
                f"{_fmt_usd(t.pnl_net):>10} "
                f"{t.exit_reason:<16}"
            )
    print()

    # --- Equity summary ---
    print("--- Equity Summary ---")
    eq = store.latest_equity()
    if eq is None:
        print("  No equity snapshots yet")
    else:
        all_trades = store.list_trades(limit=10_000)
        total_pnl = sum(t.pnl_net for t in all_trades)
        n_trades = len(all_trades)
        print(f"  Current NAV:       ${eq.nav:.2f}")
        print(f"  Unrealized PnL:    {_fmt_usd(eq.unrealized_pnl)}")
        print(f"  Total realized PnL:{_fmt_usd(total_pnl)}")
        print(f"  Closed trades:     {n_trades}")
        print(f"  Last snapshot:     {_fmt_ts(eq.ts)}")
    print()

    # --- Last 5 events ---
    print("--- Last 5 Events ---")
    events = store.recent_events(limit=5)
    if not events:
        print("  No events")
    else:
        for ev in events:
            ts_str = _fmt_ts(ev["ts"])
            payload_str = json.dumps(ev["payload"])
            if len(payload_str) > 60:
                payload_str = payload_str[:57] + "..."
            print(f"  [{ts_str}] {ev['level']:<8} {ev['event_type']:<20} {payload_str}")
    print()

    # --- Uptime ---
    print("--- Uptime ---")
    startup_events = store.recent_events(limit=1000)
    startups = [e for e in startup_events if e["event_type"] == "STARTUP"]
    if startups:
        first_start = startups[-1]["ts"]
        now = pd.Timestamp.now(tz="UTC")
        first_ts = first_start
        if first_ts.tzinfo is None:
            first_ts = first_ts.tz_localize("UTC")
        delta = now - first_ts
        hours = delta.total_seconds() / 3600
        print(f"  First STARTUP: {_fmt_ts(first_start)}")
        print(f"  Uptime:        {hours:.1f} hours ({delta.days}d {delta.seconds // 3600}h)")
    else:
        print("  No STARTUP event found")

    print()
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report Waddle bot status")
    parser.add_argument(
        "--db",
        default=None,
        help="Path to bot state database (default: all data/fleet_combo*.db)",
    )
    args = parser.parse_args()
    if args.db:
        report(args.db)
    else:
        dbs = sorted(glob.glob("data/fleet_combo*.db"))
        if not dbs:
            print("No fleet_combo*.db found — use --db to specify a database.")
            return
        for db in dbs:
            report(db)


if __name__ == "__main__":
    main()
