"""Compare all paper fleet instances side-by-side.

Reads all 4 fleet state DBs and produces a comparison table with key metrics.

Note: 4 bots x 3 symbols x 1 tick/min = 12 API calls/min (Bybit allows ~120/min).

Usage:
    uv run scripts/compare_paper_fleet.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from waddle.bot.state import StateStore


FLEET_COMBOS = {
    1: {"label": "e=2.0/x=0.5", "db": "data/fleet_combo1.db", "params": "config/fleet/combo1_params.json"},
    2: {"label": "e=2.2/x=1.0", "db": "data/fleet_combo2.db", "params": "config/fleet/combo2_params.json"},
    3: {"label": "e=2.5/x=0.5", "db": "data/fleet_combo3.db", "params": "config/fleet/combo3_params.json"},
    4: {"label": "e=2.7/x=1.0", "db": "data/fleet_combo4.db", "params": "config/fleet/combo4_params.json"},
}


def _fmt_ts(ts: pd.Timestamp | None) -> str:
    if ts is None:
        return "-"
    return ts.strftime("%Y-%m-%d %H:%M")


def _load_params_label(params_path: str) -> str:
    try:
        with open(params_path, "r", encoding="utf-8") as f:
            p = json.load(f)
        return f"e={p['entry_z']}/x={p['exit_z']}"
    except Exception:
        return "?"


def _compute_sharpe(pnls: list[float]) -> float | None:
    if len(pnls) < 3:
        return None
    mean_pnl = sum(pnls) / len(pnls)
    variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
    std_pnl = math.sqrt(variance) if variance > 0 else 0.0
    if std_pnl == 0:
        return None
    trades_per_year = len(pnls) * (8760 / max(1, len(pnls)))
    return (mean_pnl / std_pnl) * math.sqrt(trades_per_year)


def main() -> None:
    print()
    print("=" * 90)
    print("  WADDLE PAPER FLEET COMPARISON")
    print("  Rate note: 4 bots x 3 symbols x 1/min = 12 calls/min (Bybit limit ~120/min)")
    print("=" * 90)
    print()

    rows: list[dict] = []

    for combo_id, combo in FLEET_COMBOS.items():
        db_path = Path(combo["db"])
        if not db_path.exists():
            rows.append({
                "combo": combo_id,
                "params": combo["label"],
                "trades": "-",
                "win_rate": "-",
                "total_pnl": "-",
                "sharpe": "-",
                "max_dd": "-",
                "current_pos": "-",
                "funding": "-",
                "first_trade": "-",
                "last_trade": "-",
                "runtime": "-",
            })
            continue

        store = StateStore(db_path)
        trades = store.list_trades(limit=10_000)
        positions = store.list_open_positions()

        n_trades = len(trades)
        wins = sum(1 for t in trades if t.pnl_net > 0)
        win_rate = f"{100 * wins / n_trades:.0f}%" if n_trades > 0 else "-"

        total_pnl = sum(t.pnl_net for t in trades)
        funding_total = sum(t.pnl_funding for t in trades)

        pnls = [t.pnl_net for t in trades]
        sharpe = _compute_sharpe(pnls)
        sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "-"

        # Max drawdown from cumulative PnL series
        if n_trades > 0:
            sorted_trades = sorted(trades, key=lambda t: t.exit_ts)
            cum_pnl = 0.0
            peak = 0.0
            max_dd = 0.0
            for t in sorted_trades:
                cum_pnl += t.pnl_net
                if cum_pnl > peak:
                    peak = cum_pnl
                dd = peak - cum_pnl
                if dd > max_dd:
                    max_dd = dd
            max_dd_str = f"${max_dd:.2f}"
        else:
            max_dd_str = "-"

        if positions:
            pos = positions[0]
            now = pd.Timestamp.now(tz="UTC")
            entry_ts = pos.entry_ts
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize("UTC")
            bars = max(0, int((now - entry_ts).total_seconds() // 3600))
            current_pos = f"{pos.direction[:5]} {bars}h"
        else:
            current_pos = "flat"

        # Timeline
        events = store.recent_events(limit=10_000)
        startups = [e for e in events if e["event_type"] == "STARTUP"]
        if startups:
            first_start = startups[-1]["ts"]
            now = pd.Timestamp.now(tz="UTC")
            first_ts = first_start if first_start.tzinfo is not None else first_start.tz_localize("UTC")
            delta = now - first_ts
            runtime_str = f"{delta.total_seconds() / 3600:.1f}h"
        else:
            first_start = None
            runtime_str = "-"

        first_trade_ts = _fmt_ts(sorted(trades, key=lambda t: t.exit_ts)[0].exit_ts) if trades else "-"
        last_trade_ts = _fmt_ts(sorted(trades, key=lambda t: t.exit_ts)[-1].exit_ts) if trades else "-"

        rows.append({
            "combo": combo_id,
            "params": combo["label"],
            "trades": str(n_trades),
            "win_rate": win_rate,
            "total_pnl": f"${total_pnl:+.2f}",
            "sharpe": sharpe_str,
            "max_dd": max_dd_str,
            "current_pos": current_pos,
            "funding": f"${funding_total:+.2f}" if n_trades > 0 else "-",
            "first_trade": first_trade_ts,
            "last_trade": last_trade_ts,
            "runtime": runtime_str,
        })

    # --- Comparison table ---
    print("--- Comparison Table ---")
    print()
    header = f"{'Combo':>5}  {'Params':<13}  {'Trades':>6}  {'Win%':>5}  {'Total PnL':>11}  {'Sharpe':>7}  {'Max DD':>9}  {'Position':<12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['combo']:>5}  "
            f"{r['params']:<13}  "
            f"{r['trades']:>6}  "
            f"{r['win_rate']:>5}  "
            f"{r['total_pnl']:>11}  "
            f"{r['sharpe']:>7}  "
            f"{r['max_dd']:>9}  "
            f"{r['current_pos']:<12}"
        )
    print()

    # --- Funding impact ---
    print("--- Funding Impact ---")
    for r in rows:
        print(f"  Combo {r['combo']}: {r['funding']}")
    print()

    # --- Timeline ---
    print("--- Timeline ---")
    for r in rows:
        print(f"  Combo {r['combo']}: first trade {r['first_trade']}, last trade {r['last_trade']}, runtime {r['runtime']}")
    print()

    # --- Verdict ---
    scored = [r for r in rows if r["trades"] != "-" and r["trades"] != "0"]
    if len(scored) >= 2:
        print("--- Verdict ---")
        best = max(scored, key=lambda r: float(r["total_pnl"].replace("$", "").replace("+", "")))
        worst = min(scored, key=lambda r: float(r["total_pnl"].replace("$", "").replace("+", "")))
        print(f"  Best PnL:  Combo {best['combo']} ({best['params']}) at {best['total_pnl']}")
        print(f"  Worst PnL: Combo {worst['combo']} ({worst['params']}) at {worst['total_pnl']}")
        print()
    else:
        print("--- Verdict ---")
        print("  Not enough data for a verdict yet (need >= 2 combos with trades)")
        print()

    print("=" * 90)


if __name__ == "__main__":
    main()
