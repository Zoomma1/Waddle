"""Replay all 4 fleet combos over historical Bybit data and produce a comparison table.

Generates the theoretical baseline for each combo so live fleet performance can
be measured against expected. Uses the real engine tick() -- not run_pairs_reversion --
for fidelity with the live bots.

Usage:
    uv run scripts/replay_paper_fleet.py
    uv run scripts/replay_paper_fleet.py --start 2026-01-01 --end 2026-04-15
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from waddle.bot.config import load_bot_config, load_active_params
from waddle.bot.engine import tick
from waddle.bot.replay import (
    HistoricalBybitAdapter,
    ReplayClock,
    load_replay_caches,
)
from waddle.bot.state import StateStore, BotTrade


COMBOS = [
    {
        "name": "combo1",
        "bot_yaml": Path("config/fleet/combo1_bot.yaml"),
        "params_json": Path("config/fleet/combo1_params.json"),
        "replay_db": Path("data/replay_fleet_combo1.db"),
    },
    {
        "name": "combo2",
        "bot_yaml": Path("config/fleet/combo2_bot.yaml"),
        "params_json": Path("config/fleet/combo2_params.json"),
        "replay_db": Path("data/replay_fleet_combo2.db"),
    },
    {
        "name": "combo3",
        "bot_yaml": Path("config/fleet/combo3_bot.yaml"),
        "params_json": Path("config/fleet/combo3_params.json"),
        "replay_db": Path("data/replay_fleet_combo3.db"),
    },
    {
        "name": "combo4",
        "bot_yaml": Path("config/fleet/combo4_bot.yaml"),
        "params_json": Path("config/fleet/combo4_params.json"),
        "replay_db": Path("data/replay_fleet_combo4.db"),
    },
]


def _sharpe_from_trades(trades: list[BotTrade]) -> float:
    if len(trades) < 2:
        return 0.0
    pnls = [t.pnl_net for t in trades]
    mean = sum(pnls) / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return 0.0
    per_trade_sharpe = mean / std
    avg_bars = sum(t.bars_held for t in trades) / len(trades)
    trades_per_year = 8760 / max(avg_bars, 1)
    return per_trade_sharpe * math.sqrt(trades_per_year)


def _max_drawdown(trades: list[BotTrade]) -> float:
    if not trades:
        return 0.0
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        equity += t.pnl_net
        if equity > peak:
            peak = equity
        dd = equity - peak
        if dd < max_dd:
            max_dd = dd
    return max_dd


def replay_combo(
    combo: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cached_ohlcv: dict[str, pd.DataFrame],
    cached_funding: dict[str, pd.DataFrame],
) -> dict:
    config = load_bot_config(combo["bot_yaml"])
    active_params = load_active_params(combo["params_json"])

    db_path = combo["replay_db"]
    if db_path.exists():
        db_path.unlink()
    state = StateStore(db_path)
    state.init_schema()

    clock = ReplayClock(start=start)
    adapter = HistoricalBybitAdapter(clock, config.fees, cached_ohlcv, cached_funding)

    current = start
    n_ticks = 0
    n_errors = 0
    while current <= end:
        clock.set(current)
        try:
            tick(state, config, adapter, clock, combo["params_json"])
        except Exception as e:
            n_errors += 1
            if n_errors <= 3:
                print(f"  [{combo['name']}] tick at {current} failed: {type(e).__name__}: {e}")
        n_ticks += 1
        current += pd.Timedelta(hours=1)

    trades = state.list_trades(limit=100000)
    trades_sorted = sorted(trades, key=lambda t: t.exit_ts)
    open_pos = state.get_open_position()

    trade_count = len(trades_sorted)
    total_pnl = sum(t.pnl_net for t in trades_sorted)
    pnl_per_trade = total_pnl / trade_count if trade_count > 0 else 0.0
    wins = sum(1 for t in trades_sorted if t.pnl_net > 0)
    win_pct = (wins / trade_count * 100) if trade_count > 0 else 0.0
    avg_hold = sum(t.bars_held for t in trades_sorted) / trade_count if trade_count > 0 else 0.0
    max_dd = _max_drawdown(trades_sorted)
    sharpe = _sharpe_from_trades(trades_sorted)

    return {
        "name": combo["name"],
        "entry_z": active_params.entry_z,
        "exit_z": active_params.exit_z,
        "trades": trades_sorted,
        "trade_count": trade_count,
        "total_pnl": total_pnl,
        "pnl_per_trade": pnl_per_trade,
        "win_pct": win_pct,
        "avg_hold": avg_hold,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "n_ticks": n_ticks,
        "n_errors": n_errors,
        "open_position": open_pos,
    }


def format_table(results: list[dict], start: str, end: str) -> str:
    lines = []
    lines.append(f"FLEET BASELINE -- Historical replay ({start} -> {end})")
    lines.append("Bybit perpetuals, fees 0.02%/side + 0.02% slip, max_holding=48h, zscore_window=168h")
    lines.append("")
    header = f"{'Combo':<9}{'Params':<16}{'Trades':>7}{'Win%':>7}{'Total PnL':>12}{'PnL/trade':>11}{'Max DD':>10}{'Sharpe*':>9}{'Avg hold':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        params_str = f"{r['entry_z']:.1f} / {r['exit_z']:.1f}"
        lines.append(
            f"{r['name']:<9}"
            f"{params_str:<16}"
            f"{r['trade_count']:>7}"
            f"{r['win_pct']:>6.0f}%"
            f"  ${r['total_pnl']:>+8.2f}"
            f"  ${r['pnl_per_trade']:>+7.2f}"
            f"  ${r['max_dd']:>+7.2f}"
            f"{r['sharpe']:>9.2f}"
            f"{r['avg_hold']:>8.0f}h"
        )

    lines.append("")
    lines.append("* Sharpe = annualized per-trade Sharpe from closed trade PnLs")
    return "\n".join(lines)


def format_trade_list(results: list[dict], max_trades: int = 5) -> str:
    lines = []
    for r in results:
        lines.append(f"\n--- {r['name']} (entry={r['entry_z']:.1f}, exit={r['exit_z']:.1f}) "
                      f"first {min(max_trades, r['trade_count'])} trades ---")
        for t in r["trades"][:max_trades]:
            lines.append(
                f"  exit={t.exit_ts.strftime('%Y-%m-%d %H:%M')} "
                f"reason={t.exit_reason:<14} "
                f"bars={t.bars_held:>3}h "
                f"pnl_gross=${t.pnl_gross:+.2f} "
                f"fees=${t.pnl_fees:+.2f} "
                f"fund=${t.pnl_funding:+.2f} "
                f"net=${t.pnl_net:+.2f}"
            )
        if r["open_position"] is not None:
            lines.append(f"  [!] 1 position still open at replay end (not counted)")
    return "\n".join(lines)


def format_commentary(results: list[dict]) -> str:
    lines = ["\n=== COMMENTARY ==="]

    most_active = max(results, key=lambda r: r["trade_count"])
    lines.append(
        f"- {most_active['name']} ({most_active['entry_z']:.1f}/{most_active['exit_z']:.1f}) "
        f"is the most active with {most_active['trade_count']} trades -- "
        f"lowest entry threshold triggers the most signals."
    )

    profitable = [r for r in results if r["trade_count"] > 0]
    if profitable:
        best_per_trade = max(profitable, key=lambda r: r["pnl_per_trade"])
        lines.append(
            f"- {best_per_trade['name']} ({best_per_trade['entry_z']:.1f}/{best_per_trade['exit_z']:.1f}) "
            f"has the highest PnL/trade at ${best_per_trade['pnl_per_trade']:+.2f} -- "
            f"fewer but higher-conviction trades."
        )

    risk_adj = [r for r in results if r["sharpe"] != 0]
    if risk_adj:
        best_sharpe = max(risk_adj, key=lambda r: r["sharpe"])
        lines.append(
            f"- {best_sharpe['name']} ({best_sharpe['entry_z']:.1f}/{best_sharpe['exit_z']:.1f}) "
            f"has the best risk-adjusted return (Sharpe {best_sharpe['sharpe']:.2f})."
        )

    worst_pnl = min(results, key=lambda r: r["total_pnl"])
    if worst_pnl["total_pnl"] < max(r["total_pnl"] for r in results):
        lines.append(
            f"- {worst_pnl['name']} ({worst_pnl['entry_z']:.1f}/{worst_pnl['exit_z']:.1f}) "
            f"is the weakest performer at ${worst_pnl['total_pnl']:+.2f} total PnL."
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay all 4 fleet combos")
    parser.add_argument("--start", default="2026-01-01", help="Replay start date (UTC)")
    parser.add_argument("--end", default="2026-04-15", help="Replay end date (UTC)")
    args = parser.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    print(f"Fleet replay window: {start} -> {end}")
    print(f"Replaying {len(COMBOS)} combos...\n")

    ref_config = load_bot_config(COMBOS[0]["bot_yaml"])
    warmup = pd.Timedelta(days=14)
    symbols = [
        ref_config.pair.symbol_a,
        ref_config.pair.symbol_b,
        ref_config.pair.reference_btc,
    ]
    print(f"Loading caches for {symbols}...")
    cached_ohlcv, cached_funding = load_replay_caches(
        symbols=symbols,
        start=start - warmup,
        end=end,
    )
    for sym in symbols:
        print(f"  {sym}: {len(cached_ohlcv[sym])} bars, {len(cached_funding.get(sym, []))} funding")
    print()

    results = []
    for combo in COMBOS:
        print(f"--- Replaying {combo['name']} ---")
        r = replay_combo(combo, start, end, cached_ohlcv, cached_funding)
        print(
            f"    {r['n_ticks']} ticks ({r['n_errors']} errors), "
            f"{r['trade_count']} trades, PnL ${r['total_pnl']:+.2f}"
        )
        if r["open_position"] is not None:
            print(f"    [!] 1 position still open at replay end")
        results.append(r)

    print("\n" + "=" * 80)
    table = format_table(results, args.start, args.end)
    print(table)

    trade_list = format_trade_list(results)
    print(trade_list)

    commentary = format_commentary(results)
    print(commentary)

    # --- Sanity check: combo1 vs Investigation H ---
    print("\n=== SANITY CHECK: combo1 vs Investigation H ===")
    c1 = results[0]
    inv_h_trades = 31
    inv_h_pnl = 139.46
    print(f"  Investigation H: {inv_h_trades} trades, ${inv_h_pnl:+.2f} net PnL")
    print(f"  Fleet replay:    {c1['trade_count']} trades, ${c1['total_pnl']:+.2f} net PnL")
    trade_match = c1["trade_count"] == inv_h_trades
    pnl_diff = abs(c1["total_pnl"] - inv_h_pnl)
    pnl_tolerance = max(abs(inv_h_pnl) * 0.05, 5.0)
    pnl_match = pnl_diff <= pnl_tolerance
    if trade_match and pnl_match:
        print(f"  VERDICT: PASS (trade count match, PnL diff ${pnl_diff:.2f} within ${pnl_tolerance:.2f} tolerance)")
    else:
        if not trade_match:
            print(f"  VERDICT: DRIFT -- trade count {c1['trade_count']} != {inv_h_trades}")
        if not pnl_match:
            print(f"  VERDICT: DRIFT -- PnL diff ${pnl_diff:.2f} exceeds ${pnl_tolerance:.2f} tolerance")


if __name__ == "__main__":
    main()
