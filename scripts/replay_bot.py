"""Replay the bot over historical Bybit data and compare fidelity with the backtest.

This is the end-to-end validation: given identical inputs (same prices, same
params, same fees, same funding), the bot's engine should produce the same
trades as `run_pairs_reversion` run directly on the log spread. Any significant
deviation is a bug.

Usage:
    uv run scripts/replay_bot.py
    uv run scripts/replay_bot.py --start 2026-01-01 --end 2026-04-15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from waddle.bot.config import load_active_params, load_bot_config
from waddle.bot.engine import tick
from waddle.bot.replay import (
    HistoricalBybitAdapter,
    ReplayClock,
    load_replay_caches,
)
from waddle.bot.state import StateStore
from waddle.features.spread import log_spread, rolling_zscore
from waddle.strategy.pairs_reversion import run_pairs_reversion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01-01", help="Replay start date (UTC)")
    parser.add_argument("--end", default="2026-04-15", help="Replay end date (UTC)")
    parser.add_argument(
        "--db",
        default="data/waddle_bot_replay.db",
        help="Replay state db path (will be deleted)",
    )
    parser.add_argument(
        "--config",
        default="config/bot.yaml",
        help="Path to bot.yaml",
    )
    parser.add_argument(
        "--active-params",
        default="config/active_params.json",
        help="Path to active_params.json",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    print(f"Replay window: {start} -> {end}")

    config = load_bot_config(Path(args.config))
    active_params = load_active_params(Path(args.active_params))
    print(
        f"Active params: entry_z={active_params.entry_z} "
        f"exit_z={active_params.exit_z} "
        f"max_holding={active_params.max_holding_bars} "
        f"zscore_window={active_params.zscore_window}"
    )

    db_path = Path(args.db)
    if db_path.exists():
        db_path.unlink()
    state = StateStore(db_path)
    state.init_schema()

    warmup = pd.Timedelta(days=14)
    cached_ohlcv, cached_funding = load_replay_caches(
        symbols=[
            config.pair.symbol_a,
            config.pair.symbol_b,
            config.pair.reference_btc,
        ],
        start=start - warmup,
        end=end,
    )
    print(
        f"Cache bars: "
        f"{config.pair.symbol_a}={len(cached_ohlcv[config.pair.symbol_a])}, "
        f"{config.pair.symbol_b}={len(cached_ohlcv[config.pair.symbol_b])}, "
        f"{config.pair.reference_btc}={len(cached_ohlcv[config.pair.reference_btc])}"
    )

    clock = ReplayClock(start=start)
    adapter = HistoricalBybitAdapter(clock, config.fees, cached_ohlcv, cached_funding)

    current = start
    n_ticks = 0
    n_errors = 0
    while current <= end:
        clock.set(current)
        try:
            tick(state, config, adapter, clock, Path(args.active_params))
        except Exception as e:
            n_errors += 1
            if n_errors <= 3:
                print(f"  tick at {current} failed: {type(e).__name__}: {e}")
        n_ticks += 1
        current = current + pd.Timedelta(hours=1)

    bot_trades = state.list_trades(limit=100000)
    bot_trades_sorted = sorted(bot_trades, key=lambda t: t.exit_ts)
    bot_trade_count = len(bot_trades_sorted)
    bot_pnl_net = sum(t.pnl_net for t in bot_trades_sorted)

    open_pos = state.get_open_position()
    open_pos_note = ""
    if open_pos is not None:
        open_pos_note = (
            f"  WARNING: 1 position still open at replay end "
            f"(entry_ts={open_pos.entry_ts}, direction={open_pos.direction}), not counted"
        )

    print(
        f"\n[BOT REPLAY] {n_ticks} ticks ({n_errors} errors), "
        f"{bot_trade_count} closed trades, net PnL ${bot_pnl_net:+.2f}"
    )
    if open_pos_note:
        print(open_pos_note)

    bars_a_full = cached_ohlcv[config.pair.symbol_a]
    bars_b_full = cached_ohlcv[config.pair.symbol_b]
    aligned_full = bars_a_full[["close"]].join(
        bars_b_full[["close"]], how="inner", lsuffix="_a", rsuffix="_b"
    )

    spread_full = log_spread(aligned_full["close_a"], aligned_full["close_b"])
    zscore_full = rolling_zscore(spread_full, window=active_params.zscore_window)

    in_window = (spread_full.index >= start) & (spread_full.index <= end)
    spread_window = spread_full[in_window]
    zscore_window = zscore_full.loc[spread_window.index]

    oracle_trades = run_pairs_reversion(
        spread_window,
        zscore_window,
        entry_z=active_params.entry_z,
        exit_z=active_params.exit_z,
        max_holding=active_params.max_holding_bars,
        stop_loss_z=None,
        fee_per_side=config.fees.fee_per_side,
        slippage_per_side=config.fees.slippage_per_side,
    )
    oracle_pnl_log = sum(t.pnl for t in oracle_trades)
    oracle_pnl_usd = oracle_pnl_log * config.risk.leg_notional_usd

    print(
        f"[ORACLE] run_pairs_reversion: {len(oracle_trades)} trades, "
        f"net PnL ${oracle_pnl_usd:+.2f} (log pnl {oracle_pnl_log:+.6f})"
    )

    print("\n[COMPARISON]")
    trade_diff = bot_trade_count - len(oracle_trades)
    pnl_diff = bot_pnl_net - oracle_pnl_usd
    print(
        f"  Trade count: bot={bot_trade_count}, "
        f"oracle={len(oracle_trades)}, diff={trade_diff}"
    )
    print(
        f"  Net PnL:     bot=${bot_pnl_net:+.2f}, "
        f"oracle=${oracle_pnl_usd:+.2f}, diff=${pnl_diff:+.2f}"
    )

    abs_tolerance = max(abs(oracle_pnl_usd) * 0.05, 1.0)
    if (
        abs(pnl_diff) <= abs_tolerance
        and bot_trade_count == len(oracle_trades)
    ):
        print("  VERDICT: PASS — bot matches oracle within 5% tolerance")
    else:
        print("  VERDICT: DRIFT — investigate")

    max_print = 8
    print(f"\n  Bot trades (first {max_print}):")
    for t in bot_trades_sorted[:max_print]:
        print(
            f"    exit_ts={t.exit_ts} reason={t.exit_reason:<14} "
            f"bars={t.bars_held:>3} "
            f"pnl_gross=${t.pnl_gross:+.2f} pnl_fees=${t.pnl_fees:+.2f} "
            f"pnl_fund=${t.pnl_funding:+.2f} pnl_net=${t.pnl_net:+.2f}"
        )
    print(f"\n  Oracle trades (first {max_print}):")
    for t in oracle_trades[:max_print]:
        usd = t.pnl * config.risk.leg_notional_usd
        print(
            f"    exit_ts={t.exit_ts} reason={t.exit_reason:<14} "
            f"bars={t.bars_held:>3} pnl_log={t.pnl:+.6f} (usd=${usd:+.2f})"
        )


if __name__ == "__main__":
    main()
