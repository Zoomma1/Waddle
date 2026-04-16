"""Walk-forward sweep on SOL/AVAX using Bybit linear perpetual data + funding.

This script mirrors `scripts/walk_forward_sweep.py` for the SOL/AVAX pair
but:
    1. Loads price bars from the Bybit perpetual table, not Binance spot.
    2. Loads the funding rate history and applies it to each trade's PnL.
    3. Reports three scenarios side by side:
         - Binance spot baseline   (ADR-007 baseline, no funding by construction)
         - Bybit no-funding        (Bybit prices, ignore funding — naive)
         - Bybit with funding      (Bybit prices + funding PnL per trade)

Funding PnL model
-----------------
Bybit linear perpetuals settle funding every 8h (00:00, 08:00, 16:00 UTC).
At each settlement, the party that is long a leg pays `funding_rate * notional`
if funding is positive, receives it if funding is negative.

For a dollar-neutral LONG_SPREAD position (long A, short B, equal notionals):
    funding_pnl_per_settlement = funding_B - funding_A
(you pay funding_A on your long A, receive funding_B on your short B, so net
you gain `-funding_A + funding_B`).

For SHORT_SPREAD (short A, long B) the sign flips:
    funding_pnl_per_settlement = funding_A - funding_B

We accumulate funding settlements that fall strictly after the entry
timestamp and up to and including the exit timestamp. A trade that enters
at 08:00:00 does not pay the 08:00 settlement (it's settled just before);
a trade that exits at 16:00:00 does capture the 16:00 settlement (it's
settled just after). This is a conservative convention consistent with
Bybit's docs.

Metrics
-------
We report the median test Sharpe as the "honest" statistic (mean is outlier-
sensitive on a small sample; ADR-007 uses the median as the baseline number).
We also report total PnL, mean sharpe, positive window %, total trades, and
funding impact (funding PnL as a fraction of gross PnL).
"""
from __future__ import annotations

from collections import Counter
from statistics import mean, median, stdev

import pandas as pd

from waddle.data.storage import (
    load_funding_rates,
    load_ohlcv,
    load_ohlcv_bybit,
)
from waddle.features.spread import log_spread, rolling_zscore
from waddle.strategy.pairs_reversion import Trade, metrics_dict, run_pairs_reversion

ROLLING_WINDOW = 24 * 7

TRAIN_DAYS = 90
TEST_DAYS = 30
STEP_DAYS = 15
BARS_PER_DAY = 24

PAIR_A_SPOT = "SOL/USDT"
PAIR_B_SPOT = "AVAX/USDT"
PAIR_A_PERP = "SOL/USDT:USDT"
PAIR_B_PERP = "AVAX/USDT:USDT"

PARAM_GRID = [
    {"entry_z": 1.8, "exit_z": 0.5, "max_holding": 48, "stop_loss_z": None},
    {"entry_z": 2.0, "exit_z": 0.5, "max_holding": 48, "stop_loss_z": None},
    {"entry_z": 2.0, "exit_z": 1.0, "max_holding": 48, "stop_loss_z": None},
    {"entry_z": 2.2, "exit_z": 0.5, "max_holding": 48, "stop_loss_z": None},
    {"entry_z": 2.2, "exit_z": 1.0, "max_holding": 48, "stop_loss_z": None},
    {"entry_z": 2.5, "exit_z": 0.5, "max_holding": 48, "stop_loss_z": None},
    {"entry_z": 2.5, "exit_z": 1.0, "max_holding": 48, "stop_loss_z": None},
    {"entry_z": 2.7, "exit_z": 1.0, "max_holding": 48, "stop_loss_z": None},
]

FEE_PER_SIDE = 0.0002
SLIPPAGE_PER_SIDE = 0.0002


def _combo_label(combo: dict) -> str:
    return f"entry={combo['entry_z']}/exit={combo['exit_z']}"


def _calmar(m: dict) -> float:
    if m["n_trades"] < 3:
        return -float("inf")
    if m["max_dd"] == 0:
        return float("inf") if m["total_pnl"] > 0 else 0.0
    return m["total_pnl"] / abs(m["max_dd"])


def _trade_funding_pnl(
    trade: Trade,
    funding_a: pd.Series,
    funding_b: pd.Series,
) -> float:
    """Net funding PnL for a single dollar-neutral pair trade.

    Sums the settlements held strictly after entry and up to and including
    exit. Sign convention:
        LONG_SPREAD  -> receive funding_b, pay funding_a => + (f_b - f_a)
        SHORT_SPREAD -> receive funding_a, pay funding_b => + (f_a - f_b)
    """
    entry_ts = trade.entry_ts
    exit_ts = trade.exit_ts
    settlements_a = funding_a[(funding_a.index > entry_ts) & (funding_a.index <= exit_ts)]
    settlements_b = funding_b[(funding_b.index > entry_ts) & (funding_b.index <= exit_ts)]
    joined = pd.DataFrame({"a": settlements_a, "b": settlements_b}).fillna(0.0)
    if len(joined) == 0:
        return 0.0
    if trade.direction == "LONG_SPREAD":
        net = (joined["b"] - joined["a"]).sum()
    else:
        net = (joined["a"] - joined["b"]).sum()
    return float(net)


def _apply_funding(
    trades: list[Trade],
    funding_a: pd.Series,
    funding_b: pd.Series,
) -> tuple[list[Trade], float]:
    """Return a new list of Trades with `pnl` adjusted by funding PnL.

    Second return value is the total funding PnL accumulated across all
    trades (useful for reporting the impact).
    """
    if not trades:
        return [], 0.0
    total_funding = 0.0
    adjusted: list[Trade] = []
    for t in trades:
        f = _trade_funding_pnl(t, funding_a, funding_b)
        total_funding += f
        adjusted.append(
            Trade(
                entry_ts=t.entry_ts,
                exit_ts=t.exit_ts,
                direction=t.direction,
                entry_spread=t.entry_spread,
                exit_spread=t.exit_spread,
                entry_z=t.entry_z,
                exit_z=t.exit_z,
                bars_held=t.bars_held,
                pnl=t.pnl + f,
                exit_reason=t.exit_reason,
            )
        )
    return adjusted, total_funding


def load_scenario_data(use_bybit: bool) -> dict:
    if use_bybit:
        a = load_ohlcv_bybit(PAIR_A_PERP)
        b = load_ohlcv_bybit(PAIR_B_PERP)
        funding_a = load_funding_rates(PAIR_A_PERP)["funding_rate"]
        funding_b = load_funding_rates(PAIR_B_PERP)["funding_rate"]
    else:
        a = load_ohlcv(PAIR_A_SPOT)
        b = load_ohlcv(PAIR_B_SPOT)
        funding_a = pd.Series(dtype=float)
        funding_b = pd.Series(dtype=float)

    aligned = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
    spread = log_spread(aligned["close_a"], aligned["close_b"])
    zscore = rolling_zscore(spread, window=ROLLING_WINDOW)
    return {
        "aligned": aligned,
        "spread": spread,
        "zscore": zscore,
        "funding_a": funding_a,
        "funding_b": funding_b,
    }


def make_windows(n_bars: int) -> list[tuple[int, int, int]]:
    train_bars = TRAIN_DAYS * BARS_PER_DAY
    test_bars = TEST_DAYS * BARS_PER_DAY
    step_bars = STEP_DAYS * BARS_PER_DAY
    windows: list[tuple[int, int, int]] = []
    idx = 0
    while idx + train_bars + test_bars <= n_bars:
        windows.append((idx, idx + train_bars, idx + train_bars + test_bars))
        idx += step_bars
    return windows


def run_walk_forward(
    data: dict,
    windows: list[tuple[int, int, int]],
    apply_funding: bool,
) -> dict:
    spread = data["spread"]
    zscore = data["zscore"]
    funding_a = data["funding_a"]
    funding_b = data["funding_b"]

    per_window: list[dict] = []
    total_gross_pnl = 0.0
    total_funding_pnl = 0.0
    total_fee_cost = 0.0

    for win_i, (tr_start, tr_end, te_end) in enumerate(windows):
        spread_tr = spread.iloc[tr_start:tr_end]
        zscore_tr = zscore.iloc[tr_start:tr_end]
        spread_te = spread.iloc[tr_end:te_end]
        zscore_te = zscore.iloc[tr_end:te_end]
        train_days = (spread_tr.index[-1] - spread_tr.index[0]).total_seconds() / 86400
        test_days = (spread_te.index[-1] - spread_te.index[0]).total_seconds() / 86400

        best_combo = None
        best_train_m = None
        best_calmar = -float("inf")
        for combo in PARAM_GRID:
            train_trades_raw = run_pairs_reversion(
                spread_tr,
                zscore_tr,
                **combo,
                fee_per_side=FEE_PER_SIDE,
                slippage_per_side=SLIPPAGE_PER_SIDE,
            )
            if apply_funding and train_trades_raw:
                train_trades, _ = _apply_funding(train_trades_raw, funding_a, funding_b)
            else:
                train_trades = train_trades_raw
            m = metrics_dict(train_trades, period_days=train_days)
            c = _calmar(m)
            if c > best_calmar:
                best_calmar = c
                best_combo = combo
                best_train_m = m

        if best_combo is None:
            continue

        test_trades_raw = run_pairs_reversion(
            spread_te,
            zscore_te,
            **best_combo,
            fee_per_side=FEE_PER_SIDE,
            slippage_per_side=SLIPPAGE_PER_SIDE,
        )
        if apply_funding and test_trades_raw:
            test_trades, window_funding = _apply_funding(test_trades_raw, funding_a, funding_b)
        else:
            test_trades = test_trades_raw
            window_funding = 0.0

        test_m = metrics_dict(test_trades, period_days=test_days)

        per_window.append({
            "window_i": win_i,
            "test_start": spread_te.index[0],
            "test_end": spread_te.index[-1],
            "best_combo": best_combo,
            "test_m": test_m,
            "n_trades": test_m["n_trades"],
            "funding_pnl": window_funding,
            "pnl_net_funding": test_m["total_pnl"],
        })
        total_funding_pnl += window_funding

    return _aggregate(per_window, total_funding_pnl)


def _aggregate(window_results: list[dict], total_funding_pnl: float) -> dict:
    active = [r for r in window_results if r["n_trades"] > 0]
    if not active:
        return {
            "n_windows_total": len(window_results),
            "n_windows_active": 0,
            "mean_test_sharpe": 0.0,
            "median_test_sharpe": 0.0,
            "std_test_sharpe": 0.0,
            "positive_pct": 0.0,
            "total_test_pnl": 0.0,
            "mean_test_dd": 0.0,
            "total_test_trades": 0,
            "top_combo": None,
            "total_funding_pnl": total_funding_pnl,
            "funding_share": 0.0,
            "funding_per_trade": 0.0,
        }
    sharpes = [r["test_m"]["sharpe"] for r in active]
    pnls = [r["test_m"]["total_pnl"] for r in active]
    dds = [r["test_m"]["max_dd"] for r in active]
    ns = [r["test_m"]["n_trades"] for r in active]
    total_pnl = sum(pnls)
    total_trades = sum(ns)
    n_positive = sum(1 for s in sharpes if s > 0)
    combo_counts = Counter(_combo_label(r["best_combo"]) for r in active)
    top_combo = combo_counts.most_common(1)[0][0] if combo_counts else None

    funding_share = (total_funding_pnl / total_pnl) if total_pnl else 0.0
    funding_per_trade = (total_funding_pnl / total_trades) if total_trades else 0.0

    return {
        "n_windows_total": len(window_results),
        "n_windows_active": len(active),
        "mean_test_sharpe": mean(sharpes),
        "median_test_sharpe": median(sharpes),
        "std_test_sharpe": stdev(sharpes) if len(sharpes) > 1 else 0.0,
        "positive_pct": n_positive / len(active),
        "total_test_pnl": total_pnl,
        "mean_test_dd": mean(dds),
        "total_test_trades": total_trades,
        "top_combo": top_combo,
        "total_funding_pnl": total_funding_pnl,
        "funding_share": funding_share,
        "funding_per_trade": funding_per_trade,
        "window_results": active,
    }


def print_scenario_table(scenarios: list[tuple[str, dict]]) -> None:
    print()
    print("=" * 128)
    print("SOL/AVAX walk-forward — Bybit futures + funding validation")
    print("=" * 128)
    header = (
        f"{'scenario':<24} "
        f"{'n_win':>6} "
        f"{'mean_sh':>9} "
        f"{'med_sh':>9} "
        f"{'std_sh':>8} "
        f"{'pos_%':>7} "
        f"{'tot_pnl':>10} "
        f"{'mean_dd':>10} "
        f"{'trades':>7} "
        f"{'fund_pnl':>10} "
        f"{'fund_/tr':>10}"
    )
    print(header)
    print("-" * len(header))
    for label, agg in scenarios:
        if agg["n_windows_active"] == 0:
            print(f"{label:<24} (no active windows)")
            continue
        print(
            f"{label:<24} "
            f"{agg['n_windows_active']:>6} "
            f"{agg['mean_test_sharpe']:>+9.2f} "
            f"{agg['median_test_sharpe']:>+9.2f} "
            f"{agg['std_test_sharpe']:>8.2f} "
            f"{agg['positive_pct'] * 100:>6.1f}% "
            f"{agg['total_test_pnl'] * 100:>+9.2f}% "
            f"{agg['mean_test_dd'] * 100:>+9.2f}% "
            f"{agg['total_test_trades']:>7} "
            f"{agg['total_funding_pnl'] * 100:>+9.2f}% "
            f"{agg['funding_per_trade'] * 100:>+9.4f}%"
        )

    print()
    print("Columns:")
    print("  n_win    : walk-forward windows with at least one test trade")
    print("  mean_sh  : mean OOS Sharpe (outlier-sensitive)")
    print("  med_sh   : median OOS Sharpe (honest baseline number)")
    print("  std_sh   : stdev of OOS Sharpes (variance / stability)")
    print("  pos_%    : fraction of active windows with positive OOS Sharpe")
    print("  tot_pnl  : cumulative OOS PnL across all test windows (log-return units)")
    print("  mean_dd  : mean of max_dd per test window")
    print("  trades   : total OOS trades")
    print("  fund_pnl : total funding PnL across all test trades (log-return units)")
    print("  fund_/tr : mean funding PnL per trade (log-return units)")


def print_funding_impact(with_funding_agg: dict, no_funding_agg: dict) -> None:
    print()
    print("=" * 92)
    print("FUNDING IMPACT BREAKDOWN — SOL/AVAX on Bybit")
    print("=" * 92)

    delta_total = with_funding_agg["total_test_pnl"] - no_funding_agg["total_test_pnl"]
    delta_mean_sharpe = with_funding_agg["mean_test_sharpe"] - no_funding_agg["mean_test_sharpe"]
    delta_med_sharpe = with_funding_agg["median_test_sharpe"] - no_funding_agg["median_test_sharpe"]

    print(f"  Total PnL without funding:   {no_funding_agg['total_test_pnl'] * 100:+.2f}%")
    print(f"  Total PnL with    funding:   {with_funding_agg['total_test_pnl'] * 100:+.2f}%")
    print(f"  Funding contribution total:  {delta_total * 100:+.2f}%")
    if no_funding_agg["total_test_pnl"]:
        print(f"  Funding / gross PnL:         {delta_total / no_funding_agg['total_test_pnl'] * 100:+.2f}%")
    print()
    print(f"  Mean test Sharpe without fd: {no_funding_agg['mean_test_sharpe']:+.2f}")
    print(f"  Mean test Sharpe with fd:    {with_funding_agg['mean_test_sharpe']:+.2f}")
    print(f"  Delta mean Sharpe:           {delta_mean_sharpe:+.2f}")
    print()
    print(f"  Median Sharpe without fd:    {no_funding_agg['median_test_sharpe']:+.2f}")
    print(f"  Median Sharpe with fd:       {with_funding_agg['median_test_sharpe']:+.2f}")
    print(f"  Delta median Sharpe:         {delta_med_sharpe:+.2f}")
    print()

    wr = with_funding_agg.get("window_results", [])
    if wr:
        f_pnls = [w["funding_pnl"] for w in wr]
        print(f"  Per-window funding PnL stats (N={len(f_pnls)}):")
        print(f"    min      = {min(f_pnls) * 100:+.4f}%")
        print(f"    median   = {median(f_pnls) * 100:+.4f}%")
        print(f"    mean     = {mean(f_pnls) * 100:+.4f}%")
        print(f"    max      = {max(f_pnls) * 100:+.4f}%")
        if with_funding_agg["total_test_trades"]:
            per_trade = with_funding_agg["total_funding_pnl"] / with_funding_agg["total_test_trades"]
            print(f"    per trade= {per_trade * 100:+.4f}% (avg)")


def _days_span(spread: pd.Series) -> float:
    return (spread.index[-1] - spread.index[0]).total_seconds() / 86400


def main() -> None:
    print("=" * 128)
    print("WADDLE — Bybit futures walk-forward validation on SOL/AVAX")
    print("=" * 128)
    print(f"Pair (perp): {PAIR_A_PERP} / {PAIR_B_PERP}")
    print(f"Windows: train={TRAIN_DAYS}d / test={TEST_DAYS}d / step={STEP_DAYS}d")
    print(f"Fees: {FEE_PER_SIDE * 100:.2f}%/side fee + {SLIPPAGE_PER_SIDE * 100:.2f}%/side slip")
    print(f"Grid: {len(PARAM_GRID)} combos")

    bybit_data = load_scenario_data(use_bybit=True)
    spot_data = load_scenario_data(use_bybit=False)

    print(
        f"\nBybit data:   {len(bybit_data['aligned'])} aligned bars "
        f"({bybit_data['aligned'].index[0]} -> {bybit_data['aligned'].index[-1]})  "
        f"span={_days_span(bybit_data['aligned']['close_a']):.1f}d"
    )
    print(
        f"Binance data: {len(spot_data['aligned'])} aligned bars "
        f"({spot_data['aligned'].index[0]} -> {spot_data['aligned'].index[-1]})  "
        f"span={_days_span(spot_data['aligned']['close_a']):.1f}d"
    )

    f_a = bybit_data["funding_a"]
    f_b = bybit_data["funding_b"]
    print(
        f"Funding SOL: {len(f_a)} settlements  "
        f"(mean={f_a.mean() * 100:+.4f}%, max|r|={f_a.abs().max() * 100:.4f}%)"
    )
    print(
        f"Funding AVAX: {len(f_b)} settlements  "
        f"(mean={f_b.mean() * 100:+.4f}%, max|r|={f_b.abs().max() * 100:.4f}%)"
    )

    bybit_windows = make_windows(len(bybit_data["aligned"]))
    spot_windows = make_windows(len(spot_data["aligned"]))
    print(f"\nWalk-forward windows: bybit={len(bybit_windows)}, binance={len(spot_windows)}")

    print("Running: Binance spot baseline")
    binance_agg = run_walk_forward(spot_data, spot_windows, apply_funding=False)
    print("Running: Bybit prices, NO funding (naive)")
    bybit_no_fund = run_walk_forward(bybit_data, bybit_windows, apply_funding=False)
    print("Running: Bybit prices, WITH funding (realistic)")
    bybit_with_fund = run_walk_forward(bybit_data, bybit_windows, apply_funding=True)

    scenarios = [
        ("Binance spot baseline", binance_agg),
        ("Bybit no funding", bybit_no_fund),
        ("Bybit with funding", bybit_with_fund),
    ]
    print_scenario_table(scenarios)
    print_funding_impact(bybit_with_fund, bybit_no_fund)


if __name__ == "__main__":
    main()
