"""Walk-forward sweep on SOL/AVAX with optional mandatory filters.

ADR-007 says we cannot deploy the SOL/AVAX baseline without two filters:
    1. Rolling Engle-Granger p-value < 0.10 (cointegration active)
    2. BTC close > 60000 USDT (no trading in an untested bear regime)

This script runs the exact same walk-forward logic as scripts/walk_forward_sweep.py,
but only on SOL/AVAX, and compares four scenarios side-by-side:

    baseline      : no filters
    +cointegration: entry gated by rolling 90d ADF p-value < 0.10
    +bear halt    : entry gated by BTC close > 60000
    +both         : both filters combined

The goal is to answer the deployment question: do these mandatory filters
*improve* or *hurt* the SOL/AVAX edge? If the filter removes the 60% of time
it excludes because that time was unprofitable, Sharpe goes up. If it throws
away real profit, Sharpe goes down — and we'd have to rethink the threshold.

The cointegration p-value series and the bear halt mask are pre-computed ONCE
on the full 18 months of data, then sliced per window. No recomputation.
"""
from __future__ import annotations

from collections import Counter
from statistics import mean, median, stdev

import pandas as pd

from waddle.data.storage import load_ohlcv
from waddle.features.cointegration import (
    bear_halt_mask,
    cointegration_mask,
    rolling_engle_granger_pvalue,
)
from waddle.features.spread import log_spread, rolling_zscore
from waddle.strategy.pairs_reversion import metrics_dict, run_pairs_reversion

ROLLING_WINDOW = 24 * 7

TRAIN_DAYS = 90
TEST_DAYS = 30
STEP_DAYS = 15
BARS_PER_DAY = 24

COINT_WINDOW_BARS = 90 * 24  # 2160 = 90 days
COINT_STEP_BARS = 24          # recompute daily
COINT_THRESHOLD = 0.10
BEAR_THRESHOLD = 60000.0

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

PAIR_A = "SOL/USDT"
PAIR_B = "AVAX/USDT"
BTC_SYMBOL = "BTC/USDT"

FEE_PER_SIDE = 0.0002
SLIPPAGE_PER_SIDE = 0.0002


def _calmar(m: dict) -> float:
    if m["n_trades"] < 3:
        return -float("inf")
    if m["max_dd"] == 0:
        return float("inf") if m["total_pnl"] > 0 else 0.0
    return m["total_pnl"] / abs(m["max_dd"])


def _combo_label(combo: dict) -> str:
    return f"entry={combo['entry_z']}/exit={combo['exit_z']}"


def load_data() -> dict:
    sol = load_ohlcv(PAIR_A)
    avax = load_ohlcv(PAIR_B)
    btc = load_ohlcv(BTC_SYMBOL)
    aligned = sol.join(avax, how="inner", lsuffix="_a", rsuffix="_b")
    spread = log_spread(aligned["close_a"], aligned["close_b"])
    zscore = rolling_zscore(spread, window=ROLLING_WINDOW)
    return {
        "aligned": aligned,
        "spread": spread,
        "zscore": zscore,
        "btc_close": btc["close"],
    }


def precompute_filters(data: dict) -> dict:
    aligned = data["aligned"]
    print(f"Precomputing rolling Engle-Granger p-value "
          f"(window={COINT_WINDOW_BARS}b, step={COINT_STEP_BARS}b)...")
    pvalue = rolling_engle_granger_pvalue(
        aligned["close_a"],
        aligned["close_b"],
        window_bars=COINT_WINDOW_BARS,
        step_bars=COINT_STEP_BARS,
    )
    coint = cointegration_mask(pvalue, threshold=COINT_THRESHOLD)
    bear = bear_halt_mask(data["btc_close"], aligned.index, threshold=BEAR_THRESHOLD)
    both = coint & bear

    valid_pvalue = pvalue.dropna()
    print(f"  p-value range: min={valid_pvalue.min():.4f} "
          f"median={valid_pvalue.median():.4f} max={valid_pvalue.max():.4f}")
    print(f"  cointegration active: {coint.sum()}/{len(coint)} "
          f"({100 * coint.mean():.1f}% of bars)")
    print(f"  bear halt safe:       {bear.sum()}/{len(bear)} "
          f"({100 * bear.mean():.1f}% of bars)")
    print(f"  both filters pass:    {both.sum()}/{len(both)} "
          f"({100 * both.mean():.1f}% of bars)")
    print(f"  BTC close range:      min={data['btc_close'].min():.0f} "
          f"max={data['btc_close'].max():.0f}")
    return {
        "pvalue": pvalue,
        "cointegration_mask": coint,
        "bear_halt_mask": bear,
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


def run_scenario(
    data: dict,
    windows: list[tuple[int, int, int]],
    use_coint: bool,
    use_bear: bool,
    filters: dict,
) -> dict:
    spread = data["spread"]
    zscore = data["zscore"]

    coint_mask_full = filters["cointegration_mask"] if use_coint else None
    bear_mask_full = filters["bear_halt_mask"] if use_bear else None

    window_results = []
    for win_i, (tr_start, tr_end, te_end) in enumerate(windows):
        spread_tr = spread.iloc[tr_start:tr_end]
        zscore_tr = zscore.iloc[tr_start:tr_end]
        spread_te = spread.iloc[tr_end:te_end]
        zscore_te = zscore.iloc[tr_end:te_end]

        coint_tr = coint_mask_full.iloc[tr_start:tr_end] if coint_mask_full is not None else None
        bear_tr = bear_mask_full.iloc[tr_start:tr_end] if bear_mask_full is not None else None
        coint_te = coint_mask_full.iloc[tr_end:te_end] if coint_mask_full is not None else None
        bear_te = bear_mask_full.iloc[tr_end:te_end] if bear_mask_full is not None else None

        train_days_real = (spread_tr.index[-1] - spread_tr.index[0]).total_seconds() / 86400
        test_days_real = (spread_te.index[-1] - spread_te.index[0]).total_seconds() / 86400

        best_combo = None
        best_train_m = None
        best_calmar = -float("inf")
        for combo in PARAM_GRID:
            trades = run_pairs_reversion(
                spread_tr,
                zscore_tr,
                **combo,
                fee_per_side=FEE_PER_SIDE,
                slippage_per_side=SLIPPAGE_PER_SIDE,
                cointegration_mask=coint_tr,
                regime_mask=bear_tr,
            )
            m = metrics_dict(trades, period_days=train_days_real)
            c = _calmar(m)
            if c > best_calmar:
                best_calmar = c
                best_combo = combo
                best_train_m = m

        if best_combo is None:
            continue

        test_trades = run_pairs_reversion(
            spread_te,
            zscore_te,
            **best_combo,
            fee_per_side=FEE_PER_SIDE,
            slippage_per_side=SLIPPAGE_PER_SIDE,
            cointegration_mask=coint_te,
            regime_mask=bear_te,
        )
        test_m = metrics_dict(test_trades, period_days=test_days_real)

        window_results.append({
            "window_i": win_i,
            "best_combo": best_combo,
            "train_m": best_train_m,
            "test_m": test_m,
        })

    return _aggregate(window_results)


def _aggregate(results: list[dict]) -> dict:
    active_windows = [r for r in results if r["test_m"]["n_trades"] > 0]
    if not active_windows:
        return {
            "n_windows_total": len(results),
            "n_windows_active": 0,
            "mean_test_sharpe": 0.0,
            "median_test_sharpe": 0.0,
            "std_test_sharpe": 0.0,
            "positive_pct": 0.0,
            "total_test_pnl": 0.0,
            "mean_test_dd": 0.0,
            "total_test_trades": 0,
            "top_combo": None,
        }

    test_sharpes = [r["test_m"]["sharpe"] for r in active_windows]
    test_pnls = [r["test_m"]["total_pnl"] for r in active_windows]
    test_dds = [r["test_m"]["max_dd"] for r in active_windows]
    test_ns = [r["test_m"]["n_trades"] for r in active_windows]

    n_positive = sum(1 for s in test_sharpes if s > 0)
    combos = Counter(_combo_label(r["best_combo"]) for r in active_windows)
    top_combo = combos.most_common(1)[0][0] if combos else None

    return {
        "n_windows_total": len(results),
        "n_windows_active": len(active_windows),
        "mean_test_sharpe": mean(test_sharpes),
        "median_test_sharpe": median(test_sharpes),
        "std_test_sharpe": stdev(test_sharpes) if len(test_sharpes) > 1 else 0.0,
        "positive_pct": n_positive / len(active_windows),
        "total_test_pnl": sum(test_pnls),
        "mean_test_dd": mean(test_dds),
        "total_test_trades": sum(test_ns),
        "top_combo": top_combo,
    }


def print_comparison(scenarios: list[tuple[str, dict]]) -> None:
    print()
    print("=" * 112)
    print("SOL/AVAX walk-forward — filter impact comparison")
    print("=" * 112)
    header = (
        f"{'scenario':<22} "
        f"{'n_wins':>7} "
        f"{'mean_sh':>9} "
        f"{'med_sh':>9} "
        f"{'std_sh':>9} "
        f"{'pos_%':>7} "
        f"{'tot_pnl':>10} "
        f"{'mean_dd':>10} "
        f"{'tot_trd':>8} "
        f"{'top_combo':<22}"
    )
    print(header)
    print("-" * len(header))
    for label, agg in scenarios:
        if agg["n_windows_active"] == 0:
            print(f"{label:<22} (no active windows)")
            continue
        print(
            f"{label:<22} "
            f"{agg['n_windows_active']:>7} "
            f"{agg['mean_test_sharpe']:>+9.2f} "
            f"{agg['median_test_sharpe']:>+9.2f} "
            f"{agg['std_test_sharpe']:>9.2f} "
            f"{agg['positive_pct'] * 100:>6.1f}% "
            f"{agg['total_test_pnl'] * 100:>+9.2f}% "
            f"{agg['mean_test_dd'] * 100:>+9.2f}% "
            f"{agg['total_test_trades']:>8} "
            f"{agg['top_combo'] or '':<22}"
        )
    print()
    print("Columns:")
    print("  n_wins  : walk-forward windows where at least one test trade happened")
    print("  mean_sh : mean OOS Sharpe across active windows")
    print("  med_sh  : median OOS Sharpe (less outlier-sensitive — honest baseline)")
    print("  std_sh  : standard deviation of OOS Sharpes (variance / stability)")
    print("  pos_%   : fraction of active windows with positive OOS Sharpe")
    print("  tot_pnl : sum of OOS PnL across all test windows (log-return units)")
    print("  mean_dd : mean of max_dd observed per test window (negative = loss)")
    print("  tot_trd : total number of trades executed across all test windows")
    print("  top_combo: entry/exit combo most often picked by Calmar on train")


def main() -> None:
    print("=" * 112)
    print("WADDLE — SOL/AVAX walk-forward sweep, with mandatory ADR-007 filters")
    print("=" * 112)
    print(f"Pair: {PAIR_A} vs {PAIR_B}")
    print(f"Windows: train={TRAIN_DAYS}d / test={TEST_DAYS}d / step={STEP_DAYS}d")
    print(f"Fees: {FEE_PER_SIDE * 100:.2f}%/side fee + {SLIPPAGE_PER_SIDE * 100:.2f}%/side slip")
    print(f"Grid: {len(PARAM_GRID)} combos")
    print(f"Cointegration filter: p-value < {COINT_THRESHOLD} on rolling {COINT_WINDOW_BARS}b window")
    print(f"Bear halt filter:     BTC/USDT > {BEAR_THRESHOLD:.0f} USDT")

    data = load_data()
    print(f"\nData: {len(data['aligned'])} bars loaded "
          f"({data['aligned'].index[0].date()} -> {data['aligned'].index[-1].date()})")

    filters = precompute_filters(data)

    windows = make_windows(len(data["aligned"]))
    print(f"\nRolling windows generated: {len(windows)}")

    scenarios = [
        ("baseline", False, False),
        ("+cointegration", True, False),
        ("+bear halt", False, True),
        ("+both", True, True),
    ]
    results: list[tuple[str, dict]] = []
    for label, use_coint, use_bear in scenarios:
        print(f"Running scenario: {label} (coint={use_coint}, bear={use_bear})")
        agg = run_scenario(data, windows, use_coint, use_bear, filters)
        results.append((label, agg))

    print_comparison(results)


if __name__ == "__main__":
    main()
