"""Walk-forward sweep over cointegration filter thresholds on SOL/AVAX.

Background: the investigation D report (2026-04-16) showed that gating new entries
on Engle-Granger ADF p-value < 0.10 destroyed ~90% of the SOL/AVAX edge:

    baseline      : median Sharpe +2.24, PnL +108.47%, 155 trades
    +coint 0.10   : median Sharpe  0.00, PnL  +10.62%,  84 trades

That's a catastrophic result at the threshold recommended by ADR-007. This script
answers the follow-up question: **is there ANY threshold where the cointegration
filter actually helps (or at least doesn't hurt)?**

Strategy:
    1. Precompute the rolling Engle-Granger p-value ONCE on the full 18 months.
       This is the expensive part (~15s). Thresholds just re-mask the same series.
    2. Precompute the bear-halt mask ONCE (it's cheap but consistent with D).
    3. Loop over thresholds in [None, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.90].
       None = no cointegration filter (baseline reference, bear halt still ON).
    4. Bear halt stays ON in every scenario — it's free insurance (cost zero on
       this dataset, BTC never dips below 60k), not part of the sweep.
    5. Print a comparison table with the headline metrics + two decisive columns:
        - `mask True %` : fraction of bars where the filter lets trades through.
        - `mean PnL/trade` : trade-quality indicator. If the filter genuinely
          removes bad trades, this should rise as the threshold tightens. If it
          drops or stays flat, the filter is just noise.

The real question: does `mean PnL/trade` curve up at ANY threshold? If yes, the
ADF filter is doing real work — we just need to find the sweet spot. If no,
the 90-day ADF p-value is fundamentally the wrong signal for a 48h-holding
strategy, and ADR-007 needs a different gating metric.
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

COINT_WINDOW_BARS = 90 * 24
COINT_STEP_BARS = 24
BEAR_THRESHOLD = 60000.0

# None = baseline reference (no cointegration filter, bear halt still ON).
# The rest sweep the ADF p-value threshold from extremely strict (0.05) to
# extremely loose (0.90, i.e. "only refuse when the test is almost maximally
# inconclusive"). If even 0.90 degrades the edge, the signal is intrinsically
# noise — no threshold will save it.
THRESHOLDS: list[float | None] = [None, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.90]

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


def precompute_pvalue_and_bear(data: dict) -> dict:
    aligned = data["aligned"]
    print(
        f"Precomputing rolling Engle-Granger p-value "
        f"(window={COINT_WINDOW_BARS}b, step={COINT_STEP_BARS}b) — this is the expensive part..."
    )
    pvalue = rolling_engle_granger_pvalue(
        aligned["close_a"],
        aligned["close_b"],
        window_bars=COINT_WINDOW_BARS,
        step_bars=COINT_STEP_BARS,
    )
    bear = bear_halt_mask(data["btc_close"], aligned.index, threshold=BEAR_THRESHOLD)

    valid = pvalue.dropna()
    print(
        f"  p-value range: min={valid.min():.4f} "
        f"median={valid.median():.4f} max={valid.max():.4f}"
    )
    print(
        f"  bear halt safe (BTC > {BEAR_THRESHOLD:.0f}): "
        f"{bear.sum()}/{len(bear)} ({100 * bear.mean():.1f}% of bars)"
    )
    return {"pvalue": pvalue, "bear_halt_mask": bear}


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
    coint_mask_full: pd.Series | None,
    bear_mask_full: pd.Series,
) -> dict:
    """Run the walk-forward loop for one (threshold) scenario.

    coint_mask_full = None means "no cointegration filter" — the baseline
    reference. bear_mask_full is always provided and always applied.
    """
    spread = data["spread"]
    zscore = data["zscore"]

    window_results = []
    for win_i, (tr_start, tr_end, te_end) in enumerate(windows):
        spread_tr = spread.iloc[tr_start:tr_end]
        zscore_tr = zscore.iloc[tr_start:tr_end]
        spread_te = spread.iloc[tr_end:te_end]
        zscore_te = zscore.iloc[tr_end:te_end]

        coint_tr = (
            coint_mask_full.iloc[tr_start:tr_end] if coint_mask_full is not None else None
        )
        coint_te = (
            coint_mask_full.iloc[tr_end:te_end] if coint_mask_full is not None else None
        )
        bear_tr = bear_mask_full.iloc[tr_start:tr_end]
        bear_te = bear_mask_full.iloc[tr_end:te_end]

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
            "total_test_trades": 0,
            "mean_pnl_per_trade": 0.0,
            "top_combo": None,
        }

    test_sharpes = [r["test_m"]["sharpe"] for r in active_windows]
    test_pnls = [r["test_m"]["total_pnl"] for r in active_windows]
    test_ns = [r["test_m"]["n_trades"] for r in active_windows]

    n_positive = sum(1 for s in test_sharpes if s > 0)
    combos = Counter(_combo_label(r["best_combo"]) for r in active_windows)
    top_combo = combos.most_common(1)[0][0] if combos else None

    total_trades = sum(test_ns)
    total_pnl = sum(test_pnls)
    mean_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0.0

    return {
        "n_windows_total": len(results),
        "n_windows_active": len(active_windows),
        "mean_test_sharpe": mean(test_sharpes),
        "median_test_sharpe": median(test_sharpes),
        "std_test_sharpe": stdev(test_sharpes) if len(test_sharpes) > 1 else 0.0,
        "positive_pct": n_positive / len(active_windows),
        "total_test_pnl": total_pnl,
        "total_test_trades": total_trades,
        "mean_pnl_per_trade": mean_pnl_per_trade,
        "top_combo": top_combo,
    }


def print_comparison(scenarios: list[tuple[str, float, dict]]) -> None:
    print()
    print("=" * 124)
    print("SOL/AVAX walk-forward — cointegration threshold sweep (bear halt ON everywhere)")
    print("=" * 124)
    header = (
        f"{'threshold':<12} "
        f"{'mask_T%':>8} "
        f"{'n_wins':>7} "
        f"{'mean_sh':>9} "
        f"{'med_sh':>9} "
        f"{'pos_%':>7} "
        f"{'tot_pnl':>10} "
        f"{'n_trd':>7} "
        f"{'pnl/trd':>10} "
        f"{'top_combo':<22}"
    )
    print(header)
    print("-" * len(header))
    for label, mask_true_pct, agg in scenarios:
        if agg["n_windows_active"] == 0:
            print(f"{label:<12} {mask_true_pct * 100:>7.1f}% (no active windows)")
            continue
        print(
            f"{label:<12} "
            f"{mask_true_pct * 100:>7.1f}% "
            f"{agg['n_windows_active']:>7} "
            f"{agg['mean_test_sharpe']:>+9.2f} "
            f"{agg['median_test_sharpe']:>+9.2f} "
            f"{agg['positive_pct'] * 100:>6.1f}% "
            f"{agg['total_test_pnl'] * 100:>+9.2f}% "
            f"{agg['total_test_trades']:>7} "
            f"{agg['mean_pnl_per_trade'] * 100:>+9.4f}% "
            f"{agg['top_combo'] or '':<22}"
        )
    print()
    print("Columns:")
    print("  threshold : cointegration ADF p-value cap (None = filter off, bear halt only)")
    print("  mask_T%   : fraction of bars where the coint filter lets trades through")
    print("              (at None: 100% — no filter; at 0.05: very strict; at 0.90: almost open)")
    print("  n_wins    : walk-forward windows with at least one test trade")
    print("  mean_sh   : mean OOS Sharpe across active windows")
    print("  med_sh    : median OOS Sharpe (outlier-resistant — the honest baseline)")
    print("  pos_%     : fraction of active windows with positive OOS Sharpe")
    print("  tot_pnl   : sum of OOS PnL across all test windows (log-return units, ~%)")
    print("  n_trd     : total trades executed across all test windows")
    print("  pnl/trd   : mean PnL per trade = tot_pnl / n_trd — trade quality indicator")
    print("              If filter discriminates (kills bad trades), this should RISE as")
    print("              threshold tightens. If noise, it stays flat or drops.")
    print("  top_combo : entry/exit combo most often selected on train by Calmar")


def main() -> None:
    print("=" * 124)
    print("WADDLE — SOL/AVAX walk-forward cointegration THRESHOLD SWEEP")
    print("=" * 124)
    print(f"Pair: {PAIR_A} vs {PAIR_B}")
    print(f"Windows: train={TRAIN_DAYS}d / test={TEST_DAYS}d / step={STEP_DAYS}d")
    print(f"Fees: {FEE_PER_SIDE * 100:.2f}%/side fee + {SLIPPAGE_PER_SIDE * 100:.2f}%/side slip")
    print(f"Grid: {len(PARAM_GRID)} combos")
    print(f"Thresholds tested: {THRESHOLDS}")
    print(f"Bear halt (BTC > {BEAR_THRESHOLD:.0f}): ALWAYS ON in every scenario")

    data = load_data()
    print(
        f"\nData: {len(data['aligned'])} bars loaded "
        f"({data['aligned'].index[0].date()} -> {data['aligned'].index[-1].date()})"
    )

    cached = precompute_pvalue_and_bear(data)
    pvalue_series = cached["pvalue"]
    bear_full = cached["bear_halt_mask"]

    windows = make_windows(len(data["aligned"]))
    print(f"\nRolling windows generated: {len(windows)}")

    results: list[tuple[str, float, dict]] = []
    for threshold in THRESHOLDS:
        if threshold is None:
            label = "None"
            coint_full = None
            mask_true_pct = 1.0  # no filter = 100% pass-through
        else:
            label = f"{threshold:.2f}"
            coint_full = cointegration_mask(pvalue_series, threshold=threshold)
            mask_true_pct = float(coint_full.mean())

        print(
            f"Running scenario: threshold={label}  "
            f"(coint mask lets through {mask_true_pct * 100:.1f}% of bars)"
        )
        agg = run_scenario(data, windows, coint_full, bear_full)
        results.append((label, mask_true_pct, agg))

    print_comparison(results)


if __name__ == "__main__":
    main()
