"""Walk-forward sweep over rolling half-life filter thresholds on SOL/AVAX.

Background: ADR-008 (2026-04-15) retired the Engle-Granger ADF cointegration
filter after investigation E proved that no threshold could improve the SOL/AVAX
baseline. The diagnostic: the ADF test operates on a 90-day window while the
strategy holds for 48h max — the horizons are decoupled, and PnL/trade *dropped*
as the filter tightened (the signature of a noisy / anti-correlated filter).

ADR-008 §"Exploration d'alternatives" named rolling half-life as the main
candidate (a) because it's a *direct* measure of mean-reversion speed and lives
on a compatible time scale (30-60 days of lookback, expressing "how fast does a
shock revert?"). This script answers the empirical question: does the half-life
filter improve, maintain, or degrade the SOL/AVAX baseline?

Critical metric: **mean PnL/trade**. If it RISES as the threshold tightens, the
filter is discriminating well (removes bad trades). If it stays flat or drops,
the filter is just noise — same failure mode as the ADF filter.

Strategy:
    1. Precompute the rolling half-life series ONCE on the full 18 months.
    2. Bear halt ON in every scenario (free insurance, not part of the sweep).
    3. Loop over half-life thresholds: [None, 10, 20, 30, 45, 60, 90] days.
       None = no half-life filter (baseline = ADR-008 config, bear halt only).
    4. Same walk-forward config as walk_forward_sweep.py + walk_forward_threshold_sweep.py:
       train=90d / test=30d / step=15d, 8-combo grid, Calmar selection.
    5. Print comparison table with the headline metrics + mask_True% + PnL/trade.
"""
from __future__ import annotations

from collections import Counter
from statistics import mean, median, stdev

import pandas as pd

from waddle.data.storage import load_ohlcv
from waddle.features.cointegration import (
    bear_halt_mask,
    half_life_mask,
    rolling_half_life,
)
from waddle.features.spread import log_spread, rolling_zscore
from waddle.strategy.pairs_reversion import metrics_dict, run_pairs_reversion

ROLLING_WINDOW = 24 * 7

TRAIN_DAYS = 90
TEST_DAYS = 30
STEP_DAYS = 15
BARS_PER_DAY = 24

# Half-life rolling window: 30 days of hourly bars. Deliberately much shorter
# than the ADF 90-day window — we want the signal at a scale compatible with the
# 48h holding period. Rebuild once per day.
HALF_LIFE_WINDOW_BARS = 30 * 24
HALF_LIFE_STEP_BARS = 24

BEAR_THRESHOLD = 60000.0

# None = baseline reference (no half-life filter, bear halt still ON — this is
# the ADR-008 config). The rest sweep thresholds in days. 10 is very strict
# (only trade when the spread literally reverts within ~1.5 weeks on average),
# 90 is very lax (almost anything passes). If the curve has a "knee" somewhere,
# we'll find it here.
THRESHOLDS: list[float | None] = [None, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0]

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


def precompute_halflife_and_bear(data: dict) -> dict:
    aligned = data["aligned"]
    spread = data["spread"]
    print(
        f"Precomputing rolling AR(1) half-life "
        f"(window={HALF_LIFE_WINDOW_BARS}b={HALF_LIFE_WINDOW_BARS // 24}d, "
        f"step={HALF_LIFE_STEP_BARS}b)..."
    )
    hl = rolling_half_life(
        spread,
        window_bars=HALF_LIFE_WINDOW_BARS,
        step_bars=HALF_LIFE_STEP_BARS,
    )
    bear = bear_halt_mask(data["btc_close"], aligned.index, threshold=BEAR_THRESHOLD)

    valid = hl.replace([float("inf"), float("-inf")], float("nan")).dropna()
    inf_count = int((hl == float("inf")).sum())
    if len(valid) > 0:
        print(
            f"  half-life range (days, finite only): "
            f"min={valid.min():.2f} median={valid.median():.2f} "
            f"mean={valid.mean():.2f} max={valid.max():.2f}"
        )
        print(
            f"  quantiles: p25={valid.quantile(0.25):.2f} "
            f"p50={valid.quantile(0.50):.2f} "
            f"p75={valid.quantile(0.75):.2f} "
            f"p90={valid.quantile(0.90):.2f}"
        )
    else:
        print("  half-life: no finite values (!)")
    print(f"  +inf values (explosive / no reversion): {inf_count} bars")
    print(f"  NaN bars (warmup): {int(hl.isna().sum())} bars")
    print(
        f"  bear halt safe (BTC > {BEAR_THRESHOLD:.0f}): "
        f"{bear.sum()}/{len(bear)} ({100 * bear.mean():.1f}% of bars)"
    )
    return {"half_life": hl, "bear_halt_mask": bear}


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
    hl_mask_full: pd.Series | None,
    bear_mask_full: pd.Series,
) -> dict:
    """Run the walk-forward loop for one (threshold) scenario.

    hl_mask_full = None means "no half-life filter" — baseline reference.
    bear_mask_full is always provided and always applied.
    """
    spread = data["spread"]
    zscore = data["zscore"]

    window_results = []
    for win_i, (tr_start, tr_end, te_end) in enumerate(windows):
        spread_tr = spread.iloc[tr_start:tr_end]
        zscore_tr = zscore.iloc[tr_start:tr_end]
        spread_te = spread.iloc[tr_end:te_end]
        zscore_te = zscore.iloc[tr_end:te_end]

        hl_tr = hl_mask_full.iloc[tr_start:tr_end] if hl_mask_full is not None else None
        hl_te = hl_mask_full.iloc[tr_end:te_end] if hl_mask_full is not None else None
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
                cointegration_mask=hl_tr,
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
            cointegration_mask=hl_te,
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
    print("=" * 130)
    print("SOL/AVAX walk-forward — HALF-LIFE threshold sweep (bear halt ON everywhere)")
    print("=" * 130)
    header = (
        f"{'threshold':<14} "
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
            print(f"{label:<14} {mask_true_pct * 100:>7.1f}% (no active windows)")
            continue
        print(
            f"{label:<14} "
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
    print("  threshold : half-life cap in days (None = filter off, bear halt only)")
    print("  mask_T%   : fraction of bars where the half-life filter lets trades through")
    print("              (at None: 100% — no filter; lower threshold = stricter)")
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
    print("=" * 130)
    print("WADDLE — SOL/AVAX walk-forward HALF-LIFE threshold sweep (Investigation F)")
    print("=" * 130)
    print(f"Pair: {PAIR_A} vs {PAIR_B}")
    print(f"Windows: train={TRAIN_DAYS}d / test={TEST_DAYS}d / step={STEP_DAYS}d")
    print(f"Fees: {FEE_PER_SIDE * 100:.2f}%/side fee + {SLIPPAGE_PER_SIDE * 100:.2f}%/side slip")
    print(f"Grid: {len(PARAM_GRID)} combos")
    print(
        f"Half-life lookback: {HALF_LIFE_WINDOW_BARS // 24}d "
        f"(vs ADF test which used 90d — we deliberately match the strategy horizon)"
    )
    print(f"Thresholds tested (days): {THRESHOLDS}")
    print(f"Bear halt (BTC > {BEAR_THRESHOLD:.0f}): ALWAYS ON in every scenario")

    data = load_data()
    print(
        f"\nData: {len(data['aligned'])} bars loaded "
        f"({data['aligned'].index[0].date()} -> {data['aligned'].index[-1].date()})"
    )

    cached = precompute_halflife_and_bear(data)
    hl_series = cached["half_life"]
    bear_full = cached["bear_halt_mask"]

    windows = make_windows(len(data["aligned"]))
    print(f"\nRolling windows generated: {len(windows)}")

    results: list[tuple[str, float, dict]] = []
    for threshold in THRESHOLDS:
        if threshold is None:
            label = "None"
            hl_full = None
            mask_true_pct = 1.0
        else:
            label = f"{threshold:.0f}d"
            hl_full = half_life_mask(hl_series, threshold_days=threshold)
            mask_true_pct = float(hl_full.mean())

        print(
            f"Running scenario: threshold={label}  "
            f"(half-life mask lets through {mask_true_pct * 100:.1f}% of bars)"
        )
        agg = run_scenario(data, windows, hl_full, bear_full)
        results.append((label, mask_true_pct, agg))

    print_comparison(results)


if __name__ == "__main__":
    main()
