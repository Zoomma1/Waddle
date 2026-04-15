"""Rolling walk-forward parameter sweep — multi-pair edition.

The first rolling walk-forward (2026-04-15) on BTC/ETH over 6 months showed
no robust edge (ADR-006). Two hypotheses to eliminate before changing the
model itself:

1. The dataset was too short — 6 months = 4-6 trades per 30d test window,
   pure noise on that sample size.
2. BTC/ETH is the most arbitraged crypto pair on earth; mean-reversion
   inefficiencies may exist on less saturated pairs.

This script runs the same walk-forward sweep on 18 months of data across
multiple pairs and reports a cross-pair summary at the end.

Output has three levels:
- PER-WINDOW table per pair (what combo won each window, train/test metrics)
- PER-PAIR aggregates (stability, mean/std test metrics, generalization ratio)
- CROSS-PAIR summary (rank pairs by robustness to spot candidates)

Rule of thumb for a "candidate" pair:
- Generalization ratio (test/train Sharpe) >= 0.6
- Mean test Sharpe > 0 after fees
- Stability of best-train combo >= 40% (no combo dominates completely but one is clearly preferred)
"""
from __future__ import annotations

from collections import Counter
from statistics import mean, stdev

import pandas as pd

from waddle.data.storage import load_ohlcv
from waddle.features.spread import log_spread, rolling_zscore
from waddle.strategy.pairs_reversion import metrics_dict, run_pairs_reversion

ROLLING_WINDOW = 24 * 7  # 168h

TRAIN_DAYS = 90
TEST_DAYS = 30
STEP_DAYS = 15
BARS_PER_DAY = 24

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

PAIRS_TO_TEST = [
    ("BTC/USDT", "ETH/USDT"),
    ("SOL/USDT", "ETH/USDT"),
    ("SOL/USDT", "AVAX/USDT"),
    ("BNB/USDT", "ETH/USDT"),
]

FEE_PER_SIDE = 0.0002
SLIPPAGE_PER_SIDE = 0.0002


def _combo_label(combo: dict) -> str:
    return f"entry={combo['entry_z']}/exit={combo['exit_z']}"


def _pair_label(a: str, b: str) -> str:
    return f"{a.split('/')[0]}/{b.split('/')[0]}"


def _calmar(m: dict) -> float:
    if m["n_trades"] < 3:
        return -float("inf")
    if m["max_dd"] == 0:
        return float("inf") if m["total_pnl"] > 0 else 0.0
    return m["total_pnl"] / abs(m["max_dd"])


def load_aligned_pair(symbol_a: str, symbol_b: str) -> pd.DataFrame:
    a = load_ohlcv(symbol_a)
    b = load_ohlcv(symbol_b)
    return a.join(b, how="inner", lsuffix="_a", rsuffix="_b")


def run_walk_forward_sweep(symbol_a: str, symbol_b: str) -> dict:
    aligned = load_aligned_pair(symbol_a, symbol_b)
    total_days = (aligned.index[-1] - aligned.index[0]).total_seconds() / 86400

    spread = log_spread(aligned["close_a"], aligned["close_b"])
    zscore = rolling_zscore(spread, window=ROLLING_WINDOW)

    train_bars = TRAIN_DAYS * BARS_PER_DAY
    test_bars = TEST_DAYS * BARS_PER_DAY
    step_bars = STEP_DAYS * BARS_PER_DAY

    windows: list[tuple[int, int, int]] = []
    idx = 0
    while idx + train_bars + test_bars <= len(aligned):
        windows.append((idx, idx + train_bars, idx + train_bars + test_bars))
        idx += step_bars

    results = []
    for win_i, (tr_start, tr_end, te_end) in enumerate(windows):
        spread_tr = spread.iloc[tr_start:tr_end]
        zscore_tr = zscore.iloc[tr_start:tr_end]
        spread_te = spread.iloc[tr_end:te_end]
        zscore_te = zscore.iloc[tr_end:te_end]

        train_days_real = (spread_tr.index[-1] - spread_tr.index[0]).total_seconds() / 86400
        test_days_real = (spread_te.index[-1] - spread_te.index[0]).total_seconds() / 86400

        best_combo = None
        best_train_m = None
        best_calmar = -float("inf")
        for combo in PARAM_GRID:
            trades = run_pairs_reversion(
                spread_tr, zscore_tr, **combo,
                fee_per_side=FEE_PER_SIDE,
                slippage_per_side=SLIPPAGE_PER_SIDE,
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
            spread_te, zscore_te, **best_combo,
            fee_per_side=FEE_PER_SIDE,
            slippage_per_side=SLIPPAGE_PER_SIDE,
        )
        test_m = metrics_dict(test_trades, period_days=test_days_real)

        results.append({
            "window_i": win_i,
            "train_start": spread_tr.index[0],
            "train_end": spread_tr.index[-1],
            "test_start": spread_te.index[0],
            "test_end": spread_te.index[-1],
            "best_combo": best_combo,
            "train_m": best_train_m,
            "test_m": test_m,
        })

    return {
        "pair_a": symbol_a,
        "pair_b": symbol_b,
        "total_days": total_days,
        "n_windows": len(windows),
        "results": results,
    }


def _print_per_window(pair_result: dict) -> None:
    results = pair_result["results"]
    if not results:
        print("  (no valid windows)")
        return
    header = (
        f"{'win':>3} {'train_start':<12} {'best_combo':<20} "
        f"{'tr_shrp':>8} {'te_shrp':>8} {'te/tr':>7} "
        f"{'tr_pnl':>9} {'te_pnl':>9} {'te_dd':>9} {'te_n':>5}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        tr = r["train_m"]
        te = r["test_m"]
        ratio = te["sharpe"] / tr["sharpe"] if tr["sharpe"] != 0 else 0.0
        print(
            f"{r['window_i']:>3} "
            f"{r['train_start'].strftime('%Y-%m-%d'):<12} "
            f"{_combo_label(r['best_combo']):<20} "
            f"{tr['sharpe']:>+8.2f} "
            f"{te['sharpe']:>+8.2f} "
            f"{ratio:>6.2f}x "
            f"{tr['total_pnl'] * 100:>+8.2f}% "
            f"{te['total_pnl'] * 100:>+8.2f}% "
            f"{te['max_dd'] * 100:>+8.2f}% "
            f"{te['n_trades']:>5}"
        )


def _aggregate_pair(pair_result: dict) -> dict:
    results = pair_result["results"]
    if not results:
        return {
            "n_results": 0,
            "top_combo": None,
            "stability": 0.0,
            "mean_train_sharpe": 0.0,
            "mean_test_sharpe": 0.0,
            "std_test_sharpe": 0.0,
            "mean_test_pnl": 0.0,
            "mean_test_dd": 0.0,
            "mean_test_n": 0.0,
            "gen_ratio": 0.0,
            "n_positive_test_windows": 0,
        }

    counts = Counter(_combo_label(r["best_combo"]) for r in results)
    top_label, top_n = counts.most_common(1)[0]
    stability = top_n / len(results)

    test_sharpes = [r["test_m"]["sharpe"] for r in results]
    train_sharpes = [r["train_m"]["sharpe"] for r in results]
    test_pnls = [r["test_m"]["total_pnl"] for r in results]
    test_dds = [r["test_m"]["max_dd"] for r in results]
    test_ns = [r["test_m"]["n_trades"] for r in results]

    mean_train_sharpe = mean(train_sharpes)
    mean_test_sharpe = mean(test_sharpes)
    std_test_sharpe = stdev(test_sharpes) if len(test_sharpes) > 1 else 0.0
    gen_ratio = mean_test_sharpe / mean_train_sharpe if mean_train_sharpe != 0 else 0.0
    n_positive = sum(1 for s in test_sharpes if s > 0)

    return {
        "n_results": len(results),
        "top_combo": top_label,
        "top_combo_count": top_n,
        "stability": stability,
        "mean_train_sharpe": mean_train_sharpe,
        "mean_test_sharpe": mean_test_sharpe,
        "std_test_sharpe": std_test_sharpe,
        "mean_test_pnl": mean(test_pnls),
        "mean_test_dd": mean(test_dds),
        "mean_test_n": mean(test_ns),
        "gen_ratio": gen_ratio,
        "n_positive_test_windows": n_positive,
        "combo_counts": counts,
    }


def _print_pair_aggregate(pair_result: dict, agg: dict) -> None:
    pair = _pair_label(pair_result["pair_a"], pair_result["pair_b"])

    print(f"\n  [STABILITY — best-on-train combo frequency]")
    if agg["n_results"] == 0:
        print("    (no valid results)")
    else:
        for label, n in agg["combo_counts"].most_common():
            pct = n / agg["n_results"] * 100
            bar = "#" * int(pct / 5)
            print(f"    {label:<22} {n:>3}/{agg['n_results']:<3} ({pct:>5.1f}%) {bar}")

    print(f"\n  [AGGREGATES]")
    print(f"    Mean train Sharpe:        {agg['mean_train_sharpe']:+.2f}")
    print(f"    Mean test  Sharpe:        {agg['mean_test_sharpe']:+.2f}  (std {agg['std_test_sharpe']:.2f})")
    print(f"    Generalization ratio:     {agg['gen_ratio']:.2f}x")
    print(f"    Mean test PnL:            {agg['mean_test_pnl'] * 100:+.2f}%")
    print(f"    Mean test max DD:         {agg['mean_test_dd'] * 100:+.2f}%")
    print(f"    Mean test n_trades:       {agg['mean_test_n']:.1f}")
    print(f"    Positive test windows:    {agg['n_positive_test_windows']}/{agg['n_results']}")


def _print_cross_pair_summary(all_aggs: list[tuple[str, dict]]) -> None:
    print("\n" + "#" * 108)
    print("CROSS-PAIR SUMMARY — ranked by mean test Sharpe")
    print("#" * 108)
    header = (
        f"{'pair':<14} {'windows':>8} {'top_combo':<20} {'stability':>10} "
        f"{'tr_shrp':>9} {'te_shrp':>9} {'gen':>7} {'te_pnl':>9} {'pos':>8}"
    )
    print(header)
    print("-" * len(header))

    sorted_aggs = sorted(all_aggs, key=lambda t: t[1]["mean_test_sharpe"], reverse=True)
    for pair, agg in sorted_aggs:
        if agg["n_results"] == 0:
            print(f"{pair:<14} (no results)")
            continue
        pos_str = f"{agg['n_positive_test_windows']}/{agg['n_results']}"
        print(
            f"{pair:<14} {agg['n_results']:>8} "
            f"{agg['top_combo']:<20} "
            f"{agg['stability'] * 100:>8.0f}% "
            f"{agg['mean_train_sharpe']:>+9.2f} "
            f"{agg['mean_test_sharpe']:>+9.2f} "
            f"{agg['gen_ratio']:>6.2f}x "
            f"{agg['mean_test_pnl'] * 100:>+8.2f}% "
            f"{pos_str:>8}"
        )

    print("\n[CANDIDATE CRITERIA] — a pair is a candidate if it hits all three:")
    print("  - gen_ratio >= 0.60  (test Sharpe doesn't collapse vs train)")
    print("  - mean test Sharpe > 0  (makes money after fees on average)")
    print("  - stability >= 40%  (one combo clearly preferred)")
    print()

    candidates = [
        (pair, agg)
        for pair, agg in sorted_aggs
        if agg["n_results"] > 0
        and agg["gen_ratio"] >= 0.60
        and agg["mean_test_sharpe"] > 0
        and agg["stability"] >= 0.40
    ]
    if candidates:
        print("[CANDIDATES FOUND]")
        for pair, agg in candidates:
            print(
                f"  > {pair:<14} combo={agg['top_combo']} "
                f"gen={agg['gen_ratio']:.2f}x  "
                f"te_shrp={agg['mean_test_sharpe']:+.2f}  "
                f"stability={agg['stability'] * 100:.0f}%"
            )
    else:
        print("[NO CANDIDATES] — no pair passes all three criteria.")
        print("  Next step: filtre de régime, cointégration formelle, ou pivot stratégique.")


def main() -> None:
    print("=" * 108)
    print("WADDLE — Multi-pair walk-forward sweep")
    print("=" * 108)
    print(f"Windows: train={TRAIN_DAYS}d / test={TEST_DAYS}d / step={STEP_DAYS}d")
    print(f"Fees: {FEE_PER_SIDE * 100:.2f}%/side fee + {SLIPPAGE_PER_SIDE * 100:.2f}%/side slip")
    print(f"Grid: {len(PARAM_GRID)} combos")
    print(f"Pairs: {', '.join(_pair_label(a, b) for a, b in PAIRS_TO_TEST)}")

    all_aggs: list[tuple[str, dict]] = []
    for symbol_a, symbol_b in PAIRS_TO_TEST:
        pair = _pair_label(symbol_a, symbol_b)
        print("\n" + "=" * 108)
        print(f"PAIR: {pair}  [{symbol_a} vs {symbol_b}]")
        print("=" * 108)

        pair_result = run_walk_forward_sweep(symbol_a, symbol_b)
        print(
            f"  data: {pair_result['total_days']:.1f} days, "
            f"{pair_result['n_windows']} rolling windows generated"
        )

        _print_per_window(pair_result)
        agg = _aggregate_pair(pair_result)
        _print_pair_aggregate(pair_result, agg)
        all_aggs.append((pair, agg))

    _print_cross_pair_summary(all_aggs)


if __name__ == "__main__":
    main()
