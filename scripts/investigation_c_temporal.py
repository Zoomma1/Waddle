"""Investigation C — Temporal stability of the SOL/AVAX edge.

Question: is the +4.43 mean test Sharpe observed on the 18-month walk-forward
sweep distributed across the whole period, or concentrated on a specific
regime / lucky sub-period?

Five analyses:
  1. Cumulative out-of-sample PnL — does the equity curve grow smoothly or
     in a couple of big jumps?
  2. H1 vs H2 split — re-run the sweep separately on first 9 months and
     second 9 months.
  3. Rolling log-return correlation SOL vs AVAX (30d and 90d windows).
  4. Outlier windows (top-5 / bottom-5 by test Sharpe) and their period.
  5. Regime analysis by BTC price bucket.

Pass criterion C: mean test Sharpe > +1.0 on H1 AND on H2 independently.
"""
from __future__ import annotations

from collections import Counter
from statistics import mean, stdev

import pandas as pd

from waddle.data.storage import load_ohlcv
from waddle.features.correlation import rolling_log_return_correlation
from waddle.features.spread import log_spread, rolling_zscore
from waddle.strategy.pairs_reversion import (
    metrics_dict,
    run_pairs_reversion,
    Trade,
)

# ---- config identical to walk_forward_sweep.py -----------------------------
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

FEE_PER_SIDE = 0.0002
SLIPPAGE_PER_SIDE = 0.0002

SYMBOL_A = "SOL/USDT"
SYMBOL_B = "AVAX/USDT"
BTC_SYMBOL = "BTC/USDT"


# ---- helpers (adapted from walk_forward_sweep) -----------------------------
def _combo_label(combo: dict) -> str:
    return f"entry={combo['entry_z']}/exit={combo['exit_z']}"


def _calmar(m: dict) -> float:
    if m["n_trades"] < 3:
        return -float("inf")
    if m["max_dd"] == 0:
        return float("inf") if m["total_pnl"] > 0 else 0.0
    return m["total_pnl"] / abs(m["max_dd"])


def _load_aligned_pair(symbol_a: str, symbol_b: str) -> pd.DataFrame:
    a = load_ohlcv(symbol_a)
    b = load_ohlcv(symbol_b)
    return a.join(b, how="inner", lsuffix="_a", rsuffix="_b")


def _build_zscore(aligned: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    spread = log_spread(aligned["close_a"], aligned["close_b"])
    zscore = rolling_zscore(spread, window=ROLLING_WINDOW)
    return spread, zscore


def _iter_windows(n: int) -> list[tuple[int, int, int]]:
    train_bars = TRAIN_DAYS * BARS_PER_DAY
    test_bars = TEST_DAYS * BARS_PER_DAY
    step_bars = STEP_DAYS * BARS_PER_DAY
    windows = []
    idx = 0
    while idx + train_bars + test_bars <= n:
        windows.append((idx, idx + train_bars, idx + train_bars + test_bars))
        idx += step_bars
    return windows


def _run_single_window(
    spread: pd.Series,
    zscore: pd.Series,
    tr_start: int,
    tr_end: int,
    te_end: int,
) -> dict:
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
        return {}

    test_trades = run_pairs_reversion(
        spread_te, zscore_te, **best_combo,
        fee_per_side=FEE_PER_SIDE,
        slippage_per_side=SLIPPAGE_PER_SIDE,
    )
    test_m = metrics_dict(test_trades, period_days=test_days_real)

    return {
        "train_start": spread_tr.index[0],
        "train_end": spread_tr.index[-1],
        "test_start": spread_te.index[0],
        "test_end": spread_te.index[-1],
        "best_combo": best_combo,
        "train_m": best_train_m,
        "test_m": test_m,
        "test_trades": test_trades,
    }


def _walk_forward_on_slice(
    spread: pd.Series,
    zscore: pd.Series,
    label: str,
) -> list[dict]:
    """Run sweep on an already-sliced (spread, zscore). Returns per-window results."""
    n = len(spread)
    windows = _iter_windows(n)
    results = []
    for win_i, (tr_start, tr_end, te_end) in enumerate(windows):
        r = _run_single_window(spread, zscore, tr_start, tr_end, te_end)
        if not r:
            continue
        r["window_i"] = win_i
        r["slice"] = label
        results.append(r)
    return results


def _aggregate(results: list[dict]) -> dict:
    if not results:
        return {
            "n_results": 0,
            "mean_train_sharpe": 0.0,
            "mean_test_sharpe": 0.0,
            "std_test_sharpe": 0.0,
            "n_positive": 0,
            "combo_counts": Counter(),
            "mean_test_pnl": 0.0,
            "mean_test_n": 0.0,
        }
    test_sh = [r["test_m"]["sharpe"] for r in results]
    train_sh = [r["train_m"]["sharpe"] for r in results]
    test_pnls = [r["test_m"]["total_pnl"] for r in results]
    test_ns = [r["test_m"]["n_trades"] for r in results]
    counts = Counter(_combo_label(r["best_combo"]) for r in results)
    return {
        "n_results": len(results),
        "mean_train_sharpe": mean(train_sh),
        "mean_test_sharpe": mean(test_sh),
        "std_test_sharpe": stdev(test_sh) if len(test_sh) > 1 else 0.0,
        "n_positive": sum(1 for s in test_sh if s > 0),
        "combo_counts": counts,
        "mean_test_pnl": mean(test_pnls),
        "mean_test_n": mean(test_ns),
    }


# ---- analysis 1: cumulative OOS PnL -----------------------------------------
def _cumulative_oos_equity(results: list[dict]) -> pd.Series:
    """Concatenate all test trades chronologically and return cumulative PnL."""
    all_trades: list[Trade] = []
    for r in results:
        all_trades.extend(r["test_trades"])
    if not all_trades:
        return pd.Series(dtype=float)
    data = [(t.exit_ts, t.pnl) for t in all_trades]
    s = pd.Series({ts: pnl for ts, pnl in data}).sort_index()
    # if multiple trades exit at same ts, sum them — unlikely at 1h granularity
    s = s.groupby(s.index).sum()
    return s.cumsum()


def _find_biggest_jumps(equity: pd.Series, n: int = 3) -> list[tuple[pd.Timestamp, float]]:
    """Return the n largest one-step upward jumps in cumulative equity."""
    if equity.empty:
        return []
    steps = equity.diff().dropna()
    top = steps.sort_values(ascending=False).head(n)
    return [(ts, float(v)) for ts, v in top.items()]


# ---- analysis 3: rolling correlation ----------------------------------------
def _corr_summary(corr: pd.Series) -> dict:
    clean = corr.dropna()
    if clean.empty:
        return {}
    return {
        "mean": float(clean.mean()),
        "min": float(clean.min()),
        "p10": float(clean.quantile(0.10)),
        "p90": float(clean.quantile(0.90)),
        "max": float(clean.max()),
        "min_ts": clean.idxmin(),
        "max_ts": clean.idxmax(),
    }


def _monthly_samples(series: pd.Series) -> pd.Series:
    """Take the last observation each month."""
    clean = series.dropna()
    return clean.resample("1MS").last().dropna()


# ---- analysis 5: regime bucket ----------------------------------------------
def _btc_regime_at(ts: pd.Timestamp, btc_close: pd.Series) -> str:
    # look up closest earlier or equal timestamp
    snap = btc_close.loc[:ts]
    if snap.empty:
        return "unknown"
    price = float(snap.iloc[-1])
    if price > 75000:
        return "bull"
    if price < 60000:
        return "bear"
    return "chop"


# ---- reporting --------------------------------------------------------------
def _print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _print_subheader(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> None:
    _print_header("INVESTIGATION C — Temporal stability of SOL/AVAX edge")
    print(f"Pair: {SYMBOL_A} / {SYMBOL_B}")
    print(f"Window config: train={TRAIN_DAYS}d / test={TEST_DAYS}d / step={STEP_DAYS}d")
    print(f"Grid size: {len(PARAM_GRID)} combos")
    print(f"Fees: {FEE_PER_SIDE * 100:.2f}%/side fee + {SLIPPAGE_PER_SIDE * 100:.2f}%/side slip")

    # ---- data loading ------------------------------------------------------
    aligned = _load_aligned_pair(SYMBOL_A, SYMBOL_B)
    btc = load_ohlcv(BTC_SYMBOL)
    btc_close = btc["close"]

    total_days = (aligned.index[-1] - aligned.index[0]).total_seconds() / 86400
    print(f"Data: {len(aligned)} aligned bars | {aligned.index[0].date()} -> {aligned.index[-1].date()} | {total_days:.0f} days")

    spread, zscore = _build_zscore(aligned)

    # ============================================================
    # 1 — full sweep on 18 months + cumulative OOS PnL
    # ============================================================
    _print_header("1 — CUMULATIVE OOS PnL (full 18 months)")
    full_results = _walk_forward_on_slice(spread, zscore, "FULL")
    full_agg = _aggregate(full_results)
    print(
        f"Windows generated: {full_agg['n_results']} | "
        f"mean train Sharpe: {full_agg['mean_train_sharpe']:+.2f} | "
        f"mean test Sharpe: {full_agg['mean_test_sharpe']:+.2f} "
        f"(std {full_agg['std_test_sharpe']:.2f}) | "
        f"positive: {full_agg['n_positive']}/{full_agg['n_results']}"
    )

    equity = _cumulative_oos_equity(full_results)
    if equity.empty:
        print("(no trades — cannot build equity curve)")
    else:
        _print_subheader("Cumulative OOS PnL at monthly sampling points")
        monthly = _monthly_samples(equity)
        for ts, v in monthly.items():
            print(f"  {ts.date()}   {v * 100:+8.2f}%")

        _print_subheader("Biggest one-trade jumps in cumulative PnL")
        jumps = _find_biggest_jumps(equity, n=5)
        for ts, v in jumps:
            print(f"  {ts.strftime('%Y-%m-%d %H:%M'):<17}   {v * 100:+7.2f}%")

        total_gain = float(equity.iloc[-1])
        top_5_sum = sum(v for _, v in jumps)
        share = (top_5_sum / total_gain) if total_gain > 0 else 0.0
        print(
            f"\n  Total OOS gain: {total_gain * 100:+.2f}% | "
            f"Top-5 trades contribute: {top_5_sum * 100:+.2f}% ({share * 100:.0f}% of total)"
        )

    # ============================================================
    # 2 — H1 / H2 split
    # ============================================================
    _print_header("2 — H1 / H2 TEMPORAL SPLIT")
    midpoint = aligned.index[0] + (aligned.index[-1] - aligned.index[0]) / 2
    print(f"Midpoint: {midpoint.date()}")

    h1_mask = aligned.index <= midpoint
    h2_mask = aligned.index > midpoint
    h1_spread = spread[h1_mask]
    h1_zscore = zscore[h1_mask]
    h2_spread = spread[h2_mask]
    h2_zscore = zscore[h2_mask]
    print(
        f"H1: {h1_spread.index[0].date()} -> {h1_spread.index[-1].date()} | {len(h1_spread)} bars"
    )
    print(
        f"H2: {h2_spread.index[0].date()} -> {h2_spread.index[-1].date()} | {len(h2_spread)} bars"
    )

    h1_results = _walk_forward_on_slice(h1_spread, h1_zscore, "H1")
    h2_results = _walk_forward_on_slice(h2_spread, h2_zscore, "H2")
    h1_agg = _aggregate(h1_results)
    h2_agg = _aggregate(h2_results)

    _print_subheader("H1 vs H2 aggregates")
    row = "  {:<4} {:>7} {:>11} {:>12} {:>10} {:>10} {:>10}"
    print(row.format("slice", "windows", "mean_train", "mean_test", "std_test", "pos_pct", "avg_n"))
    for label, a in (("H1", h1_agg), ("H2", h2_agg)):
        if a["n_results"] == 0:
            print(f"  {label:<4} (no windows)")
            continue
        pos_pct = a["n_positive"] / a["n_results"] * 100
        print(
            row.format(
                label,
                a["n_results"],
                f"{a['mean_train_sharpe']:+.2f}",
                f"{a['mean_test_sharpe']:+.2f}",
                f"{a['std_test_sharpe']:.2f}",
                f"{pos_pct:.0f}%",
                f"{a['mean_test_n']:.1f}",
            )
        )

    _print_subheader("Top-3 combos per half")
    for label, a in (("H1", h1_agg), ("H2", h2_agg)):
        print(f"  {label}:")
        if not a["combo_counts"]:
            print("    (no combos)")
            continue
        for combo, n in a["combo_counts"].most_common(3):
            pct = n / a["n_results"] * 100
            print(f"    {combo:<22} {n}/{a['n_results']} ({pct:.0f}%)")

    # ============================================================
    # 3 — rolling log-return correlation SOL vs AVAX
    # ============================================================
    _print_header("3 — ROLLING LOG-RETURN CORRELATION (SOL vs AVAX)")
    corr_30d = rolling_log_return_correlation(
        aligned["close_a"], aligned["close_b"], window=30 * BARS_PER_DAY
    )
    corr_90d = rolling_log_return_correlation(
        aligned["close_a"], aligned["close_b"], window=90 * BARS_PER_DAY
    )

    for label, corr in (("30d (720 bars)", corr_30d), ("90d (2160 bars)", corr_90d)):
        _print_subheader(label)
        summary = _corr_summary(corr)
        if not summary:
            print("  (no data)")
            continue
        print(f"  mean={summary['mean']:.3f}  min={summary['min']:.3f}  p10={summary['p10']:.3f}  p90={summary['p90']:.3f}  max={summary['max']:.3f}")
        print(f"  min at {summary['min_ts'].date()}  |  max at {summary['max_ts'].date()}")

        print("  Monthly samples:")
        monthly = _monthly_samples(corr)
        for ts, v in monthly.items():
            bar_len = int((v - 0.3) * 30) if v > 0.3 else 0
            bar = "#" * max(0, min(bar_len, 30))
            print(f"    {ts.date()}   {v:+.3f}  {bar}")

    # ============================================================
    # 4 — Outlier windows
    # ============================================================
    _print_header("4 — OUTLIER WINDOWS")
    by_sharpe = sorted(full_results, key=lambda r: r["test_m"]["sharpe"], reverse=True)

    header = "  {:>3} {:<12} {:<22} {:>10} {:>10} {:>8}"
    line = "  {:>3} {:<12} {:<22} {:>+10.2f} {:>+9.2f}% {:>8}"

    _print_subheader("TOP 5 test Sharpe")
    print(header.format("win", "test_start", "best_combo", "te_sharpe", "te_pnl", "n_trades"))
    for r in by_sharpe[:5]:
        print(line.format(
            r["window_i"],
            r["test_start"].strftime("%Y-%m-%d"),
            _combo_label(r["best_combo"]),
            r["test_m"]["sharpe"],
            r["test_m"]["total_pnl"] * 100,
            r["test_m"]["n_trades"],
        ))

    _print_subheader("BOTTOM 5 test Sharpe")
    print(header.format("win", "test_start", "best_combo", "te_sharpe", "te_pnl", "n_trades"))
    for r in by_sharpe[-5:]:
        print(line.format(
            r["window_i"],
            r["test_start"].strftime("%Y-%m-%d"),
            _combo_label(r["best_combo"]),
            r["test_m"]["sharpe"],
            r["test_m"]["total_pnl"] * 100,
            r["test_m"]["n_trades"],
        ))

    # ============================================================
    # 5 — Regime analysis by BTC price bucket
    # ============================================================
    _print_header("5 — REGIME ANALYSIS (BTC price at window train start)")
    regime_buckets: dict[str, list[dict]] = {"bull": [], "chop": [], "bear": []}
    for r in full_results:
        reg = _btc_regime_at(r["train_start"], btc_close)
        if reg in regime_buckets:
            regime_buckets[reg].append(r)

    row = "  {:<6} {:>8} {:>12} {:>12} {:>10} {:>10}"
    print(row.format("regime", "windows", "mean_test", "std_test", "pos_pct", "avg_n"))
    for regime in ("bull", "chop", "bear"):
        bucket = regime_buckets[regime]
        if not bucket:
            print(f"  {regime:<6} (no windows)")
            continue
        test_sh = [r["test_m"]["sharpe"] for r in bucket]
        test_ns = [r["test_m"]["n_trades"] for r in bucket]
        pos_pct = sum(1 for s in test_sh if s > 0) / len(test_sh) * 100
        std_sh = stdev(test_sh) if len(test_sh) > 1 else 0.0
        print(
            row.format(
                regime,
                len(bucket),
                f"{mean(test_sh):+.2f}",
                f"{std_sh:.2f}",
                f"{pos_pct:.0f}%",
                f"{mean(test_ns):.1f}",
            )
        )

    # ============================================================
    # Verdict
    # ============================================================
    _print_header("VERDICT — Criterion C")
    h1_pass = h1_agg["mean_test_sharpe"] > 1.0
    h2_pass = h2_agg["mean_test_sharpe"] > 1.0
    print(f"  Criterion: mean test Sharpe > +1.0 on H1 AND on H2 independently")
    print(f"  H1 mean test Sharpe: {h1_agg['mean_test_sharpe']:+.2f}  ->  {'PASS' if h1_pass else 'FAIL'}")
    print(f"  H2 mean test Sharpe: {h2_agg['mean_test_sharpe']:+.2f}  ->  {'PASS' if h2_pass else 'FAIL'}")
    if h1_pass and h2_pass:
        print("  OVERALL: PASS — edge appears distributed across both halves.")
    else:
        print("  OVERALL: FAIL — edge is not stable across the full period.")


if __name__ == "__main__":
    main()
