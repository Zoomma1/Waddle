"""Investigation A — SOL/AVAX variance & robustness.

Re-runs the walk-forward sweep ONLY on SOL/AVAX and stress-tests the +4.43
mean test Sharpe headline number four ways:

    1. Outlier analysis      — median, trimmed mean, top/bottom windows
    2. Minimum trades filter — drop windows with < 4/5/6 trades
    3. Fee stress test       — re-run sweep at 1x / 2x / 4x baseline fees
    4. Restricted combo set  — keep only combos picked >= 3 times across windows

Window geometry, param grid, Calmar selection, and fee baseline match
`walk_forward_sweep.py` exactly. No cointegration, no regime analysis, no
temporal split — those belong to the sibling investigations B and C.
"""
from __future__ import annotations

from collections import Counter
from statistics import mean, median, stdev

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

PAIR_A = "SOL/USDT"
PAIR_B = "AVAX/USDT"

# Bybit futures maker baseline, used as fee 1x
BASE_FEE = 0.0002
BASE_SLIP = 0.0002

FEE_SCENARIOS = [
    ("1x (Bybit futures maker)", 1),
    ("2x (stress)", 2),
    ("4x (retail cold wallet)", 4),
]


def _combo_label(combo: dict) -> str:
    return f"entry={combo['entry_z']}/exit={combo['exit_z']}"


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


def run_walk_forward(
    spread: pd.Series,
    zscore: pd.Series,
    aligned: pd.DataFrame,
    fee_per_side: float,
    slip_per_side: float,
) -> list[dict]:
    """Run a full SOL/AVAX walk-forward sweep at a given fee level.

    Returns the same per-window results as `walk_forward_sweep.py`:
    train metrics, test metrics, and the Calmar-best combo picked on train.
    """
    train_bars = TRAIN_DAYS * BARS_PER_DAY
    test_bars = TEST_DAYS * BARS_PER_DAY
    step_bars = STEP_DAYS * BARS_PER_DAY

    windows: list[tuple[int, int, int]] = []
    idx = 0
    while idx + train_bars + test_bars <= len(aligned):
        windows.append((idx, idx + train_bars, idx + train_bars + test_bars))
        idx += step_bars

    results: list[dict] = []
    for win_i, (tr_start, tr_end, te_end) in enumerate(windows):
        spread_tr = spread.iloc[tr_start:tr_end]
        zscore_tr = zscore.iloc[tr_start:tr_end]
        spread_te = spread.iloc[tr_end:te_end]
        zscore_te = zscore.iloc[tr_end:te_end]

        train_days_real = (
            spread_tr.index[-1] - spread_tr.index[0]
        ).total_seconds() / 86400
        test_days_real = (
            spread_te.index[-1] - spread_te.index[0]
        ).total_seconds() / 86400

        best_combo = None
        best_train_m = None
        best_calmar = -float("inf")
        for combo in PARAM_GRID:
            trades = run_pairs_reversion(
                spread_tr,
                zscore_tr,
                **combo,
                fee_per_side=fee_per_side,
                slippage_per_side=slip_per_side,
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
            fee_per_side=fee_per_side,
            slippage_per_side=slip_per_side,
        )
        test_m = metrics_dict(test_trades, period_days=test_days_real)

        results.append(
            {
                "window_i": win_i,
                "train_start": spread_tr.index[0],
                "train_end": spread_tr.index[-1],
                "test_start": spread_te.index[0],
                "test_end": spread_te.index[-1],
                "best_combo": best_combo,
                "train_m": best_train_m,
                "test_m": test_m,
            }
        )
    return results


def _trimmed_mean(values: list[float], trim_frac: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = int(len(sorted_v) * trim_frac)
    trimmed = sorted_v[k : len(sorted_v) - k] if k > 0 else sorted_v
    if not trimmed:
        return 0.0
    return float(mean(trimmed))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def _summarize(sharpes: list[float], label: str) -> dict:
    if not sharpes:
        return {
            "label": label,
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "trimmed_mean_10": 0.0,
            "std": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "positive_pct": 0.0,
        }
    return {
        "label": label,
        "n": len(sharpes),
        "mean": float(mean(sharpes)),
        "median": float(median(sharpes)),
        "trimmed_mean_10": _trimmed_mean(sharpes, 0.10),
        "std": float(stdev(sharpes)) if len(sharpes) > 1 else 0.0,
        "p25": _percentile(sharpes, 0.25),
        "p50": _percentile(sharpes, 0.50),
        "p75": _percentile(sharpes, 0.75),
        "positive_pct": sum(1 for s in sharpes if s > 0) / len(sharpes) * 100,
    }


def _print_stat_row(s: dict) -> None:
    print(
        f"  {s['label']:<45} n={s['n']:>3} "
        f"mean={s['mean']:>+7.2f}  median={s['median']:>+7.2f}  "
        f"trim10={s['trimmed_mean_10']:>+7.2f}  std={s['std']:>6.2f}  "
        f"pos={s['positive_pct']:>5.1f}%"
    )


def _print_header(title: str) -> None:
    print("\n" + "=" * 108)
    print(title)
    print("=" * 108)


def investigation_1_outliers(results: list[dict]) -> dict:
    """Section 1 — outlier analysis on 1x fee sweep."""
    _print_header("[1] OUTLIER ANALYSIS — SOL/AVAX (Bybit futures maker fees 1x)")
    sharpes = [r["test_m"]["sharpe"] for r in results]
    full = _summarize(sharpes, "Full sample")
    print(
        f"  n={full['n']}  "
        f"mean={full['mean']:+.2f}  median={full['median']:+.2f}  "
        f"trim10={full['trimmed_mean_10']:+.2f}  std={full['std']:.2f}"
    )
    print(
        f"  p25={full['p25']:+.2f}  p50={full['p50']:+.2f}  p75={full['p75']:+.2f}  "
        f"positive={full['positive_pct']:.1f}%"
    )

    ordered = sorted(results, key=lambda r: r["test_m"]["sharpe"], reverse=True)
    top3 = ordered[:3]
    bot3 = ordered[-3:]

    print("\n  TOP-3 test Sharpe windows (most generous to the edge):")
    print(
        f"    {'win':>3}  {'train_start':<12}  {'combo':<20}  "
        f"{'tr_shrp':>8}  {'te_shrp':>8}  {'te_n':>5}"
    )
    for r in top3:
        print(
            f"    {r['window_i']:>3}  "
            f"{r['train_start'].strftime('%Y-%m-%d'):<12}  "
            f"{_combo_label(r['best_combo']):<20}  "
            f"{r['train_m']['sharpe']:>+8.2f}  "
            f"{r['test_m']['sharpe']:>+8.2f}  "
            f"{r['test_m']['n_trades']:>5}"
        )

    print("\n  BOTTOM-3 test Sharpe windows (worst-case):")
    print(
        f"    {'win':>3}  {'train_start':<12}  {'combo':<20}  "
        f"{'tr_shrp':>8}  {'te_shrp':>8}  {'te_n':>5}"
    )
    for r in bot3:
        print(
            f"    {r['window_i']:>3}  "
            f"{r['train_start'].strftime('%Y-%m-%d'):<12}  "
            f"{_combo_label(r['best_combo']):<20}  "
            f"{r['train_m']['sharpe']:>+8.2f}  "
            f"{r['test_m']['sharpe']:>+8.2f}  "
            f"{r['test_m']['n_trades']:>5}"
        )

    ordered_ts = sorted(results, key=lambda r: r["test_m"]["sharpe"])
    without_top3 = ordered_ts[:-3]  # drop the 3 highest
    sharpes_wo = [r["test_m"]["sharpe"] for r in without_top3]
    wo_summary = _summarize(sharpes_wo, "Without top-3 windows")

    print("\n  Sample after dropping the 3 best test Sharpe windows:")
    print(
        f"    n={wo_summary['n']}  "
        f"mean={wo_summary['mean']:+.2f}  median={wo_summary['median']:+.2f}  "
        f"trim10={wo_summary['trimmed_mean_10']:+.2f}  std={wo_summary['std']:.2f}"
    )

    return {
        "full": full,
        "top3": top3,
        "bottom3": bot3,
        "without_top3": wo_summary,
    }


def investigation_2_min_trades(results: list[dict]) -> dict:
    """Section 2 — aggregate only on windows with >= N test trades."""
    _print_header("[2] MINIMUM TRADES FILTER — SOL/AVAX")
    print(
        "  Rationale: a Sharpe built on 2-3 test trades is noise. Drop low-activity "
        "windows\n  and see whether the edge survives when forced onto a larger sample."
    )
    print()

    base_sharpes = [r["test_m"]["sharpe"] for r in results]
    base = _summarize(base_sharpes, "All windows (no filter)")
    _print_stat_row(base)

    out = {"no_filter": base, "thresholds": {}}
    for thr in (4, 5, 6):
        kept = [r for r in results if r["test_m"]["n_trades"] >= thr]
        sharpes = [r["test_m"]["sharpe"] for r in kept]
        s = _summarize(sharpes, f"windows with test n_trades >= {thr}")
        _print_stat_row(s)
        out["thresholds"][thr] = s

    base_mean = base["mean"]
    print()
    print("  Mean Sharpe drop vs no-filter baseline:")
    for thr, s in out["thresholds"].items():
        drop = s["mean"] - base_mean
        pct = (s["mean"] / base_mean * 100) if base_mean != 0 else 0.0
        print(
            f"    >= {thr} trades:  mean={s['mean']:+.2f}  "
            f"delta={drop:+.2f}  ({pct:.0f}% of baseline)"
        )
    return out


def investigation_3_fee_stress(
    spread: pd.Series,
    zscore: pd.Series,
    aligned: pd.DataFrame,
) -> dict:
    """Section 3 — re-run the sweep at 2x and 4x fees."""
    _print_header("[3] FEE STRESS TEST — SOL/AVAX")
    print(
        "  Rationale: if the edge dies at 2x fees, it won't survive real-world "
        "slippage or a bad day at the exchange."
    )
    print()

    out: dict = {"scenarios": {}}
    for label, mult in FEE_SCENARIOS:
        fee = BASE_FEE * mult
        slip = BASE_SLIP * mult
        res = run_walk_forward(spread, zscore, aligned, fee, slip)
        sharpes = [r["test_m"]["sharpe"] for r in res]
        pnls = [r["test_m"]["total_pnl"] for r in res]
        s = _summarize(sharpes, label)
        s["mean_test_pnl_pct"] = float(mean(pnls)) * 100 if pnls else 0.0
        print(
            f"  fee={fee * 100:.2f}%/side slip={slip * 100:.2f}%/side  "
            f"[{label}]"
        )
        _print_stat_row(s)
        print(f"    mean test PnL: {s['mean_test_pnl_pct']:+.2f}%")
        out["scenarios"][label] = s

    base_sharpe = out["scenarios"][FEE_SCENARIOS[0][0]]["mean"]
    print()
    print("  Mean Sharpe degradation curve (baseline = 1x):")
    for label, _ in FEE_SCENARIOS:
        s = out["scenarios"][label]
        delta = s["mean"] - base_sharpe
        print(
            f"    {label:<30} mean={s['mean']:+.2f}  "
            f"median={s['median']:+.2f}  delta={delta:+.2f}"
        )
    return out


def investigation_4_restricted_combos(results: list[dict]) -> dict:
    """Section 4 — keep only combos picked >= 3 times (or top-4 fallback)."""
    _print_header("[4] RESTRICTED COMBO SET — SOL/AVAX")
    print(
        "  Rationale: if every window picks a different combo, the 'edge' might be "
        "curve-fit.\n  Force the aggregates onto the most-picked combos only."
    )
    print()

    counts = Counter(_combo_label(r["best_combo"]) for r in results)
    print("  Combo pick frequency across all windows:")
    for label, n in counts.most_common():
        bar = "#" * int(n / len(results) * 40) if results else ""
        print(f"    {label:<22} {n:>3}/{len(results):<3}  {bar}")

    freq_kept = {label for label, n in counts.items() if n >= 3}
    if len(freq_kept) >= 4:
        kept_labels = freq_kept
        rule = ">= 3 picks"
    else:
        kept_labels = {label for label, _ in counts.most_common(4)}
        rule = "top-4 by pick count (fallback)"
    print(f"\n  Rule applied: {rule}")
    print(f"  Kept combos:  {sorted(kept_labels)}")

    kept_results = [r for r in results if _combo_label(r["best_combo"]) in kept_labels]
    kept_sharpes = [r["test_m"]["sharpe"] for r in kept_results]
    base_sharpes = [r["test_m"]["sharpe"] for r in results]

    base = _summarize(base_sharpes, "All combos (baseline)")
    restricted = _summarize(kept_sharpes, f"Restricted ({rule})")

    print()
    _print_stat_row(base)
    _print_stat_row(restricted)

    return {"rule": rule, "kept_labels": sorted(kept_labels), "baseline": base, "restricted": restricted}


def main() -> None:
    print("=" * 108)
    print("WADDLE — Investigation A: SOL/AVAX variance & robustness")
    print("=" * 108)
    print(f"Pair:       {PAIR_A} vs {PAIR_B}")
    print(
        f"Windows:    train={TRAIN_DAYS}d / test={TEST_DAYS}d / step={STEP_DAYS}d"
    )
    print(
        f"Grid:       {len(PARAM_GRID)} combos, Calmar selection on train"
    )
    print(
        f"Base fees:  {BASE_FEE * 100:.2f}%/side fee + {BASE_SLIP * 100:.2f}%/side slip "
        "(Bybit futures maker)"
    )

    aligned = load_aligned_pair(PAIR_A, PAIR_B)
    total_days = (aligned.index[-1] - aligned.index[0]).total_seconds() / 86400
    spread = log_spread(aligned["close_a"], aligned["close_b"])
    zscore = rolling_zscore(spread, window=ROLLING_WINDOW)

    print(
        f"Data:       {total_days:.1f} days, {len(aligned)} aligned hourly bars"
    )

    base_results = run_walk_forward(spread, zscore, aligned, BASE_FEE, BASE_SLIP)
    print(f"Windows:    {len(base_results)} rolling walk-forward windows computed")

    out1 = investigation_1_outliers(base_results)
    out2 = investigation_2_min_trades(base_results)
    out3 = investigation_3_fee_stress(spread, zscore, aligned)
    out4 = investigation_4_restricted_combos(base_results)

    _print_header("[VERDICT] — decision criterion (investigation plan)")
    print(
        "  Decision rule: Median test Sharpe > +1.5 after outlier removal AND "
        "edge > 0 with 2x fees"
    )
    print()

    median_no_top3 = out1["without_top3"]["median"]
    mean_2x = out3["scenarios"]["2x (stress)"]["mean"]
    median_2x = out3["scenarios"]["2x (stress)"]["median"]

    crit_a = median_no_top3 > 1.5
    crit_b = mean_2x > 0
    passed = crit_a and crit_b

    print(f"  Median test Sharpe (dropping top-3 outliers): {median_no_top3:+.2f}")
    print(f"     criterion > +1.5 : {'PASS' if crit_a else 'FAIL'}")
    print(
        f"  Mean test Sharpe at 2x fees:                   {mean_2x:+.2f}  "
        f"(median {median_2x:+.2f})"
    )
    print(f"     criterion >   0   : {'PASS' if crit_b else 'FAIL'}")
    print()
    if passed:
        verdict = "EDGE ROBUST"
    else:
        base_mean = out1["full"]["mean"]
        median_full = out1["full"]["median"]
        if base_mean > 0 and (median_full > 0 or mean_2x > 0):
            verdict = "EDGE FRAGILE"
        else:
            verdict = "EDGE ARTEFACT"
    print(f"  BOTTOM LINE: {verdict}")


if __name__ == "__main__":
    main()
