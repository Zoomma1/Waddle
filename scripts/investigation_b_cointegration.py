"""Investigation B — Formal cointegration tests on candidate pairs.

For each of (BTC/ETH, SOL/ETH, SOL/AVAX, BNB/ETH) this script runs:
  1. Engle-Granger 2-step test (OLS + ADF on residuals) -> beta, alpha, p-value
  2. AR(1) half-life of mean reversion on the residual spread
  3. Rolling 90d Engle-Granger (step 30d) -> stability of cointegration over time
  4. Rolling log-return correlation over full sample as sanity check

The test is "is there a linear combination of log(A) and log(B) that is
stationary" — if yes, the spread has a unit-root-free residual and there's
a theoretical force pulling it back to equilibrium. That's what makes
mean-reversion trading work in principle.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from waddle.data.storage import load_ohlcv
from waddle.features.correlation import rolling_log_return_correlation


PAIRS = [
    ("BTC/USDT", "ETH/USDT"),
    ("SOL/USDT", "ETH/USDT"),
    ("SOL/USDT", "AVAX/USDT"),
    ("BNB/USDT", "ETH/USDT"),
]


def _label(symbol: str) -> str:
    return symbol.split("/")[0]


def load_pair_closes(sym_a: str, sym_b: str) -> pd.DataFrame:
    a = load_ohlcv(sym_a)["close"].rename("a")
    b = load_ohlcv(sym_b)["close"].rename("b")
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    return df


def engle_granger(df: pd.DataFrame) -> dict:
    """Step 1: OLS of log(a) on log(b) with intercept.
    Step 2: ADF on residuals.
    """
    log_a = np.log(df["a"])
    log_b = np.log(df["b"])

    X = sm.add_constant(log_b)
    ols = sm.OLS(log_a, X).fit()
    alpha = float(ols.params.iloc[0])
    beta = float(ols.params.iloc[1])

    residual = log_a - beta * log_b - alpha

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_stat, adf_p, _, _, crit, _ = adfuller(residual.dropna(), autolag="AIC")

    return {
        "alpha": alpha,
        "beta": beta,
        "residual": residual,
        "adf_stat": float(adf_stat),
        "adf_p": float(adf_p),
        "crit_1": float(crit["1%"]),
        "crit_5": float(crit["5%"]),
        "crit_10": float(crit["10%"]),
    }


def half_life(residual: pd.Series) -> dict:
    """Fit AR(1): spread_t = mu + rho * spread_{t-1} + eps.
    Half-life in hours = -ln(2)/ln(rho) for 0 < rho < 1.
    """
    s = residual.dropna()
    lag = s.shift(1).dropna()
    cur = s.loc[lag.index]

    X = sm.add_constant(lag.rename("lag"))
    ols = sm.OLS(cur, X).fit()
    mu = float(ols.params.iloc[0])
    rho = float(ols.params.iloc[1])

    if rho >= 1.0:
        hl_hours = float("inf")
        status = "no mean reversion (rho>=1)"
    elif rho <= 0.0:
        hl_hours = -np.log(2) / np.log(abs(rho)) if abs(rho) < 1 else float("inf")
        status = "oscillatory (rho<=0)"
    else:
        hl_hours = -np.log(2) / np.log(rho)
        status = "ok"

    return {
        "mu": mu,
        "rho": rho,
        "hl_hours": hl_hours,
        "hl_days": hl_hours / 24.0 if np.isfinite(hl_hours) else float("inf"),
        "status": status,
    }


def rolling_engle_granger(df: pd.DataFrame, window_days: int = 90, step_days: int = 30) -> dict:
    """Rolling Engle-Granger cointegration test.
    Returns fraction of windows with p < 0.05 and per-window series.
    """
    window_h = window_days * 24
    step_h = step_days * 24
    n = len(df)

    starts = list(range(0, n - window_h + 1, step_h))
    p_values = []
    window_labels = []

    for start in starts:
        end = start + window_h
        sub = df.iloc[start:end]
        try:
            res = engle_granger(sub)
            p_values.append(res["adf_p"])
        except Exception:
            p_values.append(np.nan)
        window_labels.append(sub.index[0].strftime("%Y-%m-%d"))

    p_arr = np.array(p_values, dtype=float)
    valid = ~np.isnan(p_arr)
    n_total = int(valid.sum())
    n_coint = int(((p_arr < 0.05) & valid).sum())
    frac = n_coint / n_total if n_total else float("nan")

    return {
        "n_total": n_total,
        "n_coint": n_coint,
        "frac_coint": frac,
        "p_values": p_arr,
        "window_starts": window_labels,
    }


def corr_sanity(df: pd.DataFrame, window_h: int = 168) -> dict:
    c = rolling_log_return_correlation(df["a"], df["b"], window=window_h).dropna()
    return {
        "mean_corr": float(c.mean()),
        "min_corr": float(c.min()),
        "max_corr": float(c.max()),
    }


def run():
    print("=" * 78)
    print("INVESTIGATION B — Formal cointegration of candidate pairs")
    print("=" * 78)
    print()
    print("Data: full 18 months of 1h bars loaded from data/waddle.db")
    print("Method:")
    print("  - Engle-Granger 2-step: OLS(log_a ~ log_b) + ADF on residuals")
    print("  - AR(1) half-life on residual spread")
    print("  - Rolling 90d Engle-Granger, step 30d")
    print("  - Rolling 168h log-return correlation (sanity check)")
    print()

    results = []

    for sym_a, sym_b in PAIRS:
        la, lb = _label(sym_a), _label(sym_b)
        pair_name = f"{la}/{lb}"
        print("-" * 78)
        print(f"PAIR: {pair_name}")
        print("-" * 78)

        df = load_pair_closes(sym_a, sym_b)
        print(f"  bars: {len(df)}  range: {df.index.min().date()} -> {df.index.max().date()}")

        eg = engle_granger(df)
        hl = half_life(eg["residual"])
        roll = rolling_engle_granger(df, window_days=90, step_days=30)
        corr = corr_sanity(df)

        coint_5pct = "YES" if eg["adf_p"] < 0.05 else "NO"
        coint_1pct = "YES" if eg["adf_p"] < 0.01 else "NO"

        print()
        print(f"  Engle-Granger (full sample):")
        print(f"    log_a = {eg['alpha']:+.4f} + {eg['beta']:+.4f} * log_b + residual")
        print(f"    ADF statistic  : {eg['adf_stat']:.4f}")
        print(f"    ADF p-value    : {eg['adf_p']:.6f}")
        print(f"    Critical values: 1%={eg['crit_1']:.3f}  5%={eg['crit_5']:.3f}  10%={eg['crit_10']:.3f}")
        print(f"    Cointegrated @5% : {coint_5pct}")
        print(f"    Cointegrated @1% : {coint_1pct}")
        print()
        print(f"  Half-life of mean reversion (AR(1) on residual):")
        print(f"    mu  = {hl['mu']:+.6f}")
        print(f"    rho = {hl['rho']:+.6f}")
        print(f"    status = {hl['status']}")
        if np.isfinite(hl["hl_hours"]):
            print(f"    half-life: {hl['hl_hours']:.1f} hours = {hl['hl_days']:.2f} days")
        else:
            print(f"    half-life: infinite (no mean reversion)")
        print()
        print(f"  Rolling 90d Engle-Granger (step 30d):")
        print(f"    {roll['n_coint']}/{roll['n_total']} windows cointegrated @5% "
              f"= {roll['frac_coint']*100:.1f}%")
        print()
        print(f"  Log-return correlation sanity (168h rolling):")
        print(f"    mean={corr['mean_corr']:.3f}  min={corr['min_corr']:.3f}  max={corr['max_corr']:.3f}")
        print()

        results.append({
            "pair": pair_name,
            "beta": eg["beta"],
            "alpha": eg["alpha"],
            "adf_stat": eg["adf_stat"],
            "adf_p": eg["adf_p"],
            "coint_5pct": coint_5pct,
            "coint_1pct": coint_1pct,
            "rho": hl["rho"],
            "hl_days": hl["hl_days"],
            "hl_status": hl["status"],
            "roll_frac": roll["frac_coint"],
            "roll_n_coint": roll["n_coint"],
            "roll_n_total": roll["n_total"],
            "mean_corr": corr["mean_corr"],
            "min_corr": corr["min_corr"],
        })

    # Summary table
    print("=" * 78)
    print("SUMMARY TABLE")
    print("=" * 78)
    print()
    header = f"{'pair':<12} {'beta':>8} {'alpha':>10} {'ADF p':>10} {'coint@5%':>10} " \
             f"{'HL days':>10} {'roll %':>10} {'mean_corr':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        hl_str = f"{r['hl_days']:.2f}" if np.isfinite(r["hl_days"]) else "inf"
        print(
            f"{r['pair']:<12} "
            f"{r['beta']:>8.3f} "
            f"{r['alpha']:>10.4f} "
            f"{r['adf_p']:>10.6f} "
            f"{r['coint_5pct']:>10} "
            f"{hl_str:>10} "
            f"{r['roll_frac']*100:>9.1f}% "
            f"{r['mean_corr']:>10.3f}"
        )
    print()

    # Ranking comparisons
    print("-" * 78)
    print("RANKING COMPARISON")
    print("-" * 78)
    wf_ranking = [
        ("SOL/AVAX", 4.43),
        ("BTC/ETH",  0.78),
        ("BNB/ETH",  0.72),
        ("SOL/ETH",  0.67),
    ]
    print("Walk-forward sweep (OOS Sharpe):")
    for name, sh in wf_ranking:
        print(f"  {name:<10}  Sharpe {sh:+.2f}")
    print()

    by_p = sorted(results, key=lambda r: r["adf_p"])
    print("Cointegration strength (lowest ADF p-value first):")
    for r in by_p:
        print(f"  {r['pair']:<10}  ADF p={r['adf_p']:.6f}  HL={r['hl_days']:.2f}d  rolling={r['roll_frac']*100:.0f}%")
    print()

    by_hl = sorted(
        [r for r in results if np.isfinite(r["hl_days"])],
        key=lambda r: r["hl_days"],
    )
    print("Speed of mean reversion (shortest half-life first):")
    for r in by_hl:
        print(f"  {r['pair']:<10}  HL={r['hl_days']:.2f}d  rho={r['rho']:.4f}")
    print()

    # Criterion B verdict on SOL/AVAX
    sol_avax = next(r for r in results if r["pair"] == "SOL/AVAX")
    print("=" * 78)
    print("CRITERION B VERDICT — SOL/AVAX")
    print("=" * 78)
    passes_p = sol_avax["adf_p"] < 0.05
    passes_hl = np.isfinite(sol_avax["hl_days"]) and sol_avax["hl_days"] < 30
    verdict = "PASS" if (passes_p and passes_hl) else "FAIL"
    print(f"  ADF p-value < 0.05 ?  {'YES' if passes_p else 'NO':<4}  "
          f"(p = {sol_avax['adf_p']:.6f})")
    print(f"  Half-life  < 30 days ? {'YES' if passes_hl else 'NO':<4}  "
          f"(HL = {sol_avax['hl_days']:.2f} days)")
    print(f"  VERDICT: {verdict}")
    print()


if __name__ == "__main__":
    run()
