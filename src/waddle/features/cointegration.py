"""Rolling cointegration and regime filters for pairs trading.

The Engle-Granger rolling p-value is a time-varying measure of whether the
linear combination of two log-prices is stationary over a recent window.
When the p-value is low, the mean reversion hypothesis is backed by recent
evidence and we're willing to enter new trades. When it drifts above the
threshold, the relationship has weakened or broken — we stop opening new
positions until it recovers.

See ADR-007 §"Conditions de déploiement obligatoires" for why these filters
are mandatory on the SOL/AVAX baseline.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def _engle_granger_pvalue(log_a: np.ndarray, log_b: np.ndarray) -> float:
    """Single-window Engle-Granger: OLS residuals + ADF p-value.

    Returns NaN if the ADF call fails numerically (rare, happens on degenerate
    windows with near-constant residuals).
    """
    X = sm.add_constant(log_b)
    try:
        ols = sm.OLS(log_a, X).fit()
        alpha = float(ols.params[0])
        beta = float(ols.params[1])
        residual = log_a - beta * log_b - alpha
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, p_value, *_ = adfuller(residual, autolag="AIC")
        return float(p_value)
    except Exception:
        return float("nan")


def rolling_engle_granger_pvalue(
    price_a: pd.Series,
    price_b: pd.Series,
    window_bars: int = 2160,
    step_bars: int = 24,
) -> pd.Series:
    """Rolling Engle-Granger ADF p-value indexed by timestamp.

    For each refresh point (every `step_bars`), fit OLS of log(a) on log(b)
    over the trailing `window_bars` and run ADF on the residuals. The result
    is the p-value for "residuals have a unit root" — low p-value means we
    reject the unit root and accept cointegration.

    Values between refresh points are forward-filled (the p-value from the
    most recent complete window applies until the next refresh). Before the
    first complete window (warmup), the series is NaN.

    Parameters
    ----------
    window_bars : trailing window for the cointegration test (default 2160 = 90 days).
    step_bars   : refresh period (default 24 = once per day on hourly data).
    """
    if len(price_a) != len(price_b):
        raise ValueError("price_a and price_b must have the same length")
    if not price_a.index.equals(price_b.index):
        raise ValueError("price_a and price_b must share the same index")

    log_a_arr = np.log(price_a.to_numpy(dtype=float))
    log_b_arr = np.log(price_b.to_numpy(dtype=float))
    n = len(price_a)

    pvalues = np.full(n, np.nan, dtype=float)

    for end in range(window_bars, n + 1, step_bars):
        start = end - window_bars
        p = _engle_granger_pvalue(log_a_arr[start:end], log_b_arr[start:end])
        pvalues[end - 1] = p

    series = pd.Series(pvalues, index=price_a.index, name="eg_pvalue")
    return series.ffill()


def cointegration_mask(
    pvalue_series: pd.Series,
    threshold: float = 0.10,
) -> pd.Series:
    """Boolean mask: True = cointegration active (p-value < threshold).

    NaN p-values become False — we refuse to trade without a valid test result
    (conservative default: safer to skip an entry than to enter blind).
    """
    mask = pvalue_series < threshold
    return mask.where(pvalue_series.notna(), False).astype(bool)


def bear_halt_mask(
    btc_close: pd.Series,
    reference_index: pd.Index,
    threshold: float = 60000.0,
) -> pd.Series:
    """Boolean mask aligned to reference_index: True = safe to trade (BTC above threshold).

    The BTC series is reindexed onto the reference index (the pair's own
    timeline) with forward-fill, so a missing BTC bar inherits the last known
    close. Any remaining NaN becomes False (no trading without a BTC reading).
    """
    aligned = btc_close.reindex(reference_index, method="ffill")
    mask = aligned > threshold
    return mask.where(aligned.notna(), False).astype(bool)


def _ar1_half_life_days(y: np.ndarray) -> float:
    """Single-window AR(1) half-life of mean reversion, in days.

    Fits spread_t = alpha + rho * spread_{t-1} + eps via OLS on the provided
    window (assumed hourly). Returns:
      - half-life in days  = -ln(2) / ln(rho) / 24, if 0 < rho < 1
      - +inf               if rho >= 1 (no mean reversion / explosive / unit root)
      - finite negative    if rho <= 0 (oscillatory — rare, returned as-is,
                             but half_life_mask() treats negatives as tradable)
      - NaN                if OLS fails or variance is degenerate
    """
    if y.size < 10:
        return float("nan")
    y_prev = y[:-1]
    y_next = y[1:]
    if not np.all(np.isfinite(y_prev)) or not np.all(np.isfinite(y_next)):
        return float("nan")
    x_mean = y_prev.mean()
    y_mean = y_next.mean()
    dx = y_prev - x_mean
    dy = y_next - y_mean
    denom = float((dx * dx).sum())
    if denom <= 0.0:
        return float("nan")
    rho = float((dx * dy).sum() / denom)
    if not np.isfinite(rho):
        return float("nan")
    if rho >= 1.0:
        return float("inf")
    if rho <= 0.0:
        # oscillatory regime — log of a negative is undefined, but the window
        # is clearly reverting fast. Return a conservative-but-tradable small
        # positive value so the mask lets it through.
        return 0.0
    hl_hours = -np.log(2.0) / np.log(rho)
    return float(hl_hours / 24.0)


def rolling_half_life(
    spread: pd.Series,
    window_bars: int = 720,
    step_bars: int = 24,
) -> pd.Series:
    """Rolling AR(1) half-life of mean reversion for the spread series, in days.

    At each refresh point (every `step_bars`), fit AR(1) over the trailing
    `window_bars`:
        spread_t = alpha + rho * spread_{t-1} + eps_t

    Half-life in hours = -ln(2) / ln(rho) if 0 < rho < 1. Converted to days.

    Aligned to the strategy horizon: default window_bars = 720 = 30 days,
    step_bars = 24 = recompute once per day. Much shorter than the 90-day ADF
    test — deliberately, because the strategy's holding horizon is 48h and the
    filter signal should live on a compatible time scale.

    Returns a series indexed like `spread`, forward-filled between refreshes.
    - Before first complete window: NaN (warmup)
    - rho >= 1 (explosive / unit root): +inf (treat as "won't revert")
    - rho <= 0 (oscillatory): 0.0 (tradable — fast reversion)
    - OLS numerical failure: NaN
    """
    if not isinstance(spread, pd.Series):
        raise TypeError("spread must be a pandas Series")
    arr = spread.to_numpy(dtype=float)
    n = arr.size

    hl = np.full(n, np.nan, dtype=float)
    for end in range(window_bars, n + 1, step_bars):
        start = end - window_bars
        window = arr[start:end]
        if not np.all(np.isfinite(window)):
            continue
        hl[end - 1] = _ar1_half_life_days(window)

    series = pd.Series(hl, index=spread.index, name="half_life_days")
    return series.ffill()


def half_life_mask(
    half_life_series: pd.Series,
    threshold_days: float = 30.0,
) -> pd.Series:
    """Boolean mask: True = half-life is short enough to trade.

    True iff 0 <= half_life_series < threshold_days.
    NaN or +inf -> False (conservative: no trading without a valid, short
    half-life). Negative values (from the oscillatory edge case) also become
    False in the strict form, but rolling_half_life() clamps those to 0.0
    already, so in practice the only non-tradable outputs are NaN and +inf.
    """
    valid = half_life_series.notna() & np.isfinite(half_life_series)
    mask = valid & (half_life_series >= 0) & (half_life_series < threshold_days)
    return mask.astype(bool)
