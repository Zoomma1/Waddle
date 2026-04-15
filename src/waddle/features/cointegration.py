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
