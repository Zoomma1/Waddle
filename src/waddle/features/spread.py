"""Spread and rolling statistics for pairs trading."""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_spread(price_a: pd.Series, price_b: pd.Series) -> pd.Series:
    """Log spread = log(A) - log(B).

    Preferred over raw ratio for stat arb: additive, symmetric, and the PnL
    of a long-A/short-B position scales linearly with changes in log spread.
    """
    return np.log(price_a) - np.log(price_b)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score using only past data at each point (no lookahead bias)."""
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return (series - mean) / std
