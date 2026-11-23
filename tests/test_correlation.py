import numpy as np
import pandas as pd
import pytest

from waddle.features.correlation import rolling_log_return_correlation


def _prices(seed: int, n: int = 60) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))


def test_power_of_same_asset_has_perfect_correlation():
    a = _prices(0)
    b = a**2  # log returns are exactly 2x those of a
    corr = rolling_log_return_correlation(a, b, window=10)
    assert corr.dropna().tolist() == pytest.approx([1.0] * len(corr.dropna()))


def test_inverse_asset_has_perfect_anticorrelation():
    a = _prices(1)
    corr = rolling_log_return_correlation(a, 1.0 / a, window=10)
    assert corr.dropna().tolist() == pytest.approx([-1.0] * len(corr.dropna()))


def test_warmup_length_is_window_bars_nan():
    # diff() burns one bar, rolling(window) burns window-1 more
    a = _prices(2, n=30)
    b = _prices(3, n=30)
    corr = rolling_log_return_correlation(a, b, window=5)
    assert corr.iloc[:5].isna().all()
    assert corr.iloc[5:].notna().all()


def test_uses_returns_not_price_levels():
    # a and b trend together (price corr ~1) but their returns are independent
    n = 200
    rng = np.random.default_rng(4)
    trend = np.exp(np.linspace(0, 3, n))
    a = pd.Series(trend * np.exp(rng.normal(0, 0.01, n)))
    b = pd.Series(trend * np.exp(rng.normal(0, 0.01, n)))
    assert a.corr(b) > 0.99
    corr = rolling_log_return_correlation(a, b, window=n - 1)
    assert abs(corr.iloc[-1]) < 0.3
