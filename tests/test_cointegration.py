import numpy as np
import pandas as pd
import pytest

from waddle.features.cointegration import (
    bear_halt_mask,
    cointegration_mask,
    half_life_mask,
    rolling_engle_granger_pvalue,
    rolling_half_life,
)

N = 400
IDX = pd.date_range("2024-01-01", periods=N, freq="h", tz="UTC")


def _cointegrated_pair() -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(42)
    log_b = np.cumsum(rng.normal(0, 0.01, N)) + 5
    noise = np.zeros(N)
    for i in range(1, N):
        noise[i] = 0.5 * noise[i - 1] + rng.normal(0, 0.005)
    log_a = 0.8 * log_b + noise + 1
    return pd.Series(np.exp(log_a), index=IDX), pd.Series(np.exp(log_b), index=IDX)


def _independent_walks() -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(42)
    a = np.cumsum(rng.normal(0, 0.01, N)) + 5
    b = np.cumsum(rng.normal(0, 0.01, N)) + 5
    return pd.Series(np.exp(a), index=IDX), pd.Series(np.exp(b), index=IDX)


# --- rolling_engle_granger_pvalue ---

def test_pvalue_rejects_length_mismatch():
    a = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="same length"):
        rolling_engle_granger_pvalue(a, a.iloc[:2], window_bars=2, step_bars=1)


def test_pvalue_rejects_index_mismatch():
    a = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    b = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 3])
    with pytest.raises(ValueError, match="same index"):
        rolling_engle_granger_pvalue(a, b, window_bars=2, step_bars=1)


def test_pvalue_is_nan_before_first_window_then_forward_filled():
    a, b = _cointegrated_pair()
    p = rolling_engle_granger_pvalue(a, b, window_bars=300, step_bars=50)
    assert p.name == "eg_pvalue"
    assert p.iloc[:299].isna().all()
    assert p.iloc[299:].notna().all()
    # refresh points at bars 299, 349, 399; constant in between
    assert p.iloc[299:349].nunique() == 1
    assert p.iloc[349:399].nunique() == 1


def test_pvalue_low_for_cointegrated_pair():
    a, b = _cointegrated_pair()
    p = rolling_engle_granger_pvalue(a, b, window_bars=300, step_bars=50)
    assert p.iloc[299] < 1e-6


def test_pvalue_high_for_independent_random_walks():
    a, b = _independent_walks()
    p = rolling_engle_granger_pvalue(a, b, window_bars=300, step_bars=100)
    assert p.iloc[299] > 0.1


def test_pvalue_degenerate_constant_prices_gives_all_nan():
    k = pd.Series([100.0] * N, index=IDX)
    p = rolling_engle_granger_pvalue(k, k, window_bars=300, step_bars=50)
    assert p.isna().all()


# --- cointegration_mask ---

def test_cointegration_mask_is_strictly_below_threshold_and_nan_is_false():
    p = pd.Series([0.01, 0.10, 0.11, np.nan])
    mask = cointegration_mask(p, threshold=0.10)
    assert mask.tolist() == [True, False, False, False]
    assert mask.dtype == bool


# --- bear_halt_mask ---

def test_bear_halt_mask_threshold_is_strict():
    ref = pd.date_range("2024-01-01", periods=3, freq="h")
    btc = pd.Series([59_999.0, 60_000.0, 60_001.0], index=ref)
    mask = bear_halt_mask(btc, ref, threshold=60_000.0)
    assert mask.tolist() == [False, False, True]


def test_bear_halt_mask_forward_fills_missing_btc_bar():
    ref = pd.date_range("2024-01-01", periods=3, freq="h")
    btc = pd.Series([70_000.0], index=ref[:1])
    assert bear_halt_mask(btc, ref).tolist() == [True, True, True]


def test_bear_halt_mask_false_before_first_btc_reading():
    ref = pd.date_range("2024-01-01", periods=3, freq="h")
    btc = pd.Series([70_000.0], index=ref[1:2])
    assert bear_halt_mask(btc, ref).tolist() == [False, True, True]


# --- half life ---

def _geometric(rho: float, n: int = 40) -> pd.Series:
    return pd.Series(10.0 * rho ** np.arange(n))


def test_half_life_matches_ar1_formula_in_days():
    rho = 0.9
    hl = rolling_half_life(_geometric(rho), window_bars=20, step_bars=5)
    expected_days = (-np.log(2.0) / np.log(rho)) / 24.0
    assert hl.iloc[19:].tolist() == pytest.approx([expected_days] * 21)


def test_half_life_warmup_is_nan_and_values_forward_filled():
    hl = rolling_half_life(_geometric(0.9, n=30), window_bars=20, step_bars=5)
    assert hl.name == "half_life_days"
    assert hl.iloc[:19].isna().all()
    assert hl.iloc[19:].notna().all()


def test_half_life_explosive_series_is_inf():
    hl = rolling_half_life(_geometric(1.1, n=30), window_bars=20, step_bars=5)
    assert np.isinf(hl.iloc[-1]) and hl.iloc[-1] > 0


def test_half_life_oscillatory_series_is_clamped_to_zero():
    hl = rolling_half_life(_geometric(-0.5, n=30), window_bars=20, step_bars=5)
    assert hl.iloc[-1] == 0.0


def test_half_life_constant_series_is_nan():
    hl = rolling_half_life(pd.Series([1.0] * 30), window_bars=20, step_bars=5)
    assert hl.isna().all()


def test_half_life_window_with_nan_keeps_previous_value():
    # characterization: a non-finite window is skipped, ffill then carries the stale value
    s = _geometric(0.9, n=30)
    s.iloc[24] = np.nan
    hl = rolling_half_life(s, window_bars=20, step_bars=5)
    clean = rolling_half_life(_geometric(0.9, n=30), window_bars=20, step_bars=5)
    assert hl.iloc[29] == pytest.approx(clean.iloc[19])


def test_half_life_requires_a_series():
    with pytest.raises(TypeError):
        rolling_half_life(np.zeros(30))  # type: ignore[arg-type]


# --- half_life_mask ---

def test_half_life_mask_bounds_nan_inf_and_negatives():
    hl = pd.Series([0.0, 29.99, 30.0, np.inf, np.nan, -1.0])
    mask = half_life_mask(hl, threshold_days=30.0)
    assert mask.tolist() == [True, True, False, False, False, False]
