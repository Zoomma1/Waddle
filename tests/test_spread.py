import numpy as np
import pandas as pd
import pytest

from waddle.features.spread import log_spread, rolling_zscore


def test_log_spread_is_log_a_minus_log_b():
    a = pd.Series([100.0, 110.0])
    b = pd.Series([50.0, 55.0])
    out = log_spread(a, b)
    assert out.tolist() == pytest.approx([np.log(2.0), np.log(2.0)])


def test_log_spread_sign_follows_which_leg_is_higher():
    a = pd.Series([10.0])
    b = pd.Series([20.0])
    assert log_spread(a, b).iloc[0] < 0
    assert log_spread(b, a).iloc[0] > 0


def test_rolling_zscore_known_value_uses_sample_std():
    # window [1,2,3]: mean 2, sample std (ddof=1) 1 -> z of last point = 1
    z = rolling_zscore(pd.Series([1.0, 2.0, 3.0]), window=3)
    assert z.iloc[:2].isna().all()
    assert z.iloc[2] == pytest.approx(1.0)


def test_rolling_zscore_has_no_lookahead():
    base = pd.Series(np.sin(np.arange(40) / 3.0))
    changed = base.copy()
    changed.iloc[30:] = 99.0
    z_base = rolling_zscore(base, window=10)
    z_changed = rolling_zscore(changed, window=10)
    pd.testing.assert_series_equal(z_base.iloc[:30], z_changed.iloc[:30])


def test_rolling_zscore_constant_series_is_nan():
    z = rolling_zscore(pd.Series([5.0] * 10), window=4)
    assert z.isna().all()
