"""Rolling correlation on log returns."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_log_return_correlation(
    price_a: pd.Series,
    price_b: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling Pearson correlation of log returns over a window.

    Why log returns and not prices: two assets that both drift upward can show
    a Pearson price correlation near 1.0 while their returns are nearly
    uncorrelated — the drift hides the noise. For pairs trading, what matters
    is whether A and B *move together day-to-day*, which is exactly the
    correlation of their returns.
    """
    log_ret_a = np.log(price_a).diff()
    log_ret_b = np.log(price_b).diff()
    return log_ret_a.rolling(window=window).corr(log_ret_b)
