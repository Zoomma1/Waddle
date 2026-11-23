"""Validate OHLCV DataFrames."""
from __future__ import annotations

import pandas as pd

_COLUMNS = ["open", "high", "low", "close", "volume"]


def _index_is_valid(index: pd.Index) -> bool:
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        return False
    if str(index.tz) != "UTC":
        return False
    return bool(index.is_monotonic_increasing and index.is_unique)


def validate_ohlcv(df: pd.DataFrame) -> list[str]:
    """Return the problem codes found in an OHLCV frame (empty list = valid)."""
    if any(c not in df.columns for c in _COLUMNS):
        return ["missing-columns"]

    problems: list[str] = []
    if not _index_is_valid(df.index):
        problems.append("bad-index")

    data = df[_COLUMNS]
    if bool(data.isna().to_numpy().any()):
        problems.append("nan-values")

    o, h, l, c, v = (data[col] for col in _COLUMNS)
    body_high = pd.concat([o, c], axis=1).max(axis=1, skipna=False)
    body_low = pd.concat([o, c], axis=1).min(axis=1, skipna=False)
    if bool((h < body_high).any()):
        problems.append("high-below-body")
    if bool((l > body_low).any()):
        problems.append("low-above-body")
    if bool((h < l).any()):
        problems.append("high-below-low")
    if bool((v < 0).any()):
        problems.append("negative-volume")
    return problems
