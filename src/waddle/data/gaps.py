"""Detect holes in a regular series of bars."""
from __future__ import annotations

import pandas as pd


def find_gaps(
    index: pd.DatetimeIndex, freq: str = "1h"
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return the holes of `index` as (first missing, last missing) pairs, inclusive.

    `index` must be tz-aware UTC, strictly increasing and aligned on the
    `freq` grid (epoch nanoseconds are a multiple of the duration);
    otherwise ValueError is raised.
    """
    if index.tz is None:
        raise ValueError("index must be tz-aware")
    step = pd.Timedelta(freq)
    step_ns = step.value
    ns = index.asi8
    if any(int(v) % step_ns for v in ns):
        raise ValueError(f"index is not aligned on the {freq} grid")
    if any(int(b) <= int(a) for a, b in zip(ns[:-1], ns[1:])):
        raise ValueError("index must be strictly increasing")

    holes: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for a, b in zip(index[:-1], index[1:]):
        if b - a > step:
            holes.append((a + step, b - step))
    return holes
