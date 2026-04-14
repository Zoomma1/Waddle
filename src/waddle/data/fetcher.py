"""Fetch OHLCV bars from crypto exchanges via CCXT."""
from __future__ import annotations

import time

import ccxt
import pandas as pd

_MAX_BARS_PER_CALL = 1000


def _bars_to_df(raw: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(
        raw,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """Fetch the most recent `limit` OHLCV bars for a symbol (single CCXT call)."""
    exchange = getattr(ccxt, exchange_id)()
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    return _bars_to_df(raw)


def fetch_ohlcv_range(
    symbol: str,
    since_ms: int,
    until_ms: int,
    timeframe: str = "1h",
    exchange_id: str = "binance",
    pause_ms: int = 200,
) -> pd.DataFrame:
    """Fetch all OHLCV bars for a symbol in [since_ms, until_ms] with pagination.

    CCXT exchanges cap single calls (Binance = 1000 bars). We loop with the
    `since` cursor until we've covered the full range, pausing briefly between
    calls to stay polite with the API.
    """
    exchange = getattr(ccxt, exchange_id)()
    all_bars: list[list[float]] = []
    cursor = since_ms

    while cursor < until_ms:
        bars = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=cursor,
            limit=_MAX_BARS_PER_CALL,
        )
        if not bars:
            break
        all_bars.extend(bars)
        last_ts = bars[-1][0]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        if len(bars) < _MAX_BARS_PER_CALL:
            break
        time.sleep(pause_ms / 1000)

    df = _bars_to_df(all_bars)
    until_pd = pd.Timestamp(until_ms, unit="ms", tz="UTC")
    return df[df.index <= until_pd]
