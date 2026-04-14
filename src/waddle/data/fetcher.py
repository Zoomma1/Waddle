"""Fetch OHLCV bars from crypto exchanges via CCXT."""
from __future__ import annotations

import ccxt
import pandas as pd


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """Fetch OHLCV bars for a symbol from a CCXT-supported exchange.

    Returns a DataFrame indexed by UTC timestamp with columns:
    open, high, low, close, volume.
    """
    exchange = getattr(ccxt, exchange_id)()
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(
        raw,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    return df
