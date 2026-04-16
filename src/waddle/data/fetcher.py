"""Fetch OHLCV bars and funding rates from crypto exchanges via CCXT."""
from __future__ import annotations

import time

import ccxt
import pandas as pd

_MAX_BARS_PER_CALL = 1000
_MAX_FUNDING_PER_CALL = 200


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


def _bybit_exchange() -> ccxt.bybit:
    """Bybit client configured for USDT-margined linear perpetuals (swap).

    Symbol format on this client is `BASE/USDT:USDT` (CCXT unified notation
    for a linear perpetual settled in USDT). `enableRateLimit=True` makes
    CCXT throttle requests automatically based on the exchange's rate limit
    profile, so we don't hammer the 120 req/min ceiling.
    """
    return ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })


def fetch_ohlcv_range_bybit(
    symbol: str,
    since_ms: int,
    until_ms: int,
    timeframe: str = "1h",
    pause_ms: int = 150,
) -> pd.DataFrame:
    """Fetch all OHLCV bars for a Bybit LINEAR PERPETUAL in [since_ms, until_ms].

    Bybit's linear perpetuals are distinct from spot: they have no expiry,
    price tracks spot via a funding mechanism, and fees are typically lower
    (ADR-004). This function targets those contracts, not spot.

    Symbol format expected: `BTC/USDT:USDT` (CCXT unified notation for a
    USDT-margined perp). The settlement suffix `:USDT` tells CCXT which
    product type to hit.
    """
    exchange = _bybit_exchange()
    exchange.load_markets()
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
        # Bybit occasionally returns fewer bars than requested even when more
        # data exists downstream. Do not break on undersized batches — only
        # trust an empty return or a stalled cursor as "end of data".
        time.sleep(pause_ms / 1000)

    seen_ts: set[int] = set()
    deduped: list[list[float]] = []
    for row in all_bars:
        ts = int(row[0])
        if ts in seen_ts:
            continue
        seen_ts.add(ts)
        deduped.append(row)

    df = _bars_to_df(deduped)
    until_pd = pd.Timestamp(until_ms, unit="ms", tz="UTC")
    return df[df.index <= until_pd]


def fetch_funding_rate_history(
    symbol: str,
    since_ms: int,
    until_ms: int,
    pause_ms: int = 150,
) -> pd.DataFrame:
    """Fetch paginated funding rate history for a Bybit linear perpetual.

    Returns a DataFrame indexed by UTC timestamp of the funding settlement
    with a single column `funding_rate` (float, e.g. 0.0001 = 0.01%).

    Bybit settles funding every 8h (three times per day at 00:00, 08:00 and
    16:00 UTC). Each row represents a single settlement event, and the rate
    shown is the fraction of notional paid by longs to shorts for that period
    (negative = shorts pay longs). For a dollar-neutral long-A/short-B trade,
    the funding PnL at a given settlement is `funding_B - funding_A` on the
    notional held through the settlement.
    """
    exchange = _bybit_exchange()
    exchange.load_markets()
    all_rates: list[dict] = []
    cursor = since_ms
    last_ts: int | None = None

    while cursor < until_ms:
        batch = exchange.fetch_funding_rate_history(
            symbol,
            since=cursor,
            limit=_MAX_FUNDING_PER_CALL,
        )
        if not batch:
            break
        all_rates.extend(batch)
        batch_last = int(batch[-1]["timestamp"])
        if last_ts is not None and batch_last <= last_ts:
            break
        last_ts = batch_last
        cursor = batch_last + 1
        if len(batch) < _MAX_FUNDING_PER_CALL:
            break
        time.sleep(pause_ms / 1000)

    if not all_rates:
        return pd.DataFrame(columns=["funding_rate"]).rename_axis("timestamp")

    df = pd.DataFrame([
        {
            "timestamp": pd.to_datetime(int(r["timestamp"]), unit="ms", utc=True),
            "funding_rate": float(r["fundingRate"]),
        }
        for r in all_rates
    ]).drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()

    until_pd = pd.Timestamp(until_ms, unit="ms", tz="UTC")
    since_pd = pd.Timestamp(since_ms, unit="ms", tz="UTC")
    return df[(df.index >= since_pd) & (df.index <= until_pd)]
