"""Backfill 18 months of hourly majors into the local SQLite db.

Extended from 6 to 18 months and to 5 symbols after ADR-006 showed that 6
months of BTC/ETH gave no statistically robust signal. The additional symbols
enable testing alt/ETH and L1/L1 pairs alongside the BTC/ETH baseline.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from waddle.data.fetcher import fetch_ohlcv_range
from waddle.data.storage import DEFAULT_DB, save_ohlcv

LOOKBACK_DAYS = 540
SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "AVAX/USDT")
TIMEFRAME = "1h"


def main() -> None:
    until = datetime.now(timezone.utc)
    since = until - timedelta(days=LOOKBACK_DAYS)
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(until.timestamp() * 1000)

    print(f"Backfilling {LOOKBACK_DAYS}d of {TIMEFRAME} data into {DEFAULT_DB}")
    print(f"  {since.isoformat()}  ->  {until.isoformat()}")

    for symbol in SYMBOLS:
        print(f"\n=> {symbol}")
        df = fetch_ohlcv_range(symbol, since_ms=since_ms, until_ms=until_ms, timeframe=TIMEFRAME)
        written = save_ohlcv(df, symbol=symbol)
        print(f"   fetched {len(df)} bars  |  saved {written} rows")
        if len(df) > 0:
            print(f"   range: {df.index[0]}  ->  {df.index[-1]}")


if __name__ == "__main__":
    main()
