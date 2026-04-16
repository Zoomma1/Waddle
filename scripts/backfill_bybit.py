"""Backfill 540d of hourly OHLCV + funding rates from Bybit linear perpetuals.

The existing `backfill.py` targets Binance spot for historical baselining.
This script targets Bybit linear perpetuals — the actual deployment venue per
[[ADR-004]]. We fetch both price bars and funding rate history so the walk-
forward can compute a realistic PnL with funding integrated.

Runs two fetches per symbol:
    1. OHLCV bars (1h)           -> ohlcv_bybit table
    2. Funding rate history      -> funding_rates table

Bybit has a dedicated USDT perpetual for each of these assets, listed well
before our data horizon, so 540 days should return a full contiguous series.
If a symbol returns less than expected, the per-symbol summary at the bottom
will show the actual covered window so the caller can decide whether to
shrink the backtest range.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from waddle.data.fetcher import fetch_funding_rate_history, fetch_ohlcv_range_bybit
from waddle.data.storage import (
    DEFAULT_DB,
    save_funding_rates,
    save_ohlcv_bybit,
)

LOOKBACK_DAYS = 540
SYMBOLS = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "AVAX/USDT:USDT",
)
TIMEFRAME = "1h"


def main() -> None:
    until = datetime.now(timezone.utc)
    since = until - timedelta(days=LOOKBACK_DAYS)
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(until.timestamp() * 1000)

    print("=" * 92)
    print(f"Bybit linear perpetual backfill -> {DEFAULT_DB}")
    print("=" * 92)
    print(f"Window: {since.isoformat()}  ->  {until.isoformat()}  ({LOOKBACK_DAYS}d)")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Symbols: {', '.join(SYMBOLS)}")

    summary: list[tuple[str, int, str, str, int, str, str]] = []

    for symbol in SYMBOLS:
        print(f"\n=> {symbol}")

        print("   [OHLCV] fetching...")
        bars_df = fetch_ohlcv_range_bybit(
            symbol,
            since_ms=since_ms,
            until_ms=until_ms,
            timeframe=TIMEFRAME,
        )
        bars_written = save_ohlcv_bybit(bars_df, symbol=symbol)
        ohlcv_first = bars_df.index[0].isoformat() if len(bars_df) else "-"
        ohlcv_last = bars_df.index[-1].isoformat() if len(bars_df) else "-"
        print(f"          {len(bars_df)} bars fetched, {bars_written} rows saved")
        if len(bars_df) > 0:
            print(f"          range: {ohlcv_first}  ->  {ohlcv_last}")
        else:
            print("          !! no bars returned")

        print("   [FUNDING] fetching...")
        fund_df = fetch_funding_rate_history(
            symbol,
            since_ms=since_ms,
            until_ms=until_ms,
        )
        fund_written = save_funding_rates(fund_df, symbol=symbol)
        fund_first = fund_df.index[0].isoformat() if len(fund_df) else "-"
        fund_last = fund_df.index[-1].isoformat() if len(fund_df) else "-"
        print(f"          {len(fund_df)} settlements fetched, {fund_written} rows saved")
        if len(fund_df) > 0:
            print(f"          range: {fund_first}  ->  {fund_last}")
            mn = fund_df["funding_rate"].min()
            mx = fund_df["funding_rate"].max()
            mean = fund_df["funding_rate"].mean()
            print(
                f"          stats:  min={mn * 100:+.4f}%  "
                f"mean={mean * 100:+.4f}%  max={mx * 100:+.4f}% (per 8h)"
            )
        else:
            print("          !! no funding rates returned")

        summary.append((
            symbol,
            len(bars_df),
            ohlcv_first,
            ohlcv_last,
            len(fund_df),
            fund_first,
            fund_last,
        ))

    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)
    header = f"{'symbol':<22} {'ohlcv':>7} {'ohlcv first':<20} {'ohlcv last':<20} {'fund':>5}"
    print(header)
    print("-" * len(header))
    for row in summary:
        sym, n_o, f_o, l_o, n_f, _f_f, _f_l = row
        print(f"{sym:<22} {n_o:>7} {f_o:<20} {l_o:<20} {n_f:>5}")


if __name__ == "__main__":
    main()
