"""Sanity check: fetch 100h of BTC/USDT and ETH/USDT, print tails and spread."""
from __future__ import annotations

from waddle.data.fetcher import fetch_ohlcv


def main() -> None:
    frames = {}
    for symbol in ("BTC/USDT", "ETH/USDT"):
        df = fetch_ohlcv(symbol, timeframe="1h", limit=100)
        frames[symbol] = df
        print(f"\n=== {symbol} ===")
        print(df.tail())
        print(f"shape: {df.shape}")

    btc_close = frames["BTC/USDT"]["close"]
    eth_close = frames["ETH/USDT"]["close"]
    ratio = btc_close / eth_close
    print("\n=== BTC/ETH close ratio (last 5) ===")
    print(ratio.tail())
    print(f"mean: {ratio.mean():.4f}  std: {ratio.std():.4f}")


if __name__ == "__main__":
    main()
