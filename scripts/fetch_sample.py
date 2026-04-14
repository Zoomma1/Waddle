"""Fetch BTC+ETH OHLCV, compute log spread + rolling z-score, print signals."""
from __future__ import annotations

from waddle.data.fetcher import fetch_ohlcv
from waddle.features.spread import log_spread, rolling_zscore

ROLLING_WINDOW = 24
SIGNAL_THRESHOLD = 1.5


def main() -> None:
    btc = fetch_ohlcv("BTC/USDT", timeframe="1h", limit=200)
    eth = fetch_ohlcv("ETH/USDT", timeframe="1h", limit=200)

    spread = log_spread(btc["close"], eth["close"])
    zscore = rolling_zscore(spread, window=ROLLING_WINDOW)

    print("\n=== log spread (last 5) ===")
    print(spread.tail())

    print(f"\n=== rolling z-score (window={ROLLING_WINDOW}h, last 5) ===")
    print(zscore.tail())

    valid = zscore.dropna()
    print("\n=== stats on rolling z-score (dropna) ===")
    print(f"  count: {len(valid)}")
    print(f"  min:   {valid.min():.3f}")
    print(f"  max:   {valid.max():.3f}")
    print(f"  mean:  {valid.mean():.3f}")
    print(f"  std:   {valid.std():.3f}")

    extremes = zscore[zscore.abs() > SIGNAL_THRESHOLD]
    print(f"\n=== {len(extremes)} potential signals |z| > {SIGNAL_THRESHOLD} ===")
    if len(extremes) > 0:
        for ts, z in extremes.items():
            direction = "SHORT spread" if z > 0 else "LONG spread"
            print(f"  {ts}  z={z:+.3f}  -> {direction}")
    else:
        print("  (none — spread stayed within normal bounds)")


if __name__ == "__main__":
    main()
