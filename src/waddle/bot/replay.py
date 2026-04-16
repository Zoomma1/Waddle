"""Replay primitives for offline fidelity testing of the bot.

This module provides a drop-in replacement for `BybitAdapter` and the `Clock`
protocol that reads all market data from local SQLite (populated via
`scripts/backfill_bybit.py`) instead of the live exchange. The engine / tick
loop cannot tell the difference — which is the whole point. Feeding the same
historical inputs into both the bot and `run_pairs_reversion` should yield the
same trades, within slippage noise.

Key properties:
- `ReplayClock.now()` is a manually-advanced pd.Timestamp. `sleep()` is a no-op
  so replay runs as fast as Python can iterate.
- `HistoricalBybitAdapter.fetch_recent_bars()` returns ONLY bars whose timestamp
  is <= clock.now() — strict no-lookahead semantics. As the clock advances, more
  bars become visible, mirroring the live case.
- Slippage math is applied identically to `BybitAdapter` so the paper fills have
  the same cost structure. Any deviation here would make the comparison against
  `run_pairs_reversion` meaningless (apples vs oranges).

The module is offline-only: no ccxt calls anywhere. If a symbol is missing from
the caches the adapter raises KeyError rather than falling back to the network.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from waddle.bot.config import FeesConfig
from waddle.bot.exchange import PaperFill
from waddle.data.storage import load_funding_rates, load_ohlcv_bybit


_LONG_SPREAD = "LONG_SPREAD"
_SHORT_SPREAD = "SHORT_SPREAD"


class ReplayClock:
    """Clock whose `now()` returns a fixed timestamp advanced manually via `advance`.

    `sleep()` is a no-op so the engine loop runs as fast as possible during replay.
    This satisfies the `Clock` Protocol from `waddle.bot.engine`.
    """

    def __init__(self, start: pd.Timestamp):
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        self._now = start

    def now(self) -> pd.Timestamp:
        return self._now

    def sleep(self, seconds: float) -> None:
        return None

    def advance(self, delta: pd.Timedelta) -> None:
        self._now = self._now + delta

    def set(self, ts: pd.Timestamp) -> None:
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        self._now = ts


class HistoricalBybitAdapter:
    """Offline, clock-aware Bybit adapter backed by SQLite caches.

    Matches the public surface of `BybitAdapter`: `fetch_recent_bars`,
    `fetch_latest_price`, `fetch_funding_payment`, `paper_execute_entry`,
    `paper_execute_exit`. The engine accepts either class interchangeably.

    `cached_ohlcv` must map each required symbol to a full-history DataFrame
    indexed by UTC timestamp with at least an `open/high/low/close/volume` set
    of columns. `cached_funding` maps each required symbol to a DataFrame
    indexed by UTC timestamp with a `funding_rate` column. Symbols that do
    not appear in funding are treated as zero-funding (fetch returns 0.0).
    """

    def __init__(
        self,
        clock: ReplayClock,
        fees_config: FeesConfig,
        cached_ohlcv: dict[str, pd.DataFrame],
        cached_funding: dict[str, pd.DataFrame],
    ):
        self._clock = clock
        self._fees = fees_config
        self._cached_ohlcv = cached_ohlcv
        self._cached_funding = cached_funding

    def fetch_recent_bars(
        self,
        symbol: str,
        limit: int = 200,
        timeframe: str = "1h",
    ) -> pd.DataFrame:
        """Return the last `limit` bars with index <= current replay time.

        Strict no-lookahead: any bar with timestamp > clock.now() is hidden.
        The `timeframe` argument is ignored (the cache is already at the
        configured timeframe — 1h for the SOL/AVAX baseline).
        """
        full = self._cached_ohlcv.get(symbol)
        if full is None:
            raise KeyError(
                f"HistoricalBybitAdapter: symbol {symbol!r} not in cached_ohlcv"
            )
        visible = full[full.index <= self._clock.now()]
        return visible.tail(limit)

    def fetch_latest_price(self, symbol: str) -> float:
        """Return the close price of the latest visible bar.

        This stands in for the live `fetch_ticker` mid price; during replay we
        only have OHLCV bars, so we use the close of the most recent completed
        bar (which is also what the engine uses for the spread calculation).
        """
        bars = self.fetch_recent_bars(symbol, limit=1)
        if bars.empty:
            raise RuntimeError(
                f"HistoricalBybitAdapter: no visible bars for {symbol} at {self._clock.now()}"
            )
        return float(bars["close"].iloc[-1])

    def fetch_funding_payment(
        self,
        symbol: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> float:
        """Sum cached funding settlements in (start_ts, end_ts].

        Same conservative convention as `BybitAdapter`: a trade entered at a
        settlement time does NOT pay that settlement, but one that exits at a
        settlement time DOES capture it. Missing symbol -> 0.0.
        """
        df = self._cached_funding.get(symbol)
        if df is None or df.empty:
            return 0.0

        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")

        mask = (df.index > start_ts) & (df.index <= end_ts)
        window = df.loc[mask]
        if window.empty:
            return 0.0
        return float(window["funding_rate"].sum())

    def paper_execute_entry(
        self,
        direction: str,
        price_a: float,
        price_b: float,
        notional_per_leg_usd: float,
    ) -> PaperFill:
        """Simulate entering a dollar-neutral pair trade — identical to `BybitAdapter`."""
        if notional_per_leg_usd <= 0:
            raise ValueError(
                f"notional_per_leg_usd must be > 0, got {notional_per_leg_usd}"
            )
        if price_a <= 0 or price_b <= 0:
            raise ValueError(
                f"prices must be > 0, got price_a={price_a}, price_b={price_b}"
            )

        slip = self._fees.slippage_per_side
        fee = self._fees.fee_per_side

        if direction == _LONG_SPREAD:
            fill_price_a = price_a * (1 + slip)
            fill_price_b = price_b * (1 - slip)
        elif direction == _SHORT_SPREAD:
            fill_price_a = price_a * (1 - slip)
            fill_price_b = price_b * (1 + slip)
        else:
            raise ValueError(
                f"direction must be '{_LONG_SPREAD}' or '{_SHORT_SPREAD}', got '{direction}'"
            )

        size_a_units = notional_per_leg_usd / fill_price_a
        size_b_units = notional_per_leg_usd / fill_price_b
        notional_usd = 2 * notional_per_leg_usd
        total_fee_usd = 2 * fee * notional_per_leg_usd

        return PaperFill(
            fill_price_a=fill_price_a,
            fill_price_b=fill_price_b,
            size_a_units=size_a_units,
            size_b_units=size_b_units,
            notional_usd=notional_usd,
            total_fee_usd=total_fee_usd,
        )

    def paper_execute_exit(
        self,
        direction: str,
        price_a: float,
        price_b: float,
        size_a_units: float,
        size_b_units: float,
    ) -> PaperFill:
        """Simulate closing a dollar-neutral pair trade — identical to `BybitAdapter`."""
        if size_a_units <= 0 or size_b_units <= 0:
            raise ValueError(
                f"sizes must be > 0, got size_a_units={size_a_units}, size_b_units={size_b_units}"
            )
        if price_a <= 0 or price_b <= 0:
            raise ValueError(
                f"prices must be > 0, got price_a={price_a}, price_b={price_b}"
            )

        slip = self._fees.slippage_per_side
        fee = self._fees.fee_per_side

        if direction == _LONG_SPREAD:
            fill_price_a = price_a * (1 - slip)
            fill_price_b = price_b * (1 + slip)
        elif direction == _SHORT_SPREAD:
            fill_price_a = price_a * (1 + slip)
            fill_price_b = price_b * (1 - slip)
        else:
            raise ValueError(
                f"direction must be '{_LONG_SPREAD}' or '{_SHORT_SPREAD}', got '{direction}'"
            )

        leg_a_notional = size_a_units * fill_price_a
        leg_b_notional = size_b_units * fill_price_b
        notional_usd = leg_a_notional + leg_b_notional
        total_fee_usd = fee * (leg_a_notional + leg_b_notional)

        return PaperFill(
            fill_price_a=fill_price_a,
            fill_price_b=fill_price_b,
            size_a_units=size_a_units,
            size_b_units=size_b_units,
            notional_usd=notional_usd,
            total_fee_usd=total_fee_usd,
        )


def load_replay_caches(
    symbols: list[str],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    db_path: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load full Bybit OHLCV + funding history for `symbols`, optionally clipped.

    Clipping is applied to BOTH endpoints and is inclusive. `start` should
    usually include the warmup period (e.g. replay_start - 14 days) so the
    first tick's z-score is already valid. Pass `None` for either bound to
    keep the raw database range.

    Returns `(ohlcv_by_symbol, funding_by_symbol)`. Symbols with no funding
    entries map to an empty DataFrame (the adapter treats that as zero).
    """
    ohlcv_map: dict[str, pd.DataFrame] = {}
    funding_map: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        if db_path is None:
            bars = load_ohlcv_bybit(symbol)
        else:
            bars = load_ohlcv_bybit(symbol, db_path=db_path)
        if start is not None:
            bars = bars[bars.index >= start]
        if end is not None:
            bars = bars[bars.index <= end]
        ohlcv_map[symbol] = bars

        if db_path is None:
            funding = load_funding_rates(symbol)
        else:
            funding = load_funding_rates(symbol, db_path=db_path)
        if start is not None:
            funding = funding[funding.index >= start]
        if end is not None:
            funding = funding[funding.index <= end]
        funding_map[symbol] = funding

    return ohlcv_map, funding_map
