"""Exchange adapter for the paper trading bot.

Thin wrapper around `ccxt.bybit` providing:
- Read-only primitives: recent OHLCV bars, latest price, funding rate history
- Paper execution primitives: simulated adverse-slippage fills for pair trades

V1 is paper-only — no live order submission. The `enable_live_execution` flag
is plumbed through for forward-compat but always defaults to False and, if
toggled, the current methods do not actually transmit orders. V2 will add
dedicated `live_execute_*` methods.

Slippage / fees convention (MUST match backtest semantics — see
`src/waddle/strategy/pairs_reversion.py` where
`cost_per_trade = 4 * (fee_per_side + slippage_per_side)`):

- Slippage is applied to the fill price in an adverse direction:
  * Buying  -> fill = price * (1 + slippage_per_side)
  * Selling -> fill = price * (1 - slippage_per_side)
- Fees are NOT baked into the fill price; they are returned in
  `PaperFill.total_fee_usd` and deducted by the engine at trade close.
- Roundtrip slippage impact = 4 * slippage_per_side * notional  (2 legs * 2 sides)
- Roundtrip fee impact       = 4 * fee_per_side       * notional
- Sum = 4 * (fee + slip) * notional — identical to the backtest cost model.

Symbol convention: `BASE/USDT:USDT` (CCXT unified notation for a USDT-margined
linear perpetual on Bybit). The `:USDT` settlement suffix is required.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import ccxt
import pandas as pd

from waddle.bot.config import ExchangeConfig, FeesConfig


_LOGGER = logging.getLogger(__name__)

_LONG_SPREAD = "LONG_SPREAD"
_SHORT_SPREAD = "SHORT_SPREAD"


@dataclass
class PaperFill:
    """Result of a simulated paper fill for a dollar-neutral pair trade (both legs).

    `size_a_units` and `size_b_units` are expressed in the native unit of each
    asset (e.g. SOL, AVAX), NOT in USD. `notional_usd` is the total dollar
    exposure across both legs at the fill prices used.
    """
    fill_price_a: float
    fill_price_b: float
    size_a_units: float
    size_b_units: float
    notional_usd: float
    total_fee_usd: float


class BybitAdapter:
    """Read-only + paper-fill adapter wrapping `ccxt.bybit` in swap mode."""

    def __init__(
        self,
        exchange_config: ExchangeConfig,
        fees_config: FeesConfig,
        enable_live_execution: bool = False,
    ):
        if exchange_config.name != "bybit":
            raise ValueError(
                f"BybitAdapter only supports name='bybit', got '{exchange_config.name}'"
            )
        if enable_live_execution:
            raise NotImplementedError(
                "Live execution is disabled in V1 — paper mode only. See ADR-009 and"
                " design doc §'Anti-objectifs (V1)'."
            )

        self._exchange_config = exchange_config
        self._fees_config = fees_config
        self._client = ccxt.bybit({
            "enableRateLimit": True,
            "options": {"defaultType": exchange_config.type},
        })
        if exchange_config.testnet:
            self._client.set_sandbox_mode(True)

    def fetch_recent_bars(
        self,
        symbol: str,
        limit: int = 200,
        timeframe: str = "1h",
    ) -> pd.DataFrame:
        """Fetch the last `limit` COMPLETED bars for `symbol`.

        Over-fetches by 2 internally so that after stripping the
        currently-forming bar (if ccxt returns it), we still return `limit`
        closed bars. Without the buffer, `limit=1` could return 0 rows
        whenever the only bar is in-progress — which happens most of the time
        on short polling intervals.
        """
        raw = self._client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit + 2)
        if not raw:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
            ).rename_axis("timestamp")

        df = pd.DataFrame(
            raw,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")

        bar_span_ms = self._client.parse_timeframe(timeframe) * 1000
        now_ms = self._client.milliseconds()
        last_bar_open_ms = int(df.index[-1].timestamp() * 1000)
        if now_ms < last_bar_open_ms + bar_span_ms:
            df = df.iloc[:-1]

        return df.tail(limit)

    def fetch_latest_price(self, symbol: str) -> float:
        """Current mid price from `fetch_ticker`.

        Used at execution time for a more realistic fill reference than the
        previous bar close. Returns `(bid + ask) / 2` when both sides are
        quoted, else falls back to `last`. Raises if ccxt returns no usable
        price (the engine will catch and skip the tick).
        """
        ticker = self._client.fetch_ticker(symbol)
        bid = ticker.get("bid")
        ask = ticker.get("ask")
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return float((bid + ask) / 2)
        last = ticker.get("last")
        if last is not None and last > 0:
            return float(last)
        raise RuntimeError(f"fetch_ticker({symbol}) returned no usable price: {ticker}")

    def fetch_funding_payment(
        self,
        symbol: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> float:
        """Sum of funding rates settled strictly within (start_ts, end_ts].

        Bybit perpetuals settle funding every 8h at 00:00, 08:00, 16:00 UTC.
        This returns the CUMULATIVE RATE (a fraction, e.g. 0.0003 = 0.03%).
        The caller applies sign + notional to turn it into a USD PnL impact:
        a long leg PAYS when funding > 0, a short leg RECEIVES it.

        ADR-009 flagged funding as a second-order factor (+0.18%/year on the
        SOL/AVAX pair in the same regime), so the contract here is "best
        effort": on any ccxt failure we log a warning and return 0.0 rather
        than crashing the bot.
        """
        try:
            start_ms = int(start_ts.tz_convert("UTC").timestamp() * 1000) \
                if start_ts.tzinfo is not None else int(start_ts.tz_localize("UTC").timestamp() * 1000)
            end_ms = int(end_ts.tz_convert("UTC").timestamp() * 1000) \
                if end_ts.tzinfo is not None else int(end_ts.tz_localize("UTC").timestamp() * 1000)

            batch = self._client.fetch_funding_rate_history(
                symbol,
                since=start_ms,
                limit=200,
            )
        except Exception as exc:
            _LOGGER.warning(
                "fetch_funding_payment(%s) failed, returning 0.0: %s", symbol, exc
            )
            return 0.0

        if not batch:
            return 0.0

        total = 0.0
        for row in batch:
            ts_ms = int(row["timestamp"])
            if ts_ms <= start_ms:
                continue
            if ts_ms > end_ms:
                continue
            total += float(row["fundingRate"])
        return total

    def paper_execute_entry(
        self,
        direction: str,
        price_a: float,
        price_b: float,
        notional_per_leg_usd: float,
    ) -> PaperFill:
        """Simulate entering a dollar-neutral pair trade.

        LONG_SPREAD = long A + short B -> buy A (adverse: pay up), sell B (adverse: get hit).
        SHORT_SPREAD = short A + long B -> sell A (adverse: get hit), buy B (adverse: pay up).

        Returns a `PaperFill` whose fill prices include slippage but NOT fees.
        Fees are accumulated in `total_fee_usd` for the engine to debit at close.
        """
        if notional_per_leg_usd <= 0:
            raise ValueError(f"notional_per_leg_usd must be > 0, got {notional_per_leg_usd}")
        if price_a <= 0 or price_b <= 0:
            raise ValueError(f"prices must be > 0, got price_a={price_a}, price_b={price_b}")

        slip = self._fees_config.slippage_per_side
        fee = self._fees_config.fee_per_side

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
        """Simulate closing a dollar-neutral pair trade.

        `direction` here is the ORIGINAL entry direction (the trade being
        closed). Closing a LONG_SPREAD means selling A (adverse: get hit) and
        buying B (adverse: pay up) — the opposite slippage orientation of the
        entry. Same logic mirrored for SHORT_SPREAD.

        `size_a_units` / `size_b_units` come from the original entry fill and
        are kept constant — we flatten the same quantity we opened. The exit
        notional is re-computed from the fill prices and is typically slightly
        different from the entry notional (that difference is the gross PnL).
        """
        if size_a_units <= 0 or size_b_units <= 0:
            raise ValueError(
                f"sizes must be > 0, got size_a_units={size_a_units}, size_b_units={size_b_units}"
            )
        if price_a <= 0 or price_b <= 0:
            raise ValueError(f"prices must be > 0, got price_a={price_a}, price_b={price_b}")

        slip = self._fees_config.slippage_per_side
        fee = self._fees_config.fee_per_side

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
