"""Core polling loop for the paper trading bot.

A single synchronous loop that:
    1. reloads the adaptive params (hot reload) every tick
    2. fetches recent bars from the exchange
    3. computes log_spread + rolling z-score
    4. applies a bear-halt regime filter on BTC
    5. runs the pair mean-reversion state machine (one tick at a time)
    6. persists positions, trades, equity, and events via StateStore

Design constraints:
- `tick()` is pure: all dependencies are injected via arguments, no module-level state.
- PnL convention mirrors `waddle.strategy.pairs_reversion.run_pairs_reversion`:
  gross_pnl in USD = sign * leg_notional_usd * (exit_spread - entry_spread),
  where sign = +1 for LONG_SPREAD and -1 for SHORT_SPREAD.
- Fee model matches the backtest: 4 * leg_notional * (fee_per_side + slippage_per_side)
  per roundtrip (2 legs * 2 sides each).
- No directional stop-loss (ADR-003). Only mean_reversion, stop_time, or bot_stop.
- The loop never crashes on a transient exception — it logs an ERROR event and
  continues. KeyboardInterrupt triggers a graceful SHUTDOWN event and exits.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import pandas as pd

from waddle.bot.config import ActiveParams, BotConfig, load_active_params
from waddle.bot.exchange import BybitAdapter
from waddle.bot.state import Position, StateStore
from waddle.features.spread import log_spread, rolling_zscore


class Clock(Protocol):
    def now(self) -> pd.Timestamp: ...
    def sleep(self, seconds: float) -> None: ...


class RealClock:
    """Wall-clock implementation for production use."""

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC")

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def run_engine(
    state: StateStore,
    config: BotConfig,
    exchange: BybitAdapter,
    clock: Clock,
    active_params_path: Path = Path("config/active_params.json"),
) -> None:
    """Run the engine in a synchronous polling loop until interrupted.

    On KeyboardInterrupt (Ctrl+C) we log a SHUTDOWN event and return cleanly;
    open positions are LEFT open so a restart can pick them back up.

    Any unexpected exception in `tick()` is caught, logged as an ERROR event,
    and the loop continues — we do not crash on transient network errors.
    """
    state.log_event(
        clock.now(),
        "INFO",
        "STARTUP",
        {
            "mode": config.mode,
            "pair": f"{config.pair.symbol_a}/{config.pair.symbol_b}",
            "polling_interval_seconds": config.engine.polling_interval_seconds,
            "lookback_bars": config.engine.lookback_bars,
        },
    )
    try:
        while True:
            try:
                tick(state, config, exchange, clock, active_params_path)
            except Exception as e:  # noqa: BLE001 — we deliberately catch everything
                state.log_event(
                    clock.now(),
                    "ERROR",
                    "TICK_FAILED",
                    {"error": str(e), "type": type(e).__name__},
                )
            clock.sleep(config.engine.polling_interval_seconds)
    except KeyboardInterrupt:
        state.log_event(
            clock.now(),
            "INFO",
            "SHUTDOWN",
            {"reason": "keyboard_interrupt"},
        )


def tick(
    state: StateStore,
    config: BotConfig,
    exchange: BybitAdapter,
    clock: Clock,
    active_params_path: Path,
) -> None:
    """One iteration of the engine loop. Pure function — all deps injected."""
    now = clock.now()

    active_params = load_active_params(active_params_path)

    bars_a = exchange.fetch_recent_bars(
        config.pair.symbol_a, limit=config.engine.lookback_bars
    )
    bars_b = exchange.fetch_recent_bars(
        config.pair.symbol_b, limit=config.engine.lookback_bars
    )
    bars_btc = exchange.fetch_recent_bars(config.pair.reference_btc, limit=1)

    aligned = bars_a[["close"]].join(
        bars_b[["close"]], how="inner", lsuffix="_a", rsuffix="_b"
    )
    if len(aligned) < active_params.zscore_window + 1:
        state.log_event(
            now,
            "WARNING",
            "TICK_SKIPPED",
            {"reason": "insufficient_bars", "n": int(len(aligned))},
        )
        return

    spread = log_spread(aligned["close_a"], aligned["close_b"])
    zscore = rolling_zscore(spread, window=active_params.zscore_window)

    last_ts = aligned.index[-1]
    last_spread = float(spread.iloc[-1])
    last_z = float(zscore.iloc[-1])
    if pd.isna(last_z):
        state.log_event(
            now,
            "WARNING",
            "TICK_SKIPPED",
            {"reason": "nan_zscore"},
        )
        return

    if bars_btc is None or len(bars_btc) == 0:
        state.log_event(
            now,
            "WARNING",
            "TICK_SKIPPED",
            {"reason": "no_btc_reference_bar"},
        )
        return
    btc_close = float(bars_btc["close"].iloc[-1])
    bear_halt_ok = btc_close >= config.risk.bear_halt_btc_threshold

    position = state.get_open_position()

    if position is None:
        _try_open(
            state,
            config,
            exchange,
            now,
            last_ts,
            last_spread,
            last_z,
            active_params,
            bear_halt_ok,
        )
    else:
        _try_close(
            state,
            config,
            exchange,
            now,
            last_ts,
            last_spread,
            last_z,
            position,
            active_params,
        )

    _record_equity(state, config, now, last_spread)


def _try_open(
    state: StateStore,
    config: BotConfig,
    exchange: BybitAdapter,
    now: pd.Timestamp,
    last_ts: pd.Timestamp,
    last_spread: float,
    last_z: float,
    active_params: ActiveParams,
    bear_halt_ok: bool,
) -> None:
    """Decide whether to open a new position, and if so, execute + persist."""
    if not bear_halt_ok:
        if last_z >= active_params.entry_z or last_z <= -active_params.entry_z:
            state.log_event(
                now,
                "INFO",
                "ENTRY_SKIPPED",
                {
                    "reason": "bear_halt",
                    "z": last_z,
                    "bear_threshold": config.risk.bear_halt_btc_threshold,
                },
            )
        return

    direction: str | None = None
    if last_z >= active_params.entry_z:
        direction = "SHORT_SPREAD"
    elif last_z <= -active_params.entry_z:
        direction = "LONG_SPREAD"

    if direction is None:
        return

    try:
        price_a = exchange.fetch_latest_price(config.pair.symbol_a)
        price_b = exchange.fetch_latest_price(config.pair.symbol_b)
    except Exception as e:  # noqa: BLE001
        state.log_event(
            now,
            "WARNING",
            "ENTRY_SKIPPED",
            {"reason": "fetch_latest_failed", "error": str(e)},
        )
        return

    fill = exchange.paper_execute_entry(
        direction=direction,
        price_a=price_a,
        price_b=price_b,
        notional_per_leg_usd=config.risk.leg_notional_usd,
    )

    position_id = state.open_position(
        symbol_a=config.pair.symbol_a,
        symbol_b=config.pair.symbol_b,
        direction=direction,
        entry_ts=last_ts,
        entry_spread=last_spread,
        entry_z=last_z,
        entry_price_a=fill.fill_price_a,
        entry_price_b=fill.fill_price_b,
        size_a=fill.size_a_units,
        size_b=fill.size_b_units,
        active_params_snapshot=asdict(active_params),
    )

    state.log_event(
        now,
        "INFO",
        "ENTRY",
        {
            "position_id": position_id,
            "direction": direction,
            "entry_z": last_z,
            "entry_spread": last_spread,
            "entry_price_a": fill.fill_price_a,
            "entry_price_b": fill.fill_price_b,
            "notional": config.risk.leg_notional_usd,
            "entry_fee_usd": fill.total_fee_usd,
        },
    )


def _try_close(
    state: StateStore,
    config: BotConfig,
    exchange: BybitAdapter,
    now: pd.Timestamp,
    last_ts: pd.Timestamp,
    last_spread: float,
    last_z: float,
    position: Position,
    active_params: ActiveParams,
) -> None:
    """Decide whether to close the open position, and if so, execute + persist."""
    entry_ts = position.entry_ts
    if entry_ts.tzinfo is None:
        entry_ts = entry_ts.tz_localize("UTC")
    close_ts = last_ts
    if close_ts.tzinfo is None:
        close_ts = close_ts.tz_localize("UTC")
    bars_held = max(0, int((close_ts - entry_ts).total_seconds() // 3600))

    exit_reason: str | None = None
    if abs(last_z) <= active_params.exit_z:
        exit_reason = "mean_reversion"
    elif bars_held >= active_params.max_holding_bars:
        exit_reason = "stop_time"
    # ADR-003: no directional stop-loss.

    if exit_reason is None:
        return

    try:
        price_a = exchange.fetch_latest_price(config.pair.symbol_a)
        price_b = exchange.fetch_latest_price(config.pair.symbol_b)
    except Exception as e:  # noqa: BLE001
        state.log_event(
            now,
            "WARNING",
            "EXIT_SKIPPED",
            {"reason": "fetch_latest_failed", "error": str(e)},
        )
        return

    exit_fill = exchange.paper_execute_exit(
        direction=position.direction,
        price_a=price_a,
        price_b=price_b,
        size_a_units=position.size_a,
        size_b_units=position.size_b,
    )

    sign = 1 if position.direction == "LONG_SPREAD" else -1
    leg_notional = config.risk.leg_notional_usd
    pnl_gross = sign * leg_notional * (last_spread - position.entry_spread)

    fee_per_side = config.fees.fee_per_side
    slip_per_side = config.fees.slippage_per_side
    total_fees_usd = 4 * leg_notional * fee_per_side
    total_slip_usd = 4 * leg_notional * slip_per_side
    pnl_fees = -(total_fees_usd + total_slip_usd)

    try:
        rate_a = exchange.fetch_funding_payment(
            config.pair.symbol_a, position.entry_ts, last_ts
        )
        rate_b = exchange.fetch_funding_payment(
            config.pair.symbol_b, position.entry_ts, last_ts
        )
        pnl_funding = sign * leg_notional * (rate_b - rate_a)
    except Exception as e:  # noqa: BLE001
        pnl_funding = 0.0
        state.log_event(
            now,
            "WARNING",
            "FUNDING_FETCH_FAILED",
            {"error": str(e)},
        )

    pnl_net = pnl_gross + pnl_fees + pnl_funding

    if position.id is None:
        raise RuntimeError("Cannot close a position whose id is None")

    state.close_position(
        position_id=position.id,
        exit_ts=last_ts,
        exit_spread=last_spread,
        exit_z=last_z,
        exit_price_a=exit_fill.fill_price_a,
        exit_price_b=exit_fill.fill_price_b,
        pnl_gross=pnl_gross,
        pnl_fees=pnl_fees,
        pnl_funding=pnl_funding,
        pnl_net=pnl_net,
        bars_held=bars_held,
        exit_reason=exit_reason,
    )

    state.log_event(
        now,
        "INFO",
        "EXIT",
        {
            "position_id": position.id,
            "exit_reason": exit_reason,
            "bars_held": bars_held,
            "exit_z": last_z,
            "exit_spread": last_spread,
            "pnl_gross": pnl_gross,
            "pnl_fees": pnl_fees,
            "pnl_funding": pnl_funding,
            "pnl_net": pnl_net,
        },
    )


def _record_equity(
    state: StateStore,
    config: BotConfig,
    now: pd.Timestamp,
    last_spread: float,
) -> None:
    """Compute and persist the current equity snapshot.

    NAV = initial_cash + realized_pnl + unrealized_pnl
    cash = initial_cash + realized_pnl (excludes unrealized)
    """
    trades = state.list_trades(limit=10_000)
    realized_pnl = sum(t.pnl_net for t in trades)

    position = state.get_open_position()
    if position is not None:
        sign = 1 if position.direction == "LONG_SPREAD" else -1
        unrealized_pnl = sign * config.risk.leg_notional_usd * (
            last_spread - position.entry_spread
        )
        open_count = 1
    else:
        unrealized_pnl = 0.0
        open_count = 0

    initial_cash = config.risk.total_notional_usd
    nav = initial_cash + realized_pnl + unrealized_pnl
    cash = initial_cash + realized_pnl

    state.record_equity(
        ts=now,
        nav=nav,
        cash=cash,
        open_position_count=open_count,
        unrealized_pnl=unrealized_pnl,
    )
