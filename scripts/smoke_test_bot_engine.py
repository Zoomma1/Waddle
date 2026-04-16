"""Smoke test for waddle.bot.engine.tick().

Runs 5 offline scenarios against a real StateStore + fake exchange + fake clock:
    1. No-signal tick (|z| below entry threshold) -> no open position
    2. Entry tick (|z| >= entry_z) -> 1 open position, SHORT_SPREAD
    3. Exit tick (|z| <= exit_z while in position) -> 0 open, 1 closed trade
    4. Bear halt (BTC < threshold) -> entry blocked, event logged
    5. Hot reload (rewrite active_params.json between ticks) -> new params take effect

Run with: uv run scripts/smoke_test_bot_engine.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from waddle.bot.config import (
    BotConfig,
    EngineConfig,
    ExchangeConfig,
    FeesConfig,
    PairConfig,
    RiskConfig,
    write_active_params,
)
from waddle.bot.engine import tick
from waddle.bot.exchange import PaperFill
from waddle.bot.state import StateStore


SMOKE_DB = Path("data/waddle_bot_engine_smoke.db")
SMOKE_PARAMS_PATH = Path("data/waddle_bot_engine_smoke_params.json")


def _hr(title: str) -> None:
    print(f"\n--- {title} ---")


# === Fakes ==================================================================

class FakeClock:
    def __init__(self, fixed_ts: pd.Timestamp):
        self._ts = fixed_ts

    def now(self) -> pd.Timestamp:
        return self._ts

    def sleep(self, seconds: float) -> None:
        pass


class FakeBybitAdapter:
    """Minimal fake — returns preset bars, prices, funding. No network."""

    def __init__(self, symbol_a: str, symbol_b: str, symbol_btc: str):
        self._symbol_a = symbol_a
        self._symbol_b = symbol_b
        self._symbol_btc = symbol_btc
        self.bars_a: pd.DataFrame | None = None
        self.bars_b: pd.DataFrame | None = None
        self.bars_btc: pd.DataFrame | None = None
        self.price_a: float = 100.0
        self.price_b: float = 25.0
        self.price_btc: float = 80000.0
        self.funding_a: float = 0.0
        self.funding_b: float = 0.0

    def fetch_recent_bars(self, symbol: str, limit: int = 200, timeframe: str = "1h") -> pd.DataFrame:
        if symbol == self._symbol_btc:
            assert self.bars_btc is not None, "bars_btc not set on fake"
            return self.bars_btc.tail(limit)
        if symbol == self._symbol_a:
            assert self.bars_a is not None, "bars_a not set on fake"
            return self.bars_a.tail(limit)
        if symbol == self._symbol_b:
            assert self.bars_b is not None, "bars_b not set on fake"
            return self.bars_b.tail(limit)
        raise ValueError(f"FakeBybitAdapter: unknown symbol {symbol}")

    def fetch_latest_price(self, symbol: str) -> float:
        if symbol == self._symbol_btc:
            return self.price_btc
        if symbol == self._symbol_a:
            return self.price_a
        if symbol == self._symbol_b:
            return self.price_b
        raise ValueError(f"FakeBybitAdapter: unknown symbol {symbol}")

    def fetch_funding_payment(
        self, symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp
    ) -> float:
        if symbol == self._symbol_a:
            return self.funding_a
        if symbol == self._symbol_b:
            return self.funding_b
        return 0.0

    def paper_execute_entry(
        self,
        direction: str,
        price_a: float,
        price_b: float,
        notional_per_leg_usd: float,
    ) -> PaperFill:
        return PaperFill(
            fill_price_a=price_a,
            fill_price_b=price_b,
            size_a_units=notional_per_leg_usd / price_a,
            size_b_units=notional_per_leg_usd / price_b,
            notional_usd=2 * notional_per_leg_usd,
            total_fee_usd=0.20,
        )

    def paper_execute_exit(
        self,
        direction: str,
        price_a: float,
        price_b: float,
        size_a_units: float,
        size_b_units: float,
    ) -> PaperFill:
        return PaperFill(
            fill_price_a=price_a,
            fill_price_b=price_b,
            size_a_units=size_a_units,
            size_b_units=size_b_units,
            notional_usd=size_a_units * price_a + size_b_units * price_b,
            total_fee_usd=0.20,
        )


# === Test config & params helpers ==========================================

SYMBOL_A = "SOL/USDT:USDT"
SYMBOL_B = "AVAX/USDT:USDT"
SYMBOL_BTC = "BTC/USDT:USDT"

ZSCORE_WINDOW = 168
LEG_NOTIONAL = 500.0
TOTAL_NOTIONAL = 1000.0


def _make_config() -> BotConfig:
    return BotConfig(
        pair=PairConfig(symbol_a=SYMBOL_A, symbol_b=SYMBOL_B, reference_btc=SYMBOL_BTC),
        exchange=ExchangeConfig(name="bybit", type="swap", testnet=False),
        fees=FeesConfig(fee_per_side=0.0002, slippage_per_side=0.0002),
        risk=RiskConfig(
            total_notional_usd=TOTAL_NOTIONAL,
            leg_notional_usd=LEG_NOTIONAL,
            bear_halt_btc_threshold=60000.0,
        ),
        engine=EngineConfig(polling_interval_seconds=60, lookback_bars=200),
        mode="paper",
        state_db=str(SMOKE_DB),
    )


def _write_params(
    entry_z: float = 2.5,
    exit_z: float = 1.0,
    max_holding_bars: int = 48,
    zscore_window: int = ZSCORE_WINDOW,
) -> None:
    write_active_params(
        entry_z=entry_z,
        exit_z=exit_z,
        max_holding_bars=max_holding_bars,
        zscore_window=zscore_window,
        train_calmar=1.0,
        train_sharpe=1.0,
        train_window_days=90,
        selected_by="smoke_test",
        json_path=SMOKE_PARAMS_PATH,
    )


# === Synthetic bar generators ==============================================

_RNG = np.random.default_rng(42)


def _build_bars(
    n_bars: int,
    last_ts: pd.Timestamp,
    close_a: np.ndarray,
    close_b: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build two aligned OHLCV DataFrames with the given close arrays."""
    index = pd.date_range(end=last_ts, periods=n_bars, freq="h", tz="UTC")
    df_a = pd.DataFrame(
        {
            "open": close_a,
            "high": close_a,
            "low": close_a,
            "close": close_a,
            "volume": np.ones(n_bars),
        },
        index=index,
    )
    df_b = pd.DataFrame(
        {
            "open": close_b,
            "high": close_b,
            "low": close_b,
            "close": close_b,
            "volume": np.ones(n_bars),
        },
        index=index,
    )
    df_a.index.name = "timestamp"
    df_b.index.name = "timestamp"
    return df_a, df_b


def _build_bars_with_target_z(
    n_bars: int,
    last_ts: pd.Timestamp,
    zscore_window: int,
    target_last_z: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct bars so that the last z-score of log_spread equals target_last_z.

    Strategy: build a neutral random spread for all bars except the last. The
    last spread value is then computed analytically from the rolling mean/std
    of the prior (window-1) + itself so that the final z-score equals target.
    Since we control the series, we also set fixed prices: price_b constant,
    price_a derived from price_b * exp(spread).
    """
    noise = _RNG.normal(0.0, 0.01, size=n_bars - 1)
    hist_spread = 1.0 + noise  # centred around 1.0

    prior_window = hist_spread[-(zscore_window - 1):]
    prior_mean = float(np.mean(prior_window))
    prior_std = float(np.std(prior_window, ddof=1))

    denom_mean_contrib = 1.0 / zscore_window

    def spread_for_target(z: float) -> float:
        low, high = -10.0, 10.0
        for _ in range(100):
            mid = 0.5 * (low + high)
            last_window = np.append(prior_window, mid)
            m = float(np.mean(last_window))
            s = float(np.std(last_window, ddof=1))
            if s == 0.0:
                z_mid = 0.0
            else:
                z_mid = (mid - m) / s
            if z_mid < z:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    last_spread_val = spread_for_target(target_last_z)

    full_spread = np.concatenate([hist_spread, [last_spread_val]])

    price_b = 25.0 * np.ones(n_bars)
    price_a = price_b * np.exp(full_spread)

    return _build_bars(n_bars, last_ts, price_a, price_b)


def _build_btc_bars(last_ts: pd.Timestamp, btc_price: float) -> pd.DataFrame:
    index = pd.date_range(end=last_ts, periods=5, freq="h", tz="UTC")
    arr = np.full(5, btc_price)
    df = pd.DataFrame(
        {
            "open": arr,
            "high": arr,
            "low": arr,
            "close": arr,
            "volume": np.ones(5),
        },
        index=index,
    )
    df.index.name = "timestamp"
    return df


# === Scenarios ==============================================================

def scenario_1_no_signal(state: StateStore, config: BotConfig) -> None:
    _hr("Scenario 1: no-signal tick")
    now = pd.Timestamp("2026-04-15 10:00:00", tz="UTC")
    last_bar_ts = now.floor("h")

    _write_params(entry_z=2.5, exit_z=1.0)

    bars_a, bars_b = _build_bars_with_target_z(
        n_bars=ZSCORE_WINDOW + 10,
        last_ts=last_bar_ts,
        zscore_window=ZSCORE_WINDOW,
        target_last_z=0.5,
    )
    adapter = FakeBybitAdapter(SYMBOL_A, SYMBOL_B, SYMBOL_BTC)
    adapter.bars_a = bars_a
    adapter.bars_b = bars_b
    adapter.bars_btc = _build_btc_bars(last_bar_ts, btc_price=80_000.0)
    adapter.price_a = float(bars_a["close"].iloc[-1])
    adapter.price_b = float(bars_b["close"].iloc[-1])
    adapter.price_btc = 80_000.0

    clock = FakeClock(now)

    assert state.get_open_position() is None, "pre: no open positions"
    tick(state, config, adapter, clock, SMOKE_PARAMS_PATH)
    assert state.get_open_position() is None, "post: still no open positions"

    equity = state.latest_equity()
    assert equity is not None, "equity snapshot should have been recorded"
    assert equity.open_position_count == 0
    print(f"  no position opened (|z|=0.5 < entry_z=2.5)")
    print(f"  equity snapshot recorded: nav=${equity.nav:.2f}, open={equity.open_position_count}")


def scenario_2_entry(state: StateStore, config: BotConfig) -> None:
    _hr("Scenario 2: entry tick (SHORT_SPREAD)")
    now = pd.Timestamp("2026-04-16 10:00:00", tz="UTC")
    last_bar_ts = now.floor("h")

    _write_params(entry_z=2.5, exit_z=1.0)

    bars_a, bars_b = _build_bars_with_target_z(
        n_bars=ZSCORE_WINDOW + 10,
        last_ts=last_bar_ts,
        zscore_window=ZSCORE_WINDOW,
        target_last_z=3.0,
    )
    adapter = FakeBybitAdapter(SYMBOL_A, SYMBOL_B, SYMBOL_BTC)
    adapter.bars_a = bars_a
    adapter.bars_b = bars_b
    adapter.bars_btc = _build_btc_bars(last_bar_ts, btc_price=80_000.0)
    adapter.price_a = float(bars_a["close"].iloc[-1])
    adapter.price_b = float(bars_b["close"].iloc[-1])
    adapter.price_btc = 80_000.0

    clock = FakeClock(now)

    assert state.get_open_position() is None
    tick(state, config, adapter, clock, SMOKE_PARAMS_PATH)

    pos = state.get_open_position()
    assert pos is not None, "a position should have been opened"
    assert pos.direction == "SHORT_SPREAD", f"expected SHORT_SPREAD, got {pos.direction}"
    assert pos.symbol_a == SYMBOL_A
    assert pos.symbol_b == SYMBOL_B
    assert pos.entry_z >= 2.5 - 1e-6, f"entry_z should be >= 2.5, got {pos.entry_z}"
    assert pos.active_params_snapshot is not None

    print(f"  opened position id={pos.id}  dir={pos.direction}")
    print(f"  entry_ts={pos.entry_ts}  entry_z={pos.entry_z:+.4f}")
    print(f"  entry_spread={pos.entry_spread:+.6f}")
    print(f"  entry_price_a={pos.entry_price_a:.4f}  entry_price_b={pos.entry_price_b:.4f}")
    print(f"  size_a={pos.size_a:.6f}  size_b={pos.size_b:.6f}")


def scenario_3_exit(state: StateStore, config: BotConfig) -> None:
    _hr("Scenario 3: exit tick (mean_reversion)")
    now = pd.Timestamp("2026-04-17 14:00:00", tz="UTC")
    last_bar_ts = now.floor("h")

    _write_params(entry_z=2.5, exit_z=1.0)

    bars_a, bars_b = _build_bars_with_target_z(
        n_bars=ZSCORE_WINDOW + 10,
        last_ts=last_bar_ts,
        zscore_window=ZSCORE_WINDOW,
        target_last_z=0.3,
    )
    adapter = FakeBybitAdapter(SYMBOL_A, SYMBOL_B, SYMBOL_BTC)
    adapter.bars_a = bars_a
    adapter.bars_b = bars_b
    adapter.bars_btc = _build_btc_bars(last_bar_ts, btc_price=80_000.0)
    adapter.price_a = float(bars_a["close"].iloc[-1])
    adapter.price_b = float(bars_b["close"].iloc[-1])
    adapter.price_btc = 80_000.0
    adapter.funding_a = 0.0001
    adapter.funding_b = 0.00005

    clock = FakeClock(now)

    pre_pos = state.get_open_position()
    assert pre_pos is not None, "scenario 3 requires an open position from scenario 2"
    entry_spread = pre_pos.entry_spread
    entry_direction = pre_pos.direction
    pos_id = pre_pos.id

    tick(state, config, adapter, clock, SMOKE_PARAMS_PATH)

    assert state.get_open_position() is None, "position should be closed"

    trades = state.list_trades(limit=10)
    assert len(trades) >= 1
    closed = [t for t in trades if t.position_id == pos_id]
    assert len(closed) == 1, f"expected exactly 1 closed trade for position {pos_id}"
    t = closed[0]
    assert t.exit_reason == "mean_reversion", f"expected mean_reversion, got {t.exit_reason}"
    assert isinstance(t.pnl_net, float)
    assert isinstance(t.pnl_gross, float)

    sign = 1 if entry_direction == "LONG_SPREAD" else -1
    expected_gross = sign * LEG_NOTIONAL * (t.exit_spread - entry_spread)
    assert abs(t.pnl_gross - expected_gross) < 1e-6, (
        f"pnl_gross {t.pnl_gross} != expected {expected_gross}"
    )

    expected_fees = -4 * LEG_NOTIONAL * (config.fees.fee_per_side + config.fees.slippage_per_side)
    assert abs(t.pnl_fees - expected_fees) < 1e-6, (
        f"pnl_fees {t.pnl_fees} != expected {expected_fees}"
    )

    print(f"  closed trade id={t.id}  position_id={t.position_id}  reason={t.exit_reason}")
    print(f"  bars_held={t.bars_held}")
    print(f"  entry_spread={entry_spread:+.6f}  exit_spread={t.exit_spread:+.6f}")
    print(f"  pnl_gross={t.pnl_gross:+.4f}  pnl_fees={t.pnl_fees:+.4f}  "
          f"pnl_funding={t.pnl_funding:+.4f}  pnl_net={t.pnl_net:+.4f}")
    print(f"  gross PnL formula check OK ({sign} * {LEG_NOTIONAL} * "
          f"({t.exit_spread:+.4f} - {entry_spread:+.4f}))")


def scenario_4_bear_halt(state: StateStore, config: BotConfig) -> None:
    _hr("Scenario 4: bear halt blocks entry")
    now = pd.Timestamp("2026-04-18 10:00:00", tz="UTC")
    last_bar_ts = now.floor("h")

    _write_params(entry_z=2.5, exit_z=1.0)

    bars_a, bars_b = _build_bars_with_target_z(
        n_bars=ZSCORE_WINDOW + 10,
        last_ts=last_bar_ts,
        zscore_window=ZSCORE_WINDOW,
        target_last_z=3.0,
    )
    adapter = FakeBybitAdapter(SYMBOL_A, SYMBOL_B, SYMBOL_BTC)
    adapter.bars_a = bars_a
    adapter.bars_b = bars_b
    adapter.bars_btc = _build_btc_bars(last_bar_ts, btc_price=55_000.0)
    adapter.price_a = float(bars_a["close"].iloc[-1])
    adapter.price_b = float(bars_b["close"].iloc[-1])
    adapter.price_btc = 55_000.0

    clock = FakeClock(now)

    assert state.get_open_position() is None
    tick(state, config, adapter, clock, SMOKE_PARAMS_PATH)

    assert state.get_open_position() is None, "bear halt should have blocked entry"

    events = state.recent_events(limit=20)
    skipped = [e for e in events if e["event_type"] == "ENTRY_SKIPPED"]
    bear_halt_events = [e for e in skipped if e["payload"].get("reason") == "bear_halt"]
    assert len(bear_halt_events) >= 1, "expected at least one ENTRY_SKIPPED with reason=bear_halt"

    latest = bear_halt_events[0]
    print(f"  bear halt correctly triggered (BTC=55k < threshold=60k)")
    print(f"  event logged: {latest['event_type']} reason={latest['payload'].get('reason')} "
          f"z={latest['payload'].get('z'):+.4f}")


def scenario_5_hot_reload(state: StateStore, config: BotConfig) -> None:
    _hr("Scenario 5: hot reload of active_params.json")
    now = pd.Timestamp("2026-04-19 10:00:00", tz="UTC")
    last_bar_ts = now.floor("h")

    _write_params(entry_z=2.5, exit_z=1.0)

    bars_a, bars_b = _build_bars_with_target_z(
        n_bars=ZSCORE_WINDOW + 10,
        last_ts=last_bar_ts,
        zscore_window=ZSCORE_WINDOW,
        target_last_z=1.5,
    )
    adapter = FakeBybitAdapter(SYMBOL_A, SYMBOL_B, SYMBOL_BTC)
    adapter.bars_a = bars_a
    adapter.bars_b = bars_b
    adapter.bars_btc = _build_btc_bars(last_bar_ts, btc_price=80_000.0)
    adapter.price_a = float(bars_a["close"].iloc[-1])
    adapter.price_b = float(bars_b["close"].iloc[-1])
    adapter.price_btc = 80_000.0

    clock = FakeClock(now)

    assert state.get_open_position() is None
    tick(state, config, adapter, clock, SMOKE_PARAMS_PATH)
    assert state.get_open_position() is None, "tick 1: z=1.5 < entry_z=2.5, no entry expected"
    print(f"  tick 1 with entry_z=2.5: no entry (|z|=1.5 < 2.5) OK")

    _write_params(entry_z=1.0, exit_z=0.2)

    with open(SMOKE_PARAMS_PATH, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert reloaded["entry_z"] == 1.0, "params file not updated"
    print(f"  rewrote active_params.json: entry_z=1.0 exit_z=0.2")

    tick(state, config, adapter, clock, SMOKE_PARAMS_PATH)
    pos = state.get_open_position()
    assert pos is not None, "tick 2: with hot-reloaded entry_z=1.0, |z|=1.5 should open"
    assert pos.direction == "SHORT_SPREAD"
    assert pos.active_params_snapshot is not None
    assert pos.active_params_snapshot.get("entry_z") == 1.0, (
        "snapshotted entry_z should be the reloaded 1.0, not the original 2.5"
    )
    print(f"  tick 2 with entry_z=1.0: entry executed, SHORT_SPREAD id={pos.id}")
    print(f"  snapshotted entry_z in position = {pos.active_params_snapshot.get('entry_z')}")


# === Main ===================================================================

def main() -> None:
    _hr("Step 0: fresh DB + fresh params file")
    if SMOKE_DB.exists():
        SMOKE_DB.unlink()
        print(f"  deleted existing {SMOKE_DB}")
    if SMOKE_PARAMS_PATH.exists():
        SMOKE_PARAMS_PATH.unlink()
        print(f"  deleted existing {SMOKE_PARAMS_PATH}")
    SMOKE_DB.parent.mkdir(parents=True, exist_ok=True)

    state = StateStore(SMOKE_DB)
    state.init_schema()
    config = _make_config()
    print(f"  StateStore initialized at {SMOKE_DB}")
    print(f"  Config: pair={config.pair.symbol_a}/{config.pair.symbol_b}  "
          f"leg=${config.risk.leg_notional_usd}")

    scenario_1_no_signal(state, config)
    scenario_2_entry(state, config)
    scenario_3_exit(state, config)
    scenario_4_bear_halt(state, config)
    scenario_5_hot_reload(state, config)

    _hr("SUMMARY")
    all_trades = state.list_trades(limit=100)
    all_events = state.recent_events(limit=100)
    print(f"  closed trades total: {len(all_trades)}")
    print(f"  events logged total: {len(all_events)}")
    print(f"  event types seen:    {sorted({e['event_type'] for e in all_events})}")
    print("\n*** ENGINE SMOKE TEST PASSED ***")


if __name__ == "__main__":
    main()
