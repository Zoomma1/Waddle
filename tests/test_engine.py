import ccxt
import numpy as np
import pandas as pd
import pytest

from waddle.bot.config import (
    BotConfig,
    EngineConfig,
    ExchangeConfig,
    FeesConfig,
    PairConfig,
    RiskConfig,
    write_active_params,
)
from waddle.bot.engine import run_engine, tick
from waddle.bot.replay import HistoricalBybitAdapter, ReplayClock
from waddle.bot.state import StateStore

T0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")
N = 40
LAST = T0 + pd.Timedelta(hours=N - 1)
SLIP = 0.0003
FEE = 0.0002
LEG = 500.0
TOTAL = 1000.0


def _h(n):
    return T0 + pd.Timedelta(hours=n)


CONFIG = BotConfig(
    pair=PairConfig("A", "B", "BTC"),
    exchange=ExchangeConfig("bybit", "swap", False),
    fees=FeesConfig(fee_per_side=FEE, slippage_per_side=SLIP),
    risk=RiskConfig(total_notional_usd=TOTAL, leg_notional_usd=LEG, bear_halt_btc_threshold=60000.0),
    engine=EngineConfig(polling_interval_seconds=60, lookback_bars=30),
    mode="paper",
    state_db="unused",
)


def _bars(closes, n=None):
    idx = pd.date_range(T0, periods=len(closes), freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0}, index=idx
    )


def _spreads(last: float | None, n: int = N) -> np.ndarray:
    """Small deterministic wiggle, then `last` (None -> mean of previous 9 => z == 0)."""
    s = 0.002 * np.sin(np.arange(n))
    s[-1] = s[-10:-1].mean() if last is None else last
    return s


class World:
    def __init__(self, tmp_path, *, last_spread, btc=70000.0, funding=None, n=N, adapter_cls=HistoricalBybitAdapter, clock=None, flat=False, **param_overrides):
        self.state = StateStore(tmp_path / "bot.db")
        self.state.init_schema()
        spread = _spreads(last_spread, n)
        a = pd.Series(np.full(n, 100.0) if flat else 100.0 * np.exp(spread))
        ohlcv = {"A": _bars(a.tolist()), "B": _bars([100.0] * n)}
        if btc is not None:
            ohlcv["BTC"] = _bars([btc] * n)
        self.spread = spread
        self.clock = clock or ReplayClock(_h(n - 1))
        self.exchange = adapter_cls(self.clock, CONFIG.fees, ohlcv, funding or {})
        self.params_path = tmp_path / "active_params.json"
        params = dict(
            entry_z=2.0, exit_z=0.5, max_holding_bars=10, zscore_window=10,
            train_calmar=1.0, train_sharpe=1.0, train_window_days=90,
        )
        params.update(param_overrides)
        write_active_params(**params, json_path=self.params_path)

    def tick(self):
        tick(self.state, CONFIG, self.exchange, self.clock, self.params_path)

    def events(self, event_type=None):
        ev = self.state.recent_events(limit=100)
        return [e for e in ev if event_type is None or e["event_type"] == event_type]

    def open_position(self, direction, hours_ago, entry_spread):
        return self.state.open_position(
            "A", "B", direction, LAST - pd.Timedelta(hours=hours_ago),
            entry_spread, 2.5 if direction == "SHORT_SPREAD" else -2.5,
            100.0, 100.0, LEG / 100.0, LEG / 100.0, {"entry_z": 2.0},
        )


# --- entries ---

def test_tick_opens_short_spread_on_positive_spike(tmp_path):
    w = World(tmp_path, last_spread=0.05)
    w.tick()
    pos = w.state.get_open_position()
    assert pos is not None
    assert pos.direction == "SHORT_SPREAD"
    assert pos.entry_ts == LAST
    assert pos.entry_z >= 2.0
    assert pos.entry_spread == pytest.approx(0.05)
    last_close_a = 100.0 * np.exp(0.05)
    assert pos.entry_price_a == pytest.approx(last_close_a * (1 - SLIP))
    assert pos.entry_price_b == pytest.approx(100.0 * (1 + SLIP))
    assert pos.size_a == pytest.approx(LEG / pos.entry_price_a)
    assert pos.active_params_snapshot["entry_z"] == 2.0
    [entry] = w.events("ENTRY")
    assert entry["payload"]["direction"] == "SHORT_SPREAD"
    assert entry["payload"]["entry_fee_usd"] == pytest.approx(2 * FEE * LEG)


def test_tick_opens_long_spread_on_negative_spike(tmp_path):
    w = World(tmp_path, last_spread=-0.05)
    w.tick()
    pos = w.state.get_open_position()
    assert pos.direction == "LONG_SPREAD"
    assert pos.entry_z <= -2.0


def test_tick_does_nothing_when_z_inside_band(tmp_path):
    w = World(tmp_path, last_spread=0.002 * np.sin(N - 1))
    w.tick()
    assert w.state.get_open_position() is None
    assert w.events("ENTRY") == []


def test_tick_records_equity_snapshot_at_clock_time(tmp_path):
    w = World(tmp_path, last_spread=0.05)
    w.tick()
    snap = w.state.latest_equity()
    assert snap.ts == LAST
    assert snap.open_position_count == 1
    assert snap.nav == pytest.approx(TOTAL)  # opened at the current spread
    assert snap.unrealized_pnl == pytest.approx(0.0)


def test_bear_halt_blocks_entry_and_logs_when_z_triggers(tmp_path):
    w = World(tmp_path, last_spread=0.05, btc=59999.0)
    w.tick()
    assert w.state.get_open_position() is None
    [ev] = w.events("ENTRY_SKIPPED")
    assert ev["payload"]["reason"] == "bear_halt"
    assert ev["payload"]["bear_threshold"] == 60000.0


def test_bear_halt_silent_when_no_entry_signal(tmp_path):
    w = World(tmp_path, last_spread=0.002 * np.sin(N - 1), btc=1.0)
    w.tick()
    assert w.events("ENTRY_SKIPPED") == []


def test_bear_halt_threshold_is_inclusive_in_engine(tmp_path):
    # characterization: engine uses >=, whereas features.bear_halt_mask uses strict >.
    w = World(tmp_path, last_spread=0.05, btc=60000.0)
    w.tick()
    assert w.state.get_open_position() is not None


# --- skips ---

def test_insufficient_bars_skips_tick_without_equity(tmp_path):
    w = World(tmp_path, last_spread=0.05, n=10)
    w.tick()
    [ev] = w.events("TICK_SKIPPED")
    assert ev["payload"] == {"reason": "insufficient_bars", "n": 10}
    assert w.state.latest_equity() is None


def test_flat_prices_give_nan_zscore_skip(tmp_path):
    w = World(tmp_path, last_spread=0.0, flat=True)
    w.tick()
    [ev] = w.events("TICK_SKIPPED")
    assert ev["payload"]["reason"] == "nan_zscore"


def test_missing_btc_reference_skips_tick(tmp_path):
    w = World(tmp_path, last_spread=0.05, btc=None)
    w.tick()
    [ev] = w.events("TICK_SKIPPED")
    assert ev["payload"]["reason"] == "no_btc_reference_bar"
    assert w.state.get_open_position() is None


class _FailingPrice(HistoricalBybitAdapter):
    def fetch_latest_price(self, symbol):
        if symbol == "A":
            raise RuntimeError("ticker down")
        return super().fetch_latest_price(symbol)


def test_entry_skipped_when_latest_price_fetch_fails(tmp_path):
    w = World(tmp_path, last_spread=0.05, adapter_cls=_FailingPrice)
    w.tick()
    assert w.state.get_open_position() is None
    [ev] = w.events("ENTRY_SKIPPED")
    assert ev["payload"]["reason"] == "fetch_latest_failed"


def test_exit_skipped_when_latest_price_fetch_fails(tmp_path):
    w = World(tmp_path, last_spread=None, adapter_cls=_FailingPrice)
    w.open_position("SHORT_SPREAD", hours_ago=3, entry_spread=0.05)
    w.tick()
    assert w.state.get_open_position() is not None
    [ev] = w.events("EXIT_SKIPPED")
    assert ev["payload"]["reason"] == "fetch_latest_failed"


# --- exits ---

def test_mean_reversion_close_books_pnl_fees_and_funding(tmp_path):
    funding = {
        # entry bar is LAST-3h: that settlement must NOT be paid
        "A": pd.DataFrame({"funding_rate": [0.5, 0.001]}, index=[LAST - pd.Timedelta(hours=3), LAST - pd.Timedelta(hours=1)]),
        "B": pd.DataFrame({"funding_rate": [0.003]}, index=[LAST - pd.Timedelta(hours=2)]),
    }
    w = World(tmp_path, last_spread=None, funding=funding)
    w.open_position("SHORT_SPREAD", hours_ago=3, entry_spread=0.05)
    w.tick()

    assert w.state.get_open_position() is None
    [t] = w.state.list_trades()
    last_spread = float(w.spread[-1])
    assert t.exit_reason == "mean_reversion"
    assert t.bars_held == 3
    assert t.exit_ts == LAST
    assert abs(t.exit_z) <= 0.5
    assert t.pnl_gross == pytest.approx(-1 * LEG * (last_spread - 0.05))
    assert t.pnl_fees == pytest.approx(-4 * LEG * (FEE + SLIP))
    assert t.pnl_funding == pytest.approx(-1 * LEG * (0.003 - 0.001))
    assert t.pnl_net == pytest.approx(t.pnl_gross + t.pnl_fees + t.pnl_funding)
    [ev] = w.events("EXIT")
    assert ev["payload"]["pnl_net"] == pytest.approx(t.pnl_net)

    snap = w.state.latest_equity()
    assert snap.open_position_count == 0
    assert snap.cash == pytest.approx(TOTAL + t.pnl_net)
    assert snap.nav == pytest.approx(snap.cash)
    assert snap.unrealized_pnl == 0.0


def test_long_spread_close_has_positive_sign(tmp_path):
    w = World(tmp_path, last_spread=None)
    w.open_position("LONG_SPREAD", hours_ago=3, entry_spread=-0.05)
    w.tick()
    [t] = w.state.list_trades()
    assert t.pnl_gross == pytest.approx(LEG * (float(w.spread[-1]) + 0.05))


def test_stop_time_close_when_held_long_enough_and_z_still_extreme(tmp_path):
    w = World(tmp_path, last_spread=0.05)
    w.open_position("SHORT_SPREAD", hours_ago=12, entry_spread=0.04)
    w.tick()
    [t] = w.state.list_trades()
    assert t.exit_reason == "stop_time"
    assert t.bars_held == 12


def test_position_held_below_max_holding_with_extreme_z_stays_open(tmp_path):
    w = World(tmp_path, last_spread=0.05)
    w.open_position("SHORT_SPREAD", hours_ago=9, entry_spread=0.04)
    w.tick()
    assert w.state.list_trades() == []
    snap = w.state.latest_equity()
    assert snap.open_position_count == 1
    assert snap.unrealized_pnl == pytest.approx(-1 * LEG * (0.05 - 0.04))
    assert snap.nav == pytest.approx(TOTAL + snap.unrealized_pnl)


def test_exit_z_boundary_is_inclusive(tmp_path):
    from waddle.features.spread import log_spread, rolling_zscore

    w = World(tmp_path, last_spread=0.05, max_holding_bars=100)
    bars_a = w.exchange.fetch_recent_bars("A", limit=30)
    bars_b = w.exchange.fetch_recent_bars("B", limit=30)
    last_z = float(rolling_zscore(log_spread(bars_a["close"], bars_b["close"]), 10).iloc[-1])
    write_active_params(
        entry_z=9.0, exit_z=last_z, max_holding_bars=100, zscore_window=10,
        train_calmar=1.0, train_sharpe=1.0, train_window_days=90, json_path=w.params_path,
    )
    w.open_position("SHORT_SPREAD", hours_ago=2, entry_spread=0.04)
    w.tick()
    [t] = w.state.list_trades()
    assert t.exit_reason == "mean_reversion"


def test_stop_time_boundary_is_inclusive(tmp_path):
    w = World(tmp_path, last_spread=0.05)
    w.open_position("SHORT_SPREAD", hours_ago=10, entry_spread=0.04)
    w.tick()
    [t] = w.state.list_trades()
    assert t.exit_reason == "stop_time"


def test_cash_includes_realized_pnl_of_past_trades(tmp_path):
    w = World(tmp_path, last_spread=0.002 * np.sin(N - 1))
    pid = w.open_position("SHORT_SPREAD", hours_ago=30, entry_spread=0.0)
    w.state.close_position(pid, _h(1), 0.0, 0.0, 100.0, 100.0, 12.0, 0.0, 0.0, 12.0, 1, "mean_reversion")
    w.tick()
    snap = w.state.latest_equity()
    assert snap.cash == pytest.approx(TOTAL + 12.0)
    assert snap.nav == pytest.approx(TOTAL + 12.0)


def test_no_directional_stop_loss_adverse_move_does_not_close(tmp_path):
    # ADR-003: a spread that keeps running against the position is NOT stopped.
    w = World(tmp_path, last_spread=0.5)
    w.open_position("SHORT_SPREAD", hours_ago=2, entry_spread=0.04)
    w.tick()
    assert w.state.get_open_position() is not None


def test_open_position_still_exits_while_btc_below_bear_threshold(tmp_path):
    w = World(tmp_path, last_spread=None, btc=1.0)
    w.open_position("SHORT_SPREAD", hours_ago=3, entry_spread=0.05)
    w.tick()
    assert w.state.get_open_position() is None


class _FailingFunding(HistoricalBybitAdapter):
    def fetch_funding_payment(self, symbol, start_ts, end_ts):
        raise RuntimeError("funding api down")


def test_funding_failure_books_zero_funding_and_still_closes(tmp_path):
    w = World(tmp_path, last_spread=None, adapter_cls=_FailingFunding)
    w.open_position("SHORT_SPREAD", hours_ago=3, entry_spread=0.05)
    w.tick()
    [t] = w.state.list_trades()
    assert t.pnl_funding == 0.0
    assert len(w.events("FUNDING_FETCH_FAILED")) == 1


def test_hot_reloaded_params_are_read_every_tick(tmp_path):
    w = World(tmp_path, last_spread=0.05)
    write_active_params(
        entry_z=5.0, exit_z=0.5, max_holding_bars=10, zscore_window=10,
        train_calmar=1.0, train_sharpe=1.0, train_window_days=90, json_path=w.params_path,
    )
    w.tick()
    assert w.state.get_open_position() is None  # z ~2.8 < 5


# --- run_engine ---

class _StopAfter(ReplayClock):
    def __init__(self, start, sleeps_before_interrupt):
        super().__init__(start)
        self.sleeps = 0
        self._limit = sleeps_before_interrupt

    def sleep(self, seconds):
        self.sleeps += 1
        if self.sleeps >= self._limit:
            raise KeyboardInterrupt


def test_run_engine_logs_startup_ticks_and_clean_shutdown(tmp_path):
    w = World(tmp_path, last_spread=0.05, clock=_StopAfter(LAST, 1))
    run_engine(w.state, CONFIG, w.exchange, w.clock, w.params_path)
    kinds = {e["event_type"] for e in w.events()}
    assert {"STARTUP", "ENTRY", "SHUTDOWN"} <= kinds
    startup = w.events("STARTUP")[0]["payload"]
    assert startup["pair"] == "A/B" and startup["mode"] == "paper"
    assert w.events("SHUTDOWN")[0]["payload"] == {"reason": "keyboard_interrupt"}
    assert w.state.get_open_position() is not None  # left open on shutdown


def test_run_engine_survives_tick_exception_and_logs_error(tmp_path):
    w = World(tmp_path, last_spread=0.05, clock=_StopAfter(LAST, 3))
    run_engine(w.state, CONFIG, w.exchange, w.clock, tmp_path / "missing.json")
    failed = w.events("TICK_FAILED")
    assert len(failed) == 3
    assert {e["level"] for e in failed} == {"ERROR"}
    assert failed[0]["payload"]["type"] == "FileNotFoundError"


class _ExchangeDown(HistoricalBybitAdapter):
    def fetch_recent_bars(self, symbol, limit=200, timeframe="1h"):
        raise ccxt.ExchangeError("maintenance")


def test_run_engine_logs_ccxt_exchange_error_as_warning(tmp_path):
    w = World(tmp_path, last_spread=0.05, adapter_cls=_ExchangeDown, clock=_StopAfter(LAST, 1))
    run_engine(w.state, CONFIG, w.exchange, w.clock, w.params_path)
    [ev] = w.events("TICK_FAILED")
    assert ev["level"] == "WARNING"
    assert ev["payload"] == {"error": "maintenance", "type": "ExchangeError"}
