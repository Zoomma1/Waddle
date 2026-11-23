import pandas as pd
import pytest

from waddle.bot.config import FeesConfig
from waddle.bot.replay import HistoricalBybitAdapter, ReplayClock, load_replay_caches
from waddle.data.storage import save_funding_rates, save_ohlcv_bybit

T0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")
FEES = FeesConfig(fee_per_side=0.001, slippage_per_side=0.01)


def _h(n):
    return T0 + pd.Timedelta(hours=n)


def _bars(closes, start=T0):
    idx = pd.date_range(start, periods=len(closes), freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0},
        index=idx,
    )


def _funding(rates_by_hour: dict[int, float]):
    return pd.DataFrame(
        {"funding_rate": list(rates_by_hour.values())},
        index=pd.DatetimeIndex([_h(h) for h in rates_by_hour]),
    )


def _adapter(clock=None, closes=(10.0, 11.0, 12.0, 13.0), funding=None):
    clock = clock or ReplayClock(_h(3))
    return HistoricalBybitAdapter(
        clock=clock,
        fees_config=FEES,
        cached_ohlcv={"A": _bars(list(closes))},
        cached_funding=funding or {},
    )


# --- ReplayClock ---

def test_clock_naive_start_is_localized_to_utc():
    assert ReplayClock(pd.Timestamp("2024-01-01")).now() == T0


def test_clock_advance_set_and_noop_sleep():
    c = ReplayClock(T0)
    c.advance(pd.Timedelta(hours=2))
    assert c.now() == _h(2)
    c.set(pd.Timestamp("2024-01-02"))
    assert c.now() == pd.Timestamp("2024-01-02", tz="UTC")
    assert c.sleep(3600) is None
    assert c.now() == pd.Timestamp("2024-01-02", tz="UTC")


# --- fetch_recent_bars / fetch_latest_price ---

def test_recent_bars_hide_future_bars_and_respect_limit():
    clock = ReplayClock(_h(1))
    a = _adapter(clock)
    assert a.fetch_recent_bars("A")["close"].tolist() == [10.0, 11.0]
    assert a.fetch_recent_bars("A", limit=1)["close"].tolist() == [11.0]
    clock.advance(pd.Timedelta(hours=2))
    assert a.fetch_recent_bars("A")["close"].tolist() == [10.0, 11.0, 12.0, 13.0]


def test_bar_exactly_at_clock_time_is_visible():
    a = _adapter(ReplayClock(_h(2)))
    assert a.fetch_recent_bars("A").index[-1] == _h(2)


def test_unknown_symbol_raises_keyerror_not_network():
    with pytest.raises(KeyError):
        _adapter().fetch_recent_bars("ZZZ")


def test_latest_price_is_close_of_last_visible_bar():
    assert _adapter(ReplayClock(_h(1))).fetch_latest_price("A") == 11.0


def test_latest_price_without_visible_bars_raises():
    with pytest.raises(RuntimeError, match="no visible bars"):
        _adapter(ReplayClock(_h(0) - pd.Timedelta(hours=1))).fetch_latest_price("A")


# --- funding ---

def test_funding_window_is_exclusive_start_inclusive_end():
    f = {"A": _funding({0: 0.5, 2: 0.01, 4: 0.02, 6: 0.04})}
    a = _adapter(funding=f)
    # (h0, h4]: h0 excluded, h2 + h4 included, h6 outside
    assert a.fetch_funding_payment("A", _h(0), _h(4)) == pytest.approx(0.03)


def test_funding_missing_symbol_or_empty_window_is_zero():
    a = _adapter(funding={"A": _funding({2: 0.01})})
    assert a.fetch_funding_payment("B", _h(0), _h(5)) == 0.0
    assert a.fetch_funding_payment("A", _h(3), _h(5)) == 0.0
    assert a.fetch_funding_payment("A", _h(2), _h(2)) == 0.0


def test_funding_accepts_naive_timestamps_as_utc():
    a = _adapter(funding={"A": _funding({2: 0.01})})
    assert a.fetch_funding_payment("A", pd.Timestamp("2024-01-01 01:00"), pd.Timestamp("2024-01-01 03:00")) == pytest.approx(0.01)


# --- paper fills ---

def test_entry_long_spread_buys_a_sells_b_with_adverse_slippage():
    fill = _adapter().paper_execute_entry("LONG_SPREAD", 100.0, 50.0, 500.0)
    assert fill.fill_price_a == pytest.approx(101.0)
    assert fill.fill_price_b == pytest.approx(49.5)
    assert fill.size_a_units == pytest.approx(500.0 / 101.0)
    assert fill.size_b_units == pytest.approx(500.0 / 49.5)
    assert fill.notional_usd == 1000.0
    assert fill.total_fee_usd == pytest.approx(2 * 0.001 * 500.0)


def test_entry_short_spread_is_mirror_image():
    fill = _adapter().paper_execute_entry("SHORT_SPREAD", 100.0, 50.0, 500.0)
    assert fill.fill_price_a == pytest.approx(99.0)
    assert fill.fill_price_b == pytest.approx(50.5)


def test_exit_long_spread_sells_a_buys_b():
    fill = _adapter().paper_execute_exit("LONG_SPREAD", 100.0, 50.0, 5.0, 10.0)
    assert fill.fill_price_a == pytest.approx(99.0)
    assert fill.fill_price_b == pytest.approx(50.5)
    assert fill.notional_usd == pytest.approx(5.0 * 99.0 + 10.0 * 50.5)
    assert fill.total_fee_usd == pytest.approx(0.001 * fill.notional_usd)
    assert (fill.size_a_units, fill.size_b_units) == (5.0, 10.0)


def test_exit_short_spread_is_mirror_image():
    fill = _adapter().paper_execute_exit("SHORT_SPREAD", 100.0, 50.0, 5.0, 10.0)
    assert fill.fill_price_a == pytest.approx(101.0)
    assert fill.fill_price_b == pytest.approx(49.5)


@pytest.mark.parametrize(
    "call",
    [
        lambda a: a.paper_execute_entry("SIDEWAYS", 1.0, 1.0, 1.0),
        lambda a: a.paper_execute_entry("LONG_SPREAD", 1.0, 1.0, 0.0),
        lambda a: a.paper_execute_entry("LONG_SPREAD", 0.0, 1.0, 1.0),
        lambda a: a.paper_execute_exit("SIDEWAYS", 1.0, 1.0, 1.0, 1.0),
        lambda a: a.paper_execute_exit("LONG_SPREAD", 1.0, 1.0, 0.0, 1.0),
        lambda a: a.paper_execute_exit("LONG_SPREAD", 1.0, -1.0, 1.0, 1.0),
    ],
)
def test_paper_fills_validate_inputs(call):
    with pytest.raises(ValueError):
        call(_adapter())


# --- load_replay_caches ---

def test_load_replay_caches_reads_db_and_clips_inclusively(tmp_path):
    db = tmp_path / "t.db"
    save_ohlcv_bybit(_bars([1.0, 2.0, 3.0, 4.0]), "A", db_path=db)
    save_funding_rates(_funding({0: 0.1, 2: 0.2, 4: 0.3}), "A", db_path=db)
    ohlcv, funding = load_replay_caches(["A", "B"], start=_h(1), end=_h(2), db_path=db)
    assert ohlcv["A"]["close"].tolist() == [2.0, 3.0]
    assert funding["A"]["funding_rate"].tolist() == [0.2]
    assert ohlcv["B"].empty and funding["B"].empty


def test_load_replay_caches_without_bounds_keeps_everything(tmp_path):
    db = tmp_path / "t.db"
    save_ohlcv_bybit(_bars([1.0, 2.0]), "A", db_path=db)
    ohlcv, funding = load_replay_caches(["A"], db_path=db)
    assert len(ohlcv["A"]) == 2
    assert funding["A"].empty
