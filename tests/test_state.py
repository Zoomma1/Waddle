import sqlite3

import pandas as pd
import pytest

from waddle.bot.state import StateStore

T0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")


def _h(n):
    return T0 + pd.Timedelta(hours=n)


@pytest.fixture
def store(tmp_path):
    s = StateStore(tmp_path / "sub" / "bot.db")
    s.init_schema()
    return s


def _open(store, entry_ts=T0, direction="LONG_SPREAD", snapshot=None):
    return store.open_position(
        symbol_a="A",
        symbol_b="B",
        direction=direction,
        entry_ts=entry_ts,
        entry_spread=-0.5,
        entry_z=-2.5,
        entry_price_a=10.0,
        entry_price_b=20.0,
        size_a=50.0,
        size_b=25.0,
        active_params_snapshot=snapshot,
    )


def _close(store, position_id, exit_ts=None, net=1.0):
    return store.close_position(
        position_id=position_id,
        exit_ts=exit_ts or _h(5),
        exit_spread=-0.1,
        exit_z=-0.3,
        exit_price_a=11.0,
        exit_price_b=19.0,
        pnl_gross=1.5,
        pnl_fees=-0.4,
        pnl_funding=-0.1,
        pnl_net=net,
        bars_held=5,
        exit_reason="mean_reversion",
    )


def test_store_creates_parent_dir_and_init_schema_is_idempotent(store):
    assert store.db_path.exists()
    store.init_schema()
    assert store.get_open_position() is None


def test_open_position_roundtrip(store):
    pid = _open(store, snapshot={"entry_z": 2.0, "exit_z": 0.5})
    pos = store.get_open_position()
    assert pos is not None
    assert pos.id == pid
    assert pos.status == "OPEN"
    assert pos.direction == "LONG_SPREAD"
    assert pos.entry_ts == T0
    assert pos.entry_ts.tzinfo is not None
    assert (pos.entry_spread, pos.entry_z) == (-0.5, -2.5)
    assert (pos.entry_price_a, pos.entry_price_b) == (10.0, 20.0)
    assert (pos.size_a, pos.size_b) == (50.0, 25.0)
    assert pos.active_params_snapshot == {"entry_z": 2.0, "exit_z": 0.5}


def test_missing_snapshot_reads_back_as_none(store):
    _open(store)
    assert store.get_open_position().active_params_snapshot is None


def test_naive_timestamp_is_stored_as_utc(store):
    _open(store, entry_ts=pd.Timestamp("2024-01-01 00:00"))
    assert store.get_open_position().entry_ts == T0


def test_get_open_position_returns_oldest_and_list_is_ordered(store):
    late = _open(store, entry_ts=_h(3))
    early = _open(store, entry_ts=_h(1))
    assert store.get_open_position().id == early
    assert [p.id for p in store.list_open_positions()] == [early, late]


def test_close_position_writes_trade_and_flips_status(store):
    pid = _open(store)
    tid = _close(store, pid, net=0.9)
    assert store.get_open_position() is None
    [trade] = store.list_trades()
    assert trade.id == tid
    assert trade.position_id == pid
    assert trade.exit_ts == _h(5)
    assert (trade.pnl_gross, trade.pnl_fees, trade.pnl_funding, trade.pnl_net) == (1.5, -0.4, -0.1, 0.9)
    assert trade.bars_held == 5
    assert trade.exit_reason == "mean_reversion"


def test_close_unknown_position_rolls_back_and_raises(store):
    pid = _open(store)
    with pytest.raises(sqlite3.IntegrityError):
        _close(store, 999)
    assert store.list_trades() == []
    assert store.get_open_position().id == pid


def test_close_position_has_no_double_close_guard(store):
    # characterization: closing an already-closed position inserts a 2nd trade
    pid = _open(store)
    _close(store, pid)
    _close(store, pid)
    assert len(store.list_trades()) == 2


def test_list_trades_most_recent_first_and_limited(store):
    ids = [_open(store, entry_ts=_h(i)) for i in range(3)]
    for i, pid in enumerate(ids):
        _close(store, pid, exit_ts=_h(10 + i))
    trades = store.list_trades()
    assert [t.exit_ts for t in trades] == [_h(12), _h(11), _h(10)]
    assert len(store.list_trades(limit=2)) == 2


def test_equity_latest_and_replace_on_same_ts(store):
    assert store.latest_equity() is None
    store.record_equity(_h(1), nav=1000.0, cash=1000.0, open_position_count=0, unrealized_pnl=0.0)
    store.record_equity(_h(2), nav=1010.0, cash=1000.0, open_position_count=1, unrealized_pnl=10.0)
    store.record_equity(_h(2), nav=1020.0, cash=1000.0, open_position_count=1, unrealized_pnl=20.0)
    snap = store.latest_equity()
    assert snap.ts == _h(2)
    assert (snap.nav, snap.cash, snap.open_position_count, snap.unrealized_pnl) == (1020.0, 1000.0, 1, 20.0)


def test_events_roundtrip_order_and_level_filter(store):
    store.log_event(_h(1), "INFO", "A", {"k": 1})
    store.log_event(_h(3), "WARNING", "B", {"k": [1, 2]})
    store.log_event(_h(2), "INFO", "C", {})
    events = store.recent_events()
    assert [e["event_type"] for e in events] == ["B", "C", "A"]
    assert events[0]["payload"] == {"k": [1, 2]}
    assert events[0]["ts"] == _h(3)
    assert [e["event_type"] for e in store.recent_events(level="INFO")] == ["C", "A"]
    assert len(store.recent_events(limit=1)) == 1
