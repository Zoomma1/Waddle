"""Smoke test for waddle.bot.state.StateStore.

Runs a full happy-path cycle against a fresh throwaway DB:
1. open a realistic SOL/AVAX LONG_SPREAD position
2. snapshot equity + log a TICK event
3. close the position with realistic PnL
4. verify the state is coherent afterwards

Run with: uv run scripts/smoke_test_bot_state.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from waddle.bot.state import StateStore


SMOKE_DB = Path("data/waddle_bot_smoke.db")


def _hr(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> None:
    # Step 0: clean slate
    _hr("Step 0: fresh DB")
    if SMOKE_DB.exists():
        SMOKE_DB.unlink()
        print(f"  deleted existing {SMOKE_DB}")
    else:
        print(f"  {SMOKE_DB} did not exist yet")

    store = StateStore(SMOKE_DB)
    store.init_schema()
    print(f"  schema initialized at {SMOKE_DB}")

    # Step 1: open a LONG_SPREAD position
    _hr("Step 1: open LONG_SPREAD SOL/AVAX")
    entry_ts = pd.Timestamp("2026-04-15 10:00:00", tz="UTC")
    snapshot = {
        "entry_z": 2.5,
        "exit_z": 1.0,
        "max_holding_bars": 48,
        "zscore_window": 168,
        "selected_by": "smoke_test",
    }
    position_id = store.open_position(
        symbol_a="SOL/USDT:USDT",
        symbol_b="AVAX/USDT:USDT",
        direction="LONG_SPREAD",
        entry_ts=entry_ts,
        entry_spread=0.9213,
        entry_z=-2.63,
        entry_price_a=142.50,
        entry_price_b=36.80,
        size_a=500.0,
        size_b=500.0,
        active_params_snapshot=snapshot,
    )
    print(f"  opened position id={position_id}")

    # Step 2: verify get_open_position() returns it
    _hr("Step 2: verify get_open_position()")
    open_pos = store.get_open_position()
    assert open_pos is not None, "get_open_position returned None right after open"
    assert open_pos.id == position_id
    assert open_pos.direction == "LONG_SPREAD"
    assert open_pos.symbol_a == "SOL/USDT:USDT"
    assert open_pos.entry_ts == entry_ts
    assert open_pos.active_params_snapshot == snapshot
    print(f"  pos.id={open_pos.id}  dir={open_pos.direction}  "
          f"{open_pos.symbol_a}/{open_pos.symbol_b}")
    print(f"  entry_ts={open_pos.entry_ts}  entry_z={open_pos.entry_z:+.2f}")
    print(f"  size_a=${open_pos.size_a:.0f}  size_b=${open_pos.size_b:.0f}")
    print(f"  active_params_snapshot={open_pos.active_params_snapshot}")

    # Step 3: record an equity snapshot
    _hr("Step 3: record_equity()")
    equity_ts = pd.Timestamp("2026-04-15 10:00:00", tz="UTC")
    store.record_equity(
        ts=equity_ts,
        nav=1000.0,
        cash=0.0,
        open_position_count=1,
        unrealized_pnl=0.0,
    )
    latest = store.latest_equity()
    assert latest is not None
    assert latest.nav == 1000.0
    assert latest.open_position_count == 1
    print(f"  nav=${latest.nav:.2f}  cash=${latest.cash:.2f}  "
          f"open={latest.open_position_count}  unrealized=${latest.unrealized_pnl:.2f}")

    # Step 4: log a TICK event
    _hr("Step 4: log_event() TICK")
    store.log_event(
        ts=equity_ts,
        level="INFO",
        event_type="TICK",
        payload={
            "z": -2.63,
            "spread": 0.9213,
            "btc_close": 68500.0,
            "bear_halt_ok": True,
        },
    )
    print("  logged TICK event")

    # Step 5: close the position with realistic PnL
    _hr("Step 5: close_position()")
    exit_ts = pd.Timestamp("2026-04-16 14:00:00", tz="UTC")  # ~28h later
    pnl_gross = 18.42       # gross from mean reversion
    pnl_fees = -0.80        # 4 sides * $500 * 0.0004 ~= 0.80
    pnl_funding = 0.15      # positive funding received on short leg
    pnl_net = pnl_gross + pnl_fees + pnl_funding
    trade_id = store.close_position(
        position_id=position_id,
        exit_ts=exit_ts,
        exit_spread=0.9538,
        exit_z=-0.85,
        exit_price_a=145.20,
        exit_price_b=36.20,
        pnl_gross=pnl_gross,
        pnl_fees=pnl_fees,
        pnl_funding=pnl_funding,
        pnl_net=pnl_net,
        bars_held=28,
        exit_reason="mean_reversion",
    )
    print(f"  closed position id={position_id}, new trade id={trade_id}")
    print(f"  pnl_gross=${pnl_gross:+.2f}  fees=${pnl_fees:+.2f}  "
          f"funding=${pnl_funding:+.2f}  net=${pnl_net:+.2f}")

    # Step 6: verify no more open positions
    _hr("Step 6: verify no open positions")
    assert store.get_open_position() is None, "position should be closed"
    assert store.list_open_positions() == []
    print("  get_open_position() -> None OK")

    # Step 7: list trades
    _hr("Step 7: list_trades()")
    trades = store.list_trades(limit=10)
    assert len(trades) == 1
    t = trades[0]
    assert t.position_id == position_id
    assert t.exit_reason == "mean_reversion"
    assert t.bars_held == 28
    print(f"  trade id={t.id}  position_id={t.position_id}  "
          f"exit_reason={t.exit_reason}  bars_held={t.bars_held}")
    print(f"  pnl_net=${t.pnl_net:+.2f}  exit_ts={t.exit_ts}")

    # Step 8: list events
    _hr("Step 8: recent_events()")
    events = store.recent_events(limit=10)
    assert len(events) >= 1
    for e in events:
        print(f"  [{e['level']}] {e['event_type']} @ {e['ts']} payload={e['payload']}")

    # Step 9: final summary
    _hr("SUMMARY")
    all_open = store.list_open_positions()
    all_trades = store.list_trades(limit=100)
    latest_eq = store.latest_equity()
    all_events = store.recent_events(limit=100)
    print(f"  open positions:    {len(all_open)}")
    print(f"  closed trades:     {len(all_trades)}")
    print(f"  equity snapshots:  {1 if latest_eq is not None else 0} (latest nav=${latest_eq.nav:.2f})")
    print(f"  events logged:     {len(all_events)}")
    print("\nSmoke test PASSED")


if __name__ == "__main__":
    main()
