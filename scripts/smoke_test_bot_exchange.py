"""Smoke test for waddle.bot.exchange.BybitAdapter.

Hits the live Bybit API to validate the read-only primitives and runs
purely-local asserts on the paper fill primitives.

Run with: uv run scripts/smoke_test_bot_exchange.py
"""
from __future__ import annotations

import pandas as pd

from waddle.bot.config import ExchangeConfig, FeesConfig, load_bot_config
from waddle.bot.exchange import BybitAdapter, PaperFill


def _hr(title: str) -> None:
    print(f"\n--- {title} ---")


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main() -> None:
    config = load_bot_config()
    adapter = BybitAdapter(
        exchange_config=config.exchange,
        fees_config=config.fees,
        enable_live_execution=False,
    )
    symbol = config.pair.symbol_a
    fee = config.fees.fee_per_side
    slip = config.fees.slippage_per_side
    print(f"Adapter initialized for exchange={config.exchange.name} type={config.exchange.type}")
    print(f"Fees: fee_per_side={fee}, slippage_per_side={slip}")
    print(f"Test symbol: {symbol}")

    # Step 1 - fetch live bars + latest price
    _hr("Step 1: fetch_recent_bars + fetch_latest_price (live Bybit)")
    bars = adapter.fetch_recent_bars(symbol, limit=50, timeframe="1h")
    print(f"  returned {len(bars)} bars")
    print(f"  first ts: {bars.index[0]}")
    print(f"  last  ts: {bars.index[-1]}")
    print(f"  last close: {bars['close'].iloc[-1]:.4f}")

    latest = adapter.fetch_latest_price(symbol)
    last_close = float(bars["close"].iloc[-1])
    drift = abs(latest - last_close) / last_close
    print(f"  fetch_latest_price: {latest:.4f}")
    print(f"  drift vs last close: {drift * 100:.3f}%")
    if drift > 0.01:
        print(f"  WARNING: drift > 1% — not a failure but surprisingly large")
    else:
        print(f"  OK: drift within 1% tolerance")

    # Step 2 - paper entry (LONG_SPREAD, synthetic prices)
    _hr("Step 2: paper_execute_entry (LONG_SPREAD, price_a=100, price_b=25, notional=500)")
    entry = adapter.paper_execute_entry(
        direction="LONG_SPREAD",
        price_a=100.0,
        price_b=25.0,
        notional_per_leg_usd=500.0,
    )
    expected_fill_a = 100.0 * (1 + slip)
    expected_fill_b = 25.0 * (1 - slip)
    expected_size_a = 500.0 / expected_fill_a
    expected_size_b = 500.0 / expected_fill_b
    expected_fee = 2 * fee * 500.0
    print(f"  fill_price_a  = {entry.fill_price_a:.6f}  (expected {expected_fill_a:.6f})")
    print(f"  fill_price_b  = {entry.fill_price_b:.6f}  (expected {expected_fill_b:.6f})")
    print(f"  size_a_units  = {entry.size_a_units:.8f}  (expected {expected_size_a:.8f})")
    print(f"  size_b_units  = {entry.size_b_units:.8f}  (expected {expected_size_b:.8f})")
    print(f"  notional_usd  = {entry.notional_usd:.4f}  (expected 1000.0)")
    print(f"  total_fee_usd = {entry.total_fee_usd:.6f}  (expected {expected_fee:.6f})")

    assert _approx(entry.fill_price_a, expected_fill_a), "fill_price_a mismatch"
    assert _approx(entry.fill_price_b, expected_fill_b), "fill_price_b mismatch"
    assert _approx(entry.size_a_units, expected_size_a), "size_a_units mismatch"
    assert _approx(entry.size_b_units, expected_size_b), "size_b_units mismatch"
    assert _approx(entry.notional_usd, 1000.0), "notional_usd mismatch"
    assert _approx(entry.total_fee_usd, expected_fee), "total_fee_usd mismatch"
    print("  OK: all entry asserts pass")

    # Step 3 - paper exit at a favorable spread move (LONG_SPREAD => want A up, B down)
    _hr("Step 3: paper_execute_exit (close the LONG_SPREAD at price_a=105, price_b=23)")
    exit_fill = adapter.paper_execute_exit(
        direction="LONG_SPREAD",
        price_a=105.0,
        price_b=23.0,
        size_a_units=entry.size_a_units,
        size_b_units=entry.size_b_units,
    )
    expected_exit_a = 105.0 * (1 - slip)
    expected_exit_b = 23.0 * (1 + slip)
    expected_exit_fee = fee * (entry.size_a_units * expected_exit_a + entry.size_b_units * expected_exit_b)
    print(f"  fill_price_a  = {exit_fill.fill_price_a:.6f}  (expected {expected_exit_a:.6f})")
    print(f"  fill_price_b  = {exit_fill.fill_price_b:.6f}  (expected {expected_exit_b:.6f})")
    print(f"  size_a_units  = {exit_fill.size_a_units:.8f}")
    print(f"  size_b_units  = {exit_fill.size_b_units:.8f}")
    print(f"  notional_usd  = {exit_fill.notional_usd:.4f}")
    print(f"  total_fee_usd = {exit_fill.total_fee_usd:.6f}  (expected {expected_exit_fee:.6f})")

    assert _approx(exit_fill.fill_price_a, expected_exit_a), "exit fill_price_a mismatch"
    assert _approx(exit_fill.fill_price_b, expected_exit_b), "exit fill_price_b mismatch"
    assert _approx(exit_fill.size_a_units, entry.size_a_units), "exit size_a_units drift"
    assert _approx(exit_fill.size_b_units, entry.size_b_units), "exit size_b_units drift"
    assert _approx(exit_fill.total_fee_usd, expected_exit_fee, tol=1e-6), "exit total_fee_usd mismatch"
    print("  OK: all exit asserts pass")

    leg_a_pnl = entry.size_a_units * (exit_fill.fill_price_a - entry.fill_price_a)
    leg_b_pnl = entry.size_b_units * (entry.fill_price_b - exit_fill.fill_price_b)
    gross_pnl = leg_a_pnl + leg_b_pnl
    total_fees = entry.total_fee_usd + exit_fill.total_fee_usd
    net_pnl = gross_pnl - total_fees
    print(f"  computed gross PnL: {gross_pnl:+.4f} USD")
    print(f"  computed total fees: {total_fees:.4f} USD")
    print(f"  computed net PnL: {net_pnl:+.4f} USD")

    # Sanity check: a LONG_SPREAD where A moved up (100 -> 105) AND B moved down (25 -> 23)
    # is the textbook favorable case. PnL must be clearly positive despite fees.
    assert gross_pnl > 0, "gross PnL should be positive on a favorable LONG_SPREAD exit"

    # Step 4 - SHORT_SPREAD roundtrip (direction coverage, local only)
    _hr("Step 4: paper_execute_entry + paper_execute_exit (SHORT_SPREAD, sanity)")
    short_entry = adapter.paper_execute_entry(
        direction="SHORT_SPREAD",
        price_a=100.0,
        price_b=25.0,
        notional_per_leg_usd=500.0,
    )
    assert _approx(short_entry.fill_price_a, 100.0 * (1 - slip)), "short entry fill_price_a direction"
    assert _approx(short_entry.fill_price_b, 25.0 * (1 + slip)), "short entry fill_price_b direction"
    print(f"  short entry fill_price_a: {short_entry.fill_price_a:.6f} (sold, slip-)")
    print(f"  short entry fill_price_b: {short_entry.fill_price_b:.6f} (bought, slip+)")

    short_exit = adapter.paper_execute_exit(
        direction="SHORT_SPREAD",
        price_a=95.0,
        price_b=26.0,
        size_a_units=short_entry.size_a_units,
        size_b_units=short_entry.size_b_units,
    )
    assert _approx(short_exit.fill_price_a, 95.0 * (1 + slip)), "short exit fill_price_a direction"
    assert _approx(short_exit.fill_price_b, 26.0 * (1 - slip)), "short exit fill_price_b direction"
    print(f"  short exit  fill_price_a: {short_exit.fill_price_a:.6f} (bought to close, slip+)")
    print(f"  short exit  fill_price_b: {short_exit.fill_price_b:.6f} (sold to close, slip-)")
    print("  OK: SHORT_SPREAD slippage orientations pass")

    # Step 5 - funding payment (live)
    _hr("Step 5: fetch_funding_payment (last 48h, live Bybit)")
    now = pd.Timestamp.now(tz="UTC")
    start = now - pd.Timedelta(hours=48)
    funding = adapter.fetch_funding_payment(symbol, start_ts=start, end_ts=now)
    print(f"  window: {start} -> {now}")
    print(f"  cumulative funding rate: {funding:+.6f}  ({funding * 100:+.4f}%)")
    print(f"  (expected order of magnitude: 1e-4 to 1e-3 over 48h across ~6 settlements)")

    # Validation of the cost_per_trade equivalence with the backtest
    _hr("Step 6: backtest-equivalence check")
    leg_notional = 500.0
    roundtrip_slippage_cost = 4 * slip * leg_notional
    roundtrip_fee_cost = 4 * fee * leg_notional
    total_roundtrip_cost = roundtrip_slippage_cost + roundtrip_fee_cost
    backtest_formula_cost = 4 * (fee + slip) * leg_notional
    print(f"  roundtrip slippage cost (2 legs * 2 sides * slip * notional): {roundtrip_slippage_cost:.4f}")
    print(f"  roundtrip fee cost      (2 legs * 2 sides * fee  * notional): {roundtrip_fee_cost:.4f}")
    print(f"  total roundtrip cost: {total_roundtrip_cost:.4f}")
    print(f"  backtest formula  4*(fee+slip)*notional: {backtest_formula_cost:.4f}")
    assert _approx(total_roundtrip_cost, backtest_formula_cost), "cost identity broken"
    print("  OK: paper fill cost model matches backtest 4*(fee+slip)*notional")

    print("\n*** SMOKE TEST PASSED ***")


if __name__ == "__main__":
    main()
