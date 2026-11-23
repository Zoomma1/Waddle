import numpy as np
import pandas as pd
import pytest

from waddle.strategy.pairs_reversion import (
    Trade,
    annualized_return,
    equity_curve,
    max_drawdown,
    metrics_dict,
    run_pairs_reversion,
    sharpe_ratio,
)


def _series(values, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def _run(z, spread=None, **kwargs):
    z_s = _series(z)
    s = _series(spread if spread is not None else [0.0] * len(z))
    return run_pairs_reversion(s, z_s, **kwargs), z_s.index


# --- entries / exits ---

def test_short_spread_entry_and_mean_reversion_exit():
    trades, idx = _run(
        z=[0.0, 2.5, 1.5, 0.3],
        spread=[0.0, 0.5, 0.3, 0.1],
        entry_z=2.0,
        exit_z=0.5,
    )
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "SHORT_SPREAD"
    assert t.exit_reason == "mean_reversion"
    assert (t.entry_ts, t.exit_ts) == (idx[1], idx[3])
    assert t.bars_held == 2
    assert (t.entry_z, t.exit_z) == (2.5, 0.3)
    assert t.pnl == pytest.approx(0.4)  # -1 * (0.1 - 0.5)


def test_long_spread_entry_and_pnl_sign():
    trades, _ = _run(
        z=[0.0, -2.5, -1.0, -0.4],
        spread=[0.0, -0.5, -0.3, -0.2],
    )
    assert [t.direction for t in trades] == ["LONG_SPREAD"]
    assert trades[0].pnl == pytest.approx(0.3)  # +1 * (-0.2 + 0.5)


def test_entry_and_exit_thresholds_are_inclusive():
    trades, idx = _run(z=[2.0, 1.0, 0.5], entry_z=2.0, exit_z=0.5)
    assert len(trades) == 1
    assert (trades[0].entry_ts, trades[0].exit_ts) == (idx[0], idx[2])


def test_no_entry_below_threshold():
    trades, _ = _run(z=[1.99, -1.99, 0.0], entry_z=2.0)
    assert trades == []


def test_entry_bar_is_not_evaluated_for_exit():
    # z at entry is already <= exit_z-compatible only after the entry bar
    trades, _ = _run(z=[2.5, 2.5, 0.0])
    assert trades[0].bars_held == 2


def test_mean_reversion_wins_over_stop_time_on_same_bar():
    trades, _ = _run(z=[2.5, 0.5], max_holding=1)
    assert trades[0].exit_reason == "mean_reversion"


def test_stop_time_exits_when_bars_held_reaches_max_holding():
    trades, idx = _run(z=[2.5, 2.4, 2.3, 2.2, 2.1], max_holding=3)
    assert len(trades) == 1
    assert trades[0].exit_reason == "stop_time"
    assert trades[0].bars_held == 3
    assert trades[0].exit_ts == idx[3]


def test_stop_loss_short_spread_triggers_above_entry_z_plus_stop():
    trades, idx = _run(z=[2.5, 3.0, 3.4, 3.6], stop_loss_z=1.0)
    assert trades[0].exit_reason == "stop_loss"
    assert trades[0].exit_ts == idx[3]  # 3.0 and 3.4 are within entry_z + 1.0 = 3.5


def test_stop_loss_long_spread_triggers_below_entry_z_minus_stop():
    trades, idx = _run(z=[-2.5, -3.0, -3.4, -3.6], stop_loss_z=1.0)
    assert trades[0].exit_reason == "stop_loss"
    assert trades[0].exit_ts == idx[3]


def test_stop_loss_is_strict():
    # exactly entry_z + stop_loss_z does not trigger
    trades, _ = _run(z=[2.5, 3.5], stop_loss_z=1.0)
    assert trades == []


def test_no_stop_loss_by_default():
    trades, _ = _run(z=[2.5, 10.0, 20.0])
    assert trades == []


def test_nan_zscore_bars_are_skipped_and_not_counted_as_held():
    trades, _ = _run(z=[np.nan, 2.5, np.nan, np.nan, 0.0])
    assert len(trades) == 1
    assert trades[0].bars_held == 1


def test_position_still_open_at_end_of_series_produces_no_trade():
    trades, _ = _run(z=[2.5, 2.4, 2.3])
    assert trades == []


def test_can_reenter_on_bar_after_exit_not_same_bar():
    trades, idx = _run(z=[2.5, 0.0, 2.5, 0.0])
    assert len(trades) == 2
    assert trades[1].entry_ts == idx[2]


# --- costs ---

def test_costs_are_four_sides_of_fee_plus_slippage():
    trades, _ = _run(
        z=[0.0, 2.5, 0.0],
        spread=[0.0, 0.5, 0.5],  # zero gross pnl
        fee_per_side=0.001,
        slippage_per_side=0.0005,
    )
    assert trades[0].pnl == pytest.approx(-4 * 0.0015)


# --- entry gating masks ---

def test_cointegration_mask_blocks_entry_until_true():
    z = [2.5, 2.5, 0.0]
    mask = _series([0, 1, 1]).astype(bool)
    trades, idx = _run(z=z, cointegration_mask=mask)
    assert len(trades) == 1
    assert trades[0].entry_ts == idx[1]


def test_regime_mask_blocks_entry():
    mask = _series([0, 0, 0]).astype(bool)
    trades, _ = _run(z=[2.5, 2.5, 0.0], regime_mask=mask)
    assert trades == []


def test_both_masks_must_allow_entry():
    ok = _series([1, 1, 1]).astype(bool)
    ko = _series([0, 0, 0]).astype(bool)
    trades, _ = _run(z=[2.5, 0.0, 2.5, 0.0], cointegration_mask=ok, regime_mask=ko)
    assert trades == []


def test_mask_flip_does_not_force_open_position_out():
    mask = _series([1, 0, 0, 0]).astype(bool)
    trades, idx = _run(z=[2.5, 2.4, 2.3, 0.0], cointegration_mask=mask)
    assert len(trades) == 1
    assert trades[0].exit_ts == idx[3]


def test_timestamp_missing_from_mask_blocks_entry():
    z_s = _series([2.5, 2.5, 0.0])
    s = _series([0.0, 0.0, 0.0])
    mask = pd.Series([True], index=z_s.index[:1])  # only first bar covered
    trades = run_pairs_reversion(s, z_s.iloc[1:], cointegration_mask=mask)
    assert trades == []


def test_float_nan_mask_value_allows_entry_despite_docstring():
    # BUG (not fixed, out of scope): docstring says NaN mask -> False, but
    # bool(nan) is True, so a float mask with NaN lets the entry through.
    mask = _series([np.nan, np.nan, np.nan])
    trades, _ = _run(z=[2.5, 0.0, 0.0], cointegration_mask=mask)
    assert len(trades) == 1


# --- equity / drawdown ---

def _trade(exit_ts, pnl, bars=1, reason="mean_reversion"):
    ts = pd.Timestamp(exit_ts, tz="UTC")
    return Trade(ts, ts, "LONG_SPREAD", 0.0, 0.0, -2.0, 0.0, bars, pnl, reason)


def test_equity_curve_is_cumulative_pnl_sorted_by_exit_ts():
    trades = [_trade("2024-01-03", 0.3), _trade("2024-01-02", 0.1)]
    eq = equity_curve(trades)
    assert eq.tolist() == pytest.approx([0.1, 0.4])
    assert list(eq.index) == [pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC")]


def test_equity_curve_empty():
    assert equity_curve([]).empty


def test_max_drawdown_empty_equity():
    assert max_drawdown(pd.Series(dtype=float)) == {
        "max_dd": 0.0,
        "peak_ts": None,
        "trough_ts": None,
        "recovered": False,
    }


def test_max_drawdown_peak_trough_and_recovery():
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    eq = pd.Series([0.1, 0.3, 0.1, 0.2, 0.4], index=idx)
    dd = max_drawdown(eq)
    assert dd["max_dd"] == pytest.approx(-0.2)
    assert dd["peak_ts"] == idx[1]
    assert dd["trough_ts"] == idx[2]
    assert dd["recovery_ts"] == idx[4]
    assert dd["recovered"] is True


def test_max_drawdown_not_recovered():
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    dd = max_drawdown(pd.Series([0.3, 0.1, 0.2], index=idx))
    assert dd["recovered"] is False
    assert dd["recovery_ts"] is None


def test_max_drawdown_ignores_losses_before_first_gain():
    # characterization: equity starts at the first cumulative pnl, not at 0,
    # so an initial losing streak below zero is not counted as drawdown.
    idx = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    dd = max_drawdown(pd.Series([-0.1, -0.2], index=idx))
    assert dd["max_dd"] == pytest.approx(-0.1)


# --- ratios ---

def test_sharpe_ratio_known_value():
    assert sharpe_ratio(pd.Series([1.0, 3.0]), trades_per_year=4) == pytest.approx(2 * np.sqrt(2))


def test_sharpe_ratio_degenerate_inputs_are_zero():
    assert sharpe_ratio(pd.Series([1.0]), 10) == 0.0
    assert sharpe_ratio(pd.Series([1.0, 1.0, 1.0]), 10) == 0.0


def test_annualized_return_compounds():
    assert annualized_return(0.1, 365) == pytest.approx(0.1)
    assert annualized_return(0.1, 182.5) == pytest.approx(1.1**2 - 1)
    assert annualized_return(0.1, 0) == 0.0


# --- metrics_dict ---

def test_metrics_dict_empty_is_all_zero():
    m = metrics_dict([])
    assert m["n_trades"] == 0
    assert set(m) >= {"win_rate", "total_pnl", "sharpe", "by_stop_loss"}
    assert all(v == 0 or v == 0.0 for v in m.values())


def test_metrics_dict_aggregates_trades():
    trades = [
        _trade("2024-01-02", 0.2, bars=4, reason="mean_reversion"),
        _trade("2024-01-03", -0.1, bars=8, reason="stop_time"),
        _trade("2024-01-04", 0.3, bars=6, reason="stop_loss"),
    ]
    m = metrics_dict(trades)
    assert m["n_trades"] == 3
    assert m["win_rate"] == pytest.approx(2 / 3)
    assert m["total_pnl"] == pytest.approx(0.4)
    assert m["median_pnl"] == pytest.approx(0.2)
    assert m["worst"] == pytest.approx(-0.1)
    assert m["max_dd"] == pytest.approx(-0.1)
    assert m["avg_hold"] == pytest.approx(6.0)
    assert (m["by_mean_rev"], m["by_stop_time"], m["by_stop_loss"]) == (1, 1, 1)
    assert m["sharpe"] == 0.0 and m["ann_return"] == 0.0  # no period_days


def test_metrics_dict_zero_pnl_trade_is_not_a_win():
    assert metrics_dict([_trade("2024-01-02", 0.0), _trade("2024-01-03", 0.1)])["win_rate"] == 0.5


def test_metrics_dict_with_period_fills_sharpe_and_ann_return():
    trades = [_trade("2024-01-02", 0.2), _trade("2024-01-03", 0.4)]
    m = metrics_dict(trades, period_days=365)
    assert m["ann_return"] == pytest.approx(0.6)
    assert m["sharpe"] == pytest.approx((0.3 / np.std([0.2, 0.4], ddof=1)) * np.sqrt(2))
