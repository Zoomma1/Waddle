"""Pairs mean-reversion state machine.

Simple rules:
    flat + z >=  entry_z  -> open SHORT_SPREAD (short A, long B)
    flat + z <= -entry_z  -> open LONG_SPREAD  (long A, short B)
    in position + |z| <= exit_z                 -> close (mean reversion hit)
    in position + bars_held >= max_holding      -> close (stop-time)
    in position + adverse move > stop_loss_z    -> close (stop-loss)

PnL is expressed in log-spread units: `sign * (exit_spread - entry_spread)`,
where sign = +1 for LONG_SPREAD and -1 for SHORT_SPREAD. These are the (approx.)
log returns of a dollar-neutral long-A/short-B position.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Trade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    direction: str
    entry_spread: float
    exit_spread: float
    entry_z: float
    exit_z: float
    bars_held: int
    pnl: float
    exit_reason: str  # "mean_reversion", "stop_time", "stop_loss"


def run_pairs_reversion(
    spread: pd.Series,
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    max_holding: int = 48,
    stop_loss_z: float | None = None,
    fee_per_side: float = 0.0,
    slippage_per_side: float = 0.0,
    cointegration_mask: pd.Series | None = None,
    regime_mask: pd.Series | None = None,
) -> list[Trade]:
    """Run the pairs mean-reversion state machine.

    Fees and slippage model: a dollar-neutral pair trade opens 2 legs and closes
    2 legs = 4 sides. PnL is debited by `4 * (fee_per_side + slippage_per_side)`
    per trade. Both are expressed as fractions (0.001 = 0.1%).

    Entry gating (optional):
        `cointegration_mask` and `regime_mask` are boolean Series aligned to
        the zscore index. When provided, new entries are only allowed at
        timestamps where the mask is True. Existing positions are never forced
        out by a mask flip — they exit normally via mean reversion, stop-time,
        or stop-loss. This matches ADR-007: do not close positions when a
        filter deactivates, only refuse new ones.

        Missing timestamps or NaN mask values are treated as False
        (conservative default: no new trade without a valid filter reading).
    """
    cost_per_trade = 4 * (fee_per_side + slippage_per_side)
    trades: list[Trade] = []
    position: str | None = None
    entry_info: dict | None = None
    bars_in_position = 0

    def _entry_allowed(ts: pd.Timestamp) -> bool:
        if cointegration_mask is not None:
            if not bool(cointegration_mask.get(ts, False)):
                return False
        if regime_mask is not None:
            if not bool(regime_mask.get(ts, False)):
                return False
        return True

    for ts, z in zscore.items():
        if pd.isna(z):
            continue
        s = spread.loc[ts]

        if position is None:
            if z >= entry_z or z <= -entry_z:
                if not _entry_allowed(ts):
                    continue
                if z >= entry_z:
                    position = "SHORT_SPREAD"
                else:
                    position = "LONG_SPREAD"
                entry_info = {"ts": ts, "spread": s, "z": z}
                bars_in_position = 0
            continue

        assert entry_info is not None
        bars_in_position += 1

        exit_reason: str | None = None
        if abs(z) <= exit_z:
            exit_reason = "mean_reversion"
        elif bars_in_position >= max_holding:
            exit_reason = "stop_time"
        elif stop_loss_z is not None:
            if position == "LONG_SPREAD" and z < entry_info["z"] - stop_loss_z:
                exit_reason = "stop_loss"
            elif position == "SHORT_SPREAD" and z > entry_info["z"] + stop_loss_z:
                exit_reason = "stop_loss"

        if exit_reason is not None:
            sign = 1 if position == "LONG_SPREAD" else -1
            gross_pnl = sign * (s - entry_info["spread"])
            pnl = gross_pnl - cost_per_trade
            trades.append(
                Trade(
                    entry_ts=entry_info["ts"],
                    exit_ts=ts,
                    direction=position,
                    entry_spread=entry_info["spread"],
                    exit_spread=s,
                    entry_z=entry_info["z"],
                    exit_z=z,
                    bars_held=bars_in_position,
                    pnl=pnl,
                    exit_reason=exit_reason,
                )
            )
            position = None
            entry_info = None
            bars_in_position = 0

    return trades


def equity_curve(trades: list[Trade]) -> pd.Series:
    """Cumulative PnL indexed by trade exit timestamp."""
    if not trades:
        return pd.Series(dtype=float)
    data = {t.exit_ts: t.pnl for t in trades}
    return pd.Series(data).sort_index().cumsum()


def max_drawdown(equity: pd.Series) -> dict:
    """Largest peak-to-trough drawdown of an equity curve.

    A drawdown is how far we are below the running maximum of the equity curve —
    the psychological pain of watching profits evaporate. Max drawdown is the
    worst such episode over the whole series.
    """
    if equity.empty:
        return {"max_dd": 0.0, "peak_ts": None, "trough_ts": None, "recovered": False}

    running_max = equity.cummax()
    drawdown = equity - running_max
    trough_ts = drawdown.idxmin()
    max_dd_val = float(drawdown.loc[trough_ts])

    peak_level = running_max.loc[trough_ts]
    pre_trough = equity.loc[:trough_ts]
    peak_ts = pre_trough[pre_trough >= peak_level].index[0]

    post_trough = equity.loc[trough_ts:]
    recovered = post_trough[post_trough >= peak_level]
    recovery_ts = recovered.index[0] if len(recovered) > 0 else None

    return {
        "max_dd": max_dd_val,
        "peak_ts": peak_ts,
        "trough_ts": trough_ts,
        "recovery_ts": recovery_ts,
        "recovered": recovery_ts is not None,
    }


def sharpe_ratio(pnls: pd.Series, trades_per_year: float) -> float:
    """Per-trade annualized Sharpe ratio (risk-free rate = 0).

    Uses per-trade returns as the unit — simpler than per-bar and appropriate
    for a strategy that holds sparse positions. Annualizes via sqrt of the
    expected number of trades per year, under the iid-returns assumption.
    """
    if len(pnls) < 2 or pnls.std() == 0:
        return 0.0
    return float((pnls.mean() / pnls.std()) * (trades_per_year ** 0.5))


def annualized_return(total_pnl: float, period_days: float) -> float:
    """Compound a total PnL observed over `period_days` to a full year."""
    if period_days <= 0:
        return 0.0
    return float((1 + total_pnl) ** (365 / period_days) - 1)


def metrics_dict(trades: list[Trade], period_days: float | None = None) -> dict:
    """Flat numeric metrics for side-by-side comparison of parameter combos.

    `period_days` is optional: required only if you want Sharpe + annualized
    return. Sharpe requires knowing how many trades/year the strategy produces.
    """
    if not trades:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "mean_pnl": 0.0,
            "median_pnl": 0.0,
            "worst": 0.0,
            "max_dd": 0.0,
            "avg_hold": 0.0,
            "sharpe": 0.0,
            "ann_return": 0.0,
            "by_mean_rev": 0,
            "by_stop_time": 0,
            "by_stop_loss": 0,
        }
    pnls = pd.Series([t.pnl for t in trades])
    bars = pd.Series([t.bars_held for t in trades])
    equity = equity_curve(trades)
    dd = max_drawdown(equity)
    reasons = pd.Series([t.exit_reason for t in trades])

    sharpe = 0.0
    ann_ret = 0.0
    if period_days is not None and period_days > 0:
        trades_per_year = len(trades) * (365 / period_days)
        sharpe = sharpe_ratio(pnls, trades_per_year)
        ann_ret = annualized_return(float(pnls.sum()), period_days)

    return {
        "n_trades": len(trades),
        "win_rate": (pnls > 0).sum() / len(trades),
        "total_pnl": float(pnls.sum()),
        "mean_pnl": float(pnls.mean()),
        "median_pnl": float(pnls.median()),
        "worst": float(pnls.min()),
        "max_dd": dd["max_dd"],
        "avg_hold": float(bars.mean()),
        "sharpe": sharpe,
        "ann_return": ann_ret,
        "by_mean_rev": int((reasons == "mean_reversion").sum()),
        "by_stop_time": int((reasons == "stop_time").sum()),
        "by_stop_loss": int((reasons == "stop_loss").sum()),
    }
