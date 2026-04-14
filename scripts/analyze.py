"""Load BTC+ETH history, run pipeline, report sweep + Sharpe + fee sensitivity."""
from __future__ import annotations

from waddle.data.storage import load_ohlcv
from waddle.features.correlation import rolling_log_return_correlation
from waddle.features.spread import log_spread, rolling_zscore
from waddle.strategy.pairs_reversion import (
    equity_curve,
    max_drawdown,
    metrics_dict,
    run_pairs_reversion,
)

ROLLING_WINDOW = 24 * 7
CORR_WINDOW = 24 * 7

PARAM_SWEEP = [
    {"label": "baseline",        "entry_z": 2.0, "exit_z": 0.5, "max_holding": 48, "stop_loss_z": None},
    {"label": "tighter_entry",   "entry_z": 2.5, "exit_z": 0.5, "max_holding": 48, "stop_loss_z": None},
    {"label": "wider_exit",      "entry_z": 2.0, "exit_z": 1.0, "max_holding": 48, "stop_loss_z": None},
    {"label": "sweet_spot",      "entry_z": 2.5, "exit_z": 1.0, "max_holding": 48, "stop_loss_z": None},
    {"label": "aggressive_ss",   "entry_z": 3.0, "exit_z": 1.0, "max_holding": 48, "stop_loss_z": None},
]

FEE_SCENARIOS = [
    {"label": "ideal",                  "fee_per_side": 0.0,    "slippage_per_side": 0.0},
    {"label": "bybit_futures_maker",    "fee_per_side": 0.0002, "slippage_per_side": 0.0002},
    {"label": "binance_spot",           "fee_per_side": 0.001,  "slippage_per_side": 0.0003},
    {"label": "binance_spot_pessimist", "fee_per_side": 0.001,  "slippage_per_side": 0.001},
]

SWEET_SPOT = {"entry_z": 2.5, "exit_z": 1.0, "max_holding": 48, "stop_loss_z": None}


def main() -> None:
    btc = load_ohlcv("BTC/USDT")
    eth = load_ohlcv("ETH/USDT")
    aligned = btc.join(eth, how="inner", lsuffix="_btc", rsuffix="_eth")
    period_days = (aligned.index[-1] - aligned.index[0]).total_seconds() / 86400
    print(f"Aligned bars: {len(aligned)}  ({aligned.index[0]}  ->  {aligned.index[-1]})")
    print(f"Period: {period_days:.1f} days")

    spread = log_spread(aligned["close_btc"], aligned["close_eth"])
    zscore = rolling_zscore(spread, window=ROLLING_WINDOW)

    corr = rolling_log_return_correlation(
        aligned["close_btc"], aligned["close_eth"], window=CORR_WINDOW
    ).dropna()
    print(f"\n=== log-return correlation ({CORR_WINDOW}h rolling) ===")
    print(
        f"  mean: {corr.mean():.3f}  min: {corr.min():.3f}  max: {corr.max():.3f}  "
        f"bars<0.8: {(corr < 0.8).sum()}"
    )

    print("\n=== parameter sweep (gross — no fees) ===")
    _print_header()
    for combo in PARAM_SWEEP:
        label = combo["label"]
        params = {k: v for k, v in combo.items() if k != "label"}
        trades = run_pairs_reversion(spread, zscore, **params)
        m = metrics_dict(trades, period_days=period_days)
        _print_row(label, m)

    print("\n=== sweet_spot × fee scenarios ===")
    _print_header()
    for scenario in FEE_SCENARIOS:
        label = scenario["label"]
        trades = run_pairs_reversion(
            spread,
            zscore,
            **SWEET_SPOT,
            fee_per_side=scenario["fee_per_side"],
            slippage_per_side=scenario["slippage_per_side"],
        )
        m = metrics_dict(trades, period_days=period_days)
        _print_row(label, m)


def _print_header() -> None:
    header = (
        f"{'label':<24} {'N':>4} {'win%':>6} {'total':>9} {'ann':>9} "
        f"{'sharpe':>8} {'worst':>9} {'max_dd':>9} {'calmar':>8}"
    )
    print(header)
    print("-" * len(header))


def _print_row(label: str, m: dict) -> None:
    calmar = m["total_pnl"] / abs(m["max_dd"]) if m["max_dd"] != 0 else 0.0
    print(
        f"{label:<24} {m['n_trades']:>4} {m['win_rate'] * 100:>5.1f}% "
        f"{m['total_pnl'] * 100:>+8.2f}% {m['ann_return'] * 100:>+8.2f}% "
        f"{m['sharpe']:>8.2f} {m['worst'] * 100:>+8.2f}% "
        f"{m['max_dd'] * 100:>+8.2f}% {calmar:>8.2f}"
    )


if __name__ == "__main__":
    main()
