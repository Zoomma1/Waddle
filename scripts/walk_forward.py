"""Walk-forward validation — split 6 months into train/test and compare metrics.

The simplest honest walk-forward: apply the current sweet-spot params to both
halves of the data separately and see if the metrics are stable. If train and
test are close, the params generalize. If test is much worse than train, the
sweet spot was overfit to the train period.

Note: we compute spread and z-score on the FULL series (causal rolling — no
lookahead bias across the split) and THEN slice into train/test. Computing
z-score on each half separately would introduce a window warmup bias.
"""
from __future__ import annotations

from waddle.data.storage import load_ohlcv
from waddle.features.spread import log_spread, rolling_zscore
from waddle.strategy.pairs_reversion import metrics_dict, run_pairs_reversion

ROLLING_WINDOW = 24 * 7
SWEET_SPOT = {"entry_z": 2.5, "exit_z": 1.0, "max_holding": 48, "stop_loss_z": None}

SPLITS = [
    {"label": "50/50 split",   "ratio": 0.50},
    {"label": "67/33 split",   "ratio": 0.67},  # Classical train-test split
    {"label": "80/20 split",   "ratio": 0.80},  # Heavier train
]

FEE_SCENARIOS = [
    {"label": "ideal",            "fee_per_side": 0.0,    "slippage_per_side": 0.0},
    {"label": "bybit_futures",    "fee_per_side": 0.0002, "slippage_per_side": 0.0002},
]


def main() -> None:
    btc = load_ohlcv("BTC/USDT")
    eth = load_ohlcv("ETH/USDT")
    aligned = btc.join(eth, how="inner", lsuffix="_btc", rsuffix="_eth")
    period_days = (aligned.index[-1] - aligned.index[0]).total_seconds() / 86400
    print(f"Full period: {len(aligned)} bars over {period_days:.1f} days")
    print(f"Sweet spot params: {SWEET_SPOT}")

    spread = log_spread(aligned["close_btc"], aligned["close_eth"])
    zscore = rolling_zscore(spread, window=ROLLING_WINDOW)

    for fees in FEE_SCENARIOS:
        print(f"\n{'=' * 80}")
        print(f"Fee scenario: {fees['label']} (fee={fees['fee_per_side']}, slip={fees['slippage_per_side']})")
        print("=" * 80)

        for split in SPLITS:
            ratio = split["ratio"]
            split_idx = int(len(aligned) * ratio)
            split_ts = aligned.index[split_idx]

            spread_train = spread.iloc[:split_idx]
            zscore_train = zscore.iloc[:split_idx]
            spread_test = spread.iloc[split_idx:]
            zscore_test = zscore.iloc[split_idx:]

            train_days = (spread_train.index[-1] - spread_train.index[0]).total_seconds() / 86400
            test_days = (spread_test.index[-1] - spread_test.index[0]).total_seconds() / 86400

            train_trades = run_pairs_reversion(
                spread_train, zscore_train, **SWEET_SPOT,
                fee_per_side=fees["fee_per_side"],
                slippage_per_side=fees["slippage_per_side"],
            )
            test_trades = run_pairs_reversion(
                spread_test, zscore_test, **SWEET_SPOT,
                fee_per_side=fees["fee_per_side"],
                slippage_per_side=fees["slippage_per_side"],
            )

            m_train = metrics_dict(train_trades, period_days=train_days)
            m_test = metrics_dict(test_trades, period_days=test_days)

            print(f"\n--- {split['label']} (cut at {split_ts.strftime('%Y-%m-%d')}) ---")
            print(f"{'':14}{'train':>14}{'test':>14}{'test/train':>14}")
            _row("n_trades", m_train['n_trades'], m_test['n_trades'], None)
            _row("win_rate", f"{m_train['win_rate']*100:.1f}%", f"{m_test['win_rate']*100:.1f}%", None)
            _row("total_pnl", f"{m_train['total_pnl']*100:+.2f}%", f"{m_test['total_pnl']*100:+.2f}%", None)
            _row("ann_return", f"{m_train['ann_return']*100:+.2f}%", f"{m_test['ann_return']*100:+.2f}%", None)
            _row("sharpe", f"{m_train['sharpe']:.2f}", f"{m_test['sharpe']:.2f}",
                 _ratio(m_test['sharpe'], m_train['sharpe']))
            _row("max_dd", f"{m_train['max_dd']*100:+.2f}%", f"{m_test['max_dd']*100:+.2f}%", None)
            _row("worst", f"{m_train['worst']*100:+.2f}%", f"{m_test['worst']*100:+.2f}%", None)


def _row(label: str, train_val, test_val, ratio: str | None) -> None:
    ratio_str = ratio if ratio else ""
    print(f"{label:14}{str(train_val):>14}{str(test_val):>14}{ratio_str:>14}")


def _ratio(test: float, train: float) -> str:
    if train == 0:
        return "n/a"
    return f"{test / train:.2f}x"


if __name__ == "__main__":
    main()
