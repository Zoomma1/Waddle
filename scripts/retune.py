"""Walk-forward adaptive retuning — standalone.

Runs periodically (via cron or manual) to re-optimize the adaptive params
based on the most recent TRAIN_WINDOW_DAYS of Bybit data. Writes the
result to config/active_params.json. The bot reloads this file on each
tick for hot-reload of adaptive params.
"""
from __future__ import annotations

from waddle.bot.config import load_bot_config, write_active_params
from waddle.data.storage import load_ohlcv_bybit
from waddle.features.spread import log_spread, rolling_zscore
from waddle.strategy.pairs_reversion import metrics_dict, run_pairs_reversion

TRAIN_WINDOW_DAYS = 90
BARS_PER_DAY = 24
TRAIN_BARS = TRAIN_WINDOW_DAYS * BARS_PER_DAY
ZSCORE_WINDOW = 168

PARAM_GRID = [
    {"entry_z": 1.8, "exit_z": 0.5},
    {"entry_z": 2.0, "exit_z": 0.5},
    {"entry_z": 2.0, "exit_z": 1.0},
    {"entry_z": 2.2, "exit_z": 0.5},
    {"entry_z": 2.2, "exit_z": 1.0},
    {"entry_z": 2.5, "exit_z": 0.5},
    {"entry_z": 2.5, "exit_z": 1.0},
    {"entry_z": 2.7, "exit_z": 1.0},
]
MAX_HOLDING_BARS = 48


def main() -> None:
    config = load_bot_config()
    symbol_a = config.pair.symbol_a
    symbol_b = config.pair.symbol_b
    fee = config.fees.fee_per_side
    slip = config.fees.slippage_per_side

    print(f"Retune — loading last {TRAIN_WINDOW_DAYS} days of {symbol_a} / {symbol_b}")

    bars_a = load_ohlcv_bybit(symbol_a).tail(TRAIN_BARS + ZSCORE_WINDOW)
    bars_b = load_ohlcv_bybit(symbol_b).tail(TRAIN_BARS + ZSCORE_WINDOW)
    aligned = bars_a.join(bars_b, how="inner", lsuffix="_a", rsuffix="_b")

    print(f"Aligned bars: {len(aligned)} ({aligned.index[0]} -> {aligned.index[-1]})")

    spread = log_spread(aligned["close_a"], aligned["close_b"])
    zscore = rolling_zscore(spread, window=ZSCORE_WINDOW)

    train_spread = spread.tail(TRAIN_BARS)
    train_zscore = zscore.tail(TRAIN_BARS)
    train_days_real = (train_spread.index[-1] - train_spread.index[0]).total_seconds() / 86400

    best_combo = None
    best_calmar = -float("inf")
    best_sharpe = 0.0
    best_metrics = None

    print(f"\n{'combo':<24} {'n':>4} {'pnl':>9} {'dd':>9} {'sharpe':>8} {'calmar':>8}")
    print("-" * 64)

    for combo in PARAM_GRID:
        trades = run_pairs_reversion(
            train_spread, train_zscore,
            entry_z=combo["entry_z"],
            exit_z=combo["exit_z"],
            max_holding=MAX_HOLDING_BARS,
            stop_loss_z=None,
            fee_per_side=fee,
            slippage_per_side=slip,
        )
        m = metrics_dict(trades, period_days=train_days_real)
        if m["n_trades"] < 3:
            calmar = -float("inf")
        elif m["max_dd"] == 0:
            calmar = float("inf") if m["total_pnl"] > 0 else 0.0
        else:
            calmar = m["total_pnl"] / abs(m["max_dd"])

        label = f"entry={combo['entry_z']}/exit={combo['exit_z']}"
        print(f"{label:<24} {m['n_trades']:>4} "
              f"{m['total_pnl']*100:>+7.2f}% "
              f"{m['max_dd']*100:>+8.2f}% "
              f"{m['sharpe']:>+8.2f} "
              f"{calmar:>+8.2f}")

        if calmar > best_calmar:
            best_calmar = calmar
            best_combo = combo
            best_sharpe = m["sharpe"]
            best_metrics = m

    if best_combo is None:
        print("\nERROR: no valid combo selected (all rejected). active_params.json NOT updated.")
        return

    print(f"\nBest combo: entry={best_combo['entry_z']}/exit={best_combo['exit_z']}  "
          f"Calmar={best_calmar:+.2f}  Sharpe={best_sharpe:+.2f}")

    params = write_active_params(
        entry_z=best_combo["entry_z"],
        exit_z=best_combo["exit_z"],
        max_holding_bars=MAX_HOLDING_BARS,
        zscore_window=ZSCORE_WINDOW,
        train_calmar=best_calmar,
        train_sharpe=best_sharpe,
        train_window_days=TRAIN_WINDOW_DAYS,
    )

    print(f"\nWrote config/active_params.json:")
    print(f"  entry_z={params.entry_z}")
    print(f"  exit_z={params.exit_z}")
    print(f"  max_holding_bars={params.max_holding_bars}")
    print(f"  updated_at={params.updated_at}")


if __name__ == "__main__":
    main()
