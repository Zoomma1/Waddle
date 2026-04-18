# Waddle

SOL/AVAX pairs mean reversion bot. Paper trading on Bybit perpetual futures, 1h bars, live since April 2026.

---

## Strategy

SOL and AVAX are two L1 chains that trade with enough economic coupling that their log-spread tends to mean revert. When it gets stretched far enough from its rolling mean, Waddle bets on a snap back.

Signal pipeline:

```
Bybit perps 1h OHLCV
    → log_spread = log(SOL) - log(AVAX)
    → rolling z-score (168h window)
    → entry when |z| > threshold
    → exit when |z| < exit_threshold or 48h elapsed
```

Position logic:
- z > threshold → short SOL, long AVAX
- z < -threshold → long SOL, short AVAX
- No directional stop-loss (explained below)
- 48h max holding per trade
- No new entries if BTC < 60k (untested regime, halted by default)

Parameters re-tune every ~15 days via walk-forward sweep on the last 90 days of data.

---

## Backtest results

18 months OOS on Bybit perpetuals. Maker fees at 0.02%/side, funding rates integrated into PnL.

| Metric | Value |
|--------|-------|
| Sharpe ratio (median) | ~2.32 |
| Total PnL / 18 months | +95.3% |
| Positive windows | 72% |
| Max drawdown (OOS) | -6% to -10% |
| Mean reversion half-life | ~24 days |
| Funding rate impact | +0.18%/yr |

The mean Sharpe across walk-forward windows is ~4.43. Don't cite that number — it's pulled up by outlier windows. Median ~2.32 is the honest figure.

Walk-forward: rolling 90-day train, 30-day OOS, 8 parameter combos, best Calmar selected.

---

## Design decisions

**No directional stop-loss.** People expect one. In mean reversion it's the wrong instinct — a position that keeps diverging is at its highest expected return, not its lowest. Cutting early is the worst possible timing. The actual risk control is stop-time: if the trade hasn't closed in 48h, force-close regardless.

**Futures only.** Spot fees kill the edge before you make a cent. Same strategy on Binance spot (~0.1%/side) returns -2.47%/yr. It needs Bybit futures maker fees (0.02%/side) to work.

**No ADF cointegration filter.** Tested it. A 90-day rolling ADF test runs on a completely different horizon than a 48h hold, so it blocks trades right when the spread is most stretched. Removed it. SOL/AVAX have structural economic coupling — you don't need a rolling stat test reinventing that wheel every 90 days.

**Log spread, not price ratio.** `log(SOL) - log(AVAX)` is stationary-compatible and additive. `SOL/AVAX` is not. All correlations use log returns.

---

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd waddle
uv sync

# Backfill Bybit perps (last 18 months)
uv run scripts/backfill_bybit.py

# Walk-forward sweep → writes best params to config/active_params.json
uv run scripts/retune.py

# Run the bot (Ctrl+C to stop, positions survive restarts)
uv run scripts/run_bot.py
```

After any engine change, run the replay to make sure nothing drifted:

```bash
uv run scripts/replay_bot.py --start 2026-01-01 --end 2026-04-15
# Expect: 31 bot trades = 31 oracle trades, PnL diff ≤ 5%
```

---

## Status

Paper trading on Bybit since April 2026. No real capital deployed yet.

Moving to live when: Sharpe > 1.0 net of fees over 3-6 months of paper, including at least one choppy or bearish market.

---

## Stack

Python 3.13 · uv · ccxt · pandas · SQLite · Bybit perpetuals
