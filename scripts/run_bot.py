"""Waddle paper trading bot — production entry point.

Loads config, initializes state, creates the live Bybit adapter + real clock,
starts the engine loop. Ctrl+C for graceful shutdown.

Usage:
    uv run scripts/run_bot.py
    uv run scripts/run_bot.py --config path/to/bot.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from waddle.bot.config import load_bot_config
from waddle.bot.engine import RealClock, run_engine
from waddle.bot.exchange import BybitAdapter
from waddle.bot.state import StateStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/bot.yaml",
        help="Path to bot.yaml",
    )
    parser.add_argument(
        "--active-params",
        default="config/active_params.json",
        help="Path to active_params.json",
    )
    args = parser.parse_args()

    config = load_bot_config(Path(args.config))
    print("Waddle bot starting")
    print(f"  pair: {config.pair.symbol_a} / {config.pair.symbol_b}")
    print(f"  mode: {config.mode}")
    print(f"  state db: {config.state_db}")
    print(f"  polling: {config.engine.polling_interval_seconds}s")

    if config.mode != "paper":
        raise SystemExit(f"ERROR: only mode=paper supported in V1, got {config.mode!r}")

    state = StateStore(Path(config.state_db))
    state.init_schema()

    exchange = BybitAdapter(config.exchange, config.fees, enable_live_execution=False)
    clock = RealClock()

    try:
        run_engine(
            state,
            config,
            exchange,
            clock,
            active_params_path=Path(args.active_params),
        )
    except KeyboardInterrupt:
        print("\nGraceful shutdown.")


if __name__ == "__main__":
    main()
