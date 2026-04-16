"""Bot configuration loading and hot-reload of adaptive params.

The bot has two config layers:
- bot.yaml (static): pair, exchange, fees, risk, engine settings. Loaded once at startup.
- active_params.json (dynamic): entry_z, exit_z, max_holding, zscore_window. Reloaded every tick.

Split rationale: adaptive params change every 15-30 days via retune.py, the rest never changes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json

import yaml


# === Static config (loaded from bot.yaml) ===

@dataclass
class PairConfig:
    symbol_a: str
    symbol_b: str
    reference_btc: str


@dataclass
class ExchangeConfig:
    name: str
    type: str  # "swap" for USDT perpetuals
    testnet: bool


@dataclass
class FeesConfig:
    fee_per_side: float
    slippage_per_side: float


@dataclass
class RiskConfig:
    total_notional_usd: float
    leg_notional_usd: float
    bear_halt_btc_threshold: float


@dataclass
class EngineConfig:
    polling_interval_seconds: int
    lookback_bars: int


@dataclass
class BotConfig:
    pair: PairConfig
    exchange: ExchangeConfig
    fees: FeesConfig
    risk: RiskConfig
    engine: EngineConfig
    mode: str  # "paper" | "live"
    state_db: str  # path to SQLite state db (can differ from data db)


def load_bot_config(yaml_path: Path = Path("config/bot.yaml")) -> BotConfig:
    """Load and parse the static bot config. Raises if any required field is missing."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return BotConfig(
        pair=PairConfig(**raw["pair"]),
        exchange=ExchangeConfig(**raw["exchange"]),
        fees=FeesConfig(**raw["fees"]),
        risk=RiskConfig(**raw["risk"]),
        engine=EngineConfig(**raw["engine"]),
        mode=raw["mode"],
        state_db=raw["state_db"],
    )


# === Dynamic config (loaded from active_params.json on every tick) ===

@dataclass
class ActiveParams:
    entry_z: float
    exit_z: float
    max_holding_bars: int
    zscore_window: int
    updated_at: str          # ISO 8601 UTC
    selected_by: str         # "retune.py" | "manual" | etc
    train_window_days: int
    train_calmar: float
    train_sharpe: float      # for logging / debugging


def load_active_params(json_path: Path = Path("config/active_params.json")) -> ActiveParams:
    """Load the current adaptive params. Raises FileNotFoundError if file does not exist yet
    (the bot should not start without a valid active_params — require running retune.py first)."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return ActiveParams(**raw)


def write_active_params(
    entry_z: float,
    exit_z: float,
    max_holding_bars: int,
    zscore_window: int,
    train_calmar: float,
    train_sharpe: float,
    train_window_days: int,
    selected_by: str = "retune.py",
    json_path: Path = Path("config/active_params.json"),
) -> ActiveParams:
    """Write a new active_params.json. Called by retune.py. Atomic via temp-file + rename
    so a partial write cannot corrupt the file while the bot is reading it."""
    params = ActiveParams(
        entry_z=entry_z,
        exit_z=exit_z,
        max_holding_bars=max_holding_bars,
        zscore_window=zscore_window,
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        selected_by=selected_by,
        train_window_days=train_window_days,
        train_calmar=train_calmar,
        train_sharpe=train_sharpe,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = json_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(asdict(params), f, indent=2)
    tmp_path.replace(json_path)  # atomic rename
    return params
