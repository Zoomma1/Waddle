import json
from datetime import datetime, timezone

import pytest
import yaml

from waddle.bot.config import (
    ActiveParams,
    load_active_params,
    load_bot_config,
    write_active_params,
)

BOT_YAML = {
    "pair": {"symbol_a": "A/USDT:USDT", "symbol_b": "B/USDT:USDT", "reference_btc": "BTC/USDT:USDT"},
    "exchange": {"name": "bybit", "type": "swap", "testnet": True},
    "fees": {"fee_per_side": 0.0002, "slippage_per_side": 0.0003},
    "risk": {"total_notional_usd": 1000.0, "leg_notional_usd": 500.0, "bear_halt_btc_threshold": 60000.0},
    "engine": {"polling_interval_seconds": 60, "lookback_bars": 200},
    "mode": "paper",
    "state_db": "data/x.db",
}

PARAMS = dict(
    entry_z=2.0,
    exit_z=0.5,
    max_holding_bars=48,
    zscore_window=336,
    train_calmar=1.5,
    train_sharpe=1.2,
    train_window_days=90,
)


def _write_yaml(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_bot_config_parses_every_section(tmp_path):
    cfg = load_bot_config(_write_yaml(tmp_path / "bot.yaml", BOT_YAML))
    assert cfg.pair.symbol_a == "A/USDT:USDT"
    assert cfg.pair.reference_btc == "BTC/USDT:USDT"
    assert cfg.exchange.testnet is True
    assert cfg.fees.slippage_per_side == 0.0003
    assert cfg.risk.leg_notional_usd == 500.0
    assert cfg.risk.bear_halt_btc_threshold == 60000.0
    assert cfg.engine.lookback_bars == 200
    assert cfg.mode == "paper"
    assert cfg.state_db == "data/x.db"


def test_load_bot_config_raises_on_missing_section(tmp_path):
    raw = {k: v for k, v in BOT_YAML.items() if k != "risk"}
    with pytest.raises(KeyError):
        load_bot_config(_write_yaml(tmp_path / "bot.yaml", raw))


def test_load_bot_config_raises_on_missing_field(tmp_path):
    raw = {**BOT_YAML, "fees": {"fee_per_side": 0.0002}}
    with pytest.raises(TypeError):
        load_bot_config(_write_yaml(tmp_path / "bot.yaml", raw))


def test_load_bot_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bot_config(tmp_path / "nope.yaml")


def test_write_then_load_active_params_roundtrip(tmp_path):
    path = tmp_path / "cfg" / "active_params.json"
    written = write_active_params(**PARAMS, json_path=path)
    loaded = load_active_params(path)
    assert loaded == written
    assert loaded.entry_z == 2.0
    assert loaded.max_holding_bars == 48
    assert loaded.zscore_window == 336
    assert loaded.selected_by == "retune.py"


def test_write_active_params_stamps_utc_iso_seconds(tmp_path):
    p = write_active_params(**PARAMS, json_path=tmp_path / "p.json")
    parsed = datetime.fromisoformat(p.updated_at)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timezone.utc.utcoffset(None)
    assert parsed.microsecond == 0


def test_write_active_params_custom_selected_by(tmp_path):
    p = write_active_params(**PARAMS, selected_by="manual", json_path=tmp_path / "p.json")
    assert load_active_params(tmp_path / "p.json").selected_by == "manual" == p.selected_by


def test_write_active_params_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "p.json"
    write_active_params(**PARAMS, json_path=path)
    write_active_params(**{**PARAMS, "entry_z": 3.0}, json_path=path)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["p.json"]
    assert json.loads(path.read_text())["entry_z"] == 3.0


def test_load_active_params_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_active_params(tmp_path / "nope.json")


def test_load_active_params_rejects_unknown_key(tmp_path):
    path = tmp_path / "p.json"
    p = write_active_params(**PARAMS, json_path=path)
    raw = json.loads(path.read_text())
    raw["surprise"] = 1
    path.write_text(json.dumps(raw))
    with pytest.raises(TypeError):
        load_active_params(path)
    assert isinstance(p, ActiveParams)
