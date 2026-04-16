"""SQLite-backed state store for the paper trading bot.

Manages positions, trades, equity snapshots, and event logs in a dedicated
database (default `data/waddle_bot.db`) that is isolated from market data
(`data/waddle.db`). All tables are prefixed `bot_` so the two concerns never
share a primary key namespace.

Timestamp convention: pd.Timestamp in Python, unix milliseconds (int) on disk.
Conversion happens at the DB boundary via `int(ts.timestamp() * 1000)` on write
and `pd.to_datetime(ms, unit="ms", utc=True)` on read.

Transactions: `close_position` is the only operation that writes to two tables
in one logical step (INSERT into bot_trades + UPDATE of bot_positions.status).
It uses an explicit BEGIN / COMMIT / ROLLBACK block so either both rows land or
neither does.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


_SCHEMA_POSITIONS = """
CREATE TABLE IF NOT EXISTS bot_positions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_a                TEXT    NOT NULL,
    symbol_b                TEXT    NOT NULL,
    direction               TEXT    NOT NULL,
    entry_ts                INTEGER NOT NULL,
    entry_spread            REAL    NOT NULL,
    entry_z                 REAL    NOT NULL,
    entry_price_a           REAL    NOT NULL,
    entry_price_b           REAL    NOT NULL,
    size_a                  REAL    NOT NULL,
    size_b                  REAL    NOT NULL,
    status                  TEXT    NOT NULL DEFAULT 'OPEN',
    active_params_snapshot  TEXT
)
"""

_SCHEMA_POSITIONS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_bot_positions_status ON bot_positions(status)
"""

_SCHEMA_TRADES = """
CREATE TABLE IF NOT EXISTS bot_trades (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id    INTEGER NOT NULL REFERENCES bot_positions(id),
    exit_ts        INTEGER NOT NULL,
    exit_spread    REAL    NOT NULL,
    exit_z         REAL    NOT NULL,
    exit_price_a   REAL    NOT NULL,
    exit_price_b   REAL    NOT NULL,
    pnl_gross      REAL    NOT NULL,
    pnl_fees       REAL    NOT NULL,
    pnl_funding    REAL    NOT NULL,
    pnl_net        REAL    NOT NULL,
    bars_held      INTEGER NOT NULL,
    exit_reason    TEXT    NOT NULL
)
"""

_SCHEMA_TRADES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_bot_trades_position ON bot_trades(position_id)
"""

_SCHEMA_EQUITY = """
CREATE TABLE IF NOT EXISTS bot_equity_snapshots (
    ts                   INTEGER PRIMARY KEY,
    nav                  REAL    NOT NULL,
    cash                 REAL    NOT NULL,
    open_position_count  INTEGER NOT NULL,
    unrealized_pnl       REAL    NOT NULL
)
"""

_SCHEMA_EVENTS = """
CREATE TABLE IF NOT EXISTS bot_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            INTEGER NOT NULL,
    level         TEXT    NOT NULL,
    event_type    TEXT    NOT NULL,
    payload_json  TEXT    NOT NULL
)
"""

_SCHEMA_EVENTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_bot_events_ts ON bot_events(ts)
"""


@dataclass
class Position:
    id: int | None
    symbol_a: str
    symbol_b: str
    direction: str
    entry_ts: pd.Timestamp
    entry_spread: float
    entry_z: float
    entry_price_a: float
    entry_price_b: float
    size_a: float
    size_b: float
    status: str
    active_params_snapshot: dict | None


@dataclass
class BotTrade:
    id: int | None
    position_id: int
    exit_ts: pd.Timestamp
    exit_spread: float
    exit_z: float
    exit_price_a: float
    exit_price_b: float
    pnl_gross: float
    pnl_fees: float
    pnl_funding: float
    pnl_net: float
    bars_held: int
    exit_reason: str


@dataclass
class EquitySnapshot:
    ts: pd.Timestamp
    nav: float
    cash: float
    open_position_count: int
    unrealized_pnl: float


def _ts_to_ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def _ms_to_ts(ms: int) -> pd.Timestamp:
    return pd.to_datetime(ms, unit="ms", utc=True)


def _row_to_position(row: sqlite3.Row) -> Position:
    snapshot_raw = row["active_params_snapshot"]
    snapshot = json.loads(snapshot_raw) if snapshot_raw else None
    return Position(
        id=row["id"],
        symbol_a=row["symbol_a"],
        symbol_b=row["symbol_b"],
        direction=row["direction"],
        entry_ts=_ms_to_ts(row["entry_ts"]),
        entry_spread=row["entry_spread"],
        entry_z=row["entry_z"],
        entry_price_a=row["entry_price_a"],
        entry_price_b=row["entry_price_b"],
        size_a=row["size_a"],
        size_b=row["size_b"],
        status=row["status"],
        active_params_snapshot=snapshot,
    )


def _row_to_trade(row: sqlite3.Row) -> BotTrade:
    return BotTrade(
        id=row["id"],
        position_id=row["position_id"],
        exit_ts=_ms_to_ts(row["exit_ts"]),
        exit_spread=row["exit_spread"],
        exit_z=row["exit_z"],
        exit_price_a=row["exit_price_a"],
        exit_price_b=row["exit_price_b"],
        pnl_gross=row["pnl_gross"],
        pnl_fees=row["pnl_fees"],
        pnl_funding=row["pnl_funding"],
        pnl_net=row["pnl_net"],
        bars_held=row["bars_held"],
        exit_reason=row["exit_reason"],
    )


class StateStore:
    """Thin SQLite wrapper for bot state — positions, trades, equity, events."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def init_schema(self) -> None:
        """Create all tables + indexes if they don't exist. Idempotent."""
        con = self._connect()
        try:
            con.execute(_SCHEMA_POSITIONS)
            con.execute(_SCHEMA_POSITIONS_INDEX)
            con.execute(_SCHEMA_TRADES)
            con.execute(_SCHEMA_TRADES_INDEX)
            con.execute(_SCHEMA_EQUITY)
            con.execute(_SCHEMA_EVENTS)
            con.execute(_SCHEMA_EVENTS_INDEX)
            con.commit()
        finally:
            con.close()

    # === Positions ===

    def open_position(
        self,
        symbol_a: str,
        symbol_b: str,
        direction: str,
        entry_ts: pd.Timestamp,
        entry_spread: float,
        entry_z: float,
        entry_price_a: float,
        entry_price_b: float,
        size_a: float,
        size_b: float,
        active_params_snapshot: dict | None = None,
    ) -> int:
        """Insert a new OPEN position. Returns the new position id."""
        snapshot_json = json.dumps(active_params_snapshot) if active_params_snapshot is not None else None
        con = self._connect()
        try:
            cur = con.execute(
                "INSERT INTO bot_positions ("
                "symbol_a, symbol_b, direction, entry_ts, entry_spread, entry_z, "
                "entry_price_a, entry_price_b, size_a, size_b, status, active_params_snapshot"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    symbol_a,
                    symbol_b,
                    direction,
                    _ts_to_ms(entry_ts),
                    float(entry_spread),
                    float(entry_z),
                    float(entry_price_a),
                    float(entry_price_b),
                    float(size_a),
                    float(size_b),
                    "OPEN",
                    snapshot_json,
                ),
            )
            con.commit()
            new_id = cur.lastrowid
            if new_id is None:
                raise RuntimeError("open_position: lastrowid was None after INSERT")
            return int(new_id)
        finally:
            con.close()

    def get_open_position(self) -> Position | None:
        """Return the single currently-open position, or None.

        V1 assumes at most one position at a time (single pair). If the store
        ever holds more than one, the oldest (smallest entry_ts) is returned.
        """
        con = self._connect()
        try:
            row = con.execute(
                "SELECT * FROM bot_positions WHERE status = 'OPEN' "
                "ORDER BY entry_ts ASC LIMIT 1"
            ).fetchone()
            return _row_to_position(row) if row is not None else None
        finally:
            con.close()

    def list_open_positions(self) -> list[Position]:
        """All open positions, ordered by entry_ts ASC."""
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT * FROM bot_positions WHERE status = 'OPEN' ORDER BY entry_ts ASC"
            ).fetchall()
            return [_row_to_position(r) for r in rows]
        finally:
            con.close()

    def close_position(
        self,
        position_id: int,
        exit_ts: pd.Timestamp,
        exit_spread: float,
        exit_z: float,
        exit_price_a: float,
        exit_price_b: float,
        pnl_gross: float,
        pnl_fees: float,
        pnl_funding: float,
        pnl_net: float,
        bars_held: int,
        exit_reason: str,
    ) -> int:
        """Atomically insert a bot_trades row and flip the position status to CLOSED.

        Both statements run inside a single explicit transaction. If either
        fails, neither commits — the position stays OPEN and the caller sees
        the exception.
        """
        con = self._connect()
        try:
            con.execute("BEGIN")
            cur = con.execute(
                "INSERT INTO bot_trades ("
                "position_id, exit_ts, exit_spread, exit_z, exit_price_a, exit_price_b, "
                "pnl_gross, pnl_fees, pnl_funding, pnl_net, bars_held, exit_reason"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(position_id),
                    _ts_to_ms(exit_ts),
                    float(exit_spread),
                    float(exit_z),
                    float(exit_price_a),
                    float(exit_price_b),
                    float(pnl_gross),
                    float(pnl_fees),
                    float(pnl_funding),
                    float(pnl_net),
                    int(bars_held),
                    exit_reason,
                ),
            )
            new_trade_id = cur.lastrowid
            con.execute(
                "UPDATE bot_positions SET status = 'CLOSED' WHERE id = ?",
                (int(position_id),),
            )
            con.commit()
            if new_trade_id is None:
                raise RuntimeError("close_position: lastrowid was None after INSERT")
            return int(new_trade_id)
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def list_trades(self, limit: int = 100) -> list[BotTrade]:
        """Recent trades, most recent first."""
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT * FROM bot_trades ORDER BY exit_ts DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            return [_row_to_trade(r) for r in rows]
        finally:
            con.close()

    # === Equity ===

    def record_equity(
        self,
        ts: pd.Timestamp,
        nav: float,
        cash: float,
        open_position_count: int,
        unrealized_pnl: float,
    ) -> None:
        """Insert or replace the equity snapshot at ts (ts is the primary key)."""
        con = self._connect()
        try:
            con.execute(
                "INSERT OR REPLACE INTO bot_equity_snapshots "
                "(ts, nav, cash, open_position_count, unrealized_pnl) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    _ts_to_ms(ts),
                    float(nav),
                    float(cash),
                    int(open_position_count),
                    float(unrealized_pnl),
                ),
            )
            con.commit()
        finally:
            con.close()

    def latest_equity(self) -> EquitySnapshot | None:
        """Most recent equity snapshot, or None if empty."""
        con = self._connect()
        try:
            row = con.execute(
                "SELECT ts, nav, cash, open_position_count, unrealized_pnl "
                "FROM bot_equity_snapshots ORDER BY ts DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            return EquitySnapshot(
                ts=_ms_to_ts(row["ts"]),
                nav=row["nav"],
                cash=row["cash"],
                open_position_count=row["open_position_count"],
                unrealized_pnl=row["unrealized_pnl"],
            )
        finally:
            con.close()

    # === Events ===

    def log_event(
        self,
        ts: pd.Timestamp,
        level: str,
        event_type: str,
        payload: dict,
    ) -> None:
        """Insert a bot_events row. Payload is json.dumps'd."""
        con = self._connect()
        try:
            con.execute(
                "INSERT INTO bot_events (ts, level, event_type, payload_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    _ts_to_ms(ts),
                    level,
                    event_type,
                    json.dumps(payload),
                ),
            )
            con.commit()
        finally:
            con.close()

    def recent_events(self, limit: int = 50, level: str | None = None) -> list[dict]:
        """Most recent events, filtered by level if provided.

        Returned as plain dicts (not dataclasses) because events are free-form
        JSON payloads — a dict is closer to how callers actually inspect them.
        """
        con = self._connect()
        try:
            if level is None:
                rows = con.execute(
                    "SELECT id, ts, level, event_type, payload_json "
                    "FROM bot_events ORDER BY ts DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT id, ts, level, event_type, payload_json "
                    "FROM bot_events WHERE level = ? ORDER BY ts DESC LIMIT ?",
                    (level, int(limit)),
                ).fetchall()
            return [
                {
                    "id": r["id"],
                    "ts": _ms_to_ts(r["ts"]),
                    "level": r["level"],
                    "event_type": r["event_type"],
                    "payload": json.loads(r["payload_json"]),
                }
                for r in rows
            ]
        finally:
            con.close()
