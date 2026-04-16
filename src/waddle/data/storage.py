"""SQLite persistence for OHLCV bars and funding rates.

Schema mirrors the Postgres target on the homeserver — migrating later means
swapping the connection, not rewriting the writes. Two OHLCV tables on
purpose: `ohlcv` for Binance spot (existing baseline) and `ohlcv_bybit` for
Bybit linear perpetuals (deployment target). Keeping them separate avoids the
data integrity footgun of mixing two price sources with different contract
types under the same `(symbol, timestamp)` key.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB = Path("data/waddle.db")

_SCHEMA_OHLCV = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol    TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    open      REAL,
    high      REAL,
    low       REAL,
    close     REAL,
    volume    REAL,
    PRIMARY KEY (symbol, timestamp)
)
"""

_SCHEMA_OHLCV_BYBIT = """
CREATE TABLE IF NOT EXISTS ohlcv_bybit (
    symbol    TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    open      REAL,
    high      REAL,
    low       REAL,
    close     REAL,
    volume    REAL,
    PRIMARY KEY (symbol, timestamp)
)
"""

_SCHEMA_FUNDING = """
CREATE TABLE IF NOT EXISTS funding_rates (
    symbol       TEXT NOT NULL,
    timestamp    INTEGER NOT NULL,
    funding_rate REAL NOT NULL,
    PRIMARY KEY (symbol, timestamp)
)
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute(_SCHEMA_OHLCV)
    con.execute(_SCHEMA_OHLCV_BYBIT)
    con.execute(_SCHEMA_FUNDING)
    return con


def save_ohlcv(df: pd.DataFrame, symbol: str, db_path: Path = DEFAULT_DB) -> int:
    """Upsert a DataFrame of OHLCV bars for a symbol. Returns number of rows written."""
    con = _connect(db_path)
    rows = [
        (
            symbol,
            int(ts.timestamp() * 1000),
            float(r["open"]),
            float(r["high"]),
            float(r["low"]),
            float(r["close"]),
            float(r["volume"]),
        )
        for ts, r in df.iterrows()
    ]
    con.executemany(
        "INSERT OR REPLACE INTO ohlcv (symbol, timestamp, open, high, low, close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()
    con.close()
    return len(rows)


def load_ohlcv(symbol: str, db_path: Path = DEFAULT_DB) -> pd.DataFrame:
    """Load all stored bars for a symbol, indexed by UTC timestamp."""
    con = _connect(db_path)
    df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume "
        "FROM ohlcv WHERE symbol = ? ORDER BY timestamp",
        con,
        params=(symbol,),
    )
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")


def save_ohlcv_bybit(df: pd.DataFrame, symbol: str, db_path: Path = DEFAULT_DB) -> int:
    """Upsert a DataFrame of Bybit linear-perp OHLCV bars. Returns rows written.

    Stored in a separate table from `ohlcv` to keep Binance spot and Bybit
    perps cleanly isolated. Symbol uses the CCXT unified format `BTC/USDT:USDT`
    so the storage layer preserves the contract information.
    """
    con = _connect(db_path)
    rows = [
        (
            symbol,
            int(ts.timestamp() * 1000),
            float(r["open"]),
            float(r["high"]),
            float(r["low"]),
            float(r["close"]),
            float(r["volume"]),
        )
        for ts, r in df.iterrows()
    ]
    con.executemany(
        "INSERT OR REPLACE INTO ohlcv_bybit (symbol, timestamp, open, high, low, close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()
    con.close()
    return len(rows)


def load_ohlcv_bybit(symbol: str, db_path: Path = DEFAULT_DB) -> pd.DataFrame:
    """Load all stored Bybit linear-perp bars for a symbol, indexed by UTC timestamp."""
    con = _connect(db_path)
    df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume "
        "FROM ohlcv_bybit WHERE symbol = ? ORDER BY timestamp",
        con,
        params=(symbol,),
    )
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")


def save_funding_rates(df: pd.DataFrame, symbol: str, db_path: Path = DEFAULT_DB) -> int:
    """Upsert a DataFrame of funding rate settlements for a symbol.

    Expects `df` indexed by UTC timestamp with a `funding_rate` column.
    Each row represents one funding settlement event (Bybit: 8h cadence).
    """
    con = _connect(db_path)
    rows = [
        (
            symbol,
            int(ts.timestamp() * 1000),
            float(r["funding_rate"]),
        )
        for ts, r in df.iterrows()
    ]
    con.executemany(
        "INSERT OR REPLACE INTO funding_rates (symbol, timestamp, funding_rate) "
        "VALUES (?, ?, ?)",
        rows,
    )
    con.commit()
    con.close()
    return len(rows)


def load_funding_rates(symbol: str, db_path: Path = DEFAULT_DB) -> pd.DataFrame:
    """Load all stored funding rate settlements for a symbol, indexed by UTC timestamp."""
    con = _connect(db_path)
    df = pd.read_sql_query(
        "SELECT timestamp, funding_rate "
        "FROM funding_rates WHERE symbol = ? ORDER BY timestamp",
        con,
        params=(symbol,),
    )
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")
