"""SQLite persistence for OHLCV bars.

Schema mirrors the Postgres target on the homeserver — migrating later means
swapping the connection, not rewriting the writes.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB = Path("data/waddle.db")

_SCHEMA = """
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


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute(_SCHEMA)
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
