"""
SQLite schema setup and connection helper.

Tables:
  signals      — every signal emitted by the strategy engine
  trades       — every completed round-trip trade
  positions    — current open positions (reconciled against exchange)
"""

import sqlite3
from pathlib import Path


DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    signal_type TEXT    NOT NULL,   -- entry / exit / hold
    direction   TEXT    NOT NULL,   -- long / short / flat
    entry_price REAL,
    stop_price  REAL,
    tp_price    REAL,
    atr         REAL,
    reason      TEXT,
    strategy    TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    entry_ts        TEXT    NOT NULL,
    exit_ts         TEXT,
    entry_price     REAL    NOT NULL,
    exit_price      REAL,
    position_size   REAL    NOT NULL,
    pnl_usd         REAL,
    risk_usd        REAL,
    exit_reason     TEXT,           -- 'stop', 'take_profit', 'signal', 'manual'
    strategy        TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL UNIQUE,
    direction       TEXT    NOT NULL,
    entry_price     REAL    NOT NULL,
    position_size   REAL    NOT NULL,
    stop_price      REAL    NOT NULL,
    tp_price        REAL,
    opened_ts       TEXT    NOT NULL,
    stop_order_id   TEXT,
    tp_order_id     TEXT
);
"""


def get_connection(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> sqlite3.Connection:
    conn = get_connection(db_path)
    conn.executescript(DDL)
    conn.commit()
    return conn
