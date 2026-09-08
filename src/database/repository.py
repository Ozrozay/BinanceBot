"""
Database repository — all read/write operations on the SQLite database.

One class per table. Every method is safe to call from the trading loop:
failures are caught and logged, never re-raised.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class SignalRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert(
        self,
        ts: datetime,
        symbol: str,
        signal_type: str,
        direction: str,
        entry_price: Optional[float],
        stop_price: Optional[float],
        tp_price: Optional[float],
        atr: Optional[float],
        reason: str,
        strategy: str,
    ) -> Optional[int]:
        try:
            cur = self._conn.execute(
                """
                INSERT INTO signals
                  (ts, symbol, signal_type, direction, entry_price,
                   stop_price, tp_price, atr, reason, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ts.isoformat(), symbol, signal_type, direction,
                 entry_price, stop_price, tp_price, atr, reason, strategy),
            )
            self._conn.commit()
            return cur.lastrowid
        except Exception as e:
            logger.error("SignalRepository.insert failed: %s", e)
            return None

    def recent(self, limit: int = 20) -> list[sqlite3.Row]:
        try:
            return self._conn.execute(
                "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        except Exception as e:
            logger.error("SignalRepository.recent failed: %s", e)
            return []


class TradeRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def open_trade(
        self,
        symbol: str,
        direction: str,
        entry_ts: datetime,
        entry_price: float,
        position_size: float,
        risk_usd: float,
        strategy: str,
    ) -> Optional[int]:
        try:
            cur = self._conn.execute(
                """
                INSERT INTO trades
                  (symbol, direction, entry_ts, entry_price,
                   position_size, risk_usd, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (symbol, direction, entry_ts.isoformat(),
                 entry_price, position_size, risk_usd, strategy),
            )
            self._conn.commit()
            return cur.lastrowid
        except Exception as e:
            logger.error("TradeRepository.open_trade failed: %s", e)
            return None

    def close_trade(
        self,
        trade_id: int,
        exit_ts: datetime,
        exit_price: float,
        pnl_usd: float,
        exit_reason: str,
    ) -> None:
        try:
            self._conn.execute(
                """
                UPDATE trades
                SET exit_ts=?, exit_price=?, pnl_usd=?, exit_reason=?
                WHERE id=?
                """,
                (exit_ts.isoformat(), exit_price, pnl_usd, exit_reason, trade_id),
            )
            self._conn.commit()
        except Exception as e:
            logger.error("TradeRepository.close_trade failed: %s", e)

    def daily_pnl(self, date: Optional[str] = None) -> float:
        """Return total realised PnL for a given date (YYYY-MM-DD), default today."""
        day = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(pnl_usd), 0) FROM trades WHERE exit_ts LIKE ?",
                (f"{day}%",),
            ).fetchone()
            return float(row[0]) if row else 0.0
        except Exception as e:
            logger.error("TradeRepository.daily_pnl failed: %s", e)
            return 0.0

    def open_trades(self) -> list[sqlite3.Row]:
        try:
            return self._conn.execute(
                "SELECT * FROM trades WHERE exit_ts IS NULL"
            ).fetchall()
        except Exception as e:
            logger.error("TradeRepository.open_trades failed: %s", e)
            return []

    def closed_trades_since(self, date_str: str) -> list[sqlite3.Row]:
        """Return all closed trades on or after date_str (YYYY-MM-DD UTC)."""
        try:
            return self._conn.execute(
                "SELECT * FROM trades WHERE exit_ts IS NOT NULL AND exit_ts >= ? ORDER BY id DESC",
                (date_str,),
            ).fetchall()
        except Exception as e:
            logger.error("TradeRepository.closed_trades_since failed: %s", e)
            return []

    def pair_win_rate(self, symbol: str, direction: str, limit: int = 20) -> dict:
        """
        Return win rate stats for a specific pair + direction.
        Returns dict with: trades, wins, losses, win_rate, enough_data (bool)
        """
        try:
            rows = self._conn.execute(
                """
                SELECT pnl_usd FROM trades
                WHERE exit_ts IS NOT NULL
                  AND symbol = ?
                  AND direction = ?
                ORDER BY id DESC LIMIT ?
                """,
                (symbol, direction.lower(), limit),
            ).fetchall()
            if not rows:
                return {"trades": 0, "wins": 0, "losses": 0, "win_rate": None, "enough_data": False}
            wins   = sum(1 for r in rows if float(r[0] or 0) > 0)
            losses = len(rows) - wins
            return {
                "trades":      len(rows),
                "wins":        wins,
                "losses":      losses,
                "win_rate":    wins / len(rows),
                "enough_data": len(rows) >= 5,   # need at least 5 trades to judge
            }
        except Exception as e:
            logger.error("TradeRepository.pair_win_rate failed: %s", e)
            return {"trades": 0, "wins": 0, "losses": 0, "win_rate": None, "enough_data": False}

    def recent_closed(self, limit: int = 10) -> list[sqlite3.Row]:
        try:
            return self._conn.execute(
                "SELECT * FROM trades WHERE exit_ts IS NOT NULL ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        except Exception as e:
            logger.error("TradeRepository.recent_closed failed: %s", e)
            return []

    def all_closed(self) -> list[sqlite3.Row]:
        """All closed trades, newest first."""
        try:
            return self._conn.execute(
                "SELECT * FROM trades WHERE exit_ts IS NOT NULL ORDER BY id DESC"
            ).fetchall()
        except Exception as e:
            logger.error("TradeRepository.all_closed failed: %s", e)
            return []

    def consecutive_losses(self) -> int:
        """Count of consecutive losses at the tail of trade history."""
        try:
            rows = self._conn.execute(
                "SELECT pnl_usd FROM trades WHERE exit_ts IS NOT NULL ORDER BY id DESC LIMIT 20"
            ).fetchall()
            count = 0
            for r in rows:
                if float(r[0] or 0) < 0:
                    count += 1
                else:
                    break
            return count
        except Exception as e:
            logger.error("TradeRepository.consecutive_losses failed: %s", e)
            return 0

    def weekly_pnl(self, week_start: str) -> list[sqlite3.Row]:
        """All closed trades from week_start (YYYY-MM-DD) onwards."""
        try:
            return self._conn.execute(
                "SELECT * FROM trades WHERE exit_ts IS NOT NULL AND exit_ts >= ? ORDER BY id DESC",
                (week_start,),
            ).fetchall()
        except Exception as e:
            logger.error("TradeRepository.weekly_pnl failed: %s", e)
            return []


class PositionRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        position_size: float,
        stop_price: float,
        tp_price: Optional[float],
        opened_ts: datetime,
        stop_order_id: Optional[str] = None,
        tp_order_id: Optional[str] = None,
    ) -> None:
        try:
            self._conn.execute(
                """
                INSERT INTO positions
                  (symbol, direction, entry_price, position_size, stop_price,
                   tp_price, opened_ts, stop_order_id, tp_order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                  direction=excluded.direction,
                  entry_price=excluded.entry_price,
                  position_size=excluded.position_size,
                  stop_price=excluded.stop_price,
                  tp_price=excluded.tp_price,
                  opened_ts=excluded.opened_ts,
                  stop_order_id=excluded.stop_order_id,
                  tp_order_id=excluded.tp_order_id
                """,
                (symbol, direction, entry_price, position_size, stop_price,
                 tp_price, opened_ts.isoformat(), stop_order_id, tp_order_id),
            )
            self._conn.commit()
        except Exception as e:
            logger.error("PositionRepository.upsert failed: %s", e)

    def get(self, symbol: str) -> Optional[sqlite3.Row]:
        try:
            return self._conn.execute(
                "SELECT * FROM positions WHERE symbol=?", (symbol,)
            ).fetchone()
        except Exception as e:
            logger.error("PositionRepository.get failed: %s", e)
            return None

    def delete(self, symbol: str) -> None:
        try:
            self._conn.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
            self._conn.commit()
        except Exception as e:
            logger.error("PositionRepository.delete failed: %s", e)
