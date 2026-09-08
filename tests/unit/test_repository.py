"""Unit tests for database repositories — uses in-memory SQLite."""

from datetime import datetime, timezone
import pytest
from src.database.schema import init_db
from src.database.repository import SignalRepository, TradeRepository, PositionRepository


@pytest.fixture
def conn():
    c = init_db(":memory:")
    yield c
    c.close()


class TestSignalRepository:
    def test_insert_and_retrieve(self, conn):
        repo = SignalRepository(conn)
        ts = datetime.now(timezone.utc)
        row_id = repo.insert(ts, "BTC/USDT:USDT", "entry", "long",
                             100_000.0, 98_000.0, 104_000.0, 800.0, "test", "strat_v1")
        assert row_id is not None
        rows = repo.recent(limit=5)
        assert len(rows) == 1
        assert rows[0]["signal_type"] == "entry"

    def test_failure_returns_none(self, conn):
        repo = SignalRepository(conn)
        # Pass wrong type to trigger an error
        result = repo.insert(None, None, None, None, None, None, None, None, None, None)
        # Should return None without raising
        assert result is None


class TestTradeRepository:
    def test_open_and_close(self, conn):
        repo = TradeRepository(conn)
        ts = datetime.now(timezone.utc)
        trade_id = repo.open_trade("BTC/USDT:USDT", "long", ts, 100_000.0, 0.05, 50.0, "strat")
        assert trade_id is not None

        repo.close_trade(trade_id, ts, 102_000.0, 100.0, "take_profit")
        closed = repo.recent_closed(limit=1)
        assert len(closed) == 1
        assert closed[0]["pnl_usd"] == 100.0
        assert closed[0]["exit_reason"] == "take_profit"

    def test_open_trades(self, conn):
        repo = TradeRepository(conn)
        ts = datetime.now(timezone.utc)
        repo.open_trade("BTC/USDT:USDT", "long", ts, 100_000.0, 0.05, 50.0, "strat")
        assert len(repo.open_trades()) == 1

    def test_daily_pnl(self, conn):
        repo = TradeRepository(conn)
        ts = datetime.now(timezone.utc)
        tid = repo.open_trade("BTC/USDT:USDT", "long", ts, 100_000.0, 0.05, 50.0, "strat")
        repo.close_trade(tid, ts, 101_000.0, 75.0, "signal")
        pnl = repo.daily_pnl()
        assert abs(pnl - 75.0) < 0.01


class TestPositionRepository:
    def test_upsert_and_get(self, conn):
        repo = PositionRepository(conn)
        ts = datetime.now(timezone.utc)
        repo.upsert("BTC/USDT:USDT", "long", 100_000.0, 0.05, 98_000.0, 104_000.0, ts)
        pos = repo.get("BTC/USDT:USDT")
        assert pos is not None
        assert pos["direction"] == "long"
        assert pos["entry_price"] == 100_000.0

    def test_upsert_overwrites(self, conn):
        repo = PositionRepository(conn)
        ts = datetime.now(timezone.utc)
        repo.upsert("BTC/USDT:USDT", "long", 100_000.0, 0.05, 98_000.0, None, ts)
        repo.upsert("BTC/USDT:USDT", "short", 99_000.0, 0.05, 101_000.0, None, ts)
        pos = repo.get("BTC/USDT:USDT")
        assert pos["direction"] == "short"

    def test_delete(self, conn):
        repo = PositionRepository(conn)
        ts = datetime.now(timezone.utc)
        repo.upsert("BTC/USDT:USDT", "long", 100_000.0, 0.05, 98_000.0, None, ts)
        repo.delete("BTC/USDT:USDT")
        assert repo.get("BTC/USDT:USDT") is None
