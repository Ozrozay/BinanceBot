"""Unit tests for the risk manager — no exchange connection needed."""

import pytest
from datetime import datetime, timezone

from src.config import RiskConfig
from src.risk.manager import RiskAssessment, RiskManager, RiskVeto
from src.signals.signal import Direction, Signal, SignalType


def make_long_signal(entry: float, stop: float, tp: float | None = None) -> Signal:
    return Signal(
        timestamp=datetime.now(timezone.utc),
        symbol="BTC/USDT:USDT",
        signal_type=SignalType.ENTRY,
        direction=Direction.LONG,
        entry_price=entry,
        suggested_stop=stop,
        suggested_tp=tp,
        reason="test",
    )


def make_short_signal(entry: float, stop: float) -> Signal:
    return Signal(
        timestamp=datetime.now(timezone.utc),
        symbol="BTC/USDT:USDT",
        signal_type=SignalType.ENTRY,
        direction=Direction.SHORT,
        entry_price=entry,
        suggested_stop=stop,
        reason="test",
    )


class TestPositionSizing:
    def test_long_position_size(self):
        """position_size = (equity * risk_pct) / stop_distance"""
        rm = RiskManager(RiskConfig(risk_per_trade_pct=0.01, max_leverage=10))
        signal = make_long_signal(entry=50_000, stop=49_000)
        result = rm.evaluate(signal, account_equity=10_000, current_price=50_000)

        # risk_usd = 10_000 * 0.01 = 100
        # stop_distance = 50_000 - 49_000 = 1_000
        # position_size = 100 / 1_000 = 0.1 BTC
        assert abs(result.position_size - 0.1) < 1e-9
        assert abs(result.risk_usd - 100.0) < 1e-6

    def test_short_position_size(self):
        rm = RiskManager(RiskConfig(risk_per_trade_pct=0.005, max_leverage=10))
        signal = make_short_signal(entry=50_000, stop=51_000)
        result = rm.evaluate(signal, account_equity=20_000, current_price=50_000)

        # risk_usd = 20_000 * 0.005 = 100
        # stop_distance = 51_000 - 50_000 = 1_000
        # position_size = 100 / 1_000 = 0.1 BTC
        assert abs(result.position_size - 0.1) < 1e-9

    def test_leverage_cap_scales_down(self):
        """When implied leverage > cap, position is scaled down."""
        rm = RiskManager(RiskConfig(risk_per_trade_pct=0.5, max_leverage=3))
        # entry=50_000, equity=10_000, risk=50%, stop_distance=100
        # uncapped size = 5_000 / 100 = 50 BTC → notional = 2_500_000 → leverage = 250x
        signal = make_long_signal(entry=50_000, stop=49_900)
        result = rm.evaluate(signal, account_equity=10_000, current_price=50_000)

        assert result.implied_leverage <= 3.0
        # capped notional = 10_000 * 3 = 30_000; size = 30_000 / 50_000 = 0.6
        assert abs(result.position_size - 0.6) < 1e-9


class TestCircuitBreakers:
    def test_no_stop_raises(self):
        rm = RiskManager(RiskConfig())
        signal = Signal(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT:USDT",
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            entry_price=50_000,
            suggested_stop=None,
            reason="test",
        )
        with pytest.raises(RiskVeto, match="no suggested_stop"):
            rm.evaluate(signal, account_equity=10_000, current_price=50_000)

    def test_daily_loss_limit(self):
        rm = RiskManager(RiskConfig(daily_loss_limit_pct=0.03))
        rm._daily_loss_usd = 300.0   # simulate 300 loss on 10k equity = 3%
        signal = make_long_signal(entry=50_000, stop=49_000)
        with pytest.raises(RiskVeto, match="Daily loss limit"):
            rm.evaluate(signal, account_equity=10_000, current_price=50_000)

    def test_drawdown_circuit_breaker(self):
        rm = RiskManager(RiskConfig(max_drawdown_pct=0.15))
        rm._peak_equity = 10_000.0
        signal = make_long_signal(entry=50_000, stop=49_000)
        # current equity represents 16% drawdown — should trip the breaker
        with pytest.raises(RiskVeto, match="drawdown"):
            rm.evaluate(signal, account_equity=8_400, current_price=50_000)

    def test_invalid_stop_direction(self):
        """Stop on wrong side of entry should be vetoed."""
        rm = RiskManager(RiskConfig())
        signal = make_long_signal(entry=50_000, stop=51_000)  # stop above entry for long
        with pytest.raises(RiskVeto, match="non-positive"):
            rm.evaluate(signal, account_equity=10_000, current_price=50_000)
