"""Unit tests for TrendFollowingStrategy — pure logic, no network calls."""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timezone

from src.config import StrategyConfig
from src.signals.signal import Direction, SignalType
from src.strategy.trend_following import TrendFollowingStrategy


def make_config(**kwargs) -> StrategyConfig:
    defaults = dict(ema_period=5, rsi_period=5, atr_period=5, warmup_bars=10)
    defaults.update(kwargs)
    return StrategyConfig(**defaults)


def make_df(closes: list[float]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    n = len(closes)
    closes = np.array(closes, dtype=float)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        "open":  closes * 0.999,
        "high":  closes * 1.002,
        "low":   closes * 0.998,
        "close": closes,
        "volume": np.ones(n) * 1000,
    })
    return df


class TestTrendFollowingStrategy:
    def test_returns_hold_with_insufficient_bars(self):
        config = make_config()
        strat = TrendFollowingStrategy(config)
        df = make_df([100.0, 101.0])
        df = strat.compute_indicators(df)
        signal = strat.generate_signal(df)
        assert signal.signal_type == SignalType.HOLD

    def test_indicators_populated(self):
        config = make_config()
        strat = TrendFollowingStrategy(config)
        df = make_df([100 + i for i in range(30)])
        df = strat.compute_indicators(df)
        assert f"ema_{config.ema_period}" in df.columns
        assert f"rsi_{config.rsi_period}" in df.columns
        assert f"atr_{config.atr_period}" in df.columns
        # After enough bars, last row shouldn't be NaN
        assert pd.notna(df[f"ema_{config.ema_period}"].iloc[-1])

    def test_no_entry_signal_without_rsi_cross(self):
        """RSI already above 50 but no cross — should be HOLD."""
        config = make_config()
        strat = TrendFollowingStrategy(config)
        # Rising prices — EMA will be below close, RSI will be high throughout
        closes = [100 + i * 0.5 for i in range(30)]
        df = make_df(closes)
        df = strat.compute_indicators(df)
        # Force RSI to already be above 50 on both last two bars (no cross)
        df.loc[df.index[-1], f"rsi_{config.rsi_period}"] = 65.0
        df.loc[df.index[-2], f"rsi_{config.rsi_period}"] = 60.0
        signal = strat.generate_signal(df)
        # No RSI cross — should not be an entry
        assert signal.signal_type != SignalType.ENTRY or signal.direction == Direction.FLAT
