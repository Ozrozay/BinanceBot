"""
SmartScalp Strategy — derived from analysis of 12 Binance Futures trading videos.

Core concepts learned from the videos:
  - Small capital ($1) isolated futures trading
  - Short timeframe (5m/15m) momentum with trend filter
  - EMA crossover for direction bias
  - RSI for momentum confirmation
  - Volume spike for trade quality
  - ATR-based adaptive stop loss
  - Automated TP/SL execution

Strategy rules:
  LONG:
    - 15m EMA9 > EMA21 (trend is up on higher timeframe)
    - 5m RSI(14) crosses above 50 (momentum turning bullish)
    - 5m close > 5m EMA9
    - Volume > 1.5x 20-bar average (genuine move)
    - Not near a recent swing high (resistance check)

  SHORT:
    - 15m EMA9 < EMA21 (trend is down on higher timeframe)
    - 5m RSI(14) crosses below 50 (momentum turning bearish)
    - 5m close < 5m EMA9
    - Volume > 1.5x 20-bar average
    - Not near a recent swing low (support check)

  Stop loss:  entry ± ATR(14) * 1.5
  TP1:        entry ± ATR(14) * 2.0  (50% close)
  TP2:        entry ± ATR(14) * 3.5  (25% close)
  TP3:        entry ± ATR(14) * 5.0  (25% close)

  Risk per trade: fixed $1 isolated margin (FIXED_MARGIN_USD)
  Daily profit target: $1 net (DAILY_PROFIT_TARGET_USD)
  Daily loss limit: $0.50 (DAILY_LOSS_LIMIT_USD)
"""

import os
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

from src.config import StrategyConfig
from src.signals.signal import Direction, Signal, SignalType
from src.strategy.base import Strategy

logger = logging.getLogger(__name__)


class SmartScalpStrategy(Strategy):
    """
    EMA crossover + RSI momentum + volume filter scalping strategy.
    Designed for small capital ($1/trade) isolated Binance Futures trading.
    """

    # Fast/slow EMA periods for trend direction
    EMA_FAST = 9
    EMA_SLOW = 21

    # RSI settings
    RSI_PERIOD = 14
    RSI_MID    = 50

    # Volume filter: require this multiplier above rolling average
    VOLUME_MULT = float(os.environ.get("SCALP_VOLUME_MULT", "1.5"))

    # ATR multipliers
    ATR_STOP_MULT = float(os.environ.get("SCALP_ATR_STOP_MULT", "1.5"))
    ATR_TP1_MULT  = float(os.environ.get("SCALP_ATR_TP1_MULT",  "2.0"))
    ATR_TP2_MULT  = float(os.environ.get("SCALP_ATR_TP2_MULT",  "3.5"))
    ATR_TP3_MULT  = float(os.environ.get("SCALP_ATR_TP3_MULT",  "5.0"))

    # Resistance/support proximity filter (ATR multiples)
    SR_ZONE_MULT = float(os.environ.get("SCALP_SR_ZONE_MULT", "0.5"))
    SR_LOOKBACK  = int(os.environ.get("SCALP_SR_LOOKBACK", "20"))

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def name(self) -> str:
        return "smart_scalp_ema_rsi_vol"

    def required_warmup_bars(self) -> int:
        return max(self.EMA_SLOW, self.RSI_PERIOD, 14) + 30

    def compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if len(df) < self.required_warmup_bars():
            for col in ["ema_fast", "ema_slow", "rsi", "atr", "vol_avg"]:
                df[col] = float("nan")
            return df

        df["ema_fast"] = EMAIndicator(df["close"], window=self.EMA_FAST).ema_indicator()
        df["ema_slow"] = EMAIndicator(df["close"], window=self.EMA_SLOW).ema_indicator()
        df["rsi"]      = RSIIndicator(df["close"], window=self.RSI_PERIOD).rsi()
        df["atr"]      = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
        df["vol_avg"]  = df["volume"].rolling(window=20).mean()
        return df

    def generate_signal(self, data: pd.DataFrame, timestamp: datetime | None = None) -> Signal:
        df = self.compute_indicators(data)

        # Extract timestamp from last bar if not provided
        if timestamp is None:
            ts_val = df["timestamp"].iloc[-1] if "timestamp" in df.columns else None
            if ts_val is not None:
                timestamp = ts_val if hasattr(ts_val, "tzinfo") else ts_val.to_pydatetime()
            else:
                from datetime import timezone
                timestamp = datetime.now(timezone.utc)

        if len(df) < 3 or df["ema_fast"].isna().iloc[-1]:
            return self._hold(timestamp, "not enough data")

        c  = df.iloc[-1]   # current (closed) bar
        p  = df.iloc[-2]   # previous bar

        price    = float(c["close"])
        ema_fast = float(c["ema_fast"])
        ema_slow = float(c["ema_slow"])
        rsi_now  = float(c["rsi"])
        rsi_prev = float(p["rsi"])
        atr      = float(c["atr"])
        volume   = float(c["volume"])
        vol_avg  = float(c["vol_avg"])

        if atr == 0 or np.isnan(atr):
            return self._hold(timestamp, "ATR is zero")

        # ── Volume filter ────────────────────────────────────────────
        if vol_avg > 0 and volume < self.VOLUME_MULT * vol_avg:
            return self._hold(
                timestamp,
                f"low volume {volume/vol_avg:.2f}x < {self.VOLUME_MULT}x avg"
            )

        # ── Compute S/R levels ───────────────────────────────────────
        lookback = min(self.SR_LOOKBACK, len(df) - 1)
        recent_highs = df["high"].iloc[-lookback:].max()
        recent_lows  = df["low"].iloc[-lookback:].min()
        sr_zone = self.SR_ZONE_MULT * atr

        # ── LONG signal ──────────────────────────────────────────────
        # Trend: EMA fast > EMA slow (bullish)
        # Momentum: RSI crosses above 50
        # Price: above fast EMA
        # Resistance: not too close to recent high
        ema_trend_long  = ema_fast > ema_slow
        rsi_cross_long  = rsi_prev <= self.RSI_MID and rsi_now > self.RSI_MID
        price_above_ema = price > ema_fast
        not_near_resist = price < (recent_highs - sr_zone)

        if ema_trend_long and rsi_cross_long and price_above_ema and not_near_resist:
            stop = price - atr * self.ATR_STOP_MULT
            tp1  = price + atr * self.ATR_TP1_MULT
            tp2  = price + atr * self.ATR_TP2_MULT
            tp3  = price + atr * self.ATR_TP3_MULT
            logger.info(
                "SmartScalp LONG signal | price=%.5f ema_fast=%.5f rsi=%.1f vol=%.1fx",
                price, ema_fast, rsi_now, volume/vol_avg
            )
            return Signal(
                timestamp=timestamp,
                symbol=self.config.symbol,
                signal_type=SignalType.ENTRY,
                direction=Direction.LONG,
                entry_price=price,
                suggested_stop=stop,
                suggested_tp=tp1,
                suggested_tp2=tp2,
                suggested_tp3=tp3,
                atr=atr,
                reason=f"EMA9>{ema_fast:.4f} RSI cross↑{rsi_now:.1f} vol={volume/vol_avg:.1f}x",
            )

        # ── SHORT signal ─────────────────────────────────────────────
        # Trend: EMA fast < EMA slow (bearish)
        # Momentum: RSI crosses below 50
        # Price: below fast EMA
        # Support: not too close to recent low
        ema_trend_short = ema_fast < ema_slow
        rsi_cross_short = rsi_prev >= self.RSI_MID and rsi_now < self.RSI_MID
        price_below_ema = price < ema_fast
        not_near_support = price > (recent_lows + sr_zone)

        if ema_trend_short and rsi_cross_short and price_below_ema and not_near_support:
            stop = price + atr * self.ATR_STOP_MULT
            tp1  = price - atr * self.ATR_TP1_MULT
            tp2  = price - atr * self.ATR_TP2_MULT
            tp3  = price - atr * self.ATR_TP3_MULT
            logger.info(
                "SmartScalp SHORT signal | price=%.5f ema_fast=%.5f rsi=%.1f vol=%.1fx",
                price, ema_fast, rsi_now, volume/vol_avg
            )
            return Signal(
                timestamp=timestamp,
                symbol=self.config.symbol,
                signal_type=SignalType.ENTRY,
                direction=Direction.SHORT,
                entry_price=price,
                suggested_stop=stop,
                suggested_tp=tp1,
                suggested_tp2=tp2,
                suggested_tp3=tp3,
                atr=atr,
                reason=f"EMA9<{ema_fast:.4f} RSI cross↓{rsi_now:.1f} vol={volume/vol_avg:.1f}x",
            )

        # ── EXIT signals ─────────────────────────────────────────────
        # Exit long when EMA fast crosses below slow
        # Exit short when EMA fast crosses above slow
        prev_ema_fast = float(p["ema_fast"])
        prev_ema_slow = float(p["ema_slow"])

        ema_death_cross = prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow
        ema_gold_cross  = prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow

        if ema_death_cross:
            return Signal(
                timestamp=timestamp,
                symbol=self.config.symbol,
                signal_type=SignalType.EXIT,
                direction=Direction.LONG,
                entry_price=price,
                suggested_stop=price,
                suggested_tp=price,
                atr=atr,
                reason="EMA death cross — exit LONG",
            )

        if ema_gold_cross:
            return Signal(
                timestamp=timestamp,
                symbol=self.config.symbol,
                signal_type=SignalType.EXIT,
                direction=Direction.SHORT,
                entry_price=price,
                suggested_stop=price,
                suggested_tp=price,
                atr=atr,
                reason="EMA golden cross — exit SHORT",
            )

        reasons = []
        if not ema_trend_long and not ema_trend_short:
            reasons.append("EMA flat/ranging")
        if not rsi_cross_long and not rsi_cross_short:
            reasons.append(f"no RSI cross (rsi={rsi_now:.1f})")
        return self._hold(timestamp, " | ".join(reasons) if reasons else "no setup")

    # ── helpers ──────────────────────────────────────────────────────

    def _hold(self, timestamp: datetime, reason: str) -> Signal:
        return Signal(
            timestamp=timestamp,
            symbol=self.config.symbol,
            signal_type=SignalType.HOLD,
            direction=Direction.FLAT,
            entry_price=None,
            suggested_stop=None,
            suggested_tp=None,
            atr=None,
            reason=reason,
        )
