"""
Trend-following strategy: EMA trend filter + RSI pullback trigger.

Entry logic:
  LONG:  close > EMA(50)  AND  RSI crosses above 50 (was <=50 previous bar)
  SHORT: close < EMA(50)  AND  RSI crosses below 50 (was >=50 previous bar)

Exit logic:
  LONG exit:  close crosses below EMA(50)
  SHORT exit: close crosses above EMA(50)

Stop: entry ± ATR(14) * stop_multiplier
Take profit (first partial): entry ± ATR(14) * tp_multiplier

S/R filter (optional, controlled by SR_LOOKBACK env var, default 50):
  LONG entries are skipped if price is near or above a resistance level
  (i.e. there's a ceiling close overhead).
  SHORT entries are skipped if price is near or below a support level.
  "Near" is defined as within SR_ZONE_ATR_MULT * ATR of the level (default 0.5).
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

from src.config import StrategyConfig
from src.signals.signal import Direction, Signal, SignalType
from src.strategy.base import Strategy


class TrendFollowingStrategy(Strategy):
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def name(self) -> str:
        return "trend_following_ema_rsi"

    def required_warmup_bars(self) -> int:
        # EMA-50 needs 50 bars; add a buffer for RSI and ATR periods
        return max(self.config.ema_period, self.config.rsi_period, self.config.atr_period) + 10

    def compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Attach EMA, RSI, and ATR columns. Returns NaN columns if too few bars."""
        df = data.copy()

        min_window = max(self.config.ema_period, self.config.rsi_period, self.config.atr_period)
        if len(df) < min_window:
            df[f"ema_{self.config.ema_period}"] = float("nan")
            df[f"rsi_{self.config.rsi_period}"] = float("nan")
            df[f"atr_{self.config.atr_period}"] = float("nan")
            return df

        df[f"ema_{self.config.ema_period}"] = EMAIndicator(
            close=df["close"], window=self.config.ema_period
        ).ema_indicator()

        df[f"rsi_{self.config.rsi_period}"] = RSIIndicator(
            close=df["close"], window=self.config.rsi_period
        ).rsi()

        df[f"atr_{self.config.atr_period}"] = AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=self.config.atr_period
        ).average_true_range()

        if self.config.adx_threshold > 0:
            try:
                adx_ind = ADXIndicator(df["high"], df["low"], df["close"], window=14)
                df["adx"] = adx_ind.adx()
            except Exception:
                df["adx"] = np.nan

        if self.config.volume_filter > 0:
            df["rel_volume"] = df["volume"] / df["volume"].rolling(20).mean()

        # --- MACD confluence ---
        if self.config.macd_filter:
            try:
                macd_ind = MACD(
                    close=df["close"],
                    window_fast=self.config.macd_fast,
                    window_slow=self.config.macd_slow,
                    window_sign=self.config.macd_signal,
                )
                df["macd_hist"] = macd_ind.macd_diff()   # histogram = MACD - Signal
                df["macd_line"] = macd_ind.macd()
                df["macd_signal_line"] = macd_ind.macd_signal()
            except Exception:
                df["macd_hist"] = np.nan

        # --- Support & Resistance levels ---
        if self.config.sr_lookback > 0:
            df["sr_support"], df["sr_resistance"] = zip(
                *[self._compute_sr(df, i, self.config.sr_lookback) for i in range(len(df))]
            )

        return df

    @staticmethod
    def _compute_sr(df: pd.DataFrame, idx: int, lookback: int):
        """
        For row `idx`, find the nearest support and resistance levels by
        identifying pivot highs/lows in the prior `lookback` bars.

        A pivot high = local high where both neighbours are lower.
        A pivot low  = local low  where both neighbours are higher.
        Returns (support, resistance) as floats, or (nan, nan) if not enough data.
        """
        start = max(0, idx - lookback)
        end = idx  # exclude current bar
        if end - start < 5:
            return np.nan, np.nan

        window = df.iloc[start:end]
        highs = window["high"].values
        lows  = window["low"].values
        close = df.iloc[idx]["close"]

        pivot_highs = []
        pivot_lows  = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                pivot_highs.append(highs[i])
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                pivot_lows.append(lows[i])

        # Nearest resistance = lowest pivot high above current close
        resistances_above = [h for h in pivot_highs if h > close]
        resistance = min(resistances_above) if resistances_above else np.nan

        # Nearest support = highest pivot low below current close
        supports_below = [l for l in pivot_lows if l < close]
        support = max(supports_below) if supports_below else np.nan

        return support, resistance

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        ema_col = f"ema_{self.config.ema_period}"
        rsi_col = f"rsi_{self.config.rsi_period}"
        atr_col = f"atr_{self.config.atr_period}"

        if len(data) < 2:
            return self._hold(data, "not enough bars")

        curr = data.iloc[-1]
        prev = data.iloc[-2]

        # Guard: indicators must be populated (NaN during warmup)
        for col in (ema_col, rsi_col, atr_col):
            if pd.isna(curr[col]):
                return self._hold(data, f"{col} is NaN (still warming up)")

        close = curr["close"]
        ema = curr[ema_col]
        rsi_now = curr[rsi_col]
        rsi_prev = prev[rsi_col]
        atr = curr[atr_col]
        ts = self._timestamp(curr)
        symbol = self.config.symbol

        # ------------------------------------------------------------------
        # Exit signals take priority over entry signals.
        # The caller (signal generator / risk manager) tracks current direction;
        # we emit EXIT whenever the trend filter flips. The position manager
        # decides whether an exit is actually needed.
        # ------------------------------------------------------------------

        # Long exit: price crosses below EMA
        if pd.notna(prev[ema_col]) and prev["close"] > prev[ema_col] and close <= ema:
            return Signal(
                timestamp=ts, symbol=symbol,
                signal_type=SignalType.EXIT, direction=Direction.FLAT,
                reason="price crossed below EMA — long trend ended",
            )

        # Short exit: price crosses above EMA
        if pd.notna(prev[ema_col]) and prev["close"] < prev[ema_col] and close >= ema:
            return Signal(
                timestamp=ts, symbol=symbol,
                signal_type=SignalType.EXIT, direction=Direction.FLAT,
                reason="price crossed above EMA — short trend ended",
            )

        # ------------------------------------------------------------------
        # Regime filters (ADX + volume) — skip entry in weak/choppy markets
        # ------------------------------------------------------------------

        if self.config.adx_threshold > 0 and "adx" in data.columns:
            adx_val = curr.get("adx", np.nan)
            if pd.isna(adx_val) or adx_val < self.config.adx_threshold:
                return self._hold(data, f"ADX {adx_val:.1f} < {self.config.adx_threshold}")

        if self.config.volume_filter > 0 and "rel_volume" in data.columns:
            rv = curr.get("rel_volume", np.nan)
            if pd.isna(rv) or rv < self.config.volume_filter:
                return self._hold(data, f"low volume {rv:.2f}x < {self.config.volume_filter}x avg")

        # ------------------------------------------------------------------
        # MACD confluence filter — histogram must agree with trade direction
        # LONG:  macd_hist > 0  (MACD line above signal → bullish momentum)
        # SHORT: macd_hist < 0  (MACD line below signal → bearish momentum)
        # ------------------------------------------------------------------
        if self.config.macd_filter and "macd_hist" in data.columns:
            macd_hist = curr.get("macd_hist", np.nan)
            if pd.isna(macd_hist):
                return self._hold(data, "MACD not ready (NaN)")
            # We store the value and check per-direction in the entry blocks below.

        # ------------------------------------------------------------------
        # S/R zone filter — avoid entering into a wall
        # ------------------------------------------------------------------
        if self.config.sr_lookback > 0 and "sr_support" in data.columns:
            zone = atr * self.config.sr_zone_atr_mult  # tolerance band
            sr_res = curr.get("sr_resistance", np.nan)
            sr_sup = curr.get("sr_support", np.nan)

            # For LONG: skip if resistance is too close overhead (ceiling blocks upside)
            # We'll check this just before the LONG entry below, stored here for reuse.
            # For SHORT: skip if support is too close below (floor blocks downside).
            # These are evaluated per-direction in the entry blocks below.

        # ------------------------------------------------------------------
        # Entry signals
        # ------------------------------------------------------------------

        thr = self.config.rsi_entry_threshold

        # Long: above EMA, RSI crosses above threshold
        if close > ema and rsi_now > thr and pd.notna(rsi_prev) and rsi_prev <= thr:
            # MACD check: histogram must be positive (bullish momentum)
            if self.config.macd_filter and "macd_hist" in data.columns:
                macd_hist = curr.get("macd_hist", np.nan)
                if pd.notna(macd_hist) and macd_hist <= 0:
                    return self._hold(
                        data,
                        f"LONG blocked: MACD histogram {macd_hist:.6f} ≤ 0 (bearish momentum)"
                    )
            # S/R check: skip if resistance is within zone_atr_mult * ATR overhead
            if self.config.sr_lookback > 0 and "sr_resistance" in data.columns:
                sr_res = curr.get("sr_resistance", np.nan)
                zone = atr * self.config.sr_zone_atr_mult
                if pd.notna(sr_res) and (sr_res - close) < zone:
                    return self._hold(
                        data,
                        f"LONG blocked: resistance {sr_res:.5f} only {sr_res-close:.5f} away (zone={zone:.5f})"
                    )
            stop = close - atr * self.config.atr_stop_multiplier
            tp = close + atr * self.config.atr_trail_multiplier
            return Signal(
                timestamp=ts, symbol=symbol,
                signal_type=SignalType.ENTRY, direction=Direction.LONG,
                entry_price=close,
                suggested_stop=stop,
                suggested_tp=tp,
                atr=atr,
                reason=f"close>{ema:.4f} EMA, RSI crossed above {thr} ({rsi_prev:.1f}→{rsi_now:.1f})",
            )

        # Short: below EMA, RSI crosses below threshold
        if close < ema and rsi_now < thr and pd.notna(rsi_prev) and rsi_prev >= thr:
            # MACD check: histogram must be negative (bearish momentum)
            if self.config.macd_filter and "macd_hist" in data.columns:
                macd_hist = curr.get("macd_hist", np.nan)
                if pd.notna(macd_hist) and macd_hist >= 0:
                    return self._hold(
                        data,
                        f"SHORT blocked: MACD histogram {macd_hist:.6f} ≥ 0 (bullish momentum)"
                    )
            # S/R check: skip if support is within zone_atr_mult * ATR below
            if self.config.sr_lookback > 0 and "sr_support" in data.columns:
                sr_sup = curr.get("sr_support", np.nan)
                zone = atr * self.config.sr_zone_atr_mult
                if pd.notna(sr_sup) and (close - sr_sup) < zone:
                    return self._hold(
                        data,
                        f"SHORT blocked: support {sr_sup:.5f} only {close-sr_sup:.5f} away (zone={zone:.5f})"
                    )
            stop = close + atr * self.config.atr_stop_multiplier
            tp = close - atr * self.config.atr_trail_multiplier
            return Signal(
                timestamp=ts, symbol=symbol,
                signal_type=SignalType.ENTRY, direction=Direction.SHORT,
                entry_price=close,
                suggested_stop=stop,
                suggested_tp=tp,
                atr=atr,
                reason=f"close<{ema:.4f} EMA, RSI crossed below {thr} ({rsi_prev:.1f}→{rsi_now:.1f})",
            )

        return self._hold(data, "no trigger")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hold(self, data: pd.DataFrame, reason: str) -> Signal:
        ts = self._timestamp(data.iloc[-1]) if len(data) > 0 else datetime.now(timezone.utc)
        return Signal(
            timestamp=ts, symbol=self.config.symbol,
            signal_type=SignalType.HOLD, direction=Direction.FLAT,
            reason=reason,
        )

    @staticmethod
    def _timestamp(row: pd.Series) -> datetime:
        ts = row.get("timestamp", None)
        if ts is None:
            return datetime.now(timezone.utc)
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        # pandas Timestamp or int ms
        return pd.Timestamp(ts, unit="ms" if isinstance(ts, (int, float)) else None).to_pydatetime().replace(tzinfo=timezone.utc)
