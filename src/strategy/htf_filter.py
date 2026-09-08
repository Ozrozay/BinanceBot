"""
Higher Time-Frame (HTF) trend filter.

Fetches daily (or configurable) candles independently of the main 4H loop,
computes EMA on the daily close, and exposes the current HTF bias:

  LONG  — daily close > daily EMA  (uptrend on higher TF)
  SHORT — daily close < daily EMA  (downtrend on higher TF)
  FLAT  — EMA not ready (warming up)

Usage in main.py:
  htf = HTFFilter(client, symbol, timeframe="1d", ema_period=50)
  htf.refresh()                       # call once at startup + every N 4H bars
  if not htf.allows(signal.direction):
      # skip entry — trade is against the daily trend
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator

from src.signals.signal import Direction

logger = logging.getLogger(__name__)


class HTFFilter:
    def __init__(
        self,
        client,
        symbol: str,
        timeframe: str = "1d",
        ema_period: int = 50,
        lookback_bars: int = 100,
    ) -> None:
        self._client = client
        self._symbol = symbol
        self._timeframe = timeframe
        self._ema_period = ema_period
        self._lookback_bars = lookback_bars

        self._bias: Optional[Direction] = None   # LONG, SHORT, or None (flat/not ready)
        self._last_close: Optional[float] = None
        self._last_ema: Optional[float] = None
        self._last_refresh: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Fetch latest HTF candles and recompute bias. Call at startup and periodically."""
        try:
            df = self._fetch_candles()
            if df is None or len(df) < self._ema_period + 5:
                logger.warning("HTF filter: not enough bars to compute EMA (%d bars).", len(df) if df is not None else 0)
                self._bias = None
                return

            df["ema"] = EMAIndicator(close=df["close"], window=self._ema_period).ema_indicator()

            last = df.iloc[-1]
            close = float(last["close"])
            ema   = float(last["ema"])

            self._last_close = close
            self._last_ema   = ema
            self._last_refresh = datetime.now(timezone.utc)

            if pd.isna(ema):
                self._bias = None
            elif close > ema:
                self._bias = Direction.LONG
            else:
                self._bias = Direction.SHORT

            logger.info(
                "HTF filter refreshed (%s %s): close=%.5f EMA(%d)=%.5f → bias=%s",
                self._symbol, self._timeframe,
                close, self._ema_period, ema,
                self._bias.value if self._bias else "FLAT",
            )

        except Exception as e:
            logger.error("HTF filter refresh failed: %s", e, exc_info=True)
            self._bias = None

    def allows(self, direction: Direction) -> bool:
        """
        Return True if `direction` agrees with (or is not blocked by) the HTF bias.
        If HTF bias is not ready, allow all trades (fail-open).
        """
        if self._bias is None:
            return True   # warming up — don't block trades
        return self._bias == direction

    @property
    def bias(self) -> Optional[Direction]:
        return self._bias

    def status_line(self) -> str:
        """Human-readable status for Telegram /status."""
        if self._bias is None:
            return f"HTF ({self._timeframe}): not ready"
        age = ""
        if self._last_refresh:
            mins = int((datetime.now(timezone.utc) - self._last_refresh).total_seconds() / 60)
            age = f" (refreshed {mins}m ago)"
        return (
            f"HTF ({self._timeframe} EMA{self._ema_period}): "
            f"{'🟢 BULLISH' if self._bias == Direction.LONG else '🔴 BEARISH'}"
            f" — close={self._last_close:.5f} EMA={self._last_ema:.5f}{age}"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_candles(self) -> Optional[pd.DataFrame]:
        """Fetch OHLCV from Binance via the existing client."""
        try:
            # ccxt fetch_ohlcv returns list of [ts, o, h, l, c, v]
            raw = self._client.exchange.fetch_ohlcv(
                self._symbol,
                timeframe=self._timeframe,
                limit=self._lookback_bars,
            )
            if not raw:
                return None
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.dropna().reset_index(drop=True)
            return df
        except Exception as e:
            logger.error("HTF candle fetch failed: %s", e)
            return None
