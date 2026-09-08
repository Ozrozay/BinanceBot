"""
Multi-Pair Scanner — runs the full strategy filter stack across multiple
symbols every 4H bar and reports the best setup.

For each pair it:
  1. Fetches the last N candles via REST
  2. Computes all indicators (EMA, RSI, ATR, ADX, volume, MACD, S/R)
  3. Scores how many filters pass (0–6)
  4. Returns a ranked list of ScanResult objects

The scanner does NOT execute trades — it reports via Telegram and
optionally signals main.py to execute the top-ranked pair.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.trend_following import TrendFollowingStrategy
from src.config import StrategyConfig
from src.signals.signal import Direction, SignalType

logger = logging.getLogger(__name__)

# Default pairs to scan — can be overridden via SCAN_PAIRS env var
DEFAULT_SCAN_PAIRS = [
    "XRP/USDT:USDT",
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "DOGE/USDT:USDT",
    "ADA/USDT:USDT",
    "AVAX/USDT:USDT",
]


@dataclass
class ScanResult:
    symbol: str
    direction: Optional[str]        # "LONG", "SHORT", or None
    signal_type: str                # "ENTRY", "HOLD", "EXIT"
    score: int                      # 0–6 filters passed
    filters: dict                   # {"adx": True, "volume": False, ...}
    reason: str
    price: float = 0.0
    stop: float = 0.0
    tp: float = 0.0
    atr: float = 0.0


class PairScanner:
    def __init__(
        self,
        client,
        base_config: StrategyConfig,
        timeframe: str = "4h",
        lookback: int = 200,
    ) -> None:
        self._client        = client
        self._base_cfg      = base_config
        self._timeframe     = timeframe
        self._lookback      = lookback
        self._last_results  = []   # cached for Gemini chat context

    def scan(self, pairs: list[str]) -> list[ScanResult]:
        """
        Scan all pairs. Returns results sorted by score descending
        (best setups first).
        """
        results = []
        for symbol in pairs:
            try:
                result = self._scan_one(symbol)
                results.append(result)
                logger.info(
                    "Scan %s: score=%d signal=%s dir=%s reason=%s",
                    symbol, result.score, result.signal_type,
                    result.direction or "—", result.reason[:60],
                )
            except Exception as e:
                logger.warning("Scan failed for %s: %s", symbol, e)
                results.append(ScanResult(
                    symbol=symbol, direction=None,
                    signal_type="ERROR", score=0,
                    filters={}, reason=str(e),
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        self._last_results = results   # cache for Gemini chat
        return results

    def _scan_one(self, symbol: str) -> ScanResult:
        # Build a per-symbol config copy (same params, different symbol)
        cfg = StrategyConfig(
            symbol=symbol,
            timeframe=self._timeframe,
            ema_period=self._base_cfg.ema_period,
            rsi_period=self._base_cfg.rsi_period,
            atr_period=self._base_cfg.atr_period,
            atr_stop_multiplier=self._base_cfg.atr_stop_multiplier,
            atr_trail_multiplier=self._base_cfg.atr_trail_multiplier,
            rsi_entry_threshold=self._base_cfg.rsi_entry_threshold,
            adx_threshold=self._base_cfg.adx_threshold,
            volume_filter=self._base_cfg.volume_filter,
            sr_lookback=self._base_cfg.sr_lookback,
            sr_zone_atr_mult=self._base_cfg.sr_zone_atr_mult,
            macd_filter=self._base_cfg.macd_filter,
            macd_fast=self._base_cfg.macd_fast,
            macd_slow=self._base_cfg.macd_slow,
            macd_signal=self._base_cfg.macd_signal,
        )
        strategy = TrendFollowingStrategy(cfg)

        # Fetch candles
        raw = self._client.exchange.fetch_ohlcv(
            symbol, timeframe=self._timeframe, limit=self._lookback
        )
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.dropna().reset_index(drop=True)

        if len(df) < strategy.required_warmup_bars():
            return ScanResult(
                symbol=symbol, direction=None, signal_type="HOLD",
                score=0, filters={}, reason="insufficient data",
                price=float(df["close"].iloc[-1]) if len(df) else 0,
            )

        df = strategy.compute_indicators(df)
        last = df.iloc[-1]
        prev = df.iloc[-2]

        close   = float(last["close"])
        ema_col = f"ema_{cfg.ema_period}"
        rsi_col = f"rsi_{cfg.rsi_period}"
        atr_col = f"atr_{cfg.atr_period}"

        ema = float(last[ema_col]) if ema_col in df.columns and not pd.isna(last[ema_col]) else None
        rsi = float(last[rsi_col]) if rsi_col in df.columns and not pd.isna(last[rsi_col]) else None
        rsi_prev = float(prev[rsi_col]) if rsi_col in df.columns and not pd.isna(prev[rsi_col]) else None
        atr = float(last[atr_col]) if atr_col in df.columns and not pd.isna(last[atr_col]) else 0

        if ema is None or rsi is None or rsi_prev is None:
            return ScanResult(
                symbol=symbol, direction=None, signal_type="HOLD",
                score=0, filters={}, reason="indicators NaN (warming up)",
                price=close,
            )

        thr = cfg.rsi_entry_threshold

        # Determine base direction from EMA + RSI cross
        long_cross  = close > ema and rsi > thr and rsi_prev <= thr
        short_cross = close < ema and rsi < thr and rsi_prev >= thr

        if long_cross:
            direction = "LONG"
            stop = close - atr * cfg.atr_stop_multiplier
            tp   = close + atr * cfg.atr_trail_multiplier
        elif short_cross:
            direction = "SHORT"
            stop = close + atr * cfg.atr_stop_multiplier
            tp   = close - atr * cfg.atr_trail_multiplier
        else:
            # No cross — score partial filters for ranking potential
            direction = "LONG" if close > ema else "SHORT"
            stop = tp = 0.0

        # --- Score each filter ---
        filters = {}
        score = 0

        # 1. Trend (EMA)
        trend_ok = (direction == "LONG" and close > ema) or (direction == "SHORT" and close < ema)
        filters["trend"] = trend_ok
        if trend_ok: score += 1

        # 2. RSI cross
        rsi_ok = long_cross or short_cross
        filters["rsi_cross"] = rsi_ok
        if rsi_ok: score += 1

        # 3. ADX
        if cfg.adx_threshold > 0 and "adx" in df.columns:
            adx_val = float(last.get("adx", 0) or 0)
            adx_ok = not pd.isna(adx_val) and adx_val >= cfg.adx_threshold
            filters["adx"] = adx_ok
            if adx_ok: score += 1
        else:
            filters["adx"] = None   # filter disabled

        # 4. Volume
        if cfg.volume_filter > 0 and "rel_volume" in df.columns:
            rv = float(last.get("rel_volume", 0) or 0)
            vol_ok = not pd.isna(rv) and rv >= cfg.volume_filter
            filters["volume"] = vol_ok
            if vol_ok: score += 1
        else:
            filters["volume"] = None

        # 5. MACD
        if cfg.macd_filter and "macd_hist" in df.columns:
            mh = float(last.get("macd_hist", 0) or 0)
            macd_ok = (direction == "LONG" and mh > 0) or (direction == "SHORT" and mh < 0)
            filters["macd"] = macd_ok
            if macd_ok: score += 1
        else:
            filters["macd"] = None

        # 6. S/R zone clear
        if cfg.sr_lookback > 0 and "sr_resistance" in df.columns and atr > 0:
            zone = atr * cfg.sr_zone_atr_mult
            if direction == "LONG":
                sr_res = last.get("sr_resistance")
                sr_ok = pd.isna(sr_res) or (float(sr_res) - close) >= zone
            else:
                sr_sup = last.get("sr_support")
                sr_ok = pd.isna(sr_sup) or (close - float(sr_sup)) >= zone
            filters["sr_clear"] = sr_ok
            if sr_ok: score += 1
        else:
            filters["sr_clear"] = None

        signal_type = "ENTRY" if rsi_ok else "HOLD"
        reason = f"score {score}/6 — {'RSI cross ✅' if rsi_ok else 'no RSI cross'}"

        return ScanResult(
            symbol=symbol,
            direction=direction,
            signal_type=signal_type,
            score=score,
            filters=filters,
            reason=reason,
            price=close,
            stop=stop,
            tp=tp,
            atr=atr,
        )


def format_scan_report(results: list[ScanResult], htf_filter=None) -> str:
    """Format scanner results as a Telegram HTML message."""
    lines = ["🔍 <b>Multi-Pair Scan Results</b>"]

    entry_results = [r for r in results if r.signal_type == "ENTRY"]
    hold_results  = [r for r in results if r.signal_type not in ("ENTRY", "ERROR")]
    error_results = [r for r in results if r.signal_type == "ERROR"]

    if entry_results:
        lines.append("\n✅ <b>Entry Signals:</b>")
        for r in entry_results:
            # HTF check
            htf_ok = True
            htf_tag = ""
            if htf_filter:
                from src.signals.signal import Direction
                d = Direction.LONG if r.direction == "LONG" else Direction.SHORT
                htf_ok = htf_filter.allows(d)
                htf_tag = " 🟢HTF✅" if htf_ok else " 🔴HTF❌"

            base = r.symbol.split("/")[0]
            dir_icon = "📈" if r.direction == "LONG" else "📉"
            lines.append(
                f"{dir_icon} <b>{base}</b> {r.direction}  "
                f"[{r.score}/6]{htf_tag}\n"
                f"   Price: ${r.price:.5f}  Stop: ${r.stop:.5f}  TP: ${r.tp:.5f}\n"
                f"   {_filter_icons(r.filters)}"
            )
    else:
        lines.append("\n⏳ No entry signals this bar.")

    if hold_results:
        lines.append("\n📊 <b>Watching:</b>")
        for r in hold_results[:4]:   # top 4 by score
            base = r.symbol.split("/")[0]
            dir_icon = "📈" if r.direction == "LONG" else "📉"
            lines.append(
                f"  {dir_icon} {base} — {r.score}/6  {_filter_icons(r.filters)}"
            )

    if error_results:
        lines.append("\n⚠️ Errors: " + ", ".join(r.symbol.split("/")[0] for r in error_results))

    return "\n".join(lines)


def _filter_icons(filters: dict) -> str:
    """Return compact icon string for each filter."""
    labels = {
        "trend":    "EMA",
        "rsi_cross":"RSI",
        "adx":      "ADX",
        "volume":   "VOL",
        "macd":     "MACD",
        "sr_clear": "S/R",
    }
    parts = []
    for key, label in labels.items():
        val = filters.get(key)
        if val is None:
            continue   # filter disabled
        parts.append(f"{'✅' if val else '❌'}{label}")
    return "  ".join(parts)
