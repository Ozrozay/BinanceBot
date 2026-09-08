"""
backtesting.py Strategy wrapper.

This adapter makes backtesting.py call our TrendFollowingStrategy (or any Strategy
subclass) without duplicating a single line of indicator or signal logic.

How the delegation works:
  - backtesting.py calls init() once, then next() on every bar.
  - In next(), we reconstruct a DataFrame from the backtesting.py data arrays
    (all bars up to and including the current one), then call our strategy's
    generate_signal() on it.
  - The indicator columns were pre-computed on the full dataset before the
    Backtest() call, so they're available as extra columns in self.data.df.

Position sizing:
  - We replicate the risk manager's formula:
    size = (equity * risk_pct) / stop_distance
  - backtesting.py's buy()/sell() accept size in units of base currency.

Funding cost:
  - We check every bar whether we've crossed a funding time (00:00/08:00/16:00 UTC)
    while in a position, and accumulate the cost. It's deducted from reported equity
    in the metrics — backtesting.py doesn't have a native "recurring carry cost" hook.

Exit priority:
  - Exchange-side stop/TP are modelled via backtesting.py's sl=/tp= parameters.
  - Signal-based exit (EMA flip) is checked in next() and closes the position early
    if the trend filter reverses before the stop/TP is hit.
  - This covers the "no time-stop / dangling positions" known bug — both paths lead
    to an exit.
"""

import logging
from datetime import timezone

import pandas as pd
from backtesting import Strategy as BtStrategy

from src.signals.signal import Direction, SignalType
from src.strategy.base import Strategy as OurStrategy

logger = logging.getLogger(__name__)

# Funding times in UTC hours
FUNDING_HOURS = {0, 8, 16}


class BacktestStrategyWrapper(BtStrategy):
    """
    backtesting.py Strategy subclass that delegates all logic to our Strategy ABC.

    Class-level attributes are used (not __init__ args) because backtesting.py
    instantiates the strategy class itself.
    """

    # Set these before calling Backtest()
    our_strategy: OurStrategy = None       # type: ignore
    risk_per_trade_pct: float = 0.005
    max_leverage: float = 3.0
    atr_stop_multiplier: float = 1.5
    atr_trail_multiplier: float = 2.0
    use_trailing_stop: bool = False        # replace fixed TP with ATR trail
    funding_df: pd.DataFrame = None        # type: ignore
    source_df: pd.DataFrame = None         # type: ignore

    def init(self):
        self._last_signal_type = SignalType.HOLD
        self._last_direction = Direction.FLAT
        self._entry_time = None
        self._entry_price = None
        self._position_size_units = 0.0
        self._accumulated_funding_cost = 0.0
        self._last_funding_hour_checked: int | None = None
        self._current_stop_usd: float = 0.0   # tracks trailing stop in real USDT

    # FractionalBacktest internally scales all OHLCV prices by this factor
    # so it can trade in whole-number satoshi units. Our sl/tp (in real USDT)
    # must be multiplied by this same factor before passing to buy()/sell().
    _FU: float = 1 / 100e6   # default fractional_unit of FractionalBacktest

    def next(self):
        # Slice source_df (real USDT prices, indicator columns) by bar count.
        idx = len(self.data.Close)
        df = self.__class__.source_df.iloc[:idx]

        if len(df) < self.our_strategy.required_warmup_bars():
            return

        signal = self.our_strategy.generate_signal(df)

        equity = self.equity                              # real USDT (not scaled)
        close = float(df["close"].iloc[-1])              # real USDT from source_df
        bar_time = pd.Timestamp(self.data.index[-1])

        # --- Funding cost tracking ---
        if self.position and self._entry_time is not None:
            self._apply_funding_cost(bar_time, close)

        # --- Trailing stop update ---
        if self.position and self.__class__.use_trailing_stop and self._current_stop_usd > 0:
            fu = self.__class__._FU
            trail = self.__class__.atr_trail_multiplier
            atr_col = f"atr_{self.our_strategy.config.atr_period}"
            atr_val = float(df[atr_col].iloc[-1]) if atr_col in df.columns else 0.0
            if atr_val > 0:
                if self._last_direction == Direction.LONG:
                    new_stop = close - atr_val * trail
                    if new_stop > self._current_stop_usd:
                        self._current_stop_usd = new_stop
                        try:
                            self.position.sl = new_stop * fu
                        except Exception:
                            pass
                else:
                    new_stop = close + atr_val * trail
                    if new_stop < self._current_stop_usd:
                        self._current_stop_usd = new_stop
                        try:
                            self.position.sl = new_stop * fu
                        except Exception:
                            pass

        # --- Exit handling (signal-based; stop/TP handled by backtesting.py) ---
        if self.position and signal.signal_type == SignalType.EXIT:
            self.position.close()
            self._on_position_closed()
            return

        # --- Entry handling ---
        if not self.position and signal.signal_type == SignalType.ENTRY:
            if signal.suggested_stop is None:
                return

            # All prices from signal are in real USDT
            stop_usd = signal.suggested_stop
            tp_usd = signal.suggested_tp

            if signal.direction == Direction.LONG:
                stop_distance = close - stop_usd
            else:
                stop_distance = stop_usd - close

            if stop_distance <= 0:
                return

            # --- Position sizing (mirrors RiskManager, computes in real USDT) ---
            risk_usd = equity * self.risk_per_trade_pct
            size_btc = risk_usd / stop_distance
            notional = size_btc * close
            if notional / equity > self.max_leverage:
                size_btc = (equity * self.max_leverage) / close

            # Convert to satoshis (FractionalBacktest's internal unit)
            size_sats = max(int(size_btc * 1e8), 1)

            # Scale sl/tp to FractionalBacktest's internal price units
            fu = self.__class__._FU
            sl_internal = stop_usd * fu

            use_trail = self.__class__.use_trailing_stop

            if signal.direction == Direction.LONG:
                if stop_usd >= close:
                    return
                tp_internal = None if use_trail else (
                    (tp_usd * fu) if (tp_usd is not None and tp_usd > close) else None
                )
                self.buy(size=size_sats, sl=sl_internal, tp=tp_internal)
            else:
                if stop_usd <= close:
                    return
                tp_internal = None if use_trail else (
                    (tp_usd * fu) if (tp_usd is not None and tp_usd < close) else None
                )
                self.sell(size=size_sats, sl=sl_internal, tp=tp_internal)

            self._entry_time = bar_time
            self._entry_price = close
            self._position_size_units = size_btc
            self._last_direction = signal.direction
            self._current_stop_usd = stop_usd

    def _apply_funding_cost(self, bar_time: pd.Timestamp, close: float) -> None:
        """Accumulate funding cost when we cross a funding settlement hour."""
        if self.funding_df is None or self.funding_df.empty:
            return

        hour = bar_time.hour
        if hour not in FUNDING_HOURS:
            return
        if hour == self._last_funding_hour_checked:
            return  # already charged for this settlement

        self._last_funding_hour_checked = hour

        # Find the nearest funding rate at or before this bar time
        mask = self.funding_df["timestamp"] <= bar_time
        if not mask.any():
            return

        rate = float(self.funding_df.loc[mask, "funding_rate"].iloc[-1])
        notional = self._position_size_units * close

        if self._last_direction == Direction.SHORT:
            rate = -rate  # short pays negative rate, receives positive

        cost = notional * rate
        self._accumulated_funding_cost += cost

    def _on_position_closed(self) -> None:
        self._entry_time = None
        self._entry_price = None
        self._position_size_units = 0.0
        self._last_funding_hour_checked = None
        self._current_stop_usd = 0.0
