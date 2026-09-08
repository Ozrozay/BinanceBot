"""
Signal generator.

Sits between the strategy and the rest of the system.
Responsibilities:
  - Run compute_indicators + generate_signal on each new bar
  - Persist every signal to the database (observer path — never blocks trading)
  - Notify registered listeners (risk manager, logger, notifications)
"""

import logging
import sqlite3
from collections.abc import Callable

import pandas as pd

from src.signals.signal import Signal, SignalType
from src.strategy.base import Strategy

logger = logging.getLogger(__name__)

SignalListener = Callable[[Signal], None]


class SignalGenerator:
    def __init__(self, strategy: Strategy, db_conn: sqlite3.Connection) -> None:
        self.strategy = strategy
        self.db_conn = db_conn
        self._listeners: list[SignalListener] = []

    def add_listener(self, fn: SignalListener) -> None:
        """Register a callback that receives every signal. Used by the main loop."""
        self._listeners.append(fn)

    def process(self, data: pd.DataFrame) -> Signal:
        """
        Compute indicators, generate a signal, persist it, notify listeners.
        Returns the Signal so the caller can also inspect it.
        """
        data_with_indicators = self.strategy.compute_indicators(data)
        signal = self.strategy.generate_signal(data_with_indicators)

        self._persist(signal)

        if signal.signal_type != SignalType.HOLD:
            logger.info(
                "Signal: type=%s dir=%s entry=%s stop=%s tp=%s reason='%s'",
                signal.signal_type.value,
                signal.direction.value,
                f"{signal.entry_price:.2f}" if signal.entry_price else "n/a",
                f"{signal.suggested_stop:.2f}" if signal.suggested_stop else "n/a",
                f"{signal.suggested_tp:.2f}" if signal.suggested_tp else "n/a",
                signal.reason,
            )
        else:
            logger.debug("Signal: HOLD — %s", signal.reason)

        for listener in self._listeners:
            try:
                listener(signal)
            except Exception as e:
                # Observer failures must never affect the signal path
                logger.error("Signal listener error (ignored): %s", e)

        return signal

    def _persist(self, signal: Signal) -> None:
        try:
            self.db_conn.execute(
                """
                INSERT INTO signals
                  (ts, symbol, signal_type, direction, entry_price, stop_price,
                   tp_price, atr, reason, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.timestamp.isoformat(),
                    signal.symbol,
                    signal.signal_type.value,
                    signal.direction.value,
                    signal.entry_price,
                    signal.suggested_stop,
                    signal.suggested_tp,
                    signal.atr,
                    signal.reason,
                    self.strategy.name(),
                ),
            )
            self.db_conn.commit()
        except Exception as e:
            # DB write failure is never fatal to the trading loop
            logger.error("Failed to persist signal: %s", e)
