"""
Strategy interface.

Every strategy — trend-following, mean-reversion, whatever comes next — must
implement this protocol. The Risk Manager, Executor, and Backtester only ever
call generate_signal(); they know nothing about how indicators are calculated.
"""

from abc import ABC, abstractmethod

import pandas as pd

from src.signals.signal import Signal


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    Subclasses receive a DataFrame of OHLCV+indicator data and return a Signal.
    They must NOT:
      - place orders
      - access the database
      - call the Binance API
      - mutate shared state
    """

    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs and database records."""
        ...

    @abstractmethod
    def required_warmup_bars(self) -> int:
        """
        Minimum number of bars the strategy needs before generate_signal is valid.
        The data collector will not call generate_signal until this many bars are
        available.
        """
        ...

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        Evaluate the latest bar and return a Signal.

        Args:
            data: DataFrame with columns [timestamp, open, high, low, close, volume]
                  plus any indicator columns the strategy has previously computed.
                  The last row is the most recent closed bar.

        Returns:
            A Signal describing what the strategy wants to do.
        """
        ...

    def compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optional hook: compute and attach indicator columns to the DataFrame.

        Called by the data layer before generate_signal. Default is a no-op;
        override if the strategy needs to pre-compute columns on the full history
        rather than recalculating each bar.
        """
        return data
