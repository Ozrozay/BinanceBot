"""
Standard signal format shared between the Strategy layer and everything downstream.

The Strategy module produces Signal objects. The Risk Manager, Executor, and
backtester all speak this type — they never reach into strategy internals.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"   # close / no position


class SignalType(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"
    HOLD = "hold"   # no action this bar


@dataclass(frozen=True)
class Signal:
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    direction: Direction          # meaningful on ENTRY; FLAT on EXIT/HOLD

    # Price levels at signal time (strategy's best estimate; executor re-validates)
    entry_price: float | None = None
    suggested_stop: float | None = None    # strategy's ATR-based stop suggestion
    suggested_tp: float | None = None      # strategy's first take-profit suggestion

    # Metadata for logging / debugging — never drives execution decisions
    reason: str = ""              # human-readable: "EMA50 cross + RSI>50"
    atr: float | None = None      # ATR value used when computing the stop
