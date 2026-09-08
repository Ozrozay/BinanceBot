"""
Risk Manager — the gatekeeper between signals and execution.

Responsibilities:
  1. Position sizing from risk%, entry price, and stop price (never picked independently)
  2. Leverage cap enforcement
  3. Daily loss limit check
  4. Max drawdown circuit breaker
  5. Returns an approved RiskAssessment or raises RiskVeto
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone

from src.config import RiskConfig
from src.signals.signal import Direction, Signal, SignalType

logger = logging.getLogger(__name__)


class RiskVeto(Exception):
    """Raised when the risk manager blocks a signal."""


@dataclass
class RiskAssessment:
    """Everything the executor needs to place the trade, fully risk-adjusted."""
    signal: Signal
    position_size: float          # quantity in base currency (e.g. BTC)
    implied_leverage: float       # informational; already capped
    stop_price: float
    take_profit_price: float      # first partial TP
    risk_usd: float               # dollar amount at risk on this trade
    leverage_override: int = 0    # if > 0, use this leverage instead of config max
    tp2_price: float = 0.0        # second take-profit target (25% of position)
    tp3_price: float = 0.0        # third take-profit target (remaining 25%)


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config

        # State that persists across signals within a session
        self._peak_equity: float | None = None
        self._daily_loss_usd: float = 0.0
        self._daily_loss_date: date = datetime.now(timezone.utc).date()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        signal: Signal,
        account_equity: float,
        current_price: float,
    ) -> RiskAssessment:
        """
        Validate a signal and return a RiskAssessment, or raise RiskVeto.

        Args:
            signal:          The signal from the strategy.
            account_equity:  Current total equity in USDT.
            current_price:   Latest market price for leverage calculation.
        """
        if signal.signal_type != SignalType.ENTRY:
            raise ValueError("evaluate() is only for ENTRY signals; EXIT/HOLD don't need sizing")

        self._reset_daily_loss_if_new_day()
        self._update_peak_equity(account_equity)

        # --- Circuit breakers first ---
        self._check_daily_loss_limit(account_equity)
        self._check_drawdown(account_equity)

        # --- Require stop price ---
        if signal.suggested_stop is None:
            raise RiskVeto("Signal has no suggested_stop; cannot size position without a stop distance")

        entry = signal.entry_price or current_price
        stop = signal.suggested_stop

        if signal.direction == Direction.LONG:
            stop_distance = entry - stop
        elif signal.direction == Direction.SHORT:
            stop_distance = stop - entry
        else:
            raise RiskVeto(f"Cannot size a {signal.direction} direction entry")

        if stop_distance <= 0:
            raise RiskVeto(
                f"Stop distance is non-positive ({stop_distance:.4f}); "
                f"entry={entry}, stop={stop}. Signal rejected."
            )

        # --- Position sizing formula (locked in brief) ---
        # position_size = (equity * risk_pct) / stop_distance
        risk_usd = account_equity * self.config.risk_per_trade_pct
        position_size = risk_usd / stop_distance   # units of base currency

        # --- Leverage cap ---
        position_notional = position_size * entry
        implied_leverage = position_notional / account_equity

        if implied_leverage > self.config.max_leverage:
            # Scale position down to fit within leverage cap
            capped_notional = account_equity * self.config.max_leverage
            position_size = capped_notional / entry
            implied_leverage = self.config.max_leverage
            logger.warning(
                "Position sized down to fit leverage cap (%dx): "
                "notional %.2f → %.2f USDT, size %.6f",
                self.config.max_leverage, position_notional, capped_notional, position_size,
            )
            # Recalculate actual risk after cap (will be less than risk_pct)
            risk_usd = position_size * stop_distance

        # --- Take-profit level ---
        if signal.suggested_tp is not None:
            tp = signal.suggested_tp
        else:
            # Fall back to 1:2 R if strategy didn't provide one
            if signal.direction == Direction.LONG:
                tp = entry + stop_distance * self.config.risk_per_trade_pct * 200
            else:
                tp = entry - stop_distance * self.config.risk_per_trade_pct * 200
            logger.debug("No suggested_tp from strategy; computed fallback TP=%.4f", tp)

        logger.info(
            "Risk approved | dir=%s size=%.6f entry=%.2f stop=%.2f tp=%.2f "
            "risk_usd=%.2f leverage=%.2fx equity=%.2f",
            signal.direction.value, position_size, entry, stop, tp,
            risk_usd, implied_leverage, account_equity,
        )

        return RiskAssessment(
            signal=signal,
            position_size=position_size,
            implied_leverage=implied_leverage,
            stop_price=stop,
            take_profit_price=tp,
            risk_usd=risk_usd,
        )

    def record_trade_pnl(self, pnl_usd: float) -> None:
        """
        Called by the position manager when a trade closes.
        Updates daily loss tracking.
        """
        self._reset_daily_loss_if_new_day()
        if pnl_usd < 0:
            self._daily_loss_usd += abs(pnl_usd)
            logger.info("Daily loss updated: -%.2f USDT (total today: %.2f)", abs(pnl_usd), self._daily_loss_usd)

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_daily_loss_limit(self, equity: float) -> None:
        daily_limit_usd = equity * self.config.daily_loss_limit_pct
        if self._daily_loss_usd >= daily_limit_usd:
            raise RiskVeto(
                f"Daily loss limit reached: lost {self._daily_loss_usd:.2f} USDT today "
                f"(limit {daily_limit_usd:.2f} = {self.config.daily_loss_limit_pct*100:.1f}% of {equity:.2f}). "
                "Trading halted for today."
            )

    def _check_drawdown(self, equity: float) -> None:
        if self._peak_equity is None:
            return
        drawdown = (self._peak_equity - equity) / self._peak_equity
        if drawdown >= self.config.max_drawdown_pct:
            raise RiskVeto(
                f"Max drawdown circuit breaker triggered: "
                f"{drawdown*100:.1f}% drawdown from peak {self._peak_equity:.2f} USDT. "
                "Manual review required before restarting."
            )

    def _update_peak_equity(self, equity: float) -> None:
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

    def _reset_daily_loss_if_new_day(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._daily_loss_date:
            self._daily_loss_usd = 0.0
            self._daily_loss_date = today
