"""
Smart Pair Selector — scans available pairs, checks budget, picks the best trade.

Flow:
  1. Scan all configured pairs using PairScanner
  2. Filter out pairs where HTF trend opposes the signal
  3. Pick the highest-scoring pair with an ENTRY signal
  4. Calculate position size from free balance
  5. Return a ready-to-execute trade dict

Used by auto-trading loop and /trade command so the bot is never
limited to a single symbol.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from src.strategy.scanner import PairScanner, ScanResult
from src.signals.signal import Direction

logger = logging.getLogger(__name__)

# Minimum free balance required to place any trade
MIN_FREE_BALANCE_USDT = 4.0


@dataclass
class SelectedTrade:
    symbol: str
    direction: str          # "LONG" or "SHORT"
    price: float
    stop: float
    tp: float
    atr: float
    notional: float         # total position size in USDT
    size: float             # quantity in base currency
    leverage: int
    margin: float           # notional / leverage
    score: int
    reason: str


class SmartPairSelector:
    def __init__(
        self,
        scanner: PairScanner,
        pairs: list[str],
        leverage: int = 5,
        min_notional: float = 23.0,
        htf_filter=None,
    ) -> None:
        self._scanner     = scanner
        self._pairs       = pairs
        self._leverage    = leverage
        self._min_notional = min_notional
        self._htf         = htf_filter

    def select(
        self,
        free_balance: float,
        wallet_pct: float = 1.0,   # fraction of free balance to use as margin
    ) -> Optional[SelectedTrade]:
        """
        Scan all pairs and return the best trade, or None if nothing qualifies.

        wallet_pct: 1.0 = use all free balance as margin (capped at 90% for safety)
        """
        if free_balance < MIN_FREE_BALANCE_USDT:
            logger.warning("Free balance $%.2f too low to trade (min $%.2f)", free_balance, MIN_FREE_BALANCE_USDT)
            return None

        # Run scanner
        results = self._scanner.scan(self._pairs)

        # Filter: must have ENTRY signal
        entries = [r for r in results if r.signal_type == "ENTRY" and r.price > 0]

        if not entries:
            logger.info("SmartPairSelector: no entry signals across %d pairs", len(self._pairs))
            return None

        # Filter: HTF bias must agree
        if self._htf:
            entries = [
                r for r in entries
                if self._htf.allows(
                    Direction.LONG if r.direction == "LONG" else Direction.SHORT
                )
            ]
            if not entries:
                logger.info("SmartPairSelector: all entries blocked by HTF filter")
                return None

        # Pick highest score (scanner already sorted)
        best: ScanResult = entries[0]
        logger.info(
            "SmartPairSelector: best pair=%s dir=%s score=%d/6",
            best.symbol, best.direction, best.score,
        )

        # Calculate size
        margin   = min(free_balance * wallet_pct, free_balance * 0.9)  # 90% safety cap
        notional = max(margin * self._leverage, self._min_notional)
        # Don't use more margin than we have
        notional = min(notional, free_balance * 0.9 * self._leverage)
        size     = notional / best.price

        return SelectedTrade(
            symbol    = best.symbol,
            direction = best.direction,
            price     = best.price,
            stop      = best.stop,
            tp        = best.tp,
            atr       = best.atr,
            notional  = notional,
            size      = size,
            leverage  = self._leverage,
            margin    = notional / self._leverage,
            score     = best.score,
            reason    = f"auto-selected {best.symbol} score={best.score}/6",
        )

    def select_for_amount(
        self,
        free_balance: float,
        amount_usd: float,
    ) -> Optional[SelectedTrade]:
        """
        Like select() but uses a specific dollar amount as margin.
        Used by /trade $5 command.
        """
        if free_balance < MIN_FREE_BALANCE_USDT:
            return None

        # Cap amount to 90% of free balance
        margin = min(amount_usd, free_balance * 0.9)
        if margin < MIN_FREE_BALANCE_USDT:
            logger.warning("Amount $%.2f too small after safety cap", margin)
            return None

        results = self._scanner.scan(self._pairs)
        entries = [r for r in results if r.signal_type == "ENTRY" and r.price > 0]

        if self._htf:
            entries = [
                r for r in entries
                if self._htf.allows(
                    Direction.LONG if r.direction == "LONG" else Direction.SHORT
                )
            ]

        if not entries:
            return None

        best     = entries[0]
        notional = max(margin * self._leverage, self._min_notional)
        notional = min(notional, free_balance * 0.9 * self._leverage)
        size     = notional / best.price

        return SelectedTrade(
            symbol    = best.symbol,
            direction = best.direction,
            price     = best.price,
            stop      = best.stop,
            tp        = best.tp,
            atr       = best.atr,
            notional  = notional,
            size      = size,
            leverage  = self._leverage,
            margin    = notional / self._leverage,
            score     = best.score,
            reason    = f"manual ${amount_usd:.2f} → {best.symbol} score={best.score}/6",
        )
