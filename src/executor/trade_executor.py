"""
Trade Executor — the ONLY component that places orders.

Gated behind TRADING_ENV; refuses to run in mainnet_readonly mode.
After an entry fills, immediately places both STOP_MARKET and TAKE_PROFIT_MARKET
as exchange-side orders so they survive a bot restart or disconnection.
"""

import logging

from src.api.client import BinanceClient
from src.config import AppConfig, TradingEnv
from src.risk.manager import RiskAssessment
from src.signals.signal import Direction

logger = logging.getLogger(__name__)


class ExecutorBlockedError(Exception):
    """Raised when execution is attempted in a non-trade-capable environment."""


class TradeExecutor:
    def __init__(self, client: BinanceClient, config: AppConfig) -> None:
        self.client = client
        self.config = config
        self._assert_trade_capable()

    def _assert_trade_capable(self) -> None:
        if self.config.binance.trading_env == TradingEnv.MAINNET_READONLY:
            raise ExecutorBlockedError(
                "TradeExecutor cannot be instantiated in MAINNET_READONLY mode. "
                "This is intentional — the current API key cannot place orders. "
                "Provide a trade-capable key and set TRADING_ENV=testnet or TRADING_ENV=live."
            )

    # Binance Futures minimum order notional (USDT) — enforced exchange-side.
    # $20 minimum + 15% buffer to survive step-size rounding (e.g. ETH rounds
    # 0.00832 → 0.008, dropping notional below $20).
    _MIN_NOTIONAL_USDT: float = 23.0

    def execute(self, assessment: RiskAssessment) -> dict:
        """
        Place the entry order, then immediately bracket it with a stop and TP.
        Returns a dict with all three order receipts.
        """
        signal = assessment.signal
        symbol = signal.symbol
        side = "buy" if signal.direction == Direction.LONG else "sell"
        close_side = "sell" if signal.direction == Direction.LONG else "buy"

        entry_price = signal.entry_price or 0.0

        # --- Minimum notional enforcement ---
        # Binance rejects orders below $5 notional. If the risk-based size comes
        # out too small (common with tiny balances), bump up to the minimum.
        if entry_price > 0:
            min_size = self._MIN_NOTIONAL_USDT / entry_price
            if assessment.position_size < min_size:
                logger.warning(
                    "Position size %.6f (notional $%.2f) below exchange minimum $%.2f — "
                    "bumping to %.6f. Risk % will be higher than configured for this trade.",
                    assessment.position_size,
                    assessment.position_size * entry_price,
                    self._MIN_NOTIONAL_USDT,
                    min_size,
                )
                assessment.position_size = min_size

        # Round to exchange step size (XRP: 0.1, BTC: 0.001, etc.)
        # ccxt handles this automatically when placing the order

        logger.info(
            "Executing entry | symbol=%s side=%s size=%.4f stop=%.4f tp=%.4f",
            symbol, side, assessment.position_size, assessment.stop_price, assessment.take_profit_price,
        )

        # 1. Set leverage — step down until Binance accepts it for this symbol
        desired = int(assessment.leverage_override) if assessment.leverage_override > 0 else int(self.config.risk.max_leverage)
        desired = max(1, desired)
        # Binance rejects leverage above the symbol's maximum (varies per pair).
        # Try desired, then step down through common caps until one works.
        leverage_candidates = sorted(
            {desired, 125, 100, 75, 50, 25, 20, 10, 5, 1},
            reverse=True,
        )
        leverage = 1
        for lev in leverage_candidates:
            if lev > desired:
                continue   # never go above what was requested
            try:
                self.client.set_leverage(symbol, lev)
                leverage = lev
                logger.info("Leverage set to %dx for %s", leverage, symbol)
                break
            except Exception as e:
                if "-4028" in str(e) or "not valid" in str(e).lower():
                    logger.info("Leverage %dx not valid for %s — trying lower", lev, symbol)
                    continue
                raise   # unexpected error — re-raise
        # Update assessment so position sizing stays consistent
        assessment.leverage_override = leverage

        # 2. Entry order (market)
        entry_order = self.client.create_market_order(symbol, side, assessment.position_size)
        logger.info("Entry order filled: %s", entry_order)

        # Use actual fill price for stop/TP validation to avoid -2021 errors
        actual_price = float(entry_order.get("average") or entry_order.get("price") or entry_price or 0)
        if actual_price <= 0:
            actual_price = entry_price

        total_size = assessment.position_size
        orders = {"entry": entry_order, "stop": None, "take_profit": None, "tp2": None, "tp3": None}

        # 3. Stop loss (full size)
        stop_price = assessment.stop_price
        stop_valid = (
            (signal.direction == Direction.LONG and stop_price < actual_price) or
            (signal.direction == Direction.SHORT and stop_price > actual_price)
        )
        if stop_valid and stop_price and stop_price > 0:
            try:
                stop_order = self.client.create_stop_market_order(
                    symbol, close_side, total_size, stop_price
                )
                logger.info("Stop order placed at %.5f", stop_price)
                orders["stop"] = stop_order
            except Exception as e:
                logger.warning("Stop order failed (%s) — no stop placed. Monitor manually!", e)
        else:
            logger.warning("Stop price %.5f invalid vs fill %.5f — skipping.", stop_price, actual_price)

        # 4. TP1 (50% of position) — primary target
        tp1 = assessment.take_profit_price
        tp2 = getattr(assessment, "tp2_price", None)
        tp3 = getattr(assessment, "tp3_price", None)

        def _tp_valid(tp):
            if not tp or tp <= 0:
                return False
            return (
                (signal.direction == Direction.LONG  and tp > actual_price) or
                (signal.direction == Direction.SHORT and tp < actual_price)
            )

        # Split sizes: 50% / 25% / 25%
        tp1_size = round(total_size * 0.50, 8) if (tp2 or tp3) else total_size
        tp2_size = round(total_size * 0.25, 8)
        tp3_size = total_size - tp1_size - (tp2_size if _tp_valid(tp2) else 0)

        if _tp_valid(tp1):
            try:
                tp1_order = self.client.create_take_profit_market_order(
                    symbol, close_side, tp1_size, tp1
                )
                logger.info("TP1 order placed at %.5f (size=%.4f)", tp1, tp1_size)
                orders["take_profit"] = tp1_order
            except Exception as e:
                logger.warning("TP1 order failed: %s", e)

        if _tp_valid(tp2):
            try:
                tp2_order = self.client.create_take_profit_market_order(
                    symbol, close_side, tp2_size, tp2
                )
                logger.info("TP2 order placed at %.5f (size=%.4f)", tp2, tp2_size)
                orders["tp2"] = tp2_order
            except Exception as e:
                logger.warning("TP2 order failed: %s", e)

        if _tp_valid(tp3):
            try:
                tp3_order = self.client.create_take_profit_market_order(
                    symbol, close_side, tp3_size, tp3
                )
                logger.info("TP3 order placed at %.5f (size=%.4f)", tp3, tp3_size)
                orders["tp3"] = tp3_order
            except Exception as e:
                logger.warning("TP3 order failed: %s", e)

        return orders
