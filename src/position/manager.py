"""
Position Manager — tracks open positions and detects fills.

Responsibilities:
  - Load open position from DB on startup (bot restart recovery)
  - Record new positions after executor fills them
  - Detect stop/TP fills each bar by comparing bot state with exchange
  - Handle manual close via Telegram /close command
  - Feed closed-trade PnL to RiskManager for circuit breakers
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.api.client import BinanceClient
    from src.config import AppConfig
    from src.database.repository import TradeRepository, PositionRepository
    from src.notifications.telegram import TelegramNotifier
    from src.risk.manager import RiskAssessment, RiskManager

logger = logging.getLogger(__name__)


class PositionManager:
    def __init__(
        self,
        client: "BinanceClient",
        config: "AppConfig",
        trade_repo: "TradeRepository",
        position_repo: "PositionRepository",
        notifier: "TelegramNotifier",
        risk_manager: "RiskManager",
    ) -> None:
        self._client = client
        self._config = config
        self._trade_repo = trade_repo
        self._position_repo = position_repo
        self._notifier = notifier
        self._risk_manager = risk_manager
        self._current_trade_id: Optional[int] = None
        self._current: Optional[dict] = None
        self._bot_state = None   # injected by main after BotState is created

    # ------------------------------------------------------------------
    # Startup reconciliation
    # ------------------------------------------------------------------

    def load_from_db(self) -> None:
        """Reload any open position from DB on startup (handles restarts)."""
        open_trades = self._trade_repo.open_trades()
        if not open_trades:
            logger.info("No open position in DB — starting fresh.")
            return
        trade = dict(open_trades[-1])
        self._current_trade_id = trade["id"]
        self._current = trade
        logger.info(
            "Resumed open position from DB: id=%s symbol=%s dir=%s entry=%.2f",
            trade["id"], trade["symbol"], trade["direction"], trade["entry_price"],
        )

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def has_open_position(self) -> bool:
        return self._current is not None

    def get_current(self) -> Optional[dict]:
        return self._current

    # ------------------------------------------------------------------
    # Lifecycle: opened
    # ------------------------------------------------------------------

    def on_position_opened(self, assessment: "RiskAssessment", orders: dict) -> None:
        """Record a newly opened position after executor.execute() succeeds."""
        signal = assessment.signal
        entry_order = orders.get("entry", {})
        entry_price = (
            float(entry_order.get("average") or 0)
            or float(entry_order.get("price") or 0)
            or signal.entry_price
            or 0.0
        )

        trade_id = self._trade_repo.open_trade(
            symbol=signal.symbol,
            direction=signal.direction.value,
            entry_ts=signal.timestamp,
            entry_price=entry_price,
            position_size=assessment.position_size,
            risk_usd=assessment.risk_usd,
            strategy="trend_following_v1",
        )
        self._position_repo.upsert(
            symbol=signal.symbol,
            direction=signal.direction.value,
            entry_price=entry_price,
            position_size=assessment.position_size,
            stop_price=assessment.stop_price,
            tp_price=assessment.take_profit_price,
            opened_ts=signal.timestamp,
        )

        self._current_trade_id = trade_id
        self._current = {
            "id": trade_id,
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "entry_price": entry_price,
            "position_size": assessment.position_size,
            "stop_price": assessment.stop_price,
            "tp_price": assessment.take_profit_price,
        }
        logger.info("Position opened and saved: trade_id=%s entry=%.2f", trade_id, entry_price)

    # ------------------------------------------------------------------
    # Per-bar fill detection
    # ------------------------------------------------------------------

    async def check_for_fills(self) -> None:
        """
        Called each bar. If the bot thinks it has an open position but the exchange
        shows none, the stop or TP was triggered. Also moves stop to breakeven
        when TP1 is hit (position reduced but still open).
        """
        if not self.has_open_position():
            return

        symbol = self._current["symbol"]
        try:
            positions = self._client.fetch_positions(symbol)
        except Exception as e:
            logger.warning("check_for_fills: fetch_positions failed: %s", e)
            return

        # Determine how much of our symbol is still open on the exchange
        pos_size = 0.0
        for p in positions:
            contracts = float(p.get("contracts") or 0)
            if contracts == 0:
                contracts = abs(float(p.get("info", {}).get("positionAmt", 0) or 0))
            sym     = p.get("symbol", "")
            raw_sym = p.get("info", {}).get("symbol", "")
            if symbol in (sym, raw_sym) or symbol.replace("/", "").replace(":USDT", "") == raw_sym:
                pos_size = abs(contracts)
                break

        if pos_size < 1e-6:
            logger.info("Exchange shows position closed; reconciling with DB...")
            await self._reconcile_closed_position()
            return

        # --- Feature 2: Move stop to breakeven when TP1 hit (position reduced) ---
        original_size = float(self._current.get("position_size", 0))
        entry_price   = float(self._current.get("entry_price", 0))
        breakeven_set = self._current.get("breakeven_set", False)

        if (not breakeven_set and original_size > 0
                and pos_size < original_size * 0.75   # TP1 hit = position reduced by ~50%
                and entry_price > 0):
            direction = self._current["direction"]
            close_side = "sell" if direction == "long" else "buy"
            try:
                # Cancel existing stop orders and place new one at entry (breakeven)
                self._client.exchange.cancel_all_orders(symbol)
                self._client.create_stop_market_order(symbol, close_side, pos_size, entry_price)
                self._current["breakeven_set"] = True
                self._current["position_size"] = pos_size   # track reduced size
                logger.info(
                    "TP1 hit — stop moved to breakeven %.5f for %s (remaining size %.4f)",
                    entry_price, symbol, pos_size,
                )
                try:
                    await self._notifier._bot.send_message(
                        chat_id=self._notifier._chat_id,
                        text=(
                            f"🎯 <b>TP1 Hit!</b> — {symbol}\n"
                            f"50% position closed.\n"
                            f"🛡 Stop moved to <b>breakeven ${entry_price:.5f}</b>\n"
                            f"Remaining: {pos_size:.4f} units running free."
                        ),
                        parse_mode="HTML",
                    )
                except Exception:
                    pass
            except Exception as e:
                logger.warning("Could not move stop to breakeven: %s", e)

    async def _reconcile_closed_position(self) -> None:
        """Determine exit price/PnL from trade history and close the record in DB."""
        symbol = self._current["symbol"]
        direction = self._current["direction"]
        entry_price = float(self._current["entry_price"])
        size = float(self._current["position_size"])

        exit_price = entry_price  # fallback if we can't fetch trades
        try:
            my_trades = self._client.exchange.fetch_my_trades(symbol, limit=10)
            close_side = "sell" if direction == "long" else "buy"
            for t in reversed(my_trades):
                if t.get("side") == close_side and float(t.get("amount", 0)) > 0:
                    exit_price = float(t.get("price", entry_price))
                    break
        except Exception as e:
            logger.warning("Could not fetch trade history for PnL: %s — using entry as exit", e)

        pnl = (exit_price - entry_price) * size if direction == "long" else (entry_price - exit_price) * size
        exit_reason = "take_profit" if pnl >= 0 else "stop_loss"
        now = datetime.now(timezone.utc)

        self._trade_repo.close_trade(
            trade_id=self._current_trade_id,
            exit_ts=now,
            exit_price=exit_price,
            pnl_usd=pnl,
            exit_reason=exit_reason,
        )
        self._position_repo.delete(symbol)
        self._risk_manager.record_trade_pnl(pnl)

        logger.info(
            "Position closed: dir=%s entry=%.2f exit=%.2f pnl=%.2f reason=%s",
            direction, entry_price, exit_price, pnl, exit_reason,
        )

        try:
            await self._notifier.trade_closed(
                symbol=symbol,
                direction=direction,
                entry=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                reason=exit_reason,
            )
        except Exception as e:
            logger.warning("Telegram trade_closed notification failed: %s", e)

        self._current_trade_id = None
        self._current = None

        # --- Auto-pause after X consecutive losses ---
        import os as _os
        max_losses = int(_os.environ.get("MAX_CONSECUTIVE_LOSSES", "3"))
        consec = self._trade_repo.consecutive_losses()
        if consec >= max_losses and self._bot_state is not None and self._bot_state.auto_trade:
            self._bot_state.auto_trade = False
            logger.warning(
                "AUTO-PAUSED: %d consecutive losses hit. Auto-trade disabled.",
                consec,
            )
            try:
                await self._notifier._bot.send_message(
                    chat_id=self._notifier._chat_id,
                    text=(
                        f"⛔ <b>Auto-trading PAUSED</b>\n\n"
                        f"Bot recorded <b>{consec} losses in a row</b>.\n"
                        f"Auto-trade has been disabled to protect your account.\n\n"
                        f"Review your trades, then send <b>/auto on</b> to resume."
                    ),
                    parse_mode="HTML",
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Manual close (Telegram /close command)
    # ------------------------------------------------------------------

    async def get_all_open_positions(self) -> list:
        """Fetch all open futures positions from Binance (not just bot-tracked)."""
        try:
            positions = self._client.exchange.fetch_positions()
            return [
                p for p in positions
                if abs(float(p.get("contracts") or p.get("info", {}).get("positionAmt", 0) or 0)) > 1e-6
            ]
        except Exception as e:
            logger.warning("get_all_open_positions failed: %s", e)
            return []

    async def close_position_by_symbol(self, symbol: str, reason: str = "manual") -> str:
        """Close a specific position by symbol. Works even if not bot-tracked."""
        try:
            positions = self._client.exchange.fetch_positions([symbol])
            for p in positions:
                size = abs(float(p.get("contracts") or p.get("info", {}).get("positionAmt", 0) or 0))
                if size < 1e-6:
                    continue
                side = p.get("side", "long")
                close_side = "sell" if side == "long" else "buy"
                self._client.exchange.cancel_all_orders(symbol)
                order = self._client.create_market_order(symbol, close_side, size)
                # If this was bot-tracked, reconcile
                if self._current and self._current.get("symbol") == symbol:
                    await asyncio.sleep(1.5)
                    await self._reconcile_closed_position()
                return f"✅ Closed {symbol} ({side.upper()}, size={size:.4f})"
            return f"No open position found for {symbol}"
        except Exception as e:
            return f"❌ Close failed for {symbol}: {e}"

    async def close_all_positions(self) -> str:
        """Close ALL open futures positions."""
        positions = await self.get_all_open_positions()
        if not positions:
            return "No open positions to close."
        results = []
        for p in positions:
            symbol = p.get("symbol", "")
            size   = abs(float(p.get("contracts") or p.get("info", {}).get("positionAmt", 0) or 0))
            side   = p.get("side", "long")
            close_side = "sell" if side == "long" else "buy"
            try:
                self._client.exchange.cancel_all_orders(symbol)
                self._client.create_market_order(symbol, close_side, size)
                results.append(f"✅ {symbol} {side.upper()} closed")
            except Exception as e:
                results.append(f"❌ {symbol}: {e}")
        # Reconcile bot-tracked position
        if self._current:
            await asyncio.sleep(1.5)
            try:
                await self._reconcile_closed_position()
            except Exception:
                self._current = None
                self._current_trade_id = None
        return "🔴 <b>All positions closed:</b>\n" + "\n".join(results)

    async def close_position(self, reason: str = "manual") -> Optional[dict]:
        """
        Market-close the open position and cancel all bracket orders.
        Returns the closing order dict, or None if no position is open.
        """
        if not self.has_open_position():
            return None

        symbol = self._current["symbol"]
        direction = self._current["direction"]
        size = float(self._current["position_size"])
        close_side = "sell" if direction == "long" else "buy"

        logger.info(
            "Manual close: symbol=%s direction=%s size=%.6f reason=%s",
            symbol, direction, size, reason,
        )

        try:
            self._client.exchange.cancel_all_orders(symbol)
            logger.info("Cancelled all open orders for %s", symbol)
        except Exception as e:
            logger.warning("cancel_all_orders failed (proceeding with close): %s", e)

        order = self._client.create_market_order(symbol, close_side, size)

        # Brief wait for fill to land before reconciling
        await asyncio.sleep(1.5)
        await self._reconcile_closed_position()

        return order
