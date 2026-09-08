"""
Telegram notification system — observer only.

Sends alerts to your Telegram chat for key bot events:
  - Startup / shutdown
  - Entry and exit signals (with price levels)
  - Circuit breaker triggered (daily loss limit, max drawdown)
  - Trade closed (with P&L)
  - Daily summary

If Telegram is not configured (no token/chat_id in .env), all calls are silent no-ops.
A send failure never raises — it logs a warning and moves on.

Setup:
  1. Message @BotFather on Telegram → /newbot → copy the token
  2. Start a chat with your new bot, then visit:
     https://api.telegram.org/bot<TOKEN>/getUpdates
     to find your chat_id
  3. Add TELEGRAM_TOKEN and TELEGRAM_CHAT_ID to .env
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, enabled: bool = True) -> None:
        self._token = token
        self._chat_id = chat_id
        self._enabled = enabled and bool(token) and bool(chat_id)
        self._bot = None

        if self._enabled:
            try:
                from telegram import Bot
                self._bot = Bot(token=self._token)
                logger.info("Telegram notifier ready (chat_id=%s)", self._chat_id)
            except ImportError:
                logger.warning("python-telegram-bot not installed — notifications disabled")
                self._enabled = False
            except Exception as e:
                logger.warning("Telegram init failed: %s — notifications disabled", e)
                self._enabled = False

    # ------------------------------------------------------------------
    # High-level event methods (call these from the trading loop)
    # ------------------------------------------------------------------

    async def startup(self, trading_env: str, symbol: str, timeframe: str) -> None:
        env_emoji = {"mainnet_readonly": "👁", "testnet": "🟡", "live": "🔴"}.get(trading_env, "❓")
        await self._send(
            f"{env_emoji} *Bot started*\n"
            f"Env: `{trading_env}`\n"
            f"Symbol: `{symbol}` | TF: `{timeframe}`"
        )

    async def shutdown(self) -> None:
        await self._send("🛑 *Bot stopped*")

    async def signal_entry(
        self,
        direction: str,
        symbol: str,
        entry: float,
        stop: float,
        tp: Optional[float],
        reason: str,
    ) -> None:
        arrow = "📈 LONG" if direction == "long" else "📉 SHORT"
        risk_pct = abs(entry - stop) / entry * 100
        tp_str = f"`{tp:,.2f}`" if tp else "trailing"
        await self._send(
            f"{arrow} *Signal — {symbol}*\n"
            f"Entry : `{entry:,.2f}`\n"
            f"Stop  : `{stop:,.2f}` ({risk_pct:.2f}% risk)\n"
            f"TP    : {tp_str}\n"
            f"_Reason: {reason}_"
        )

    async def signal_exit(self, symbol: str, reason: str) -> None:
        await self._send(f"🚪 *Exit signal — {symbol}*\n_Reason: {reason}_")

    async def trade_closed(
        self,
        symbol: str,
        direction: str,
        pnl_usd: float,
        exit_reason: str,
    ) -> None:
        emoji = "✅" if pnl_usd >= 0 else "❌"
        sign = "+" if pnl_usd >= 0 else ""
        await self._send(
            f"{emoji} *Trade closed — {symbol}*\n"
            f"Direction : {direction}\n"
            f"P&L       : `{sign}{pnl_usd:.2f} USDT`\n"
            f"Reason    : {exit_reason}"
        )

    async def circuit_breaker(self, reason: str) -> None:
        await self._send(f"🚨 *CIRCUIT BREAKER TRIGGERED*\n{reason}")

    async def daily_summary(
        self,
        date: str,
        pnl_usd: float,
        trade_count: int,
        equity: float,
    ) -> None:
        sign = "+" if pnl_usd >= 0 else ""
        emoji = "📊"
        await self._send(
            f"{emoji} *Daily summary — {date}*\n"
            f"Trades : {trade_count}\n"
            f"P&L    : `{sign}{pnl_usd:.2f} USDT`\n"
            f"Equity : `{equity:,.2f} USDT`"
        )

    async def error(self, message: str) -> None:
        await self._send(f"⚠️ *Bot error*\n`{message}`")

    # ------------------------------------------------------------------
    # Synchronous wrapper — call from sync code (the signal listener)
    # ------------------------------------------------------------------

    def send_sync(self, coro) -> None:
        """
        Fire-and-forget a coroutine from synchronous code.
        Uses the running event loop if available, otherwise creates one.
        """
        if not self._enabled:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(coro)
            else:
                loop.run_until_complete(coro)
        except Exception as e:
            logger.warning("Telegram send_sync failed: %s", e)

    # ------------------------------------------------------------------
    # Internal sender
    # ------------------------------------------------------------------

    async def _send(self, text: str) -> None:
        if not self._enabled or self._bot is None:
            return
        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)
