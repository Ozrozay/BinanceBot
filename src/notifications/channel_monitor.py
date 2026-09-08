"""
Telegram Channel Monitor — watches multiple signal channels and auto-executes trades.

Uses Telethon (user API) to read messages from public/private channels.
When a message matches the signal format, it is parsed and passed to the
on_signal_trade callback.
"""

import asyncio
import logging
import os
from typing import Callable, Awaitable, Optional, List

from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError

from src.notifications.telegram_commands import parse_channel_signal

logger = logging.getLogger(__name__)

SESSION_FILE = "data/telegram_session"


class ChannelMonitor:
    def __init__(
        self,
        api_id: int,
        api_hash: str,
        channels: List[str],         # list of channel URLs/usernames
        notify_chat_id: str,
        on_signal_trade: Callable[[dict], Awaitable[str]],
        notify_bot,
    ) -> None:
        self._api_id          = api_id
        self._api_hash        = api_hash
        self._channels        = channels if isinstance(channels, list) else [channels]
        self._notify_chat_id  = int(notify_chat_id)
        self._on_signal_trade = on_signal_trade
        self._notify_bot      = notify_bot
        self._client: Optional[TelegramClient] = None

    async def start(self) -> None:
        os.makedirs("data", exist_ok=True)
        self._client = TelegramClient(SESSION_FILE, self._api_id, self._api_hash)
        await self._client.start()

        logger.info("Channel monitor connected. Watching %d channel(s)...", len(self._channels))

        entities = []
        for ch in self._channels:
            try:
                entity = await self._client.get_entity(ch)
                name = entity.title if hasattr(entity, "title") else ch
                logger.info("Channel resolved: %s (id=%s)", name, entity.id)
                entities.append(entity)
            except Exception as e:
                logger.error("Could not resolve channel %s: %s", ch, e)

        if not entities:
            logger.error("No channels resolved — monitor inactive.")
            return

        @self._client.on(events.NewMessage(chats=entities))
        async def on_message(event):
            text = event.message.text or ""
            if not text.strip():
                return

            # Show which channel the message came from
            try:
                chat = await event.get_chat()
                source = getattr(chat, "title", "unknown channel")
            except Exception:
                source = "unknown channel"

            logger.info("[%s] message: %s", source, text[:80].replace("\n", " "))

            signal = parse_channel_signal(text.upper())
            if not signal:
                return

            logger.info(
                "[%s] Signal: %s %s SL=%.5f",
                source, signal["direction"], signal["symbol"], signal["stop_loss"],
            )

            try:
                preview = (
                    f"📡 <b>Signal — {source}</b>\n"
                    f"Pair: {signal['symbol']}\n"
                    f"Direction: {signal['direction']}\n"
                    f"Leverage: {signal['leverage']}x\n"
                    f"Stop Loss: {signal['stop_loss']}\n"
                    f"TP1: {signal['tp1']}  TP2: {signal['tp2']}  TP3: {signal['tp3']}\n"
                    f"Size: {signal['wallet_pct']*100:.0f}% of wallet\n\n"
                    f"Executing automatically..."
                )
                await self._notify_bot.send_message(
                    chat_id=self._notify_chat_id,
                    text=preview,
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.warning("Could not send signal preview: %s", e)

            try:
                result = await self._on_signal_trade(signal)
                await self._notify_bot.send_message(
                    chat_id=self._notify_chat_id,
                    text=result,
                    parse_mode="HTML",
                )
            except Exception as e:
                err = f"❌ Auto-execution failed: {e}"
                logger.error(err, exc_info=True)
                try:
                    await self._notify_bot.send_message(
                        chat_id=self._notify_chat_id,
                        text=err,
                    )
                except Exception:
                    pass

        await self._client.run_until_disconnected()

    async def stop(self) -> None:
        if self._client:
            await self._client.disconnect()
            logger.info("Channel monitor disconnected.")
