"""
Telegram command handler — receives messages and executes bot commands.

Uses a bare Bot + manual get_updates loop (no Application/Updater) so there
is zero conflict with the TelegramNotifier's Bot instance and no hidden
lifecycle polling that fights with our explicit calls.

Supported:
  /trade        — force-execute a trade right now
  /status       — show current position and bot state
  /close        — close the open position
  /auto on|off  — enable or disable automatic signal execution
  /help         — list all commands

  Plain text "trade" or "trade now" also triggers /trade.
  Pasting a channel signal message auto-parses and executes it.

Security: only responds to the TELEGRAM_CHAT_ID from .env.
"""

import asyncio
import logging
import re
from typing import Callable, Awaitable, TYPE_CHECKING, Optional

from telegram import Bot, Update

if TYPE_CHECKING:
    from src.position.manager import PositionManager

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Channel signal parser
# ------------------------------------------------------------------

def parse_channel_signal(text: str) -> Optional[dict]:
    """
    Parse a trading channel signal like:

        AIO/ USDT/ SHORT
        Entry at market price
        Cross 5X…50X
        Take Profit Targets:
        TP1: 0.04380
        TP2: 0.03710
        TP3: 0.02890
        Stoploss: 0.052
        Use only 1% of your wallet.

    Returns a dict with keys: symbol, direction, leverage, tp1, tp2, tp3,
    stop_loss, wallet_pct — or None if not a recognisable signal.
    """
    # Direction must be present
    direction_m = re.search(r'\b(LONG|SHORT)\b', text, re.IGNORECASE)
    if not direction_m:
        return None

    # Symbol: "AIO/ USDT" or "AIO/USDT" or "BTCUSDT" etc.
    symbol_m = re.search(
        r'([A-Z0-9]{2,10})\s*/?\s*USDT',
        text, re.IGNORECASE
    )
    if not symbol_m:
        return None

    base = symbol_m.group(1).upper()
    symbol = f"{base}/USDT:USDT"   # ccxt futures format

    direction = direction_m.group(1).upper()

    # Leverage — handles "5X", "200X", "Leverage 200X", "5X…200X" (takes last/highest number)
    # If range like "5X…200X", use the highest value the signal allows
    lev_matches = re.findall(r'(\d+)\s*[xX×]', text)
    if lev_matches:
        # If it's a range (e.g. 5x...200x), use the highest
        leverage = max(int(v) for v in lev_matches)
    else:
        leverage = 5

    # TPs
    tp1_m = re.search(r'TP1[:\s]+([0-9.]+)', text, re.IGNORECASE)
    tp2_m = re.search(r'TP2[:\s]+([0-9.]+)', text, re.IGNORECASE)
    tp3_m = re.search(r'TP3[:\s]+([0-9.]+)', text, re.IGNORECASE)

    # Stop loss — handles "Stoploss:", "Stop Loss:", "SL:", "🛑 Stoploss:"
    sl_m = re.search(r'stop\s*loss[:\s]+([0-9.]+)', text, re.IGNORECASE)
    if not sl_m:
        sl_m = re.search(r'stoploss[:\s]+([0-9.]+)', text, re.IGNORECASE)
    if not sl_m:
        sl_m = re.search(r'[🛑🔴]\s*stop\s*loss[:\s]+([0-9.]+)', text, re.IGNORECASE)
    if not sl_m:
        sl_m = re.search(r'[🛑🔴]\s*stoploss[:\s]+([0-9.]+)', text, re.IGNORECASE)
    if not sl_m:
        sl_m = re.search(r'\bSL[:\s]+([0-9.]+)', text, re.IGNORECASE)

    if not sl_m:
        return None   # stop loss is mandatory

    # Wallet %
    pct_m = re.search(r'(\d+\.?\d*)\s*%', text)
    wallet_pct = float(pct_m.group(1)) / 100 if pct_m else 0.01

    return {
        "symbol":     symbol,
        "direction":  direction,
        "leverage":   leverage,   # use signal's leverage (executor will cap at account max)
        "tp1":        float(tp1_m.group(1)) if tp1_m else None,
        "tp2":        float(tp2_m.group(1)) if tp2_m else None,
        "tp3":        float(tp3_m.group(1)) if tp3_m else None,
        "stop_loss":  float(sl_m.group(1)),
        "wallet_pct": wallet_pct,
    }


class TelegramCommandHandler:
    def __init__(
        self,
        token: str,
        chat_id: str,
        bot_state,
        on_force_trade: Callable[[], Awaitable[str]],
        on_force_trade_amount: Callable[[float], Awaitable[str]],
        position_manager: "PositionManager",
        on_signal_trade: Optional[Callable[[dict], Awaitable[str]]] = None,
        chat_handler=None,
        on_scan=None,
        trade_repo=None,
        client=None,
    ) -> None:
        self._token = token
        self._chat_id = int(chat_id)
        self._bot_state = bot_state
        self._on_force_trade = on_force_trade
        self._on_force_trade_amount = on_force_trade_amount
        self._on_signal_trade = on_signal_trade
        self._position_manager = position_manager
        self._chat_handler = chat_handler
        self._on_scan = on_scan
        self._trade_repo = trade_repo
        self._client = client
        self._bot: Optional[Bot] = None
        self._pending_signal: Optional[dict] = None      # signal awaiting YES/NO
        self._awaiting_leverage: Optional[dict] = None   # signal awaiting leverage input

    async def setup(self) -> None:
        self._bot = Bot(token=self._token)
        logger.info("TelegramCommandHandler ready.")

    async def start_polling(self) -> None:
        """
        Manual get_updates loop — runs as an asyncio task alongside the WebSocket.

        On startup, a previous Application/Updater long-poll may still be live at
        Telegram's servers. We handle telegram.error.Conflict by waiting 35 seconds
        (longer than any long-poll timeout) and retrying — no crash, no user action needed.
        """
        from telegram.error import Conflict, RetryAfter

        offset = 0
        while True:
            try:
                updates = await self._bot.get_updates(
                    offset=offset,
                    timeout=30,
                    allowed_updates=["message"],
                )
                for update in updates:
                    offset = update.update_id + 1
                    await self._handle_update(update)

            except asyncio.CancelledError:
                break

            except Conflict:
                # Previous bot session still has an open long-poll at Telegram.
                # Wait for it to expire then retry automatically.
                logger.warning(
                    "Telegram conflict — previous session still active. "
                    "Waiting 35s for it to expire..."
                )
                await asyncio.sleep(35)

            except RetryAfter as e:
                wait = int(str(e).split("Retry in ")[-1].split(" ")[0]) + 1
                logger.warning("Telegram rate limit — waiting %ds...", wait)
                await asyncio.sleep(wait)

            except Exception as e:
                logger.warning("Telegram polling error: %s — retrying in 5s", e)
                await asyncio.sleep(5)

    async def shutdown(self) -> None:
        if self._bot:
            try:
                await self._bot.close()
            except Exception as e:
                logger.warning("TelegramCommandHandler shutdown error: %s", e)

    # ------------------------------------------------------------------
    # Auth guard
    # ------------------------------------------------------------------

    def _authorized(self, update: Update) -> bool:
        return bool(update.effective_chat and update.effective_chat.id == self._chat_id)

    async def _send(self, text: str, parse_mode: str = "HTML") -> None:
        if self._bot:
            try:
                await self._bot.send_message(
                    chat_id=self._chat_id, text=text, parse_mode=parse_mode
                )
            except Exception as e:
                logger.warning("Failed to send Telegram reply: %s", e)

    # ------------------------------------------------------------------
    # Update dispatcher
    # ------------------------------------------------------------------

    async def _handle_update(self, update: Update) -> None:
        if not self._authorized(update):
            return
        if not update.message or not update.message.text:
            return

        text = update.message.text.strip()

        if text.startswith("/"):
            cmd = text.split()[0].lstrip("/").lower()
            args = text.split()[1:]
            await self._dispatch_command(cmd, args)
        else:
            await self._on_text(text.lower())

    async def _dispatch_command(self, cmd: str, args: list) -> None:
        if cmd in ("trade", "t"):
            await self._cmd_trade(args)
        elif cmd == "status":
            await self._cmd_status()
        elif cmd == "close":
            await self._cmd_close(args)
        elif cmd == "closeall":
            await self._cmd_closeall()
        elif cmd == "positions":
            await self._cmd_positions()
        elif cmd == "stats":
            await self._cmd_stats()
        elif cmd == "history":
            await self._cmd_history()
        elif cmd == "pnl":
            await self._cmd_pnl()
        elif cmd == "balance":
            await self._cmd_balance()
        elif cmd == "cancel":
            await self._cmd_cancel(args)
        elif cmd == "auto":
            await self._cmd_auto(args)
        elif cmd in ("help", "start"):
            await self._cmd_help()
        elif cmd == "scan":
            await self._cmd_scan()
        elif cmd == "signal":
            # /signal followed by the full signal text pasted after the command
            # e.g. /signal AIO/USDT SHORT ...
            await self._cmd_signal(args)
        else:
            await self._send(f"Unknown command /{cmd}. Type /help for the list.")

    async def _on_text(self, text: str) -> None:
        import time as _time

        # YES/NO confirmation for pending trade alert
        t_clean = text.strip().lower()
        if t_clean in ("yes", "y", "confirm", "execute", "go", "do it"):
            # Priority 1: user pasted a signal and said YES
            if self._pending_signal and self._on_signal_trade:
                sig = self._pending_signal
                self._pending_signal = None
                self._bot_state.pending_trade = None
                await self._send(f"✅ Executing {sig['direction']} on {sig['symbol']}...")
                result = await self._on_signal_trade(sig)
                await self._send(result)
                return
            # Priority 2: scanner found a hot setup
            pt = self._bot_state.pending_trade
            if pt:
                if _time.time() > pt.expires_at:
                    self._bot_state.pending_trade = None
                    await self._send("⏰ Trade offer expired. Wait for the next signal.")
                    return
                self._bot_state.pending_trade = None
                await self._send(f"✅ Executing {pt.direction} on {pt.symbol}...")
                result = await self._on_force_trade()
                await self._send(result)
                return
            await self._send("No pending trade to confirm. Wait for a signal alert.")
            return

        if t_clean in ("no", "n", "skip", "cancel", "nope"):
            if self._pending_signal:
                sym = self._pending_signal["symbol"]
                self._pending_signal = None
                await self._send(f"❌ Signal for {sym} cancelled.")
            elif self._bot_state.pending_trade:
                sym = self._bot_state.pending_trade.symbol
                self._bot_state.pending_trade = None
                await self._send(f"❌ Trade on {sym} cancelled. Watching for next signal...")
            else:
                await self._send("No pending trade to cancel.")
            return

        # Leverage reply — user responded with a number after signal was parsed
        if self._awaiting_leverage:
            lev_m = re.match(r'^\s*(\d+)\s*[xX]?\s*$', text.strip())
            if lev_m:
                sig = self._awaiting_leverage
                chosen_lev = int(lev_m.group(1))
                sig["leverage"] = chosen_lev
                self._awaiting_leverage = None
                self._pending_signal = sig
                icon = "🟢" if sig["direction"] == "LONG" else "🔴"
                tp_line = f"TP1: {sig['tp1']}"
                if sig.get("tp2"): tp_line += f"  TP2: {sig['tp2']}"
                if sig.get("tp3"): tp_line += f"  TP3: {sig['tp3']}"
                await self._send(
                    f"✅ <b>Leverage set to {chosen_lev}x</b>\n\n"
                    f"{icon} <b>{sig['symbol']}  {sig['direction']}</b>\n"
                    f"Leverage: <b>{chosen_lev}x</b>\n"
                    f"Stop Loss: {sig['stop_loss']}\n"
                    f"{tp_line}\n"
                    f"Size: {sig['wallet_pct']*100:.0f}% of wallet\n\n"
                    f"Reply <b>YES</b> to execute  |  <b>NO</b> to cancel"
                )
                return
            elif t_clean in ("no", "n", "cancel"):
                self._awaiting_leverage = None
                await self._send("❌ Signal cancelled.")
                return
            else:
                await self._send("Please reply with a number for leverage, e.g. <b>5</b> or <b>10</b>. Or say NO to cancel.")
                return

        # "trade 1$", "trade $1", "trade 1.5$", "trade $2.50"
        m = re.match(r'^trade\s+\$?(\d+\.?\d*)\$?$', text.strip())
        if m:
            amount = float(m.group(1))
            await self._send(f"Placing ${amount:.2f} trade...")
            result = await self._on_force_trade_amount(amount)
            await self._send(result)
            return

        if text in ("trade", "trade now"):
            await self._send("Got it — executing trade...")
            result = await self._on_force_trade()
            await self._send(result)
            return

        # Signal pasted directly into chat — auto execute immediately
        signal = parse_channel_signal(text.upper())
        if signal and self._on_signal_trade:
            icon = "🟢" if signal["direction"] == "LONG" else "🔴"
            tp_line = f"TP1: {signal['tp1']}"
            if signal.get("tp2"): tp_line += f"  TP2: {signal['tp2']}"
            if signal.get("tp3"): tp_line += f"  TP3: {signal['tp3']}"
            await self._send(
                f"📡 <b>Signal received — executing now...</b>\n\n"
                f"{icon} <b>{signal['symbol']}  {signal['direction']}</b>\n"
                f"Leverage: up to {signal['leverage']}x\n"
                f"Stop Loss: {signal['stop_loss']}\n"
                f"{tp_line}\n"
                f"Size: {signal['wallet_pct']*100:.0f}% of wallet"
            )
            result = await self._on_signal_trade(signal)
            await self._send(result)
            return

        # Natural-language chat (fallback)
        if self._chat_handler:
            try:
                if hasattr(self._chat_handler, "async_handle"):
                    reply = await asyncio.wait_for(
                        self._chat_handler.async_handle(text),
                        timeout=30,
                    )
                else:
                    reply = self._chat_handler.handle(text)
            except asyncio.TimeoutError:
                reply = "⏳ Took too long to respond. Try again."
            except Exception as e:
                reply = f"⚠️ Error: {e}"
            if reply:
                await self._send(reply)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    async def _cmd_trade(self, args: list = None) -> None:
        import re
        # /trade 1 or /trade $1 or /trade 1$
        if args:
            raw = " ".join(args).strip().lstrip("$").rstrip("$")
            try:
                amount = float(raw)
                await self._send(f"Placing ${amount:.2f} trade...")
                result = await self._on_force_trade_amount(amount)
                await self._send(result)
                return
            except ValueError:
                pass
        await self._send("Checking signal and executing trade...")
        result = await self._on_force_trade()
        await self._send(result)

    async def _cmd_status(self) -> None:
        await self._send(self._build_status())

    async def _cmd_close(self, args: list = None) -> None:
        # /close          → close bot-tracked position
        # /close SOL      → close specific symbol
        # /closeall       → close everything
        if args:
            symbol_raw = args[0].upper().strip()
            # Normalise: SOL → SOL/USDT:USDT
            if "/" not in symbol_raw:
                symbol_raw = f"{symbol_raw}/USDT:USDT"
            await self._send(f"Closing {symbol_raw}...")
            result = await self._position_manager.close_position_by_symbol(symbol_raw)
            await self._send(result)
            return
        if not self._position_manager.has_open_position():
            await self._send("No bot-tracked position. Use /positions to see all open, then /close SYMBOL.")
            return
        await self._send("Closing position now...")
        try:
            await self._position_manager.close_position(reason="telegram_manual")
            await self._send("✅ Position closed.")
        except Exception as e:
            await self._send(f"❌ Close failed: {e}")
            logger.error("Manual close via Telegram failed: %s", e, exc_info=True)

    async def _cmd_closeall(self) -> None:
        await self._send("⚠️ Closing ALL open positions...")
        result = await self._position_manager.close_all_positions()
        await self._send(result, parse_mode="HTML")

    async def _cmd_positions(self) -> None:
        """Show all open Binance futures positions with unrealised P&L."""
        try:
            positions = await self._position_manager.get_all_open_positions()
        except Exception as e:
            await self._send(f"❌ Could not fetch positions: {e}")
            return
        if not positions:
            await self._send("📋 No open positions.")
            return
        lines = ["📋 <b>Open Positions</b>\n"]
        for p in positions:
            symbol    = p.get("symbol", "?")
            side      = p.get("side", "?").upper()
            size      = abs(float(p.get("contracts") or 0))
            entry     = float(p.get("entryPrice") or p.get("info", {}).get("entryPrice", 0) or 0)
            mark      = float(p.get("markPrice")  or p.get("info", {}).get("markPrice", 0)  or 0)
            upnl      = float(p.get("unrealizedPnl") or p.get("info", {}).get("unRealizedProfit", 0) or 0)
            lev       = p.get("leverage") or p.get("info", {}).get("leverage", "?")
            icon      = "🟢" if upnl >= 0 else "🔴"
            sign      = "+" if upnl >= 0 else ""
            lines.append(
                f"{icon} <b>{symbol}</b>  {side}  {lev}x\n"
                f"  Entry: ${entry:.4f}  Mark: ${mark:.4f}\n"
                f"  Size: {size:.4f}  uPnL: <b>{sign}${upnl:.2f}</b>\n"
            )
        await self._send("\n".join(lines), parse_mode="HTML")

    async def _cmd_auto(self, args: list) -> None:
        if args and args[0].lower() == "off":
            self._bot_state.auto_trade = False
            await self._send("Auto-trading <b>DISABLED</b>.\nUse /trade to trade manually.")
        else:
            self._bot_state.auto_trade = True
            await self._send("Auto-trading <b>ENABLED</b>.\nBot will execute all signals automatically.")

    async def _cmd_scan(self) -> None:
        """Trigger an on-demand multi-pair scan."""
        if not hasattr(self, '_on_scan') or self._on_scan is None:
            await self._send("Scanner not configured. Set SCANNER_ENABLED=true in .env")
            return
        await self._send("🔍 Scanning pairs... (may take 10–20s)")
        try:
            result = await self._on_scan()
            await self._send(result)
        except Exception as e:
            await self._send(f"❌ Scan failed: {e}")

    async def _cmd_signal(self, args: list) -> None:
        """Execute a pasted channel signal via /signal command."""
        full_text = " ".join(args)
        signal = parse_channel_signal(full_text.upper())
        if not signal:
            await self._send(
                "❌ Could not parse signal. Make sure it includes direction (LONG/SHORT), "
                "a stop loss (Stoploss: X.XX), and a symbol."
            )
            return
        if not self._on_signal_trade:
            await self._send("Signal trading not configured.")
            return
        preview = (
            f"📡 <b>Signal detected</b>\n"
            f"Pair: {signal['symbol']}\n"
            f"Direction: {signal['direction']}\n"
            f"Leverage: {signal['leverage']}x\n"
            f"Stop Loss: {signal['stop_loss']}\n"
            f"TP1: {signal['tp1']}  TP2: {signal['tp2']}  TP3: {signal['tp3']}\n"
            f"Size: {signal['wallet_pct']*100:.0f}% of wallet\n\n"
            f"Executing..."
        )
        await self._send(preview)
        result = await self._on_signal_trade(signal)
        await self._send(result)

    async def _cmd_stats(self) -> None:
        """All-time trading statistics."""
        if not self._trade_repo:
            await self._send("Stats not available — trade repo not configured.")
            return
        rows = self._trade_repo.all_closed()
        if not rows:
            await self._send("📊 No closed trades yet.")
            return
        pnls = [float(r["pnl_usd"] or 0) for r in rows]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total  = sum(pnls)
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        avg_win  = sum(wins)  / len(wins)   if wins   else 0
        avg_loss = sum(losses)/ len(losses) if losses else 0
        rr       = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        best     = max(pnls)
        worst    = min(pnls)
        consec   = self._trade_repo.consecutive_losses()
        sign     = "+" if total >= 0 else ""
        icon     = "🟢" if total >= 0 else "🔴"
        await self._send(
            f"📊 <b>All-Time Statistics</b>\n\n"
            f"{icon} Total P&amp;L: <b>{sign}${total:.2f} USDT</b>\n"
            f"Trades: {len(pnls)}  (✅ {len(wins)} W / ❌ {len(losses)} L)\n"
            f"Win Rate: <b>{win_rate:.1f}%</b>\n"
            f"Avg Win: <b>+${avg_win:.2f}</b>  Avg Loss: <b>${avg_loss:.2f}</b>\n"
            f"Avg R:R: <b>{rr:.2f}</b>\n"
            f"Best trade: +${best:.2f}  Worst: ${worst:.2f}\n"
            f"Current losing streak: {consec}"
        )

    async def _cmd_history(self) -> None:
        """Last 10 closed trades."""
        if not self._trade_repo:
            await self._send("History not available.")
            return
        rows = self._trade_repo.recent_closed(10)
        if not rows:
            await self._send("📋 No closed trades yet.")
            return
        lines = ["📋 <b>Last 10 Trades</b>\n"]
        for r in rows:
            pnl  = float(r["pnl_usd"] or 0)
            icon = "✅" if pnl > 0 else "❌"
            sign = "+" if pnl > 0 else ""
            sym  = r["symbol"] if "symbol" in r.keys() else "?"
            dirn = (r["direction"] or "?").upper()
            entry= float(r["entry_price"] or 0)
            exit_= float(r["exit_price"]  or 0) if "exit_price" in r.keys() else 0
            dt   = str(r["exit_ts"] or "")[:10]
            lines.append(
                f"{icon} <b>{sym}</b> {dirn}  {sign}${pnl:.2f}\n"
                f"   Entry ${entry:.4f} → Exit ${exit_:.4f}  [{dt}]"
            )
        await self._send("\n".join(lines))

    async def _cmd_pnl(self) -> None:
        """Today's P&L on demand."""
        if not self._trade_repo:
            await self._send("P&L not available.")
            return
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows  = self._trade_repo.closed_trades_since(today)
        if not rows:
            await self._send(f"📊 No closed trades today ({today} UTC).")
            return
        pnls  = [float(r["pnl_usd"] or 0) for r in rows]
        total = sum(pnls)
        wins  = sum(1 for p in pnls if p > 0)
        sign  = "+" if total >= 0 else ""
        icon  = "🟢" if total >= 0 else "🔴"
        lines = [f"📊 <b>Today's P&amp;L ({today} UTC)</b>\n",
                 f"{icon} Net: <b>{sign}${total:.2f} USDT</b>  ({wins}W / {len(pnls)-wins}L)\n"]
        for r in rows:
            p    = float(r["pnl_usd"] or 0)
            s    = "+" if p > 0 else ""
            sym  = r["symbol"] if "symbol" in r.keys() else "?"
            lines.append(f"  • {sym}: {s}${p:.2f}")
        await self._send("\n".join(lines))

    async def _cmd_balance(self) -> None:
        """Show USDT balance + open positions value."""
        if not self._client:
            await self._send("Balance not available — client not configured.")
            return
        try:
            account = await asyncio.get_event_loop().run_in_executor(
                None, self._client.fetch_account_info
            )
            free  = float(account.get("USDT", {}).get("free",  0) or 0)
            total = float(account.get("USDT", {}).get("total", 0) or 0)
            used  = total - free
        except Exception as e:
            await self._send(f"❌ Could not fetch balance: {e}")
            return

        try:
            positions = await self._position_manager.get_all_open_positions()
            upnl_total = sum(
                float(p.get("unrealizedPnl") or p.get("info", {}).get("unRealizedProfit", 0) or 0)
                for p in positions
            )
        except Exception:
            upnl_total = 0.0
            positions  = []

        sign = "+" if upnl_total >= 0 else ""
        await self._send(
            f"💰 <b>Wallet Balance</b>\n\n"
            f"Total equity: <b>${total:.2f} USDT</b>\n"
            f"Free margin:  ${free:.2f}\n"
            f"In positions: ${used:.2f}\n"
            f"Unrealised P&amp;L: <b>{sign}${upnl_total:.2f}</b>\n"
            f"Open positions: {len(positions)}"
        )

    async def _cmd_cancel(self, args: list) -> None:
        """/cancel SYMBOL — cancel all pending TP/stop orders without closing."""
        if not args:
            await self._send("Usage: /cancel SYMBOL  (e.g. /cancel SOL)")
            return
        if not self._client:
            await self._send("Client not available.")
            return
        symbol_raw = args[0].upper().strip()
        if "/" not in symbol_raw:
            symbol_raw = f"{symbol_raw}/USDT:USDT"
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._client.exchange.cancel_all_orders(symbol_raw)
            )
            await self._send(f"✅ All open orders cancelled for <b>{symbol_raw}</b>.\nPosition remains open — place new orders manually if needed.")
        except Exception as e:
            await self._send(f"❌ Cancel failed for {symbol_raw}: {e}")

    async def _cmd_help(self) -> None:
        msg = (
            "<b>OzBinanceBot Commands</b>\n\n"
            "/trade — execute a trade right now\n"
            "/status — current position and signal\n"
            "/positions — all open positions with unrealised P&amp;L\n"
            "/close — close bot-tracked position\n"
            "/close SYMBOL — close a specific position (e.g. /close SOL)\n"
            "/closeall — close ALL open positions\n"
            "/auto off — disable automatic trading\n"
            "/auto on — re-enable automatic trading\n"
            "/help — this message\n\n"
            "<b>💬 Just Chat:</b>\n"
            "You can type naturally and I'll understand:\n"
            "• <i>\"is the signal good?\"</i> — full market analysis\n"
            "• <i>\"what are my gains today?\"</i> — today's P&amp;L\n"
            "• <i>\"show my last trade\"</i> — last closed trade\n"
            "• <i>\"my win rate\"</i> — win rate &amp; stats\n"
            "• <i>\"my balance\"</i> — wallet balance\n"
            "• <i>\"trade history\"</i> — recent trades\n\n"
            "/stats — all-time win rate, R:R, best/worst trade\n"
            "/history — last 10 closed trades\n"
            "/pnl — today's P&amp;L on demand\n"
            "/balance — USDT balance + open positions value\n"
            "/cancel SYMBOL — cancel TP/stop orders (keep position open)\n"
            "/scan — scan all pairs for best setup right now\n\n"
            "<b>Channel Signals:</b>\n"
            "Paste a signal message — bot parses and executes it automatically.\n\n"
            "<i>You can also type <b>trade</b> or <b>trade now</b>.</i>"
        )
        await self._send(msg)

    # ------------------------------------------------------------------
    # Status builder
    # ------------------------------------------------------------------

    def _build_status(self) -> str:
        pos = self._position_manager.get_current()
        auto_str = "ON" if self._bot_state.auto_trade else "OFF"
        signal = self._bot_state.latest_signal

        lines = ["<b>OzBinanceBot Status</b>", f"Auto-trade: <b>{auto_str}</b>\n"]

        if pos:
            dir_str = pos.get("direction", "?").upper()
            entry = pos.get("entry_price", 0)
            stop = pos.get("stop_price", "?")
            tp = pos.get("tp_price", "?")
            lines += [
                "<b>Open Position</b>",
                f"Direction: {dir_str}",
                f"Entry: ${float(entry):,.4f}" if entry else "Entry: ?",
                f"Stop: ${float(stop):,.4f}" if isinstance(stop, (int, float)) else f"Stop: {stop}",
                f"TP: ${float(tp):,.4f}" if isinstance(tp, (int, float)) else f"TP: {tp}",
            ]
        else:
            lines.append("No open position — waiting for signal.")

        if signal:
            lines += [
                "",
                "<b>Latest Signal</b>",
                f"Type: {signal.signal_type.value.upper()}",
                f"Direction: {signal.direction.value}",
                f"Reason: {signal.reason}",
            ]

        return "\n".join(lines)
