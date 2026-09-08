"""
ClaudeChatHandler — free-form AI chat powered by Anthropic Claude.

The user can say anything to the Telegram bot and Claude will:
  - Read all live bot data (balance, indicators, position, trades, scanner)
  - Think and reason about what the user is asking
  - Reply naturally in plain English formatted for Telegram

No keyword matching. No rigid intents. Just natural conversation.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_SYSTEM = """You are a smart crypto futures trading assistant embedded in a live Binance bot.
Your job is to help the trader make good decisions by:
- Analysing the current market setup using the live data provided
- Answering questions about P&L, trades, balance, signals, and setups
- Recommending whether to trade, wait, or avoid based on the full picture
- Being concise — replies go to Telegram, keep them under 250 words
- Using <b>bold</b> for key numbers and emojis for readability
- Always mentioning risk — every trade can lose money
- Never guarantee price targets — say "around" or "approximately"

Respond in Telegram HTML using only <b>bold</b> and <i>italic</i> tags (no markdown, no bullet dashes with *, no #headings).
Use • for bullet points."""


class ClaudeChatHandler:
    def __init__(
        self,
        api_key: str,
        trade_repo,
        signal_repo,
        bot_state,
        collector,
        strategy,
        config,
        htf_filter=None,
        position_manager=None,
        client=None,
        pair_selector=None,
        scanner=None,
        scan_pairs=None,
        model: str = "claude-haiku-4-5-20251001",
        timeout_seconds: int = 25,
    ) -> None:
        self._api_key    = api_key
        self._trades     = trade_repo
        self._signals    = signal_repo
        self._state      = bot_state
        self._collector  = collector
        self._strategy   = strategy
        self._config     = config
        self._htf        = htf_filter
        self._pm         = position_manager
        self._client     = client
        self._selector   = pair_selector
        self._scanner    = scanner
        self._scan_pairs = scan_pairs or [config.strategy.symbol]
        self._model      = model
        self._timeout    = timeout_seconds
        self._anthropic  = None
        self._ready      = False
        self._setup()

    def _setup(self) -> None:
        try:
            import anthropic
            self._anthropic = anthropic.Anthropic(api_key=self._api_key)
            self._ready = True
            logger.info("ClaudeChatHandler ready (%s)", self._model)
        except Exception as e:
            logger.error("ClaudeChatHandler setup failed: %s", e)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def handle(self, text: str) -> Optional[str]:
        """Sync wrapper — falls through if not ready."""
        if not self._ready:
            return "⚠️ AI chat not available right now."
        import re
        stripped = text.strip().lower()
        if stripped.startswith("/") or stripped in ("trade", "trade now"):
            return None
        if re.match(r'^trade\s+\$?(\d+\.?\d*)\$?$', stripped):
            return None
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context — caller should use async_handle
                return None
            return loop.run_until_complete(self.async_handle(text))
        except Exception as e:
            logger.error("ClaudeChatHandler.handle error: %s", e)
            return f"⚠️ Error: {e}"

    async def async_handle(self, text: str) -> str:
        """Async entry point called from telegram_commands._on_text."""
        if not self._ready:
            return "⚠️ AI chat not available right now."
        try:
            context = await self._build_context()
            full_prompt = f"=== LIVE BOT DATA ===\n{context}\n\n=== TRADER SAYS ===\n{text}"

            loop = asyncio.get_event_loop()

            def _call():
                msg = self._anthropic.messages.create(
                    model=self._model,
                    max_tokens=600,
                    system=_SYSTEM,
                    messages=[{"role": "user", "content": full_prompt}],
                )
                return msg.content[0].text.strip()

            reply = await asyncio.wait_for(
                loop.run_in_executor(None, _call),
                timeout=self._timeout,
            )
            logger.info("Claude chat reply (%d chars) for: %s", len(reply), text[:60])
            return reply

        except asyncio.TimeoutError:
            return "⏳ Took too long to respond. Try again."
        except Exception as e:
            logger.error("Claude chat failed: %s", e)
            return f"⚠️ AI error: {e}"

    # ------------------------------------------------------------------
    # Context builder — reads from memory, no slow API calls
    # ------------------------------------------------------------------

    async def _build_context(self) -> str:
        lines = []

        # Timestamp + config
        now = datetime.now(timezone.utc)
        lines.append(f"Time: {now.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append(f"Primary pair: {self._config.strategy.symbol}  Timeframe: {self._config.strategy.timeframe}")

        # Balance (fast async)
        if self._client:
            try:
                loop = asyncio.get_event_loop()
                balances = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self._client.exchange.fetch_balance({"type": "future"})
                    ),
                    timeout=5,
                )
                free  = float(balances.get("free",  {}).get("USDT", 0) or 0)
                total = float(balances.get("total", {}).get("USDT", 0) or 0)
                lines.append(f"Wallet: total=${total:.2f} USDT  free=${free:.2f}  in_margin=${total-free:.2f}")
            except Exception:
                lines.append("Wallet: unavailable")

        # Open position
        pos = self._pm.get_current() if self._pm else None
        if pos:
            lines.append(
                f"Open position: {pos.get('direction','?').upper()} {pos.get('symbol','?')} "
                f"entry=${pos.get('entry_price',0):.5f} size={pos.get('size',0):.4f} "
                f"stop=${pos.get('stop_loss',0):.5f}"
            )
        else:
            lines.append("Open position: none")

        # HTF trend
        if self._htf and self._htf.bias:
            lines.append(f"Daily trend: {self._htf.bias.value.upper()}")
        else:
            lines.append("Daily trend: unknown")

        # Current indicators (from memory — no API call)
        try:
            df   = self._strategy.compute_indicators(self._collector.data)
            last = df.iloc[-1]
            cfg  = self._config.strategy
            close = float(last["close"])
            ema   = float(last.get(f"ema_{cfg.ema_period}", 0) or 0)
            rsi   = float(last.get(f"rsi_{cfg.rsi_period}", 0) or 0)
            adx   = float(last.get("adx", 0) or 0)
            rv    = float(last.get("rel_volume", 0) or 0)
            mh    = float(last.get("macd_hist", 0) or 0)
            atr   = float(last.get(f"atr_{cfg.atr_period}", 0) or 0)
            above = "ABOVE EMA — bullish" if close > ema else "BELOW EMA — bearish"
            lines.append(
                f"Indicators: price=${close:.5f} EMA{cfg.ema_period}=${ema:.5f} ({above}) "
                f"RSI={rsi:.1f} ADX={adx:.1f} volume={rv:.2f}x MACD_hist={mh:+.6f} ATR={atr:.5f}"
            )
        except Exception as e:
            lines.append(f"Indicators: unavailable ({e})")

        # Latest signal
        sig = self._state.latest_signal
        if sig:
            lines.append(
                f"Latest signal: {sig.signal_type.value.upper()} {sig.direction.value.upper()} — {sig.reason}"
            )
        else:
            lines.append("Latest signal: none")

        # Cached scanner results
        if self._scanner and getattr(self._scanner, "_last_results", None):
            results = self._scanner._last_results
            entries = [r for r in results if r.signal_type == "ENTRY"]
            if entries:
                lines.append(f"Scanner: {len(entries)} entry signal(s):")
                for r in entries[:3]:
                    lines.append(f"  • {r.symbol} {r.direction} score={r.score}/6 entry~${r.price:.4f} stop=${r.stop:.4f} tp=${r.tp:.4f}")
            else:
                top = sorted(results, key=lambda r: -r.score)[:3]
                lines.append("Scanner: no entries. Closest setups:")
                for r in top:
                    lines.append(f"  • {r.symbol} {r.direction} {r.score}/6 filters")
        else:
            lines.append(f"Scanner: monitoring {', '.join(s.split('/')[0] for s in self._scan_pairs)}")

        # Recent trades (from DB — fast)
        try:
            rows = self._trades.recent_closed(limit=5)
            if rows:
                lines.append("Recent closed trades:")
                for r in rows:
                    pnl = r["pnl_usd"] or 0
                    lines.append(
                        f"  {r['direction'].upper()} {r['symbol'].split('/')[0]} "
                        f"{'+' if pnl>=0 else ''}{pnl:.2f} ({r['exit_reason'] or 'unknown'})"
                    )
                total = sum(r["pnl_usd"] or 0 for r in rows)
                wins  = sum(1 for r in rows if (r["pnl_usd"] or 0) > 0)
                lines.append(f"  Last {len(rows)} summary: total={'+' if total>=0 else ''}{total:.2f} wins={wins}/{len(rows)}")
            else:
                lines.append("Recent trades: none yet")
        except Exception:
            lines.append("Recent trades: unavailable")

        return "\n".join(lines)
