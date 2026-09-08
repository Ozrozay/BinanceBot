"""
GeminiChatHandler — free-form AI chat powered by Gemini.

The user can say anything to the Telegram bot and Gemini will:
  - Read all live bot data (balance, indicators, position, trades, scanner)
  - Search Google for latest news on any coin mentioned
  - Think and reason about what the user is asking
  - Reply naturally in plain English (formatted for Telegram)

No keyword matching. No rigid intents. Just natural conversation.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# System prompt that tells Gemini who it is and what data it has
_SYSTEM = """You are a smart crypto futures trading assistant embedded in a live Binance bot.
Your job is to help the trader make good decisions by:
- Analysing the current market setup using the live data provided below
- Searching for latest news when relevant (you have Google Search)
- Answering questions about P&L, trades, balance, signals
- Recommending whether to trade, wait, or avoid based on the full picture
- Being concise — replies go to Telegram, keep them under 300 words
- Using bold for key numbers, emojis for readability
- NEVER recommend a specific entry price as a guarantee — always say "around" or "approximately"
- Always mention risk — every trade can lose money

Respond in Telegram HTML format using <b>bold</b> and <i>italic</i> only (no markdown).
"""


class GeminiChatHandler:
    """
    Drop-in replacement for ChatHandler.
    Accepts any free-form user message and returns a Gemini AI reply.
    """

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
        timeout_seconds: int = 25,
    ) -> None:
        self._api_key       = api_key
        self._trades        = trade_repo
        self._signals       = signal_repo
        self._state         = bot_state
        self._collector     = collector
        self._strategy      = strategy
        self._config        = config
        self._htf           = htf_filter
        self._pm            = position_manager
        self._client        = client
        self._pair_selector = pair_selector
        self._scanner       = scanner
        self._scan_pairs    = scan_pairs or [config.strategy.symbol]
        self._timeout       = timeout_seconds
        self._ready         = False
        self._client_obj    = None
        self._setup()

    def _setup(self) -> None:
        try:
            from google import genai
            self._genai = genai
            self._client_obj = genai.Client(api_key=self._api_key)
            self._ready = True
            logger.info("GeminiChatHandler ready (gemini-2.0-flash + Google Search)")
        except Exception as e:
            logger.error("GeminiChatHandler setup failed: %s", e)
            self._ready = False

    # ------------------------------------------------------------------
    # Public entry point  (called from telegram_commands._on_text)
    # ------------------------------------------------------------------

    def handle(self, text: str) -> Optional[str]:
        """
        Synchronous wrapper — runs the async Gemini call via a new event loop
        slice. Returns the reply string or None (falls through to next handler).
        """
        # Don't intercept slash commands or trade triggers
        stripped = text.strip().lower()
        if stripped.startswith("/") or stripped in ("trade", "trade now"):
            return None
        import re
        if re.match(r'^trade\s+\$?(\d+\.?\d*)\$?$', stripped):
            return None

        if not self._ready:
            return "⚠️ AI chat is not available right now."

        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._async_handle(text))
        except RuntimeError:
            # Already inside an event loop — use run_in_executor trick
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._async_handle(text))
                return future.result(timeout=self._timeout + 5)
        except Exception as e:
            logger.error("GeminiChatHandler.handle error: %s", e)
            return f"⚠️ AI error: {e}"

    async def async_handle(self, text: str) -> str:
        """Async version for callers that are already in an event loop."""
        if not self._ready:
            return "⚠️ AI chat is not available right now."
        return await self._async_handle(text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _async_handle(self, user_message: str) -> str:
        context = await self._build_context()
        prompt = (
            f"{_SYSTEM}\n\n"
            f"=== LIVE BOT DATA ===\n{context}\n\n"
            f"=== TRADER'S MESSAGE ===\n{user_message}"
        )

        try:
            loop = asyncio.get_event_loop()

            def _call():
                from google.genai import types
                return self._client_obj.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                    ),
                )

            response = await asyncio.wait_for(
                loop.run_in_executor(None, _call),
                timeout=self._timeout,
            )
            reply = response.text.strip()
            # Gemini sometimes uses ** markdown — convert to Telegram HTML
            reply = self._md_to_html(reply)
            logger.info("Gemini chat reply (%d chars) for: %s", len(reply), user_message[:60])
            return reply

        except asyncio.TimeoutError:
            return "⏳ AI took too long to respond. Try again in a moment."
        except Exception as e:
            logger.error("Gemini chat call failed: %s", e)
            return f"⚠️ AI error: {e}"

    async def _build_context(self) -> str:
        """Gather all live bot data into a text block for Gemini."""
        lines = []

        # 1. Timestamp
        now = datetime.now(timezone.utc)
        lines.append(f"Timestamp: {now.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append(f"Primary pair: {self._config.strategy.symbol}")
        lines.append(f"Timeframe: {self._config.strategy.timeframe}")

        # 2. Balance
        if self._client:
            try:
                balances = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.exchange.fetch_balance({"type": "future"})
                )
                free  = float(balances.get("free",  {}).get("USDT", 0) or 0)
                total = float(balances.get("total", {}).get("USDT", 0) or 0)
                lines.append(f"Wallet: total=${total:.2f} USDT, free=${free:.2f} USDT, in_margin=${total-free:.2f}")
            except Exception as e:
                lines.append(f"Wallet: unavailable ({e})")

        # 3. Open position
        pos = self._pm.get_current() if self._pm else None
        if pos:
            direction = pos.get("direction", "?").upper()
            symbol    = pos.get("symbol", "?")
            entry     = pos.get("entry_price", 0)
            size      = pos.get("size", 0)
            stop      = pos.get("stop_loss", 0)
            lines.append(
                f"Open position: {direction} {symbol} "
                f"entry=${entry:.5f} size={size:.4f} stop=${stop:.5f}"
            )
        else:
            lines.append("Open position: none")

        # 4. HTF trend
        if self._htf and self._htf.bias:
            lines.append(f"Daily trend (HTF EMA-{self._config.strategy.ema_period if hasattr(self._config.strategy,'ema_period') else 50}): {self._htf.bias.value.upper()}")
        else:
            lines.append("Daily trend: unknown")

        # 5. Current indicators on primary pair
        try:
            df   = self._strategy.compute_indicators(self._collector.data)
            last = df.iloc[-1]
            cfg  = self._config.strategy
            close = float(last["close"])
            ema   = float(last.get(f"ema_{cfg.ema_period}", 0) or 0)
            rsi   = float(last.get(f"rsi_{cfg.rsi_period}", 0) or 0)
            atr   = float(last.get(f"atr_{cfg.atr_period}", 0) or 0)
            adx   = float(last.get("adx", 0) or 0)
            rv    = float(last.get("rel_volume", 0) or 0)
            mh    = float(last.get("macd_hist", 0) or 0)

            lines.append(
                f"Indicators ({cfg.symbol} {cfg.timeframe}): "
                f"price=${close:.5f}, EMA{cfg.ema_period}=${ema:.5f}, "
                f"RSI={rsi:.1f}, ADX={adx:.1f}, volume={rv:.2f}x avg, "
                f"MACD_hist={mh:+.6f}, ATR={atr:.5f}"
            )
            price_vs_ema = "ABOVE EMA (bullish)" if close > ema else "BELOW EMA (bearish)"
            lines.append(f"Price vs EMA: {price_vs_ema}")

        except Exception as e:
            lines.append(f"Indicators: unavailable ({e})")

        # 6. Latest signal
        sig = self._state.latest_signal
        if sig:
            lines.append(
                f"Latest signal: {sig.signal_type.value.upper()} {sig.direction.value.upper()} "
                f"— reason: {sig.reason}"
            )
        else:
            lines.append("Latest signal: none")

        # 7. Scanner — use cached results (avoid slow live API scan during chat)
        if self._scanner and hasattr(self._scanner, "_last_results") and self._scanner._last_results:
            try:
                results = self._scanner._last_results
                entries = [r for r in results if r.signal_type == "ENTRY"]
                if entries:
                    lines.append(f"Scanner (last scan) — {len(entries)} entry signal(s):")
                    for r in entries[:3]:
                        lines.append(
                            f"  • {r.symbol} {r.direction} score={r.score}/6 "
                            f"entry~${r.price:.4f} stop=${r.stop:.4f} tp=${r.tp:.4f}"
                        )
                else:
                    top = sorted(results, key=lambda r: -r.score)[:3]
                    lines.append("Scanner — no entry signals in last scan. Closest:")
                    for r in top:
                        lines.append(f"  • {r.symbol} {r.direction} {r.score}/6 filters")
            except Exception as e:
                lines.append(f"Scanner cache: error ({e})")
        else:
            lines.append(f"Scanner: watching {', '.join(s.split('/')[0] for s in self._scan_pairs)}")

        # 8. Recent trades (last 5)
        try:
            rows = self._trades.recent_closed(limit=5)
            if rows:
                lines.append("Recent closed trades:")
                for r in rows:
                    pnl  = r["pnl_usd"] or 0
                    sign = "+" if pnl >= 0 else ""
                    lines.append(
                        f"  {r['direction'].upper()} {r['symbol'].split('/')[0]} "
                        f"{sign}${pnl:.2f} ({r['exit_reason'] or 'unknown'})"
                    )
                total_pnl = sum(r["pnl_usd"] or 0 for r in rows)
                wins = sum(1 for r in rows if (r["pnl_usd"] or 0) > 0)
                lines.append(f"  Summary last {len(rows)}: total={'+' if total_pnl>=0 else ''}{total_pnl:.2f}, wins={wins}/{len(rows)}")
            else:
                lines.append("Recent trades: none yet")
        except Exception:
            lines.append("Recent trades: unavailable")

        return "\n".join(lines)

    @staticmethod
    def _md_to_html(text: str) -> str:
        """Convert common Markdown bold/italic to Telegram HTML."""
        import re
        # **bold** → <b>bold</b>
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        # *italic* → <i>italic</i>
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        # ### headings → bold line
        text = re.sub(r'^#{1,3}\s+(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)
        return text
