"""
Gemini News Sentiment Filter — checks for negative news before executing a trade.

Uses Google Gemini with Google Search grounding so it has real-time access
to the latest news about any coin.

Verdict:
  GREEN  — news is neutral or positive, safe to trade
  YELLOW — mixed signals, proceed with caution (trade still executes)
  RED    — strong negative news (regulatory, hack, crash) — trade BLOCKED

Usage:
  filter = GeminiSentimentFilter(api_key="...")
  result = await filter.check("XRP", "LONG")
  if result.verdict == "RED":
      # skip trade
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    verdict: str          # "GREEN", "YELLOW", "RED"
    summary: str          # one-line explanation
    details: str          # full AI response
    coin: str
    direction: str


class GeminiSentimentFilter:
    def __init__(self, api_key: str, timeout_seconds: int = 15) -> None:
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._client = None
        self._ready = False
        self._setup()

    def _setup(self) -> None:
        try:
            from google import genai
            from google.genai import types
            self._genai = genai
            self._types = types
            self._client_obj = genai.Client(api_key=self._api_key)
            self._ready = True
            logger.info("Gemini sentiment filter ready (gemini-2.0-flash + Google Search)")
        except Exception as e:
            logger.error("Gemini setup failed: %s — sentiment checks disabled", e)
            self._ready = False

    async def check(self, coin: str, direction: str) -> SentimentResult:
        """
        Check news sentiment for a coin before entering a trade.
        Returns a SentimentResult. On any error, returns GREEN (fail-open).
        """
        if not self._ready:
            return SentimentResult(
                verdict="GREEN", coin=coin, direction=direction,
                summary="Sentiment check unavailable — proceeding",
                details="Gemini not configured",
            )

        prompt = (
            f"You are a crypto trading risk assistant. I am about to open a {direction} position on {coin}/USDT.\n\n"
            f"Search Google for the latest news about {coin} cryptocurrency in the last 24-48 hours.\n\n"
            f"Look for:\n"
            f"- Regulatory action, bans, lawsuits, SEC/government news\n"
            f"- Exchange hacks, security breaches, rug pulls\n"
            f"- Major price crashes or whale dumps\n"
            f"- Project failures, team exits, scams\n"
            f"- Positive news: partnerships, ETF approvals, major adoption\n\n"
            f"Respond in this exact format:\n"
            f"VERDICT: [GREEN / YELLOW / RED]\n"
            f"SUMMARY: [one sentence explaining the verdict]\n"
            f"DETAILS: [2-3 sentences of key news found]\n\n"
            f"GREEN = safe to trade (no bad news or positive news)\n"
            f"YELLOW = some uncertainty but not critical\n"
            f"RED = serious negative news that makes this trade very risky right now\n\n"
            f"Be conservative — only use RED for genuinely serious breaking news."
        )

        try:
            loop = asyncio.get_event_loop()

            def _call():
                from google.genai import types
                return self._client_obj.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                    ),
                )

            response = await asyncio.wait_for(
                loop.run_in_executor(None, _call),
                timeout=self._timeout,
            )
            text = response.text.strip()
            return self._parse_response(text, coin, direction)

        except asyncio.TimeoutError:
            logger.warning("Gemini sentiment check timed out after %ds — proceeding GREEN", self._timeout)
            return SentimentResult(
                verdict="GREEN", coin=coin, direction=direction,
                summary="Sentiment check timed out — proceeding",
                details="Request timed out",
            )
        except Exception as e:
            logger.error("Gemini sentiment check failed: %s — proceeding GREEN", e)
            return SentimentResult(
                verdict="GREEN", coin=coin, direction=direction,
                summary=f"Sentiment check error — proceeding",
                details=str(e),
            )

    def _parse_response(self, text: str, coin: str, direction: str) -> SentimentResult:
        """Parse Gemini's structured response."""
        verdict = "GREEN"
        summary = ""
        details = text

        for line in text.splitlines():
            line = line.strip()
            if line.startswith("VERDICT:"):
                v = line.split(":", 1)[1].strip().upper()
                if "RED" in v:
                    verdict = "RED"
                elif "YELLOW" in v:
                    verdict = "YELLOW"
                else:
                    verdict = "GREEN"
            elif line.startswith("SUMMARY:"):
                summary = line.split(":", 1)[1].strip()
            elif line.startswith("DETAILS:"):
                details = line.split(":", 1)[1].strip()

        if not summary:
            summary = text[:100]

        logger.info("Gemini sentiment for %s %s: %s — %s", coin, direction, verdict, summary)
        return SentimentResult(
            verdict=verdict, coin=coin, direction=direction,
            summary=summary, details=details,
        )

    def format_telegram(self, result: SentimentResult) -> str:
        """Format result as Telegram HTML message."""
        icons = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}
        icon = icons.get(result.verdict, "⚪")
        return (
            f"{icon} <b>Gemini News Check — {result.coin} {result.direction}</b>\n"
            f"Verdict: <b>{result.verdict}</b>\n"
            f"{result.summary}\n"
            f"<i>{result.details}</i>"
        )
