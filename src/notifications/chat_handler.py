"""
Natural-language chat handler for the Telegram bot.

Understands questions like:
  "is the signal good?"
  "what are my gains today?"
  "show my last trade"
  "how much did I make this week?"
  "what is my pnl?"
  "am I winning?"
  "what's the market doing?"
  "analyse the market"

Returns a formatted HTML string for Telegram.
"""

import logging
import re
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class ChatHandler:
    def __init__(
        self,
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
    ) -> None:
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

    def handle(self, text: str) -> Optional[str]:
        """
        Try to match text to a chat intent.
        Returns a reply string, or None if not recognised (caller falls through).
        """
        t = text.lower().strip()

        # P&L / gains queries
        if any(w in t for w in ["gain", "loss", "pnl", "profit", "earning", "win", "how much", "made today", "made this"]):
            return self._pnl_reply(t)

        # Last trade
        if any(w in t for w in ["last trade", "previous trade", "recent trade", "last position"]):
            return self._last_trade_reply()

        # Signal / market analysis
        if any(w in t for w in ["signal good", "good signal", "is the signal", "analyse", "analyze", "market doing", "what's happening", "should i trade", "setup good", "is it good"]):
            return self._analysis_reply()

        # Trade history
        if any(w in t for w in ["trade history", "all trades", "trade log", "my trades"]):
            return self._history_reply()

        # Win rate
        if any(w in t for w in ["win rate", "winrate", "how many wins", "success rate"]):
            return self._winrate_reply()

        # Balance / equity
        if any(w in t for w in ["balance", "equity", "wallet", "how much do i have", "my account"]):
            return self._balance_reply()

        # "Can I trade $X now?" / "is $X enough?"
        dollar_match = re.search(r"\$?\s*(\d+(?:\.\d+)?)", t)
        if dollar_match and any(w in t for w in ["can i trade", "trade with", "is $", "is this enough", "enough to trade", "trade now", "afford"]):
            amount = float(dollar_match.group(1))
            return self._can_i_trade_reply(amount)

        # "What's a good trade now?" / "any good setups?"
        if any(w in t for w in ["good trade", "best trade", "good setup", "best setup", "any trade", "what to trade", "what should i trade", "find me a trade", "good opportunity", "any opportunity", "scan now", "good pair"]):
            return self._best_trade_reply()

        return None   # not a chat intent — let other handlers try

    # ------------------------------------------------------------------
    # Intents
    # ------------------------------------------------------------------

    def _pnl_reply(self, text: str) -> str:
        now = datetime.now(timezone.utc)

        # Detect time window from text
        if "week" in text:
            days = 7
            label = "This week"
        elif "month" in text:
            days = 30
            label = "This month"
        elif "yesterday" in text:
            days = 2
            label = "Yesterday"
            now = now - timedelta(days=1)
        else:
            days = 1
            label = "Today"

        # Sum PnL from closed trades in window
        cutoff = (now - timedelta(days=days)).isoformat()
        try:
            rows = self._trades._conn.execute(
                "SELECT pnl_usd, direction, symbol, exit_ts, entry_price, exit_price "
                "FROM trades WHERE exit_ts IS NOT NULL AND exit_ts >= ? ORDER BY exit_ts DESC",
                (cutoff,),
            ).fetchall()
        except Exception:
            rows = []

        if not rows:
            return f"📊 <b>{label}</b>\nNo closed trades yet in this period."

        total = sum(r["pnl_usd"] or 0 for r in rows)
        wins  = sum(1 for r in rows if (r["pnl_usd"] or 0) > 0)
        losses = len(rows) - wins
        emoji = "📈" if total >= 0 else "📉"
        sign  = "+" if total >= 0 else ""

        lines = [f"{emoji} <b>{label}'s P&amp;L</b>"]
        lines.append(f"Total: <b>{sign}${total:.2f}</b>")
        lines.append(f"Trades: {len(rows)}  ✅ {wins} wins  ❌ {losses} losses")
        lines.append("")

        for r in rows[:5]:   # show up to 5 most recent
            pnl = r["pnl_usd"] or 0
            s = "✅" if pnl >= 0 else "❌"
            sign2 = "+" if pnl >= 0 else ""
            lines.append(
                f"{s} {r['direction'].upper()} {r['symbol'].split('/')[0]}  "
                f"{sign2}${pnl:.2f}"
            )

        if len(rows) > 5:
            lines.append(f"<i>…and {len(rows)-5} more</i>")

        return "\n".join(lines)

    def _last_trade_reply(self) -> str:
        try:
            rows = self._trades.recent_closed(limit=1)
        except Exception:
            rows = []

        if not rows:
            return "📋 No completed trades recorded yet."

        r = rows[0]
        pnl   = r["pnl_usd"] or 0
        emoji = "✅" if pnl >= 0 else "❌"
        sign  = "+" if pnl >= 0 else ""

        # Duration
        dur = ""
        if r["entry_ts"] and r["exit_ts"]:
            try:
                entry_dt = datetime.fromisoformat(r["entry_ts"])
                exit_dt  = datetime.fromisoformat(r["exit_ts"])
                secs = int((exit_dt - entry_dt).total_seconds())
                h, m = divmod(secs // 60, 60)
                dur = f"\nDuration: {h}h {m}m"
            except Exception:
                pass

        pct = ""
        if r["entry_price"] and r["exit_price"] and r["entry_price"] > 0:
            change = (r["exit_price"] - r["entry_price"]) / r["entry_price"] * 100
            if r["direction"] == "short":
                change = -change
            pct = f" ({'+' if change>=0 else ''}{change:.2f}%)"

        return (
            f"{emoji} <b>Last Trade — {r['symbol']}</b>\n"
            f"Direction: {r['direction'].upper()}\n"
            f"Entry: ${r['entry_price']:.5f}\n"
            f"Exit:  ${r['exit_price']:.5f}{pct}\n"
            f"P&amp;L: <b>{sign}${pnl:.2f}</b>\n"
            f"Exit reason: {r['exit_reason'] or 'unknown'}"
            f"{dur}"
        )

    def _analysis_reply(self) -> str:
        """Full confluence analysis of the current bar."""
        lines = [f"📊 <b>Market Analysis — {self._config.strategy.symbol}</b>"]

        # HTF bias
        if self._htf:
            bias = self._htf.bias
            if bias:
                icon = "🟢" if bias.value == "long" else "🔴"
                lines.append(f"Daily trend: {icon} <b>{'BULLISH' if bias.value=='long' else 'BEARISH'}</b>")
            else:
                lines.append("Daily trend: ⏳ warming up")

        # Current indicators
        try:
            df = self._strategy.compute_indicators(self._collector.data)
            last = df.iloc[-1]
            close   = float(last["close"])
            ema_col = f"ema_{self._config.strategy.ema_period}"
            rsi_col = f"rsi_{self._config.strategy.rsi_period}"
            atr_col = f"atr_{self._config.strategy.atr_period}"

            ema = float(last[ema_col]) if ema_col in df.columns else None
            rsi = float(last[rsi_col]) if rsi_col in df.columns else None
            atr = float(last[atr_col]) if atr_col in df.columns else None

            lines.append(f"Price: <b>${close:.5f}</b>")

            if ema:
                above = close > ema
                lines.append(f"EMA-{self._config.strategy.ema_period}: ${ema:.5f}  {'✅ above' if above else '❌ below'}")

            if rsi is not None:
                thr = self._config.strategy.rsi_entry_threshold
                ok = rsi > thr
                lines.append(f"RSI: {rsi:.1f}  {'✅' if ok else '⏳'} (threshold {thr})")

            if "adx" in df.columns:
                adx = float(last.get("adx", 0) or 0)
                ok = adx >= self._config.strategy.adx_threshold
                lines.append(f"ADX: {adx:.1f}  {'✅ trending' if ok else '❌ choppy'}")

            if "rel_volume" in df.columns:
                rv = float(last.get("rel_volume", 0) or 0)
                ok = rv >= self._config.strategy.volume_filter
                lines.append(f"Volume: {rv:.2f}×  {'✅' if ok else '❌ low'}")

            if "macd_hist" in df.columns:
                mh = float(last.get("macd_hist", 0) or 0)
                lines.append(f"MACD hist: {mh:+.6f}  {'✅ bullish' if mh>0 else '❌ bearish'}")

            if "sr_resistance" in df.columns and atr:
                res = last.get("sr_resistance")
                sup = last.get("sr_support")
                import numpy as np
                import pandas as pd
                if res and not pd.isna(res):
                    gap = float(res) - close
                    zone = atr * self._config.strategy.sr_zone_atr_mult
                    lines.append(f"Resistance: ${float(res):.5f}  {'❌ too close' if gap < zone else '✅ clear'}")
                if sup and not pd.isna(sup):
                    lines.append(f"Support: ${float(sup):.5f}")

        except Exception as e:
            lines.append(f"<i>Could not compute indicators: {e}</i>")

        # Current signal
        sig = self._state.latest_signal
        if sig:
            stype = sig.signal_type.value.upper()
            sdir  = sig.direction.value.upper()
            if stype == "ENTRY":
                lines.append(f"\n🚨 <b>Signal: {sdir} ENTRY</b>")
                lines.append(f"Reason: {sig.reason}")
            elif stype == "EXIT":
                lines.append(f"\n⚠️ Signal: EXIT ({sig.reason})")
            else:
                lines.append(f"\n⏳ Signal: HOLD — {sig.reason}")

        return "\n".join(lines)

    def _history_reply(self) -> str:
        try:
            rows = self._trades.recent_closed(limit=10)
        except Exception:
            rows = []

        if not rows:
            return "📋 No trade history yet."

        lines = ["📋 <b>Recent Trades</b>"]
        for r in rows:
            pnl  = r["pnl_usd"] or 0
            sign = "+" if pnl >= 0 else ""
            icon = "✅" if pnl >= 0 else "❌"
            date = (r["exit_ts"] or "")[:10]
            lines.append(
                f"{icon} {r['direction'].upper()} {r['symbol'].split('/')[0]}  "
                f"<b>{sign}${pnl:.2f}</b>  <i>{date}</i>"
            )

        total = sum(r["pnl_usd"] or 0 for r in rows)
        sign = "+" if total >= 0 else ""
        lines.append(f"\nTotal (last {len(rows)}): <b>{sign}${total:.2f}</b>")
        return "\n".join(lines)

    def _winrate_reply(self) -> str:
        try:
            rows = self._trades.recent_closed(limit=50)
        except Exception:
            rows = []

        if not rows:
            return "📊 No closed trades yet to calculate win rate."

        wins   = sum(1 for r in rows if (r["pnl_usd"] or 0) > 0)
        losses = len(rows) - wins
        rate   = wins / len(rows) * 100
        total_pnl = sum(r["pnl_usd"] or 0 for r in rows)
        avg_win  = sum(r["pnl_usd"] for r in rows if (r["pnl_usd"] or 0) > 0) / max(wins, 1)
        avg_loss = sum(r["pnl_usd"] for r in rows if (r["pnl_usd"] or 0) <= 0) / max(losses, 1)

        emoji = "🔥" if rate >= 60 else "📊" if rate >= 50 else "⚠️"
        return (
            f"{emoji} <b>Win Rate (last {len(rows)} trades)</b>\n"
            f"Wins: {wins}  Losses: {losses}  Rate: <b>{rate:.0f}%</b>\n"
            f"Avg win: +${avg_win:.2f}  Avg loss: ${avg_loss:.2f}\n"
            f"Total P&amp;L: <b>${'+' if total_pnl>=0 else ''}{total_pnl:.2f}</b>"
        )

    def _balance_reply(self) -> str:
        if not self._client:
            return "❌ Balance unavailable in read-only mode."
        try:
            account = self._client.fetch_account_info()
            free  = float(account.get("USDT", {}).get("free", 0) or 0)
            total = float(account.get("USDT", {}).get("total", 0) or 0)
            locked = total - free
            return (
                f"💰 <b>Futures Wallet</b>\n"
                f"Total: <b>${total:.2f} USDT</b>\n"
                f"Available: ${free:.2f}\n"
                f"In margin: ${locked:.2f}"
            )
        except Exception as e:
            return f"❌ Could not fetch balance: {e}"

    def _can_i_trade_reply(self, amount_usd: float) -> str:
        """Tell the user if they can trade a specific dollar amount right now."""
        leverage = self._config.strategy.max_leverage if hasattr(self._config.strategy, "max_leverage") else 5
        min_notional = 23.0
        min_margin = min_notional / leverage

        lines = [f"💵 <b>Can I trade ${amount_usd:.2f}?</b>"]

        # Check minimum
        if amount_usd < min_margin:
            lines.append(
                f"❌ <b>Too small.</b> You need at least <b>${min_margin:.2f}</b> margin "
                f"to meet Binance's minimum order size (~${min_notional:.0f} notional at {leverage}x leverage)."
            )
            lines.append(f"💡 Top up to at least <b>${min_margin + 1:.0f} USDT</b> to place trades.")
            return "\n".join(lines)

        # Check live balance
        free_balance = None
        if self._client:
            try:
                balances = self._client.exchange.fetch_balance({"type": "future"})
                free_balance = float(balances.get("free", {}).get("USDT", 0) or 0)
            except Exception:
                pass

        if free_balance is not None and amount_usd > free_balance * 0.9:
            lines.append(
                f"⚠️ <b>Not enough funds.</b> You have <b>${free_balance:.2f}</b> free "
                f"but asked to trade ${amount_usd:.2f} (90% cap = ${free_balance*0.9:.2f})."
            )
            return "\n".join(lines)

        # Scan for best pair at that amount
        if self._pair_selector and self._scanner:
            try:
                results = self._scanner.scan(self._scan_pairs)
                entries = [r for r in results if r.signal_type == "ENTRY" and r.price > 0]

                if not entries:
                    lines.append(f"✅ <b>${amount_usd:.2f} is tradeable</b> (${amount_usd * leverage:.2f} notional at {leverage}x).")
                    lines.append("⏳ But no entry signals right now. Wait for a setup.")
                    return "\n".join(lines)

                best = entries[0]
                notional = max(amount_usd * leverage, min_notional)
                size = notional / best.price
                margin_used = notional / leverage

                lines.append(f"✅ <b>Yes! You can trade ${amount_usd:.2f}</b>")
                lines.append(f"")
                lines.append(f"🏆 <b>Best setup right now:</b>")
                lines.append(f"Pair: <b>{best.symbol}</b>  {best.direction}")
                lines.append(f"Score: {best.score}/6 filters ✅")
                lines.append(f"Entry: ~${best.price:.4f}")
                lines.append(f"Stop: ${best.stop:.4f}")
                lines.append(f"Target: ${best.tp:.4f}")
                lines.append(f"")
                lines.append(f"📐 Position at {leverage}x:")
                lines.append(f"Margin used: ${margin_used:.2f}  Notional: ${notional:.2f}")
                lines.append(f"Size: {size:.4f} {best.symbol.split('/')[0]}")
                lines.append(f"")
                lines.append(f"💡 Use /trade to execute automatically.")

            except Exception as e:
                lines.append(f"✅ <b>${amount_usd:.2f} is tradeable</b> in principle.")
                lines.append(f"<i>Could not scan pairs: {e}</i>")
        else:
            notional = amount_usd * leverage
            lines.append(f"✅ <b>${amount_usd:.2f} is tradeable</b> (${notional:.2f} notional at {leverage}x leverage).")
            lines.append("Use /trade to find and execute the best pair automatically.")

        return "\n".join(lines)

    def _best_trade_reply(self) -> str:
        """Scan all pairs and report the best opportunity right now."""
        if not self._scanner:
            return "⏳ Scanner not available — try /scan instead."

        try:
            results = self._scanner.scan(self._scan_pairs)
        except Exception as e:
            return f"❌ Scan failed: {e}"

        entries = [r for r in results if r.signal_type == "ENTRY" and r.price > 0]
        holds   = [r for r in results if r.signal_type != "ENTRY"]

        lines = ["🔍 <b>Best Trades Right Now</b>"]

        if not entries:
            lines.append("")
            lines.append("⏳ <b>No entry signals</b> across any pair right now.")
            lines.append("")
            # Show top 3 closest to triggering
            top = sorted(results, key=lambda r: -r.score)[:3]
            if top:
                lines.append("📊 <b>Closest setups (waiting):</b>")
                for r in top:
                    lines.append(f"• {r.symbol.split('/')[0]}  {r.direction}  {r.score}/6 filters")
            lines.append("")
            lines.append("💡 Watch these pairs — they're building setups.")
            return "\n".join(lines)

        # Sort by score
        entries.sort(key=lambda r: -r.score)
        best = entries[0]

        # HTF check annotation
        htf_ok = True
        htf_note = ""
        if self._htf:
            from src.signals.signal import Direction
            d = Direction.LONG if best.direction == "LONG" else Direction.SHORT
            htf_ok = self._htf.allows(d)
            if not htf_ok:
                htf_note = " ⚠️ HTF opposes"

        lines.append("")
        icon = "🟢" if best.direction == "LONG" else "🔴"
        lines.append(f"{icon} <b>#{1} {best.symbol}  {best.direction}</b>{htf_note}")
        lines.append(f"Score: <b>{best.score}/6</b> filters passing")
        lines.append(f"Entry: ~${best.price:.4f}")
        lines.append(f"Stop loss: ${best.stop:.4f}")
        lines.append(f"Take profit: ${best.tp:.4f}")

        # Risk/reward
        if best.price and best.stop and best.tp:
            risk   = abs(best.price - best.stop)
            reward = abs(best.tp - best.price)
            rr = reward / risk if risk > 0 else 0
            lines.append(f"Risk/Reward: <b>{rr:.1f}R</b>")

        if best.score < 5:
            lines.append(f"<i>⚠️ Only {best.score}/6 filters — moderate confidence</i>")

        # Runner-up
        if len(entries) > 1:
            lines.append("")
            lines.append("📋 <b>Other setups:</b>")
            for r in entries[1:4]:
                icon2 = "🟢" if r.direction == "LONG" else "🔴"
                lines.append(f"{icon2} {r.symbol.split('/')[0]}  {r.direction}  {r.score}/6")

        lines.append("")
        lines.append("💡 Use /trade to execute the best pair automatically.")
        return "\n".join(lines)
