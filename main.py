"""
Bot entry point.

Startup sequence:
  1. Load .env
  2. Load and validate config — prints TRADING_ENV loudly
  3. Set up logging
  4. Initialize DB + repositories
  5. Connect to Binance, confirm auth
  6. Start Telegram notifier (outbound)
  7. Instantiate RiskManager, TradeExecutor, PositionManager (live/testnet only)
  8. Load any open position from DB (restart recovery)
  9. Fetch historical klines for warmup
 10. Start Telegram command polling as a parallel asyncio task
 11. Start WebSocket kline stream
 12. On each closed bar:
       a. check_for_fills — detect stop/TP hits
       b. generate_signal — strategy + DB persist
       c. auto-execute entry (if auto_trade=True and no open position)
       d. close position on EXIT signal
"""

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

_LOCK_FILE = Path("data/.bot.lock")


def _acquire_lock() -> None:
    """Exit immediately if another instance is already running."""
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _LOCK_FILE.exists():
        existing_pid = _LOCK_FILE.read_text().strip()
        # Check if the process is actually still alive
        try:
            os.kill(int(existing_pid), 0)
            print(
                f"[STARTUP ERROR] Bot is already running (PID {existing_pid}).\n"
                f"Stop it first:  kill {existing_pid}\n"
                f"Or force-clear: rm {_LOCK_FILE}",
                file=sys.stderr,
            )
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pass  # stale lock from a crashed run — overwrite it
    _LOCK_FILE.write_text(str(os.getpid()))


def _release_lock() -> None:
    try:
        _LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass

from src.config import TradingEnv, load_config
from src.logging.structured import setup_logging
from src.signals.signal import Signal

logger = logging.getLogger(__name__)


@dataclass
class PendingTrade:
    """A trade setup waiting for user YES/NO confirmation."""
    symbol: str
    direction: str
    price: float
    stop: float
    tp: float
    score: int
    margin: float
    notional: float
    size: float
    leverage: int
    expires_at: float   # time.time() + 300 seconds


@dataclass
class BotState:
    """Shared mutable state between the WebSocket loop and Telegram command handler."""
    latest_signal: Optional[Signal] = None
    auto_trade: bool = True
    trade_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_trade: Optional[PendingTrade] = None


async def async_main() -> None:
    setup_logging()

    try:
        config = load_config()
    except EnvironmentError as e:
        print(f"[STARTUP ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    env_banner = {
        TradingEnv.MAINNET_READONLY: "⚠  MAINNET READ-ONLY  (real data, no orders)",
        TradingEnv.TESTNET:          "TESTNET  (fake money)",
        TradingEnv.LIVE:             "LIVE TRADING  *** REAL MONEY ***",
    }
    logger.warning("=" * 70)
    logger.warning("  TRADING ENV: %s", env_banner[config.binance.trading_env])
    logger.warning("=" * 70)

    from src.api.client import BinanceClient
    from src.data.collector import MarketDataCollector
    from src.data.websocket import KlineWebSocket
    from src.database.schema import init_db
    from src.database.repository import SignalRepository, TradeRepository, PositionRepository
    from src.notifications.telegram import TelegramNotifier
    from src.signals.generator import SignalGenerator
    from src.signals.signal import SignalType
    from src.strategy.trend_following import TrendFollowingStrategy

    # --- Database ---
    db_conn = init_db(config.database.path)
    signal_repo = SignalRepository(db_conn)
    trade_repo  = TradeRepository(db_conn)
    pos_repo    = PositionRepository(db_conn)
    logger.info("Database ready: %s", config.database.path)

    # --- Telegram (outbound notifications) ---
    notifier = TelegramNotifier(
        token=config.telegram.token,
        chat_id=config.telegram.chat_id,
        enabled=config.telegram.enabled,
    )

    # --- Exchange ---
    client = BinanceClient(config)
    logger.info("Connecting to Binance (%s)...", config.binance.trading_env.value)
    try:
        account = client.fetch_account_info()
        usdt_balance = account.get("USDT", {}).get("total", "n/a")
        logger.info("Auth OK — futures wallet USDT: %s", usdt_balance)
    except Exception as e:
        logger.error("Auth failed: %s", e)
        sys.exit(1)

    # --- Strategy + data ---
    strategy  = TrendFollowingStrategy(config.strategy)
    collector = MarketDataCollector(client, config.data, config.strategy)
    signal_gen = SignalGenerator(strategy, db_conn)

    # Sentiment filter disabled — using Claude for chat instead
    sentiment_filter = None

    # --- Multi-pair scanner + smart pair selector ---
    from src.strategy.scanner import PairScanner, DEFAULT_SCAN_PAIRS, format_scan_report
    from src.strategy.pair_selector import SmartPairSelector
    scan_pairs_env = os.environ.get("SCAN_PAIRS", "")
    scan_pairs = [p.strip() for p in scan_pairs_env.split(",")] if scan_pairs_env else DEFAULT_SCAN_PAIRS
    scanner_enabled = os.environ.get("SCANNER_ENABLED", "false").lower() == "true"
    scanner = PairScanner(client=client, base_config=config.strategy, timeframe=config.strategy.timeframe) if scanner_enabled else None
    pair_selector = None   # initialised after htf_filter is ready
    _scan_bar_counter = 0

    # --- Higher Time-Frame trend filter ---
    from src.strategy.htf_filter import HTFFilter
    htf_timeframe  = os.environ.get("HTF_TIMEFRAME", "1d")
    htf_ema_period = int(os.environ.get("HTF_EMA_PERIOD", "50"))
    htf_enabled    = os.environ.get("HTF_FILTER", "false").lower() == "true"
    htf_filter = HTFFilter(
        client=client,
        symbol=config.strategy.symbol,
        timeframe=htf_timeframe,
        ema_period=htf_ema_period,
    ) if htf_enabled else None
    _htf_bar_counter = 0

    # --- Shared state ---
    bot_state = BotState()

    # --- Risk + execution (live/testnet only) ---
    risk_manager     = None
    executor         = None
    position_manager = None
    tg_cmd_handler   = None

    if config.binance.trading_env in (TradingEnv.LIVE, TradingEnv.TESTNET):
        from src.risk.manager import RiskManager
        from src.executor.trade_executor import TradeExecutor
        from src.position.manager import PositionManager
        from src.notifications.telegram_commands import TelegramCommandHandler

        risk_manager     = RiskManager(config.risk)
        executor         = TradeExecutor(client, config)
        position_manager = PositionManager(
            client=client,
            config=config,
            trade_repo=trade_repo,
            position_repo=pos_repo,
            notifier=notifier,
            risk_manager=risk_manager,
        )
        position_manager.load_from_db()
        position_manager._bot_state = bot_state

        async def force_trade() -> str:
            return await _execute_trade(
                signal=bot_state.latest_signal,
                collector=collector,
                client=client,
                risk_manager=risk_manager,
                executor=executor,
                position_manager=position_manager,
                notifier=notifier,
                bot_state=bot_state,
                pair_selector=pair_selector,
                sentiment_filter=sentiment_filter,
            )

        async def force_trade_amount(amount_usd: float) -> str:
            return await _execute_trade_amount(
                amount_usd=amount_usd,
                collector=collector,
                client=client,
                strategy=strategy,
                config=config,
                executor=executor,
                position_manager=position_manager,
                bot_state=bot_state,
                pair_selector=pair_selector,
            )

        async def execute_signal_trade(signal_dict: dict) -> str:
            return await _execute_channel_signal(
                signal_dict=signal_dict,
                client=client,
                config=config,
                executor=executor,
                position_manager=position_manager,
                bot_state=bot_state,
                trade_repo=trade_repo,
            )

        if config.telegram.enabled:
            async def run_scan() -> str:
                if not scanner:
                    return "Scanner not enabled. Set SCANNER_ENABLED=true in .env and restart."
                results = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: scanner.scan(scan_pairs)
                )
                return format_scan_report(results, htf_filter=htf_filter)

            # AI chat disabled — use basic keyword handler
            from src.notifications.chat_handler import ChatHandler
            chat_handler = ChatHandler(
                trade_repo=trade_repo,
                signal_repo=signal_repo,
                bot_state=bot_state,
                collector=collector,
                strategy=strategy,
                config=config,
                htf_filter=htf_filter,
                position_manager=position_manager,
                client=client,
            )
            tg_cmd_handler = TelegramCommandHandler(
                token=config.telegram.token,
                chat_id=config.telegram.chat_id,
                bot_state=bot_state,
                on_force_trade=force_trade,
                on_force_trade_amount=force_trade_amount,
                position_manager=position_manager,
                on_signal_trade=execute_signal_trade,
                chat_handler=chat_handler,
                on_scan=run_scan,
                trade_repo=trade_repo,
                client=client,
            )
            await tg_cmd_handler.setup()

        logger.info("Live execution active: RiskManager + TradeExecutor + PositionManager ready.")

    # --- Channel monitor (Telethon user API) ---
    channel_monitor = None
    api_id   = os.environ.get("TELEGRAM_API_ID", "")
    api_hash = os.environ.get("TELEGRAM_API_HASH", "")
    channel  = os.environ.get("SIGNAL_CHANNEL", "")

    if api_id and api_hash and channel and executor is not None:
        from src.notifications.channel_monitor import ChannelMonitor
        from telegram import Bot as TgBot
        notify_bot = TgBot(token=config.telegram.token)
        # Support comma-separated list of channels
        channels = [c.strip() for c in channel.split(",") if c.strip()]
        channel_monitor = ChannelMonitor(
            api_id=int(api_id),
            api_hash=api_hash,
            channels=channels,
            notify_chat_id=config.telegram.chat_id,
            on_signal_trade=execute_signal_trade,
            notify_bot=notify_bot,
        )
        logger.info("Channel monitor configured: %d channel(s): %s", len(channels), ", ".join(channels))

    else:
        logger.info("Read-only mode — no risk/executor/position manager loaded.")

    # --- Startup notification ---
    await notifier.startup(
        config.binance.trading_env.value,
        config.strategy.symbol,
        config.strategy.timeframe,
    )

    # --- Smart pair selector (needs htf_filter) ---
    if scanner:
        pair_selector = SmartPairSelector(
            scanner=scanner,
            pairs=scan_pairs,
            leverage=int(config.risk.max_leverage),
            min_notional=23.0,
            htf_filter=htf_filter,
        )
        logger.info("SmartPairSelector ready — will scan %d pairs per trade", len(scan_pairs))
        # Patch chat_handler now that pair_selector is ready
        try:
            if chat_handler is not None:
                chat_handler._selector = pair_selector
        except NameError:
            pass

    # --- HTF filter startup refresh ---
    if htf_filter:
        logger.info("HTF filter: fetching %s candles (%s EMA%d)...", htf_timeframe, config.strategy.symbol, htf_ema_period)
        htf_filter.refresh()
        logger.info(htf_filter.status_line())

    # --- Warmup ---
    logger.info("Loading historical data for warmup...")
    df = collector.fetch_historical()
    logger.info(
        "Warmup complete: %d bars (%s → %s)",
        len(df),
        df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M"),
        df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
    )

    initial_signal = signal_gen.process(collector.data)
    bot_state.latest_signal = initial_signal
    logger.info(
        "Initial signal: type=%s dir=%s reason='%s'",
        initial_signal.signal_type.value,
        initial_signal.direction.value,
        initial_signal.reason,
    )

    # --- Per-bar callback ---
    async def on_bar_closed(bar: dict) -> None:
        nonlocal _htf_bar_counter, _scan_bar_counter
        collector.on_kline_closed(bar)

        if not collector.is_warmed_up(strategy.required_warmup_bars()):
            return

        # 1. Check if open position was stopped/TP'd since last bar
        if position_manager:
            await position_manager.check_for_fills()

        # 2. Refresh HTF filter every 6 bars (once per ~24h on 4H timeframe)
        if htf_filter:
            _htf_bar_counter += 1
            if _htf_bar_counter >= 6:
                _htf_bar_counter = 0
                htf_filter.refresh()
                logger.info(htf_filter.status_line())

        # 3. Multi-pair scan every 6 bars (~24h on 4H timeframe)
        if scanner:
            _scan_bar_counter += 1
            if _scan_bar_counter >= 6:
                _scan_bar_counter = 0
                try:
                    import time as _time
                    logger.info("Running multi-pair scan on %d pairs...", len(scan_pairs))
                    scan_results = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: scanner.scan(scan_pairs)
                    )
                    report = format_scan_report(scan_results, htf_filter=htf_filter)
                    if config.telegram.enabled and notifier._bot:
                        await notifier._bot.send_message(
                            chat_id=config.telegram.chat_id,
                            text=report,
                            parse_mode="HTML",
                        )

                    # --- Proactive trade alert: notify if score >= 5 and no open position ---
                    if position_manager and not position_manager.has_open_position():
                        hot = [r for r in scan_results if r.signal_type == "ENTRY" and r.score >= 5]
                        # Apply HTF filter
                        if htf_filter:
                            from src.signals.signal import Direction
                            hot = [r for r in hot if htf_filter.allows(
                                Direction.LONG if r.direction == "LONG" else Direction.SHORT
                            )]
                        if hot and not bot_state.pending_trade:
                            best = hot[0]
                            # Calculate size from free balance
                            try:
                                _bals = await asyncio.get_event_loop().run_in_executor(
                                    None,
                                    lambda: client.exchange.fetch_balance({"type": "future"})
                                )
                                free_bal = float(_bals.get("free", {}).get("USDT", 0) or 0)
                            except Exception:
                                free_bal = 0.0

                            lev      = int(config.risk.max_leverage)
                            margin   = round(min(free_bal * 0.9, free_bal), 2)
                            notional = round(max(margin * lev, 23.0), 2)
                            notional = min(notional, free_bal * 0.9 * lev)
                            size     = notional / best.price if best.price else 0

                            if margin >= 2.0 and config.telegram.enabled and notifier._bot:
                                # Save pending trade
                                bot_state.pending_trade = PendingTrade(
                                    symbol=best.symbol,
                                    direction=best.direction,
                                    price=best.price,
                                    stop=best.stop,
                                    tp=best.tp,
                                    score=best.score,
                                    margin=margin,
                                    notional=notional,
                                    size=size,
                                    leverage=lev,
                                    expires_at=_time.time() + 300,
                                )
                                icon = "🟢" if best.direction == "LONG" else "🔴"
                                rr = abs(best.tp - best.price) / abs(best.price - best.stop) if abs(best.price - best.stop) > 0 else 0
                                alert = (
                                    f"🚨 <b>Trade Opportunity Found!</b>\n\n"
                                    f"{icon} <b>{best.symbol}  {best.direction}</b>\n"
                                    f"Score: <b>{best.score}/6</b> filters passing\n"
                                    f"Entry: ~${best.price:.4f}\n"
                                    f"Stop loss: ${best.stop:.4f}\n"
                                    f"Take profit: ${best.tp:.4f}\n"
                                    f"Risk/Reward: <b>{rr:.1f}R</b>\n\n"
                                    f"💰 <b>Your position:</b>\n"
                                    f"Margin: ${margin:.2f}  ×{lev} = ${notional:.2f} notional\n"
                                    f"Size: {size:.4f} {best.symbol.split('/')[0]}\n\n"
                                    f"Reply <b>YES</b> to execute now\n"
                                    f"Reply <b>NO</b> to skip\n"
                                    f"<i>(Expires in 5 minutes)</i>"
                                )
                                await notifier._bot.send_message(
                                    chat_id=config.telegram.chat_id,
                                    text=alert,
                                    parse_mode="HTML",
                                )
                                logger.info("Trade alert sent: %s %s score=%d", best.symbol, best.direction, best.score)

                except Exception as e:
                    logger.error("Scanner error: %s", e)

        # 4. Generate new signal
        signal = signal_gen.process(collector.data)
        bot_state.latest_signal = signal

        if signal.signal_type == SignalType.ENTRY:
            # HTF confluence check — skip if trade is against the daily trend
            if htf_filter and not htf_filter.allows(signal.direction):
                logger.info(
                    "HTF FILTER BLOCKED: %s signal vs daily bias=%s — skipping entry.",
                    signal.direction.value,
                    htf_filter.bias.value if htf_filter.bias else "FLAT",
                )
                return
            logger.info(
                "ENTRY SIGNAL: dir=%s entry=%.2f stop=%.2f tp=%.2f",
                signal.direction.value,
                signal.entry_price or 0,
                signal.suggested_stop or 0,
                signal.suggested_tp or 0,
            )
            await notifier.signal_entry(
                direction=signal.direction.value,
                symbol=signal.symbol,
                entry=signal.entry_price or 0,
                stop=signal.suggested_stop or 0,
                tp=signal.suggested_tp,
                reason=signal.reason,
            )

            # 3. Auto-execute if enabled, trade-capable, and no position already open
            if (
                bot_state.auto_trade
                and executor is not None
                and position_manager is not None
                and not position_manager.has_open_position()
            ):
                result = await _execute_trade(
                    signal=signal,
                    collector=collector,
                    client=client,
                    risk_manager=risk_manager,
                    executor=executor,
                    position_manager=position_manager,
                    notifier=notifier,
                    bot_state=bot_state,
                    pair_selector=pair_selector,
                    sentiment_filter=sentiment_filter,
                )
                logger.info("Auto-trade result: %s", result[:80])

        elif signal.signal_type == SignalType.EXIT:
            logger.info("EXIT SIGNAL: reason='%s'", signal.reason)
            await notifier.signal_exit(signal.symbol, signal.reason)

            if position_manager and position_manager.has_open_position():
                try:
                    await position_manager.close_position(reason=f"signal: {signal.reason}")
                except Exception as e:
                    logger.error("Failed to close on EXIT signal: %s", e)
                    await notifier.error(f"Position close failed: {e}")

    # --- Feature 3: Daily midnight P&L report ---
    async def _daily_pnl_report() -> None:
        """Send a P&L summary at midnight UTC every day."""
        from datetime import datetime, timezone, timedelta
        while True:
            try:
                now  = datetime.now(timezone.utc)
                next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
                wait_secs = (next_midnight - now).total_seconds()
                await asyncio.sleep(wait_secs)

                # Gather today's closed trades
                today = datetime.now(timezone.utc).date()
                today_str = today.strftime("%Y-%m-%d")
                try:
                    rows = trade_repo.closed_trades_since(today_str) if hasattr(trade_repo, "closed_trades_since") else []
                except Exception:
                    rows = []

                if rows:
                    total_pnl = sum(float(r.get("pnl_usd", 0)) for r in rows)
                    wins  = sum(1 for r in rows if float(r.get("pnl_usd", 0)) > 0)
                    losses = sum(1 for r in rows if float(r.get("pnl_usd", 0)) <= 0)
                    sign  = "+" if total_pnl >= 0 else ""
                    icon  = "🟢" if total_pnl >= 0 else "🔴"
                    msg = (
                        f"📊 <b>Daily P&amp;L Report — {today_str}</b>\n\n"
                        f"{icon} Net P&amp;L: <b>{sign}${total_pnl:.2f} USDT</b>\n"
                        f"Trades: {len(rows)}  (✅ {wins} W / ❌ {losses} L)\n"
                    )
                    for r in rows:
                        sym  = r.get("symbol", "?")
                        pnl  = float(r.get("pnl_usd", 0))
                        s    = "+" if pnl >= 0 else ""
                        msg += f"  • {sym}: {s}${pnl:.2f}\n"
                else:
                    msg = f"📊 <b>Daily P&amp;L — {today_str}</b>\nNo trades today."

                if config.telegram.enabled and notifier._bot:
                    await notifier._bot.send_message(
                        chat_id=config.telegram.chat_id,
                        text=msg,
                        parse_mode="HTML",
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Daily P&L report error: %s", e)
                await asyncio.sleep(60)

    # --- Feature: Weekly P&L report (every Monday 08:00 UTC) ---
    async def _weekly_pnl_report() -> None:
        from datetime import datetime, timezone, timedelta
        while True:
            try:
                now  = datetime.now(timezone.utc)
                # Next Monday 08:00 UTC
                days_until_monday = (7 - now.weekday()) % 7 or 7
                next_monday = (now + timedelta(days=days_until_monday)).replace(
                    hour=8, minute=0, second=0, microsecond=0
                )
                await asyncio.sleep((next_monday - now).total_seconds())

                week_start = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
                rows = trade_repo.weekly_pnl(week_start)
                today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

                if rows:
                    pnls  = [float(r["pnl_usd"] or 0) for r in rows]
                    total = sum(pnls)
                    wins  = sum(1 for p in pnls if p > 0)
                    sign  = "+" if total >= 0 else ""
                    icon  = "🟢" if total >= 0 else "🔴"
                    msg = (
                        f"📅 <b>Weekly P&amp;L Report</b>\n"
                        f"({week_start} → {today_str})\n\n"
                        f"{icon} Net: <b>{sign}${total:.2f} USDT</b>\n"
                        f"Trades: {len(pnls)}  (✅ {wins} W / ❌ {len(pnls)-wins} L)\n"
                    )
                    for r in rows[:10]:
                        p   = float(r["pnl_usd"] or 0)
                        s   = "+" if p > 0 else ""
                        sym = r["symbol"] if "symbol" in r.keys() else "?"
                        msg += f"  • {sym}: {s}${p:.2f}\n"
                    if len(rows) > 10:
                        msg += f"  … and {len(rows)-10} more\n"
                else:
                    msg = f"📅 <b>Weekly P&amp;L</b>\nNo trades this week."

                if config.telegram.enabled and notifier._bot:
                    await notifier._bot.send_message(
                        chat_id=config.telegram.chat_id, text=msg, parse_mode="HTML"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Weekly P&L report error: %s", e)
                await asyncio.sleep(3600)

    # --- Feature: Drawdown alert (checked every 15 min) ---
    async def _drawdown_monitor() -> None:
        from datetime import datetime, timezone
        DRAWDOWN_ALERT_PCT = float(os.environ.get("DRAWDOWN_ALERT_PCT", "10.0")) / 100.0
        peak_balance = 0.0
        alerted_at   = 0.0   # track so we don't spam
        while True:
            try:
                await asyncio.sleep(900)   # every 15 min
                if not (config.telegram.enabled and notifier._bot):
                    continue
                bal = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: client.fetch_account_info()
                )
                total = float(bal.get("USDT", {}).get("total", 0) or 0)
                if total <= 0:
                    continue
                if total > peak_balance:
                    peak_balance = total
                    alerted_at   = 0.0
                drawdown = (peak_balance - total) / peak_balance if peak_balance > 0 else 0
                if drawdown >= DRAWDOWN_ALERT_PCT and total != alerted_at:
                    alerted_at = total
                    await notifier._bot.send_message(
                        chat_id=config.telegram.chat_id,
                        text=(
                            f"📉 <b>Drawdown Alert!</b>\n\n"
                            f"Balance dropped <b>{drawdown*100:.1f}%</b> from peak.\n"
                            f"Peak: ${peak_balance:.2f}  Current: ${total:.2f}\n"
                            f"Loss: -${peak_balance - total:.2f} USDT\n\n"
                            f"Consider pausing trading with <b>/auto off</b>"
                        ),
                        parse_mode="HTML",
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Drawdown monitor error: %s", e)
                await asyncio.sleep(60)

    # --- Feature: Morning health check (08:30 UTC daily) ---
    async def _morning_health_check() -> None:
        from datetime import datetime, timezone, timedelta
        while True:
            try:
                now  = datetime.now(timezone.utc)
                next_check = now.replace(hour=8, minute=30, second=0, microsecond=0)
                if next_check <= now:
                    next_check += timedelta(days=1)
                await asyncio.sleep((next_check - now).total_seconds())

                if not (config.telegram.enabled and notifier._bot):
                    continue
                try:
                    bal    = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: client.fetch_account_info()
                    )
                    total  = float(bal.get("USDT", {}).get("total", 0) or 0)
                    free   = float(bal.get("USDT", {}).get("free",  0) or 0)
                except Exception:
                    total, free = 0.0, 0.0

                today_pnl = trade_repo.daily_pnl()
                auto_str  = "ON 🟢" if bot_state.auto_trade else "OFF 🔴"
                open_pos  = position_manager.has_open_position() if position_manager else False
                pos_str   = "Yes" if open_pos else "None"
                sign      = "+" if today_pnl >= 0 else ""

                await notifier._bot.send_message(
                    chat_id=config.telegram.chat_id,
                    text=(
                        f"☀️ <b>Good morning! Bot is running.</b>\n\n"
                        f"💰 Balance: <b>${total:.2f} USDT</b> (free: ${free:.2f})\n"
                        f"📊 Today's P&amp;L: <b>{sign}${today_pnl:.2f}</b>\n"
                        f"📂 Open position: {pos_str}\n"
                        f"🤖 Auto-trade: {auto_str}\n"
                        f"🕒 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
                    ),
                    parse_mode="HTML",
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Morning health check error: %s", e)
                await asyncio.sleep(3600)

    # --- Start Telegram command polling (parallel task) ---
    polling_task = None
    if tg_cmd_handler:
        polling_task = asyncio.create_task(tg_cmd_handler.start_polling())
        logger.info("Telegram command polling started.")

    daily_report_task   = None
    weekly_report_task  = None
    drawdown_task       = None
    health_check_task   = None
    if config.telegram.enabled and config.binance.trading_env in (TradingEnv.LIVE, TradingEnv.TESTNET):
        daily_report_task  = asyncio.create_task(_daily_pnl_report())
        weekly_report_task = asyncio.create_task(_weekly_pnl_report())
        drawdown_task      = asyncio.create_task(_drawdown_monitor())
        health_check_task  = asyncio.create_task(_morning_health_check())
        logger.info("Scheduled tasks started: daily P&L, weekly P&L, drawdown monitor, morning health check.")

    # --- Start channel monitor (parallel task) ---
    channel_task = None
    if channel_monitor:
        channel_task = asyncio.create_task(channel_monitor.start())
        logger.info("Channel monitor started: %s", channel)

    # --- WebSocket main loop ---
    ws = KlineWebSocket(config.data, config.strategy, on_bar_closed)
    logger.info(
        "Live stream starting: %s %s — Press Ctrl+C to stop.",
        config.strategy.symbol,
        config.strategy.timeframe,
    )

    try:
        await ws.run()
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
    finally:
        ws.stop()
        if polling_task:
            polling_task.cancel()
            try:
                await polling_task
            except asyncio.CancelledError:
                pass
        if channel_task:
            channel_task.cancel()
            try:
                await channel_task
            except asyncio.CancelledError:
                pass
        for _task in (daily_report_task, weekly_report_task, drawdown_task, health_check_task):
            if _task:
                _task.cancel()
                try:
                    await _task
                except asyncio.CancelledError:
                    pass
        if channel_monitor:
            await channel_monitor.stop()
        if tg_cmd_handler:
            await tg_cmd_handler.shutdown()
        await notifier.shutdown()
        db_conn.close()
        logger.info("Bot stopped cleanly.")


async def _execute_trade(
    signal: Optional[Signal],
    collector,
    client,
    risk_manager,
    executor,
    position_manager,
    notifier,
    bot_state: BotState,
    pair_selector=None,
    sentiment_filter=None,
) -> str:
    """
    Execute a trade. If pair_selector is available and no strong signal exists,
    scan all pairs and pick the best one automatically.
    """
    import datetime
    from src.signals.signal import SignalType, Direction, Signal
    from src.risk.manager import RiskVeto, RiskAssessment

    if position_manager and position_manager.has_open_position():
        pos = position_manager.get_current()
        return (
            f"Already in a {pos['direction'].upper()} position "
            f"(entry ${pos['entry_price']:,.4f}). Use /close first."
        )

    async with bot_state.trade_lock:
        try:
            account   = client.fetch_account_info()
            free_bal  = float(account.get("USDT", {}).get("free", 0) or 0)
            total_bal = float(account.get("USDT", {}).get("total", 0) or 0)
            equity    = free_bal if free_bal > 0 else total_bal

            # --- Path 1: existing ENTRY signal on the configured symbol ---
            if signal and signal.signal_type == SignalType.ENTRY:
                # Gemini news sentiment check
                if sentiment_filter:
                    coin = signal.symbol.split("/")[0]
                    sentiment = await sentiment_filter.check(coin, signal.direction.value.upper())
                    sentiment_msg = sentiment_filter.format_telegram(sentiment)
                    try:
                        await notifier._bot.send_message(
                            chat_id=notifier._chat_id, text=sentiment_msg, parse_mode="HTML"
                        )
                    except Exception:
                        pass
                    if sentiment.verdict == "RED":
                        return f"🔴 Trade BLOCKED by Gemini news check:\n{sentiment.summary}"

                assessment = risk_manager.evaluate(signal, equity, signal.entry_price or 0)
                orders     = executor.execute(assessment)
                position_manager.on_position_opened(assessment, orders)
                base = signal.symbol.split("/")[0]
                return (
                    f"✅ Trade executed!\n"
                    f"Pair: {signal.symbol}\n"
                    f"Direction: {signal.direction.value.upper()}\n"
                    f"Entry: ${assessment.signal.entry_price:,.5f}\n"
                    f"Stop: ${assessment.stop_price:,.5f}\n"
                    f"TP: ${assessment.take_profit_price:,.5f}\n"
                    f"Size: {assessment.position_size:.4f} {base}\n"
                    f"Risk: ${assessment.risk_usd:.2f} USDT"
                )

            # --- Path 2: smart pair selector ---
            if pair_selector:
                logger.info("No active signal — running smart pair selector...")
                trade = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: pair_selector.select(free_bal)
                )
                if trade is None:
                    return "🔍 Scanned all pairs — no qualifying setup right now. Bot is waiting for a signal."

                direction = Direction.LONG if trade.direction == "LONG" else Direction.SHORT
                now = datetime.datetime.now(datetime.timezone.utc)
                sig = Signal(
                    timestamp=now,
                    symbol=trade.symbol,
                    signal_type=SignalType.ENTRY,
                    direction=direction,
                    entry_price=trade.price,
                    suggested_stop=trade.stop,
                    suggested_tp=trade.tp,
                    atr=trade.atr,
                    reason=trade.reason,
                )
                assessment = RiskAssessment(
                    signal=sig,
                    position_size=trade.size,
                    implied_leverage=float(trade.leverage),
                    stop_price=trade.stop,
                    take_profit_price=trade.tp,
                    risk_usd=trade.size * abs(trade.price - trade.stop),
                )
                # Gemini news sentiment check
                if sentiment_filter:
                    coin = trade.symbol.split("/")[0]
                    sentiment = await sentiment_filter.check(coin, trade.direction)
                    sentiment_msg = sentiment_filter.format_telegram(sentiment)
                    try:
                        await notifier._bot.send_message(
                            chat_id=notifier._chat_id, text=sentiment_msg, parse_mode="HTML"
                        )
                    except Exception:
                        pass
                    if sentiment.verdict == "RED":
                        return f"🔴 Trade BLOCKED by Gemini news check:\n{sentiment.summary}"

                try:
                    client.set_leverage(trade.symbol, trade.leverage)
                except Exception as e:
                    logger.warning("Could not set leverage for %s: %s", trade.symbol, e)

                orders = executor.execute(assessment)
                position_manager.on_position_opened(assessment, orders)
                base = trade.symbol.split("/")[0]
                return (
                    f"✅ Smart trade executed!\n"
                    f"Pair: {trade.symbol}  [score {trade.score}/6]\n"
                    f"Direction: {trade.direction}\n"
                    f"Entry: ${trade.price:,.5f}\n"
                    f"Stop: ${trade.stop:,.5f}\n"
                    f"TP: ${trade.tp:,.5f}\n"
                    f"Size: {trade.size:.4f} {base} (${trade.notional:.2f} notional)\n"
                    f"Margin: ~${trade.margin:.2f} USDT"
                )

            return "No active ENTRY signal right now. Enable SCANNER_ENABLED=true for smart pair selection."

        except RiskVeto as e:
            msg = f"Trade BLOCKED by risk manager:\n{e}"
            logger.warning(msg)
            try:
                await notifier.circuit_breaker(str(e))
            except Exception:
                pass
            return msg

        except Exception as e:
            msg = f"Trade execution error: {e}"
            logger.error(msg, exc_info=True)
            try:
                await notifier.error(msg)
            except Exception:
                pass
            return msg


async def _execute_trade_amount(
    amount_usd: float,
    collector,
    client,
    strategy,
    config,
    executor,
    position_manager,
    bot_state: BotState,
    pair_selector=None,
) -> str:
    """
    Force a trade using a specific dollar amount as margin.
    If pair_selector is available, scans all pairs and picks the best one.
    Otherwise falls back to the configured symbol.
    """
    import datetime
    from src.signals.signal import Direction, Signal, SignalType
    from src.risk.manager import RiskAssessment

    async with bot_state.trade_lock:
        if position_manager and position_manager.has_open_position():
            pos = position_manager.get_current()
            return f"Already in a {pos['direction'].upper()} position (entry ${pos['entry_price']:,.4f}). Use /close first."

        try:
            account   = client.fetch_account_info()
            free_bal  = float(account.get("USDT", {}).get("free", 0) or 0)
            total_bal = float(account.get("USDT", {}).get("total", 0) or 0)
            free_bal  = free_bal if free_bal > 0 else total_bal or 1

            # --- Path 1: smart pair selector ---
            if pair_selector:
                trade = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: pair_selector.select_for_amount(free_bal, amount_usd)
                )
                if trade is None:
                    return "🔍 Scanned all pairs — no qualifying setup right now."

                direction = Direction.LONG if trade.direction == "LONG" else Direction.SHORT
                now = datetime.datetime.now(datetime.timezone.utc)
                signal = Signal(
                    timestamp=now, symbol=trade.symbol,
                    signal_type=SignalType.ENTRY, direction=direction,
                    entry_price=trade.price, suggested_stop=trade.stop,
                    suggested_tp=trade.tp, atr=trade.atr,
                    reason=trade.reason,
                )
                assessment = RiskAssessment(
                    signal=signal, position_size=trade.size,
                    implied_leverage=float(trade.leverage),
                    stop_price=trade.stop, take_profit_price=trade.tp,
                    risk_usd=trade.size * abs(trade.price - trade.stop),
                )
                try:
                    client.set_leverage(trade.symbol, trade.leverage)
                except Exception as e:
                    logger.warning("Could not set leverage for %s: %s", trade.symbol, e)

                orders = executor.execute(assessment)
                position_manager.on_position_opened(assessment, orders)
                base = trade.symbol.split("/")[0]
                return (
                    f"✅ Trade placed!\n"
                    f"Pair: {trade.symbol}  [score {trade.score}/6]\n"
                    f"Direction: {trade.direction}\n"
                    f"Entry: ${trade.price:,.5f}\n"
                    f"Stop: ${trade.stop:,.5f}\n"
                    f"TP: ${trade.tp:,.5f}\n"
                    f"Size: {trade.size:.4f} {base} (${trade.notional:.2f} notional)\n"
                    f"Margin: ~${trade.margin:.2f} USDT"
                )

            # --- Path 2: fallback to configured symbol ---
            if len(collector.data) < strategy.required_warmup_bars():
                return "Not enough data yet — warmup still in progress."

            df    = strategy.compute_indicators(collector.data)
            last  = df.iloc[-1]
            price = float(last["close"])
            ema   = float(last[f"ema_{config.strategy.ema_period}"])
            atr   = float(last[f"atr_{config.strategy.atr_period}"])

            if price >= ema:
                direction = Direction.LONG
                stop = price - atr * config.strategy.atr_stop_multiplier
                tp   = price + atr * config.strategy.atr_trail_multiplier
            else:
                direction = Direction.SHORT
                stop = price + atr * config.strategy.atr_stop_multiplier
                tp   = price - atr * config.strategy.atr_trail_multiplier

            leverage = int(config.risk.max_leverage)
            margin   = min(amount_usd, free_bal * 0.9)
            notional = max(margin * leverage, 23.0)
            notional = min(notional, free_bal * 0.9 * leverage)
            size     = notional / price

            now = datetime.datetime.now(datetime.timezone.utc)
            signal = Signal(
                timestamp=now, symbol=config.strategy.symbol,
                signal_type=SignalType.ENTRY, direction=direction,
                entry_price=price, suggested_stop=stop, suggested_tp=tp,
                atr=atr, reason=f"manual ${amount_usd:.2f}",
            )
            assessment = RiskAssessment(
                signal=signal, position_size=size,
                implied_leverage=float(leverage),
                stop_price=stop, take_profit_price=tp,
                risk_usd=size * abs(price - stop),
            )
            orders = executor.execute(assessment)
            position_manager.on_position_opened(assessment, orders)
            base = config.strategy.symbol.split("/")[0]
            return (
                f"✅ Trade placed!\n"
                f"Pair: {config.strategy.symbol}\n"
                f"Direction: {direction.value.upper()}\n"
                f"Entry: ${price:.5f}\n"
                f"Stop: ${stop:.5f}\n"
                f"TP: ${tp:.5f}\n"
                f"Size: {size:.4f} {base} (${notional:.2f} notional)\n"
                f"Margin: ~${notional/leverage:.2f} USDT"
            )

        except Exception as e:
            msg = f"Trade failed: {e}"
            logger.error(msg, exc_info=True)
            return msg


async def _execute_channel_signal(
    signal_dict: dict,
    client,
    config,
    executor,
    position_manager,
    bot_state: BotState,
    trade_repo=None,
) -> str:
    """
    Execute a parsed channel signal (from paste detection or /signal command).

    Improvements:
      - Fixed $1 margin per trade (not % of wallet)
      - Quality filter: checks historical win rate for this pair+direction
        and warns/skips if win rate is poor (configurable threshold)
    """
    import datetime
    from src.signals.signal import Direction, Signal, SignalType
    from src.risk.manager import RiskAssessment

    BINANCE_MAX_LEVERAGE = 125
    # Fixed margin per trade in USDT (overrides wallet_pct)
    FIXED_MARGIN_USD = float(os.environ.get("FIXED_MARGIN_USD", "1.0"))
    # Min win rate required to place a trade (0 = always trade, 0.4 = need 40%+)
    MIN_WIN_RATE = float(os.environ.get("MIN_WIN_RATE", "0.0"))

    try:
        symbol    = signal_dict["symbol"]
        direction = Direction.LONG if signal_dict["direction"] == "LONG" else Direction.SHORT
        stop      = float(signal_dict["stop_loss"])
        tp        = signal_dict["tp1"]
        leverage  = min(int(signal_dict.get("leverage", 5)), BINANCE_MAX_LEVERAGE)

        # ── Quality Filter ──────────────────────────────────────────
        quality_line = ""
        if trade_repo:
            stats = trade_repo.pair_win_rate(symbol, direction.value)
            if stats["enough_data"]:
                wr = stats["win_rate"]
                w  = stats["wins"]
                l  = stats["losses"]
                icon = "✅" if wr >= 0.5 else "⚠️" if wr >= 0.35 else "❌"
                quality_line = f"\n📊 {symbol} {direction.value.upper()} history: {w}W/{l}L ({wr*100:.0f}% win rate) {icon}"
                if MIN_WIN_RATE > 0 and wr < MIN_WIN_RATE:
                    return (
                        f"🚫 <b>Trade skipped — poor historical performance</b>\n"
                        f"{symbol} {direction.value.upper()}: {w}W / {l}L ({wr*100:.0f}% win rate)\n"
                        f"Required minimum: {MIN_WIN_RATE*100:.0f}%\n"
                        f"Set MIN_WIN_RATE=0 in .env to disable this filter."
                    )
            else:
                n = stats["trades"]
                quality_line = f"\n📊 {symbol} {direction.value.upper()} history: {n} trade(s) — not enough data yet"

        # ── Price & balance ─────────────────────────────────────────
        ticker = client.fetch_ticker(symbol)
        price  = float(ticker["last"])

        if direction == Direction.LONG and stop >= price:
            return f"❌ Invalid signal: LONG stop ({stop}) must be below price ({price:.5f})"
        if direction == Direction.SHORT and stop <= price:
            return f"❌ Invalid signal: SHORT stop ({stop}) must be above price ({price:.5f})"

        account       = client.fetch_account_info()
        free_balance  = float(account.get("USDT", {}).get("free",  0) or 0)
        total_balance = float(account.get("USDT", {}).get("total", 0) or 0)
        equity        = free_balance if free_balance > 0 else total_balance
        if equity < 1.0:
            return f"❌ Insufficient free balance: ${equity:.2f} USDT"

        # ── Fixed $1 margin (improvement #2) ────────────────────────
        margin   = min(FIXED_MARGIN_USD, equity * 0.9)   # never exceed 90% of balance
        notional = max(margin * leverage, 23.0)           # Binance minimum $5, we use $23 buffer
        notional = min(notional, equity * 0.9 * leverage)
        size     = notional / price

        # ── Build & execute ─────────────────────────────────────────
        now = datetime.datetime.now(datetime.timezone.utc)
        signal = Signal(
            timestamp=now,
            symbol=symbol,
            signal_type=SignalType.ENTRY,
            direction=direction,
            entry_price=price,
            suggested_stop=stop,
            suggested_tp=tp,
            reason=f"channel signal {direction.value}",
        )
        tp2 = signal_dict.get("tp2") or 0.0
        tp3 = signal_dict.get("tp3") or 0.0
        assessment = RiskAssessment(
            signal=signal,
            position_size=size,
            implied_leverage=float(leverage),
            stop_price=stop,
            take_profit_price=tp,
            risk_usd=size * abs(price - stop),
            leverage_override=leverage,
            tp2_price=float(tp2) if tp2 else 0.0,
            tp3_price=float(tp3) if tp3 else 0.0,
        )

        orders = executor.execute(assessment)
        position_manager.on_position_opened(assessment, orders)

        actual_margin = notional / leverage
        tp_str  = f"${tp:.5f}" if tp else "none"
        capped  = " (capped)" if int(signal_dict.get("leverage", 5)) > BINANCE_MAX_LEVERAGE else ""
        tp2_str = f"\nTP2: ${float(tp2):.5f} (25%)" if tp2 else ""
        tp3_str = f"\nTP3: ${float(tp3):.5f} (25%)" if tp3 else ""
        return (
            f"✅ <b>Signal trade executed!</b>{quality_line}\n\n"
            f"Pair: {symbol}\n"
            f"Direction: {direction.value.upper()}\n"
            f"Entry: ${price:.5f}\n"
            f"Stop Loss: ${stop:.5f}\n"
            f"TP1: {tp_str} (50%){tp2_str}{tp3_str}\n"
            f"Size: {size:.4f} (${notional:.2f} notional)\n"
            f"Margin: ~${actual_margin:.2f} USDT\n"
            f"Leverage: {leverage}x{capped}"
        )

    except Exception as e:
        msg = f"Signal trade failed: {e}"
        logger.error(msg, exc_info=True)
        return msg


def main() -> None:
    _acquire_lock()
    try:
        asyncio.run(async_main())
    finally:
        _release_lock()


if __name__ == "__main__":
    main()
