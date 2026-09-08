"""
Central configuration module.

All settings come from environment variables (or a .env file loaded at startup).
Nothing secret ever lives in this file — only structure and defaults.
"""

import os
from dataclasses import dataclass, field
from enum import Enum


class TradingEnv(str, Enum):
    MAINNET_READONLY = "mainnet_readonly"  # Real data, zero order capability
    TESTNET = "testnet"                    # Testnet keys, fake money
    LIVE = "live"                          # Real keys, real money — requires explicit sign-off


@dataclass
class BinanceConfig:
    api_key: str
    api_secret: str
    trading_env: TradingEnv


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.005       # 0.5% of equity per trade
    max_leverage: int = 3                   # hard cap on leverage
    daily_loss_limit_pct: float = 0.03      # halt for the day at 3% cumulative loss
    max_drawdown_pct: float = 0.15          # halt entirely at 15% drawdown from peak


@dataclass
class StrategyConfig:
    symbol: str = "BTC/USDT:USDT"          # ccxt unified symbol for BTC perp
    timeframe: str = "1h"
    ema_period: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0        # stop = entry ± (atr * multiplier)
    atr_trail_multiplier: float = 4.0       # trailing stop distance
    rsi_entry_threshold: float = 45.0       # RSI level for entry cross
    adx_threshold: float = 0.0             # skip trades when ADX < threshold (0 = off)
    volume_filter: float = 0.0             # skip when rel_volume < N × 20-bar avg (0 = off)
    sr_lookback: int = 0                   # bars to look back for S/R pivots (0 = off)
    sr_zone_atr_mult: float = 0.5          # how close to S/R (in ATR units) blocks entry
    macd_filter: bool = False              # require MACD histogram to confirm direction
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    partial_tp_r_multiple: float = 2.0      # take 50% off at 2R
    warmup_bars: int = 100                  # bars needed before strategy is ready


@dataclass
class DataConfig:
    kline_limit: int = 500                  # historical bars to fetch at warmup
    websocket_reconnect_delay: float = 5.0  # seconds before WS reconnect attempt


@dataclass
class DatabaseConfig:
    path: str = "data/trading_bot.db"


@dataclass
class TelegramConfig:
    token: str = ""
    chat_id: str = ""
    enabled: bool = False


@dataclass
class AppConfig:
    binance: BinanceConfig
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)


def load_config() -> AppConfig:
    """
    Build AppConfig from environment variables.
    Raises immediately with a clear message if required vars are missing.
    """
    missing = []

    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")
    env_str = os.environ.get("TRADING_ENV", "")

    if not api_key:
        missing.append("BINANCE_API_KEY")
    if not api_secret:
        missing.append("BINANCE_API_SECRET")
    if not env_str:
        missing.append("TRADING_ENV")

    if missing:
        raise EnvironmentError(
            f"Required environment variables not set: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in your values."
        )

    try:
        trading_env = TradingEnv(env_str)
    except ValueError:
        valid = [e.value for e in TradingEnv]
        raise EnvironmentError(
            f"TRADING_ENV='{env_str}' is not valid. Must be one of: {valid}"
        )

    telegram_token = os.environ.get("TELEGRAM_TOKEN", "")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    return AppConfig(
        binance=BinanceConfig(
            api_key=api_key,
            api_secret=api_secret,
            trading_env=trading_env,
        ),
        risk=RiskConfig(
            risk_per_trade_pct=float(os.environ.get("RISK_PER_TRADE_PCT", "0.005")),
            max_leverage=int(os.environ.get("MAX_LEVERAGE", "3")),
            daily_loss_limit_pct=float(os.environ.get("DAILY_LOSS_LIMIT_PCT", "0.03")),
            max_drawdown_pct=float(os.environ.get("MAX_DRAWDOWN_PCT", "0.15")),
        ),
        strategy=StrategyConfig(
            symbol=os.environ.get("SYMBOL", "BTC/USDT:USDT"),
            timeframe=os.environ.get("TIMEFRAME", "1h"),
            ema_period=int(os.environ.get("EMA_PERIOD", "50")),
            rsi_period=int(os.environ.get("RSI_PERIOD", "14")),
            atr_period=int(os.environ.get("ATR_PERIOD", "14")),
            atr_stop_multiplier=float(os.environ.get("ATR_STOP_MULTIPLIER", "2.0")),
            atr_trail_multiplier=float(os.environ.get("ATR_TRAIL_MULTIPLIER", "4.0")),
            rsi_entry_threshold=float(os.environ.get("RSI_ENTRY_THRESHOLD", "45.0")),
            adx_threshold=float(os.environ.get("ADX_THRESHOLD", "0.0")),
            volume_filter=float(os.environ.get("VOLUME_FILTER", "0.0")),
            sr_lookback=int(os.environ.get("SR_LOOKBACK", "0")),
            sr_zone_atr_mult=float(os.environ.get("SR_ZONE_ATR_MULT", "0.5")),
            macd_filter=os.environ.get("MACD_FILTER", "false").lower() == "true",
            macd_fast=int(os.environ.get("MACD_FAST", "12")),
            macd_slow=int(os.environ.get("MACD_SLOW", "26")),
            macd_signal=int(os.environ.get("MACD_SIGNAL", "9")),
            partial_tp_r_multiple=float(os.environ.get("PARTIAL_TP_R_MULTIPLE", "2.0")),
            warmup_bars=int(os.environ.get("WARMUP_BARS", "100")),
        ),
        data=DataConfig(
            kline_limit=int(os.environ.get("KLINE_LIMIT", "500")),
        ),
        database=DatabaseConfig(
            path=os.environ.get("DB_PATH", "data/trading_bot.db"),
        ),
        telegram=TelegramConfig(
            token=telegram_token,
            chat_id=telegram_chat_id,
            enabled=bool(telegram_token and telegram_chat_id),
        ),
    )
