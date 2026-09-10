"""
Binance API integration layer.

All ccxt calls live here. No other module should import ccxt directly.
This is the single choke point for rate-limit awareness.
"""

import logging
import ccxt

from src.config import AppConfig, TradingEnv

logger = logging.getLogger(__name__)


def build_exchange(config: AppConfig) -> ccxt.binanceusdm:
    """
    Construct and return a ccxt binanceusdm (USDT-M Futures) exchange instance.

    binanceusdm is the ccxt market type for Binance USD-margined perpetual futures.
    We pick it over 'binance' (spot) because the brief targets futures trading.
    """
    options: dict = {
        "defaultType": "future",
        "enableRateLimit": True,
        # Prevent ccxt from calling the Spot wallet API (sapi/v1/capital/config/getall)
        # which requires Spot permissions we don't have on a futures-only key.
        "fetchCurrencies": False,
    }

    if config.binance.trading_env == TradingEnv.TESTNET:
        options["testnet"] = True
        logger.info("Exchange configured for TESTNET")
    else:
        logger.info("Exchange configured for MAINNET (env=%s)", config.binance.trading_env.value)

    exchange = ccxt.binanceusdm(
        {
            "apiKey": config.binance.api_key,
            "secret": config.binance.api_secret,
            "options": options,
        }
    )

    return exchange


class BinanceClient:
    """
    Thin wrapper around a ccxt exchange that enforces the trading environment
    contract: in mainnet_readonly mode, any call that would place or modify
    orders raises immediately rather than silently doing nothing.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.env = config.binance.trading_env
        self._exchange = build_exchange(config)

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------

    def _require_trade_capable(self, action: str) -> None:
        """Raise if the current env cannot place orders."""
        if self.env == TradingEnv.MAINNET_READONLY:
            raise PermissionError(
                f"Attempted '{action}' in MAINNET_READONLY mode. "
                "This key cannot place, modify, or cancel orders. "
                "Switch TRADING_ENV to 'testnet' or 'live' with a trade-capable key."
            )

    # ------------------------------------------------------------------
    # Market data (always allowed)
    # ------------------------------------------------------------------

    def fetch_account_info(self) -> dict:
        """Return futures account balance/info. Works with the read-only key."""
        # binanceusdm is futures-only — no type param needed (it causes a v3 endpoint
        # mismatch that returns an empty/error response on newer ccxt versions).
        return self._exchange.fetch_balance()

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list:
        """Fetch historical OHLCV bars. REST call — used for warmup."""
        return self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_ticker(self, symbol: str) -> dict:
        return self._exchange.fetch_ticker(symbol)

    def fetch_exchange_info(self) -> dict:
        """
        Load markets (symbol precision, min notional, step sizes).
        Must be called before placing any order so we respect exchange constraints.
        """
        return self._exchange.load_markets(reload=True)

    def fetch_funding_rate(self, symbol: str) -> dict:
        """Current funding rate. Used by the backtester to model funding costs."""
        return self._exchange.fetch_funding_rate(symbol)

    def fetch_funding_rate_history(self, symbol: str, since: int | None = None, limit: int = 500) -> list:
        return self._exchange.fetch_funding_rate_history(symbol, since=since, limit=limit)

    # ------------------------------------------------------------------
    # Order management (gated)
    # ------------------------------------------------------------------

    def create_market_order(self, symbol: str, side: str, amount: float, params: dict | None = None) -> dict:
        self._require_trade_capable("create_market_order")
        return self._exchange.create_market_order(symbol, side, amount, params=params or {})

    def create_stop_market_order(self, symbol: str, side: str, amount: float, stop_price: float, params: dict | None = None) -> dict:
        self._require_trade_capable("create_stop_market_order")
        p = {"stopPrice": stop_price, "reduceOnly": True, **(params or {})}
        return self._exchange.create_order(symbol, "STOP_MARKET", side, amount, params=p)

    def create_take_profit_market_order(self, symbol: str, side: str, amount: float, stop_price: float, params: dict | None = None) -> dict:
        self._require_trade_capable("create_take_profit_market_order")
        p = {"stopPrice": stop_price, "reduceOnly": True, **(params or {})}
        return self._exchange.create_order(symbol, "TAKE_PROFIT_MARKET", side, amount, params=p)

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        self._require_trade_capable("cancel_order")
        return self._exchange.cancel_order(order_id, symbol)

    def fetch_open_orders(self, symbol: str) -> list:
        self._require_trade_capable("fetch_open_orders")
        return self._exchange.fetch_open_orders(symbol)

    def fetch_positions(self, symbol: str | None = None) -> list:
        self._require_trade_capable("fetch_positions")
        symbols = [symbol] if symbol else None
        return self._exchange.fetch_positions(symbols)

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        self._require_trade_capable("set_leverage")
        return self._exchange.set_leverage(leverage, symbol)

    def set_margin_mode(self, symbol: str, mode: str = "isolated") -> None:
        """
        Switch a symbol to ISOLATED or CROSS margin mode.

        Binance requires no open position on the symbol when switching.
        Silently ignores the "already set" error (-4046) so this is safe
        to call unconditionally before every trade.
        """
        self._require_trade_capable("set_margin_mode")
        try:
            self._exchange.set_margin_mode(mode, symbol)
            logger.info("Margin mode set to %s for %s", mode.upper(), symbol)
        except Exception as e:
            # -4046: "No need to change margin type." — already isolated, ignore.
            if "-4046" in str(e) or "No need to change" in str(e):
                logger.debug("Margin mode already %s for %s — skipping", mode.upper(), symbol)
            else:
                raise

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def exchange(self) -> ccxt.binanceusdm:
        """Direct access for operations not wrapped above. Use sparingly."""
        return self._exchange
