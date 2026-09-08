"""Unit tests for WebSocket message parsing — no network needed."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.config import DataConfig, StrategyConfig
from src.data.websocket import KlineWebSocket, _to_stream_symbol, _to_stream_interval


def make_ws(callback=None) -> KlineWebSocket:
    cb = callback or AsyncMock()
    return KlineWebSocket(DataConfig(), StrategyConfig(), cb)


class TestSymbolConversion:
    def test_btc_usdt_perp(self):
        assert _to_stream_symbol("BTC/USDT:USDT") == "btcusdt"

    def test_eth_usdt_perp(self):
        assert _to_stream_symbol("ETH/USDT:USDT") == "ethusdt"

    def test_interval_passthrough(self):
        assert _to_stream_interval("1h") == "1h"
        assert _to_stream_interval("15m") == "15m"


class TestMessageHandling:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_closed_bar_triggers_callback(self):
        cb = AsyncMock()
        ws = make_ws(cb)
        msg = '{"e":"kline","s":"BTCUSDT","k":{"t":1700000000000,"o":"50000","h":"51000","l":"49000","c":"50500","v":"100","x":true}}'
        self._run(ws._handle_message(msg))
        cb.assert_awaited_once()
        bar = cb.call_args[0][0]
        assert bar["close"] == "50500"
        assert bar["timestamp"] == 1700000000000

    def test_open_bar_ignored(self):
        """Bars with x=false (not yet closed) must be silently ignored."""
        cb = AsyncMock()
        ws = make_ws(cb)
        msg = '{"e":"kline","s":"BTCUSDT","k":{"t":1700000000000,"o":"50000","h":"51000","l":"49000","c":"50500","v":"100","x":false}}'
        self._run(ws._handle_message(msg))
        cb.assert_not_awaited()

    def test_non_kline_message_ignored(self):
        cb = AsyncMock()
        ws = make_ws(cb)
        self._run(ws._handle_message('{"e":"trade","p":"50000"}'))
        cb.assert_not_awaited()

    def test_malformed_json_ignored(self):
        cb = AsyncMock()
        ws = make_ws(cb)
        self._run(ws._handle_message("not json at all"))
        cb.assert_not_awaited()
