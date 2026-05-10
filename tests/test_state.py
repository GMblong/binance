"""Unit tests for utils/state.py"""
import pytest
from utils.state import BotState, MarketData


class TestBotState:
    def setup_method(self):
        self.state = BotState()

    def test_default_values(self):
        assert self.state["balance"] == 0.0
        assert self.state["is_passive"] is False
        assert self.state["api_health_status"] == "OK"

    def test_missing_key_returns_none(self):
        assert self.state["nonexistent_key"] is None

    def test_set_and_get(self):
        self.state["balance"] = 100.0
        assert self.state["balance"] == 100.0

    def test_safe_increment(self):
        self.state["heartbeat"] = 0
        self.state.safe_increment("heartbeat")
        assert self.state["heartbeat"] == 1
        self.state.safe_increment("heartbeat", 5)
        assert self.state["heartbeat"] == 6

    def test_get_with_default(self):
        assert self.state.get("missing", "fallback") == "fallback"

    def test_dict_operations(self):
        self.state["test_key"] = "value"
        assert "test_key" in self.state
        del self.state["test_key"]
        assert self.state["test_key"] is None


class TestMarketData:
    def setup_method(self):
        self.md = MarketData()

    def test_push_agg_trade(self):
        import time
        now = time.time()
        self.md.push_agg_trade("BTCUSDT", now, 0.5, 50000.0, False)
        cvd, n = self.md.get_live_cvd("BTCUSDT", window_sec=9999)
        assert n == 1
        assert cvd > 0  # Buyer aggressive = positive

    def test_push_agg_trade_seller(self):
        import time
        now = time.time()
        self.md.push_agg_trade("ETHUSDT", now, 1.0, 3000.0, True)
        cvd, n = self.md.get_live_cvd("ETHUSDT", window_sec=9999)
        assert n == 1
        assert cvd < 0  # Seller aggressive = negative

    def test_get_trades_empty(self):
        trades = self.md.get_trades("UNKNOWN")
        assert trades == []

    def test_get_live_cvd_empty(self):
        cvd, n = self.md.get_live_cvd("UNKNOWN")
        assert cvd == 0.0
        assert n == 0

    def test_microprice_none_when_empty(self):
        assert self.md.get_microprice("BTCUSDT") is None

    def test_push_best_quote_and_microprice(self):
        import time
        self.md.push_best_quote("BTCUSDT", 50000.0, 10.0, 50001.0, 5.0)
        mp = self.md.get_microprice("BTCUSDT")
        # microprice = (bid*ask_qty + ask*bid_qty) / (bid_qty + ask_qty)
        expected = (50000.0 * 5.0 + 50001.0 * 10.0) / (10.0 + 5.0)
        assert abs(mp - expected) < 0.01

    def test_depth_velocity_empty(self):
        bv, av, ss = self.md.get_depth_velocity("UNKNOWN")
        assert bv == 0.0 and av == 0.0 and ss == 0.0
