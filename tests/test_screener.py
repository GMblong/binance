"""Unit tests for coin_screener.py"""
import pytest
from coin_screener import screen_coins


class TestScreenCoins:
    def test_empty_tickers(self):
        assert screen_coins([]) == []

    def test_filters_low_volume(self):
        tickers = [
            {"s": "BTCUSDT", "q": 100_000_000, "c": 50000, "o": 49000},
            {"s": "LOWUSDT", "q": 1_000, "c": 1.0, "o": 1.0},  # Too low
        ]
        result = screen_coins(tickers, top_n=5)
        assert "BTCUSDT" in result
        assert "LOWUSDT" not in result

    def test_returns_max_top_n(self):
        tickers = [
            {"s": f"COIN{i}USDT", "q": 50_000_000 + i * 1000, "c": 10.0, "o": 9.5}
            for i in range(20)
        ]
        result = screen_coins(tickers, top_n=5)
        assert len(result) <= 5

    def test_only_usdt_pairs(self):
        tickers = [
            {"s": "BTCUSDT", "q": 100_000_000, "c": 50000, "o": 49000},
            {"s": "ETHBTC", "q": 100_000_000, "c": 0.05, "o": 0.05},
        ]
        result = screen_coins(tickers, top_n=5)
        assert "ETHBTC" not in result
