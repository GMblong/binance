"""Unit tests for utils/helpers.py"""
import pytest
from utils.helpers import get_signature, round_step


class TestGetSignature:
    def test_basic_signature(self):
        sig = get_signature("symbol=BTCUSDT&timestamp=123456", "mysecret")
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA256 hex digest

    def test_deterministic(self):
        s1 = get_signature("a=1", "key")
        s2 = get_signature("a=1", "key")
        assert s1 == s2

    def test_different_keys_different_sigs(self):
        s1 = get_signature("a=1", "key1")
        s2 = get_signature("a=1", "key2")
        assert s1 != s2

    def test_empty_query(self):
        sig = get_signature("", "secret")
        assert len(sig) == 64


class TestRoundStep:
    def test_basic_rounding(self):
        assert round_step(0.12345, 0.001) == 0.123

    def test_rounds_down(self):
        assert round_step(0.9999, 0.01) == 0.99

    def test_step_1(self):
        assert round_step(123.7, 1) == 123.0

    def test_step_0_1(self):
        assert round_step(1.567, 0.1) == 1.5

    def test_step_0_00001(self):
        assert round_step(0.123456, 0.00001) == 0.12345

    def test_zero_step_returns_value(self):
        assert round_step(1.5, 0) == 1.5

    def test_none_step_returns_value(self):
        assert round_step(1.5, None) == 1.5

    def test_infinity_returns_value(self):
        assert round_step(float('inf'), 0.01) == float('inf')

    def test_very_small_quantity(self):
        assert round_step(0.001, 0.001) == 0.001

    def test_exact_multiple(self):
        assert round_step(0.5, 0.1) == 0.5
