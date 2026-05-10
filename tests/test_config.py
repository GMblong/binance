"""Unit tests for utils/config.py"""
import pytest
from utils.config import (
    MIN_NOTIONAL_USD, MIN_VOLUME_FILTER, MIN_VOLUME_SCREENER,
    CONSEC_LOSS_COOLDOWN_SEC, MAX_CONSEC_LOSSES, ML_RETRAIN_INTERVAL_SEC,
    API_BAN_SLEEP_SEC, KLINE_MAX_CANDLES, DB_SAVE_INTERVAL_SEC,
    ACCOUNT_RISK_PERCENT, MAX_LEVERAGE, DAILY_LOSS_LIMIT_PCT, DAILY_PROFIT_TARGET_PCT,
)


class TestConfigConstants:
    def test_min_notional_positive(self):
        assert MIN_NOTIONAL_USD > 0

    def test_risk_percent_reasonable(self):
        assert 0 < ACCOUNT_RISK_PERCENT <= 0.1  # Max 10% per trade

    def test_leverage_bounded(self):
        assert 1 <= MAX_LEVERAGE <= 125

    def test_daily_limits_positive(self):
        assert 0 < DAILY_LOSS_LIMIT_PCT < 1.0
        assert 0 < DAILY_PROFIT_TARGET_PCT < 1.0

    def test_cooldown_reasonable(self):
        assert CONSEC_LOSS_COOLDOWN_SEC >= 60  # At least 1 min
        assert MAX_CONSEC_LOSSES >= 1

    def test_ml_retrain_interval(self):
        assert ML_RETRAIN_INTERVAL_SEC >= 3600  # At least 1 hour

    def test_volume_filters_ordered(self):
        assert MIN_VOLUME_FILTER <= MIN_VOLUME_SCREENER
