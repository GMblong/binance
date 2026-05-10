"""Unit tests for engine/ml_engine.py feature extraction and model logic."""
import pytest
import numpy as np
import pandas as pd
from engine.ml_engine import MLPredictor


@pytest.fixture
def predictor():
    return MLPredictor()


@pytest.fixture
def sample_df():
    """Create realistic OHLCV dataframe for testing."""
    np.random.seed(42)
    n = 250
    base_price = 50000.0
    returns = np.random.normal(0, 0.002, n)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "ot": np.arange(n) * 60000,
        "o": prices * (1 - np.random.uniform(0, 0.001, n)),
        "h": prices * (1 + np.random.uniform(0, 0.003, n)),
        "l": prices * (1 - np.random.uniform(0, 0.003, n)),
        "c": prices,
        "v": np.random.uniform(100, 1000, n),
    })
    return df


class TestFeatureEngineering:
    def test_produces_required_features(self, predictor, sample_df):
        result = predictor.feature_engineering(sample_df.copy())
        required = ['ema_9', 'ema_21', 'rsi', 'atr', 'roc_c_1', 'roc_c_5',
                    'roc_v_1', 'volatility', 'body_size', 'upper_wick', 'lower_wick',
                    'dist_ema9', 'dist_ema21', 'dist_vwap', 'cvd_roc']
        for feat in required:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_no_inf_values(self, predictor, sample_df):
        result = predictor.feature_engineering(sample_df.copy())
        features = predictor._get_feature_list(result.columns)
        for feat in features:
            if feat in result.columns:
                assert not np.isinf(result[feat]).any(), f"Inf in {feat}"

    def test_training_mode_drops_nan(self, predictor, sample_df):
        result = predictor.feature_engineering(sample_df.copy())
        features = predictor._get_feature_list(result.columns)
        for feat in features:
            if feat in result.columns:
                assert not result[feat].isna().any(), f"NaN in {feat}"

    def test_prediction_mode_returns_single_row(self, predictor):
        """Short df (< 200 rows) should return tail(1)."""
        np.random.seed(42)
        n = 100
        prices = 50000 + np.cumsum(np.random.normal(0, 10, n))
        df = pd.DataFrame({
            "ot": np.arange(n) * 60000,
            "o": prices - 5,
            "h": prices + 20,
            "l": prices - 20,
            "c": prices,
            "v": np.random.uniform(100, 500, n),
        })
        result = predictor.feature_engineering(df)
        assert len(result) == 1

    def test_macd_columns_present(self, predictor, sample_df):
        result = predictor.feature_engineering(sample_df.copy())
        assert 'MACD_12_26_9' in result.columns
        assert 'MACDs_12_26_9' in result.columns
        assert 'MACDh_12_26_9' in result.columns


class TestTripleBarrier:
    def test_labels_are_0_or_1(self, predictor):
        """Test triple barrier with clean synthetic data."""
        np.random.seed(42)
        n = 300
        prices = 100 + np.cumsum(np.random.normal(0, 0.5, n))
        df = pd.DataFrame({
            "c": prices,
            "h": prices + np.abs(np.random.normal(0, 0.3, n)),
            "l": prices - np.abs(np.random.normal(0, 0.3, n)),
            "atr": np.full(n, 1.0),
        })
        result = predictor.apply_triple_barrier(df, lookahead=10)
        assert 'target' in result.columns
        assert set(result['target'].unique()).issubset({0, 1})
        # Should have dropped last 10 rows (NaN targets)
        assert len(result) == n - 10

    def test_shorter_df_fewer_valid(self, predictor):
        """Shorter df with larger lookahead = fewer valid labels."""
        n = 50
        df = pd.DataFrame({
            "c": np.linspace(100, 110, n),
            "h": np.linspace(101, 111, n),
            "l": np.linspace(99, 109, n),
            "atr": np.full(n, 0.5),
        })
        result = predictor.apply_triple_barrier(df, lookahead=20)
        assert len(result) == n - 20


class TestPerformanceTracking:
    def test_update_performance(self, predictor):
        predictor.update_performance("BTCUSDT", True)
        predictor.update_performance("BTCUSDT", True)
        predictor.update_performance("BTCUSDT", False)
        assert predictor.performance["BTCUSDT"] == [True, True, False]

    def test_recent_win_rate(self, predictor):
        for _ in range(10):
            predictor.update_performance("ETHUSDT", True)
        for _ in range(5):
            predictor.update_performance("ETHUSDT", False)
        wr = predictor.recent_win_rate("ETHUSDT", window=15)
        assert abs(wr - 10/15) < 0.01

    def test_should_retrain_new_symbol(self, predictor):
        assert predictor.should_retrain("NEWCOIN") is True

    def test_should_retrain_after_many_trades(self, predictor):
        predictor.models["BTCUSDT"] = {"lgb": None}
        predictor.last_trained["BTCUSDT"] = 9999999999  # Far future
        predictor.trades_since_train["BTCUSDT"] = 50
        assert predictor.should_retrain("BTCUSDT") is True

    def test_performance_capped_at_50(self, predictor):
        for _ in range(60):
            predictor.update_performance("X", True)
        assert len(predictor.performance["X"]) == 50
