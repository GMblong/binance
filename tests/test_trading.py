"""Integration tests for engine/trading.py with mocked API."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from utils.state import BotState, MarketData


@pytest.fixture
def fresh_state(monkeypatch):
    """Reset bot_state and market_data for each test."""
    state = BotState()
    state["balance"] = 1000.0
    state["start_balance"] = 1000.0
    md = MarketData()
    md.prices["BTCUSDT"] = 50000.0
    monkeypatch.setattr("engine.trading.bot_state", state)
    monkeypatch.setattr("engine.trading.market_data", md)
    return state


@pytest.fixture
def mock_client():
    """Mock httpx client."""
    return AsyncMock()


def _mock_precision():
    """Return mock symbol precision data."""
    return {"tick": 0.1, "step": 0.001, "p_prec": 1, "q_prec": 3}


def _make_api_response(status_code=200, json_data=None):
    """Create a mock API response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


class TestOpenPosition:
    @pytest.mark.asyncio
    async def test_open_market_order_success(self, fresh_state, mock_client):
        ai_brain = {
            "limit_price": 50000.0,
            "sl": 1.0,
            "tp": 2.0,
            "is_market": True,
            "ts_act": 0.8,
            "ts_cb": 0.25,
            "type": "INTRA",
            "regime": "TRENDING",
            "atr_pct": 0.5,
            "ml_prob": 0.7,
            "active_features": ["EMA_CROSS"],
        }

        order_response = _make_api_response(200, {
            "orderId": 12345,
            "avgPrice": "50000.0",
            "status": "FILLED",
        })
        leverage_response = _make_api_response(200, {"maxNotionalValue": "1000000"})
        sl_tp_response = _make_api_response(200, {"orderId": 99})

        with patch("engine.trading.get_symbol_precision", return_value=_mock_precision()):
            with patch("engine.trading.binance_request") as mock_req:
                mock_req.side_effect = [
                    leverage_response,  # leverage set
                    _make_api_response(200, {}),  # margin type
                    order_response,  # main order
                    sl_tp_response,  # SL
                    sl_tp_response,  # TP
                ]
                from engine.trading import open_position_async
                result = await open_position_async(mock_client, "BTCUSDT", "BUY", "EMA_CROSS", ai_brain)

        assert result is True
        assert "BTCUSDT" in fresh_state["trades"]
        assert fresh_state["trades"]["BTCUSDT"]["side"] == "LONG"

    @pytest.mark.asyncio
    async def test_open_order_insufficient_balance(self, fresh_state, mock_client):
        fresh_state["balance"] = 0.0
        ai_brain = {
            "limit_price": 50000.0,
            "sl": 1.0,
            "tp": 2.0,
            "is_market": True,
            "ml_prob": 0.5,
        }

        with patch("engine.trading.get_symbol_precision", return_value=_mock_precision()):
            with patch("engine.trading.binance_request") as mock_req:
                mock_req.return_value = _make_api_response(200, {"maxNotionalValue": "1000000"})
                from engine.trading import open_position_async
                result = await open_position_async(mock_client, "BTCUSDT", "BUY", "TEST", ai_brain)

        # Should fail because quantity would be 0 with 0 balance
        assert result is False

    @pytest.mark.asyncio
    async def test_open_limit_order_stores_in_limit_orders(self, fresh_state, mock_client):
        ai_brain = {
            "limit_price": 49500.0,
            "sl": 1.0,
            "tp": 2.0,
            "is_market": False,
            "ml_prob": 0.65,
            "active_features": ["RSI_OB"],
        }

        order_response = _make_api_response(200, {"orderId": 67890, "status": "NEW"})
        leverage_response = _make_api_response(200, {"maxNotionalValue": "1000000"})

        with patch("engine.trading.get_symbol_precision", return_value=_mock_precision()):
            with patch("engine.trading.binance_request") as mock_req:
                mock_req.side_effect = [
                    leverage_response,
                    _make_api_response(200, {}),  # margin type
                    order_response,
                ]
                from engine.trading import open_position_async
                result = await open_position_async(mock_client, "BTCUSDT", "BUY", "RSI_OB", ai_brain)

        assert result is True
        assert "BTCUSDT" in fresh_state["limit_orders"]
        assert fresh_state["limit_orders"]["BTCUSDT"]["orderId"] == 67890

    @pytest.mark.asyncio
    async def test_open_order_api_failure(self, fresh_state, mock_client):
        ai_brain = {
            "limit_price": 50000.0,
            "sl": 1.0,
            "tp": 2.0,
            "is_market": True,
            "ml_prob": 0.7,
        }

        fail_response = _make_api_response(400, {"msg": "Insufficient margin", "code": -2019})
        leverage_response = _make_api_response(200, {"maxNotionalValue": "1000000"})

        with patch("engine.trading.get_symbol_precision", return_value=_mock_precision()):
            with patch("engine.trading.binance_request") as mock_req:
                mock_req.side_effect = [
                    leverage_response,
                    _make_api_response(200, {}),
                    fail_response,
                ]
                from engine.trading import open_position_async
                result = await open_position_async(mock_client, "BTCUSDT", "BUY", "TEST", ai_brain)

        assert result is False


class TestClosePosition:
    @pytest.mark.asyncio
    async def test_close_position_success(self, fresh_state, mock_client):
        fresh_state["trades"]["BTCUSDT"] = {
            "side": "LONG", "peak": 50000, "entry_time": 0,
            "active_features": ["EMA"],
        }

        with patch("engine.trading.get_symbol_precision", return_value=_mock_precision()):
            with patch("engine.trading.binance_request") as mock_req:
                mock_req.return_value = _make_api_response(200, {"status": "FILLED"})
                from engine.trading import close_position_async
                await close_position_async(mock_client, "BTCUSDT", "LONG", 0.1, "TEST", pnl=1.5)

        assert "CLOSED" in fresh_state["last_log"]

    @pytest.mark.asyncio
    async def test_close_updates_ml_performance(self, fresh_state, mock_client):
        fresh_state["trades"]["ETHUSDT"] = {
            "side": "SHORT", "peak": 3000, "entry_time": 0,
            "active_features": [],
        }

        with patch("engine.trading.get_symbol_precision", return_value=_mock_precision()):
            with patch("engine.trading.binance_request") as mock_req:
                mock_req.return_value = _make_api_response(200, {"status": "FILLED"})
                with patch("engine.trading.ml_predictor") as mock_ml:
                    from engine.trading import close_position_async
                    await close_position_async(mock_client, "ETHUSDT", "SHORT", 1.0, "TP_HIT", pnl=2.0)
                    mock_ml.update_performance.assert_called_once_with("ETHUSDT", True)
