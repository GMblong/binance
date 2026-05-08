"""Honest backtesting utilities for the hybrid scalper.

Key design goals (vs the legacy backtest_*.py scripts):

- Realistic cost model: taker/maker fees, slippage scaled to ATR%, funding
  rate settlement every 8h.
- Realistic fill model for limit orders: a limit order only fills when price
  TRADES THROUGH the level (+1 tick), not merely touches it.
- No look-ahead: all ML training is done with an explicit `end_time` that is
  strictly earlier than the first simulated bar.
- Walk-forward splits so ML models never see data from their own test window.
- Proper metrics: Sharpe, Sortino, Max Drawdown, Profit Factor, Expectancy,
  Win Rate, Avg Hold Time, % Limit Fill Rate.

The module is intentionally self-contained so it can be imported from either
backtest_today.py or backtest_yesterday.py.
"""

from .cost_model import CostModel, FillModel
from .metrics import compute_metrics, print_report
from .engine import EventDrivenBacktester, Order, Position, Trade

__all__ = [
    "CostModel",
    "FillModel",
    "EventDrivenBacktester",
    "Order",
    "Position",
    "Trade",
    "compute_metrics",
    "print_report",
]
