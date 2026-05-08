"""Realistic cost + fill model for Binance USDT-M futures.

Live Binance futures (as of 2025) taker is 0.04% and maker is 0.02%. Market
orders on alts also eat slippage; our model scales slippage with the
instantaneous ATR%. Funding is settled every 8h at 00:00, 08:00, 16:00 UTC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CostModel:
    """Per-leg cost in percentage of notional.

    fee_taker: 0.0004 = 0.04% per side (Binance default taker).
    fee_maker: 0.0002 = 0.02% per side (Binance default maker).
    slippage_floor_pct: minimum slippage applied even in calm markets.
    slippage_atr_coef: slippage scales with ATR%. At ATR% = 1.0, slippage
        becomes slippage_floor + slippage_atr_coef * 0.01.
    funding_per_8h_default: fallback funding rate in decimal (0.0001 = 0.01%).
        The engine can override per bar when historical funding is available.
    maker_tp: if True, the engine will treat TP exits as maker limit fills
        (fee_maker, no slippage) instead of taker market. This halves the
        round-trip cost of a profit-taking trade. Only safe when TP is
        placed as a resting LIMIT order (Binance futures supports this
        via the reduce-only flag + GTC limit). SL stays taker+slip always.
    """

    fee_taker: float = 0.0004
    fee_maker: float = 0.0002
    slippage_floor_pct: float = 0.0003  # 0.03%
    slippage_atr_coef: float = 0.30
    funding_per_8h_default: float = 0.0001
    maker_tp: bool = False

    def market_entry_cost(self, notional: float, atr_pct: float) -> float:
        """Dollar cost to enter with a MARKET order (fee + slippage)."""
        slip = self._slippage_pct(atr_pct)
        return notional * (self.fee_taker + slip)

    def market_exit_cost(self, notional: float, atr_pct: float) -> float:
        """Dollar cost to exit with a MARKET order (fee + slippage)."""
        slip = self._slippage_pct(atr_pct)
        return notional * (self.fee_taker + slip)

    def limit_entry_cost(self, notional: float) -> float:
        """Dollar cost to enter with a maker LIMIT order (fee only, no slip)."""
        return notional * self.fee_maker

    def limit_exit_cost(self, notional: float) -> float:
        """Dollar cost to exit with a maker LIMIT order (fee only, no slip).

        Only used by the engine for TP-as-maker fills when
        `CostModel.maker_tp=True`. SL exits always use stop_exit_cost.
        """
        return notional * self.fee_maker

    def stop_exit_cost(self, notional: float, atr_pct: float) -> float:
        """STOP_MARKET exit: taker fee + slippage (stops are always taker)."""
        slip = self._slippage_pct(atr_pct)
        return notional * (self.fee_taker + slip)

    def _slippage_pct(self, atr_pct: float) -> float:
        # atr_pct is in percent (e.g. 0.6 for 0.6%). Convert to decimal.
        atr_dec = max(0.0, atr_pct) / 100.0
        return self.slippage_floor_pct + self.slippage_atr_coef * atr_dec


@dataclass
class FillModel:
    """Realistic limit-order fill logic.

    A limit BUY at price P fills only if the market prints strictly below P
    (low < P - tick). Merely touching (low == P) is not enough: the queue
    usually did not clear. This is a known gotcha that makes most home-made
    backtesters look profitable when they are not.
    """

    require_penetration: bool = True
    penetration_ticks: int = 1

    def fills_long_limit(
        self, low: float, price: float, tick_size: float
    ) -> bool:
        if self.require_penetration:
            return low < price - self.penetration_ticks * tick_size
        return low <= price

    def fills_short_limit(
        self, high: float, price: float, tick_size: float
    ) -> bool:
        if self.require_penetration:
            return high > price + self.penetration_ticks * tick_size
        return high >= price

    def fills_long_stop(self, low: float, stop: float) -> bool:
        # A long stop-loss is a SELL stop, triggers when price <= stop.
        return low <= stop

    def fills_short_stop(self, high: float, stop: float) -> bool:
        return high >= stop


def is_funding_timestamp(ts_ms: int) -> bool:
    """True if this bar's open time falls on a Binance funding settlement.

    Binance perpetuals settle funding at 00:00, 08:00, 16:00 UTC. We detect
    this by checking the UTC hour and whether the minute is zero.
    """
    # ts_ms is epoch ms for the OPEN of the candle.
    seconds = ts_ms // 1000
    secs_in_day = seconds % 86400
    # funding windows: 0, 8h, 16h
    for target in (0, 8 * 3600, 16 * 3600):
        if abs(secs_in_day - target) < 60:
            return True
    return False
