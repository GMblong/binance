"""Event-driven backtester.

The loop iterates once per 1m bar globally (not per-symbol), so all cross-
symbol interactions (portfolio exposure, correlated exits, max concurrent
positions) are accurate.

The backtester is intentionally strategy-agnostic: pass in a `signal_fn`
that turns per-bar context into optional orders, and `exit_fn` that can
override the default SL/TP/trailing exits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any

import pandas as pd

from .cost_model import CostModel, FillModel, is_funding_timestamp


@dataclass
class Order:
    symbol: str
    side: str  # "BUY" / "SELL"
    kind: str  # "MARKET" / "LIMIT"
    price: float  # reference or limit price
    size_usd: float
    sl_pct: float
    tp_pct: float
    ts_act_pct: float
    ts_cb_pct: float
    tick_size: float
    atr_pct_at_signal: float
    created_bar: int
    ttl_bars: int = 15  # cancel after this many bars unfilled
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    side: str  # "LONG" / "SHORT"
    entry: float
    size_usd: float
    qty: float
    sl: float
    tp: float
    ts_act_pct: float
    ts_cb_pct: float
    atr_pct_at_entry: float
    peak: float
    entry_bar: int
    be_moved: bool = False
    entry_fee_usd: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    symbol: str
    side: str
    entry: float
    exit: float
    size_usd: float
    pnl: float
    fees: float
    hold_bars: int
    reason: str


class EventDrivenBacktester:
    """Global 1m-bar event loop over N symbols.

    Required inputs:
      - `data`: dict[symbol -> dict] with keys `"1m"`, `"15m"`, `"1h"` (pandas
         DataFrames) and optionally `"funding"` (Series indexed by ot).
      - `signal_fn(context) -> Optional[Order]`: called once per sample_bar
         per non-active symbol.
      - Optional `exit_override(pos, context) -> Optional[str]`: return a
         reason string to force-close a position at the current close.
    """

    def __init__(
        self,
        data: Dict[str, Dict[str, Any]],
        signal_fn: Callable[[Dict[str, Any]], Optional[Order]],
        starting_balance: float = 100.0,
        cost_model: Optional[CostModel] = None,
        fill_model: Optional[FillModel] = None,
        max_positions: int = 3,
        sim_bars: int = 1440,
        signal_every_n_bars: int = 5,
        exit_override: Optional[Callable[[Position, Dict[str, Any]], Optional[str]]] = None,
        on_trade_close: Optional[Callable[[Trade, Dict[str, Any]], None]] = None,
    ) -> None:
        self.data = data
        self.signal_fn = signal_fn
        self.balance = starting_balance
        self.start_balance = starting_balance
        self.cost = cost_model or CostModel()
        self.fill = fill_model or FillModel()
        self.max_positions = max_positions
        self.sim_bars = sim_bars
        self.signal_every_n_bars = max(1, signal_every_n_bars)
        self.exit_override = exit_override
        # Optional hook: called once per trade as soon as the position
        # closes. Receives (Trade, position.meta). Used by v3 runners to
        # feed the OnlineLearner its per-trade labels.
        self.on_trade_close = on_trade_close

        self.pending: List[Order] = []
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []

        # Rate-stat trackers.
        self.signals_generated = 0
        self.limit_orders_placed = 0
        self.limit_orders_filled = 0
        self.limit_orders_expired = 0

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        # Pre-compute useful series per symbol to avoid recomputing each bar.
        for sym, d in self.data.items():
            df1m = d["1m"]
            if "atr" not in d:
                tr = pd.concat(
                    [
                        df1m["h"] - df1m["l"],
                        (df1m["h"] - df1m["c"].shift(1)).abs(),
                        (df1m["l"] - df1m["c"].shift(1)).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                d["atr"] = tr.rolling(14).mean()

        for step in range(self.sim_bars):
            for sym, d in self.data.items():
                df1m = d["1m"]
                idx = len(df1m) - self.sim_bars + step
                if idx < 1:
                    continue
                bar = df1m.iloc[idx]
                atr = d["atr"].iloc[idx]
                curr_price = float(bar["c"])
                atr_pct = (
                    float(atr) / curr_price * 100 if pd.notna(atr) and curr_price > 0 else 0.5
                )

                # 1. Apply funding if the bar OPEN falls on a funding boundary.
                ts_ms = int(bar["ot"])
                if is_funding_timestamp(ts_ms):
                    self._apply_funding(sym, d, ts_ms)

                # 2. Try to fill pending limit orders.
                self._process_pending(sym, bar, atr_pct, step)

                # 3. Handle exits for open positions in this symbol.
                self._process_exits(sym, d, bar, atr_pct, step)

                # 4. Generate new signals.
                if (
                    step % self.signal_every_n_bars == 0
                    and len(self.positions) < self.max_positions
                    and not any(p.symbol == sym for p in self.positions)
                    and not any(o.symbol == sym for o in self.pending)
                ):
                    ctx = {
                        "symbol": sym,
                        "data": d,
                        "bar_idx": idx,
                        "step": step,
                        "bar": bar,
                        "atr_pct": atr_pct,
                        "balance": self.balance,
                    }
                    order = self.signal_fn(ctx)
                    if order is not None:
                        self.signals_generated += 1
                        if order.kind == "MARKET":
                            self._execute_market(sym, d, order, bar, atr_pct, step)
                        else:
                            self.pending.append(order)
                            self.limit_orders_placed += 1

            self.equity_curve.append(self.balance)

        # Close any still-open positions at the last bar.
        self._force_close_all()

        from .metrics import compute_metrics

        metrics = compute_metrics(
            [t.__dict__ for t in self.trades], self.equity_curve
        )
        metrics["fill_rate"] = (
            self.limit_orders_filled / self.limit_orders_placed
            if self.limit_orders_placed > 0
            else 0.0
        )
        metrics["signals"] = self.signals_generated
        metrics["limit_placed"] = self.limit_orders_placed
        metrics["limit_filled"] = self.limit_orders_filled
        metrics["limit_expired"] = self.limit_orders_expired
        metrics["final_balance"] = self.balance
        metrics["start_balance"] = self.start_balance
        return metrics

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _apply_funding(self, sym: str, d: Dict[str, Any], ts_ms: int) -> None:
        funding_series = d.get("funding")
        rate = self.cost.funding_per_8h_default
        if funding_series is not None:
            # Look up the nearest funding rate at or before ts_ms.
            try:
                rate = float(funding_series.asof(ts_ms))
            except Exception:
                pass
        for pos in list(self.positions):
            if pos.symbol != sym:
                continue
            notional = pos.size_usd
            # Long pays funding when rate > 0, short receives. Short pays when
            # rate < 0.
            sign = 1 if pos.side == "LONG" else -1
            cost = notional * rate * sign
            self.balance -= cost

    def _process_pending(
        self, sym: str, bar: pd.Series, atr_pct: float, step: int
    ) -> None:
        high = float(bar["h"])
        low = float(bar["l"])
        for order in list(self.pending):
            if order.symbol != sym:
                continue
            filled = False
            fill_price = order.price
            if order.side == "BUY":
                if self.fill.fills_long_limit(low, order.price, order.tick_size):
                    filled = True
                    fill_price = min(order.price, high)
            else:
                if self.fill.fills_short_limit(high, order.price, order.tick_size):
                    filled = True
                    fill_price = max(order.price, low)

            if filled:
                self._open_position_from_order(order, fill_price, atr_pct, step, maker=True)
                self.pending.remove(order)
                self.limit_orders_filled += 1
            elif step - order.created_bar >= order.ttl_bars:
                self.pending.remove(order)
                self.limit_orders_expired += 1

    def _execute_market(
        self,
        sym: str,
        d: Dict[str, Any],
        order: Order,
        bar: pd.Series,
        atr_pct: float,
        step: int,
    ) -> None:
        # Market executes at close plus slippage in the adverse direction.
        close = float(bar["c"])
        slip_dec = (
            self.cost.slippage_floor_pct + self.cost.slippage_atr_coef * (atr_pct / 100.0)
        )
        if order.side == "BUY":
            fill_price = close * (1 + slip_dec)
        else:
            fill_price = close * (1 - slip_dec)
        self._open_position_from_order(order, fill_price, atr_pct, step, maker=False)

    def _open_position_from_order(
        self,
        order: Order,
        fill_price: float,
        atr_pct: float,
        step: int,
        maker: bool,
    ) -> None:
        if len(self.positions) >= self.max_positions:
            return
        if fill_price <= 0:
            return
        qty = order.size_usd / fill_price
        if maker:
            entry_fee = self.cost.limit_entry_cost(order.size_usd)
        else:
            entry_fee = self.cost.market_entry_cost(order.size_usd, atr_pct)
        self.balance -= entry_fee

        side = "LONG" if order.side == "BUY" else "SHORT"
        if side == "LONG":
            sl = fill_price * (1 - order.sl_pct / 100)
            tp = fill_price * (1 + order.tp_pct / 100)
        else:
            sl = fill_price * (1 + order.sl_pct / 100)
            tp = fill_price * (1 - order.tp_pct / 100)
        self.positions.append(
            Position(
                symbol=order.symbol,
                side=side,
                entry=fill_price,
                size_usd=order.size_usd,
                qty=qty,
                sl=sl,
                tp=tp,
                ts_act_pct=order.ts_act_pct,
                ts_cb_pct=order.ts_cb_pct,
                atr_pct_at_entry=order.atr_pct_at_signal,
                peak=fill_price,
                entry_bar=step,
                entry_fee_usd=entry_fee,
                meta=dict(order.meta),
            )
        )

    def _process_exits(
        self,
        sym: str,
        d: Dict[str, Any],
        bar: pd.Series,
        atr_pct: float,
        step: int,
    ) -> None:
        high = float(bar["h"])
        low = float(bar["l"])
        close = float(bar["c"])

        for pos in list(self.positions):
            if pos.symbol != sym:
                continue

            exit_price = None
            reason = ""

            if pos.side == "LONG":
                pos.peak = max(pos.peak, high)
                if self.fill.fills_long_stop(low, pos.sl):
                    exit_price = pos.sl
                    reason = "SL"
                elif high >= pos.tp:
                    exit_price = pos.tp
                    reason = "TP"
                else:
                    peak_pnl = (pos.peak - pos.entry) / pos.entry * 100
                    curr_pnl = (close - pos.entry) / pos.entry * 100
                    # Break-even move: after 1R in profit, nudge SL to BE + 2x
                    # fee buffer so fee+slippage doesn't turn the winner into a
                    # loser.
                    if not pos.be_moved and peak_pnl >= pos.ts_act_pct:
                        be_buffer_pct = 2 * (self.cost.fee_taker * 100)
                        pos.sl = max(pos.sl, pos.entry * (1 + be_buffer_pct / 100))
                        pos.be_moved = True
                    if peak_pnl > pos.ts_act_pct and (peak_pnl - curr_pnl) >= pos.ts_cb_pct:
                        exit_price = close
                        reason = "TRAIL"
            else:
                pos.peak = min(pos.peak, low)
                if self.fill.fills_short_stop(high, pos.sl):
                    exit_price = pos.sl
                    reason = "SL"
                elif low <= pos.tp:
                    exit_price = pos.tp
                    reason = "TP"
                else:
                    peak_pnl = (pos.entry - pos.peak) / pos.entry * 100
                    curr_pnl = (pos.entry - close) / pos.entry * 100
                    if not pos.be_moved and peak_pnl >= pos.ts_act_pct:
                        be_buffer_pct = 2 * (self.cost.fee_taker * 100)
                        pos.sl = min(pos.sl, pos.entry * (1 - be_buffer_pct / 100))
                        pos.be_moved = True
                    if peak_pnl > pos.ts_act_pct and (peak_pnl - curr_pnl) >= pos.ts_cb_pct:
                        exit_price = close
                        reason = "TRAIL"

            if exit_price is None and self.exit_override is not None:
                ctx = {
                    "bar": bar,
                    "close": close,
                    "step": step,
                    "data": d,
                }
                forced = self.exit_override(pos, ctx)
                if forced:
                    exit_price = close
                    reason = forced

            if exit_price is not None:
                self._close_position(pos, exit_price, atr_pct, step, reason)

    def _close_position(
        self,
        pos: Position,
        exit_price: float,
        atr_pct: float,
        step: int,
        reason: str,
    ) -> None:
        # Exit routing by reason:
        #   - TP: can be a resting maker limit (reduce-only GTC on Binance).
        #         When `CostModel.maker_tp=True`, charge maker fee + 0 slip.
        #   - SL: always stop_market -> taker fee + slippage.
        #   - TRAIL/EOD/custom: taker fee + slippage (dynamic close).
        if reason == "TP" and getattr(self.cost, "maker_tp", False):
            exit_fee = self.cost.limit_exit_cost(pos.size_usd)
        elif reason == "SL":
            exit_fee = self.cost.stop_exit_cost(pos.size_usd, atr_pct)
        else:
            exit_fee = self.cost.market_exit_cost(pos.size_usd, atr_pct)
        gross = (
            (exit_price - pos.entry) / pos.entry
            if pos.side == "LONG"
            else (pos.entry - exit_price) / pos.entry
        )
        gross_usd = gross * pos.size_usd
        net_usd = gross_usd - exit_fee
        self.balance += net_usd
        fees_total = pos.entry_fee_usd + exit_fee
        self.trades.append(
            Trade(
                symbol=pos.symbol,
                side=pos.side,
                entry=pos.entry,
                exit=exit_price,
                size_usd=pos.size_usd,
                pnl=net_usd - pos.entry_fee_usd,
                fees=fees_total,
                hold_bars=step - pos.entry_bar,
                reason=reason,
            )
        )
        # Fire the close hook BEFORE removing the position so the
        # callback can still see pos.meta (e.g. the sig_id that links
        # back to the OnlineLearner's pending sample).
        if self.on_trade_close is not None:
            try:
                self.on_trade_close(self.trades[-1], dict(pos.meta))
            except Exception:
                # Never let a hook failure kill the sim loop.
                pass
        self.positions.remove(pos)

    def _force_close_all(self) -> None:
        for pos in list(self.positions):
            d = self.data[pos.symbol]
            last = d["1m"].iloc[-1]
            close = float(last["c"])
            atr = d["atr"].iloc[-1]
            atr_pct = (
                float(atr) / close * 100
                if pd.notna(atr) and close > 0
                else 0.5
            )
            self._close_position(pos, close, atr_pct, self.sim_bars, "EOD")
