"""Adapter that turns the existing hybrid-strategy heuristics into a
signal_fn compatible with backtest.engine.EventDrivenBacktester.

Design notes:
- EV gate (ev >= min_ev) is the deciding filter. A trade only fires if the
  probability-weighted payoff, AFTER expected round-trip costs, is net
  positive by at least `min_ev` percent of notional.
- Hard MTF gate: in TRENDING regime we require 1h trend == 15m trend == 1m
  direction, no exceptions. Soft mean-reversion only in RANGING.
- No MARKET entries for routine scalps. Only allowed when the signal is
  extremely strong AND price has already penetrated structure.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Dict, Any

import numpy as np
import pandas as pd

from strategies.analyzer import MarketAnalyzer
from .cost_model import CostModel
from .engine import Order


def make_signal_fn(
    ml_prob_fn: Optional[Callable[[str, pd.DataFrame, pd.DataFrame], float]] = None,
    risk_percent: float = 0.01,  # default 1% per trade (was 2%)
    min_ev_pct: float = 0.10,  # require >= 0.10% expected value after costs
    max_leverage: int = 10,  # cap leverage-implied sizing more conservatively
    cost_model: Optional[CostModel] = None,
    session_fn: Optional[Callable[[int], str]] = None,
    lead_lag_fn: Optional[Callable[[pd.DataFrame, pd.DataFrame], int]] = None,
    btc_1m: Optional[pd.DataFrame] = None,
) -> Callable[[Dict[str, Any]], Optional[Order]]:
    """Factory returning a signal_fn bound to ML + session + BTC context."""

    cost = cost_model or CostModel()

    def _signal_fn(ctx: Dict[str, Any]) -> Optional[Order]:
        symbol = ctx["symbol"]
        d = ctx["data"]
        idx = ctx["bar_idx"]
        step = ctx["step"]
        bar = ctx["bar"]
        atr_pct = ctx["atr_pct"]
        balance = ctx["balance"]

        df1m = d["1m"]
        df15m = d["15m"]
        df1h = d["1h"]

        d15_idx = min(len(df15m) - 1, idx // 15)
        d1h_idx = min(len(df1h) - 1, idx // 60)
        sub_1m = df1m.iloc[: idx + 1]
        sub_15m = df15m.iloc[: d15_idx + 1]
        sub_1h = df1h.iloc[: d1h_idx + 1]
        if len(sub_15m) < 30 or len(sub_1h) < 20 or len(sub_1m) < 50:
            return None

        price = float(bar["c"])

        # 1h/15m/1m trend direction.
        ema9_15m = MarketAnalyzer.get_ema(sub_15m["c"], 9).iloc[-1]
        ema21_15m = MarketAnalyzer.get_ema(sub_15m["c"], 21).iloc[-1]
        dir_15 = 1 if ema9_15m > ema21_15m else -1

        ema20_1h = MarketAnalyzer.get_ema(sub_1h["c"], 20).iloc[-1]
        ema50_1h = (
            MarketAnalyzer.get_ema(sub_1h["c"], 50).iloc[-1]
            if len(sub_1h) >= 50
            else ema20_1h
        )
        dir_1h = 1 if sub_1h["c"].iloc[-1] > ema20_1h and ema20_1h >= ema50_1h else (
            -1 if sub_1h["c"].iloc[-1] < ema20_1h and ema20_1h <= ema50_1h else 0
        )

        ema9_1m = MarketAnalyzer.get_ema(sub_1m["c"], 9).iloc[-1]
        ema21_1m = MarketAnalyzer.get_ema(sub_1m["c"], 21).iloc[-1]
        dir_1m = 1 if ema9_1m > ema21_1m else -1

        direction = dir_15
        regime = MarketAnalyzer.detect_regime(sub_15m)

        # --- Hard MTF gate ---------------------------------------------------
        # In TRENDING / VOLATILE, all three TFs must align. This single gate
        # is responsible for eliminating the bulk of the losing trades.
        if regime in ("TRENDING", "VOLATILE"):
            if dir_1h == 0 or dir_1h != direction or dir_1m != direction:
                return None
        else:
            # RANGING: allow soft mean reversion, but still require 1m to
            # confirm the entry side.
            if dir_1m != direction:
                return None

        # Session / lead-lag context.
        session = "QUIET"
        if session_fn is not None:
            session = session_fn(int(bar["ot"]))
        lead_lag = 0
        if lead_lag_fn is not None and btc_1m is not None:
            lead_lag = lead_lag_fn(sub_1m, btc_1m.iloc[: idx + 1])

        # Score and ML probability.
        score = MarketAnalyzer.calculate_score(
            sub_1m,
            sub_15m,
            direction,
            1.0,
            0.0,
            regime=regime,
            session=session,
            lead_lag=lead_lag,
        )

        ml_prob = 0.5
        if ml_prob_fn is not None:
            try:
                ml_prob = ml_prob_fn(symbol, sub_1m, sub_15m)
            except Exception:
                ml_prob = 0.5

        # Convert ML prob to directional probability for this trade.
        p_win_ml = ml_prob if direction == 1 else (1 - ml_prob)

        # ML nudge on score (kept modest -- EV gate handles real filtering).
        score_boost = 0
        if direction == 1 and ml_prob >= 0.60:
            score_boost = int((ml_prob - 0.55) * 40)
        elif direction == 1 and ml_prob < 0.40:
            score_boost = -int((0.45 - ml_prob) * 40)
        elif direction == -1 and ml_prob <= 0.40:
            score_boost = int((0.45 - ml_prob) * 40)
        elif direction == -1 and ml_prob > 0.60:
            score_boost = -int((ml_prob - 0.55) * 40)
        score = max(0, min(100, score + score_boost))

        # Regime-conditional score threshold (still used as a coarse filter).
        if regime == "TRENDING":
            score_min = 75
        elif regime == "VOLATILE":
            score_min = 78
        else:
            score_min = 82
        if score < score_min:
            return None

        # --- Entry price & structural SL/TP ---------------------------------
        atr = d["atr"].iloc[idx]
        if not pd.notna(atr) or atr <= 0:
            return None
        fvg = MarketAnalyzer.get_nearest_fvg(sub_1m)
        ob = MarketAnalyzer.find_nearest_order_block(sub_1m, price, direction)

        limit_p = ema9_1m
        if direction == 1:
            if fvg and fvg["type"] == "BULLISH":
                limit_p = fvg["top"]
            elif ob and ob["type"] == "BULLISH":
                limit_p = ob["top"]
        else:
            if fvg and fvg["type"] == "BEARISH":
                limit_p = fvg["bottom"]
            elif ob and ob["type"] == "BEARISH":
                limit_p = ob["bottom"]

        # Structural SL.
        recent_low_15 = sub_1m["l"].tail(15).min()
        recent_low_30 = sub_1m["l"].tail(30).min()
        recent_high_15 = sub_1m["h"].tail(15).max()
        recent_high_30 = sub_1m["h"].tail(30).max()
        buffer_pct = max(atr_pct * 0.3, 0.15)

        if direction == 1:
            struct_low = (
                recent_low_30
                if ((price - recent_low_30) / price * 100) <= 3.0
                else recent_low_15
            )
            if ob and ob["type"] == "BULLISH":
                sl_price = ob["bottom"]
            else:
                sl_price = struct_low if struct_low else price * 0.99
            sl_pct = ((price - sl_price) / price * 100) + buffer_pct
        else:
            struct_high = (
                recent_high_30
                if ((recent_high_30 - price) / price * 100) <= 3.0
                else recent_high_15
            )
            if ob and ob["type"] == "BEARISH":
                sl_price = ob["top"]
            else:
                sl_price = struct_high if struct_high else price * 1.01
            sl_pct = ((sl_price - price) / price * 100) + buffer_pct

        sl_pct = max(0.5, min(sl_pct, 3.0))

        # RR sizing: start at 1.8, stretch to 2.5 on high-confidence ML.
        rr_mult = 1.8
        if (direction == 1 and ml_prob >= 0.70) or (direction == -1 and ml_prob <= 0.30):
            rr_mult = 2.5
        if regime == "RANGING":
            rr_mult = min(rr_mult, 1.5)
        tp_pct = sl_pct * rr_mult

        # --- EV gate ---------------------------------------------------------
        # Cost per round trip (limit-in, market-out) in percent.
        cost_roundtrip_pct = (
            cost.fee_maker + cost.fee_taker + cost.slippage_floor_pct
            + cost.slippage_atr_coef * (atr_pct / 100.0)
        ) * 100

        # Use ML prob fused with score: score of 82+ corresponds to roughly
        # the top decile of historical signals, so we floor p at max(p_ml, 0.5 + score_edge).
        score_edge = (score - score_min) / 100.0 * 0.4  # max ~0.1 boost
        p_win = max(p_win_ml, 0.5 + score_edge)
        ev_pct = p_win * tp_pct - (1 - p_win) * sl_pct - cost_roundtrip_pct
        if ev_pct < min_ev_pct:
            return None

        # --- Sizing ----------------------------------------------------------
        # Risk X% of balance on SL. Cap leverage hard so a single whipsaw
        # cannot liquidate.
        risk_usd = balance * risk_percent
        size_usd = risk_usd / (sl_pct / 100)
        max_notional = balance * max_leverage
        size_usd = min(size_usd, max_notional)
        if size_usd <= 0:
            return None

        # --- Order kind ------------------------------------------------------
        # Breakout detection for market orders. We are VERY strict: only
        # allow MARKET if (a) 1m truly broke the prior 15 bars' extreme AND
        # (b) ML prob > 0.72 in our direction AND (c) EV remains positive
        # after paying taker+slip TWICE (we subtract one extra slip).
        prev_15_high = sub_1m["h"].iloc[-16:-1].max()
        prev_15_low = sub_1m["l"].iloc[-16:-1].min()
        is_breakout = (direction == 1 and price > prev_15_high) or (
            direction == -1 and price < prev_15_low
        )
        is_market = False
        if is_breakout and (
            (direction == 1 and ml_prob >= 0.72)
            or (direction == -1 and ml_prob <= 0.28)
        ):
            # Market-entry EV check with taker fee both sides.
            market_cost_pct = (
                cost.fee_taker * 2
                + cost.slippage_floor_pct * 2
                + cost.slippage_atr_coef * (atr_pct / 100.0) * 2
            ) * 100
            if (p_win * tp_pct - (1 - p_win) * sl_pct - market_cost_pct) >= min_ev_pct:
                is_market = True

        if not is_market:
            dist = abs(price - limit_p) / price * 100
            if dist > 0.6:
                return None  # Too far, skip rather than chase.
            # Final EV gate with maker+taker cost already considered above.

        side = "BUY" if direction == 1 else "SELL"
        order = Order(
            symbol=symbol,
            side=side,
            kind="MARKET" if is_market else "LIMIT",
            price=price if is_market else float(limit_p),
            size_usd=size_usd,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            ts_act_pct=sl_pct * 0.8,
            ts_cb_pct=max(sl_pct * 0.4, 0.25),
            tick_size=d.get("tick", price * 1e-4),
            atr_pct_at_signal=atr_pct,
            created_bar=step,
            ttl_bars=15,
            meta={
                "score": score,
                "ml_prob": ml_prob,
                "regime": regime,
                "ev_pct": ev_pct,
                "p_win": p_win,
            },
        )
        return order

    return _signal_fn
