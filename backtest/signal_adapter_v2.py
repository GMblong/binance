"""Signal adapter v2 — fuses technical score + ml_v2 + microstructure.

Compared to `signal_adapter.py` (v1):
  - Uses ml_v2 regime-routed model instead of per-symbol LightGBM.
  - Adds microstructure signals: CVD divergence/absorption, funding skew,
    BTC beta-gap, liquidation cascade reversion.
  - Builds P_trade via logit-space fusion (fusion.fuse_probabilities)
    instead of ad-hoc score boosts.
  - EV gate remains the final go/no-go but now uses calibrated P_trade
    from the fusion, which makes the EV number meaningful.

All the v1 guarantees (hard MTF gate, structural SL, maker-limit default,
cost-aware RR, leverage cap via sizing) are preserved.
"""

from __future__ import annotations

from typing import Callable, Optional, Dict, Any

import numpy as np
import pandas as pd

from strategies.analyzer import MarketAnalyzer
from backtest.cost_model import CostModel
from backtest.engine import Order

from microstructure.cvd import cvd_divergence, cvd_absorption
from microstructure.funding import funding_crowding_signal, funding_skew
from microstructure.liquidations import liq_cascade_signal
from microstructure.lead_lag import btc_beta_gap_signal

from ml_v2.fusion import fuse_probabilities, flow_prob, score_to_prob


def make_signal_fn_v2(
    v2_predictor,
    btc_1m: Optional[pd.DataFrame] = None,
    funding_per_symbol: Optional[Dict[str, pd.Series]] = None,
    cross_section: Optional[pd.DataFrame] = None,
    risk_percent: float = 0.01,
    min_ev_pct: float = 0.10,
    max_leverage: int = 10,
    cost_model: Optional[CostModel] = None,
    fusion_weights: Optional[Dict[str, float]] = None,
    session_fn: Optional[Callable[[int], str]] = None,
) -> Callable[[Dict[str, Any]], Optional[Order]]:
    """Return a `signal_fn` compatible with EventDrivenBacktester."""

    cost = cost_model or CostModel()
    funding_per_symbol = funding_per_symbol or {}

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
        if len(sub_15m) < 30 or len(sub_1h) < 20 or len(sub_1m) < 60:
            return None

        price = float(bar["c"])

        # Directions
        ema9_15m = MarketAnalyzer.get_ema(sub_15m["c"], 9).iloc[-1]
        ema21_15m = MarketAnalyzer.get_ema(sub_15m["c"], 21).iloc[-1]
        dir_15 = 1 if ema9_15m > ema21_15m else -1

        ema20_1h = MarketAnalyzer.get_ema(sub_1h["c"], 20).iloc[-1]
        ema50_1h = (
            MarketAnalyzer.get_ema(sub_1h["c"], 50).iloc[-1]
            if len(sub_1h) >= 50 else ema20_1h
        )
        last_1h_close = sub_1h["c"].iloc[-1]
        if last_1h_close > ema20_1h and ema20_1h >= ema50_1h:
            dir_1h = 1
        elif last_1h_close < ema20_1h and ema20_1h <= ema50_1h:
            dir_1h = -1
        else:
            dir_1h = 0

        ema9_1m = MarketAnalyzer.get_ema(sub_1m["c"], 9).iloc[-1]
        ema21_1m = MarketAnalyzer.get_ema(sub_1m["c"], 21).iloc[-1]
        dir_1m = 1 if ema9_1m > ema21_1m else -1

        direction = dir_15
        regime = MarketAnalyzer.detect_regime(sub_15m)

        # Hard MTF gate (same as v1)
        if regime in ("TRENDING", "VOLATILE"):
            if dir_1h == 0 or dir_1h != direction or dir_1m != direction:
                return None
        else:
            if dir_1m != direction:
                return None

        # ---- Build signal probabilities ----
        # 1) Technical score (v1 analyzer).
        session = session_fn(int(bar["ot"])) if session_fn else "QUIET"
        tech_score = MarketAnalyzer.calculate_score(
            sub_1m, sub_15m, direction, 1.0, 0.0,
            regime=regime, session=session, lead_lag=0,
        )
        p_tech = score_to_prob(tech_score, direction)

        # 2) ml_v2 prediction (regime-routed, calibrated).
        btc_close_slice = None
        if btc_1m is not None:
            btc_close_slice = btc_1m["c"].iloc[: idx + 1]
        funding_series = funding_per_symbol.get(symbol)
        p_ml_up = 0.5
        if v2_predictor is not None and v2_predictor.is_ready():
            try:
                p_ml_up = v2_predictor.predict_prob(
                    symbol=symbol,
                    df_1m=sub_1m,
                    btc_close=btc_close_slice,
                    funding_series=funding_series,
                    cross_section=cross_section,
                )
            except Exception:
                p_ml_up = 0.5
        p_ml = p_ml_up if direction == 1 else (1.0 - p_ml_up)

        # 3) Flow prob from microstructure signals.
        cvd_div = cvd_divergence(sub_1m)
        cvd_abs = cvd_absorption(sub_1m)
        funding_rate = None
        oi_roc_pct = None
        price_range_pct = None
        try:
            rng_15 = (sub_1m["h"].tail(15).max() - sub_1m["l"].tail(15).min()) / price * 100
            price_range_pct = float(rng_15)
        except Exception:
            price_range_pct = None
        if funding_series is not None and len(funding_series) > 0:
            try:
                funding_rate = float(funding_series.iloc[-1])
            except Exception:
                funding_rate = None
        if "oi" in sub_1m.columns and len(sub_1m) > 20:
            try:
                oi_roc_pct = float(
                    (sub_1m["oi"].iloc[-1] / sub_1m["oi"].iloc[-15] - 1.0) * 100
                )
            except Exception:
                oi_roc_pct = None
        funding_sig = funding_crowding_signal(
            funding_rate, oi_roc_pct, price_range_pct, direction
        )
        liq_sig = liq_cascade_signal(sub_1m, direction)
        beta_sig = 0
        if btc_close_slice is not None and len(btc_close_slice) >= 150:
            try:
                beta_sig = btc_beta_gap_signal(sub_1m["c"], btc_close_slice, direction)
            except Exception:
                beta_sig = 0
        # Use the v1 orderbook imbalance proxy from mid-range position.
        try:
            last_row = sub_1m.iloc[-1]
            rng_last = last_row["h"] - last_row["l"]
            ob_imb = 0.0
            if rng_last > 0:
                pos = (last_row["c"] - last_row["l"]) / rng_last  # 0..1
                ob_imb = (pos - 0.5) * 2.0  # -1..1
        except Exception:
            ob_imb = 0.0

        p_flow = flow_prob(
            ob_imbalance=ob_imb,
            cvd_div=cvd_div,
            cvd_abs=cvd_abs,
            funding_sig=funding_sig,
            liq_sig=liq_sig,
            beta_gap=beta_sig,
            direction=direction,
        )

        # 4) Fuse.
        p_trade, _fused_logit = fuse_probabilities(
            p_tech, p_ml, p_flow, weights=fusion_weights
        )

        # ---- Structural SL / entry / RR ----
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

        # RR stretches with calibrated confidence.
        if p_trade >= 0.70:
            rr_mult = 2.5
        elif p_trade >= 0.60:
            rr_mult = 2.0
        else:
            rr_mult = 1.6
        if regime == "RANGING":
            rr_mult = min(rr_mult, 1.5)
        tp_pct = sl_pct * rr_mult

        # ---- EV gate ----
        cost_roundtrip_pct = (
            cost.fee_maker + cost.fee_taker + cost.slippage_floor_pct
            + cost.slippage_atr_coef * (atr_pct / 100.0)
        ) * 100
        ev_pct = p_trade * tp_pct - (1 - p_trade) * sl_pct - cost_roundtrip_pct
        if ev_pct < min_ev_pct:
            return None

        # Entry kind: be very conservative with market orders in v2.
        prev_15_high = sub_1m["h"].iloc[-16:-1].max()
        prev_15_low = sub_1m["l"].iloc[-16:-1].min()
        is_breakout = (direction == 1 and price > prev_15_high) or (
            direction == -1 and price < prev_15_low
        )
        is_market = False
        if is_breakout and p_trade >= 0.72:
            market_cost_pct = (
                2 * cost.fee_taker + 2 * cost.slippage_floor_pct
                + 2 * cost.slippage_atr_coef * (atr_pct / 100.0)
            ) * 100
            if (p_trade * tp_pct - (1 - p_trade) * sl_pct - market_cost_pct) >= min_ev_pct:
                is_market = True

        if not is_market:
            dist = abs(price - limit_p) / price * 100
            if dist > 0.6:
                return None

        # ---- Sizing ----
        risk_usd = balance * risk_percent
        size_usd = risk_usd / (sl_pct / 100)
        max_notional = balance * max_leverage
        size_usd = min(size_usd, max_notional)
        if size_usd <= 0:
            return None

        side = "BUY" if direction == 1 else "SELL"
        return Order(
            symbol=symbol,
            side=side,
            kind="MARKET" if is_market else "LIMIT",
            price=price if is_market else float(limit_p),
            size_usd=size_usd,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            ts_act_pct=sl_pct * 0.8,
            ts_cb_pct=max(sl_pct * 0.5, 0.25),
            tick_size=d.get("tick", price * 1e-4),
            atr_pct_at_signal=atr_pct,
            created_bar=step,
            ttl_bars=15,
            meta={
                "score": int(tech_score),
                "p_tech": p_tech,
                "p_ml": p_ml,
                "p_flow": p_flow,
                "p_trade": p_trade,
                "ev_pct": ev_pct,
                "regime": regime,
                "cvd_div": cvd_div,
                "cvd_abs": cvd_abs,
                "funding_sig": funding_sig,
                "liq_sig": liq_sig,
                "beta_sig": beta_sig,
            },
        )

    return _signal_fn
