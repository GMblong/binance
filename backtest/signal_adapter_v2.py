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

TF-aware: `mtf_ratio` and `htf_ratio` let the adapter operate on any base
timeframe. For example, on 5m bars: mtf_ratio=3 (15m MTF) and htf_ratio=12
(1h HTF). On 1m bars: mtf_ratio=15 (15m MTF) and htf_ratio=60 (1h HTF).

Structural lookbacks (recent_low_N, prev breakout window, FVG/OB search,
tail_ttl for pending orders) scale down on higher TFs since 15 bars of 5m
is already 75 minutes.
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
    mtf_ratio: int = 15,
    htf_ratio: int = 60,
    struct_short: int = 15,
    struct_long: int = 30,
    breakout_window: int = 16,
    trail_ttl_bars: int = 15,
    min_1m_warmup: int = 60,
) -> Callable[[Dict[str, Any]], Optional[Order]]:
    """Return a `signal_fn` compatible with EventDrivenBacktester.

    mtf_ratio / htf_ratio: ratio of mid/high TF to base TF (e.g. for 5m
    base: 3 for 15m MTF, 12 for 1h HTF).
    struct_short / struct_long: windows for recent swing-high/low.
    breakout_window: bars to look back for prior extreme in breakout test.
    trail_ttl_bars: how many bars a pending limit stays live.
    min_1m_warmup: minimum bars needed on the base TF (technically not
    always 1m anymore — name kept for continuity).
    """

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

        df_base = d["1m"]
        df_mtf = d["15m"]
        df_htf = d["1h"]

        mtf_idx = min(len(df_mtf) - 1, idx // mtf_ratio)
        htf_idx = min(len(df_htf) - 1, idx // htf_ratio)
        sub_base = df_base.iloc[: idx + 1]
        sub_mtf = df_mtf.iloc[: mtf_idx + 1]
        sub_htf = df_htf.iloc[: htf_idx + 1]
        if (
            len(sub_mtf) < 30
            or len(sub_htf) < 20
            or len(sub_base) < min_1m_warmup
        ):
            return None

        price = float(bar["c"])

        # Directions
        ema9_mtf = MarketAnalyzer.get_ema(sub_mtf["c"], 9).iloc[-1]
        ema21_mtf = MarketAnalyzer.get_ema(sub_mtf["c"], 21).iloc[-1]
        dir_mtf = 1 if ema9_mtf > ema21_mtf else -1

        ema20_htf = MarketAnalyzer.get_ema(sub_htf["c"], 20).iloc[-1]
        ema50_htf = (
            MarketAnalyzer.get_ema(sub_htf["c"], 50).iloc[-1]
            if len(sub_htf) >= 50
            else ema20_htf
        )
        last_htf_close = sub_htf["c"].iloc[-1]
        if last_htf_close > ema20_htf and ema20_htf >= ema50_htf:
            dir_htf = 1
        elif last_htf_close < ema20_htf and ema20_htf <= ema50_htf:
            dir_htf = -1
        else:
            dir_htf = 0

        ema9_base = MarketAnalyzer.get_ema(sub_base["c"], 9).iloc[-1]
        ema21_base = MarketAnalyzer.get_ema(sub_base["c"], 21).iloc[-1]
        dir_base = 1 if ema9_base > ema21_base else -1

        direction = dir_mtf
        regime = MarketAnalyzer.detect_regime(sub_mtf)

        # Hard MTF gate (same as v1)
        if regime in ("TRENDING", "VOLATILE"):
            if dir_htf == 0 or dir_htf != direction or dir_base != direction:
                return None
        else:
            if dir_base != direction:
                return None

        # ---- Build signal probabilities ----
        # 1) Technical score (v1 analyzer).
        session = session_fn(int(bar["ot"])) if session_fn else "QUIET"
        tech_score = MarketAnalyzer.calculate_score(
            sub_base,
            sub_mtf,
            direction,
            1.0,
            0.0,
            regime=regime,
            session=session,
            lead_lag=0,
        )
        p_tech = score_to_prob(tech_score, direction)

        # 2) ml_v2 prediction (regime-routed, calibrated).
        btc_close_slice = None
        if btc_1m is not None:
            btc_close_slice = btc_1m["c"].iloc[: idx + 1]
        funding_series = funding_per_symbol.get(symbol)
        p_ml_up = 0.5
        ml_regime_auc = 0.5
        if v2_predictor is not None and v2_predictor.is_ready():
            try:
                p_ml_up = v2_predictor.predict_prob(
                    symbol=symbol,
                    df_1m=sub_base,
                    btc_close=btc_close_slice,
                    funding_series=funding_series,
                    cross_section=cross_section,
                )
                # Per-regime AUC: if this regime's model is < 0.51 AUC,
                # treat its output as random (P_ml = 0.5). This prevents
                # a weak-regime model from dragging a good tech+flow
                # signal into a bad trade.
                ml_regime_auc = v2_predictor.last_regime_auc(regime)
                if ml_regime_auc < 0.51:
                    p_ml_up = 0.5
            except Exception:
                p_ml_up = 0.5
        p_ml = p_ml_up if direction == 1 else (1.0 - p_ml_up)

        # 3) Flow prob from microstructure signals.
        cvd_div = cvd_divergence(sub_base)
        cvd_abs = cvd_absorption(sub_base)
        funding_rate = None
        oi_roc_pct = None
        price_range_pct = None
        try:
            rng_n = min(struct_short, len(sub_base))
            rng_15 = (
                sub_base["h"].tail(rng_n).max() - sub_base["l"].tail(rng_n).min()
            ) / price * 100
            price_range_pct = float(rng_15)
        except Exception:
            price_range_pct = None
        if funding_series is not None and len(funding_series) > 0:
            try:
                funding_rate = float(funding_series.iloc[-1])
            except Exception:
                funding_rate = None
        if "oi" in sub_base.columns and len(sub_base) > 20:
            try:
                oi_roc_pct = float(
                    (sub_base["oi"].iloc[-1] / sub_base["oi"].iloc[-15] - 1.0)
                    * 100
                )
            except Exception:
                oi_roc_pct = None
        funding_sig = funding_crowding_signal(
            funding_rate, oi_roc_pct, price_range_pct, direction
        )
        liq_sig = liq_cascade_signal(sub_base, direction)
        beta_sig = 0
        if btc_close_slice is not None and len(btc_close_slice) >= 150:
            try:
                beta_sig = btc_beta_gap_signal(
                    sub_base["c"], btc_close_slice, direction
                )
            except Exception:
                beta_sig = 0
        # Use the v1 orderbook imbalance proxy from mid-range position.
        try:
            last_row = sub_base.iloc[-1]
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
        fvg = MarketAnalyzer.get_nearest_fvg(sub_base)
        ob = MarketAnalyzer.find_nearest_order_block(sub_base, price, direction)
        limit_p = ema9_base
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

        recent_low_s = sub_base["l"].tail(struct_short).min()
        recent_low_l = sub_base["l"].tail(struct_long).min()
        recent_high_s = sub_base["h"].tail(struct_short).max()
        recent_high_l = sub_base["h"].tail(struct_long).max()
        buffer_pct = max(atr_pct * 0.3, 0.15)
        if direction == 1:
            struct_low = (
                recent_low_l
                if ((price - recent_low_l) / price * 100) <= 3.0
                else recent_low_s
            )
            if ob and ob["type"] == "BULLISH":
                sl_price = ob["bottom"]
            else:
                sl_price = struct_low if struct_low else price * 0.99
            sl_pct = ((price - sl_price) / price * 100) + buffer_pct
        else:
            struct_high = (
                recent_high_l
                if ((recent_high_l - price) / price * 100) <= 3.0
                else recent_high_s
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
            cost.fee_maker
            + cost.fee_taker
            + cost.slippage_floor_pct
            + cost.slippage_atr_coef * (atr_pct / 100.0)
        ) * 100
        ev_pct = p_trade * tp_pct - (1 - p_trade) * sl_pct - cost_roundtrip_pct
        if ev_pct < min_ev_pct:
            return None

        # Entry kind: be very conservative with market orders in v2.
        prev_hi = sub_base["h"].iloc[-breakout_window:-1].max()
        prev_lo = sub_base["l"].iloc[-breakout_window:-1].min()
        is_breakout = (direction == 1 and price > prev_hi) or (
            direction == -1 and price < prev_lo
        )
        is_market = False
        if is_breakout and p_trade >= 0.72:
            market_cost_pct = (
                2 * cost.fee_taker
                + 2 * cost.slippage_floor_pct
                + 2 * cost.slippage_atr_coef * (atr_pct / 100.0)
            ) * 100
            if (
                p_trade * tp_pct - (1 - p_trade) * sl_pct - market_cost_pct
            ) >= min_ev_pct:
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
            ttl_bars=trail_ttl_bars,
            meta={
                "score": int(tech_score),
                "p_tech": p_tech,
                "p_ml": p_ml,
                "p_flow": p_flow,
                "p_trade": p_trade,
                "ev_pct": ev_pct,
                "regime": regime,
                "ml_regime_auc": ml_regime_auc,
                "cvd_div": cvd_div,
                "cvd_abs": cvd_abs,
                "funding_sig": funding_sig,
                "liq_sig": liq_sig,
                "beta_sig": beta_sig,
            },
        )

    return _signal_fn
