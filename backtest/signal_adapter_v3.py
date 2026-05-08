"""Signal adapter v3 — v2 fusion + online learner + adaptive EV gate.

v3 is a last-ditch attempt to make 1m/5m scalping with live-learning
ML profitable. It keeps everything v2 got right (hard MTF gate, EV
gate, structural SL, TF-aware lookbacks, regime-routed LightGBM) and
adds THREE things:

1. Online learning: a SGDClassifier is updated from each closed trade.
   Its output P_online is fused alongside P_tech, P_ml, P_flow. When
   the frozen LightGBM drifts or the regime shifts, the online head
   adapts within 20-50 trades.

2. Maker-maker exit accounting: callers enable `CostModel.maker_tp=True`
   so that TP fills charge maker fee (0.02%) instead of taker (0.04%).
   This halves round-trip cost for winning trades and is what actually
   shifts fee-dominated scalping into (marginally) positive EV.

3. Adaptive EV gate: the minimum EV threshold is scaled by the
   classifier's recent AUC. When the model has no edge (best AUC <
   0.52), `min_ev_pct` is bumped aggressively so only screaming
   tech+flow setups can trade. When AUC > 0.55, we ease up a bit.

Public entry point `make_signal_fn_v3` also returns an OnlineLearner
handle the engine uses to call record_signal() on each generated
order and record_outcome() on each closed trade.
"""

from __future__ import annotations

from typing import Callable, Optional, Dict, Any, Tuple

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
from ml_v2.features import FEATURE_COLUMNS, build_feature_matrix
from ml_v2.online import OnlineLearner


def _scale_min_ev(base_min_ev: float, best_auc: float) -> float:
    """Adaptive EV gate: demand more edge when the model has less.

    AUC <= 0.50 -> 2.5x base min EV (very strict -- basically only
    tech+flow setups get through).
    AUC in [0.50, 0.55] -> linear 2.5x -> 1.0x.
    AUC >= 0.55 -> 0.8x (slightly easier so we don't starve).
    """
    if best_auc <= 0.50:
        return base_min_ev * 2.5
    if best_auc >= 0.55:
        return base_min_ev * 0.8
    # Linear interp in [0.50, 0.55]
    frac = (best_auc - 0.50) / 0.05
    return base_min_ev * (2.5 - 1.7 * frac)


def make_signal_fn_v3(
    v2_predictor,
    online_learner: Optional[OnlineLearner] = None,
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
    sym_id_map: Optional[Dict[str, int]] = None,
    adaptive_ev: bool = True,
    online_blend_weight: float = 0.3,
) -> Tuple[Callable[[Dict[str, Any]], Optional[Order]], OnlineLearner]:
    """Build a v3 signal_fn plus an `OnlineLearner` handle.

    The caller is expected to:
      - pass `cost_model=CostModel(maker_tp=True)` for maker-maker exits.
      - after each Order returned, note the signal_id from meta["sig_id"]
        and call `online_learner.record_signal(sig_id, feature_row,
        direction, step)` — the runner does this.
      - after each trade closes, call
        `online_learner.record_outcome(sig_id, 1_if_profitable_else_0)`.

    `adaptive_ev=True` scales min_ev_pct by recent classifier AUC so
    the gate tightens when the model degrades.
    `online_blend_weight` controls how loud the online head is in the
    final fusion. Default 0.3 is modest; can be raised once the live
    AUC stabilizes above 0.55.
    """

    cost = cost_model or CostModel()
    funding_per_symbol = funding_per_symbol or {}
    sym_id_map = sym_id_map or {}
    if online_learner is None:
        online_learner = OnlineLearner(feature_cols=list(FEATURE_COLUMNS))

    # Monotonic signal id counter -- engine needs it to reconcile
    # trades with learner state.
    _sig_counter = {"n": 0}

    def _next_sig_id(symbol: str, step: int) -> str:
        _sig_counter["n"] += 1
        return f"{symbol}@{step}#{_sig_counter['n']}"

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

        # Hard MTF gate (same as v2)
        if regime in ("TRENDING", "VOLATILE"):
            if dir_htf == 0 or dir_htf != direction or dir_base != direction:
                return None
        else:
            if dir_base != direction:
                return None

        # ---- Build signal probabilities ----
        # 1) Technical score.
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

        # 2) Build a unified feature row once -- used both for the frozen
        # LightGBM predict_prob path AND for the online learner.
        btc_close_slice = None
        if btc_1m is not None:
            btc_close_slice = btc_1m["c"].iloc[: idx + 1]
        funding_series = funding_per_symbol.get(symbol)
        try:
            feat_full = build_feature_matrix(
                sub_base.tail(400),
                btc_close=(
                    btc_close_slice.tail(400) if btc_close_slice is not None else None
                ),
                funding_series=funding_series,
                cross_section=cross_section,
                symbol=symbol,
                sym_id_map=sym_id_map,
            )
            feat_row = (
                feat_full.iloc[-1][FEATURE_COLUMNS].copy()
                if not feat_full.empty
                else None
            )
            # Fill NaN with column median to avoid garbage in the online model.
            if feat_row is not None and feat_row.isna().any():
                med = (
                    feat_full[FEATURE_COLUMNS].tail(50).median(numeric_only=True)
                )
                feat_row = feat_row.fillna(med).fillna(0.0)
        except Exception:
            feat_row = None

        # 3) Frozen LightGBM (ml_v2).
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
                ml_regime_auc = v2_predictor.last_regime_auc(regime)
                if ml_regime_auc < 0.51:
                    p_ml_up = 0.5
            except Exception:
                p_ml_up = 0.5
        p_ml = p_ml_up if direction == 1 else (1.0 - p_ml_up)

        # 4) Online learner.
        p_online_up = 0.5
        online_auc = 0.5
        if online_learner is not None and feat_row is not None:
            try:
                p_online_up = online_learner.predict_up(feat_row)
                online_auc = online_learner.recent_auc()
                # Same safety gate: if learner has no edge yet, neutralize.
                if online_auc < 0.51:
                    p_online_up = 0.5
            except Exception:
                p_online_up = 0.5
        p_online = p_online_up if direction == 1 else (1.0 - p_online_up)

        # 5) Flow prob from microstructure signals.
        cvd_div = cvd_divergence(sub_base)
        cvd_abs = cvd_absorption(sub_base)
        funding_rate = None
        oi_roc_pct = None
        price_range_pct = None
        try:
            rng_n = min(struct_short, len(sub_base))
            rng_15 = (
                sub_base["h"].tail(rng_n).max()
                - sub_base["l"].tail(rng_n).min()
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
        try:
            last_row = sub_base.iloc[-1]
            rng_last = last_row["h"] - last_row["l"]
            ob_imb = 0.0
            if rng_last > 0:
                pos = (last_row["c"] - last_row["l"]) / rng_last
                ob_imb = (pos - 0.5) * 2.0
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

        # 6) Fuse tech + ml + flow FIRST using the usual 3-way weights.
        p_base_trade, _ = fuse_probabilities(
            p_tech, p_ml, p_flow, weights=fusion_weights
        )
        # Now blend in the online head. Using a soft logit blend so the
        # online output acts as a pure nudge, weighted by its blend_weight
        # (0.3 by default; caller can lower for safety during warmup).
        from ml_v2.fusion import logit, sigmoid

        w_online = online_blend_weight
        if online_learner is not None and online_learner.is_ready():
            # Only let the online head participate once it has both an
            # AUC signal and a reasonable sample count.
            if online_auc < 0.51:
                w_online *= 0.0
        else:
            w_online = 0.0
        if w_online > 0:
            fused_logit = (
                (1.0 - w_online) * logit(p_base_trade)
                + w_online * logit(p_online)
            )
            p_trade = sigmoid(fused_logit)
        else:
            p_trade = p_base_trade

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

        # ---- EV gate (maker-aware + adaptive) ----
        # When maker_tp=True, the TP leg pays maker (fee_maker) instead of
        # taker+slippage. Entry is still maker (we route everything via
        # LIMIT below unless the breakout branch fires). Round-trip cost:
        #   maker-maker   = fee_maker + fee_maker                 (2 * 0.02% = 0.04%)
        #   maker-stop    = fee_maker + fee_taker + slip          (~0.09-0.15%)
        # Our EV estimate uses a weighted average assuming the trade wins
        # with probability `p_trade`: wins exit via maker TP, losses via
        # stop taker. This matches the real cost distribution.
        slip_pct = (
            cost.slippage_floor_pct
            + cost.slippage_atr_coef * (atr_pct / 100.0)
        ) * 100  # in %
        if getattr(cost, "maker_tp", False):
            cost_win_pct = (cost.fee_maker + cost.fee_maker) * 100
            cost_loss_pct = (cost.fee_maker + cost.fee_taker) * 100 + slip_pct
        else:
            cost_win_pct = (cost.fee_maker + cost.fee_taker) * 100 + slip_pct
            cost_loss_pct = (cost.fee_maker + cost.fee_taker) * 100 + slip_pct

        ev_pct = (
            p_trade * (tp_pct - cost_win_pct)
            - (1 - p_trade) * (sl_pct + cost_loss_pct)
        )

        # Adaptive EV threshold: when neither frozen ML nor online head
        # has any edge, demand a much bigger tech+flow setup to fire.
        best_auc = max(ml_regime_auc, online_auc)
        eff_min_ev = (
            _scale_min_ev(min_ev_pct, best_auc) if adaptive_ev else min_ev_pct
        )
        if ev_pct < eff_min_ev:
            return None

        # Entry kind: be very conservative with market orders in v3. If
        # maker_tp is on, avoid market entries entirely when we have any
        # ML signal at all -- we want the full maker-maker round trip.
        prev_hi = sub_base["h"].iloc[-breakout_window:-1].max()
        prev_lo = sub_base["l"].iloc[-breakout_window:-1].min()
        is_breakout = (direction == 1 and price > prev_hi) or (
            direction == -1 and price < prev_lo
        )
        is_market = False
        if is_breakout and p_trade >= 0.75 and not getattr(cost, "maker_tp", False):
            market_cost_pct = (
                2 * cost.fee_taker
                + 2 * cost.slippage_floor_pct
                + 2 * cost.slippage_atr_coef * (atr_pct / 100.0)
            ) * 100
            if (
                p_trade * tp_pct - (1 - p_trade) * sl_pct - market_cost_pct
            ) >= eff_min_ev:
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
        sig_id = _next_sig_id(symbol, step)
        order = Order(
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
                "sig_id": sig_id,
                "score": int(tech_score),
                "p_tech": p_tech,
                "p_ml": p_ml,
                "p_online": p_online,
                "p_flow": p_flow,
                "p_trade": p_trade,
                "ev_pct": ev_pct,
                "eff_min_ev": eff_min_ev,
                "regime": regime,
                "ml_regime_auc": ml_regime_auc,
                "online_auc": online_auc,
                "cvd_div": cvd_div,
                "cvd_abs": cvd_abs,
                "funding_sig": funding_sig,
                "liq_sig": liq_sig,
                "beta_sig": beta_sig,
            },
        )

        # Record the feature snapshot now so the learner can update
        # immediately when the position closes.
        if online_learner is not None and feat_row is not None:
            try:
                online_learner.record_signal(
                    sig_id, feat_row, direction, created_step=step
                )
            except Exception:
                pass
        return order

    return _signal_fn, online_learner
