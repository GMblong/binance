"""
Scalper Pro Strategy
--------------------
Pro-grade scalping signal engine focused on three pillars:

  1. Regime classification (TREND_UP / TREND_DOWN / RANGE / SQUEEZE / VOLATILE)
  2. Confluence across 1m / 5m / 15m / 1h (no redundant features)
  3. Execution plan with fixed-R structure and partial exits

Design goals:
- NO look-ahead: every indicator uses only the candles up to `now`.
- NO feature double-counting: each score contribution is orthogonal.
- Strict filter-first, score-later: bad setups are rejected early and cheaply.
- Output is deterministic and fully backtestable.

Why this should outperform the old hybrid scoring model:
- The old model piled ~15 overlapping features into a score and used a fixed
  threshold. Two-third of the "plus points" were just restating the same trend.
- Here we require SEPARATE pillars to agree. If trend, momentum, and
  microstructure all say the same thing, conviction is genuine.

Public entry point: `ScalperPro.generate_signal(ctx) -> Optional[Signal]`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import math

import numpy as np
import pandas as pd


# ---------- Indicator primitives (vectorised, pure) ----------

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["h"], df["l"], df["c"]
    tr = pd.concat([(h - l).abs(), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder's ADX. Values > 20 indicate a trending market."""
    h, l, c = df["h"], df["l"], df["c"]
    up = h.diff()
    down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(h - l).abs(), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / length, adjust=False).mean()


def bollinger(series: pd.Series, length: int = 20, num_std: float = 2.0):
    ma = series.rolling(length).mean()
    sd = series.rolling(length).std()
    return ma, ma + num_std * sd, ma - num_std * sd


def keltner(df: pd.DataFrame, length: int = 20, mult: float = 1.5):
    ma = df["c"].ewm(span=length, adjust=False).mean()
    a = atr(df, length)
    return ma, ma + mult * a, ma - mult * a


def vwap(df: pd.DataFrame, window: int = 120) -> pd.Series:
    """Rolling VWAP approximation. Use window ~2h of 1m candles."""
    tp = (df["h"] + df["l"] + df["c"]) / 3.0
    num = (tp * df["v"]).rolling(window).sum()
    den = df["v"].rolling(window).sum().replace(0, np.nan)
    return num / den


def wick_cvd(df: pd.DataFrame) -> pd.Series:
    """Cumulative volume delta proxied from candle wicks.

    When Binance `taker buy volume` (tbv) is available on the dataframe,
    prefer the exact buy-minus-sell computation; otherwise fall back to the
    wick-weighted estimate used by the old engine.
    """
    if "tbv" in df.columns:
        buy = df["tbv"].astype(float)
        sell = df["v"].astype(float) - buy
        return (buy - sell).cumsum()
    rng = (df["h"] - df["l"]).replace(0, np.nan)
    buy = df["v"] * (df["c"] - df["l"]) / rng
    sell = df["v"] * (df["h"] - df["c"]) / rng
    return (buy - sell).fillna(0).cumsum()


def z_score(series: pd.Series, length: int = 20) -> pd.Series:
    mu = series.rolling(length).mean()
    sd = series.rolling(length).std().replace(0, np.nan)
    return (series - mu) / sd


# ---------- Regime & context ----------

@dataclass
class Regime:
    name: str            # TREND_UP, TREND_DOWN, RANGE, SQUEEZE, VOLATILE
    strength: float      # 0..1, how clean the regime is
    atr_pct: float       # ATR/price as fraction
    adx: float


def classify_regime(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> Regime:
    """Regime from 15m + 1h. We use ADX for trend strength, BB/KC for squeeze."""
    if len(df_15m) < 50 or len(df_1h) < 30:
        return Regime("RANGE", 0.0, 0.0, 0.0)

    close = df_15m["c"]
    a = atr(df_15m, 14).iloc[-1]
    atr_p = float(a / close.iloc[-1])
    adx_v = float(adx(df_15m, 14).iloc[-1])

    mid, bb_up, bb_lo = bollinger(close, 20, 2.0)
    km, kc_up, kc_lo = keltner(df_15m, 20, 1.5)
    bb_width = float((bb_up.iloc[-1] - bb_lo.iloc[-1]) / close.iloc[-1])
    kc_width = float((kc_up.iloc[-1] - kc_lo.iloc[-1]) / close.iloc[-1])
    squeeze = bb_width < kc_width  # TTM-style squeeze

    ema50_1h = ema(df_1h["c"], 50).iloc[-1]
    ema20_1h = ema(df_1h["c"], 20).iloc[-1]
    htf_up = df_1h["c"].iloc[-1] > ema50_1h and ema20_1h > ema50_1h
    htf_dn = df_1h["c"].iloc[-1] < ema50_1h and ema20_1h < ema50_1h

    # High-vol: ATR > 1.5x recent median ATR AND price stretched from 20-EMA
    atr_med = float(atr(df_15m, 14).rolling(50).median().iloc[-1] or a)
    vol_ratio = a / max(atr_med, 1e-12)
    stretched = abs(close.iloc[-1] - ema(close, 20).iloc[-1]) / close.iloc[-1] > 1.5 * atr_p

    if vol_ratio > 1.8 and stretched:
        return Regime("VOLATILE", min(1.0, vol_ratio / 3.0), atr_p, adx_v)
    if squeeze and adx_v < 18:
        return Regime("SQUEEZE", max(0.0, 1.0 - bb_width / max(kc_width, 1e-9)), atr_p, adx_v)
    if adx_v > 22 and htf_up:
        return Regime("TREND_UP", min(1.0, adx_v / 40.0), atr_p, adx_v)
    if adx_v > 22 and htf_dn:
        return Regime("TREND_DOWN", min(1.0, adx_v / 40.0), atr_p, adx_v)
    return Regime("RANGE", 1.0 - min(adx_v / 30.0, 1.0), atr_p, adx_v)


@dataclass
class Features:
    """All indicator snapshots at the decision candle."""
    price: float
    atr_1m: float
    ema9_1m: float
    ema21_1m: float
    ema50_1m: float
    vwap_1m: float
    rsi_5m: float
    ema9_15m: float
    ema21_15m: float
    adx_15m: float
    bb_pctb: float        # %B: 0 = lower band, 1 = upper band
    vol_z: float          # 1m volume z-score
    cvd_slope: float      # short-term slope of cumulative volume delta
    body_ratio: float     # last candle body / range
    htf_bias: int         # +1 / -1 / 0 from 1h
    regime: Regime


def build_features(df_1m: pd.DataFrame, df_5m: pd.DataFrame,
                   df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> Optional[Features]:
    if any(d is None or len(d) < 60 for d in (df_1m, df_5m, df_15m)):
        return None
    if df_1h is None or len(df_1h) < 30:
        return None

    price = float(df_1m["c"].iloc[-1])
    a1 = float(atr(df_1m, 14).iloc[-1])
    e9 = float(ema(df_1m["c"], 9).iloc[-1])
    e21 = float(ema(df_1m["c"], 21).iloc[-1])
    e50 = float(ema(df_1m["c"], 50).iloc[-1])
    vw = float(vwap(df_1m, 120).iloc[-1])
    rsi5 = float(rsi(df_5m["c"], 14).iloc[-1])
    e9_15 = float(ema(df_15m["c"], 9).iloc[-1])
    e21_15 = float(ema(df_15m["c"], 21).iloc[-1])
    adx15 = float(adx(df_15m, 14).iloc[-1])

    mid, up, lo = bollinger(df_1m["c"], 20, 2.0)
    width = float((up.iloc[-1] - lo.iloc[-1])) or 1e-12
    pctb = float((df_1m["c"].iloc[-1] - lo.iloc[-1]) / width)

    vol_z_ = float(z_score(df_1m["v"], 20).iloc[-1] or 0)
    cvd = wick_cvd(df_1m.tail(120))
    cvd_slope = float(cvd.diff(5).iloc[-1] / max(df_1m["v"].tail(20).mean(), 1e-9))

    last = df_1m.iloc[-1]
    rng = max(float(last["h"] - last["l"]), 1e-9)
    body_ratio = float(abs(last["c"] - last["o"]) / rng)

    ema20_1h = float(ema(df_1h["c"], 20).iloc[-1])
    ema50_1h = float(ema(df_1h["c"], 50).iloc[-1])
    htf_bias = 1 if (df_1h["c"].iloc[-1] > ema50_1h and ema20_1h > ema50_1h) else \
               -1 if (df_1h["c"].iloc[-1] < ema50_1h and ema20_1h < ema50_1h) else 0

    reg = classify_regime(df_15m, df_1h)
    return Features(price, a1, e9, e21, e50, vw, rsi5, e9_15, e21_15, adx15,
                    pctb, vol_z_, cvd_slope, body_ratio, htf_bias, reg)


# ---------- Signal generation ----------

@dataclass
class Signal:
    side: str                    # LONG or SHORT
    entry_type: str              # MARKET or LIMIT
    entry: float
    sl: float
    tp1: float                   # partial exit at R=1
    tp2: float                   # runner target at R=tp2_r
    r_unit: float                # distance from entry to SL in price
    setup: str                   # TREND_PULLBACK, RANGE_FADE, SQUEEZE_BREAKOUT
    confidence: float            # 0..1
    reasons: Dict[str, Any]


class ScalperPro:
    """Deterministic signal engine. Stateless; feed it features."""

    # Risk/reward architecture. Tight R with partial exit produces much higher
    # expectancy than the old 2.08R target-only scheme because partials lock in
    # profit before the usual 1m reversal kills the trade.
    TP1_R = 1.0
    TP2_R = 2.0
    PARTIAL_PCT = 0.6  # % of position closed at TP1

    def _trend_pullback(self, f: Features) -> Optional[Signal]:
        """Trade WITH the 15m trend on a pullback to 1m EMA21/VWAP."""
        direction = 0
        if f.ema9_15m > f.ema21_15m and f.htf_bias >= 0 and f.regime.name in ("TREND_UP", "RANGE"):
            direction = 1
        elif f.ema9_15m < f.ema21_15m and f.htf_bias <= 0 and f.regime.name in ("TREND_DOWN", "RANGE"):
            direction = -1
        else:
            return None
        if f.regime.adx < 18:
            return None

        # Pullback condition: price between 1m EMA9 and EMA21 (or touching VWAP)
        touched = min(f.ema21_1m, f.vwap_1m) * 0.9990 <= f.price <= max(f.ema21_1m, f.vwap_1m) * 1.0010
        near_vwap = abs(f.price - f.vwap_1m) / f.price < 0.0015
        if not (touched or near_vwap):
            return None

        # Momentum confirmation: RSI not against us, CVD slope agrees
        if direction == 1 and (f.rsi_5m < 45 or f.cvd_slope < 0):
            return None
        if direction == -1 and (f.rsi_5m > 55 or f.cvd_slope > 0):
            return None

        r = 1.1 * f.atr_1m  # 1.1 ATR stop: tight but respects 1m noise
        if direction == 1:
            sl = f.price - r
            tp1 = f.price + r * self.TP1_R
            tp2 = f.price + r * self.TP2_R
        else:
            sl = f.price + r
            tp1 = f.price - r * self.TP1_R
            tp2 = f.price - r * self.TP2_R

        conf = min(1.0, 0.4 + f.regime.strength * 0.3 + min(f.regime.adx / 40, 1.0) * 0.3)
        return Signal(
            side="LONG" if direction == 1 else "SHORT",
            entry_type="MARKET", entry=f.price,
            sl=sl, tp1=tp1, tp2=tp2, r_unit=r,
            setup="TREND_PULLBACK", confidence=conf,
            reasons={"adx": f.regime.adx, "rsi5": f.rsi_5m, "cvd": f.cvd_slope},
        )

    def _range_fade(self, f: Features) -> Optional[Signal]:
        """Fade Bollinger extremes inside a range with RSI extreme + absorption."""
        if f.regime.name != "RANGE" or f.regime.adx > 20:
            return None
        # Need low volatility regime
        if f.regime.atr_pct > 0.008:
            return None

        direction = 0
        if f.bb_pctb < 0.05 and f.rsi_5m < 30 and f.cvd_slope > 0 and f.body_ratio < 0.6:
            direction = 1
        elif f.bb_pctb > 0.95 and f.rsi_5m > 70 and f.cvd_slope < 0 and f.body_ratio < 0.6:
            direction = -1
        else:
            return None

        # Mean-reversion targets: tighter stop, TP to VWAP/mid-range
        r = 0.8 * f.atr_1m
        if direction == 1:
            sl = f.price - r
            tp1 = f.price + r * self.TP1_R
            tp2 = f.vwap_1m if f.vwap_1m > f.price + r else f.price + r * self.TP2_R
        else:
            sl = f.price + r
            tp1 = f.price - r * self.TP1_R
            tp2 = f.vwap_1m if f.vwap_1m < f.price - r else f.price - r * self.TP2_R

        conf = 0.5 + (1 - min(f.regime.adx / 20, 1.0)) * 0.3
        return Signal(
            side="LONG" if direction == 1 else "SHORT",
            entry_type="MARKET", entry=f.price,
            sl=sl, tp1=tp1, tp2=tp2, r_unit=r,
            setup="RANGE_FADE", confidence=conf,
            reasons={"pctb": f.bb_pctb, "rsi5": f.rsi_5m, "body": f.body_ratio},
        )

    def _squeeze_breakout(self, f: Features) -> Optional[Signal]:
        """Enter momentum breakout after a Bollinger/Keltner squeeze with volume burst."""
        if f.regime.name != "SQUEEZE":
            return None
        if f.vol_z < 1.5:  # breakout needs a volume surge
            return None

        # Direction from 1m candle body + CVD slope
        last_body = f.price - f.ema9_1m
        direction = 0
        if last_body > 0 and f.cvd_slope > 0 and f.bb_pctb > 0.8 and f.htf_bias >= 0:
            direction = 1
        elif last_body < 0 and f.cvd_slope < 0 and f.bb_pctb < 0.2 and f.htf_bias <= 0:
            direction = -1
        else:
            return None

        r = 1.3 * f.atr_1m
        if direction == 1:
            sl = f.price - r
            tp1 = f.price + r * self.TP1_R
            tp2 = f.price + r * self.TP2_R
        else:
            sl = f.price + r
            tp1 = f.price - r * self.TP1_R
            tp2 = f.price - r * self.TP2_R

        conf = min(1.0, 0.5 + min(f.vol_z / 3.0, 0.3) + f.regime.strength * 0.2)
        return Signal(
            side="LONG" if direction == 1 else "SHORT",
            entry_type="MARKET", entry=f.price,
            sl=sl, tp1=tp1, tp2=tp2, r_unit=r,
            setup="SQUEEZE_BREAKOUT", confidence=conf,
            reasons={"vol_z": f.vol_z, "cvd": f.cvd_slope, "pctb": f.bb_pctb},
        )

    def generate_signal(self, f: Features,
                        ml_prob: Optional[float] = None,
                        ml_min: float = 0.55) -> Optional[Signal]:
        """Try each setup in order of priority; return the first valid one.

        `ml_prob` (if provided) must be the probability that a LONG trade
        would hit TP1 before SL given the current features (produced by
        `ml_engine_v2` which trains labels aligned with this exit model).
        """
        for cand in (self._trend_pullback(f), self._squeeze_breakout(f), self._range_fade(f)):
            if cand is None:
                continue
            # ML gate: directional probability must exceed minimum.
            if ml_prob is not None:
                p = ml_prob if cand.side == "LONG" else 1.0 - ml_prob
                if p < ml_min:
                    continue
                cand.confidence = min(1.0, cand.confidence * 0.6 + p * 0.4)
            return cand
        return None


# ---------- Position sizing ----------

def fractional_kelly_risk(base_risk: float, win_rate: float, rr: float,
                          fraction: float = 0.5, floor: float = 0.25,
                          ceil: float = 1.5) -> float:
    """Return a multiplier on `base_risk` using fractional Kelly.

    We cap between 0.25x and 1.5x of base to avoid volatile sizing on small
    samples. `win_rate` should be a smoothed estimate over recent 30 trades.
    """
    if win_rate <= 0 or rr <= 0:
        return base_risk
    k = win_rate - (1 - win_rate) / rr
    if k <= 0:
        return base_risk * floor
    mult = max(floor, min(ceil, k * fraction / 0.125))  # 0.125 is half-Kelly @ 50% / 2R
    return base_risk * mult


def position_notional(balance: float, risk_pct: float,
                      entry: float, sl: float, max_leverage: float = 20.0) -> float:
    """Notional USD such that (entry -> sl) loss equals `balance * risk_pct`."""
    dist_pct = abs(entry - sl) / entry
    if dist_pct <= 0:
        return 0.0
    notional = (balance * risk_pct) / dist_pct
    return min(notional, balance * max_leverage)
