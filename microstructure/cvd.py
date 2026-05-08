"""Cumulative Volume Delta built from Binance kline taker-buy-volume.

Binance futures klines include a field `tbv` (taker buy base volume) as
the 10th column. CVD per bar = 2*tbv - total_volume (i.e. takers - makers
on the buy/sell axis, sign-normalized).

Signals:
- `cvd_divergence`: price makes a new high/low but CVD doesn't -> reversal
  bias.
- `cvd_absorption`: strong directional volume prints with flat price ->
  walls absorbing, expect reversal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


def compute_cvd(df: pd.DataFrame, tbv_col: str = "tbv") -> pd.Series:
    """Build CVD from Binance kline fields. Requires `tbv` (taker buy
    volume) and `v` (total volume). Returns a running sum in base units.

    If `tbv` is missing we fall back to the wick-based proxy already
    used in strategies/analyzer.py so callers never crash.
    """
    if tbv_col in df.columns and df[tbv_col].notna().any():
        delta = 2.0 * df[tbv_col] - df["v"]
    else:
        # Fallback: wick-based delta (matches the existing engine)
        rng = (df["h"] - df["l"]).replace(0, np.nan)
        up_weight = (df["c"] - df["l"]) / rng
        down_weight = (df["h"] - df["c"]) / rng
        delta = df["v"] * (up_weight - down_weight)
        delta = delta.fillna(0)
    return delta.cumsum()


def cvd_divergence(
    df: pd.DataFrame, lookback: int = 20, tbv_col: str = "tbv"
) -> int:
    """Return -1 / 0 / +1.

    +1 = bullish divergence (price lower-low, CVD higher-low -> absorb sellers)
    -1 = bearish divergence (price higher-high, CVD lower-high -> absorb buyers)
    """
    if len(df) < lookback + 1:
        return 0
    cvd = compute_cvd(df.tail(lookback + 5), tbv_col=tbv_col)
    sub = df.tail(lookback)
    px = sub["c"]
    cvd_sub = cvd.iloc[-lookback:]
    if len(px) < lookback or len(cvd_sub) < lookback:
        return 0
    try:
        px_hh = px.iloc[-1] >= px.iloc[:-1].max()
        px_ll = px.iloc[-1] <= px.iloc[:-1].min()
        cvd_hh = cvd_sub.iloc[-1] >= cvd_sub.iloc[:-1].max()
        cvd_ll = cvd_sub.iloc[-1] <= cvd_sub.iloc[:-1].min()
        if px_ll and not cvd_ll:
            return 1
        if px_hh and not cvd_hh:
            return -1
    except Exception:
        return 0
    return 0


def cvd_absorption(
    df: pd.DataFrame, lookback: int = 10, vol_mult: float = 2.0,
    price_tol_atr: float = 0.3, tbv_col: str = "tbv",
) -> int:
    """Return -1 / 0 / +1 for absorption signal.

    +1 = heavy sell pressure (CVD dumps) but price stays flat relative to
         ATR -> bid wall absorbing, bullish.
    -1 = symmetric bearish absorption.
    """
    if len(df) < lookback + 15:
        return 0
    sub = df.tail(lookback)
    cvd = compute_cvd(df.tail(lookback + 5), tbv_col=tbv_col).tail(lookback)
    try:
        tr = pd.concat(
            [sub["h"] - sub["l"],
             (sub["h"] - sub["c"].shift()).abs(),
             (sub["l"] - sub["c"].shift()).abs()], axis=1,
        ).max(axis=1)
        atr = tr.rolling(14, min_periods=3).mean().iloc[-1]
        if not np.isfinite(atr) or atr <= 0:
            return 0
        price_move = abs(sub["c"].iloc[-1] - sub["c"].iloc[0])
        cvd_move = cvd.iloc[-1] - cvd.iloc[0]
        avg_vol = df["v"].tail(50).mean()
        recent_vol = sub["v"].mean()
        flat_price = price_move <= price_tol_atr * atr
        big_vol = recent_vol >= vol_mult * avg_vol
        if flat_price and big_vol:
            if cvd_move < 0:
                return 1  # sellers absorbed
            if cvd_move > 0:
                return -1  # buyers absorbed
    except Exception:
        return 0
    return 0
