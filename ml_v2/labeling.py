"""Triple-barrier labels (de Prado style).

For each bar i we set:
  - PT (profit target) = close[i] + pt_atr_mult * ATR[i]
  - SL (stop loss)     = close[i] - sl_atr_mult * ATR[i]
  - Horizon H bars

Label:
  2 = UP_PT first
  0 = DOWN_SL first
  1 = TIMEOUT (neither hit within H)

The 3-class formulation lets the model learn "directionless / chop" bars
as a real class, rather than being forced to pick UP or DOWN.

For a directional binary target, use `collapse_to_binary` which maps
{UP_PT: 1, TIMEOUT: NaN, DOWN_SL: 0} and drops timeouts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = pd.concat([
        df["h"] - df["l"],
        (df["h"] - df["c"].shift()).abs(),
        (df["l"] - df["c"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=3).mean()


def triple_barrier_labels(
    df: pd.DataFrame,
    pt_atr_mult: float = 1.5,
    sl_atr_mult: float = 1.0,
    horizon: int = 15,
    vol_scale: bool = True,
) -> pd.Series:
    """Return a Series of labels in {0, 1, 2} aligned with df.index.

    Bars where we don't have `horizon` lookahead are labeled NaN.
    """
    atr = _atr(df, 14)
    if vol_scale:
        base_atr = atr.rolling(200, min_periods=30).median().bfill()
        ratio = (atr / base_atr).clip(lower=0.6, upper=2.0).fillna(1.0)
    else:
        ratio = pd.Series(1.0, index=df.index)

    c = df["c"].values
    h = df["h"].values
    lo = df["l"].values
    a = atr.values
    r = ratio.values
    n = len(df)
    out = np.full(n, np.nan, dtype=float)

    for i in range(n - horizon):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        pt = c[i] + pt_atr_mult * a[i] * r[i]
        sl = c[i] - sl_atr_mult * a[i] * r[i]
        label = 1  # timeout default
        for j in range(1, horizon + 1):
            idx = i + j
            if h[idx] >= pt:
                label = 2
                break
            if lo[idx] <= sl:
                label = 0
                break
        out[i] = label
    return pd.Series(out, index=df.index, name="tb_label")


def collapse_to_binary(labels: pd.Series) -> pd.Series:
    """Collapse 3-class TB labels to {0, 1}. Drops timeouts (label=1)."""
    s = labels.copy()
    s = s.where(s != 1, np.nan)
    s = s.map({0.0: 0, 2.0: 1})
    return s
