"""Liquidation cascade proxy (from klines).

The proper signal is the `@forceOrder` websocket stream, which we use
live. But for backtests we need a historical proxy. A liquidation
cascade looks like:

- Sudden range expansion (range >= 3x ATR14)
- Volume spike (>= 3x 50-bar mean)
- Large body-to-range ratio in one direction (> 0.65 = conviction flush)

After such a bar, the pattern is a) short-term continuation for 1-3
bars then b) reversion to the pre-cascade level. The tradeable edge is
the reversion, NOT the continuation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def detect_liq_cascade(
    df: pd.DataFrame,
    atr_mult: float = 3.0,
    vol_mult: float = 3.0,
    body_min: float = 0.65,
) -> Optional[dict]:
    """Return dict describing the cascade on the last bar, or None.

    Keys: `side` ("UP"/"DOWN"), `bar_idx`, `entry_fade` (price level to
    target for reversion), `strength` (0-1).
    """
    if len(df) < 60:
        return None
    try:
        tr = pd.concat(
            [df["h"] - df["l"],
             (df["h"] - df["c"].shift()).abs(),
             (df["l"] - df["c"].shift()).abs()], axis=1,
        ).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1]
        mean_v50 = df["v"].tail(50).mean()
        last = df.iloc[-1]
        rng = last["h"] - last["l"]
        body = abs(last["c"] - last["o"])
        if not np.isfinite(atr14) or atr14 <= 0 or rng <= 0:
            return None
        range_ratio = rng / atr14
        vol_ratio = last["v"] / (mean_v50 + 1e-8)
        body_ratio = body / rng
        if range_ratio < atr_mult or vol_ratio < vol_mult or body_ratio < body_min:
            return None
        side = "UP" if last["c"] > last["o"] else "DOWN"
        # Fade target: midpoint of the bar (reversion level)
        mid = (last["h"] + last["l"]) / 2.0
        strength = min(1.0, (range_ratio / atr_mult) * (vol_ratio / vol_mult) / 4.0)
        return {
            "side": side,
            "entry_fade": float(mid),
            "strength": float(strength),
            "range_ratio": float(range_ratio),
            "vol_ratio": float(vol_ratio),
            "body_ratio": float(body_ratio),
        }
    except Exception:
        return None


def liq_cascade_signal(df: pd.DataFrame, direction_hint: int) -> int:
    """Return -1 / 0 / +1 based on cascade reversion opportunity.

    We only return a signal that AGREES with the contrarian (reversion)
    side of the cascade AND with our `direction_hint`. This prevents us
    from shorting into a cascade that is still roaring.
    """
    casc = detect_liq_cascade(df)
    if casc is None:
        return 0
    # Contrarian side of a DOWN flush = long (+1)
    if casc["side"] == "DOWN" and direction_hint == 1:
        return 1
    if casc["side"] == "UP" and direction_hint == -1:
        return -1
    return 0
