"""Funding-rate microstructure.

When perpetual funding gets extreme (e.g. > 0.05% per 8h), the losing
side of the funding payment is typically over-leveraged retail. The
classic pattern: extreme long funding + rising OI + sideways price
= short-squeeze setup TO THE DOWNSIDE (longs get liquidated first).

`funding_crowding_signal` returns a directional bias from
contrarian extremes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


EXTREME_THRESHOLD_POS = 0.0005   # +0.05% per 8h
EXTREME_THRESHOLD_NEG = -0.0003  # -0.03% per 8h (shorts less likely to crowd)


def funding_skew(current_rate: float) -> float:
    """Return a signed z-like value in [-1, +1] for how extreme this rate is."""
    if current_rate is None:
        return 0.0
    if current_rate > EXTREME_THRESHOLD_POS:
        # Longs are paying to stay long -> over-crowded -> bearish
        z = (current_rate - EXTREME_THRESHOLD_POS) / EXTREME_THRESHOLD_POS
        return -min(1.0, z)
    if current_rate < EXTREME_THRESHOLD_NEG:
        z = (EXTREME_THRESHOLD_NEG - current_rate) / abs(EXTREME_THRESHOLD_NEG)
        return min(1.0, z)
    return 0.0


def funding_crowding_signal(
    current_rate: Optional[float],
    oi_change_pct: Optional[float],
    price_range_pct: Optional[float],
    direction_hint: int = 0,
) -> int:
    """Return -1 / 0 / +1 for contrarian funding trade setup.

    +1 = contrarian long  (funding very negative + price flat + OI rising)
    -1 = contrarian short (funding very positive + price flat + OI rising)

    `direction_hint` is your current trend direction; we ONLY signal when
    it agrees with the contrarian side (i.e. we don't fight the primary
    trend, we just boost confidence when a contrarian setup aligns).
    """
    if current_rate is None:
        return 0
    sk = funding_skew(current_rate)
    if abs(sk) < 0.5:
        return 0
    flat_price = price_range_pct is not None and price_range_pct < 1.5
    rising_oi = oi_change_pct is not None and oi_change_pct > 0.5
    if not flat_price or not rising_oi:
        return 0
    if sk < 0 and (direction_hint == 0 or direction_hint == -1):
        return -1
    if sk > 0 and (direction_hint == 0 or direction_hint == 1):
        return 1
    return 0
