"""BTC -> alt lead-lag via rolling OLS beta.

Rather than the simple "5-bar return" heuristic used in v1, here we
regress alt returns on BTC returns at lags 1..5 minutes over a rolling
120-minute window. We then compute the "beta gap" = predicted alt
return - realized alt return. A large positive beta-gap means BTC has
already moved and the alt has not caught up -> expected to catch up.

For backtests we only need the two close-price series; `btc_beta` takes
two aligned series and returns (beta, gap) at the current bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


def btc_beta(
    alt_close: pd.Series,
    btc_close: pd.Series,
    window: int = 120,
    lag: int = 1,
) -> Tuple[float, float]:
    """Return (beta, beta_gap_pct) at the last index.

    beta: slope of alt_return on btc_return over `window` bars with
    `lag` bars of BTC lead. If not enough data, returns (1.0, 0.0).
    beta_gap_pct: percent catch-up expected, positive = alt is lagging up.
    """
    n = min(len(alt_close), len(btc_close))
    if n < window + lag + 2:
        return 1.0, 0.0
    try:
        alt = alt_close.iloc[-window - lag:].pct_change().dropna()
        btc = btc_close.iloc[-window - lag:].pct_change().dropna()
        if len(alt) < window or len(btc) < window:
            return 1.0, 0.0
        alt_r = alt.iloc[lag:].values
        btc_r = btc.iloc[:-lag].values if lag > 0 else btc.values
        m = min(len(alt_r), len(btc_r))
        alt_r = alt_r[-m:]
        btc_r = btc_r[-m:]
        if m < 30:
            return 1.0, 0.0
        # OLS via numpy
        var_b = np.var(btc_r)
        if var_b <= 0:
            return 1.0, 0.0
        cov = np.mean(btc_r * alt_r) - np.mean(btc_r) * np.mean(alt_r)
        beta = float(cov / var_b)
        # Recent BTC move that alt has not yet caught up to
        recent_btc_ret = float(btc_close.iloc[-1] / btc_close.iloc[-lag - 1] - 1.0)
        expected_alt = beta * recent_btc_ret
        actual_alt = float(alt_close.iloc[-1] / alt_close.iloc[-lag - 1] - 1.0)
        gap = expected_alt - actual_alt
        return beta, gap * 100.0
    except Exception:
        return 1.0, 0.0


def btc_beta_gap_signal(
    alt_close: pd.Series,
    btc_close: pd.Series,
    direction_hint: int,
    min_gap_pct: float = 0.15,
    min_beta: float = 0.5,
) -> int:
    """Return -1 / 0 / +1. Fires only when gap is in our favor AND beta
    is meaningful (> 0.5). This filters stable low-beta coins that
    don't actually track BTC."""
    beta, gap = btc_beta(alt_close, btc_close, window=120, lag=3)
    if abs(beta) < min_beta:
        return 0
    if direction_hint == 1 and gap >= min_gap_pct:
        return 1
    if direction_hint == -1 and gap <= -min_gap_pct:
        return -1
    return 0
