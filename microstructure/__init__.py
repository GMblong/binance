"""Microstructure signals (Phase 3).

These are quantities that are hard/impossible for a human to read off a
price chart but have a real, measurable effect on next-bar direction:

- Order-book L5 imbalance + liquidity void detection
- Cumulative volume delta (CVD) built from Binance's taker-buy-volume
  field in klines, so it is usable in backtests (no live trade feed).
- Funding-rate skew & extreme detection
- Liquidation cascade proxy (large range + volume spike, high wick ratio)
- BTC -> alt rolling beta and beta-gap (alt lagging vs. its regressed
  prediction)

Each sub-module exposes pure functions that take pandas DataFrames /
series and return either a scalar signal, a Series, or a dict. Nothing
in this package depends on a live websocket, so the same code runs in
backtests and in the live bot.
"""

from .orderbook import (
    ob_l5_imbalance,
    detect_liquidity_void,
    fetch_depth_snapshot,
)
from .cvd import compute_cvd, cvd_divergence, cvd_absorption
from .funding import funding_skew, funding_crowding_signal
from .liquidations import detect_liq_cascade, liq_cascade_signal
from .lead_lag import btc_beta, btc_beta_gap_signal

__all__ = [
    "ob_l5_imbalance",
    "detect_liquidity_void",
    "fetch_depth_snapshot",
    "compute_cvd",
    "cvd_divergence",
    "cvd_absorption",
    "funding_skew",
    "funding_crowding_signal",
    "detect_liq_cascade",
    "liq_cascade_signal",
    "btc_beta",
    "btc_beta_gap_signal",
]
