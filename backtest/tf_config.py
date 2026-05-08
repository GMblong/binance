"""Timeframe-aware configuration for the backtest + ML stack.

Motivation: 1m crypto bars are dominated by HFT/market-maker noise that
no causal model from OHLCV can beat after fees. Moving the base bar to
5m or 15m gives a dramatically better signal-to-noise ratio:

  - Realized per-bar return dispersion ~sqrt(5) or sqrt(15) times larger
    relative to fixed fee cost.
  - ATR-based PT/SL widens naturally, so TP 1.5x ATR clears spread+fee
    with margin instead of being consumed by them.
  - The same 3 candles of structure (FVG, OB, CHoCH) become much more
    meaningful on 5m than they ever are on 1m.

This module is a SINGLE source of truth for every knob that depends on
the base timeframe: which interval strings to fetch, how many bars per
training chunk, how the hard MTF gate maps to higher timeframes, and
how many bars are in a day for simulation length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TFConfig:
    """Configuration for one base timeframe (e.g. 5m).

    base_interval: Binance kline interval string for the primary bar.
    mtf_interval: Mid timeframe for trend/regime (~3x base).
    htf_interval: Higher timeframe for HTF bias (~12x base).
    bars_per_day: How many base bars are in 24h (used for sim length).
    mtf_ratio: Integer ratio `mtf / base` (e.g. 3 for 5m->15m).
    htf_ratio: Integer ratio `htf / base` (e.g. 12 for 5m->1h).
    default_train_chunks: How many fetches of 1500 bars to request for
        training. Aim for >= 10k samples per model even after dropna.
    horizon_bars: Triple-barrier lookahead in base bars. Chosen so the
        label's wall-clock horizon is comparable across TFs (~150min).
    signal_every_n_bars: How often the engine re-evaluates per-symbol
        signals. For 5m we evaluate every bar; for 1m every 5 bars so
        the total simulated signal rate is roughly constant.
    regime_atr_quantile: If set, trainer will override the hard-coded
        regime thresholds with per-regime quantiles computed from the
        training corpus, so bands auto-calibrate to each TF's realized
        volatility distribution. None means "use hard-coded defaults".
    fvg_lookback: Number of base bars to scan for FVG / OB structure.
    trail_ttl_bars: TTL for pending limit orders before they expire.
    """

    name: str
    base_interval: str
    mtf_interval: str
    htf_interval: str
    bars_per_day: int
    mtf_ratio: int
    htf_ratio: int
    default_train_chunks: int
    horizon_bars: int
    signal_every_n_bars: int
    regime_atr_quantile: bool
    fvg_lookback: int
    trail_ttl_bars: int


TF_CONFIGS: Dict[str, TFConfig] = {
    "1m": TFConfig(
        name="1m",
        base_interval="1m",
        mtf_interval="15m",
        htf_interval="1h",
        bars_per_day=1440,
        mtf_ratio=15,
        htf_ratio=60,
        default_train_chunks=14,
        horizon_bars=30,
        signal_every_n_bars=5,
        regime_atr_quantile=True,
        fvg_lookback=200,
        trail_ttl_bars=15,
    ),
    "5m": TFConfig(
        name="5m",
        base_interval="5m",
        mtf_interval="15m",
        htf_interval="1h",
        bars_per_day=288,
        mtf_ratio=3,
        htf_ratio=12,
        # 1500 * 5min = 7500 minutes = ~5.2 days per chunk, so 4 chunks
        # gives ~3 weeks of 5m data which is plenty for LightGBM.
        default_train_chunks=4,
        # horizon_bars=6 -> 30 min lookahead, matches live hold-time
        # distributions of the scalper.
        horizon_bars=6,
        signal_every_n_bars=1,
        regime_atr_quantile=True,
        fvg_lookback=80,
        trail_ttl_bars=5,
    ),
    "15m": TFConfig(
        name="15m",
        base_interval="15m",
        mtf_interval="1h",
        htf_interval="4h",
        bars_per_day=96,
        mtf_ratio=4,
        htf_ratio=16,
        default_train_chunks=3,
        horizon_bars=4,
        signal_every_n_bars=1,
        regime_atr_quantile=True,
        fvg_lookback=50,
        trail_ttl_bars=3,
    ),
}


def get(tf_name: str) -> TFConfig:
    """Resolve a TFConfig by base-interval string, raising on unknown."""
    if tf_name not in TF_CONFIGS:
        raise ValueError(
            f"Unknown TF '{tf_name}'. Available: {list(TF_CONFIGS.keys())}"
        )
    return TF_CONFIGS[tf_name]
