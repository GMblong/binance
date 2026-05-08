"""Feature matrix for ml_v2.

Inputs expected in the symbol DataFrame (1m bars):
  ot, o, h, l, c, v, (tbv optional)

Plus external aligned series:
  btc_close: pd.Series of BTC close prices at the same ot timestamps
  funding_series: Series indexed by funding-time ms, value = rate
  cross_section: DataFrame with columns = symbols, values = 1h returns
                 (aligned on ot). Optional; when absent, those features
                 are zero-filled.

Output: a DataFrame with all engineered features. Rows with NaNs (due
to rolling windows) are retained so the caller can decide whether to
dropna per-use.

The FEATURE_COLUMNS list is the canonical ordered list. ALWAYS index
X[features] using this list so train/infer stay aligned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict

from microstructure.cvd import compute_cvd
from microstructure.lead_lag import btc_beta
from microstructure.funding import funding_skew


FEATURE_COLUMNS = [
    # Price momentum
    "ret_1", "ret_3", "ret_5", "ret_15", "ret_60",
    # Volatility
    "atr_pct", "atr_ratio",
    # Trend structure
    "ema9_slope", "dist_ema21", "dist_ema50", "dist_ema200",
    # RSI + MACD
    "rsi14", "macd_hist",
    # Candle shape
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
    # Flow
    "cvd_roc3", "cvd_roc10", "vol_z",
    # BTC context
    "btc_ret_1", "btc_ret_5", "btc_ret_15", "btc_beta", "btc_gap",
    # Time of day (cyc encoded)
    "tod_sin", "tod_cos", "dow",
    # Funding & OI (if available)
    "funding_rate", "funding_skew",
    "oi_roc3",
    # Cross-section
    "cs_rank_1h",
    # Microstructure extras
    "range_expansion", "vol_spike",
    # Symbol embedding (set by trainer/predictor, 0 if unknown)
    "sym_id",
]


def _safe_ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=max(3, span // 3)).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = pd.concat([
        df["h"] - df["l"],
        (df["h"] - df["c"].shift()).abs(),
        (df["l"] - df["c"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=3).mean()


def build_feature_matrix(
    df: pd.DataFrame,
    btc_close: Optional[pd.Series] = None,
    funding_series: Optional[pd.Series] = None,
    cross_section: Optional[pd.DataFrame] = None,
    symbol: Optional[str] = None,
    sym_id_map: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Compute feature matrix for a single symbol's 1m bars.

    Returns a DataFrame whose columns are a SUPERSET of FEATURE_COLUMNS;
    missing features are filled with 0 so downstream indexing with
    FEATURE_COLUMNS always works.
    """
    if len(df) < 60:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    out = pd.DataFrame(index=df.index)
    c = df["c"]
    v = df["v"]

    # Returns
    out["ret_1"] = c.pct_change(1)
    out["ret_3"] = c.pct_change(3)
    out["ret_5"] = c.pct_change(5)
    out["ret_15"] = c.pct_change(15)
    out["ret_60"] = c.pct_change(60)

    # Volatility
    atr = _atr(df, 14)
    out["atr_pct"] = atr / c * 100
    atr_long = atr.rolling(200, min_periods=30).median()
    out["atr_ratio"] = atr / atr_long.replace(0, np.nan)

    # Trend structure
    e9 = _safe_ema(c, 9)
    e21 = _safe_ema(c, 21)
    e50 = _safe_ema(c, 50)
    e200 = _safe_ema(c, 200)
    out["ema9_slope"] = e9.diff(3) / c
    out["dist_ema21"] = (c - e21) / e21
    out["dist_ema50"] = (c - e50) / e50
    out["dist_ema200"] = (c - e200) / e200

    # RSI + MACD
    out["rsi14"] = _rsi(c, 14) / 100.0
    macd_fast = _safe_ema(c, 12)
    macd_slow = _safe_ema(c, 26)
    macd = macd_fast - macd_slow
    macd_sig = _safe_ema(macd, 9)
    out["macd_hist"] = (macd - macd_sig) / c

    # Candle shape
    rng = (df["h"] - df["l"]).replace(0, np.nan)
    out["body_ratio"] = (df["c"] - df["o"]).abs() / rng
    out["upper_wick_ratio"] = (df["h"] - df[["o", "c"]].max(axis=1)) / rng
    out["lower_wick_ratio"] = (df[["o", "c"]].min(axis=1) - df["l"]) / rng

    # Flow
    cvd = compute_cvd(df)
    out["cvd_roc3"] = cvd.pct_change(3)
    out["cvd_roc10"] = cvd.pct_change(10)
    vol_mean = v.rolling(50, min_periods=10).mean()
    vol_std = v.rolling(50, min_periods=10).std().replace(0, np.nan)
    out["vol_z"] = (v - vol_mean) / vol_std

    # BTC context
    if btc_close is not None and len(btc_close) >= len(df):
        btc = btc_close.reindex(df.index).ffill()
        out["btc_ret_1"] = btc.pct_change(1)
        out["btc_ret_5"] = btc.pct_change(5)
        out["btc_ret_15"] = btc.pct_change(15)
        # Rolling beta approx: last-window beta at every step would be
        # expensive. We compute once using the last window and broadcast,
        # plus recent beta_gap. For training we rely on per-bar BTC lag
        # returns above; single beta is good enough as a "how reactive is
        # this coin to BTC" slow feature.
        try:
            beta_last, gap_last = btc_beta(c, btc, window=120, lag=3)
        except Exception:
            beta_last, gap_last = 1.0, 0.0
        out["btc_beta"] = beta_last
        out["btc_gap"] = gap_last
    else:
        out["btc_ret_1"] = 0.0
        out["btc_ret_5"] = 0.0
        out["btc_ret_15"] = 0.0
        out["btc_beta"] = 1.0
        out["btc_gap"] = 0.0

    # Time of day (minute of UTC day, cyclical)
    try:
        ts = pd.to_datetime(df["ot"].astype("int64"), unit="ms", utc=True)
        minute_of_day = ts.dt.hour * 60 + ts.dt.minute
        out["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
        out["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
        out["dow"] = ts.dt.dayofweek.astype(float) / 6.0
    except Exception:
        out["tod_sin"] = 0.0
        out["tod_cos"] = 1.0
        out["dow"] = 0.0

    # Funding
    if funding_series is not None and len(funding_series) > 0:
        try:
            ts_ms = df["ot"].astype("int64")
            aligned = pd.Series(
                funding_series.reindex(ts_ms.values, method="ffill").values,
                index=df.index,
            )
            out["funding_rate"] = aligned.astype(float)
            out["funding_skew"] = aligned.apply(funding_skew)
        except Exception:
            out["funding_rate"] = 0.0
            out["funding_skew"] = 0.0
    else:
        out["funding_rate"] = 0.0
        out["funding_skew"] = 0.0

    # OI ROC (optional, if df has 'oi')
    if "oi" in df.columns:
        out["oi_roc3"] = df["oi"].pct_change(3)
    else:
        out["oi_roc3"] = 0.0

    # Cross-section rank
    if cross_section is not None and symbol is not None:
        try:
            sym_ret = cross_section[symbol].reindex(df.index).ffill()
            ranks = cross_section.reindex(df.index).ffill().rank(axis=1, pct=True)
            out["cs_rank_1h"] = ranks[symbol]
        except Exception:
            out["cs_rank_1h"] = 0.5
    else:
        out["cs_rank_1h"] = 0.5

    # Microstructure extras
    out["range_expansion"] = (df["h"] - df["l"]) / atr.replace(0, np.nan)
    out["vol_spike"] = v / vol_mean.replace(0, np.nan)

    # Symbol embedding
    sid = 0
    if sym_id_map is not None and symbol in sym_id_map:
        sid = sym_id_map[symbol]
    out["sym_id"] = sid

    # Clean inf/NaN policy: replace inf with NaN, keep NaN for caller.
    out = out.replace([np.inf, -np.inf], np.nan)

    # Guarantee all canonical columns exist
    for col in FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    return out
