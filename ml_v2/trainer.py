"""Pooled trainer across symbols.

Builds one feature matrix per symbol, stacks them with a `sym_id`
feature that acts as a categorical embedding, then trains one LightGBM
per regime. This matters because:

- Low-sample alts benefit from BTC/ETH history shape.
- A single pooled model is easier to deploy + calibrate than N per-symbol
  models.
- The symbol embedding lets the model still learn symbol-specific
  quirks.

Regime bucketing can be either hard-coded (legacy) or data-driven
(recommended): set `regime_quantile=True` to `train_pooled` and the
trainer will compute per-corpus quantiles of `atr_pct` and `|dist_ema21|`
so the bands auto-calibrate to each timeframe's realized volatility.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .labeling import triple_barrier_labels, collapse_to_binary
from .features import build_feature_matrix, FEATURE_COLUMNS
from .models import train_lgbm_with_cv

REGIMES = ("TRENDING", "VOLATILE", "RANGING")


def _regime_of(
    atr_pct: float,
    dist_ema21: float,
    atr_vol_hi: float = 1.2,
    atr_trend_lo: float = 0.20,
    dist_trend_lo: float = 0.0015,
) -> str:
    """Regime bucketing (data-driven when thresholds come from quantiles).

      VOLATILE = atr_pct >= atr_vol_hi                      (news/chop spike)
      TRENDING = atr_pct >= atr_trend_lo AND |dist|>=dist_trend_lo
      RANGING  = everything else

    Default thresholds are the hard-coded 1m defaults. When
    `train_pooled` runs with `regime_quantile=True` it replaces these
    numbers with per-corpus quantiles before calling this function.
    """
    if not np.isfinite(atr_pct) or not np.isfinite(dist_ema21):
        return "RANGING"
    if atr_pct > atr_vol_hi:
        return "VOLATILE"
    if atr_pct > atr_trend_lo and abs(dist_ema21) > dist_trend_lo:
        return "TRENDING"
    return "RANGING"


def assign_regime(
    features: pd.DataFrame,
    atr_vol_hi: float = 1.2,
    atr_trend_lo: float = 0.20,
    dist_trend_lo: float = 0.0015,
) -> pd.Series:
    """Coarse per-bar regime label from cheap features present in
    `features`. We don't train a regime classifier -- this is
    intentional: the "regime" is just a routing key so each sub-model
    specializes.

    Thresholds are accepted as parameters so inference and training can
    share a single policy source.
    """
    atr = features.get("atr_pct")
    dist = features.get("dist_ema21")
    if atr is None or dist is None:
        return pd.Series(["RANGING"] * len(features), index=features.index)
    return pd.Series(
        [
            _regime_of(a, d, atr_vol_hi, atr_trend_lo, dist_trend_lo)
            for a, d in zip(atr, dist)
        ],
        index=features.index,
    )


def compute_regime_quantile_thresholds(
    feats_concat: pd.DataFrame,
    vol_q: float = 0.80,
    trend_atr_q: float = 0.40,
    trend_dist_q: float = 0.50,
) -> Dict[str, float]:
    """Data-driven regime thresholds from the training corpus.

    - `atr_vol_hi`  = `vol_q` quantile of atr_pct (top 20% = VOLATILE).
    - `atr_trend_lo` = `trend_atr_q` quantile (anything below is RANGING).
    - `dist_trend_lo` = `trend_dist_q` quantile of |dist_ema21|.

    This gives ~40% RANGING, ~40% TRENDING, ~20% VOLATILE regardless of
    TF. Much better than fixed thresholds that were capturing 90%+
    RANGING on 1m crypto.

    If the corpus is too small or the columns are missing, returns safe
    defaults matching the hard-coded policy.
    """
    defaults = {
        "atr_vol_hi": 1.2,
        "atr_trend_lo": 0.20,
        "dist_trend_lo": 0.0015,
    }
    try:
        if "atr_pct" not in feats_concat.columns:
            return defaults
        atr = feats_concat["atr_pct"].dropna()
        if len(atr) < 500:
            return defaults
        vol_hi = float(atr.quantile(vol_q))
        trend_lo = float(atr.quantile(trend_atr_q))
        if "dist_ema21" in feats_concat.columns:
            dist_abs = feats_concat["dist_ema21"].abs().dropna()
            dist_lo = (
                float(dist_abs.quantile(trend_dist_q))
                if len(dist_abs) >= 500
                else defaults["dist_trend_lo"]
            )
        else:
            dist_lo = defaults["dist_trend_lo"]
        # Sanity clamps so a freakish quantile doesn't make the band empty.
        vol_hi = max(0.3, min(vol_hi, 5.0))
        trend_lo = max(0.05, min(trend_lo, vol_hi - 0.05))
        dist_lo = max(1e-4, min(dist_lo, 0.05))
        return {
            "atr_vol_hi": vol_hi,
            "atr_trend_lo": trend_lo,
            "dist_trend_lo": dist_lo,
        }
    except Exception:
        return defaults


def train_pooled(
    per_symbol: Dict[str, pd.DataFrame],
    btc_close: pd.Series,
    funding_per_symbol: Optional[Dict[str, pd.Series]] = None,
    cross_section: Optional[pd.DataFrame] = None,
    pt_atr_mult: float = 1.5,
    sl_atr_mult: float = 1.0,
    horizon: int = 30,
    n_splits: int = 5,
    embargo: int = 30,
    regime_quantile: bool = False,
) -> Dict[str, Any]:
    """Train one model per regime on the pooled corpus.

    `horizon` is in base bars, not minutes. Defaults to 30 to roughly
    match 1m bar hold times (~30 minutes). For 5m base bars the caller
    should pass horizon=6 (also ~30 min wall clock).

    `regime_quantile=True` replaces the hard-coded band thresholds with
    quantile-based ones computed from the training corpus itself, so
    ~20% of bars land in VOLATILE and ~40% in TRENDING regardless of
    timeframe. Strongly recommended for any TF other than 1m.

    Returns a dict:
      `{"models": {regime: model}, "sym_id_map": {...}, "metrics": {...},
        "regime_thresholds": {...}}`.
    """
    funding_per_symbol = funding_per_symbol or {}
    sym_id_map = {s: i + 1 for i, s in enumerate(sorted(per_symbol.keys()))}

    # First pass: build all feature frames and labels so we can compute
    # corpus-wide quantile thresholds before bucketing anything.
    per_sym_feats: Dict[str, pd.DataFrame] = {}
    per_sym_y: Dict[str, pd.Series] = {}
    for sym, df in per_symbol.items():
        if len(df) < 200:
            continue
        lbl = triple_barrier_labels(df, pt_atr_mult, sl_atr_mult, horizon)
        y_bin = collapse_to_binary(lbl)
        feats = build_feature_matrix(
            df,
            btc_close=btc_close,
            funding_series=funding_per_symbol.get(sym),
            cross_section=cross_section,
            symbol=sym,
            sym_id_map=sym_id_map,
        )
        per_sym_feats[sym] = feats
        per_sym_y[sym] = y_bin

    if not per_sym_feats:
        return {
            "models": {},
            "sym_id_map": sym_id_map,
            "metrics": {},
            "regime_thresholds": None,
        }

    # Compute regime thresholds.
    if regime_quantile:
        concat = pd.concat(per_sym_feats.values(), ignore_index=True)
        thresholds = compute_regime_quantile_thresholds(concat)
    else:
        thresholds = {
            "atr_vol_hi": 1.2,
            "atr_trend_lo": 0.20,
            "dist_trend_lo": 0.0015,
        }

    all_X: List[pd.DataFrame] = []
    all_y: List[pd.Series] = []
    all_regime: List[pd.Series] = []
    all_w: List[pd.Series] = []

    for sym, feats in per_sym_feats.items():
        y_bin = per_sym_y[sym]
        regime = assign_regime(
            feats,
            atr_vol_hi=thresholds["atr_vol_hi"],
            atr_trend_lo=thresholds["atr_trend_lo"],
            dist_trend_lo=thresholds["dist_trend_lo"],
        )
        # Do NOT drop rows where only a handful of columns are NaN -- those
        # are warm-up bars. Require the label and the core price features
        # only; fill the rest with the per-column median.
        core_cols = [
            c for c in ["atr_pct", "ret_1", "dist_ema21", "rsi14"]
            if c in feats.columns
        ]
        joined = feats.join(y_bin.rename("y")).join(regime.rename("regime"))
        joined = joined.dropna(subset=["y"] + core_cols)
        # Fill remaining NaNs in canonical feature list so LightGBM doesn't
        # silently mark them as a category.
        for col in FEATURE_COLUMNS:
            if col in joined.columns:
                joined[col] = joined[col].astype(float)
                if joined[col].isna().any():
                    med = joined[col].median()
                    fill = 0.0 if not np.isfinite(med) else float(med)
                    joined[col] = joined[col].fillna(fill)
            else:
                joined[col] = 0.0
        n_after = len(joined)
        if n_after < 150:
            continue
        # Sample weight: down-weight low ATR regime bars
        w = 1.0 + (joined["atr_pct"].clip(0.1, 3.0) / 3.0)
        all_X.append(joined[FEATURE_COLUMNS])
        all_y.append(joined["y"].astype(int))
        all_regime.append(joined["regime"])
        all_w.append(w)

    if not all_X:
        return {
            "models": {},
            "sym_id_map": sym_id_map,
            "metrics": {},
            "regime_thresholds": thresholds,
        }

    X_all = pd.concat(all_X, ignore_index=False)
    y_all = pd.concat(all_y, ignore_index=False)
    r_all = pd.concat(all_regime, ignore_index=False)
    w_all = pd.concat(all_w, ignore_index=False)

    # Keep original time ordering so PurgedKFold is meaningful.
    order = np.argsort(X_all.index.values)
    X_all = X_all.iloc[order].reset_index(drop=True)
    y_all = y_all.iloc[order].reset_index(drop=True)
    r_all = r_all.iloc[order].reset_index(drop=True)
    w_all = w_all.iloc[order].reset_index(drop=True)

    models: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {
        "per_regime": {},
        "total_rows": int(len(X_all)),
    }
    # Minimum samples to train a regime-specific model. With 5 symbols *
    # 10_000 bars we get ~50k rows, so even the least common regime
    # usually has several thousand. 150 is low but safe.
    min_regime_samples = 150
    for regime in REGIMES:
        mask = (r_all == regime).values
        if mask.sum() < min_regime_samples:
            metrics["per_regime"][regime] = {
                "skipped": True,
                "n": int(mask.sum()),
            }
            continue
        X_r = X_all.loc[mask].reset_index(drop=True)
        y_r = y_all.loc[mask].reset_index(drop=True)
        w_r = w_all.loc[mask].reset_index(drop=True)
        model, met = train_lgbm_with_cv(
            X_r,
            y_r,
            sample_weight=w_r,
            n_splits=n_splits,
            embargo=embargo,
        )
        models[regime] = model
        metrics["per_regime"][regime] = {**met, "n": int(mask.sum())}

    return {
        "models": models,
        "sym_id_map": sym_id_map,
        "metrics": metrics,
        "regime_thresholds": thresholds,
        # Expose the final pooled corpus (X, y) so callers can warm-
        # start the OnlineLearner from the same labels the frozen
        # LightGBM just trained on. Without this, every sweep wasted
        # its first ~30 trades in pure warmup.
        "warm_corpus": {
            "X": X_all,
            "y": y_all,
        },
    }
