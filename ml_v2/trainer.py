"""Pooled trainer across symbols.

Builds one feature matrix per symbol, stacks them with a `sym_id`
feature that acts as a categorical embedding, then trains one LightGBM
per regime. This matters because:

- Low-sample alts benefit from BTC/ETH history shape.
- A single pooled model is easier to deploy + calibrate than N per-symbol
  models.
- The symbol embedding lets the model still learn symbol-specific
  quirks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .labeling import triple_barrier_labels, collapse_to_binary
from .features import build_feature_matrix, FEATURE_COLUMNS
from .models import train_lgbm_with_cv

REGIMES = ("TRENDING", "VOLATILE", "RANGING")


def _regime_of(atr_pct: float, dist_ema21: float) -> str:
    """Regime bucketing for routing pooled model inference.

    Thresholds deliberately looser than the v1 defaults: observed in real
    training runs that atr>1.5 & dist>0.5% only captured ~2k samples of
    TRENDING out of 50k, starving the model. New bands push more bars
    into TRENDING so LightGBM can learn a usable edge there.

      VOLATILE = atr_pct >= 1.2%                  (actual chop/news spike)
      TRENDING = atr_pct >= 0.20% AND |dist|>=0.15% (persistent directional)
      RANGING  = everything else
    """
    if not np.isfinite(atr_pct) or not np.isfinite(dist_ema21):
        return "RANGING"
    if atr_pct > 1.2:
        return "VOLATILE"
    if atr_pct > 0.20 and abs(dist_ema21) > 0.0015:
        return "TRENDING"
    return "RANGING"


def assign_regime(features: pd.DataFrame) -> pd.Series:
    """Coarse per-bar regime label from cheap features present in
    `features`. We don't train a regime classifier -- this is intentional:
    the "regime" is just a routing key so each sub-model specializes."""
    atr = features.get("atr_pct")
    dist = features.get("dist_ema21")
    if atr is None or dist is None:
        return pd.Series(["RANGING"] * len(features), index=features.index)
    return pd.Series([_regime_of(a, d) for a, d in zip(atr, dist)],
                     index=features.index)


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
) -> Dict[str, Any]:
    """Train one model per regime on the pooled corpus.

    `horizon` default is 30 minutes (was 15). Rationale: in live runs the
    EventDrivenBacktester sees average hold = 100-130 bars; the 15m
    triple-barrier was therefore labelling too much chop as TIMEOUT and
    starving the directional classes. 30m aligns the training label
    horizon closer to actual exit distributions.

    Returns a dict: `{"models": {regime: model}, "sym_id_map": {...},
    "metrics": {...}}`.
    """
    funding_per_symbol = funding_per_symbol or {}
    sym_id_map = {s: i + 1 for i, s in enumerate(sorted(per_symbol.keys()))}

    all_X: List[pd.DataFrame] = []
    all_y: List[pd.Series] = []
    all_regime: List[pd.Series] = []
    all_w: List[pd.Series] = []

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
        regime = assign_regime(feats)
        n_before = len(feats)
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
        return {"models": {}, "sym_id_map": sym_id_map, "metrics": {}}

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
    metrics: Dict[str, Any] = {"per_regime": {}, "total_rows": int(len(X_all))}
    # Minimum samples to train a regime-specific model. With 5 symbols *
    # 10_000 bars we get ~50k rows, so even the least common regime
    # usually has several thousand. 150 is low but safe.
    min_regime_samples = 150
    for regime in REGIMES:
        mask = (r_all == regime).values
        if mask.sum() < min_regime_samples:
            metrics["per_regime"][regime] = {"skipped": True, "n": int(mask.sum())}
            continue
        X_r = X_all.loc[mask].reset_index(drop=True)
        y_r = y_all.loc[mask].reset_index(drop=True)
        w_r = w_all.loc[mask].reset_index(drop=True)
        model, met = train_lgbm_with_cv(
            X_r, y_r, sample_weight=w_r,
            n_splits=n_splits, embargo=embargo,
        )
        models[regime] = model
        metrics["per_regime"][regime] = {**met, "n": int(mask.sum())}

    return {"models": models, "sym_id_map": sym_id_map, "metrics": metrics}
