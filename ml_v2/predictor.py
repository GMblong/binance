"""Inference wrapper for the pooled regime-conditional model.

Usage:
    predictor = V2Predictor(bundle)
    p_up = predictor.predict_prob(
        symbol="BTCUSDT",
        df_1m=sub_df,
        btc_close=btc_close,
        funding_series=funding,
        cross_section=cs,
    )
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from .features import build_feature_matrix, FEATURE_COLUMNS
from .trainer import assign_regime


class V2Predictor:
    def __init__(self, bundle: Dict[str, Any]):
        self.models: Dict[str, Any] = bundle.get("models", {}) or {}
        self.sym_id_map: Dict[str, int] = bundle.get("sym_id_map", {}) or {}
        self.metrics: Dict[str, Any] = bundle.get("metrics", {}) or {}
        # Extract per-regime AUC so callers can weight ML dynamically.
        self.regime_auc: Dict[str, float] = {}
        for reg, m in self.metrics.get("per_regime", {}).items():
            aucs = (m or {}).get("fold_auc", [])
            if aucs:
                self.regime_auc[reg] = float(sum(aucs) / len(aucs))

    def is_ready(self) -> bool:
        return bool(self.models)

    def best_auc(self) -> float:
        """Best CV-fold AUC across regimes (useful for fusion weight)."""
        if not self.regime_auc:
            return 0.5
        return max(self.regime_auc.values())

    def last_regime_auc(self, regime: str) -> float:
        """AUC for a specific regime (0.5 if not available)."""
        return float(self.regime_auc.get(regime, 0.5))

    def predict_prob(
        self,
        symbol: str,
        df_1m: pd.DataFrame,
        btc_close: Optional[pd.Series] = None,
        funding_series: Optional[pd.Series] = None,
        cross_section: Optional[pd.DataFrame] = None,
    ) -> float:
        """Return calibrated P(UP within horizon) for the LAST bar of
        df_1m. Returns 0.5 if no suitable model is available."""
        if not self.models or len(df_1m) < 60:
            return 0.5
        feats = build_feature_matrix(
            df_1m.tail(400),
            btc_close=btc_close.tail(400) if btc_close is not None else None,
            funding_series=funding_series,
            cross_section=cross_section,
            symbol=symbol,
            sym_id_map=self.sym_id_map,
        )
        if feats.empty:
            return 0.5
        regime_series = assign_regime(feats)
        if regime_series.empty:
            return 0.5
        regime = str(regime_series.iloc[-1])
        model = self.models.get(regime)
        if model is None:
            # Fallback: use any trained regime model
            if not self.models:
                return 0.5
            model = next(iter(self.models.values()))
        row = feats.iloc[[-1]][FEATURE_COLUMNS].copy()
        # Replace remaining NaNs with column median from this row's context
        if row.isna().any().any():
            ctx = feats[FEATURE_COLUMNS].tail(50)
            med = ctx.median(numeric_only=True)
            row = row.fillna(med).fillna(0.0)
        try:
            p = float(model.predict_proba(row)[0][1])
            if not np.isfinite(p):
                return 0.5
            return max(0.01, min(0.99, p))
        except Exception:
            return 0.5
