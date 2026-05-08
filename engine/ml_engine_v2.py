"""
ML Engine v2
------------
Key fix vs ml_engine.py: training labels are produced by the EXACT same
exit rule that the live strategy (scalper_pro) uses. Without alignment,
the probability output is uncorrelated with realised outcome and misleads
the signal gate.

Label generation:
  For each historical candle i, simulate a hypothetical LONG entry at close[i]
  with SL = close[i] - R and TP1 = close[i] + R, where R = 1.1 * ATR(14, 1m).
  Walk forward up to `lookahead` candles. Label = 1 if TP1 is touched before
  SL, else 0 (SL hit or timeout).

  Since exits are symmetric in the live model, P(LONG hit TP1) and P(SHORT
  hit TP1) are related by market drift. We only train one directional model
  and use 1-p for the short side. This is approximate but empirically
  monotonic with true short-side probability.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["h"], df["l"], df["c"]
    tr = pd.concat([(h - l).abs(), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def _features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c, h, l, v = df["c"], df["h"], df["l"], df["v"]
    out["ret_1"] = c.pct_change(1)
    out["ret_5"] = c.pct_change(5)
    out["ret_15"] = c.pct_change(15)
    out["ema9_ratio"] = c / c.ewm(span=9, adjust=False).mean() - 1
    out["ema21_ratio"] = c / c.ewm(span=21, adjust=False).mean() - 1
    out["ema50_ratio"] = c / c.ewm(span=50, adjust=False).mean() - 1
    out["atr_pct"] = _atr(df, 14) / c
    out["range_pct"] = (h - l) / c
    out["body_pct"] = (c - df["o"]).abs() / c
    # Volume features
    vol_mean = v.rolling(20).mean()
    out["vol_z"] = (v - vol_mean) / v.rolling(20).std().replace(0, np.nan)
    out["vol_ratio"] = v / vol_mean.replace(0, np.nan)
    # Momentum
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - 100 / (1 + rs)
    # Position relative to recent extremes
    out["pos_in_range_20"] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min()).replace(0, np.nan)
    out["pos_in_range_60"] = (c - l.rolling(60).min()) / (h.rolling(60).max() - l.rolling(60).min()).replace(0, np.nan)
    # Wick proxies
    rng = (h - l).replace(0, np.nan)
    out["upper_wick"] = (h - df[["o", "c"]].max(axis=1)) / rng
    out["lower_wick"] = (df[["o", "c"]].min(axis=1) - l) / rng
    return out.replace([np.inf, -np.inf], np.nan)


def _label_tp_before_sl(df: pd.DataFrame, r_mult: float = 1.1, lookahead: int = 12) -> pd.Series:
    """Label: 1 if TP1 (entry + R) hit before SL (entry - R) within `lookahead`."""
    a = _atr(df, 14).values
    c = df["c"].values
    h = df["h"].values
    l = df["l"].values
    n = len(df)
    out = np.full(n, np.nan)
    for i in range(n - lookahead):
        if np.isnan(a[i]):
            continue
        r = r_mult * a[i]
        tp = c[i] + r
        sl = c[i] - r
        for j in range(1, lookahead + 1):
            if h[i + j] >= tp:
                out[i] = 1.0
                break
            if l[i + j] <= sl:
                out[i] = 0.0
                break
        else:
            out[i] = 0.0  # timeout counts as a miss
    return pd.Series(out, index=df.index)


class MLPredictorV2:
    """Per-symbol calibrated classifier. Safe to use without LightGBM installed
    (falls back to a trivial 0.5 estimator)."""

    def __init__(self):
        self.models: dict = {}
        self.last_trained: dict = {}
        self.feature_names: dict = {}

    def train(self, df_1m: pd.DataFrame, symbol: str,
              r_mult: float = 1.1, lookahead: int = 12, min_rows: int = 600) -> bool:
        if not _HAS_LGBM:
            return False
        if df_1m is None or len(df_1m) < min_rows:
            return False
        feats = _features(df_1m)
        y = _label_tp_before_sl(df_1m, r_mult=r_mult, lookahead=lookahead)
        data = feats.join(y.rename("y")).dropna()
        if len(data) < 400:
            return False

        feat_cols = [c for c in data.columns if c != "y"]
        X = data[feat_cols].values
        yv = data["y"].values.astype(int)

        # Time-based split: no shuffle, train on the first 85%.
        X_tr, X_te, y_tr, y_te = train_test_split(X, yv, test_size=0.15, shuffle=False)
        base = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.04, max_depth=5, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            n_jobs=2, random_state=42, verbose=-1,
        )
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        model.fit(X_tr, y_tr)
        self.models[symbol] = model
        self.feature_names[symbol] = feat_cols
        self.last_trained[symbol] = time.time()
        return True

    def predict(self, df_1m: pd.DataFrame, symbol: str) -> float:
        model = self.models.get(symbol)
        if model is None:
            return 0.5
        feats = _features(df_1m.tail(300)).dropna()
        if feats.empty:
            return 0.5
        feat_cols = self.feature_names[symbol]
        x = feats[feat_cols].iloc[[-1]].values
        try:
            return float(model.predict_proba(x)[0][1])
        except Exception:
            return 0.5


ml_v2 = MLPredictorV2()
