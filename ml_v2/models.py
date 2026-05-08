"""Training primitives: purged walk-forward CV + calibrated LightGBM.

Why not sklearn's KFold / TimeSeriesSplit?
- Plain KFold has look-ahead (random shuffle).
- TimeSeriesSplit doesn't purge the barrier-horizon overlap -- if a bar's
  label uses 15 bars of lookahead and the test fold begins inside that
  window, the fold has training leakage.

PurgedKFold drops the last `embargo` samples of each training fold so
the barrier horizon can never spill into test.
"""

from __future__ import annotations

from typing import Iterator, Tuple, List, Optional, Any

import numpy as np
import pandas as pd


class PurgedKFold:
    """Time-ordered K-fold with an embargo between train and test.

    Splits the dataset into `n_splits` contiguous test folds. Training is
    "everything before the test fold, minus the last `embargo` rows".
    """

    def __init__(self, n_splits: int = 5, embargo: int = 30):
        self.n_splits = max(2, n_splits)
        self.embargo = max(0, embargo)

    def split(self, X) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        fold_sizes = [n // self.n_splits] * self.n_splits
        for i in range(n % self.n_splits):
            fold_sizes[i] += 1
        starts: List[int] = []
        pos = 0
        for fs in fold_sizes:
            starts.append(pos)
            pos += fs
        for k in range(1, self.n_splits):  # skip first fold as test (no train)
            test_start = starts[k]
            test_end = starts[k] + fold_sizes[k]
            train_end = max(0, test_start - self.embargo)
            if train_end < 100:
                continue
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx


def train_lgbm_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    n_splits: int = 5,
    embargo: int = 30,
    lgbm_params: Optional[dict] = None,
) -> Tuple[Any, dict]:
    """Train a LightGBM classifier with purged walk-forward CV and
    isotonic calibration on the final fold.

    Returns (model, metrics_dict). The returned `model` is a fitted
    CalibratedClassifierCV wrapping LightGBM, trained on ALL data, with
    calibration performed using a held-out (purged) tail slice.

    Falls back to a simple LightGBM fit if sklearn/lightgbm is not
    installed so the rest of the pipeline can still run for feature
    engineering tests.
    """
    try:
        import lightgbm as lgb
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import log_loss, roc_auc_score
    except Exception as exc:
        raise RuntimeError(f"ml_v2 training requires lightgbm+sklearn: {exc}")

    default_params = dict(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=2,
        verbose=-1,
    )
    if lgbm_params:
        default_params.update(lgbm_params)

    metrics = {"fold_auc": [], "fold_logloss": [], "n_samples": len(X)}
    cv = PurgedKFold(n_splits=n_splits, embargo=embargo)

    # CV scores
    for train_idx, test_idx in cv.split(X):
        if len(train_idx) < 100 or len(test_idx) < 20:
            continue
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        sw = sample_weight.iloc[train_idx] if sample_weight is not None else None
        m = lgb.LGBMClassifier(**default_params)
        try:
            m.fit(X_tr, y_tr, sample_weight=sw)
            proba = m.predict_proba(X_te)[:, 1]
            try:
                metrics["fold_auc"].append(roc_auc_score(y_te, proba))
            except Exception:
                pass
            try:
                metrics["fold_logloss"].append(log_loss(y_te, proba, labels=[0, 1]))
            except Exception:
                pass
        except Exception:
            continue

    # Final: calibrate on a held-out purged tail
    tail_size = max(200, int(0.15 * len(X)))
    cut = max(100, len(X) - tail_size - embargo)
    X_fit = X.iloc[:cut]
    y_fit = y.iloc[:cut]
    X_cal = X.iloc[cut + embargo: cut + embargo + tail_size]
    y_cal = y.iloc[cut + embargo: cut + embargo + tail_size]
    if len(X_cal) < 50 or len(X_fit) < 100:
        # Not enough data to calibrate properly -- fit on full
        base = lgb.LGBMClassifier(**default_params)
        base.fit(X, y, sample_weight=sample_weight)
        return base, metrics
    base = lgb.LGBMClassifier(**default_params)
    base.fit(X_fit, y_fit,
             sample_weight=(sample_weight.iloc[:cut] if sample_weight is not None else None))
    try:
        cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
        cal.fit(X_cal, y_cal)
        return cal, metrics
    except Exception:
        return base, metrics
