"""ML v2 (Phase 4): regime-conditional, walk-forward, fused model.

Replaces the single per-symbol LightGBM in engine/ml_engine.py with:

- Triple-barrier labels (asymmetric PT/SL, explicit timeout class)
- Expanded features: BTC lags, time-of-day, cross-section rank,
  microstructure signals (orderbook imbalance, CVD, funding skew,
  BTC beta-gap).
- Purged walk-forward splits (no calibration leakage).
- Pooled training across symbols with a symbol embedding so low-sample
  alts benefit from BTC/ETH history.
- Regime-conditional inference (separate models per TRENDING / VOLATILE /
  RANGING).
- Probability fusion: final P_trade = sigmoid(w_t*logit(P_tech) +
  w_m*logit(P_ml) + w_f*logit(P_flow)).

See `trainer.train_pooled`, `predictor.V2Predictor`, and
`fusion.fuse_probabilities` for entry points.
"""

from .labeling import triple_barrier_labels
from .features import build_feature_matrix, FEATURE_COLUMNS
from .models import train_lgbm_with_cv, PurgedKFold
from .trainer import train_pooled
from .predictor import V2Predictor
from .fusion import fuse_probabilities, logit, sigmoid

__all__ = [
    "triple_barrier_labels",
    "build_feature_matrix",
    "FEATURE_COLUMNS",
    "train_lgbm_with_cv",
    "PurgedKFold",
    "train_pooled",
    "V2Predictor",
    "fuse_probabilities",
    "logit",
    "sigmoid",
]
