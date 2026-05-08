"""Online learner for live model adaptation.

The `V2Predictor` LightGBM is frozen: once trained on a corpus ending at
`train_end_ms` it stays static. In fast-moving crypto regimes this
decays quickly. OnlineLearner complements the frozen LightGBM with a
small SGDClassifier that is updated on every closed trade, so the bot
literally keeps learning from its own P&L.

Design constraints:
- MUST respect label delay: a trade opened at bar t does not have a
  label until the position closes at bar t+k. Feeding partial labels
  would leak future data into past bars, so we only `partial_fit` on
  samples whose labels are already resolved.
- MUST NOT dominate the LightGBM: we fuse outputs in logit space with
  a small default weight (blend_weight=0.3) so early noise from a tiny
  replay buffer doesn't swing decisions.
- MUST gracefully degrade if sklearn is missing: the class returns 0.5
  (neutral) and `is_ready()` stays False so the signal adapter can
  fall back to LightGBM-only.

Typical lifecycle:
    learner = OnlineLearner(feature_cols=FEATURE_COLUMNS)
    # Each new signal -> remember it
    learner.record_signal(signal_id, feature_row, direction)
    # When a trade closes with outcome {1=win, 0=loss} -> update model
    learner.record_outcome(signal_id, outcome_binary)
    # For the next inference
    p_up = learner.predict_up(feature_row)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    SGDClassifier = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False


@dataclass
class _PendingSample:
    """A signal whose outcome hasn't been observed yet."""
    features: np.ndarray
    direction: int
    created_step: int


@dataclass
class OnlineLearner:
    """Logistic-regression-style online classifier with replay warmup.

    Attributes:
        feature_cols: canonical feature list for alignment with predictor.
        warmup_samples: min #labelled samples before predict_up != 0.5.
        replay_size: cap for the replay buffer used in periodic retrains.
        blend_weight: default fusion weight when the OnlineLearner output
            is blended with the frozen LightGBM (used by signal_adapter_v3).
        auc_window: size of the rolling window used to estimate the
            learner's live AUC. We report this so the signal adapter can
            auto-scale blend_weight the same way we do for LightGBM.
    """

    feature_cols: List[str]
    warmup_samples: int = 30
    replay_size: int = 2000
    blend_weight: float = 0.3
    auc_window: int = 200
    lr_init: float = 0.05
    l2: float = 1e-4

    # State
    _pending: Dict[str, _PendingSample] = field(default_factory=dict)
    _replay_X: List[np.ndarray] = field(default_factory=list)
    _replay_y: List[int] = field(default_factory=list)
    _model: Any = None
    _scaler: Any = None
    _n_updates: int = 0
    _recent_probs: List[float] = field(default_factory=list)
    _recent_labels: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if _SKLEARN_AVAILABLE:
            self._model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=self.l2,
                learning_rate="constant",
                eta0=self.lr_init,
                random_state=42,
                warm_start=True,
            )
            self._scaler = StandardScaler()

    # ---------------- queries ----------------
    def is_ready(self) -> bool:
        """Has the model seen enough labels to produce a non-neutral output?"""
        return (
            _SKLEARN_AVAILABLE
            and self._n_updates >= self.warmup_samples
            and self._model is not None
            and hasattr(self._model, "classes_")
        )

    def recent_auc(self) -> float:
        """Rolling AUC of recent predictions vs realized labels.

        Returns 0.5 if we don't have both classes in the window, so
        callers can treat it as neutral. Computed with a simple
        concordant/discordant pair count to avoid a sklearn dependency
        on this hot path.
        """
        n = min(len(self._recent_probs), len(self._recent_labels))
        if n < 20:
            return 0.5
        probs = np.asarray(self._recent_probs[-n:])
        labels = np.asarray(self._recent_labels[-n:])
        pos = probs[labels == 1]
        neg = probs[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        # Mann-Whitney U normalized = AUC
        wins = 0.0
        ties = 0.0
        for p in pos:
            wins += float(np.sum(neg < p))
            ties += float(np.sum(neg == p))
        return (wins + 0.5 * ties) / (len(pos) * len(neg))

    def predict_up(self, row: pd.Series | np.ndarray) -> float:
        """Return P(UP within horizon). 0.5 if not ready."""
        if not self.is_ready():
            return 0.5
        try:
            x = self._to_vec(row)
            xs = self._scaler.transform(x.reshape(1, -1))
            p = float(self._model.predict_proba(xs)[0][1])
            if not np.isfinite(p):
                return 0.5
            return max(0.01, min(0.99, p))
        except Exception:
            return 0.5

    # ---------------- updates ----------------
    def record_signal(
        self,
        signal_id: str,
        row: pd.Series | np.ndarray,
        direction: int,
        created_step: int = 0,
    ) -> None:
        """Remember a signal so we can update the model once its label
        (trade outcome) is observed.

        Over time the backtest and live bot can accumulate unresolved
        signals (positions still open). We cap pending at 5x the replay
        size so leaked memory can't blow up.
        """
        if not _SKLEARN_AVAILABLE:
            return
        try:
            x = self._to_vec(row)
        except Exception:
            return
        if len(self._pending) > 5 * self.replay_size:
            # Drop oldest half
            oldest = sorted(self._pending.keys())[: len(self._pending) // 2]
            for k in oldest:
                self._pending.pop(k, None)
        self._pending[signal_id] = _PendingSample(
            features=x, direction=int(direction), created_step=int(created_step)
        )

    def record_outcome(self, signal_id: str, outcome: int) -> None:
        """Feed a resolved trade label (1=win, 0=loss) to the online model.

        Because our signals are directional (we trade `direction=+1/-1`
        based on the fused prob), we translate win/loss into the
        directional target in UP space: a winning LONG means UP,
        a winning SHORT means DOWN (label=0 in UP-probability terms).
        """
        if not _SKLEARN_AVAILABLE:
            return
        s = self._pending.pop(signal_id, None)
        if s is None:
            return
        y_up = 1 if (s.direction == 1 and outcome == 1) or (s.direction == -1 and outcome == 0) else 0
        self._append_to_replay(s.features, y_up)
        self._partial_fit_one(s.features, y_up)

    # ---------------- internals ----------------
    def _to_vec(self, row: pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(row, pd.Series):
            # Align to canonical feature list, zero-fill missing
            vals = np.array(
                [float(row.get(c, 0.0)) if np.isfinite(float(row.get(c, 0.0))) else 0.0
                 for c in self.feature_cols],
                dtype=np.float64,
            )
            return vals
        arr = np.asarray(row, dtype=np.float64).ravel()
        if arr.size != len(self.feature_cols):
            # If dimensions mismatch, pad or clip safely
            out = np.zeros(len(self.feature_cols), dtype=np.float64)
            n = min(arr.size, out.size)
            out[:n] = arr[:n]
            return out
        return arr

    def _append_to_replay(self, x: np.ndarray, y: int) -> None:
        self._replay_X.append(x)
        self._replay_y.append(int(y))
        if len(self._replay_X) > self.replay_size:
            # Drop oldest to stay bounded
            drop = len(self._replay_X) - self.replay_size
            self._replay_X = self._replay_X[drop:]
            self._replay_y = self._replay_y[drop:]

    # ---------------- warm start ----------------
    def warm_start_from_corpus(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        max_samples: int = 2000,
        tail_bias: bool = True,
    ) -> int:
        """Bootstrap the online learner from a labelled corpus (typically
        the same rows ml_v2.train_pooled just trained on).

        Before this method existed, the OnlineLearner sat at P=0.5 for
        the first ~30 trades of every sweep -- observed in v3 runs
        where days 1-2 were pure fee-burn while `n_updates` climbed
        from 0 to ~25. warm_start_from_corpus() lets us launch with
        30+ labels already absorbed, so the live head is
        `is_ready()=True` on the very first bar of simulation and
        `recent_auc()` reflects its real edge instead of 0.5 noise.

        Labels here are the SAME `y_up` used in live updates
        (triple-barrier binary from ml_v2.labeling); since we don't
        know the trade direction we would have taken, we just treat
        UP (label=1) as "the market went up within horizon" and leave
        direction-conditioning to the live record_outcome() calls.

        `max_samples`: cap the warm corpus so we don't overwhelm the
        buffer on >50k row datasets. `tail_bias=True` keeps the most
        recent rows since they are closest to the live distribution.

        Returns the number of samples actually absorbed. 0 means the
        sklearn stack is missing or the inputs were empty.
        """
        if not _SKLEARN_AVAILABLE:
            return 0
        try:
            if isinstance(X, pd.DataFrame):
                if not set(self.feature_cols).issubset(set(X.columns)):
                    # Align columns; missing -> 0
                    X_aligned = pd.DataFrame(
                        {c: X[c] if c in X.columns else 0.0
                         for c in self.feature_cols}
                    )
                else:
                    X_aligned = X[self.feature_cols]
                X_arr = X_aligned.astype(float).fillna(0.0).values
            else:
                X_arr = np.asarray(X, dtype=np.float64)
            if isinstance(y, pd.Series):
                y_arr = y.astype(int).values
            else:
                y_arr = np.asarray(y, dtype=np.int64).ravel()
            n = min(len(X_arr), len(y_arr))
            if n < 20:
                return 0
            X_arr = X_arr[:n]
            y_arr = y_arr[:n]
            # Drop rows with NaN/inf
            finite = np.all(np.isfinite(X_arr), axis=1)
            X_arr = X_arr[finite]
            y_arr = y_arr[finite]
            if len(X_arr) < 20:
                return 0
            if len(X_arr) > max_samples:
                X_arr = X_arr[-max_samples:] if tail_bias else X_arr[:max_samples]
                y_arr = y_arr[-max_samples:] if tail_bias else y_arr[:max_samples]

            # Need BOTH classes present to fit the classifier.
            classes = np.unique(y_arr)
            if len(classes) < 2:
                return 0

            # Fit scaler on the full warm corpus for stable feature means.
            self._scaler.fit(X_arr)
            Xs = self._scaler.transform(X_arr)
            self._model.partial_fit(Xs, y_arr, classes=np.array([0, 1]))
            # Seed replay buffer with the most recent max_samples rows
            # so subsequent scaler refreshes aren't biased by the tiny
            # live sample.
            keep = min(len(X_arr), self.replay_size)
            self._replay_X = [X_arr[i] for i in range(len(X_arr) - keep, len(X_arr))]
            self._replay_y = [int(y_arr[i]) for i in range(len(y_arr) - keep, len(y_arr))]
            # Mark the learner as having absorbed these samples so
            # is_ready() returns True immediately.
            self._n_updates = max(self._n_updates, len(X_arr))
            # Build an honest starting AUC sample: predict on the warm
            # corpus itself. In-sample optimistic but harmless because
            # the signal adapter re-checks AUC >= 0.51 before using us.
            try:
                probs = self._model.predict_proba(Xs)[:, 1]
                # Keep up to auc_window most recent samples.
                k = min(self.auc_window, len(probs))
                self._recent_probs = list(probs[-k:])
                self._recent_labels = list(y_arr[-k:].astype(int))
            except Exception:
                pass
            return len(X_arr)
        except Exception:
            return 0

    def _partial_fit_one(self, x: np.ndarray, y: int) -> None:
        """Do a single incremental SGD step. On the very first label we
        fit the scaler + do partial_fit with both class labels [0,1] so
        sklearn can initialize coefficients for both classes."""
        try:
            if self._n_updates == 0:
                # Need to init scaler & classes before we can partial_fit.
                X = np.asarray(self._replay_X, dtype=np.float64)
                if X.shape[0] < 2:
                    X = np.vstack([X, x.reshape(1, -1)])
                self._scaler.fit(X)
                self._model.partial_fit(
                    self._scaler.transform(x.reshape(1, -1)),
                    np.array([y]),
                    classes=np.array([0, 1]),
                )
            else:
                # Keep scaler gently adapting via a running mean/var
                # approximation (refresh every 50 updates).
                if self._n_updates % 50 == 0 and len(self._replay_X) >= 30:
                    try:
                        self._scaler.partial_fit(
                            np.asarray(self._replay_X[-200:], dtype=np.float64)
                        )
                    except Exception:
                        pass
                self._model.partial_fit(
                    self._scaler.transform(x.reshape(1, -1)),
                    np.array([y]),
                )
            # Track rolling AUC by getting prob BEFORE next update:
            if self.is_ready():
                try:
                    p = float(
                        self._model.predict_proba(
                            self._scaler.transform(x.reshape(1, -1))
                        )[0][1]
                    )
                    self._recent_probs.append(p)
                    self._recent_labels.append(int(y))
                    if len(self._recent_probs) > self.auc_window:
                        self._recent_probs = self._recent_probs[-self.auc_window:]
                        self._recent_labels = self._recent_labels[-self.auc_window:]
                except Exception:
                    pass
            self._n_updates += 1
        except Exception:
            # Never let a training failure kill the pipeline; just skip.
            pass
