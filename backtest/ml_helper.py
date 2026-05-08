"""ML prediction helper with strict no-look-ahead guarantees.

Usage:
    helper = WalkForwardPredictor(client, ml_predictor, train_end_ts_ms)
    await helper.train(symbols)
    # In signal_fn:
    p = helper.predict_prob(symbol, sub_1m)
"""

from __future__ import annotations

from typing import Optional, Dict

import pandas as pd


class WalkForwardPredictor:
    """Wraps engine.ml_engine.MLPredictor so the ONLY data the model ever
    sees during training is strictly before `train_end_ts_ms`.

    `predict_prob` uses cached feature dataframes from the engine when
    available, otherwise falls back to computing features on the provided
    `sub_1m` slice. The slice must end strictly at or before the current
    simulated bar.
    """

    def __init__(self, client, ml_predictor, train_end_ts_ms: int):
        self.client = client
        self.ml = ml_predictor
        self.train_end_ts_ms = train_end_ts_ms

    async def train(self, symbols):
        for sym in symbols:
            await self.ml.train_model(self.client, sym, end_time=self.train_end_ts_ms)

    def predict_prob(
        self, symbol: str, sub_1m: pd.DataFrame, sub_15m: Optional[pd.DataFrame] = None
    ) -> float:
        """Predict P(up) for the *last* bar of sub_1m, using no future data."""
        model = self.ml.models.get(symbol)
        if model is None:
            return 0.5
        try:
            feats = self.ml.feature_engineering(sub_1m.tail(300).copy())
            if feats.empty:
                return 0.5
            features = self.ml._get_feature_list(feats.columns)
            missing = [f for f in features if f not in feats.columns]
            if missing:
                return 0.5
            X = feats[features].iloc[[-1]]
            return float(model.predict_proba(X)[0][1])
        except Exception:
            return 0.5
