"""Probability fusion for Phase 4.

Given three probabilities in [0,1]:
  - P_tech: from the technical score (converted from 0..100 -> 0..1)
  - P_ml  : from the regime-conditional V2 model
  - P_flow: from microstructure flow aggregate (orderbook imbalance,
            CVD divergence, funding crowding, liquidation cascade)

We combine them in logit space:

  fused_logit = w_t * logit(P_tech) + w_m * logit(P_ml) + w_f * logit(P_flow)
  P_trade     = sigmoid(fused_logit / w_sum)

The default weights are conservative: ML is the strongest single input
(weight 1.0), tech score is 0.6, flow is 0.5. Live tuning can override
these via `set_weights`.
"""

from __future__ import annotations

import math
from typing import Tuple, Optional


_EPS = 1e-6


def logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    # Numerically-stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


DEFAULT_WEIGHTS = {"tech": 0.6, "ml": 1.0, "flow": 0.5}


def fuse_probabilities(
    p_tech: float,
    p_ml: float,
    p_flow: float,
    weights: Optional[dict] = None,
) -> Tuple[float, float]:
    """Return (P_trade, fused_logit).

    weights: optional dict with keys tech/ml/flow.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})
    wsum = max(_EPS, w["tech"] + w["ml"] + w["flow"])
    fused = (
        w["tech"] * logit(p_tech)
        + w["ml"] * logit(p_ml)
        + w["flow"] * logit(p_flow)
    ) / wsum
    return sigmoid(fused), fused


def score_to_prob(score_0_100: float, direction: int) -> float:
    """Map a 0..100 technical score to a directional probability.

    Score is directionless (how strong the setup is). A score=75 with
    direction=+1 should produce ~0.62 P(UP); score=75 with direction=-1
    should produce ~0.38.
    """
    base = min(0.95, max(0.05, score_0_100 / 100.0))
    # Neutral score = 50 -> p=0.5; scaling factor compresses extremes
    centered = 0.5 + (base - 0.5) * 0.5
    if direction >= 0:
        return centered
    return 1.0 - centered


def flow_prob(
    ob_imbalance: float = 0.0,
    cvd_div: int = 0,
    cvd_abs: int = 0,
    funding_sig: int = 0,
    liq_sig: int = 0,
    beta_gap: int = 0,
    direction: int = 1,
) -> float:
    """Aggregate microstructure signals to a single directional P(UP).

    Each input contributes a centered logit offset; the magnitudes are
    tuned to keep P_flow in [0.2, 0.8] under typical conditions.
    """
    acc = 0.0
    acc += 1.2 * float(ob_imbalance)
    acc += 0.6 * float(cvd_div)
    acc += 0.8 * float(cvd_abs)
    acc += 0.5 * float(funding_sig)
    acc += 0.7 * float(liq_sig)
    acc += 0.6 * float(beta_gap)
    # Direction hint: if direction is -1, flip the axis so the same acc
    # reads P(DIR) symmetrically on the short side.
    if direction < 0:
        acc = -acc
    p_dir = sigmoid(acc)
    return p_dir if direction >= 0 else 1.0 - p_dir
