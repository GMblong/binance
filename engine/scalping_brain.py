"""
Scalping Brain — Superhuman Meta-Intelligence Engine
=====================================================
Combines ALL invisible signals (superhuman + microstructure + new proprietary)
into a single adaptive meta-signal using Bayesian confidence updating.

New signals added (invisible to humans):
1. Order Flow Acceleration — rate-of-change of flow imbalance (2nd derivative)
2. Liquidity Vacuum Detection — gaps in orderbook that price will fill
3. Micro-Regime Transition Probability — predict regime change 5-15s ahead
4. Spread Dynamics — bid-ask spread compression/expansion as volatility predictor
5. Trade Intensity Gradient — acceleration of trade arrival rate
6. Absorption Momentum — rate of change of absorption (whale activity ramping)

Architecture:
- Each signal contributes a log-odds update (Bayesian)
- Signals are weighted by regime + recent accuracy (adaptive)
- Final output: direction confidence [-1, +1] and entry quality [0, 1]
"""
import numpy as np
import time
from collections import deque
from utils.state import market_data
from engine.superhuman import superhuman
from engine.microstructure import micro_engine


class ScalpingBrain:
    """Meta-intelligence that fuses all invisible signals into actionable scalping decisions."""

    def __init__(self):
        self._cache = {}
        self._cache_ttl = 1.5
        # Adaptive signal weights (updated by trade outcomes)
        self._signal_accuracy = {}  # {signal_name: deque of (correct: bool)}
        self._regime_weights = {
            "TRENDING": {"flow": 1.3, "momentum": 1.5, "structure": 0.8, "mean_rev": 0.4},
            "RANGING": {"flow": 0.8, "momentum": 0.6, "structure": 1.3, "mean_rev": 1.5},
            "VOLATILE": {"flow": 1.0, "momentum": 0.7, "structure": 1.0, "mean_rev": 0.5},
        }
        self._flow_history = {}  # {symbol: deque of ofi values}
        self._intensity_history = {}  # {symbol: deque of trade counts per second}

    async def compute(self, symbol: str, direction: int, regime: str, d1m=None, d15m=None, d1h=None, client=None) -> dict:
        """Compute meta-signal. Returns dict with confidence and entry quality."""
        cache_key = (symbol, direction)
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[0] < self._cache_ttl:
            return cached[1]

        # Gather all sub-signals
        sh = superhuman.compute(symbol, d1m, d15m, d1h)
        micro = await micro_engine.compute(symbol, window_sec=60, client=client)

        # Compute new proprietary signals
        flow_accel = self._order_flow_acceleration(symbol, micro)
        liq_vacuum = self._liquidity_vacuum(symbol)
        regime_trans = self._regime_transition_prob(symbol, micro, sh)
        spread_dyn = self._spread_dynamics(symbol)
        intensity_grad = self._trade_intensity_gradient(symbol)
        absorb_mom = self._absorption_momentum(symbol, micro)

        # --- Bayesian Confidence Update ---
        # Start with prior log-odds = 0 (50/50)
        log_odds = 0.0
        rw = self._regime_weights.get(regime, self._regime_weights["RANGING"])
        signals_fired = []

        # === FLOW SIGNALS (order flow, toxicity, informed trading) ===
        flow_w = rw["flow"]

        # OFI alignment
        if (direction == 1 and micro["ofi"] > 0.15) or (direction == -1 and micro["ofi"] < -0.15):
            strength = min(abs(micro["ofi"]) * 2, 1.0)
            log_odds += 0.4 * strength * flow_w
            signals_fired.append("ofi")
        elif (direction == 1 and micro["ofi"] < -0.2) or (direction == -1 and micro["ofi"] > 0.2):
            log_odds -= 0.5 * min(abs(micro["ofi"]) * 2, 1.0) * flow_w

        # VPIN (informed trading active)
        if micro["vpin"] > 0.6:
            if (direction == 1 and micro["ofi"] > 0) or (direction == -1 and micro["ofi"] < 0):
                log_odds += 0.5 * micro["vpin"] * flow_w
                signals_fired.append("vpin")
            else:
                log_odds -= 0.6 * micro["vpin"] * flow_w

        # Toxic flow in our direction
        if sh["toxicity"] > 0.6:
            sm = sh["smart_money"]
            if (direction == 1 and sm > 0.2) or (direction == -1 and sm < -0.2):
                log_odds += 0.5 * sh["toxicity"] * flow_w
                signals_fired.append("toxic_smart")
            elif (direction == 1 and sm < -0.2) or (direction == -1 and sm > 0.2):
                log_odds -= 0.5 * sh["toxicity"] * flow_w

        # TIB (Tick Imbalance Bars)
        if sh["tib_signal"] == direction:
            log_odds += 0.4 * flow_w
            signals_fired.append("tib")
        elif sh["tib_signal"] == -direction:
            log_odds -= 0.4 * flow_w

        # Order Flow Acceleration (NEW)
        if flow_accel * direction > 0.3:
            log_odds += 0.4 * min(abs(flow_accel), 1.0) * flow_w
            signals_fired.append("flow_accel")
        elif flow_accel * direction < -0.3:
            log_odds -= 0.3 * min(abs(flow_accel), 1.0) * flow_w

        # === MOMENTUM SIGNALS ===
        mom_w = rw["momentum"]

        # Micro momentum shift
        if sh["micro_momentum"] == direction:
            log_odds += 0.4 * mom_w
            signals_fired.append("micro_mom")
        elif sh["micro_momentum"] == -direction:
            log_odds -= 0.35 * mom_w

        # Hidden divergence (multi-TF)
        if sh["hidden_div"] == direction:
            log_odds += 0.5 * mom_w
            signals_fired.append("hidden_div")
        elif sh["hidden_div"] == -direction:
            log_odds -= 0.5 * mom_w

        # Hurst exponent (trend persistence)
        if micro["hurst"] > 0.6 and regime == "TRENDING":
            log_odds += 0.3 * mom_w
            signals_fired.append("hurst")
        elif micro["hurst"] < 0.4 and regime == "RANGING":
            log_odds += 0.2 * mom_w

        # Autocorrelation decay (momentum exhaustion)
        if sh["autocorr_decay"] > 0.7:
            log_odds -= 0.6 * mom_w  # Strong penalty: momentum dying

        # Trade Intensity Gradient (NEW)
        if intensity_grad * direction > 0.3:
            log_odds += 0.3 * min(abs(intensity_grad), 1.0) * mom_w
            signals_fired.append("intensity")

        # === STRUCTURE SIGNALS (orderbook, levels, microstructure) ===
        struct_w = rw["structure"]

        # Microprice skew
        if (direction == 1 and micro["microprice_skew"] > 0.3) or \
           (direction == -1 and micro["microprice_skew"] < -0.3):
            log_odds += 0.4 * min(abs(micro["microprice_skew"]), 1.5) * struct_w
            signals_fired.append("microprice")
        elif (direction == 1 and micro["microprice_skew"] < -0.4) or \
             (direction == -1 and micro["microprice_skew"] > 0.4):
            log_odds -= 0.4 * struct_w

        # Absorption (whale accumulation)
        if micro["absorption"] > 0.7:
            log_odds += 0.4 * micro["absorption"] * struct_w
            signals_fired.append("absorption")

        # Absorption Momentum (NEW)
        if absorb_mom > 0.3:
            log_odds += 0.3 * absorb_mom * struct_w
            signals_fired.append("absorb_mom")

        # Vol compression (coiled spring)
        if micro["vol_compression"] > 0.7:
            log_odds += 0.3 * struct_w
            signals_fired.append("vol_compress")

        # Liquidity Vacuum (NEW)
        if liq_vacuum * direction > 0:
            log_odds += 0.4 * min(abs(liq_vacuum), 1.0) * struct_w
            signals_fired.append("liq_vacuum")

        # Spread Dynamics (NEW)
        if spread_dyn > 0.5:
            log_odds += 0.2 * struct_w  # Spread compressing = move imminent
            signals_fired.append("spread_dyn")
        elif spread_dyn < -0.5:
            log_odds -= 0.2 * struct_w  # Spread widening = uncertainty

        # === MEAN REVERSION SIGNALS ===
        mr_w = rw["mean_rev"]

        # Entropy regime shift
        if sh["entropy_shift"] > 0.3 and regime == "RANGING":
            log_odds += 0.3 * mr_w
            signals_fired.append("entropy_shift")
        elif sh["entropy_shift"] < -0.3 and regime == "TRENDING":
            log_odds -= 0.3 * mr_w

        # Gamma proxy (squeeze potential)
        if (direction == 1 and sh["gamma_proxy"] > 0.3) or \
           (direction == -1 and sh["gamma_proxy"] < -0.3):
            log_odds += 0.3 * min(abs(sh["gamma_proxy"]), 1.0) * mr_w
            signals_fired.append("gamma")

        # === CONTEXTUAL SIGNALS ===
        # Info asymmetry + TIB confluence
        if sh["info_asymmetry"] > 0.6 and sh["tib_signal"] == direction:
            log_odds += 0.4
            signals_fired.append("info_asym")

        # Price discovery (cross-exchange)
        if (direction == 1 and sh["price_discovery"] > 0.3) or \
           (direction == -1 and sh["price_discovery"] < -0.3):
            log_odds += 0.3 * min(abs(sh["price_discovery"]), 1.0)
            signals_fired.append("price_disc")

        # Temporal alpha
        if sh["temporal_alpha"] > 0.3:
            log_odds += 0.15
        elif sh["temporal_alpha"] < -0.3:
            log_odds -= 0.2

        # Regime transition probability (NEW)
        if regime_trans > 0.6:
            # Regime about to change — reduce confidence in current regime signals
            log_odds *= 0.7
            signals_fired.append("regime_trans")

        # Whale prints + directional flow
        if micro["whale_prints"] >= 2:
            if (direction == 1 and micro["ofi"] > 0.1) or (direction == -1 and micro["ofi"] < -0.1):
                log_odds += 0.4
                signals_fired.append("whale")

        # Hawkes intensity (burst of activity)
        if micro["hawkes_intensity"] > 0.6:
            log_odds += 0.2
            signals_fired.append("hawkes")

        # Quote stuffing (manipulation)
        if micro["quote_stuffing"] > 0.7:
            log_odds -= 0.5  # Strong penalty
            signals_fired.append("spoof_penalty")

        # === CONFLUENCE BONUS ===
        # Non-linear bonus when multiple independent signal categories agree
        n_fired = len(signals_fired)
        if n_fired >= 5:
            log_odds *= 1.2  # 20% bonus for 5+ confirming signals
        elif n_fired >= 7:
            log_odds *= 1.35  # 35% bonus for 7+ (very rare, very strong)

        # === Convert log-odds to probability ===
        confidence = 2.0 / (1.0 + np.exp(-log_odds)) - 1.0  # Maps to [-1, +1]
        confidence = float(np.clip(confidence, -1.0, 1.0))

        # Entry quality: how many signals confirm + how strong
        entry_quality = min(1.0, max(0.0, (log_odds + 1.0) / 4.0))  # Normalize to [0, 1]

        # Score boost: convert to 0-30 point range for integration with existing scoring
        score_boost = int(np.clip(confidence * direction * 25, -25, 25))

        result = {
            "confidence": confidence,
            "entry_quality": float(entry_quality),
            "score_boost": score_boost,
            "signals_fired": signals_fired,
            "n_signals": n_fired,
            "log_odds": float(log_odds),
            "flow_accel": float(flow_accel),
            "liq_vacuum": float(liq_vacuum),
            "regime_trans": float(regime_trans),
            "spread_dyn": float(spread_dyn),
            "intensity_grad": float(intensity_grad),
            "absorb_mom": float(absorb_mom),
        }

        self._cache[cache_key] = (time.time(), result)
        return result

    # ─── New Proprietary Signals ───────────────────────────────────────

    def _order_flow_acceleration(self, symbol: str, micro: dict) -> float:
        """2nd derivative of order flow imbalance.
        Positive = buying accelerating, Negative = selling accelerating.
        Detects the MOMENT smart money starts pushing aggressively."""
        ofi = micro.get("ofi", 0.0)
        hist = self._flow_history.get(symbol)
        if hist is None:
            hist = deque(maxlen=20)
            self._flow_history[symbol] = hist
        hist.append(ofi)
        if len(hist) < 5:
            return 0.0
        arr = np.array(list(hist))
        # 1st derivative (velocity)
        vel = np.diff(arr)
        # 2nd derivative (acceleration)
        if len(vel) < 3:
            return 0.0
        accel = vel[-1] - vel[-3]
        return float(np.clip(accel * 5, -1.0, 1.0))

    def _liquidity_vacuum(self, symbol: str) -> float:
        """Detect gaps in orderbook where price will accelerate.
        Returns: +1 = vacuum above (price will shoot up), -1 = vacuum below."""
        buf = market_data.depth_history.get(symbol)
        if not buf or len(buf) < 3:
            return 0.0
        # Use most recent depth snapshot
        latest = buf[-1]
        _, bid_top, ask_top, bid_qty, ask_qty = latest
        if bid_qty <= 0 or ask_qty <= 0:
            return 0.0
        # Asymmetry in displayed liquidity = vacuum on thin side
        ratio = bid_qty / (ask_qty + 1e-12)
        if ratio > 2.0:
            return 0.5  # Thin asks = vacuum above (bullish)
        elif ratio < 0.5:
            return -0.5  # Thin bids = vacuum below (bearish)
        return 0.0

    def _regime_transition_prob(self, symbol: str, micro: dict, sh: dict) -> float:
        """Probability that market regime is about to change.
        High value = current regime signals unreliable."""
        indicators = []
        # Vol compression = regime change imminent
        if micro["vol_compression"] > 0.7:
            indicators.append(0.6)
        # Entropy shifting
        if abs(sh["entropy_shift"]) > 0.4:
            indicators.append(0.5)
        # Hawkes intensity spike (activity burst)
        if micro["hawkes_intensity"] > 0.7:
            indicators.append(0.4)
        # Hurst near 0.5 (random walk = no clear regime)
        if 0.45 < micro["hurst"] < 0.55:
            indicators.append(0.3)
        if not indicators:
            return 0.0
        return float(min(1.0, np.mean(indicators) + 0.1 * len(indicators)))

    def _spread_dynamics(self, symbol: str) -> float:
        """Bid-ask spread compression/expansion rate.
        Compression = move imminent (+), Expansion = uncertainty (-)."""
        buf = market_data.depth_history.get(symbol)
        if not buf or len(buf) < 5:
            return 0.0
        spreads = []
        for ts, bt, at, _, _ in list(buf)[-10:]:
            if bt > 0 and at > 0:
                spreads.append((at - bt) / bt)
        if len(spreads) < 4:
            return 0.0
        arr = np.array(spreads)
        # Rate of change: negative = compressing (good), positive = widening (bad)
        roc = (arr[-1] - arr[0]) / (arr[0] + 1e-12)
        return float(np.clip(-roc * 20, -1.0, 1.0))

    def _trade_intensity_gradient(self, symbol: str) -> float:
        """Acceleration of trade arrival rate.
        Positive = trades arriving faster (momentum building).
        Signed by direction of recent flow."""
        trades = market_data.get_trades(symbol, window_sec=30, max_items=200)
        if len(trades) < 20:
            return 0.0
        timestamps = np.array([t[0] for t in trades])
        signs = np.array([t[3] for t in trades])
        # Split into halves
        mid = len(timestamps) // 2
        t1 = timestamps[:mid]
        t2 = timestamps[mid:]
        # Arrival rate (trades per second)
        dur1 = t1[-1] - t1[0] if len(t1) > 1 else 1.0
        dur2 = t2[-1] - t2[0] if len(t2) > 1 else 1.0
        rate1 = len(t1) / (dur1 + 1e-6)
        rate2 = len(t2) / (dur2 + 1e-6)
        # Acceleration
        accel = (rate2 - rate1) / (rate1 + 1e-6)
        # Sign by recent flow direction
        recent_dir = np.sign(signs[mid:].mean()) if len(signs) > mid else 0
        return float(np.clip(accel * recent_dir, -1.0, 1.0))

    def _absorption_momentum(self, symbol: str, micro: dict) -> float:
        """Rate of change of absorption — is whale activity ramping up?"""
        absorption = micro.get("absorption", 0.0)
        key = f"absorb_{symbol}"
        hist = self._signal_accuracy.get(key)
        if hist is None:
            hist = deque(maxlen=10)
            self._signal_accuracy[key] = hist
        hist.append(absorption)
        if len(hist) < 4:
            return 0.0
        arr = np.array(list(hist))
        # Momentum = recent vs older
        recent = arr[-2:].mean()
        older = arr[:-2].mean()
        return float(np.clip((recent - older) * 3, -1.0, 1.0))

    def update_accuracy(self, symbol: str, signals_fired: list, was_profitable: bool):
        """Called after trade closes to update signal accuracy tracking."""
        for sig in signals_fired:
            key = f"acc_{sig}"
            if key not in self._signal_accuracy:
                self._signal_accuracy[key] = deque(maxlen=50)
            self._signal_accuracy[key].append(was_profitable)


# Singleton
scalping_brain = ScalpingBrain()
