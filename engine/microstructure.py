"""
Microstructure Alpha Engine
============================
12 institutional-grade signals invisible to human traders.
Computed from tick-level trade data + orderbook snapshots.
"""
import numpy as np
import time
from collections import deque
from utils.state import market_data


class MicrostructureEngine:
    """Computes microstructure alpha features from tick data."""

    def __init__(self):
        self._cache = {}  # {symbol: (ts, features_dict)}
        self._cache_ttl = 1.5  # seconds

    def compute(self, symbol: str, window_sec: int = 60) -> dict:
        """Compute all 12 microstructure features. Returns dict or cached."""
        cached = self._cache.get(symbol)
        if cached and time.time() - cached[0] < self._cache_ttl:
            return cached[1]

        trades = market_data.get_trades(symbol, window_sec=window_sec, max_items=600)
        n = len(trades)

        result = {
            "vpin": 0.0,        # Volume-sync Probability of Informed Trading
            "kyle_lambda": 0.0, # Price impact per unit volume
            "ofi": 0.0,         # Order Flow Imbalance (normalized)
            "hurst": 0.5,       # Hurst exponent (0.5=random, >0.5=trending, <0.5=mean-rev)
            "entropy": 1.0,     # Shannon entropy of returns (1=random, <1=predictable)
            "microprice_skew": 0.0,  # Microprice vs mid deviation
            "absorption": 0.0,  # Absorption rate (hidden accumulation)
            "whale_prints": 0,  # Count of outlier trades (>3σ)
            "quote_stuffing": 0.0,  # Quote churn rate
            "hawkes_intensity": 0.0, # Self-exciting trade intensity
            "vol_compression": 0.0,  # Volatility compression ratio
            "skewness": 0.0,    # Return distribution skew
            "kurtosis": 3.0,    # Return distribution kurtosis (3=normal)
            "toxic_flow_ratio": 0.0,  # Informed vs noise trader ratio
            "informed_prob": 0.0,     # Probability trade is informed (PIN model)
            "price_discovery_score": 0.0,  # Hasbrouck info share
        }

        if n < 20:
            self._cache[symbol] = (time.time(), result)
            return result

        prices = np.array([t[1] for t in trades])
        qtys = np.array([t[2] for t in trades])
        signs = np.array([t[3] for t in trades])
        timestamps = np.array([t[0] for t in trades])

        # --- 1. VPIN (Volume-synchronized Probability of Informed Trading) ---
        result["vpin"] = self._vpin(qtys, signs)

        # --- 2. Kyle's Lambda (price impact) ---
        result["kyle_lambda"] = self._kyle_lambda(prices, qtys, signs)

        # --- 3. Order Flow Imbalance ---
        result["ofi"] = self._ofi(qtys, signs)

        # --- 4. Hurst Exponent (R/S method) ---
        result["hurst"] = self._hurst(prices)

        # --- 5. Shannon Entropy ---
        result["entropy"] = self._entropy(prices)

        # --- 6. Microprice Skew ---
        result["microprice_skew"] = self._microprice_skew(symbol, prices)

        # --- 7. Absorption Rate ---
        result["absorption"] = self._absorption(prices, qtys, signs)

        # --- 8. Whale Prints ---
        result["whale_prints"] = self._whale_prints(qtys)

        # --- 9. Quote Stuffing ---
        result["quote_stuffing"] = self._quote_stuffing(symbol)

        # --- 10. Hawkes Intensity ---
        result["hawkes_intensity"] = self._hawkes_intensity(timestamps)

        # --- 11. Volatility Compression ---
        result["vol_compression"] = self._vol_compression(prices)

        # --- 12. Higher-order stats (skewness + kurtosis) ---
        sk, ku = self._higher_order_stats(prices)
        result["skewness"] = sk
        result["kurtosis"] = ku

        # --- 13. Toxic Flow Ratio (informed vs noise) ---
        result["toxic_flow_ratio"] = self._toxic_flow_ratio(qtys, signs, timestamps)

        # --- 14. Informed Trade Probability (simplified PIN) ---
        result["informed_prob"] = self._informed_probability(qtys, signs)

        # --- 15. Price Discovery Score (Hasbrouck info share proxy) ---
        result["price_discovery_score"] = self._price_discovery_score(symbol, prices, timestamps)

        self._cache[symbol] = (time.time(), result)
        market_data.micro_alpha[symbol] = result
        return result

    # ─── Feature Implementations ───────────────────────────────────────

    def _vpin(self, qtys: np.ndarray, signs: np.ndarray, n_buckets: int = 10) -> float:
        """VPIN: fraction of volume that is 'informed' (directional).
        High VPIN (>0.7) = smart money active before news/move."""
        dollar_vol = qtys  # already in base qty; use raw for bucket
        total_vol = dollar_vol.sum()
        if total_vol <= 0:
            return 0.0
        bucket_size = total_vol / n_buckets
        if bucket_size <= 0:
            return 0.0

        vpin_sum = 0.0
        bucket_buy = 0.0
        bucket_sell = 0.0
        bucket_vol = 0.0
        buckets_done = 0

        for i in range(len(qtys)):
            q = qtys[i]
            if signs[i] > 0:
                bucket_buy += q
            else:
                bucket_sell += q
            bucket_vol += q
            if bucket_vol >= bucket_size:
                vpin_sum += abs(bucket_buy - bucket_sell) / (bucket_vol + 1e-12)
                bucket_buy = bucket_sell = bucket_vol = 0.0
                buckets_done += 1

        if buckets_done == 0:
            return abs(signs.sum()) / (len(signs) + 1e-12)
        return vpin_sum / buckets_done

    def _kyle_lambda(self, prices: np.ndarray, qtys: np.ndarray, signs: np.ndarray) -> float:
        """Kyle's Lambda: regression of ΔP on signed volume.
        High lambda = low liquidity, each trade moves price a lot."""
        if len(prices) < 10:
            return 0.0
        dp = np.diff(prices)
        signed_vol = (qtys[1:] * signs[1:])
        denom = np.dot(signed_vol, signed_vol)
        if denom == 0:
            return 0.0
        lam = np.dot(dp, signed_vol) / denom
        return float(lam)

    def _ofi(self, qtys: np.ndarray, signs: np.ndarray) -> float:
        """Order Flow Imbalance: net signed volume / total volume.
        Range [-1, +1]. Strong positive = aggressive buying dominates."""
        total = qtys.sum()
        if total <= 0:
            return 0.0
        net = (qtys * signs).sum()
        return float(net / total)

    def _hurst(self, prices: np.ndarray) -> float:
        """Hurst exponent via R/S analysis.
        H > 0.5 = trending (persistent), H < 0.5 = mean-reverting."""
        n = len(prices)
        if n < 20:
            return 0.5
        log_p = np.log(prices + 1e-12)
        returns = np.diff(log_p)
        if len(returns) < 10:
            return 0.5

        # Simplified R/S for speed
        splits = [len(returns)]
        if len(returns) >= 40:
            splits = [len(returns) // 4, len(returns) // 2, len(returns)]

        rs_values = []
        ns = []
        for size in splits:
            if size < 5:
                continue
            chunk = returns[:size]
            mean_r = chunk.mean()
            dev = np.cumsum(chunk - mean_r)
            r = dev.max() - dev.min()
            s = chunk.std()
            if s > 0:
                rs_values.append(r / s)
                ns.append(size)

        if len(rs_values) < 2:
            # Fallback: variance ratio
            if len(returns) >= 20:
                var1 = np.var(returns[:len(returns)//2])
                var2 = np.var(returns)
                ratio = var2 / (2 * var1 + 1e-12)
                return min(1.0, max(0.0, 0.5 + (ratio - 1.0) * 0.3))
            return 0.5

        log_rs = np.log(rs_values)
        log_ns = np.log(ns)
        # Linear regression slope = Hurst
        slope = (log_rs[-1] - log_rs[0]) / (log_ns[-1] - log_ns[0] + 1e-12)
        return float(max(0.0, min(1.0, slope)))

    def _entropy(self, prices: np.ndarray, n_bins: int = 8) -> float:
        """Shannon entropy of discretized returns.
        Low entropy = predictable pattern. High = random."""
        if len(prices) < 10:
            return 1.0
        returns = np.diff(np.log(prices + 1e-12))
        if returns.std() == 0:
            return 0.0
        # Discretize into bins
        bins = np.linspace(returns.min() - 1e-12, returns.max() + 1e-12, n_bins + 1)
        hist, _ = np.histogram(returns, bins=bins)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(n_bins)
        return float(entropy / max_entropy) if max_entropy > 0 else 1.0

    def _microprice_skew(self, symbol: str, prices: np.ndarray) -> float:
        """Deviation of microprice from mid-price (normalized by spread).
        Positive = fair value above mid (bullish), negative = bearish."""
        mp = market_data.get_microprice(symbol)
        if mp is None:
            return 0.0
        q = market_data.best_quote.get(symbol)
        if not q:
            return 0.0
        _, bb, _, ba, _ = q
        spread = ba - bb
        if spread <= 0:
            return 0.0
        mid = (bb + ba) / 2.0
        return float((mp - mid) / spread)

    def _absorption(self, prices: np.ndarray, qtys: np.ndarray, signs: np.ndarray) -> float:
        """Absorption: large volume consumed without price movement.
        High absorption = whale accumulating silently. Range [0, 1]."""
        if len(prices) < 10:
            return 0.0
        # Split into chunks of 10 trades
        chunk_size = max(5, len(prices) // 5)
        absorptions = []
        for i in range(0, len(prices) - chunk_size, chunk_size):
            chunk_p = prices[i:i + chunk_size]
            chunk_q = qtys[i:i + chunk_size]
            price_move = abs(chunk_p[-1] - chunk_p[0]) / (chunk_p[0] + 1e-12)
            vol_consumed = chunk_q.sum()
            if vol_consumed > 0:
                # Low price move per unit volume = absorption
                absorptions.append(1.0 - min(1.0, price_move * 1000 / (vol_consumed + 1e-12) * chunk_p.mean()))

        if not absorptions:
            return 0.0
        return float(max(0.0, min(1.0, np.mean(absorptions))))

    def _whale_prints(self, qtys: np.ndarray) -> int:
        """Count trades with size > 3 standard deviations above mean."""
        if len(qtys) < 20:
            return 0
        mean_q = qtys.mean()
        std_q = qtys.std()
        if std_q == 0:
            return 0
        threshold = mean_q + 3.0 * std_q
        return int(np.sum(qtys > threshold))

    def _quote_stuffing(self, symbol: str) -> float:
        """Quote stuffing: rapid orderbook changes without fills.
        Uses depth_history churn rate. High = HFT manipulation."""
        buf = market_data.depth_history.get(symbol)
        if not buf or len(buf) < 6:
            return 0.0
        now = time.time()
        recent = [(ts, bt, at, tbq, taq) for ts, bt, at, tbq, taq in buf if now - ts < 10]
        if len(recent) < 4:
            return 0.0
        # Measure total absolute change in top-of-book
        changes = 0.0
        for i in range(1, len(recent)):
            changes += abs(recent[i][3] - recent[i - 1][3])
            changes += abs(recent[i][4] - recent[i - 1][4])
        avg_size = sum(r[3] + r[4] for r in recent) / len(recent)
        if avg_size <= 0:
            return 0.0
        # Normalize: churn relative to average displayed size
        stuffing = changes / (avg_size * len(recent))
        return float(min(1.0, stuffing))

    def _hawkes_intensity(self, timestamps: np.ndarray, decay: float = 0.1) -> float:
        """Hawkes process: self-exciting intensity.
        High = trades clustering (burst incoming). Normalized [0, 1]."""
        if len(timestamps) < 10:
            return 0.0
        now = timestamps[-1]
        # Intensity = sum of exponential kernels from past events
        diffs = now - timestamps[:-1]
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            return 0.0
        intensity = np.sum(np.exp(-decay * diffs))
        # Normalize by expected baseline (n_trades / window)
        window = timestamps[-1] - timestamps[0]
        if window <= 0:
            return 0.0
        baseline = len(timestamps) / window
        normalized = intensity / (baseline * 10 + 1e-8)
        return float(min(1.0, normalized))

    def _vol_compression(self, prices: np.ndarray) -> float:
        """Volatility compression: recent vol / historical vol.
        Low ratio = coiled spring about to explode. Range [0, 1]."""
        if len(prices) < 30:
            return 0.0
        log_ret = np.diff(np.log(prices + 1e-12))
        if len(log_ret) < 20:
            return 0.0
        recent_vol = np.std(log_ret[-10:])
        hist_vol = np.std(log_ret)
        if hist_vol <= 0:
            return 0.0
        ratio = recent_vol / hist_vol
        # Invert: low ratio = high compression score
        compression = max(0.0, min(1.0, 1.0 - ratio))
        return float(compression)

    def _higher_order_stats(self, prices: np.ndarray):
        """Skewness and kurtosis of tick returns."""
        if len(prices) < 20:
            return 0.0, 3.0
        log_ret = np.diff(np.log(prices + 1e-12))
        n = len(log_ret)
        if n < 10:
            return 0.0, 3.0
        mean = log_ret.mean()
        std = log_ret.std()
        if std == 0:
            return 0.0, 3.0
        centered = (log_ret - mean) / std
        skew = float(np.mean(centered ** 3))
        kurt = float(np.mean(centered ** 4))
        return skew, kurt

    def _toxic_flow_ratio(self, qtys: np.ndarray, signs: np.ndarray, timestamps: np.ndarray) -> float:
        """Toxic Flow Ratio: fraction of volume from informed traders.
        Large trades in bursts = informed. Small uniform = noise."""
        n = len(qtys)
        if n < 20:
            return 0.0
        median_q = np.median(qtys)
        large_mask = qtys > median_q * 2
        large_vol = qtys[large_mask].sum()
        total_vol = qtys.sum()
        size_ratio = large_vol / (total_vol + 1e-12)

        if n >= 10:
            iat = np.diff(timestamps)
            iat = iat[iat > 0]
            cv = iat.std() / (iat.mean() + 1e-12) if len(iat) > 5 else 0.0
            burst_score = min(1.0, cv / 2.5)
        else:
            burst_score = 0.0

        if large_mask.sum() >= 3:
            consistency = abs(signs[large_mask].mean())
        else:
            consistency = 0.0

        return float(min(1.0, size_ratio * 0.4 + burst_score * 0.3 + consistency * 0.3))

    def _informed_probability(self, qtys: np.ndarray, signs: np.ndarray) -> float:
        """Simplified PIN model: imbalance / (imbalance + 2*noise)."""
        if len(qtys) < 20:
            return 0.0
        buy_vol = qtys[signs > 0].sum()
        sell_vol = qtys[signs < 0].sum()
        total = buy_vol + sell_vol
        if total <= 0:
            return 0.0
        imbalance = abs(buy_vol - sell_vol)
        noise_proxy = 2 * min(buy_vol, sell_vol)
        return float(min(1.0, imbalance / (imbalance + noise_proxy + 1e-12)))

    def _price_discovery_score(self, symbol: str, prices: np.ndarray, timestamps: np.ndarray) -> float:
        """Hasbrouck info share proxy: does Binance lead or lag?
        Returns [0,1] where 1 = Binance fully leads."""
        from engine.multi_exchange import bybit_feed, okx_feed

        binance_price = prices[-1] if len(prices) > 0 else 0
        bybit_price = bybit_feed.prices.get(symbol, 0)
        okx_price = okx_feed.prices.get(symbol, 0)

        if binance_price <= 0:
            return 0.5
        if bybit_price <= 0 and okx_price <= 0:
            return 0.7

        divergences = []
        if bybit_price > 0:
            divergences.append((bybit_price - binance_price) / (binance_price + 1e-12))
        if okx_price > 0:
            divergences.append((okx_price - binance_price) / (binance_price + 1e-12))

        avg_div = np.mean(divergences)
        # Negative div = others lower = Binance leads
        discovery = 0.5 - avg_div * 50
        return float(np.clip(discovery, 0.0, 1.0))


# Singleton
micro_engine = MicrostructureEngine()
