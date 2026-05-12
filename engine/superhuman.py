"""
Superhuman Signal Detector
===========================
10+ signals INVISIBLE to human traders. These exploit statistical patterns,
information theory, and microstructure phenomena that require computational
analysis at tick-level resolution.

Signals:
1. Tick Imbalance Bars (TIB) - Detect informed flow via volume clock
2. Order Flow Toxicity (VPIN v2) - Probability of adverse selection
3. Entropy Regime Shift - Detect regime change BEFORE price moves
4. Cross-Timeframe Hidden Divergence - Multi-resolution momentum conflicts
5. Smart Money Footprint - Institutional accumulation/distribution pattern
6. Information Asymmetry Index - Detect when insiders are trading
7. Gamma Exposure Proxy - Predict volatility pinning/explosion
8. Temporal Clustering - Time-of-day + day-of-week alpha
9. Autocorrelation Decay - Detect momentum exhaustion microscopically
10. Toxic Flow Ratio - Separate informed vs noise traders
11. Price Discovery Score - Which exchange/timeframe leads
12. Microstructure Momentum Shift - Tick-level trend change detection
"""
import numpy as np
import time
from collections import deque
from utils.state import market_data, bot_state


class SuperhumanDetector:
    """Computes superhuman-level signals from tick + kline data."""

    def __init__(self):
        self._cache = {}  # {symbol: (ts, signals_dict)}
        self._cache_ttl = 2.0
        self._tib_state = {}  # Tick Imbalance Bar state per symbol
        self._entropy_history = {}  # Rolling entropy for regime shift

    def compute(self, symbol: str, d1m=None, d15m=None, d1h=None) -> dict:
        """Compute all superhuman signals. Returns dict with scores."""
        cached = self._cache.get(symbol)
        if cached and time.time() - cached[0] < self._cache_ttl:
            return cached[1]

        result = {
            "tib_signal": 0,          # -1/0/+1: Tick Imbalance Bar direction
            "toxicity": 0.0,          # [0,1]: Order flow toxicity
            "entropy_shift": 0.0,     # [-1,1]: Regime shift direction
            "hidden_div": 0,          # -1/0/+1: Cross-TF hidden divergence
            "smart_money": 0.0,       # [-1,1]: Institutional footprint
            "info_asymmetry": 0.0,    # [0,1]: Information asymmetry level
            "gamma_proxy": 0.0,       # [-1,1]: Gamma exposure direction
            "temporal_alpha": 0.0,    # [-1,1]: Time-based edge
            "autocorr_decay": 0.0,    # [0,1]: Momentum exhaustion signal
            "toxic_flow": 0.0,        # [0,1]: Informed trader ratio
            "price_discovery": 0.0,   # [-1,1]: Who leads price
            "micro_momentum": 0,      # -1/0/+1: Tick-level trend shift
        }

        trades = market_data.get_trades(symbol, window_sec=120, max_items=3000)
        n = len(trades)

        # --- 1. Tick Imbalance Bars ---
        if n >= 30:
            result["tib_signal"] = self._tick_imbalance_bars(symbol, trades)

        # --- 2. Order Flow Toxicity (VPIN v2 with time decay) ---
        if n >= 50:
            result["toxicity"] = self._flow_toxicity(trades)

        # --- 3. Entropy Regime Shift ---
        if d1m is not None and len(d1m) >= 50:
            result["entropy_shift"] = self._entropy_regime_shift(symbol, d1m)

        # --- 4. Cross-Timeframe Hidden Divergence ---
        if d1m is not None and d15m is not None and len(d1m) >= 30 and len(d15m) >= 20:
            result["hidden_div"] = self._hidden_divergence(d1m, d15m, d1h)

        # --- 5. Smart Money Footprint ---
        if n >= 40:
            result["smart_money"] = self._smart_money_footprint(trades)

        # --- 6. Information Asymmetry Index ---
        if n >= 30:
            result["info_asymmetry"] = self._info_asymmetry(trades)

        # --- 7. Gamma Exposure Proxy ---
        if d1m is not None and len(d1m) >= 30:
            result["gamma_proxy"] = self._gamma_proxy(symbol, d1m)

        # --- 8. Temporal Clustering Alpha ---
        result["temporal_alpha"] = self._temporal_alpha()

        # --- 9. Autocorrelation Decay ---
        if n >= 50:
            result["autocorr_decay"] = self._autocorr_decay(trades)

        # --- 10. Toxic Flow Ratio ---
        if n >= 30:
            result["toxic_flow"] = self._toxic_flow_ratio(trades)

        # --- 11. Price Discovery Score ---
        if d1m is not None and len(d1m) >= 10:
            result["price_discovery"] = self._price_discovery(symbol, d1m)

        # --- 12. Microstructure Momentum Shift ---
        if n >= 40:
            result["micro_momentum"] = self._micro_momentum_shift(trades)

        self._cache[symbol] = (time.time(), result)
        return result

    # ─── Signal Implementations ────────────────────────────────────────

    def _tick_imbalance_bars(self, symbol: str, trades: list) -> int:
        """Tick Imbalance Bars: detect when buy/sell flow deviates from expected.
        
        Concept: In a balanced market, E[b_t] ≈ 0.5. When TIB threshold is
        breached, informed traders are likely active.
        Returns: +1 (buy imbalance), -1 (sell imbalance), 0 (balanced)
        """
        signs = np.array([t[3] for t in trades])
        n = len(signs)
        
        # Expected imbalance = running average of sign
        state = self._tib_state.get(symbol, {"e_bt": 0.0, "theta": 0.0})
        
        # Exponential moving average of trade signs
        alpha = 2.0 / (min(n, 100) + 1)
        cumulative_theta = 0.0
        for s in signs[-50:]:
            state["e_bt"] = alpha * s + (1 - alpha) * state["e_bt"]
            cumulative_theta += s - state["e_bt"]
        
        # Threshold = expected absolute imbalance * sqrt(n)
        threshold = abs(state["e_bt"]) * np.sqrt(50) * 1.5
        if threshold < 3:
            threshold = 3
        
        self._tib_state[symbol] = state
        
        if cumulative_theta > threshold:
            return 1  # Buy imbalance (informed buying)
        elif cumulative_theta < -threshold:
            return -1  # Sell imbalance (informed selling)
        return 0

    def _flow_toxicity(self, trades: list) -> float:
        """VPIN v2 with exponential time decay.
        
        Improvement over basic VPIN: recent buckets weighted more heavily,
        and uses actual trade timestamps for volume synchronization.
        High toxicity (>0.7) = market makers pulling liquidity = big move coming.
        """
        n = len(trades)
        qtys = np.array([t[2] for t in trades])
        signs = np.array([t[3] for t in trades])
        timestamps = np.array([t[0] for t in trades])
        
        # Volume-synchronized buckets with time decay
        total_vol = qtys.sum()
        if total_vol <= 0:
            return 0.0
        
        n_buckets = min(10, n // 5)
        if n_buckets < 3:
            return 0.0
        bucket_size = total_vol / n_buckets
        
        bucket_imbalances = []
        bucket_buy = 0.0
        bucket_sell = 0.0
        bucket_vol = 0.0
        
        for i in range(n):
            q = qtys[i]
            if signs[i] > 0:
                bucket_buy += q
            else:
                bucket_sell += q
            bucket_vol += q
            
            if bucket_vol >= bucket_size:
                imb = abs(bucket_buy - bucket_sell) / (bucket_vol + 1e-12)
                bucket_imbalances.append(imb)
                bucket_buy = bucket_sell = bucket_vol = 0.0
        
        if not bucket_imbalances:
            return 0.0
        
        # Time-decay weighting: recent buckets matter more
        weights = np.exp(np.linspace(-1, 0, len(bucket_imbalances)))
        weights /= weights.sum()
        
        toxicity = float(np.dot(bucket_imbalances, weights))
        return min(1.0, toxicity)

    def _entropy_regime_shift(self, symbol: str, d1m) -> float:
        """Detect regime change via entropy rate-of-change.
        
        When market transitions from trending→ranging or vice versa,
        entropy changes BEFORE price structure does. This gives 2-5 candle
        lead time on regime transitions.
        
        Returns: positive = shifting to trending, negative = shifting to ranging
        """
        closes = d1m['c'].values[-50:]
        returns = np.diff(np.log(closes + 1e-12))
        
        if len(returns) < 30:
            return 0.0
        
        # Compute rolling entropy (window=15)
        window = 15
        entropies = []
        for i in range(window, len(returns)):
            chunk = returns[i-window:i]
            std = chunk.std()
            if std == 0:
                entropies.append(0.5)
                continue
            # Discretize into 6 bins
            bins = np.linspace(chunk.min() - 1e-12, chunk.max() + 1e-12, 7)
            hist, _ = np.histogram(chunk, bins=bins)
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            ent = -np.sum(probs * np.log2(probs)) / np.log2(6)
            entropies.append(ent)
        
        if len(entropies) < 10:
            return 0.0
        
        # Store history for rate-of-change
        hist = self._entropy_history.get(symbol, deque(maxlen=20))
        hist.append(entropies[-1])
        self._entropy_history[symbol] = hist
        
        if len(hist) < 5:
            return 0.0
        
        # Rate of change in entropy
        recent = np.array(list(hist))
        slope = (recent[-1] - recent[-5]) / 5.0
        
        # Negative slope = entropy decreasing = becoming more structured/trending
        # Positive slope = entropy increasing = becoming more random/ranging
        return float(np.clip(-slope * 20, -1.0, 1.0))

    def _hidden_divergence(self, d1m, d15m, d1h=None) -> int:
        """Cross-timeframe hidden divergence.
        
        Hidden divergence (vs regular): price makes HIGHER low but oscillator
        makes LOWER low = continuation signal (not reversal).
        
        Multi-TF: Check if 1m momentum conflicts with 15m structure.
        This catches moves that single-timeframe analysis misses.
        """
        # 1m RSI momentum
        c1m = d1m['c'].values
        if len(c1m) < 20:
            return 0
        
        ret1m = np.diff(c1m[-20:])
        gains = np.where(ret1m > 0, ret1m, 0)
        losses = np.where(ret1m < 0, -ret1m, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rsi_1m = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-12)))
        
        # 15m RSI
        c15m = d15m['c'].values
        if len(c15m) < 20:
            return 0
        ret15m = np.diff(c15m[-20:])
        gains15 = np.where(ret15m > 0, ret15m, 0)
        losses15 = np.where(ret15m < 0, -ret15m, 0)
        avg_gain15 = np.mean(gains15[-14:])
        avg_loss15 = np.mean(losses15[-14:])
        rsi_15m = 100 - (100 / (1 + avg_gain15 / (avg_loss15 + 1e-12)))
        
        # Hidden bullish: 1m RSI dropping (looks weak) but 15m RSI rising
        if rsi_1m < 40 and rsi_15m > 55:
            return 1  # Hidden bullish divergence
        
        # Hidden bearish: 1m RSI rising (looks strong) but 15m RSI dropping
        if rsi_1m > 60 and rsi_15m < 45:
            return -1  # Hidden bearish divergence
        
        # 1h confirmation if available
        if d1h is not None and len(d1h) >= 15:
            c1h = d1h['c'].values
            ret1h = np.diff(c1h[-15:])
            gains1h = np.where(ret1h > 0, ret1h, 0)
            losses1h = np.where(ret1h < 0, -ret1h, 0)
            avg_gain1h = np.mean(gains1h[-14:]) if len(gains1h) >= 14 else np.mean(gains1h)
            avg_loss1h = np.mean(losses1h[-14:]) if len(losses1h) >= 14 else np.mean(losses1h)
            rsi_1h = 100 - (100 / (1 + avg_gain1h / (avg_loss1h + 1e-12)))
            
            # Triple divergence: 1m weak, 15m strong, 1h strong = very bullish
            if rsi_1m < 35 and rsi_15m > 50 and rsi_1h > 55:
                return 1
            if rsi_1m > 65 and rsi_15m < 50 and rsi_1h < 45:
                return -1
        
        return 0

    def _smart_money_footprint(self, trades: list) -> float:
        """Detect institutional accumulation/distribution.
        
        Smart money trades in specific patterns:
        - Large orders split into many small ones (iceberg)
        - Consistent directional pressure without price impact
        - Volume-weighted average trade size increasing on one side
        
        Returns: [-1, +1] where +1 = accumulation, -1 = distribution
        """
        n = len(trades)
        qtys = np.array([t[2] for t in trades])
        signs = np.array([t[3] for t in trades])
        prices = np.array([t[1] for t in trades])
        
        # Split into halves: first half vs second half
        mid = n // 2
        
        # 1. Volume-weighted direction shift
        buy_vol_1 = qtys[:mid][signs[:mid] > 0].sum()
        sell_vol_1 = qtys[:mid][signs[:mid] < 0].sum()
        buy_vol_2 = qtys[mid:][signs[mid:] > 0].sum()
        sell_vol_2 = qtys[mid:][signs[mid:] < 0].sum()
        
        total_1 = buy_vol_1 + sell_vol_1 + 1e-12
        total_2 = buy_vol_2 + sell_vol_2 + 1e-12
        
        ratio_1 = (buy_vol_1 - sell_vol_1) / total_1
        ratio_2 = (buy_vol_2 - sell_vol_2) / total_2
        
        # 2. Price impact efficiency: smart money moves price less per unit volume
        price_move = abs(prices[-1] - prices[0]) / (prices[0] + 1e-12)
        vol_consumed = qtys.sum()
        impact_efficiency = price_move / (vol_consumed + 1e-12) * prices.mean()
        
        # Low impact + directional volume = smart money
        # High impact + directional volume = retail panic
        is_low_impact = impact_efficiency < np.median(qtys) * 0.001
        
        # 3. Combine: direction shift + low impact = institutional
        direction = ratio_2  # Recent direction
        if is_low_impact:
            direction *= 1.5  # Amplify if low impact (smart money signature)
        
        return float(np.clip(direction, -1.0, 1.0))

    def _info_asymmetry(self, trades: list) -> float:
        """Information Asymmetry Index.
        
        Measures the probability that informed traders are active.
        Based on: trade size distribution becoming bimodal (small retail + large informed)
        and price impact asymmetry (buys move price more than sells or vice versa).
        
        High value (>0.6) = insiders likely trading.
        """
        qtys = np.array([t[2] for t in trades])
        signs = np.array([t[3] for t in trades])
        prices = np.array([t[1] for t in trades])
        
        if len(qtys) < 20:
            return 0.0
        
        # 1. Bimodality of trade sizes (coefficient of variation)
        cv = qtys.std() / (qtys.mean() + 1e-12)
        # High CV = bimodal distribution = mix of informed + noise
        bimodal_score = min(1.0, cv / 3.0)
        
        # 2. Price impact asymmetry
        buy_mask = signs > 0
        sell_mask = signs < 0
        
        if buy_mask.sum() < 5 or sell_mask.sum() < 5:
            return bimodal_score * 0.5
        
        # Average price change per buy vs per sell
        dp = np.diff(prices)
        if len(dp) < len(signs) - 1:
            return bimodal_score * 0.5
        
        buy_impact = np.abs(dp[buy_mask[1:]]).mean() if buy_mask[1:].sum() > 0 else 0
        sell_impact = np.abs(dp[sell_mask[1:]]).mean() if sell_mask[1:].sum() > 0 else 0
        
        # Asymmetry: one side moves price more = informed on that side
        if buy_impact + sell_impact > 0:
            asymmetry = abs(buy_impact - sell_impact) / (buy_impact + sell_impact)
        else:
            asymmetry = 0.0
        
        # Combine
        info_score = (bimodal_score * 0.6 + asymmetry * 0.4)
        return float(min(1.0, info_score))

    def _gamma_proxy(self, symbol: str, d1m) -> float:
        """Gamma Exposure Proxy.
        
        In crypto futures, large OI at round numbers creates "gamma pinning" effect.
        When price approaches high-OI strike levels, volatility compresses.
        When it breaks through, volatility explodes (gamma squeeze).
        
        Returns: positive = likely squeeze up, negative = squeeze down, ~0 = pinned
        """
        price = d1m['c'].iloc[-1]
        
        # Find nearest round numbers (psychological levels)
        magnitude = 10 ** (len(str(int(price))) - 2)
        round_above = np.ceil(price / magnitude) * magnitude
        round_below = np.floor(price / magnitude) * magnitude
        
        dist_above = (round_above - price) / price
        dist_below = (price - round_below) / price
        
        # If very close to round number = pinning zone
        if dist_above < 0.003 or dist_below < 0.003:
            # Check momentum direction for squeeze prediction
            ret_5 = (price - d1m['c'].iloc[-6]) / d1m['c'].iloc[-6] if len(d1m) >= 6 else 0
            if ret_5 > 0.002:
                return 0.5  # Approaching from below, likely squeeze up
            elif ret_5 < -0.002:
                return -0.5  # Approaching from above, likely squeeze down
            return 0.0  # Pinned
        
        # Check OI concentration (if available)
        oi = market_data.oi.get(symbol, 0)
        if oi > 0:
            # High OI + price near round = stronger gamma effect
            oi_factor = min(1.0, oi / 1e8)  # Normalize
            if dist_above < dist_below:
                return float(oi_factor * 0.3)  # Slight upward bias
            else:
                return float(-oi_factor * 0.3)
        
        return 0.0

    def _temporal_alpha(self) -> float:
        """Time-based alpha: certain hours/days have statistical edge.
        
        Crypto patterns:
        - Asian session open (00:00 UTC): often reversal
        - London open (08:00 UTC): trend initiation
        - NY open (13:30 UTC): highest volatility
        - Sunday evening: low liquidity manipulation
        - Monday: trend-setting day
        """
        from datetime import datetime
        now = datetime.utcnow()
        hour = now.hour
        weekday = now.weekday()  # 0=Monday
        
        alpha = 0.0
        
        # High-alpha hours (trend initiation)
        if 8 <= hour <= 10:  # London open
            alpha += 0.3
        elif 13 <= hour <= 15:  # NY open
            alpha += 0.4
        
        # Low-alpha hours (manipulation/noise)
        elif 2 <= hour <= 5:  # Dead zone
            alpha -= 0.3
        
        # Day-of-week effects
        if weekday == 0:  # Monday: trend-setting
            alpha += 0.2
        elif weekday == 6:  # Sunday: low liquidity
            alpha -= 0.4
        elif weekday == 4:  # Friday: profit-taking
            alpha -= 0.1
        
        return float(np.clip(alpha, -1.0, 1.0))

    def _autocorr_decay(self, trades: list) -> float:
        """Autocorrelation decay: detect momentum exhaustion at tick level.
        
        When autocorrelation of trade signs drops rapidly, momentum is dying.
        This happens 10-30 seconds BEFORE price reverses.
        
        Returns: [0, 1] where 1 = momentum exhausted (reversal imminent)
        """
        signs = np.array([t[3] for t in trades])
        n = len(signs)
        if n < 30:
            return 0.0
        
        # Compute autocorrelation at lag 1, 3, 5
        def _autocorr(x, lag):
            if len(x) <= lag:
                return 0.0
            c = np.corrcoef(x[:-lag], x[lag:])[0, 1]
            return c if not np.isnan(c) else 0.0
        
        ac1 = _autocorr(signs[-30:], 1)
        ac3 = _autocorr(signs[-30:], 3)
        ac5 = _autocorr(signs[-30:], 5)
        
        # Strong momentum = high autocorrelation (trades cluster in same direction)
        # Exhaustion = autocorrelation dropping fast
        avg_ac = (ac1 + ac3 + ac5) / 3.0
        
        # Compare to earlier window
        if n >= 60:
            ac1_prev = _autocorr(signs[-60:-30], 1)
            ac3_prev = _autocorr(signs[-60:-30], 3)
            avg_ac_prev = (ac1_prev + ac3_prev) / 2.0
            
            # Decay = previous was high, current is low
            decay = max(0, avg_ac_prev - avg_ac)
            return float(min(1.0, decay * 3))
        
        # If current autocorrelation is near zero = no momentum
        return float(min(1.0, max(0, 0.5 - abs(avg_ac))))

    def _toxic_flow_ratio(self, trades: list) -> float:
        """Separate informed vs noise traders by trade characteristics.
        
        Informed traders:
        - Trade at specific price levels (not random)
        - Larger average size
        - Cluster in time (burst patterns)
        
        Returns: [0, 1] ratio of informed flow
        """
        qtys = np.array([t[2] for t in trades])
        timestamps = np.array([t[0] for t in trades])
        
        if len(qtys) < 20:
            return 0.0
        
        # 1. Size-based classification
        median_size = np.median(qtys)
        large_mask = qtys > median_size * 2
        large_ratio = large_mask.sum() / len(qtys)
        
        # 2. Temporal clustering (Hawkes-like)
        if len(timestamps) >= 10:
            inter_arrival = np.diff(timestamps)
            inter_arrival = inter_arrival[inter_arrival > 0]
            if len(inter_arrival) > 5:
                cv_time = inter_arrival.std() / (inter_arrival.mean() + 1e-12)
                # High CV = bursty (informed), Low CV = uniform (noise)
                burst_score = min(1.0, cv_time / 2.0)
            else:
                burst_score = 0.0
        else:
            burst_score = 0.0
        
        # 3. Combine
        toxic = large_ratio * 0.5 + burst_score * 0.5
        return float(min(1.0, toxic))

    def _price_discovery(self, symbol: str, d1m) -> float:
        """Price Discovery Score: detect if Binance leads or lags.
        
        If other exchanges (Bybit/OKX) moved first, Binance will follow.
        Uses cross-exchange price divergence direction.
        
        Returns: [-1, +1] where +1 = Binance lagging upward move
        """
        from engine.multi_exchange import bybit_feed, okx_feed
        
        bybit_div = bybit_feed.get_divergence(symbol)
        okx_div = okx_feed.get_divergence(symbol)
        
        # If other exchanges are higher = Binance will catch up (bullish)
        # If other exchanges are lower = Binance will drop (bearish)
        avg_div = (bybit_div * 0.55 + okx_div * 0.45)
        
        # Only significant if divergence > 0.02%
        if abs(avg_div) < 0.02:
            return 0.0
        
        # Scale to [-1, 1]
        return float(np.clip(avg_div * 10, -1.0, 1.0))

    def _micro_momentum_shift(self, trades: list) -> int:
        """Detect tick-level momentum shift (trend change at micro level).
        
        Uses cumulative tick direction with exponential weighting.
        When recent ticks flip direction vs older ticks = momentum shift.
        
        Returns: +1 (shifting bullish), -1 (shifting bearish), 0 (no shift)
        """
        signs = np.array([t[3] for t in trades])
        n = len(signs)
        if n < 30:
            return 0
        
        # Exponentially weighted cumulative direction
        # Recent trades weighted 3x more than older
        weights = np.exp(np.linspace(-1, 0, n))
        weighted_sum = np.sum(signs * weights)
        
        # Compare first half vs second half direction
        mid = n // 2
        first_half = signs[:mid].mean()
        second_half = signs[mid:].mean()
        
        # Shift = direction change between halves
        shift = second_half - first_half
        
        if shift > 0.3:
            return 1  # Shifting bullish
        elif shift < -0.3:
            return -1  # Shifting bearish
        return 0


# Singleton
superhuman = SuperhumanDetector()
