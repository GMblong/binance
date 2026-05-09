import pandas as pd
import numpy as np
from functools import lru_cache
import time
from numba import njit, prange

# Pre-computed indicator cache to avoid recalculating on every tick
_indicator_cache = {}  # {(symbol, interval, last_ot): {indicators}}
_CACHE_TTL = 2.0  # seconds


def _cache_key(df, prefix=""):
    """Generate cache key from dataframe's last open time."""
    if df is None or df.empty:
        return None
    return (prefix, float(df.iloc[-1]['ot']), len(df))


@njit(cache=True)
def _ema_loop(arr, alpha):
    """Numba-accelerated EMA computation."""
    n = len(arr)
    out = np.empty(n)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i-1]
    return out


@njit(cache=True)
def _rsi_loop(vals, length):
    """Numba-accelerated RSI computation."""
    n = len(vals)
    rsi = np.empty(n)
    alpha = 1.0 / length
    avg_gain = 0.0
    avg_loss = 0.0
    
    for i in range(n):
        if i == 0:
            delta = 0.0
        else:
            delta = vals[i] - vals[i-1]
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        avg_gain = alpha * gain + (1.0 - alpha) * avg_gain
        avg_loss = alpha * loss + (1.0 - alpha) * avg_loss
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


@njit(cache=True)
def _atr_loop(high, low, close, length):
    """Numba-accelerated ATR computation."""
    n = len(high)
    tr = np.empty(n)
    atr = np.empty(n)
    alpha = 2.0 / (length + 1)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
    
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i-1]
    return atr


@njit(cache=True)
def _adx_loop(high, low, close, length):
    """Numba-accelerated ADX computation."""
    n = len(high)
    alpha = 2.0 / (length + 1)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
    
    # EMA smoothing
    atr_s = np.empty(n)
    pdi_s = np.empty(n)
    mdi_s = np.empty(n)
    atr_s[0] = tr[0]
    pdi_s[0] = plus_dm[0]
    mdi_s[0] = minus_dm[0]
    for i in range(1, n):
        atr_s[i] = alpha * tr[i] + (1.0 - alpha) * atr_s[i-1]
        pdi_s[i] = alpha * plus_dm[i] + (1.0 - alpha) * pdi_s[i-1]
        mdi_s[i] = alpha * minus_dm[i] + (1.0 - alpha) * mdi_s[i-1]
    
    dx = np.empty(n)
    adx = np.empty(n)
    for i in range(n):
        pdi = 100.0 * pdi_s[i] / (atr_s[i] + 1e-8)
        mdi = 100.0 * mdi_s[i] / (atr_s[i] + 1e-8)
        dx[i] = abs(pdi - mdi) / (pdi + mdi + 1e-8) * 100.0
    
    adx[0] = dx[0]
    for i in range(1, n):
        adx[i] = alpha * dx[i] + (1.0 - alpha) * adx[i-1]
    
    return adx[-1]


@njit(cache=True)
def _hmm_forward(returns, trans, means, stds):
    """Numba-accelerated HMM forward pass."""
    n_states = 2
    log_alpha = np.array([-0.5, -0.5])
    
    for r in returns:
        log_emit = np.empty(n_states)
        for s in range(n_states):
            log_emit[s] = -0.5 * ((r - means[s]) / stds[s])**2 - np.log(stds[s])
        
        new_alpha = np.empty(n_states)
        for j in range(n_states):
            a = log_alpha[0] + np.log(trans[0, j])
            b = log_alpha[1] + np.log(trans[1, j])
            mx = max(a, b)
            new_alpha[j] = log_emit[j] + mx + np.log(np.exp(a - mx) + np.exp(b - mx))
        log_alpha = new_alpha
    
    return 0 if log_alpha[0] > log_alpha[1] else 1


class MarketAnalyzer:
    # HMM parameters (pre-fitted on typical crypto regimes)
    # State 0: Low volatility / Mean-reverting
    # State 1: High volatility / Trending
    _hmm_transition = np.array([[0.95, 0.05], [0.10, 0.90]])  # Sticky states
    _hmm_means = np.array([0.0, 0.0])  # Returns are ~0 mean in both
    _hmm_stds = np.array([0.003, 0.012])  # Low vol vs high vol

    @staticmethod
    def detect_hmm_regime(df, lookback=50):
        """Hidden Markov Model regime detection using numba-accelerated Viterbi."""
        if len(df) < lookback:
            return "UNKNOWN"
        try:
            returns = df['c'].pct_change().tail(lookback).dropna().values.astype(np.float64)
            if len(returns) < 10:
                return "UNKNOWN"
            
            trans = MarketAnalyzer._hmm_transition
            means = MarketAnalyzer._hmm_means
            stds = MarketAnalyzer._hmm_stds
            
            current_state = _hmm_forward(returns, trans, means, stds)
            return "MEAN_REVERT" if current_state == 0 else "MOMENTUM"
        except:
            return "UNKNOWN"
    @staticmethod
    def get_ema(series, length):
        vals = series.values if hasattr(series, 'values') else np.asarray(series, dtype=np.float64)
        alpha = 2.0 / (length + 1)
        result = _ema_loop(vals, alpha)
        return pd.Series(result, index=series.index if hasattr(series, 'index') else None)

    @staticmethod
    def get_rsi(series, length=14):
        vals = series.values if hasattr(series, 'values') else np.asarray(series, dtype=np.float64)
        rsi = _rsi_loop(vals, length)
        return pd.Series(rsi, index=series.index if hasattr(series, 'index') else None)

    @staticmethod
    def get_atr(df, length=14):
        h = df['h'].values.astype(np.float64)
        l = df['l'].values.astype(np.float64)
        c = df['c'].values.astype(np.float64)
        atr = _atr_loop(h, l, c, length)
        return pd.Series(atr, index=df.index)

    @staticmethod
    def get_adx(df, length=14):
        """Calculate ADX using numba JIT. Returns scalar value."""
        try:
            if len(df) < length * 2: return 25
            h = df['h'].values.astype(np.float64)
            l = df['l'].values.astype(np.float64)
            c = df['c'].values.astype(np.float64)
            return float(_adx_loop(h, l, c, length))
        except:
            return 25

    @staticmethod
    def get_volume_profile(df, bins=20):
        try:
            if len(df) < 30: return None
            c = df['c'].values
            l_vals = df['l'].values
            h_vals = df['h'].values
            v = df['v'].values
            
            price_min, price_max = float(np.min(l_vals)), float(np.max(h_vals))
            if price_max == price_min: return None
            
            bin_size = (price_max - price_min) / bins
            bin_idx = np.clip(((c - price_min) / bin_size).astype(int), 0, bins-1)
            
            # Aggregate volume by bin using numpy bincount
            profile = np.bincount(bin_idx, weights=v, minlength=bins)
            poc_bin = int(np.argmax(profile))
            poc_price = price_min + (poc_bin * bin_size) + (bin_size / 2)
            
            # Value Area (70% of volume)
            total_vol = profile.sum()
            sorted_bins = np.argsort(profile)[::-1]
            va_vol = 0
            va_bins = []
            for b in sorted_bins:
                va_vol += profile[b]
                va_bins.append(b)
                if va_vol >= total_vol * 0.7: break
            
            va_high = price_min + (max(va_bins) * bin_size) + bin_size
            va_low = price_min + (min(va_bins) * bin_size)
            
            return {"poc": poc_price, "vah": va_high, "val": va_low}
        except: return None

    @staticmethod
    def detect_vsa_signals(df):
        """Advanced VSA (Volume Spread Analysis) detection - numpy optimized."""
        try:
            if len(df) < 5: return 0
            v = df['v'].values
            o = df['o'].values
            h = df['h'].values
            l = df['l'].values
            c = df['c'].values
            
            curr_v = v[-1]
            curr_o, curr_h, curr_l, curr_c = o[-1], h[-1], l[-1], c[-1]
            vol_sma = np.mean(v[-20:]) if len(v) >= 20 else np.mean(v)
            
            is_high_vol = curr_v > (vol_sma * 1.5)
            is_low_vol = curr_v < (vol_sma * 0.7)
            body = abs(curr_c - curr_o)
            range_ = curr_h - curr_l
            if range_ == 0: return 0
            
            # 1. Absorption
            lower_wick = min(curr_o, curr_c) - curr_l
            if is_high_vol and lower_wick > (range_ * 0.6) and body < (range_ * 0.3):
                return 1
                
            upper_wick = curr_h - max(curr_o, curr_c)
            if is_high_vol and upper_wick > (range_ * 0.6) and body < (range_ * 0.3):
                return -1

            # 2. No Demand / No Supply
            avg_range = np.mean(h[-10:] - l[-10:])
            if is_low_vol and range_ < (avg_range * 0.8) and curr_c < curr_o:
                return 1
            if is_low_vol and range_ < (avg_range * 0.8) and curr_c > curr_o:
                return -1

            # 3. Stopping Volume
            if is_high_vol and curr_c < c[-2] and curr_c > curr_l + (range_ * 0.4):
                return 1

            # 4. Volume-Price Divergence
            if curr_h > np.max(h[-5:-1]) and curr_v < np.mean(v[-5:-1]) * 0.8:
                return -1
            if curr_l < np.min(l[-5:-1]) and curr_v < np.mean(v[-5:-1]) * 0.8:
                return 1

            return 0
        except: return 0

    @staticmethod
    def detect_regime(df):
        if len(df) < 30: return "RANGING"
        try:
            c = df["c"].values.astype(np.float64)
            n = len(c)
            # Use numba EMA
            ema20 = _ema_loop(c, 2.0 / 21)
            
            # Fast ATR%
            h = df['h'].values.astype(np.float64)
            l = df['l'].values.astype(np.float64)
            atr_vals = _atr_loop(h, l, c, 14)
            atr_pct = (atr_vals / c) * 100
            
            # Rolling mean of last 14
            curr_atr_p = np.mean(atr_pct[-14:])
            lookback = min(100, n)
            atr_p75 = np.percentile(atr_pct[-lookback:], 75)
            
            if curr_atr_p > atr_p75: return "VOLATILE"
            
            # Distance from EMA
            dist = ((c[-30:] - ema20[-30:]) / ema20[-30:]) * 100
            dist_std = np.std(dist)
            curr_dist = dist[-1]
            if dist_std > 0 and abs(curr_dist) > dist_std * 1.2: return "TRENDING"
            return "RANGING"
        except: return "RANGING"

    @staticmethod
    def detect_wyckoff_phase(df):
        """Detect Wyckoff market phase: ACCUMULATION, DISTRIBUTION, MARKUP, MARKDOWN.
        
        Logic:
        - ACCUMULATION: Price near lows, volume declining, range tightening (spring setup)
        - DISTRIBUTION: Price near highs, volume declining, range tightening (UTAD setup)
        - MARKUP: Price breaking above range with volume expansion
        - MARKDOWN: Price breaking below range with volume expansion
        """
        if len(df) < 30: return "UNKNOWN"
        try:
            c = df['c'].values
            v = df['v'].values
            h = df['h'].values
            l = df['l'].values
            
            # Define range from last 30 candles (more responsive)
            range_high = np.max(h[-30:])
            range_low = np.min(l[-30:])
            price_now = c[-1]
            
            # Volume trend
            vol_recent = np.mean(v[-5:])
            vol_baseline = np.mean(v[-30:-5]) if len(v) > 5 else np.mean(v)
            vol_declining = vol_recent < vol_baseline * 0.8
            vol_expanding = vol_recent > vol_baseline * 1.3
            
            # Range tightening
            recent_range = np.std(c[-8:]) / (np.mean(c[-8:]) + 1e-8)
            baseline_range = np.std(c[-30:]) / (np.mean(c[-30:]) + 1e-8)
            range_tightening = recent_range < baseline_range * 0.7
            
            # Position in range (0=bottom, 1=top)
            range_size = range_high - range_low
            if range_size == 0: return "UNKNOWN"
            position = (price_now - range_low) / range_size
            
            # Breakout detection
            prev_high = np.max(h[-15:-1])
            prev_low = np.min(l[-15:-1])
            
            if price_now > prev_high and vol_expanding:
                return "MARKUP"
            if price_now < prev_low and vol_expanding:
                return "MARKDOWN"
            if position < 0.35 and (vol_declining or range_tightening):
                return "ACCUMULATION"
            if position > 0.65 and (vol_declining or range_tightening):
                return "DISTRIBUTION"
            # Softer fallback: range tightening alone with position bias
            if range_tightening:
                return "ACCUMULATION" if position < 0.4 else "DISTRIBUTION" if position > 0.6 else "UNKNOWN"
            return "UNKNOWN"
        except:
            return "UNKNOWN"

    @staticmethod
    def detect_multi_candle_fake(df, direction):
        """Detect 3-candle fake move sequences that trap traders.
        
        Patterns detected:
        1. Fake Breakout: Break level → Rejection → Reversal (3 candles)
        2. Stop Hunt Sequence: Sweep → Absorption → Impulse opposite
        3. Exhaustion Pattern: Big move → Doji/Indecision → Reversal
        
        Returns: True if current move is likely fake/manipulation.
        """
        if len(df) < 5: return False
        try:
            c3 = df.iloc[-3]  # 3 candles ago
            c2 = df.iloc[-2]  # 2 candles ago
            c1 = df.iloc[-1]  # Current candle
            
            vol_avg = df['v'].tail(20).mean()
            atr_avg = (df['h'] - df['l']).tail(20).mean()
            
            # --- Pattern 1: Fake Breakout Trap ---
            # Candle 1: Big directional move (the "bait")
            # Candle 2: Rejection/wick (the "trap")
            # Candle 3: Reversal (the "kill")
            c3_body = c3['c'] - c3['o']
            c2_range = c2['h'] - c2['l']
            c2_body = abs(c2['c'] - c2['o'])
            c1_body = c1['c'] - c1['o']
            
            if direction == 1:  # Checking if LONG signal is fake
                # Fake bull: big green → upper wick rejection → red candle
                if (c3_body > atr_avg * 0.5 and  # Big green candle
                    (c2['h'] - max(c2['o'], c2['c'])) > c2_range * 0.5 and  # Upper wick rejection
                    c1_body < 0):  # Current is red
                    return True
            else:  # Checking if SHORT signal is fake
                # Fake bear: big red → lower wick rejection → green candle
                if (c3_body < -atr_avg * 0.5 and  # Big red candle
                    (min(c2['o'], c2['c']) - c2['l']) > c2_range * 0.5 and  # Lower wick rejection
                    c1_body > 0):  # Current is green
                    return True
            
            # --- Pattern 2: Stop Hunt Sequence ---
            # Low volume spike beyond level → high volume absorption → reversal
            c3_vol_low = c3['v'] < vol_avg * 0.8
            c2_vol_high = c2['v'] > vol_avg * 1.5
            
            if direction == 1:
                # Fake bull: low-vol spike up → high-vol selling → price drops
                if (c3['h'] > df['h'].iloc[-20:-3].max() and c3_vol_low and
                    c2_vol_high and c2['c'] < c2['o'] and c1['c'] < c2['c']):
                    return True
            else:
                # Fake bear: low-vol spike down → high-vol buying → price rises
                if (c3['l'] < df['l'].iloc[-20:-3].min() and c3_vol_low and
                    c2_vol_high and c2['c'] > c2['o'] and c1['c'] > c2['c']):
                    return True
            
            # --- Pattern 3: Exhaustion + Indecision + Reversal ---
            c3_range = c3['h'] - c3['l']
            c3_body_abs = abs(c3_body)
            c2_is_doji = c2_body < c2_range * 0.2 if c2_range > 0 else False
            
            if direction == 1:
                # Big green → doji → red = exhaustion top
                if (c3_body > atr_avg * 0.8 and c2_is_doji and c1_body < 0 and
                    c3['v'] > vol_avg * 1.3):
                    return True
            else:
                # Big red → doji → green = exhaustion bottom
                if (c3_body < -atr_avg * 0.8 and c2_is_doji and c1_body > 0 and
                    c3['v'] > vol_avg * 1.3):
                    return True
            
            return False
        except:
            return False

    @staticmethod
    def detect_structure(df):
        if len(df) < 15: return "CHOP", False, None, None
        try:
            highs = df['h'].values
            lows = df['l'].values
            recent_high = float(np.max(highs[-15:]))
            recent_low = float(np.min(lows[-15:]))
            
            # Vectorized fractal detection (lookback 2, lookforward 2)
            pivot_highs = []
            pivot_lows = []
            n = len(highs)
            for i in range(max(n-20, 2), n-2):
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    pivot_highs.append(highs[i])
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    pivot_lows.append(lows[i])
                    
            if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
                if pivot_highs[-1] > pivot_highs[-2] and pivot_lows[-1] > pivot_lows[-2]:
                    return "BULLISH", False, recent_high, recent_low
                if pivot_highs[-1] < pivot_highs[-2] and pivot_lows[-1] < pivot_lows[-2]:
                    return "BEARISH", False, recent_high, recent_low
                    
            # Fallback to simple momentum
            if highs[-1] > highs[-2] and lows[-1] > lows[-2]: 
                return "BULLISH", False, recent_high, recent_low
            if highs[-1] < highs[-2] and lows[-1] < lows[-2]: 
                return "BEARISH", False, recent_high, recent_low
                
            return "CHOP", False, recent_high, recent_low
        except: return "CHOP", False, None, None

    @staticmethod
    def calculate_score(d1m, d15m, direction, imbalance=1.0, funding=0.0, regime="RANGING", neural_weights=None, session="QUIET", lead_lag=0, return_features=False):
        """Returns int score (0-100). If return_features=True, returns (score, list_of_active_feature_keys).

        Feature keys follow `REGIME:FEATURE` convention so the feedback loop
        (see close_position_async) can credit/penalize the right weight bucket.
        """
        try:
            score = 0
            nw = neural_weights or {}
            active = []  # Features that actually contributed positively

            # 1. Macro Trend Alignment (15m) & Soft Mean Reversion (H1)
            ema9_15 = MarketAnalyzer.get_ema(d15m["c"], 9).iloc[-1]
            ema21_15 = MarketAnalyzer.get_ema(d15m["c"], 21).iloc[-1]
            dir_15 = 1 if ema9_15 > ema21_15 else -1
            
            rsi_15m = MarketAnalyzer.get_rsi(d15m["c"], 14).iloc[-1]
            
            if direction == dir_15: 
                score += 30 # Heavy weight for macro alignment
                active.append(f"{regime}:htf_align")
            else:
                # Soft alignment: Allow mean reversion if ranging and RSI extreme
                if regime == "RANGING":
                    vsa_sig = MarketAnalyzer.detect_vsa_signals(d1m)
                    # Counter-trend valid only if RSI is extreme OR there's a strong VSA reversal signal
                    if (direction == 1 and rsi_15m < 35) or (direction == -1 and rsi_15m > 65) or vsa_sig == direction:
                        score -= 15 # Mild penalty, but allowed to trade (Mean Reversion)
                        active.append(f"{regime}:mean_rev")
                    else:
                        if return_features:
                            return 0, []
                        return 0
                else:
                    if return_features:
                        return 0, []
                    return 0  # Still strict in trending/volatile markets

            # 2. Micro Momentum (1m)
            ema9 = MarketAnalyzer.get_ema(d1m["c"], 9).iloc[-1]
            ema21 = MarketAnalyzer.get_ema(d1m["c"], 21).iloc[-1]
            if direction == 1 and ema9 > ema21:
                score += 10
                active.append(f"{regime}:micro_mom")
            if direction == -1 and ema9 < ema21:
                score += 10
                active.append(f"{regime}:micro_mom")

            # 3. RSI Momentum (15m is more reliable than 1m)
            rsi = MarketAnalyzer.get_rsi(d15m["c"], 14).iloc[-1]
            if direction == 1 and 40 < rsi < 65:
                score += 10  # Room to grow upward
                active.append(f"{regime}:rsi_zone")
            if direction == -1 and 35 < rsi < 60:
                score += 10  # Room to drop downward
                active.append(f"{regime}:rsi_zone")

            # 4. Institutional Data (Orderbook & Funding)
            if direction == 1:
                if imbalance > 1.2:
                    score += 10
                    active.append(f"{regime}:ob_imb")
                elif imbalance < 0.8:
                    score -= 15  # Sell wall blocks LONG
            elif direction == -1:
                if imbalance < 0.8:
                    score += 10
                    active.append(f"{regime}:ob_imb")
                elif imbalance > 1.2:
                    score -= 15  # Buy wall blocks SHORT

            # Funding Rate (Contrarian Indicator)
            if direction == 1 and funding > 0.0005: score -= 10
            if direction == -1 and funding < -0.0005: score -= 10
            
            # 5. Smart Money & Anomalies (Weighted by Neural Weights)
            liq_w = nw.get(f"{regime}:liq", 1.0)
            ob_w = nw.get(f"{regime}:ob", 1.0)
            div_w = nw.get(f"{regime}:div", 1.0)

            sweep = MarketAnalyzer.detect_liquidity_sweep(d1m)
            if direction == 1 and sweep == 1:
                score += int(15 * liq_w)
                active.append(f"{regime}:liq")
            elif direction == -1 and sweep == -1:
                score += int(15 * liq_w)
                active.append(f"{regime}:liq")

            if MarketAnalyzer.detect_volume_anomaly(d1m):
                score += 5  # High volume supports the move
                active.append(f"{regime}:vol_anom")

            div = MarketAnalyzer.detect_rsi_divergence(d15m)
            if direction == 1 and div == 1:
                score += int(15 * div_w)
                active.append(f"{regime}:div")
            elif direction == -1 and div == -1:
                score += int(15 * div_w)
                active.append(f"{regime}:div")

            fvg = MarketAnalyzer.get_nearest_fvg(d1m)
            if fvg:
                if direction == 1 and fvg["type"] == "BULLISH":
                    score += int(10 * liq_w)
                    active.append(f"{regime}:fvg")
                elif direction == -1 and fvg["type"] == "BEARISH":
                    score += int(10 * liq_w)
                    active.append(f"{regime}:fvg")

            ob = MarketAnalyzer.find_nearest_order_block(d1m, d1m['c'].iloc[-1], direction)
            if ob:
                score += int(10 * ob_w)
                active.append(f"{regime}:ob")

            # 6. Advanced Analysis (Volume Profile, VSA, Fractal Confluence)
            # Fractal Confluence: Check if 1m structure aligns with 15m
            s1m, _, _, _ = MarketAnalyzer.detect_structure(d1m)
            s15m, _, _, _ = MarketAnalyzer.detect_structure(d15m)
            if direction == 1 and s1m == "BULLISH" and s15m == "BULLISH":
                score += 10
                active.append(f"{regime}:struct")
            elif direction == -1 and s1m == "BEARISH" and s15m == "BEARISH":
                score += 10
                active.append(f"{regime}:struct")

            # Volume Profile (POC)
            vp = MarketAnalyzer.get_volume_profile(d15m)
            if vp:
                price = d1m['c'].iloc[-1]
                if direction == 1 and price > vp['poc']:
                    score += 5
                    active.append(f"{regime}:poc")
                elif direction == -1 and price < vp['poc']:
                    score += 5
                    active.append(f"{regime}:poc")

            # VSA (Absorption & Supply/Demand)
            vsa_sig = MarketAnalyzer.detect_vsa_signals(d1m)
            if vsa_sig == direction:
                score += 15
                active.append(f"{regime}:vsa")

            # 7. Session & Lead-Lag Context
            if session in ["LONDON", "NEW_YORK"]:
                # Trend following is more reliable in these sessions
                if regime == "TRENDING":
                    score += 5
                    active.append(f"{regime}:session")

            if lead_lag == direction:
                score += 15  # Strongly follow the leader
                active.append(f"{regime}:lead_lag")

            final = max(0, min(score, 100))
            if return_features:
                return final, active
            return final
        except:
            if return_features:
                return 0, []
            return 0

    @staticmethod
    def detect_liquidity_sweep(df):
        try:
            if len(df) < 20: return 0
            # Check for sweep of previous 20-candle high/low for better accuracy
            prev_high = df['h'].iloc[-21:-1].max()
            prev_low = df['l'].iloc[-21:-1].min()
            
            curr_h = df['h'].iloc[-1]
            curr_l = df['l'].iloc[-1]
            curr_c = df['c'].iloc[-1]
            
            # Bullish Sweep: Price dipped below prev_low but closed above it
            if curr_l < prev_low and curr_c > prev_low:
                return 1
            # Bearish Sweep: Price spiked above prev_high but closed below it
            if curr_h > prev_high and curr_c < prev_high:
                return -1
            return 0
        except: return 0

    @staticmethod
    def get_structure_levels(df):
        return df['h'].max(), df['l'].min()
    
    @staticmethod
    def find_nearest_order_block(df, price, dir):
        try:
            if len(df) < 20: return None
            atr = MarketAnalyzer.get_atr(df, 14).iloc[-1]
            # Scan deeper (up to 50 candles back)
            for i in range(len(df)-3, max(0, len(df)-50), -1):
                body = abs(df['c'].iloc[i] - df['o'].iloc[i])
                impulse = abs(df['c'].iloc[i+1] - df['o'].iloc[i+1])
                
                # Ensure the following move is impulsive (at least 1.5x ATR)
                if impulse < atr * 1.5: continue

                if dir == 1 and df['c'].iloc[i] < df['o'].iloc[i]:
                    # Bullish OB: Last down candle before an impulse
                    if df['c'].iloc[i+1] > df['h'].iloc[i]:
                        ob_top, ob_bottom = df['h'].iloc[i], df['l'].iloc[i]
                        # Check mitigation
                        mitigated = False
                        for j in range(i+2, len(df)):
                            if df['l'].iloc[j] < ob_top:
                                mitigated = True
                                break
                        if not mitigated:
                            return {"top": ob_top, "bottom": ob_bottom, "type": "BULLISH"}
                elif dir == -1 and df['c'].iloc[i] > df['o'].iloc[i]:
                    # Bearish OB: Last up candle before a drop
                    if df['c'].iloc[i+1] < df['l'].iloc[i]:
                        ob_top, ob_bottom = df['h'].iloc[i], df['l'].iloc[i]
                        mitigated = False
                        for j in range(i+2, len(df)):
                            if df['h'].iloc[j] > ob_top:
                                mitigated = True
                                break
                        if not mitigated:
                            return {"top": ob_top, "bottom": ob_bottom, "type": "BEARISH"}
        except: return None
        return None

    @staticmethod
    def predict_liquidation_clusters(df):
        """Estimates liquidation clusters based on recent price extremes and common leverage tiers (M2)."""
        try:
            if len(df) < 30: return None
            # Find major swings
            recent_high = df['h'].tail(30).max()
            recent_low = df['l'].tail(30).min()
            
            tiers = [25, 50] # 4% and 2% leverage bands
            
            clusters = {"short_liq": [], "long_liq": []}
            
            for lev in tiers:
                margin = 1.0 / lev
                # Late longs entering at recent high get liquidated if price drops
                clusters["long_liq"].append(recent_high * (1 - margin))
                # Late shorts entering at recent low get liquidated if price spikes
                clusters["short_liq"].append(recent_low * (1 + margin))
                
            return clusters
        except: return None
    
    @staticmethod
    def detect_volatility_breakout(df):
        try:
            if len(df) < 20: return False
            atr = MarketAnalyzer.get_atr(df, 14)
            current_range = df['h'].iloc[-1] - df['l'].iloc[-1]
            return current_range > (atr.iloc[-2] * 2) 
        except: return False

    @staticmethod
    def detect_rsi_divergence(df):
        """Enhanced RSI Divergence Detection."""
        try:
            if len(df) < 40: return 0
            rsi = MarketAnalyzer.get_rsi(df['c'], 14)
            
            # Check for Bullish Divergence (Lower Low in Price, Higher Low in RSI)
            # Find local lows
            p_l1 = df['l'].iloc[-5:].min()
            p_l2 = df['l'].iloc[-25:-10].min()
            r_l1 = rsi.iloc[-5:].min()
            r_l2 = rsi.iloc[-25:-10].min()
            
            if p_l1 < p_l2 and r_l1 > r_l2 and r_l1 < 40:
                return 1 # Bullish Divergence
                
            # Check for Bearish Divergence (Higher High in Price, Lower High in RSI)
            p_h1 = df['h'].iloc[-5:].max()
            p_h2 = df['h'].iloc[-25:-10].max()
            r_h1 = rsi.iloc[-5:].max()
            r_h2 = rsi.iloc[-25:-10].max()
            
            if p_h1 > p_h2 and r_h1 < r_h2 and r_h1 > 60:
                return -1 # Bearish Divergence
                
            return 0
        except: return 0

    @staticmethod
    def detect_volume_anomaly(df):
        try:
            if len(df) < 20: return False
            vol_sma = df['v'].rolling(window=20).mean()
            return df['v'].iloc[-1] > (vol_sma.iloc[-2] * 2.5) 
        except: return False

    @staticmethod
    def detect_sweep(df): return False
    
    @staticmethod
    def get_nearest_fvg(df):
        try:
            if len(df) < 5: return None
            # Scan up to 30 candles back
            for i in range(len(df)-1, max(2, len(df)-30), -1):
                # Bullish FVG: Low of candle i > High of candle i-2
                if df['l'].iloc[i] > df['h'].iloc[i-2] and df['c'].iloc[i-1] > df['o'].iloc[i-1]:
                    fvg_top, fvg_bottom = df['l'].iloc[i], df['h'].iloc[i-2]
                    mitigated = False
                    for j in range(i+1, len(df)):
                        if df['l'].iloc[j] < fvg_top:
                            mitigated = True
                            break
                    if not mitigated:
                        return {"top": fvg_top, "bottom": fvg_bottom, "type": "BULLISH"}
                # Bearish FVG: High of candle i < Low of candle i-2
                elif df['h'].iloc[i] < df['l'].iloc[i-2] and df['c'].iloc[i-1] < df['o'].iloc[i-1]:
                    fvg_top, fvg_bottom = df['l'].iloc[i-2], df['h'].iloc[i]
                    mitigated = False
                    for j in range(i+1, len(df)):
                        if df['h'].iloc[j] > fvg_bottom:
                            mitigated = True
                            break
                    if not mitigated:
                        return {"top": fvg_top, "bottom": fvg_bottom, "type": "BEARISH"}
        except: return None
        return None
