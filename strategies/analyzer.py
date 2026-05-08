import pandas as pd
import numpy as np
from functools import lru_cache

class MarketAnalyzer:
    # HMM parameters (pre-fitted on typical crypto regimes)
    # State 0: Low volatility / Mean-reverting
    # State 1: High volatility / Trending
    _hmm_transition = np.array([[0.95, 0.05], [0.10, 0.90]])  # Sticky states
    _hmm_means = np.array([0.0, 0.0])  # Returns are ~0 mean in both
    _hmm_stds = np.array([0.003, 0.012])  # Low vol vs high vol

    @staticmethod
    def detect_hmm_regime(df, lookback=50):
        """Hidden Markov Model regime detection using Viterbi-like forward pass.
        
        Returns: 'MEAN_REVERT' or 'MOMENTUM' based on most likely current state.
        """
        if len(df) < lookback:
            return "UNKNOWN"
        try:
            returns = df['c'].pct_change().tail(lookback).dropna().values
            if len(returns) < 10:
                return "UNKNOWN"
            
            trans = MarketAnalyzer._hmm_transition
            means = MarketAnalyzer._hmm_means
            stds = MarketAnalyzer._hmm_stds
            n_states = 2
            
            # Forward algorithm (log space for numerical stability)
            log_alpha = np.zeros(n_states)
            log_alpha[0] = -0.5  # Slight prior for low-vol state
            log_alpha[1] = -0.5
            
            for r in returns:
                # Emission probability (Gaussian)
                log_emit = np.array([
                    -0.5 * ((r - means[s]) / stds[s])**2 - np.log(stds[s])
                    for s in range(n_states)
                ])
                # Transition + previous state
                new_alpha = np.zeros(n_states)
                for j in range(n_states):
                    new_alpha[j] = log_emit[j] + np.logaddexp(
                        log_alpha[0] + np.log(trans[0, j]),
                        log_alpha[1] + np.log(trans[1, j])
                    )
                log_alpha = new_alpha
            
            # Most likely current state
            current_state = np.argmax(log_alpha)
            return "MEAN_REVERT" if current_state == 0 else "MOMENTUM"
        except:
            return "UNKNOWN"
    @staticmethod
    def get_ema(series, length):
        return series.ewm(span=length, adjust=False).mean()

    @staticmethod
    def get_rsi(series, length=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def get_atr(df, length=14):
        high_low = df['h'] - df['l']
        high_cp = (df['h'] - df['c'].shift(1)).abs()
        low_cp = (df['l'] - df['c'].shift(1)).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        return tr.rolling(window=length).mean()

    @staticmethod
    def get_adx(df, length=14):
        """Calculate ADX (Average Directional Index). Returns scalar value."""
        try:
            if len(df) < length * 2: return 25  # Default neutral
            high, low, close = df['h'], df['l'], df['c']
            plus_dm = high.diff().clip(lower=0)
            minus_dm = (-low.diff()).clip(lower=0)
            # Zero out when other is larger
            plus_dm[plus_dm < minus_dm] = 0
            minus_dm[minus_dm < plus_dm] = 0
            tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.ewm(span=length, adjust=False).mean()
            plus_di = 100 * (plus_dm.ewm(span=length, adjust=False).mean() / atr)
            minus_di = 100 * (minus_dm.ewm(span=length, adjust=False).mean() / atr)
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)) * 100
            adx = dx.ewm(span=length, adjust=False).mean()
            return float(adx.iloc[-1])
        except:
            return 25

    @staticmethod
    def get_volume_profile(df, bins=20):
        try:
            if len(df) < 30: return None
            # Calculate price bins
            price_min, price_max = df['l'].min(), df['h'].max()
            if price_max == price_min: return None
            
            bin_size = (price_max - price_min) / bins
            df['bin'] = ((df['c'] - price_min) / bin_size).astype(int).clip(0, bins-1)
            
            # Aggregate volume by bin
            profile = df.groupby('bin')['v'].sum()
            poc_bin = profile.idxmax()
            poc_price = price_min + (poc_bin * bin_size) + (bin_size / 2)
            
            # Value Area (70% of volume)
            sorted_profile = profile.sort_values(ascending=False)
            total_vol = profile.sum()
            va_vol = 0
            va_bins = []
            for b, v in sorted_profile.items():
                va_vol += v
                va_bins.append(b)
                if va_vol >= total_vol * 0.7: break
            
            va_high = price_min + (max(va_bins) * bin_size) + bin_size
            va_low = price_min + (min(va_bins) * bin_size)
            
            return {"poc": poc_price, "vah": va_high, "val": va_low}
        except: return None

    @staticmethod
    def detect_vsa_signals(df):
        """Advanced VSA (Volume Spread Analysis) detection."""
        try:
            if len(df) < 5: return 0
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            vol_sma = df['v'].tail(20).mean()
            
            is_high_vol = curr['v'] > (vol_sma * 1.5)
            is_low_vol = curr['v'] < (vol_sma * 0.7)
            body = abs(curr['c'] - curr['o'])
            range_ = curr['h'] - curr['l']
            
            # 1. Absorption
            lower_wick = min(curr['o'], curr['c']) - curr['l']
            if is_high_vol and lower_wick > (range_ * 0.6) and body < (range_ * 0.3):
                return 1 # Bullish Absorption
                
            upper_wick = curr['h'] - max(curr['o'], curr['c'])
            if is_high_vol and upper_wick > (range_ * 0.6) and body < (range_ * 0.3):
                return -1 # Bearish Absorption

            # 2. No Demand / No Supply
            # No Supply: Low volume, narrow range, down candle (Smart money not interested in selling)
            if is_low_vol and range_ < (df['h'] - df['l']).tail(10).mean() * 0.8 and curr['c'] < curr['o']:
                return 1 # Potential Bullish Reversal / Strength
                
            # No Demand: Low volume, narrow range, up candle (Smart money not interested in buying)
            if is_low_vol and range_ < (df['h'] - df['l']).tail(10).mean() * 0.8 and curr['c'] > curr['o']:
                return -1 # Potential Bearish Reversal / Weakness

            # 3. Stopping Volume
            if is_high_vol and curr['c'] < prev['c'] and curr['c'] > curr['l'] + (range_ * 0.4):
                return 1 # Stopping Volume (Bullish)

            # 4. Volume-Price Divergence
            # Price making higher high but volume is lower (Weakness)
            if curr['h'] > df['h'].iloc[-5:-1].max() and curr['v'] < df['v'].iloc[-5:-1].mean() * 0.8:
                return -1 # Bearish Divergence (Exhaustion)
            # Price making lower low but volume is lower (Weakness)
            if curr['l'] < df['l'].iloc[-5:-1].min() and curr['v'] < df['v'].iloc[-5:-1].mean() * 0.8:
                return 1 # Bullish Divergence (Exhaustion)

            return 0
        except: return 0

    @staticmethod
    def detect_regime(df):
        if len(df) < 30: return "RANGING"
        try:
            close = df["c"]
            ema20 = MarketAnalyzer.get_ema(close, 20)
            atr = MarketAnalyzer.get_atr(df, 14)
            atr_pct = (atr / close) * 100
            
            curr_atr_p = atr_pct.iloc[-1]
            atr_p75 = atr_pct.tail(100).quantile(0.75) if len(atr_pct) >= 100 else atr_pct.quantile(0.75)
            
            if curr_atr_p > atr_p75: return "VOLATILE"
            
            # Z-score of price distance from EMA for trend detection
            dist = ((close - ema20) / ema20 * 100).tail(30)
            dist_std = dist.std()
            curr_dist = dist.iloc[-1]
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
            # Multi-candle structure detection (H10) using Rolling Max/Min (Fractals)
            # Find the actual swing points in the recent window
            recent_high = df['h'].tail(15).max()
            recent_low = df['l'].tail(15).min()
            
            # Simple fractal detection (lookback 2, lookforward 2)
            highs = df['h'].values
            lows = df['l'].values
            
            pivot_highs = []
            pivot_lows = []
            for i in range(2, len(df)-2):
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    pivot_highs.append(highs[i])
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    pivot_lows.append(lows[i])
                    
            if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
                if pivot_highs[-1] > pivot_highs[-2] and pivot_lows[-1] > pivot_lows[-2]:
                    return "BULLISH", False, recent_high, recent_low
                if pivot_highs[-1] < pivot_highs[-2] and pivot_lows[-1] < pivot_lows[-2]:
                    return "BEARISH", False, recent_high, recent_low
                    
            # Fallback to simple momentum if no clear pivots
            h = df['h'].tail(5)
            l = df['l'].tail(5)
            if h.iloc[-1] > h.iloc[-2] and l.iloc[-1] > l.iloc[-2]: 
                return "BULLISH", False, recent_high, recent_low
            if h.iloc[-1] < h.iloc[-2] and l.iloc[-1] < l.iloc[-2]: 
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
