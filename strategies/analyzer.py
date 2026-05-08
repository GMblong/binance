import pandas as pd
import numpy as np

class MarketAnalyzer:
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
            # Percentile-based: adapt to each coin's own volatility history
            atr_p75 = atr_pct.tail(100).quantile(0.75) if len(atr_pct) >= 100 else atr_pct.quantile(0.75)
            atr_p25 = atr_pct.tail(100).quantile(0.25) if len(atr_pct) >= 100 else atr_pct.quantile(0.25)
            
            if curr_atr_p > atr_p75: return "VOLATILE"
            
            # Z-score of price distance from EMA for trend detection
            dist = ((close - ema20) / ema20 * 100).tail(30)
            dist_std = dist.std()
            curr_dist = dist.iloc[-1]
            if dist_std > 0 and abs(curr_dist) > dist_std * 1.2: return "TRENDING"
            return "RANGING"
        except: return "RANGING"

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
    def calculate_score(d1m, d15m, direction, imbalance=1.0, funding=0.0, regime="RANGING", neural_weights=None, session="QUIET", lead_lag=0):
        try:
            score = 0
            nw = neural_weights or {}
            
            # 1. Macro Trend Alignment (15m) & Soft Mean Reversion (H1)
            ema9_15 = MarketAnalyzer.get_ema(d15m["c"], 9).iloc[-1]
            ema21_15 = MarketAnalyzer.get_ema(d15m["c"], 21).iloc[-1]
            dir_15 = 1 if ema9_15 > ema21_15 else -1
            
            rsi_15m = MarketAnalyzer.get_rsi(d15m["c"], 14).iloc[-1]
            
            if direction == dir_15: 
                score += 30 # Heavy weight for macro alignment
            else:
                # Soft alignment: Allow mean reversion if ranging and RSI extreme
                if regime == "RANGING":
                    vsa_sig = MarketAnalyzer.detect_vsa_signals(d1m)
                    # Counter-trend valid only if RSI is extreme OR there's a strong VSA reversal signal
                    if (direction == 1 and rsi_15m < 35) or (direction == -1 and rsi_15m > 65) or vsa_sig == direction:
                        score -= 15 # Mild penalty, but allowed to trade (Mean Reversion)
                    else:
                        return 0
                else:
                    return 0 # Still strict in trending/volatile markets
                
            # 2. Micro Momentum (1m)
            ema9 = MarketAnalyzer.get_ema(d1m["c"], 9).iloc[-1]
            ema21 = MarketAnalyzer.get_ema(d1m["c"], 21).iloc[-1]
            if direction == 1 and ema9 > ema21: score += 10
            if direction == -1 and ema9 < ema21: score += 10
            
            # 3. RSI Momentum (15m is more reliable than 1m)
            rsi = MarketAnalyzer.get_rsi(d15m["c"], 14).iloc[-1]
            if direction == 1 and 40 < rsi < 65: score += 10 # Room to grow upward
            if direction == -1 and 35 < rsi < 60: score += 10 # Room to drop downward
            
            # 4. Institutional Data (Orderbook & Funding)
            if direction == 1:
                if imbalance > 1.2: score += 10 # Buy wall supports LONG
                elif imbalance < 0.8: score -= 15 # Sell wall blocks LONG
            elif direction == -1:
                if imbalance < 0.8: score += 10 # Sell wall supports SHORT
                elif imbalance > 1.2: score -= 15 # Buy wall blocks SHORT
                
            # Funding Rate (Contrarian Indicator)
            if direction == 1 and funding > 0.0005: score -= 10
            if direction == -1 and funding < -0.0005: score -= 10
            
            # 5. Smart Money & Anomalies (Weighted by Neural Weights)
            liq_w = nw.get(f"{regime}:liq", 1.0)
            ob_w = nw.get(f"{regime}:ob", 1.0)
            div_w = nw.get(f"{regime}:div", 1.0)

            sweep = MarketAnalyzer.detect_liquidity_sweep(d1m)
            if direction == 1 and sweep == 1: score += int(15 * liq_w)
            elif direction == -1 and sweep == -1: score += int(15 * liq_w)
            
            if MarketAnalyzer.detect_volume_anomaly(d1m):
                score += 5 # High volume supports the move
            
            div = MarketAnalyzer.detect_rsi_divergence(d15m)
            if direction == 1 and div == 1: score += int(15 * div_w)
            elif direction == -1 and div == -1: score += int(15 * div_w)
            
            fvg = MarketAnalyzer.get_nearest_fvg(d1m)
            if fvg:
                if direction == 1 and fvg["type"] == "BULLISH": score += int(10 * liq_w)
                elif direction == -1 and fvg["type"] == "BEARISH": score += int(10 * liq_w)
                
            ob = MarketAnalyzer.find_nearest_order_block(d1m, d1m['c'].iloc[-1], direction)
            if ob: score += int(10 * ob_w)

            # 6. Advanced Analysis (Volume Profile, VSA, Fractal Confluence)
            # Fractal Confluence: Check if 1m structure aligns with 15m
            s1m, _, _, _ = MarketAnalyzer.detect_structure(d1m)
            s15m, _, _, _ = MarketAnalyzer.detect_structure(d15m)
            if direction == 1 and s1m == "BULLISH" and s15m == "BULLISH": score += 10
            elif direction == -1 and s1m == "BEARISH" and s15m == "BEARISH": score += 10
            
            # Volume Profile (POC)
            vp = MarketAnalyzer.get_volume_profile(d15m)
            if vp:
                price = d1m['c'].iloc[-1]
                if direction == 1 and price > vp['poc']: score += 5 
                elif direction == -1 and price < vp['poc']: score += 5 
            
            # VSA (Absorption & Supply/Demand)
            vsa_sig = MarketAnalyzer.detect_vsa_signals(d1m)
            if vsa_sig == direction: score += 15
            
            # 7. Session & Lead-Lag Context
            if session in ["LONDON", "NEW_YORK"]:
                # Trend following is more reliable in these sessions
                if regime == "TRENDING": score += 5
            
            if lead_lag == direction:
                score += 15 # Strongly follow the leader

            return max(0, min(score, 100))
        except: return 0

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
