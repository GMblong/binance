import pandas as pd
import pandas_ta as ta

class MarketAnalyzer:
    @staticmethod
    def get_structure_levels(df, window=5):
        try:
            df_h = df['h'].rolling(window=window*2+1, center=True).max()
            df_l = df['l'].rolling(window=window*2+1, center=True).min()
            hh = df['h'] == df_h
            ll = df['l'] == df_l
            
            recent_highs = df[hh.fillna(False)]['h']
            recent_lows = df[ll.fillna(False)]['l']
            
            rh = recent_highs.iloc[-1] if len(recent_highs) > 0 else df['h'].max()
            rl = recent_lows.iloc[-1] if len(recent_lows) > 0 else df['l'].min()
            return rh, rl
        except:
            return df['h'].max(), df['l'].min()
            
    @staticmethod
    def detect_rsi_divergence(df):
        if len(df) < 30: return False
        try:
            rsi = ta.rsi(df["c"], 14)
            if rsi is None or rsi.empty: return False
            
            # Look at last 15 candles for divergence
            p_curr = df['c'].iloc[-1]
            p_prev = df['c'].iloc[-15:-5].min()
            r_curr = rsi.iloc[-1]
            r_prev = rsi.iloc[-15:-5].min()
            
            # Bullish Divergence: Price Lower Low, RSI Higher Low
            if p_curr < p_prev and r_curr > r_prev:
                return "BULL_DIV"
            
            p_prev_h = df['c'].iloc[-15:-5].max()
            r_prev_h = rsi.iloc[-15:-5].max()
            
            # Bearish Divergence: Price Higher High, RSI Lower High
            if p_curr > p_prev_h and r_curr < r_prev_h:
                return "BEAR_DIV"
        except: pass
        return False

    @staticmethod
    def detect_volume_anomaly(df):
        if len(df) < 25: return False
        try:
            vol_sma = df['v'].rolling(20).mean()
            curr_vol = df['v'].iloc[-1]
            prev_vol = df['v'].iloc[-2]
            
            # 3x Volume Spike
            if curr_vol > vol_sma.iloc[-1] * 3 or prev_vol > vol_sma.iloc[-2] * 3:
                is_bull = df['c'].iloc[-1] > df['o'].iloc[-1]
                return "BULL_VOL" if is_bull else "BEAR_VOL"
        except: pass
        return False

    @staticmethod
    def find_nearest_order_block(df, current_price, direction):
        if len(df) < 50: return None
        try:
            # Enhanced OB: Look for the last candle of opposite direction 
            # that was swallowed by a massive institutional move (impulse)
            body_sizes = (df['c'] - df['o']).abs()
            avg_body = body_sizes.tail(30).mean()
            
            if direction == 1: # Looking for Bullish OB (Support)
                for i in range(len(df)-2, 10, -1):
                    # Impulse check: current candle is bullish and > 2.0x average
                    if df['c'].iloc[i] > df['o'].iloc[i] and body_sizes.iloc[i] > (avg_body * 2.0):
                        # Find the last bearish candle before this impulse
                        for j in range(i-1, i-6, -1):
                            if df['c'].iloc[j] < df['o'].iloc[j]:
                                return {
                                    "high": df['h'].iloc[j], 
                                    "low": df['l'].iloc[j], 
                                    "mid": (df['h'].iloc[j] + df['l'].iloc[j]) / 2,
                                    "index": j,
                                    "type": "BULL_OB"
                                }
            else: # Looking for Bearish OB (Resistance)
                for i in range(len(df)-2, 10, -1):
                    if df['c'].iloc[i] < df['o'].iloc[i] and body_sizes.iloc[i] > (avg_body * 2.0):
                        for j in range(i-1, i-6, -1):
                            if df['c'].iloc[j] > df['o'].iloc[j]:
                                return {
                                    "high": df['h'].iloc[j], 
                                    "low": df['l'].iloc[j], 
                                    "mid": (df['h'].iloc[j] + df['l'].iloc[j]) / 2,
                                    "index": j,
                                    "type": "BEAR_OB"
                                }
        except: pass
        return None

    @staticmethod
    def predict_liquidation_clusters(df):
        if len(df) < 50: return None
        try:
            # Institutional Liquidity: Look for 'Equal Highs/Lows' or major swing points
            highs = df['h'].tail(60)
            lows = df['l'].tail(60)
            
            # Find clusters of highs/lows within 0.1% of each other
            def find_clusters(series):
                sorted_vals = sorted(series.tolist())
                clusters = []
                for i in range(len(sorted_vals)-1):
                    if abs(sorted_vals[i+1] - sorted_vals[i]) / sorted_vals[i] < 0.001:
                        clusters.append(sorted_vals[i])
                return clusters

            upper_clusters = find_clusters(highs)
            lower_clusters = find_clusters(lows)
            
            # Use max/min of clusters or absolute peaks as major liquidity zones
            main_upper = max(upper_clusters) if upper_clusters else highs.max()
            main_lower = min(lower_clusters) if lower_clusters else lows.min()
            
            # Second layer: Secondary clusters for multi-TP
            secondary_upper = min(upper_clusters) if upper_clusters else main_upper
            secondary_lower = max(lower_clusters) if lower_clusters else main_lower

            curr_c = df['c'].iloc[-1]
            curr_l = df['l'].iloc[-1]
            curr_h = df['h'].iloc[-1]
            
            # SWEEP LOGIC: Price poked below support but closed above (wick rejection)
            sweep = None
            if curr_l < main_lower and curr_c > main_lower:
                sweep = "LIQ_SWEPT_BULL"
            elif curr_h > main_upper and curr_c < main_upper:
                sweep = "LIQ_SWEPT_BEAR"
                
            return {
                "upper": main_upper, 
                "lower": main_lower, 
                "upper_inner": secondary_upper,
                "lower_inner": secondary_lower,
                "sweep": sweep
            }
        except: return None

    @staticmethod
    def detect_volatility_breakout(df):
        if len(df) < 20: return False
        try:
            # 1. Price Squeeze Check
            bb = ta.bbands(df["c"], length=20, std=2)
            if bb is None: return False
            bw = (bb.iloc[-1, 2] - bb.iloc[-1, 0]) / bb.iloc[-1, 1] * 100
            
            # 2. Simple Linear Regression on Volume (Manual)
            y = df['v'].tail(10).tolist()
            x = list(range(len(y)))
            n = len(y)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(i*j for i,j in zip(x,y))
            sum_x2 = sum(i**2 for i in x)
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
            
            # Predictive: Tight Squeeze + Rising Volume Slope
            if bw < 1.5 and slope > 0:
                return "VOL_BREAKOUT_IMMINENT"
        except: pass
        return False

    @staticmethod
    def detect_regime(df):
        if len(df) < 20: return "RANGING"
        try:
            atr = ta.atr(df["h"], df["l"], df["c"], length=14).iloc[-2]
            atr_pct = (atr / df["c"].iloc[-2]) * 100
            adx = ta.adx(df["h"], df["l"], df["c"], length=14).iloc[-2, 0]
            rsi = ta.rsi(df["c"], 14).iloc[-2]
            
            bb = ta.bbands(df["c"], length=20, std=2)
            bb_width = 2.0
            if bb is not None:
                bb_width = (bb.iloc[-1, 2] - bb.iloc[-1, 0]) / bb.iloc[-1, 1] * 100

            if adx > 25:
                if (rsi > 70 or rsi < 30) and atr_pct > 1.2: return "EXHAUSTION"
                return "VOLATILE" if atr_pct > 1.5 else "TRENDING"
            else:
                if bb_width < 1.0: return "SQUEEZE"
                return "VOLATILE" if atr_pct > 2.0 else "RANGING"
        except: return "RANGING"

    @staticmethod
    def detect_sweep(df, window=5):
        if len(df) < 20: return False
        
        # Identify recent pivots
        df['hh_p'] = df['h'] == df['h'].rolling(window=window*2+1, center=True).max()
        df['ll_p'] = df['l'] == df['l'].rolling(window=window*2+1, center=True).min()
        
        pivots_h = df[df['hh_p'].fillna(False)]['h'].tail(3).tolist()
        pivots_l = df[df['ll_p'].fillna(False)]['l'].tail(3).tolist()
        
        if not pivots_h or not pivots_l: return False
        
        last_high = pivots_h[-1]
        last_low = pivots_l[-1]
        
        curr_h = df['h'].iloc[-1]
        curr_l = df['l'].iloc[-1]
        curr_c = df['c'].iloc[-1]
        
        # BULL SWEEP: Price went below last low (swept liquidity) but closed above it
        if curr_l < last_low and curr_c > last_low:
            return "BULL_SWEEP"
            
        # BEAR SWEEP: Price went above last high but closed below it
        if curr_h > last_high and curr_c < last_high:
            return "BEAR_SWEEP"
            
        return False

    @staticmethod
    def detect_structure(df, window=5):
        if len(df) < 20: return "CHOP", False
        
        df['hh'] = df['h'] == df['h'].rolling(window=window*2+1, center=True).max()
        df['ll'] = df['l'] == df['l'].rolling(window=window*2+1, center=True).min()
        
        pivots_h = df[df['hh'].fillna(False)]['h'].tail(3).tolist()
        pivots_l = df[df['ll'].fillna(False)]['l'].tail(3).tolist()
        
        struct = "CHOP"
        if len(pivots_h) >= 2 and len(pivots_l) >= 2:
            if pivots_h[-1] > pivots_h[-2] and pivots_l[-1] > pivots_l[-2]:
                struct = "BULLISH"
            elif pivots_h[-1] < pivots_h[-2] and pivots_l[-1] < pivots_l[-2]:
                struct = "BEARISH"
        
        fvg = False
        body_sizes = (df['c'] - df['o']).abs()
        avg_body = body_sizes.tail(20).mean()
        
        if len(df) >= 4:
            gap_size_bull = df['l'].iloc[-2] - df['h'].iloc[-4]
            if gap_size_bull > 0 and body_sizes.iloc[-3] > avg_body:
                fvg = "BULL_FVG"
                
            gap_size_bear = df['l'].iloc[-4] - df['h'].iloc[-2]
            if gap_size_bear > 0 and body_sizes.iloc[-3] > avg_body:
                fvg = "BEAR_FVG"
                
        return struct, fvg

    @staticmethod
    def get_nearest_fvg(df):
        """Identifies the most recent Fair Value Gap for aggressive entry."""
        if len(df) < 5: return None
        try:
            # Bullish FVG (Gap between Candle 1 High and Candle 3 Low)
            # 1 [H] ... 2 [Big Body] ... 3 [L]
            c1_h = df['h'].iloc[-3]
            c3_l = df['l'].iloc[-1]
            if c3_l > c1_h:
                return {"type": "BULL_FVG", "top": c3_l, "bottom": c1_h, "mid": (c3_l + c1_h) / 2}
            
            # Bearish FVG
            c1_l = df['l'].iloc[-3]
            c3_h = df['h'].iloc[-1]
            if c3_h < c1_l:
                return {"type": "BEAR_FVG", "top": c1_l, "bottom": c3_h, "mid": (c1_l + c3_h) / 2}
        except: pass
        return None

    @staticmethod
    def calculate_score(d1m, d5m, trend_dir, predictive_flags=None, neural_weights=None, oi_delta=0):
        score = 0
        predictive_flags = predictive_flags or {}
        weights = neural_weights or {"liq": 1.0, "ml": 1.0, "ob": 1.0, "div": 1.0}
        
        rsi = ta.rsi(d1m["c"], 14).iloc[-2]
        if pd.isna(rsi): return 0
        
        # --- INSTITUTIONAL CONFIRMATION (OI) ---
        # Price Up + OI Up = Strong Bullish
        # Price Down + OI Up = Strong Bearish
        if trend_dir == 1 and oi_delta > 0.01: # 1% increase in OI
            score += 25
        elif trend_dir == -1 and oi_delta > 0.01:
            score += 25
        elif oi_delta < -0.02: # OI dropping fast (Liquidation or Profit Taking)
            score -= 20
        
        # Base RSI Logic (Max +20)
        if trend_dir == 1:
            if 40 <= rsi <= 65: score += 20 
            elif rsi > 75: score -= 30 
        else:
            if 35 <= rsi <= 60: score += 20 
            elif rsi < 25: score -= 30 
        
        # Institutional Confirmations (Massive Bonuses)
        div = MarketAnalyzer.detect_rsi_divergence(d1m)
        if trend_dir == 1 and div == "BULL_DIV": score += (25 * weights.get("div", 1.0))
        elif trend_dir == -1 and div == "BEAR_DIV": score += (25 * weights.get("div", 1.0))

        vol_anomaly = MarketAnalyzer.detect_volume_anomaly(d1m)
        if trend_dir == 1 and vol_anomaly == "BULL_VOL": score += 20
        elif trend_dir == -1 and vol_anomaly == "BEAR_VOL": score += 20

        vwap = (d1m['v'] * (d1m['h'] + d1m['l'] + d1m['c']) / 3).cumsum() / d1m['v'].cumsum()
        if trend_dir == 1 and (d1m['c'].iloc[-2] > vwap.iloc[-2]): score += 15
        elif trend_dir == -1 and (d1m['c'].iloc[-2] < vwap.iloc[-2]): score += 15
        
        # Momentum & Structure (Max +50)
        vol_ma = d1m["v"].rolling(20).mean().iloc[-2]
        v_closed = d1m["v"].iloc[-2]
        if v_closed > vol_ma * 1.3:
            score += 15
            
        ema9_1m = ta.ema(d1m["c"], 9).iloc[-2]
        ema21_1m = ta.ema(d1m["c"], 21).iloc[-2]
        if (trend_dir == 1 and ema9_1m > ema21_1m) or (trend_dir == -1 and ema9_1m < ema21_1m):
            score += 15

        struct_1m, fvg_1m = MarketAnalyzer.detect_structure(d1m)
        if (trend_dir == 1 and struct_1m == "BULLISH") or (trend_dir == -1 and struct_1m == "BEARISH"):
            score += 20
        elif fvg_1m:
            score += 10
        
        # --- ULTIMATE PREDICTIVE BONUSES ---
        if predictive_flags.get("liq") == "LIQ_SWEPT_BULL" and trend_dir == 1: 
            score += (30 * weights.get("liq", 1.0))
        elif predictive_flags.get("liq") == "LIQ_SWEPT_BEAR" and trend_dir == -1: 
            score += (30 * weights.get("liq", 1.0))
        
        if predictive_flags.get("ml") == "VOL_BREAKOUT_IMMINENT": 
            score += (25 * weights.get("ml", 1.0))
        
        return min(max(score, 0), 100)
