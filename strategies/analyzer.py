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
    def detect_regime(df):
        if len(df) < 30: return "RANGING"
        try:
            close = df["c"]
            ema20 = MarketAnalyzer.get_ema(close, 20)
            atr = MarketAnalyzer.get_atr(df, 14)
            atr_pct = (atr / close) * 100
            
            curr_c = close.iloc[-1]
            curr_ema = ema20.iloc[-1]
            curr_atr_p = atr_pct.iloc[-1]
            
            if curr_atr_p > 1.5: return "VOLATILE"
            if abs(curr_c - curr_ema) / curr_ema * 100 > 0.5: return "TRENDING"
            return "RANGING"
        except: return "RANGING"

    @staticmethod
    def detect_structure(df):
        if len(df) < 10: return "CHOP", False
        try:
            # Simple Higher High / Lower Low logic
            h = df['h'].tail(5)
            l = df['l'].tail(5)
            if h.iloc[-1] > h.iloc[-2] and l.iloc[-1] > l.iloc[-2]: return "BULLISH", False
            if h.iloc[-1] < h.iloc[-2] and l.iloc[-1] < l.iloc[-2]: return "BEARISH", False
            return "CHOP", False
        except: return "CHOP", False

    @staticmethod
    def calculate_score(d1m, d15m, direction, imbalance=1.0, funding=0.0):
        try:
            score = 0
            
            # 1. Macro Trend Alignment (15m) - CRITICAL
            ema9_15 = MarketAnalyzer.get_ema(d15m["c"], 9).iloc[-1]
            ema21_15 = MarketAnalyzer.get_ema(d15m["c"], 21).iloc[-1]
            dir_15 = 1 if ema9_15 > ema21_15 else -1
            if direction == dir_15: 
                score += 35 # Heavy weight for macro alignment
            else:
                return 0 # Do not trade against the 15m trend
                
            # 2. Micro Momentum (1m)
            ema9 = MarketAnalyzer.get_ema(d1m["c"], 9).iloc[-1]
            ema21 = MarketAnalyzer.get_ema(d1m["c"], 21).iloc[-1]
            if direction == 1 and ema9 > ema21: score += 10
            if direction == -1 and ema9 < ema21: score += 10
            
            # 3. RSI Momentum (15m is more reliable than 1m)
            rsi = MarketAnalyzer.get_rsi(d15m["c"], 14).iloc[-1]
            if direction == 1 and 40 < rsi < 65: score += 15 # Room to grow upward
            if direction == -1 and 35 < rsi < 60: score += 15 # Room to drop downward
            
            # 4. Institutional Data (Orderbook & Funding)
            # Imbalance > 1.2 means strong bid wall (buyers). < 0.8 means strong ask wall (sellers).
            if direction == 1:
                if imbalance > 1.2: score += 10 # Buy wall supports LONG
                elif imbalance < 0.8: score -= 15 # Sell wall blocks LONG
            elif direction == -1:
                if imbalance < 0.8: score += 10 # Sell wall supports SHORT
                elif imbalance > 1.2: score -= 15 # Buy wall blocks SHORT
                
            # Funding Rate (Contrarian Indicator)
            # If funding is very high positive, market is heavily long (crowded), bad for new longs.
            if direction == 1 and funding > 0.0005: score -= 10
            if direction == -1 and funding < -0.0005: score -= 10
            
            # 5. Smart Money & Anomalies (1m / 15m)
            sweep = MarketAnalyzer.detect_liquidity_sweep(d1m)
            if direction == 1 and sweep == 1: score += 20 # High conviction reversal
            elif direction == -1 and sweep == -1: score += 20
            
            if MarketAnalyzer.detect_volume_anomaly(d1m):
                score += 10 # High volume supports the move
            
            div = MarketAnalyzer.detect_rsi_divergence(d15m)
            if direction == 1 and div == 1: score += 15
            elif direction == -1 and div == -1: score += 15
            
            fvg = MarketAnalyzer.get_nearest_fvg(d1m)
            if fvg:
                if direction == 1 and fvg["type"] == "BULLISH": score += 10
                elif direction == -1 and fvg["type"] == "BEARISH": score += 10
                
            ob = MarketAnalyzer.find_nearest_order_block(d1m, d1m['c'].iloc[-1], direction)
            if ob: score += 10

            return max(0, min(score, 100))
        except: return 0

    @staticmethod
    def detect_liquidity_sweep(df):
        try:
            if len(df) < 20: return 0
            # Check for sweep of previous 15-candle high/low
            prev_high = df['h'].iloc[-15:-1].max()
            prev_low = df['l'].iloc[-15:-1].min()
            
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
            # Scan deeper (up to 50 candles back)
            for i in range(len(df)-3, max(0, len(df)-50), -1):
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
                            if df['h'].iloc[j] > ob_bottom:
                                mitigated = True
                                break
                        if not mitigated:
                            return {"top": ob_top, "bottom": ob_bottom, "type": "BEARISH"}
        except: return None
        return None

    @staticmethod
    def predict_liquidation_clusters(df): return None
    
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
        try:
            if len(df) < 30: return 0
            rsi = MarketAnalyzer.get_rsi(df['c'], 14)
            p_l = df['l'].iloc[-5:].min(); pp_l = df['l'].iloc[-20:-5].min()
            r_l = rsi.iloc[-5:].min(); pr_l = rsi.iloc[-20:-5].min()
            if p_l < pp_l and r_l > pr_l: return 1
                
            p_h = df['h'].iloc[-5:].max(); pp_h = df['h'].iloc[-20:-5].max()
            r_h = rsi.iloc[-5:].max(); pr_h = rsi.iloc[-20:-5].max()
            if p_h > pp_h and r_h < pr_h: return -1
        except: return 0
        return 0

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
