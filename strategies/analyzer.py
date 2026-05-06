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
    def calculate_score(d1m, d15m, direction):
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
            
            # 4. Smart Money & Anomalies (1m / 15m)
            if MarketAnalyzer.detect_volume_anomaly(d1m):
                score += 10 # High volume supports the move
            
            div = MarketAnalyzer.detect_rsi_divergence(d15m)
            if direction == 1 and div == 1: score += 20
            elif direction == -1 and div == -1: score += 20
            
            fvg = MarketAnalyzer.get_nearest_fvg(d1m)
            if fvg:
                if direction == 1 and fvg["type"] == "BULLISH": score += 10
                elif direction == -1 and fvg["type"] == "BEARISH": score += 10
                
            ob = MarketAnalyzer.find_nearest_order_block(d1m, d1m['c'].iloc[-1], direction)
            if ob: score += 10

            return min(score, 100)
        except: return 0

    @staticmethod
    def get_structure_levels(df):
        return df['h'].max(), df['l'].min()
    
    @staticmethod
    def find_nearest_order_block(df, price, dir):
        try:
            if len(df) < 10: return None
            for i in range(len(df)-2, max(0, len(df)-20), -1):
                if dir == 1 and df['c'].iloc[i] < df['o'].iloc[i]: 
                    return {"top": df['h'].iloc[i], "bottom": df['l'].iloc[i]}
                elif dir == -1 and df['c'].iloc[i] > df['o'].iloc[i]: 
                    return {"top": df['h'].iloc[i], "bottom": df['l'].iloc[i]}
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
            if len(df) < 3: return None
            if df['l'].iloc[-1] > df['h'].iloc[-3] and df['c'].iloc[-2] > df['o'].iloc[-2]:
                return {"top": df['l'].iloc[-1], "bottom": df['h'].iloc[-3], "type": "BULLISH"}
            if df['h'].iloc[-1] < df['l'].iloc[-3] and df['c'].iloc[-2] < df['o'].iloc[-2]:
                return {"top": df['l'].iloc[-3], "bottom": df['h'].iloc[-1], "type": "BEARISH"}
        except: return None
        return None
