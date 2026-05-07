import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time
from utils.state import market_data, bot_state

def get_current_session():
    """Returns the current global trading session."""
    now = datetime.utcnow().time()
    
    # Session Windows (UTC)
    # Asia: 00:00 - 09:00
    # London: 08:00 - 17:00
    # New York: 13:00 - 22:00
    
    if dt_time(13, 0) <= now <= dt_time(22, 0): return "NEW_YORK"
    if dt_time(8, 0) <= now <= dt_time(17, 0): return "LONDON"
    if dt_time(0, 0) <= now <= dt_time(9, 0): return "ASIA"
    return "QUIET"

def calculate_market_volatility():
    """Calculates a global volatility index based on top symbols."""
    vols = []
    for symbol in market_data.current_scan_list:
        df = market_data.klines.get(symbol, {}).get("15m")
        if df is not None and not df.empty:
            # 15m ATR / Price as a percentage
            high_low = df['h'] - df['l']
            avg_price = df['c'].rolling(14).mean()
            vol = (high_low.rolling(14).mean() / avg_price) * 100
            if not pd.isna(vol.iloc[-1]):
                vols.append(vol.iloc[-1])
    
    if vols:
        avg_vol = sum(vols) / len(vols)
        # Normalize: 1.0 is "normal", > 1.5 is high, < 0.7 is low
        bot_state["market_vol"] = round(avg_vol / 1.0, 2) # Assuming 1.0% is baseline
    return bot_state["market_vol"]

def get_symbol_correlation(sym1, sym2):
    """Calculates correlation between two symbols based on 15m klines."""
    try:
        df1 = market_data.klines.get(sym1, {}).get("15m")
        df2 = market_data.klines.get(sym2, {}).get("15m")
        
        if df1 is None or df2 is None or df1.empty or df2.empty:
            return 0.5 # Neutral if no data
            
        # Use returns for correlation
        ret1 = df1['c'].pct_change().tail(20)
        ret2 = df2['c'].pct_change().tail(20)
        
        corr = ret1.corr(ret2)
        return corr if not pd.isna(corr) else 0.5
    except:
        return 0.5

def is_correlated_exposure(new_symbol, new_side):
    """Checks if opening a new position adds too much correlation to existing ones."""
    active_positions = bot_state.get("active_positions", [])
    if not active_positions:
        return False
        
    for pos in active_positions:
        pos_sym = pos['symbol']
        pos_side = "LONG" if float(pos['positionAmt']) > 0 else "SHORT"
        
        # If same direction, check correlation
        if pos_side == new_side:
            corr = get_symbol_correlation(new_symbol, pos_sym)
            if corr > 0.80: # Highly correlated
                return True
                
    return False

def calculate_kelly_risk(symbol, win_rate=0.5, rr=2.0):
    """Calculates a fractional Kelly multiplier based on historical performance."""
    try:
        # K = W - ((1 - W) / R)
        kelly = win_rate - ((1 - win_rate) / rr)
        # Use Half-Kelly for safety and cap it between 0.5x and 1.5x of base risk
        multiplier = max(0.5, min(1.5, (kelly * 0.5) / 0.125)) # 0.125 is half-kelly for 50% WR/2RR
        return round(multiplier, 2)
    except:
        return 1.0

def detect_lead_lag(symbol, leader="BTCUSDT"):
    """
    Detects if the symbol is lagging behind a market leader.
    Returns: 1 (Lagging Bullish), -1 (Lagging Bearish), 0 (Neutral)
    """
    try:
        df_sym = market_data.klines.get(symbol, {}).get("1m")
        df_lead = market_data.klines.get(leader, {}).get("1m")
        
        if df_sym is None or df_lead is None or len(df_sym) < 5 or len(df_lead) < 5:
            return 0
            
        sym_ret = (df_sym['c'].iloc[-1] - df_sym['c'].iloc[-5]) / df_sym['c'].iloc[-5] * 100
        lead_ret = (df_lead['c'].iloc[-1] - df_lead['c'].iloc[-5]) / df_lead['c'].iloc[-5] * 100
        
        # If leader moved > 0.3% and sym moved < 0.1% in same direction
        if lead_ret > 0.3 and sym_ret < 0.1: return 1 # Bullish lag
        if lead_ret < -0.3 and sym_ret > -0.1: return -1 # Bearish lag
        
        return 0
    except:
        return 0
