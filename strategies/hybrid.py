import time
import pandas as pd
import asyncio
from datetime import datetime
from utils.config import API_URL, API_KEY
from utils.state import bot_state, market_data
from strategies.analyzer import MarketAnalyzer
from engine.ml_engine import ml_predictor
from utils.logger import log_error

api_sem = asyncio.Semaphore(10) # Increased to allow true concurrency
busy_symbols = set()

async def get_btc_trend(client):
    try:
        headers = {'X-MBX-APIKEY': API_KEY, 'User-Agent': 'Mozilla/5.0'}
        res = await client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": "BTCUSDT", "interval": "1m", "limit": 100}, headers=headers, timeout=10)
        if res.status_code == 200:
            df = pd.DataFrame(res.json()).iloc[:, [0, 1, 2, 3, 4, 5]].astype(float)
            df.columns = ["ot", "o", "h", "l", "c", "v"]
            ema = MarketAnalyzer.get_ema(df["c"], 20).iloc[-1]
            bot_state["btc_state"] = "BULLISH" if df["c"].iloc[-1] > ema else "BEARISH"
            bot_state["btc_dir"] = 1 if df["c"].iloc[-1] > ema else -1
    except: pass

async def analyze_hybrid_async(client, symbol):
    if symbol in busy_symbols: return None
    try:
        now = time.time()
        if symbol not in market_data.klines: market_data.klines[symbol] = {}
        
        last_p = market_data.last_prime.get(symbol, 0)
        if (now - last_p) > 60: 
            busy_symbols.add(symbol)
            try:
                async with api_sem:
                    headers = {'X-MBX-APIKEY': API_KEY, 'User-Agent': 'Mozilla/5.0'}
                    
                    # Fetch 1m, 15m, and 1h data all at once (concurrently)
                    t1 = client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "1m", "limit": 200}, headers=headers, timeout=15)
                    t15 = client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "15m", "limit": 100}, headers=headers, timeout=15)
                    t1h = client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "1h", "limit": 50}, headers=headers, timeout=15)
                    
                    results = await asyncio.gather(t1, t15, t1h, return_exceptions=True)
                    
                    if not any(isinstance(r, Exception) for r in results):
                        res1, res15, res1h = results
                        if res1.status_code == 200 and res15.status_code == 200 and res1h.status_code == 200:
                            def proc(data):
                                df = pd.DataFrame(data).iloc[:, [0, 1, 2, 3, 4, 5, 9]]
                                df.columns = ["ot", "o", "h", "l", "c", "v", "tbv"]
                                for col in ["o", "h", "l", "c", "v", "tbv"]: df[col] = df[col].astype(float)
                                return df
                            market_data.klines[symbol]["1m"] = proc(res1.json())
                            market_data.klines[symbol]["15m"] = proc(res15.json())
                            market_data.klines[symbol]["1h"] = proc(res1h.json())
                            market_data.last_prime[symbol] = now
                    else:
                        print(f"[{symbol}] Network error fetching klines: {results}")
            finally:
                busy_symbols.discard(symbol)

        k = market_data.klines.get(symbol, {})
        if "1m" not in k or "15m" not in k or "1h" not in k: return None
        
        d1m, d15m, d1h = k["1m"], k["15m"], k["1h"]
        price = d1m["c"].iloc[-1]
        
        # --- SMART INDICATORS ---
        ema9_15m = MarketAnalyzer.get_ema(d15m["c"], 9).iloc[-1]
        ema21_15m = MarketAnalyzer.get_ema(d15m["c"], 21).iloc[-1]
        ema9_1m = MarketAnalyzer.get_ema(d1m["c"], 9).iloc[-1]
        atr = MarketAnalyzer.get_atr(d1m, 14).iloc[-1]
        
        # --- HIGHER TIMEFRAME (HTF) BIAS ---
        ema20_1h = MarketAnalyzer.get_ema(d1h["c"], 20).iloc[-1]
        htf_direction = 1 if d1h["c"].iloc[-1] > ema20_1h else -1
        
        # Primary trend is determined by 15m (Macro Bias)
        direction = 1 if ema9_15m > ema21_15m else -1
        
        # Multi-Timeframe Alignment: Ensure 15m trend aligns with 1h trend
        mtf_aligned = (direction == htf_direction)
        
        # --- INSTITUTIONAL DATA GATHERING ---
        from engine.api import get_orderbook_imbalance
        imbalance = await get_orderbook_imbalance(client, symbol)
        funding = market_data.funding.get(symbol, 0)
        oi = market_data.oi.get(symbol, 0)
        
        score = MarketAnalyzer.calculate_score(d1m, d15m, direction, imbalance, funding)
        regime = MarketAnalyzer.detect_regime(d15m)
        struct, _ = MarketAnalyzer.detect_structure(d1m)
        
        # If HTF is not aligned, we penalize the score significantly
        if not mtf_aligned:
            score -= 40 # Heavy penalty for fighting the 1h trend
            
        # --- MACHINE LEARNING INTEGRATION ---
        # Inject current OI and Funding into the dataframe for feature engineering
        d15m_with_institutional = d15m.copy()
        d15m_with_institutional['oi'] = oi
        d15m_with_institutional['funding'] = funding
        
        # Predict UP probability (0.0 to 1.0)
        ml_prob = await ml_predictor.predict(client, symbol, d15m_with_institutional)
        ml_score_boost = 0
        if direction == 1 and ml_prob > 0.6:
            ml_score_boost = int((ml_prob - 0.5) * 40) # Up to +20 score
        elif direction == -1 and ml_prob < 0.4:
            ml_score_boost = int((0.5 - ml_prob) * 40) # Up to +20 score
            
        # Severe penalty if ML strongly disagrees with TA direction
        if (direction == 1 and ml_prob < 0.4) or (direction == -1 and ml_prob > 0.6):
            ml_score_boost = -30
        
        score = min(max(score + ml_score_boost, 0), 100) # Ensure score is between 0 and 100
        
        # --- SMART ENTRY (SMC / Pullback Logic) ---
        fvg = MarketAnalyzer.get_nearest_fvg(d1m)
        ob = MarketAnalyzer.find_nearest_order_block(d1m, price, direction)
        
        if direction == 1:
            if ob and ob["type"] == "BULLISH":
                limit_p = ob["top"] * 1.0005 # Front-run offset
            elif fvg and fvg["type"] == "BULLISH":
                limit_p = fvg["top"] * 1.0005
            else:
                limit_p = price * 0.9995 # Default 0.05% discount
                if price > ema9_1m: limit_p = (price + ema9_1m) / 2 # Midpoint entry
        else:
            if ob and ob["type"] == "BEARISH":
                limit_p = ob["bottom"] * 0.9995 # Front-run offset
            elif fvg and fvg["type"] == "BEARISH":
                limit_p = fvg["bottom"] * 0.9995
            else:
                limit_p = price * 1.0005 # Default 0.05% premium
                if price < ema9_1m: limit_p = (price + ema9_1m) / 2 # Midpoint entry

        # --- SMART VOLATILITY RISK (ATR Based) ---
        # SL = 2x ATR (minimum 0.7%, maximum 3%)
        sl_pct = max(0.7, min(3.0, (atr * 2 / price * 100)))
        
        # Adaptive RR: If ML is very sure (>0.85), go for bigger target
        rr_mult = 2.5 if ml_prob > 0.85 or ml_prob < 0.15 else 1.5
        tp_pct = sl_pct * rr_mult
        
        # Breakout detection for Aggressive Entry
        prev_5_high = d1m['h'].iloc[-6:-1].max()
        prev_5_low = d1m['l'].iloc[-6:-1].min()
        is_breakout = (direction == 1 and price > prev_5_high) or (direction == -1 and price < prev_5_low)
        
        signal = "WAIT"
        dist = abs(price - limit_p) / price * 100
        
        if score >= 75:
            # ULTRA-SNIPER LOGIC:
            # 1. Mode Agresif (High Conviction + Breakout): Market Order (SCALP-LONG/SHORT)
            # 2. Mode Sabar (Score 75-84): Limit Order (Wait for dist <= 0.3)
            if (score >= 85 and is_breakout) or dist <= 0.1:
                signal = "SCALP-LONG" if direction == 1 else "SCALP-SHORT"
                limit_p = price # Market entry
            elif dist <= 0.3:
                signal = "SCALP-LONG" if direction == 1 else "SCALP-SHORT"
                # Keep the limit_p as calculated for Limit Order
        
        # If signal is still WAIT but score is high, it means we are waiting for pullback (Limit Order)
        # But hybrid_trader.py needs a signal to open an order.
        # Actually, the trader opens a LIMIT order if the signal is SCALP-*.
        # So we should return the signal if score >= 75 and dist <= 0.3.

        return {
            "sym": symbol.replace("USDT", ""), "price": f"{price:,.4f}", "limit": limit_p,
            "sig": signal, "score": score, "struct": struct[:4], "dir": direction, "dir_15m": direction,
            "regime": regime, "ai": {
                "tp": tp_pct, "sl": sl_pct, "limit_price": limit_p,
                "ts_act": sl_pct * 0.8, "ts_cb": sl_pct * 0.3, # Dynamic Trailing
                "type": "SCALP", "regime": regime, "score": score, "ml_prob": round(ml_prob, 2),
                "is_market": (score >= 85 and is_breakout)
            }
        }
    except Exception as e:
        if symbol in busy_symbols: busy_symbols.discard(symbol)
        log_error(f"Analysis Crash ({symbol}): {str(e)}")
        return None
