import time
import pandas as pd
import asyncio
from datetime import datetime
from utils.config import API_URL, API_KEY
from utils.state import bot_state, market_data
from strategies.analyzer import MarketAnalyzer
from engine.ml_engine import ml_predictor
from utils.logger import log_error

api_sem = asyncio.Semaphore(1) 
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
            async with api_sem:
                headers = {'X-MBX-APIKEY': API_KEY, 'User-Agent': 'Mozilla/5.0'}
                res1 = await client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "1m", "limit": 200}, headers=headers, timeout=10)
                await asyncio.sleep(0.3)
                res15 = await client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "15m", "limit": 100}, headers=headers, timeout=10)
                
                if res1.status_code == 200 and res15.status_code == 200:
                    def proc(data):
                        df = pd.DataFrame(data).iloc[:, [0, 1, 2, 3, 4, 5, 9]]
                        df.columns = ["ot", "o", "h", "l", "c", "v", "tbv"]
                        for col in ["o", "h", "l", "c", "v", "tbv"]: df[col] = df[col].astype(float)
                        return df
                    market_data.klines[symbol]["1m"] = proc(res1.json())
                    market_data.klines[symbol]["15m"] = proc(res15.json())
                    market_data.last_prime[symbol] = now
            busy_symbols.discard(symbol)

        k = market_data.klines.get(symbol, {})
        if "1m" not in k or "15m" not in k: return None
        
        d1m, d15m = k["1m"], k["15m"]
        price = d1m["c"].iloc[-1]
        
        # --- SMART INDICATORS ---
        ema9_15m = MarketAnalyzer.get_ema(d15m["c"], 9).iloc[-1]
        ema21_15m = MarketAnalyzer.get_ema(d15m["c"], 21).iloc[-1]
        ema9_1m = MarketAnalyzer.get_ema(d1m["c"], 9).iloc[-1]
        atr = MarketAnalyzer.get_atr(d1m, 14).iloc[-1]
        
        # Primary trend is determined by 15m (Macro Bias)
        direction = 1 if ema9_15m > ema21_15m else -1
        
        score = MarketAnalyzer.calculate_score(d1m, d15m, direction)
        regime = MarketAnalyzer.detect_regime(d15m)
        struct, _ = MarketAnalyzer.detect_structure(d1m)
        
        # --- MACHINE LEARNING INTEGRATION ---
        # Predict UP probability (0.0 to 1.0)
        ml_prob = await ml_predictor.predict(client, symbol, d15m)
        ml_score_boost = 0
        if direction == 1 and ml_prob > 0.6:
            ml_score_boost = int((ml_prob - 0.5) * 40) # Up to +20 score
        elif direction == -1 and ml_prob < 0.4:
            ml_score_boost = int((0.5 - ml_prob) * 40) # Up to +20 score
            
        # Severe penalty if ML strongly disagrees with TA direction
        if (direction == 1 and ml_prob < 0.4) or (direction == -1 and ml_prob > 0.6):
            ml_score_boost = -30
        
        score = min(score + ml_score_boost, 100)
        
        # --- SMART ENTRY (Pullback Logic) ---
        # Goal: Place limit order between current price and 1m EMA9 to catch small dips
        if direction == 1:
            limit_p = price * 0.9995 # Default 0.05% discount
            if price > ema9_1m: limit_p = (price + ema9_1m) / 2 # Midpoint entry
        else:
            limit_p = price * 1.0005 # Default 0.05% premium
            if price < ema9_1m: limit_p = (price + ema9_1m) / 2 # Midpoint entry

        # --- SMART VOLATILITY RISK (ATR Based) ---
        # SL = 2x ATR (minimum 0.7%, maximum 3%)
        sl_pct = max(0.7, min(3.0, (atr * 2 / price * 100)))
        tp_pct = sl_pct * 1.5 # 1:1.5 Risk/Reward ratio
        
        signal = "WAIT"
        if score >= 75:
            # Check if price is not too far from our limit (max 0.3% gap)
            dist = abs(price - limit_p) / price * 100
            if dist <= 0.3:
                signal = "SCALP-LONG" if direction == 1 else "SCALP-SHORT"

        return {
            "sym": symbol.replace("USDT", ""), "price": f"{price:,.4f}", "limit": limit_p,
            "sig": signal, "score": score, "struct": struct[:4], "dir": direction, "dir_15m": direction,
            "regime": regime, "ai": {
                "tp": tp_pct, "sl": sl_pct, "limit_price": limit_p,
                "ts_act": sl_pct * 0.8, "ts_cb": sl_pct * 0.3, # Dynamic Trailing
                "type": "SCALP", "regime": regime, "score": score, "ml_prob": round(ml_prob, 2)
            }
        }
    except Exception as e:
        if symbol in busy_symbols: busy_symbols.discard(symbol)
        log_error(f"Analysis Crash ({symbol}): {str(e)}")
        return None
