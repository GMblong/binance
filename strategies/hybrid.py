import time
import pandas as pd
import asyncio
from datetime import datetime
from utils.config import API_URL, API_KEY
from utils.state import bot_state, market_data
from strategies.analyzer import MarketAnalyzer
from engine.ml_engine import ml_predictor
from engine.api import get_orderbook_imbalance
from utils.intelligence import get_current_session, detect_lead_lag, calculate_market_volatility
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
    if symbol in busy_symbols: 
        with open("sync_debug.log", "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {symbol}: Skipping - in busy_symbols\n")
        return None
    try:
        now = time.time()
        with open("sync_debug.log", "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {symbol}: Starting analysis cycle\n")
        if symbol not in market_data.klines: market_data.klines[symbol] = {}
        
        last_p = market_data.last_prime.get(symbol, 0)
        # Force sync if data is missing
        if "1m" not in market_data.klines.get(symbol, {}) or (now - last_p) > 60: 
            with open("sync_debug.log", "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {symbol}: Attempting sync. last_p: {last_p}, now: {now}\n")
            busy_symbols.add(symbol)
            try:
                async with api_sem:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    if API_KEY: headers['X-MBX-APIKEY'] = API_KEY
                    
                    res1 = await client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "1m", "limit": 200}, headers=headers, timeout=15)
                    res15 = await client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "15m", "limit": 100}, headers=headers, timeout=15)
                    res1h = await client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "1h", "limit": 50}, headers=headers, timeout=15)
                    
                    with open("sync_debug.log", "a") as f:
                        f.write(f"[{time.strftime('%H:%M:%S')}] {symbol}: Statuses: {res1.status_code}, {res15.status_code}, {res1h.status_code}\n")
                    
                    if res1.status_code == 200 and res15.status_code == 200 and res1h.status_code == 200:
                        data1, data15, data1h = res1.json(), res15.json(), res1h.json()
                        with open("sync_debug.log", "a") as f:
                            f.write(f"[{time.strftime('%H:%M:%S')}] {symbol}: Received JSON lengths: {len(data1)}, {len(data15)}, {len(data1h)}\n")

                        def proc(data):
                            df = pd.DataFrame(data).iloc[:, [0, 1, 2, 3, 4, 5, 9]]
                            df.columns = ["ot", "o", "h", "l", "c", "v", "tbv"]
                            for col in ["o", "h", "l", "c", "v", "tbv"]: df[col] = df[col].astype(float)
                            return df
                        market_data.klines[symbol]["1m"] = proc(data1)
                        market_data.klines[symbol]["15m"] = proc(data15)
                        market_data.klines[symbol]["1h"] = proc(data1h)
                        market_data.last_prime[symbol] = now

                    else:
                        # SET LAST_PRIME TO AVOID SPAMMING ON FAILURE (Backoff 30 seconds)
                        market_data.last_prime[symbol] = now - 30 
                        with open("sync_debug.log", "a") as f:
                            f.write(f"[{time.strftime('%H:%M:%S')}] {symbol}: API Fail - Statuses: {res1.status_code}, {res15.status_code}, {res1h.status_code}\n")
            finally:
                busy_symbols.discard(symbol)

        k = market_data.klines.get(symbol, {})
        # Re-check cache *after* sync attempt
        if "1m" not in k or "15m" not in k or "1h" not in k: 
            with open("sync_debug.log", "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {symbol}: Still missing data. Found keys: {list(k.keys())}\n")
            return None

        d1m, d15m, d1h = k["1m"], k["15m"], k["1h"]
        
        with open("sync_debug.log", "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {symbol}: Proceeding to indicator calculation\n")

        try:
            price = d1m["c"].iloc[-1]
            
            # --- SMART INDICATORS ---
            ema9_15m = MarketAnalyzer.get_ema(d15m["c"], 9).iloc[-1]
            ema21_15m = MarketAnalyzer.get_ema(d15m["c"], 21).iloc[-1]
            ema9_1m = MarketAnalyzer.get_ema(d1m["c"], 9).iloc[-1]
            atr = MarketAnalyzer.get_atr(d1m, 14).iloc[-1]
            
            direction = 1 if ema9_15m > ema21_15m else -1
            
            with open("sync_debug.log", "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {symbol}: Indicators: Ema9_15: {ema9_15m}, Ema21_15: {ema21_15m}, Dir: {direction}\n")
            
            # ... [rest of the logic]
        except Exception as e:
            with open("sync_debug.log", "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {symbol}: Calculation Error: {str(e)}\n")
            return None
        
        # --- HIGHER TIMEFRAME (HTF) BIAS ---
        htf_direction = direction
        mtf_aligned = True
        
        # --- INSTITUTIONAL DATA GATHERING ---
        imbalance = await get_orderbook_imbalance(client, symbol)
        funding = market_data.funding.get(symbol, 0.0)
        regime = MarketAnalyzer.detect_regime(d15m)
        session = get_current_session()
        lead_lag = detect_lead_lag(symbol)
        
        # --- PER-SYMBOL DYNAMIC WEIGHTS ---
        # Try to get specific weights for this symbol, fallback to global weights
        s_weights = bot_state.get("sym_weights", {}).get(symbol)
        if not s_weights:
            s_weights = bot_state.get("neural_weights")
            
        score = MarketAnalyzer.calculate_score(d1m, d15m, direction, imbalance, funding, regime, s_weights, session, lead_lag)
        struct, _, recent_high, recent_low = MarketAnalyzer.detect_structure(d1m)
        
        # --- MACHINE LEARNING INTEGRATION ---
        # Predict UP probability (0.0 to 1.0)
        ml_prob = await ml_predictor.predict(client, symbol, d1m)
        ml_score_boost = 0
        ml_w = s_weights.get(f"{regime}:ml", 1.0) if s_weights else 1.0

        # Logika Baru (Scaled Confidence): 
        # Hanya boost jika sangat yakin (>65% atau <35%), dan penalti berskala jika tidak setuju
        if direction == 1:
            if ml_prob >= 0.65:
                ml_score_boost = int((ml_prob - 0.6) * 50 * ml_w)
            elif ml_prob < 0.40:
                ml_score_boost = int((ml_prob - 0.4) * 50 * ml_w) # Hasil negatif
        elif direction == -1:
            if ml_prob <= 0.35:
                ml_score_boost = int((0.4 - ml_prob) * 50 * ml_w)
            elif ml_prob > 0.60:
                ml_score_boost = int((0.6 - ml_prob) * 50 * ml_w) # Hasil negatif
        
        score = min(max(score + ml_score_boost, 0), 100)
        
        # --- ENTRY LOGIC (Full Sniper - Limit Only) ---
        # Default target: EMA9 (Pullback Level)
        limit_p = ema9_1m
        
        # Try to find FVG for a more precise institutional entry
        fvg = MarketAnalyzer.get_nearest_fvg(d1m)
        if direction == 1:
            if fvg and fvg["type"] == "BULLISH":
                # Front-run the FVG top slightly for better fill
                limit_p = fvg["top"]
        else:
            if fvg and fvg["type"] == "BEARISH":
                limit_p = fvg["bottom"]

        # --- SMART MARKET STRUCTURE SL & TP ---
        base_atr_pct = (atr / price) * 100
        buffer_pct = base_atr_pct * 0.2 # 0.2x ATR buffer
        
        ob = MarketAnalyzer.find_nearest_order_block(d1m, price, direction)
        
        if direction == 1:
            # LONG SL: Below Order Block or Recent Low
            sl_price = ob["bottom"] if ob else (recent_low if recent_low else price * (1 - 0.01))
            sl_pct = ((price - sl_price) / price * 100) + buffer_pct
        else:
            # SHORT SL: Above Order Block or Recent High
            sl_price = ob["top"] if ob else (recent_high if recent_high else price * (1 + 0.01))
            sl_pct = ((sl_price - price) / price * 100) + buffer_pct
            
        # Fallback & Safety bounds
        sl_pct = max(0.5, min(sl_pct, 4.0)) # Hard cap min 0.5%, max 4.0%
        
        # Adaptive RR based on Regime and ML Confidence
        rr_mult = 2.0
        if regime == "RANGING":
            rr_mult = 1.5  # Quick scalps in ranging market
        elif regime == "TRENDING":
            if (direction == 1 and ml_prob > 0.65) or (direction == -1 and ml_prob < 0.35):
                rr_mult = 2.5 # Higher conviction = higher reward target
        
        tp_pct = sl_pct * rr_mult
        
        # Breakout detection (structural, not just 5 candles which is too noisy)
        prev_15_high = d1m['h'].iloc[-16:-1].max()
        prev_15_low = d1m['l'].iloc[-16:-1].min()
        is_breakout = (direction == 1 and price > prev_15_high) or (direction == -1 and price < prev_15_low)
        
        # Logika Baru: Adaptive Threshold berdasarkan Regime & Breakout
        threshold = 75
        if regime == "VOLATILE":
            threshold = 70 if is_breakout else 85 
        elif regime == "RANGING":
            threshold = 85  
        elif regime == "TRENDING":
            threshold = 75 if is_breakout else 80
            
        # --- ADAPTIVE EXECUTION ENGINE ---
        # 1. Decide Entry Mode based on Regime and Score
        is_market_order = False
        
        if regime == "TRENDING":
            # In Trending markets, ONLY FOMO/Market Execute if it's a breakout AND conviction is high (>=88)
            # or if the overall score is absolutely extreme (>=95)
            if (is_breakout and score >= 88) or score >= 95:
                is_market_order = True
                limit_p = price # Market Execution
        elif regime == "VOLATILE":
            # In Volatile markets, we ONLY use market orders if it's a confirmed breakout with extreme score
            if is_breakout and score >= 95:
                is_market_order = True
                limit_p = price
        # In RANGING mode, we stay 100% Sniper (Limit Only)

        signal = "WAIT"
        dist = abs(price - limit_p) / price * 100
        
        if score >= threshold:
            # Sniper logic: Only signal if we are reasonably close to our target or if it's a market order
            if is_market_order or dist <= 0.6:
                signal = "SCALP-LONG" if direction == 1 else "SCALP-SHORT"
        
        return {
            "sym": symbol.replace("USDT", ""), "price": f"{price:,.4f}", "limit": float(limit_p),
            "sig": signal, "score": int(score), "struct": struct[:4], "dir": int(direction), "dir_15m": int(direction),
            "regime": regime, "ai": {
                "tp": float(tp_pct), "sl": float(sl_pct), "limit_price": float(limit_p),
                "ts_act": float(sl_pct * 0.8), "ts_cb": float(sl_pct * 0.3), 
                "type": "SCALP", "regime": regime, "score": int(score), "ml_prob": float(round(ml_prob, 2)),
                "is_market": is_market_order
            }
        }
    except Exception as e:
        if symbol in busy_symbols: busy_symbols.discard(symbol)
        log_error(f"Analysis Crash ({symbol}): {str(e)}")
        return None
