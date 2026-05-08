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
        return None
    try:
        now = time.time()
        if symbol not in market_data.klines: market_data.klines[symbol] = {}
        
        last_p = market_data.last_prime.get(symbol, 0)
        # Force sync if data is missing
        if "1m" not in market_data.klines.get(symbol, {}) or (now - last_p) > 60: 
            busy_symbols.add(symbol)
            try:
                async with api_sem:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    if API_KEY: headers['X-MBX-APIKEY'] = API_KEY
                    
                    res1 = await client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "1m", "limit": 200}, headers=headers, timeout=15)
                    res15 = await client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "15m", "limit": 100}, headers=headers, timeout=15)
                    res1h = await client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "1h", "limit": 50}, headers=headers, timeout=15)
                    
                    
                    if res1.status_code == 200 and res15.status_code == 200 and res1h.status_code == 200:
                        data1, data15, data1h = res1.json(), res15.json(), res1h.json()

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
            finally:
                busy_symbols.discard(symbol)

        k = market_data.klines.get(symbol, {})
        # Re-check cache *after* sync attempt
        if "1m" not in k or "15m" not in k or "1h" not in k: 
            return None

        d1m, d15m, d1h = k["1m"], k["15m"], k["1h"]
        

        try:
            price = d1m["c"].iloc[-1]
            
            # --- SMART INDICATORS ---
            ema9_15m = MarketAnalyzer.get_ema(d15m["c"], 9).iloc[-1]
            ema21_15m = MarketAnalyzer.get_ema(d15m["c"], 21).iloc[-1]
            ema9_1m = MarketAnalyzer.get_ema(d1m["c"], 9).iloc[-1]
            atr = MarketAnalyzer.get_atr(d1m, 14).iloc[-1]
            
            direction = 1 if ema9_15m > ema21_15m else -1
            
            
        except Exception as e:
            return None
        
        # --- HIGHER TIMEFRAME (HTF) BIAS ---
        # Compute 1h trend using both EMA20 and EMA50 so we can enforce a HARD
        # gate (1h direction must agree with 15m direction in trending/volatile
        # regimes). Ambiguous 1h (price between EMA20 and EMA50) returns 0.
        try:
            ema20_1h = MarketAnalyzer.get_ema(d1h["c"], 20).iloc[-1]
            ema50_1h = (
                MarketAnalyzer.get_ema(d1h["c"], 50).iloc[-1]
                if len(d1h) >= 50
                else ema20_1h
            )
            last_1h_close = d1h["c"].iloc[-1]
            if last_1h_close > ema20_1h and ema20_1h >= ema50_1h:
                htf_direction = 1
            elif last_1h_close < ema20_1h and ema20_1h <= ema50_1h:
                htf_direction = -1
            else:
                htf_direction = 0
        except Exception:
            htf_direction = direction

        # 1m direction for hard-gate confirmation.
        try:
            ema9_1m_val = MarketAnalyzer.get_ema(d1m["c"], 9).iloc[-1]
            ema21_1m_val = MarketAnalyzer.get_ema(d1m["c"], 21).iloc[-1]
            dir_1m_val = 1 if ema9_1m_val > ema21_1m_val else -1
        except Exception:
            dir_1m_val = direction
        mtf_aligned = True

        # --- INSTITUTIONAL DATA GATHERING ---
        imbalance = await get_orderbook_imbalance(client, symbol)
        funding = market_data.funding.get(symbol, 0.0)
        regime = MarketAnalyzer.detect_regime(d15m)
        session = get_current_session()
        lead_lag = detect_lead_lag(symbol)

        # --- HARD MTF GATE ---
        # In TRENDING / VOLATILE we require 1h == 15m == 1m direction. In
        # RANGING we still require 1m to confirm the side (otherwise we are
        # just a coin-flip). This single filter removes the bulk of the
        # losing trades that used to sneak through on an ML score boost.
        if regime in ("TRENDING", "VOLATILE"):
            if htf_direction == 0 or htf_direction != direction or dir_1m_val != direction:
                return None
        else:
            if dir_1m_val != direction:
                return None
        
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
        
        # --- LIQUIDATION MAGNET (M2) ---
        liq = MarketAnalyzer.predict_liquidation_clusters(d15m)
        if liq:
            if "liq_map" not in bot_state: bot_state["liq_map"] = {}
            bot_state["liq_map"][symbol] = liq
            
            # If LONG, price is drawn up to short_liq (liquidating shorters)
            if direction == 1 and any(p > price for p in liq["short_liq"]):
                score = min(score + 5, 100)
            # If SHORT, price is drawn down to long_liq (liquidating longers)
            elif direction == -1 and any(p < price for p in liq["long_liq"]):
                score = min(score + 5, 100)
        
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

        # --- SMART MARKET STRUCTURE SL & TP (H6) ---
        base_atr_pct = (atr / price) * 100
        buffer_pct = max(base_atr_pct * 0.3, 0.15) # 0.3x ATR or 0.15% minimum buffer
        
        ob = MarketAnalyzer.find_nearest_order_block(d1m, price, direction)
        
        # Multi-level Swing Search for better safety
        recent_low_L1 = d1m['l'].tail(15).min()
        recent_low_L2 = d1m['l'].tail(30).min()
        recent_high_L1 = d1m['h'].tail(15).max()
        recent_high_L2 = d1m['h'].tail(30).max()
        
        if direction == 1:
            # LONG SL: Try to use L2 if it's within reasonable risk (< 3.0%), otherwise L1
            struct_low = recent_low_L2 if ((price - recent_low_L2) / price * 100) <= 3.0 else recent_low_L1
            sl_price = ob["bottom"] if ob else (struct_low if struct_low else price * (1 - 0.01))
            sl_pct = ((price - sl_price) / price * 100) + buffer_pct
        else:
            # SHORT SL: Try to use L2 if it's within reasonable risk (< 3.0%), otherwise L1
            struct_high = recent_high_L2 if ((recent_high_L2 - price) / price * 100) <= 3.0 else recent_high_L1
            sl_price = ob["top"] if ob else (struct_high if struct_high else price * (1 + 0.01))
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

        # --- THRESHOLD + EV GATE ---
        # Score threshold is still used as a coarse filter, but the final
        # go/no-go is the EV gate: expected value per trade AFTER costs must
        # exceed a minimum positive number.
        threshold = 75
        if regime == "VOLATILE":
            threshold = 78 if is_breakout else 85
        elif regime == "RANGING":
            threshold = 85
        elif regime == "TRENDING":
            threshold = 78 if is_breakout else 82

        # Estimate per-round-trip cost (maker entry + taker exit + slippage
        # scaled with ATR%). Matches backtest/cost_model.py constants so the
        # live gate and the backtest gate agree.
        cost_fee_maker = 0.02   # % taker-side fee
        cost_fee_taker = 0.04   # %
        slip_floor = 0.03       # %
        slip_atr_coef = 0.30
        roundtrip_cost_pct = (
            cost_fee_maker + cost_fee_taker + slip_floor + slip_atr_coef * base_atr_pct
        )

        # Directional win probability from ML + score edge. Score beyond the
        # regime floor adds a small edge; ML prob is the dominant input.
        p_ml = ml_prob if direction == 1 else (1 - ml_prob)
        score_edge = max(0.0, (score - threshold) / 100.0) * 0.4  # max ~0.1
        p_win = max(p_ml, 0.5 + score_edge)
        ev_pct = p_win * tp_pct - (1 - p_win) * sl_pct - roundtrip_cost_pct
        MIN_EV_PCT = 0.10  # require at least +0.10% expected value per trade

        # --- ADAPTIVE EXECUTION ENGINE ---
        # MARKET entries are reserved for the rare case where all of:
        #   - structural 15-bar breakout has already printed,
        #   - ML directional prob >= 0.72 in our favor,
        #   - EV after taker-both-sides still >= MIN_EV_PCT,
        # are true. Otherwise we always use a limit order at a structurally
        # meaningful level (FVG / OB / EMA9).
        is_market_order = False
        if is_breakout:
            strong_ml = (direction == 1 and ml_prob >= 0.72) or (
                direction == -1 and ml_prob <= 0.28
            )
            if strong_ml and score >= 90:
                market_cost_pct = (
                    2 * cost_fee_taker + 2 * slip_floor
                    + 2 * slip_atr_coef * base_atr_pct
                )
                if (p_win * tp_pct - (1 - p_win) * sl_pct - market_cost_pct) >= MIN_EV_PCT:
                    is_market_order = True
                    limit_p = price

        signal = "WAIT"
        dist = abs(price - limit_p) / price * 100

        if score >= threshold and ev_pct >= MIN_EV_PCT:
            if is_market_order or dist <= 0.6:
                signal = "SCALP-LONG" if direction == 1 else "SCALP-SHORT"
        
        return {
            "sym": symbol.replace("USDT", ""), "price": f"{price:,.4f}", "limit": float(limit_p),
            "sig": signal, "score": int(score), "struct": struct[:4], "dir": int(direction), "dir_15m": int(direction),
            "regime": regime, "ai": {
                "tp": float(tp_pct), "sl": float(sl_pct), "limit_price": float(limit_p),
                "ts_act": float(sl_pct * 0.8),
                # Trailing callback: used to be sl*0.3 which was too tight --
                # even a 0.3% wiggle closed winners too early. Raise to 0.5
                # with a hard floor so small SLs don't produce micro callbacks.
                "ts_cb": float(max(sl_pct * 0.5, 0.25)),
                "type": "SCALP", "regime": regime, "score": int(score),
                "ml_prob": float(round(ml_prob, 2)),
                "p_win": float(round(p_win, 3)),
                "ev_pct": float(round(ev_pct, 3)),
                "is_market": is_market_order
            }
        }
    except Exception as e:
        if symbol in busy_symbols: busy_symbols.discard(symbol)
        log_error(f"Analysis Crash ({symbol}): {str(e)}")
        return None
