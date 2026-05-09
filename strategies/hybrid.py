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
from engine.multi_exchange import aggregate_flow
from engine.depth_predictor import depth_predictor
from engine.sentiment import sentiment_filter

api_sem = asyncio.Semaphore(15)  # Higher concurrency for parallel analysis
busy_symbols = set()

# Analysis result cache - avoid re-analyzing if data hasn't changed
_analysis_cache = {}  # {symbol: (last_kline_ot, result)}
_indicator_cache = {}  # {(symbol, interval): (last_ot, indicators_dict)}

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
        return _analysis_cache.get(symbol, (0, None))[1]
    try:
        now = time.time()
        if symbol not in market_data.klines: market_data.klines[symbol] = {}
        
        # Check cache - if kline data hasn't changed, return cached result
        k = market_data.klines.get(symbol, {})
        if "1m" in k and not k["1m"].empty:
            last_ot = float(k["1m"].iloc[-1]['ot'])
            cached = _analysis_cache.get(symbol)
            if cached and cached[0] == last_ot and (now - cached[2]) < 3.0:
                return cached[1]
        
        last_p = market_data.last_prime.get(symbol, 0)
        # Force sync if data is missing
        if "1m" not in market_data.klines.get(symbol, {}) or (now - last_p) > 60: 
            busy_symbols.add(symbol)
            try:
                async with api_sem:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    if API_KEY: headers['X-MBX-APIKEY'] = API_KEY
                    
                    res1, res15, res1h = await asyncio.gather(
                        client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "1m", "limit": 200}, headers=headers, timeout=15),
                        client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "15m", "limit": 100}, headers=headers, timeout=15),
                        client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "1h", "limit": 50}, headers=headers, timeout=15),
                    )
                    
                    
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
            
            # ADX filter: only trust EMA cross when trend is strong
            adx_val = MarketAnalyzer.get_adx(d15m, 14)
            if adx_val > 20:
                direction = 1 if ema9_15m > ema21_15m else -1
            else:
                # Weak trend: use RSI mean-reversion logic
                rsi_15m = MarketAnalyzer.get_rsi(d15m["c"], 14).iloc[-1]
                if rsi_15m < 35:
                    direction = 1  # Oversold = buy
                elif rsi_15m > 65:
                    direction = -1  # Overbought = sell
                else:
                    direction = 1 if ema9_15m > ema21_15m else -1
            
            
        except Exception as e:
            return None
        
        # --- HIGHER TIMEFRAME (HTF) BIAS ---
        # 1h EMA confirmation: don't trade against the hourly trend
        ema20_1h = MarketAnalyzer.get_ema(d1h["c"], 20).iloc[-1] if len(d1h) >= 20 else price
        ema50_1h = MarketAnalyzer.get_ema(d1h["c"], 50).iloc[-1] if len(d1h) >= 50 else ema20_1h
        htf_direction = 1 if d1h["c"].iloc[-1] > ema50_1h else -1
        # Multi-timeframe alignment: 15m direction must agree with 1h
        mtf_aligned = (direction == htf_direction)
        
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
            
        score, active_features = MarketAnalyzer.calculate_score(
            d1m, d15m, direction, imbalance, funding, regime, s_weights,
            session, lead_lag, return_features=True,
        )
        struct, _, recent_high, recent_low = MarketAnalyzer.detect_structure(d1m)
        
        # Penalize if 15m and 1h disagree (counter-trend = high risk)
        if not mtf_aligned:
            score = int(score * 0.6)  # 40% penalty for fighting HTF
        
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

        # --- REAL CVD FROM aggTrade (tick-level aggressive flow) ---
        # Falls back to 0 if buffer empty (symbol not in top-N subscription).
        # CVD is *normalized* by recent dollar volume so the threshold is
        # comparable across coins.
        cvd_60s, cvd_n = market_data.get_live_cvd(symbol, window_sec=60)
        cvd_300s, _ = market_data.get_live_cvd(symbol, window_sec=300)
        if cvd_n >= 10:  # Need at least 10 trades in the window
            # Normalize by the 1m candle's typical dollar volume
            try:
                last_candle = d1m.iloc[-1]
                norm_dv = float(last_candle['v'] * last_candle['c']) + 1.0
                cvd_ratio_60 = cvd_60s / norm_dv     # ~[-1, +1]
                cvd_ratio_300 = cvd_300s / (norm_dv * 5.0)
            except Exception:
                cvd_ratio_60 = cvd_ratio_300 = 0.0

            # Strong aggressive flow in our direction → bonus
            if direction == 1:
                if cvd_ratio_60 > 0.25 and cvd_ratio_300 > 0.15:
                    score = min(score + 10, 100)
                    active_features.append(f"{regime}:cvd_live")
                elif cvd_ratio_60 < -0.25:  # Strong sells against LONG → penalty
                    score = max(score - 15, 0)
            else:  # SHORT
                if cvd_ratio_60 < -0.25 and cvd_ratio_300 < -0.15:
                    score = min(score + 10, 100)
                    active_features.append(f"{regime}:cvd_live")
                elif cvd_ratio_60 > 0.25:
                    score = max(score - 15, 0)
        
        # --- ENTRY LOGIC (Smart Sniper with Fake-Move Detection) ---
        
        # Pre-calculate needed values
        base_atr_pct = (atr / price) * 100
        ob = MarketAnalyzer.find_nearest_order_block(d1m, price, direction)
        fvg = MarketAnalyzer.get_nearest_fvg(d1m)
        
        # 1. Validate move is REAL (not fake/manipulation)
        is_fake_move = False
        
        # Check volume confirmation: real moves have above-average volume
        vol_avg = d1m['v'].tail(20).mean()
        vol_recent = d1m['v'].tail(3).mean()
        vol_confirmed = vol_recent > vol_avg * 1.2
        
        # Check candle body quality: fake moves have long wicks, small bodies
        last_candle = d1m.iloc[-1]
        candle_range = last_candle['h'] - last_candle['l']
        candle_body = abs(last_candle['c'] - last_candle['o'])
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        
        # Fake move indicators:
        # - Low volume + big wick = stop hunt / manipulation
        # - Price spike but close near open = rejection
        if not vol_confirmed and body_ratio < 0.3:
            is_fake_move = True
        
        # Check if price is overextended from EMA (chasing)
        dist_from_ema21 = abs(price - MarketAnalyzer.get_ema(d1m["c"], 21).iloc[-1]) / price * 100
        is_overextended = dist_from_ema21 > base_atr_pct * 2.5
        
        # --- MULTI-CANDLE FAKE MOVE DETECTION (3-candle sequence) ---
        if not is_fake_move and len(d1m) >= 5:
            is_fake_move = MarketAnalyzer.detect_multi_candle_fake(d1m, direction)
        
        # --- WYCKOFF PHASE CONTEXT ---
        wyckoff = MarketAnalyzer.detect_wyckoff_phase(d15m)
        # Block entries against Wyckoff phase
        if direction == 1 and wyckoff == "DISTRIBUTION":
            score = int(score * 0.5)  # Heavy penalty: buying into distribution
        elif direction == -1 and wyckoff == "ACCUMULATION":
            score = int(score * 0.5)  # Heavy penalty: shorting into accumulation
        elif direction == 1 and wyckoff == "MARKUP":
            score = min(score + 10, 100)  # Bonus: buying in markup phase
            active_features.append(f"{regime}:wyckoff")
        elif direction == -1 and wyckoff == "MARKDOWN":
            score = min(score + 10, 100)  # Bonus: shorting in markdown phase
            active_features.append(f"{regime}:wyckoff")
        
        # --- ORDERFLOW MICROSTRUCTURE ---
        bid_vel, ask_vel, spoof_score = market_data.get_depth_velocity(symbol)
        # If spoofing detected, treat as fake move
        if spoof_score > 0.6:
            is_fake_move = True
        # Depth velocity confirmation
        if direction == 1 and bid_vel > 0 and ask_vel < 0:
            score = min(score + 5, 100)  # Bids growing, asks shrinking = bullish
            active_features.append(f"{regime}:depth_vel")
        elif direction == -1 and ask_vel > 0 and bid_vel < 0:
            score = min(score + 5, 100)  # Asks growing, bids shrinking = bearish
            active_features.append(f"{regime}:depth_vel")
        
        # --- HMM REGIME CONTEXT ---
        hmm_regime = MarketAnalyzer.detect_hmm_regime(d15m)
        if hmm_regime == "MOMENTUM" and regime == "TRENDING":
            score = min(score + 5, 100)  # Double confirmation: rule-based + HMM agree
            active_features.append(f"{regime}:hmm")
        elif hmm_regime == "MEAN_REVERT" and regime == "RANGING":
            score = min(score + 5, 100)  # Both say mean-revert
            active_features.append(f"{regime}:hmm")
        
        # --- ICEBERG ORDER DETECTION ---
        iceberg_bid, iceberg_ask = market_data.detect_iceberg(symbol)
        if direction == 1 and iceberg_bid:
            score = min(score + 10, 100)  # Hidden buyer supporting price
            active_features.append(f"{regime}:iceberg")
        elif direction == -1 and iceberg_ask:
            score = min(score + 10, 100)  # Hidden seller capping price
            active_features.append(f"{regime}:iceberg")
        elif direction == 1 and iceberg_ask:
            score = max(score - 10, 0)  # Hidden seller blocks our long
        elif direction == -1 and iceberg_bid:
            score = max(score - 10, 0)  # Hidden buyer blocks our short
        
        # --- MULTI-EXCHANGE DIVERGENCE (Aggregate: Binance + Bybit + OKX) ---
        avg_div, cvd_bias = aggregate_flow.get_cross_exchange_signal(symbol)
        if direction == 1 and avg_div > 0.03:
            score = min(score + 8, 100)
            active_features.append(f"{regime}:xchange")
        elif direction == -1 and avg_div < -0.03:
            score = min(score + 8, 100)
            active_features.append(f"{regime}:xchange")
        # Cross-exchange CVD alignment
        if direction == 1 and cvd_bias > 0.3:
            score = min(score + 5, 100)
            active_features.append(f"{regime}:xcvd")
        elif direction == -1 and cvd_bias < -0.3:
            score = min(score + 5, 100)
            active_features.append(f"{regime}:xcvd")
        
        # --- PREDICTIVE DEPTH (Real vs Fake Wall) ---
        bid_real, ask_real = depth_predictor.predict(symbol)
        if direction == 1 and bid_real > 0.7:
            score = min(score + 5, 100)  # Real bid wall = support confirmed
            active_features.append(f"{regime}:depth_pred")
        elif direction == -1 and ask_real > 0.7:
            score = min(score + 5, 100)  # Real ask wall = resistance confirmed
            active_features.append(f"{regime}:depth_pred")
        elif direction == 1 and ask_real > 0.8:
            score = max(score - 8, 0)  # Real ask wall blocks our long
        elif direction == -1 and bid_real > 0.8:
            score = max(score - 8, 0)  # Real bid wall blocks our short
        
        # --- SENTIMENT FILTER ---
        sentiment = sentiment_filter.get_sentiment(symbol)
        if sentiment < -0.5 and direction == 1:
            score = max(score - 20, 0)  # Bearish event, don't go long
        elif sentiment > 0.5 and direction == -1:
            score = max(score - 20, 0)  # Bullish event, don't go short
        liq_bias = sentiment_filter.get_liq_bias(symbol)
        if liq_bias == direction:
            score = min(score + 5, 100)  # Liquidation cascade in our direction
            active_features.append(f"{regime}:liq_cascade")
        
        # 2. Smart limit price placement
        # Priority: Order Block > FVG > VWAP-area > EMA21 pullback
        vwap_price = None
        if len(d1m) >= 60:
            tp = (d1m['h'] + d1m['l'] + d1m['c']) / 3
            vwap_price = (tp * d1m['v']).tail(60).sum() / d1m['v'].tail(60).sum()
        
        if direction == 1:
            # LONG: place limit at support levels (below current price)
            if ob and ob["top"] < price and ob["top"] > price * 0.97:
                limit_p = ob["top"]  # Order block top = strong support
            elif fvg and fvg["type"] == "BULLISH" and fvg["top"] < price:
                limit_p = fvg["top"]  # FVG fill level
            elif vwap_price and vwap_price < price and vwap_price > price * 0.98:
                limit_p = vwap_price  # VWAP as dynamic support
            else:
                # EMA21 pullback (deeper than EMA9 = better entry)
                ema21_1m = MarketAnalyzer.get_ema(d1m["c"], 21).iloc[-1]
                limit_p = ema21_1m if ema21_1m < price else ema9_1m
        else:
            # SHORT: place limit at resistance levels (above current price)
            if ob and ob["bottom"] > price and ob["bottom"] < price * 1.03:
                limit_p = ob["bottom"]  # Order block bottom = strong resistance
            elif fvg and fvg["type"] == "BEARISH" and fvg["bottom"] > price:
                limit_p = fvg["bottom"]  # FVG fill level
            elif vwap_price and vwap_price > price and vwap_price < price * 1.02:
                limit_p = vwap_price
            else:
                ema21_1m = MarketAnalyzer.get_ema(d1m["c"], 21).iloc[-1]
                limit_p = ema21_1m if ema21_1m > price else ema9_1m

        # --- SMART MARKET STRUCTURE SL & TP (H6) ---
        atr_15m = MarketAnalyzer.get_atr(d15m, 14).iloc[-1] if len(d15m) >= 14 else atr
        atr_15m_pct = (atr_15m / price) * 100
        buffer_pct = max(base_atr_pct * 0.3, 0.12)
        
        # Multi-timeframe swing levels
        recent_low_L1 = d1m['l'].tail(15).min()
        recent_low_L2 = d1m['l'].tail(30).min()
        recent_high_L1 = d1m['h'].tail(15).max()
        recent_high_L2 = d1m['h'].tail(30).max()
        # 15m structure for TP targets
        htf_high = d15m['h'].tail(20).max() if len(d15m) >= 20 else recent_high_L2
        htf_low = d15m['l'].tail(20).min() if len(d15m) >= 20 else recent_low_L2
        
        if direction == 1:
            struct_low = recent_low_L2 if ((price - recent_low_L2) / price * 100) <= 3.0 else recent_low_L1
            sl_price = ob["bottom"] if ob else (struct_low if struct_low else price * (1 - 0.01))
            sl_pct = ((price - sl_price) / price * 100) + buffer_pct
        else:
            struct_high = recent_high_L2 if ((recent_high_L2 - price) / price * 100) <= 3.0 else recent_high_L1
            sl_price = ob["top"] if ob else (struct_high if struct_high else price * (1 + 0.01))
            sl_pct = ((sl_price - price) / price * 100) + buffer_pct
        
        # ML-adjusted SL: tighter when high confidence, wider when uncertain
        if ml_prob > 0.7 or ml_prob < 0.3:
            sl_pct *= 0.85  # High confidence = tighter SL (less risk per trade)
        elif 0.45 < ml_prob < 0.55:
            sl_pct *= 1.15  # Uncertain = wider SL (give room)
        
        sl_pct = max(0.4, min(sl_pct, 3.5))
        
        # --- SMART TP: Structure-based targets + Liquidation Magnets ---
        # Base RR from regime
        rr_mult = 2.0
        if regime == "RANGING":
            rr_mult = 1.5
        elif regime == "TRENDING":
            rr_mult = 2.5 if ((direction == 1 and ml_prob > 0.65) or (direction == -1 and ml_prob < 0.35)) else 2.0
        elif regime == "VOLATILE":
            rr_mult = 1.8  # Quick exits in volatile
        
        # Try to use structural TP target (next resistance/support)
        struct_tp_pct = sl_pct * rr_mult  # Default
        if direction == 1:
            # TP target = next HTF resistance or liquidation cluster
            struct_dist = ((htf_high - price) / price) * 100
            if struct_dist > sl_pct * 1.2:  # Only if it gives at least 1.2R
                struct_tp_pct = min(struct_dist, sl_pct * 3.5)
            # Check liquidation magnet as extended target
            if liq and any(p > price for p in liq.get("short_liq", [])):
                liq_target = min(p for p in liq["short_liq"] if p > price)
                liq_dist = ((liq_target - price) / price) * 100
                if liq_dist > struct_tp_pct and liq_dist < sl_pct * 4:
                    struct_tp_pct = liq_dist  # Extend TP to liquidation magnet
        else:
            struct_dist = ((price - htf_low) / price) * 100
            if struct_dist > sl_pct * 1.2:
                struct_tp_pct = min(struct_dist, sl_pct * 3.5)
            if liq and any(p < price for p in liq.get("long_liq", [])):
                liq_target = max(p for p in liq["long_liq"] if p < price)
                liq_dist = ((price - liq_target) / price) * 100
                if liq_dist > struct_tp_pct and liq_dist < sl_pct * 4:
                    struct_tp_pct = liq_dist
        
        tp_pct = max(sl_pct * 1.3, struct_tp_pct)  # Minimum 1.3R
        
        # --- TRAILING STOP PARAMETERS (ML-adaptive) ---
        # Activation: earlier in trending, later in ranging
        ts_act = sl_pct * 0.7 if regime == "TRENDING" else sl_pct * 0.9
        # Callback: tighter when ML confident
        ts_cb = base_atr_pct * 0.5 if (ml_prob > 0.65 or ml_prob < 0.35) else base_atr_pct * 0.7
        ts_cb = max(0.15, min(ts_cb, sl_pct * 0.4))
        
        # Breakout detection (validated with volume + body)
        prev_15_high = d1m['h'].iloc[-16:-1].max()
        prev_15_low = d1m['l'].iloc[-16:-1].min()
        raw_breakout = (direction == 1 and price > prev_15_high) or (direction == -1 and price < prev_15_low)
        # Real breakout = structure break + volume + strong body
        is_breakout = raw_breakout and vol_confirmed and body_ratio > 0.5
        
        # Logika Baru: Adaptive Threshold berdasarkan Regime & Breakout
        threshold = 75
        if regime == "VOLATILE":
            threshold = 70 if is_breakout else 85
        elif regime == "RANGING":
            threshold = 82
        elif regime == "TRENDING":
            threshold = 68 if is_breakout else 75
            
        # --- ADAPTIVE EXECUTION ENGINE ---
        is_market_order = False
        
        # Market orders ONLY for confirmed breakouts with extreme conviction
        if regime == "TRENDING" and is_breakout and score >= 85 and not is_fake_move:
            is_market_order = True
            limit_p = price
        # Everything else = limit order (sniper mode)

        signal = "WAIT"
        dist = abs(price - limit_p) / price * 100
        
        # Block entry if fake move or overextended
        if is_fake_move or is_overextended:
            signal = "WAIT"
        elif score >= threshold:
            # Limit order must be at a meaningful distance (not just chasing)
            if is_market_order:
                signal = "SCALP-LONG" if direction == 1 else "SCALP-SHORT"
            elif dist >= 0.15 and dist <= 2.5:
                # Good sniper distance: not too close (chasing), not too far (won't fill)
                signal = "SCALP-LONG" if direction == 1 else "SCALP-SHORT"
            elif dist < 0.3:
                # Too close - force limit deeper into structure
                if direction == 1:
                    limit_p = price * (1 - 0.003)  # Force 0.3% below
                else:
                    limit_p = price * (1 + 0.003)  # Force 0.3% above
                signal = "SCALP-LONG" if direction == 1 else "SCALP-SHORT"
        
        result = {
            "sym": symbol.replace("USDT", ""), "price": f"{price:,.4f}", "limit": float(limit_p),
            "sig": signal, "score": int(score), "struct": struct[:4], "dir": int(direction), "dir_15m": int(direction),
            "regime": regime, "ai": {
                "tp": float(tp_pct), "sl": float(sl_pct), "limit_price": float(limit_p),
                "ts_act": float(ts_act), "ts_cb": float(ts_cb),
                "type": "SCALP", "regime": regime, "score": int(score), "ml_prob": float(round(ml_prob, 2)),
                "is_market": is_market_order, "atr_pct": float(base_atr_pct),
                "active_features": list(active_features),
            }
        }
        # Cache result keyed by last kline open time
        cache_ot = float(d1m.iloc[-1]['ot']) if not d1m.empty else 0
        _analysis_cache[symbol] = (cache_ot, result, time.time())
        return result
    except Exception as e:
        if symbol in busy_symbols: busy_symbols.discard(symbol)
        log_error(f"Analysis Crash ({symbol}): {str(e)}")
        return None
