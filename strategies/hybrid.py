import time
import pandas as pd
import pandas_ta as ta
from utils.config import API_URL
from utils.state import bot_state, market_data
from strategies.analyzer import MarketAnalyzer
from utils.intelligence import get_current_session

from engine.api import binance_request, get_orderbook_imbalance, get_btc_dominance

last_btc_update = 0

async def get_btc_trend(client):
    global last_btc_update
    now = time.time()

    # --- AUTONOMOUS SELF-TUNING (Machine Learning Adjustment) ---
    total_trades = bot_state["wins"] + bot_state["losses"]
    if total_trades >= 3:
        win_rate = bot_state["wins"] / total_trades
        if win_rate < 0.40: # If underperforming, increase strictness
            bot_state["ai_confidence"] = 1.2 
        elif win_rate > 0.65: # If performing well, can be slightly more aggressive
            bot_state["ai_confidence"] = 0.9
        else:
            bot_state["ai_confidence"] = 1.0

    if now - last_btc_update < 60 and bot_state["btc_state"] != "INITIALIZING":
        return

    try:
        await get_btc_dominance(client)
        # Fetch 15m, 1h, and 4h for multi-context analysis

        res_15m = await client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": "BTCUSDT", "interval": "15m", "limit": 50}, timeout=10)
        res_1h = await client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": "BTCUSDT", "interval": "1h", "limit": 50}, timeout=10)
        res_4h = await client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": "BTCUSDT", "interval": "4h", "limit": 50}, timeout=10)

        if res_15m.status_code != 200: return

        def process_df(data):
            df = pd.DataFrame(data).iloc[:, [0, 1, 2, 3, 4, 5, 9]]
            df.columns = ["ot", "o", "h", "l", "c", "v", "tbv"]
            df[["o", "h", "l", "c", "v", "tbv"]] = df[["o", "h", "l", "c", "v", "tbv"]].astype(float)
            return df

        d15 = process_df(res_15m.json())
        d1h = process_df(res_1h.json())
        d4h = process_df(res_4h.json())

        last_btc_update = now

        # 1. VOLATILITY GUARD (Crash/Pump Protection)
        last_change = ((d15["c"].iloc[-1] - d15["o"].iloc[-1]) / d15["o"].iloc[-1]) * 100
        if abs(last_change) > 1.2:
            bot_state["btc_state"] = "DANGER"
            bot_state["btc_dir"] = -99
            bot_state["btc_trend"] = -1 if last_change < 0 else 1
            return

        # 2. INDICATORS
        ema10_1h = ta.ema(d1h["c"], 10).iloc[-1]
        ema10_4h = ta.ema(d4h["c"], 10).iloc[-1]
        adx_1h = ta.adx(d1h["h"], d1h["l"], d1h["c"], length=14).iloc[-1, 0]
        
        # Bollinger Bands for Squeeze detection
        bb = ta.bbands(d1h["c"], length=20, std=2)
        bb_width = (bb.iloc[-1, 2] - bb.iloc[-1, 0]) / bb.iloc[-1, 1] * 100

        # 3. REGIME LOGIC
        if bb_width < 1.0:
            bot_state["btc_state"] = "SQUEEZE"
            bot_state["btc_dir"] = 0
            bot_state["btc_trend"] = 0
        elif d1h["c"].iloc[-1] > ema10_1h and d4h["c"].iloc[-1] < ema10_4h:
            bot_state["btc_state"] = "BULL_TRAP"
            bot_state["btc_dir"] = -1 # Only allow shorts in bull trap
            bot_state["btc_trend"] = -1
        elif d1h["c"].iloc[-1] < ema10_1h and d4h["c"].iloc[-1] > ema10_4h:
            bot_state["btc_state"] = "BEAR_TRAP"
            bot_state["btc_dir"] = 1 # Only allow longs in bear trap
            bot_state["btc_trend"] = 1
        elif adx_1h < 20:
            bot_state["btc_state"] = "RANGING"
            bot_state["btc_dir"] = 0
            bot_state["btc_trend"] = 0
        else:
            is_bull = d1h["c"].iloc[-1] > ema10_1h and d4h["c"].iloc[-1] > ema10_4h
            bot_state["btc_state"] = "BULLISH" if is_bull else "BEARISH"
            bot_state["btc_dir"] = 1 if is_bull else -1
            bot_state["btc_trend"] = 1 if is_bull else -1
            
    except Exception as e:
        bot_state["last_log"] = f"[red]BTC Smart Guard Err: {str(e)[:20]}[/]"

async def analyze_hybrid_async(client, symbol):
    try:
        now = time.time()
        intervals = ["1m", "5m", "15m", "1h"]
        needed_rest = []
        
        async with market_data.lock:
            if symbol not in market_data.klines:
                market_data.klines[symbol] = {}
            
            last_p = market_data.last_prime.get(symbol, 0)
            
            # Dynamic Staleness: 15s if WS is lagging, 10m if WS is healthy
            is_ws_lagging = (now - bot_state.get("ws_last_msg", 0)) > 5
            stale_threshold = 15 if is_ws_lagging else 600
            is_stale = (now - last_p) > stale_threshold
            
            for i in intervals:
                if i not in market_data.klines[symbol] or is_stale:
                    needed_rest.append(i)
        
        if needed_rest:
            success_count = 0
            for idx, i in enumerate(needed_rest):
                try:
                    # bot_state["last_log"] = f"[dim]Syncing {symbol} {i}...[/]"
                    res = await client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": i, "limit": 250}, timeout=10)
                    if res.status_code == 200:
                        data = res.json()
                        df = pd.DataFrame(data).iloc[:, [0, 1, 2, 3, 4, 5, 9]]
                        df.columns = ["ot", "o", "h", "l", "c", "v", "tbv"]
                        df[["o", "h", "l", "c", "v", "tbv"]] = df[["o", "h", "l", "c", "v", "tbv"]].astype(float)
                        async with market_data.lock:
                            market_data.klines[symbol][i] = df
                        success_count += 1
                except Exception as e: 
                    bot_state["last_log"] = f"[red]Fetch {symbol} {i} Fail: {str(e)[:20]}[/]"

            if success_count == len(needed_rest):
                market_data.last_prime[symbol] = now

        # Skip analysis if we don't have all intervals
        if not all(i in market_data.klines.get(symbol, {}) for i in intervals):
            return None

            
        d1m = market_data.klines[symbol]["1m"]
        d5m = market_data.klines[symbol]["5m"]
        d15m = market_data.klines[symbol]["15m"]
        d1h = market_data.klines[symbol]["1h"]
        
        if d1m.empty or d5m.empty or d15m.empty or d1h.empty: return None
        
        price = d1m["c"].iloc[-1]
        
        # --- MARKET REGIME DETECTOR ---
        regime = MarketAnalyzer.detect_regime(d15m)
        if regime == "TRENDING":
            rr_min = 2.0
            lev_max = 20
            gap_max = 0.25 # More room in trending
        elif regime == "VOLATILE":
            rr_min = 1.5
            lev_max = 10
            gap_max = 0.35 # Even more room in volatile
        else:
            rr_min = 1.2
            lev_max = 15
            gap_max = 0.15 # Strict in ranging

        struct_1m, fvg_1m = MarketAnalyzer.detect_structure(d1m)
        struct_5m, _ = MarketAnalyzer.detect_structure(d5m)
        struct_15m, fvg_15m = MarketAnalyzer.detect_structure(d15m)
        struct_1h, _ = MarketAnalyzer.detect_structure(d1h)
        
        ema200_1h = ta.ema(d1h["c"], 200)
        major_trend = 1 if (ema200_1h is not None and not ema200_1h.empty and d1h["c"].iloc[-1] > ema200_1h.iloc[-1]) else -1
        
        ema9 = ta.ema(d1m["c"], 9)
        ema21 = ta.ema(d1m["c"], 21)
        if ema9 is None or ema21 is None: return None
        direction = 1 if ema9.iloc[-1] > ema21.iloc[-1] else -1

        ema9_15m = ta.ema(d15m["c"], 9)
        ema21_15m = ta.ema(d15m["c"], 21)
        dir_intra = 1 if (ema9_15m is not None and not ema9_15m.empty and ema9_15m.iloc[-1] > ema21_15m.iloc[-1]) else -1
        
        ema9_5m = ta.ema(d5m["c"], 9)
        ema21_5m = ta.ema(d5m["c"], 21)
        dir_5m = 1 if (ema9_5m is not None and not ema9_5m.empty and ema9_5m.iloc[-1] > ema21_5m.iloc[-1]) else -1

        # --- QUANTUM CONTEXT (v4.0) ---
        # 1. Predictive & Momentum Factors
        liq_1m = MarketAnalyzer.predict_liquidation_clusters(d1m)
        ml_breakout = MarketAnalyzer.detect_volatility_breakout(d1m)
        sweep_1m = MarketAnalyzer.detect_sweep(d1m)
        div_1m = MarketAnalyzer.detect_rsi_divergence(d1m)
        vol_anomaly = MarketAnalyzer.detect_volume_anomaly(d1m)
        
        # 2. Confluence Analysis
        # CVD RADAR: Aggressive Market Orders (Tape Reading)
        cvd_ratio = d1m["tbv"].tail(3).sum() / d1m["v"].tail(3).sum() if d1m["v"].tail(3).sum() > 0 else 0.5
        
        # Strong Momentum factors that allow for rule bending
        has_conviction = (sweep_1m or div_1m or ml_breakout or (isinstance(liq_1m, str) and "SWEPT" in liq_1m) or vol_anomaly)
        
        # Dynamic Confluence: 4-TF Sync OR (3-TF Sync + High Conviction)
        perfect_sync = (direction == dir_5m == dir_intra == major_trend)
        # Calculate how many TFs are aligned with the signal direction
        tf_match_count = sum([direction == dir_5m, direction == dir_intra, direction == major_trend]) + 1
        
        # OMNISCIENT OVERRIDE: Extreme Order Flow bypasses standard synchronization
        god_mode_override = (cvd_ratio > 0.70 and direction == 1) or (cvd_ratio < 0.30 and direction == -1)
        dynamic_sync = perfect_sync or (tf_match_count >= 3 and has_conviction) or god_mode_override

        # --- SCORING & L2 FLOW ---
        p_flags = {"liq": liq_1m, "ml": ml_breakout}
        conf = bot_state.get("ai_confidence", 1.0)
        
        # Session Awareness
        session = get_current_session()
        if session in ["LONDON", "NEW_YORK"]:
            conf *= 1.1 
        
        # Context-Aware Weight Selection
        all_weights = bot_state.get("neural_weights", {})
        current_weights = {
            feat: all_weights.get(f"{regime}:{feat}", 1.0) 
            for feat in ["liq", "ml", "ob", "div"]
        }
        
        # Calculate OI Delta
        curr_oi = market_data.oi.get(symbol, 0)
        prev_oi = market_data.prev_oi.get(symbol, curr_oi)
        oi_delta = (curr_oi - prev_oi) / prev_oi if prev_oi > 0 else 0
        market_data.prev_oi[symbol] = curr_oi
        
        # --- NARRATIVE/SECTOR BREADTH (v9.0 Supreme) ---
        sector_bonus = 0
        bullish_radar = [r for r in bot_state.get("last_scan_results", []) if r.get("score", 0) >= 70 and r.get("dir") == 1]
        bearish_radar = [r for r in bot_state.get("last_scan_results", []) if r.get("score", 0) >= 70 and r.get("dir") == -1]
        if direction == 1 and len(bullish_radar) >= 4: sector_bonus = 15
        elif direction == -1 and len(bearish_radar) >= 4: sector_bonus = 15
        
        score_scalp = MarketAnalyzer.calculate_score(d1m, d5m, direction, predictive_flags=p_flags, neural_weights=current_weights, oi_delta=oi_delta) + sector_bonus
        score_intra = MarketAnalyzer.calculate_score(d15m, d1h, dir_intra, neural_weights=current_weights, oi_delta=oi_delta) + sector_bonus

        # --- L2 ORDERBOOK FLOW (SUPREME) ---
        l2_ratio = 1.0
        if score_scalp >= 50:
            l2_ratio = await get_orderbook_imbalance(client, symbol)

        # --- SQUEEZE RADAR (Funding & OI) ---
        funding = market_data.funding.get(symbol, 0.01)
        is_sqz_long = funding < -0.01 
        is_sqz_short = funding > 0.03
        
        # Order Block Hunter (15m is strong for OB)
        ob_zone = MarketAnalyzer.find_nearest_order_block(d15m, price, direction)
        in_ob = False
        if ob_zone:
            if direction == 1: in_ob = price <= ob_zone["high"] * 1.001
            else: in_ob = price >= ob_zone["low"] * 0.999

        signal = "SCANNING"
        
        # Market Regime Gap adjustment
        gap_max = 0.15 # Default
        if regime == "TRENDING": gap_max = 0.25
        elif regime == "VOLATILE": gap_max = 0.35
        eff_gap = gap_max
        
        # --- SCALPING FIRST PRIORITY (SNIPER MODE) ---
        ema9_val = ema9.iloc[-1]
        dist_ema = abs(price - ema9_val) / price * 100
        
        if score_scalp >= (75 * conf):
            # Adaptive Execution (Smart FOMO): Drastically expand gap if breakout confirmed
            # Absolute Execution: If score is 90%+, ignore GAP filter almost entirely (allow up to 5%)
            if score_scalp >= 90: eff_gap = 5.0
            elif ml_breakout or vol_anomaly: eff_gap = gap_max * 3.0
            elif (div_1m or in_ob or is_sqz_long or is_sqz_short or sweep_1m): eff_gap = gap_max * 1.5
            else: eff_gap = gap_max
            
            if dist_ema <= eff_gap:
                if direction == 1 and major_trend == 1:
                    # L2 Verification: Block Longs if Sell Wall is present
                    if l2_ratio < 0.6: signal = "WAIT (L2_WALL)"
                    # Perfect Alignment OR SMC Sweep OR OB Confirmation OR SQUEEZE OR LIQ HUNT OR ML
                    elif dynamic_sync and (struct_1m == "BULLISH" or fvg_1m == "BULL_FVG" or sweep_1m == "BULL_SWEEP" or in_ob or div_1m == "BULL_DIV" or is_sqz_long or (isinstance(liq_1m, str) and "BULL" in liq_1m) or ml_breakout):
                        if liq_1m == "LIQ_SWEPT_BULL": signal = "LIQ-HUNT"
                        elif ml_breakout: signal = "ML-BO"
                        elif is_sqz_long: signal = "SQZ-LONG"
                        else: signal = "SCALP-LONG"
                elif direction == -1 and major_trend == -1:
                    # L2 Verification: Block Shorts if Buy Wall is present
                    if l2_ratio > 1.6: signal = "WAIT (L2_WALL)"
                    elif dynamic_sync and (struct_1m == "BEARISH" or fvg_1m == "BEAR_FVG" or sweep_1m == "BEAR_SWEEP" or in_ob or div_1m == "BEAR_DIV" or is_sqz_short or (isinstance(liq_1m, str) and "BEAR" in liq_1m) or ml_breakout):
                        if liq_1m == "LIQ_SWEPT_BEAR": signal = "LIQ-HUNT"
                        elif ml_breakout: signal = "ML-BO"
                        elif is_sqz_short: signal = "SQZ-SHORT"
                        else: signal = "SCALP-SHORT"

        # --- INTRADAY SECOND PRIORITY ---
        if signal == "SCANNING" and score_intra >= (70 * conf): 
            ema9_15m_val = ema9_15m.iloc[-1]
            dist_ema_15m = abs(price - ema9_15m_val) / price * 100
            
            # Intraday also uses Dynamic Confluence
            if dist_ema_15m <= (gap_max * 1.5): 
                if dynamic_sync and dir_intra == direction:
                    if struct_15m == "BULLISH" or fvg_15m == "BULL_FVG":
                        signal = "INTRA-LONG"
                    elif struct_15m == "BEARISH" or fvg_15m == "BEAR_FVG":
                        signal = "INTRA-SHORT"

        score_to_show = score_intra if "INTRA" in signal else score_scalp

        if signal == "SCANNING": 
            # --- DETAILED WAIT REASON (HIGH PROBABILITY FEEDBACK) ---
            wait_reason = "WAIT"
            if score_scalp < (75 * conf) and score_intra < (70 * conf): 
                wait_reason = "WAIT (SCORE)"
            elif not dynamic_sync:
                wait_reason = "WAIT (SYNC)" # Multi-TF not aligned
            elif major_trend == 1 and direction == -1:
                wait_reason = "WAIT (TR-MICRO)" 
            elif major_trend == -1 and direction == 1:
                wait_reason = "WAIT (TR-MICRO)" 
            elif (direction == 1 and struct_1m != "BULLISH" and sweep_1m != "BULL_SWEEP" and not in_ob and not div_1m):
                wait_reason = "WAIT (SMC/OB)"
            elif (direction == -1 and struct_1m != "BEARISH" and sweep_1m != "BEAR_SWEEP" and not in_ob and not div_1m):
                wait_reason = "WAIT (SMC/OB)"
            elif dist_ema > eff_gap: 
                wait_reason = "WAIT (GAP)"
            elif USE_BTC_FILTER and ((direction == 1 and bot_state["btc_dir"] == -1) or (direction == -1 and bot_state["btc_dir"] == 1)):
                wait_reason = "WAIT (BTC)"
            
            return {
                "sym": symbol.replace("USDT", ""), 
                "price": f"{price:,.4f}", 
                "sig": wait_reason, 
                "score": max(score_intra, score_scalp),
                "struct": f"{struct_1h[:4]}/{struct_15m[:4]}",
                "dir": direction, 
                "dir_1m": direction,
                "dir_15m": dir_intra,
                "ai": None,
                "regime": regime,
                "sync": dynamic_sync,
                "sweep": sweep_1m,
                "div": div_1m,
                "ob": True if ob_zone else False
            }

        # --- QUANTUM SL & TP (Volatility Adjusted) ---
        rh_1m, rl_1m = MarketAnalyzer.get_structure_levels(d1m)
        rh_5m, rl_5m = MarketAnalyzer.get_structure_levels(d5m)
        
        atr = ta.atr(d1m["h"], d1m["l"], d1m["c"], length=14)
        atr_val = atr.iloc[-2] if atr is not None and not atr.empty else price * 0.005
        
        # Volatility Multiplier: If market is crazy, give more room
        vol_mult = 1.0
        if bot_state["market_vol"] > 1.5: vol_mult = 1.5
        elif bot_state["market_vol"] < 0.7: vol_mult = 0.8
        
        # --- ADAPTIVE ENTRY (v8.0 PREDATOR) ---
        vol_sma = d1m["v"].rolling(20).mean().iloc[-1]
        is_aggressive = score_to_show >= 85 and d1m["v"].iloc[-1] > (vol_sma * 1.5)
        
        fvg_data = MarketAnalyzer.get_nearest_fvg(d1m)
        ema9_1m = ta.ema(d1m["c"], 9).iloc[-1]

        # --- DYNAMIC TARGETS (Institutional Grade) ---
        # 1. Stop Loss: Always behind recent structure or liquidity pool
        if "LONG" in signal:
            if isinstance(liq_1m, dict) and liq_1m.get("lower"): sl_base = liq_1m["lower"]
            else: sl_base = rl_5m
            sl_price = sl_base - (atr_val * 1.0 * vol_mult)
            sl_pct = ((price - sl_price) / price) * 100
        else:
            if isinstance(liq_1m, dict) and liq_1m.get("upper"): sl_base = liq_1m["upper"]
            else: sl_base = rh_5m
            sl_price = sl_base + (atr_val * 1.0 * vol_mult)
            sl_pct = ((sl_price - price) / price) * 100
        
        sl_pct = max(0.6, min(2.5, sl_pct)) # Safety bounds
        
        # 2. Take Profit: Look for "Undeveloped" liquidity or targets
        tp_price = None
        if "LONG" in signal:
            # Target 1: Bearish OB on 15m (Major Resistance)
            ob_res = MarketAnalyzer.find_nearest_order_block(d15m, price, -1)
            # Target 2: High Liquidity Pool
            liq_target = liq_1m.get("upper") if isinstance(liq_1m, dict) else None
            
            if ob_res: tp_price = ob_res["low"] * 0.999 # Exit just before OB
            elif liq_target: tp_price = liq_target * 0.999
            
            if tp_price:
                tp_pct = ((tp_price - price) / price) * 100
            else:
                tp_pct = sl_pct * (rr_min + 0.5 if "INTRA" in signal else rr_min)
        else:
            # Target 1: Bullish OB on 15m (Major Support)
            ob_supp = MarketAnalyzer.find_nearest_order_block(d15m, price, 1)
            # Target 2: Low Liquidity Pool
            liq_target = liq_1m.get("lower") if isinstance(liq_1m, dict) else None
            
            if ob_supp: tp_price = ob_supp["high"] * 1.001 # Exit just before OB
            elif liq_target: tp_price = liq_target * 1.001
            
            if tp_price:
                tp_pct = ((price - tp_price) / price) * 100
            else:
                tp_pct = sl_pct * (rr_min + 0.5 if "INTRA" in signal else rr_min)

        # Ensure Minimum RR of 1.2 even if target is very close
        tp_pct = max(tp_pct, sl_pct * 1.2)
        
        # Entry Logic
        if "LONG" in signal:
            if is_aggressive:
                if fvg_data and fvg_data["type"] == "BULL_FVG": limit_price = fvg_data["mid"]
                else: limit_price = max(ema9_1m, (price + rl_1m) / 2)
                entry_mode = "MOMENTUM"
            else:
                ob_zone_local = MarketAnalyzer.find_nearest_order_block(d1m, price, 1)
                if ob_zone_local: limit_price = (ob_zone_local["high"] + ob_zone_local["low"]) / 2
                else: limit_price = min(price, (price + rl_1m) / 2)
                entry_mode = "PULLBACK"
        else:
            if is_aggressive:
                if fvg_data and fvg_data["type"] == "BEAR_FVG": limit_price = fvg_data["mid"]
                else: limit_price = min(ema9_1m, (price + rh_1m) / 2)
                entry_mode = "MOMENTUM"
            else:
                ob_zone_local = MarketAnalyzer.find_nearest_order_block(d1m, price, -1)
                if ob_zone_local: limit_price = (ob_zone_local["high"] + ob_zone_local["low"]) / 2
                else: limit_price = max(price, (price + rh_1m) / 2)
                entry_mode = "PULLBACK"

        atr_pct = (atr_val / price) * 100

        score_to_show = score_intra if "INTRA" in signal else score_scalp

        ai_brain = {
            "sl": sl_pct, 
            "tp": tp_pct, 
            "limit_price": limit_price,
            "ts_act": sl_pct * 0.8, 
            "ts_cb": sl_pct * 0.25,
            "type": "INTRA" if "INTRA" in signal else "SCALP",
            "regime": regime,
            "lev_max": lev_max,
            "score": score_to_show,
            "entry_mode": entry_mode,
            "trail_struct": rl_5m if "LONG" in signal else rh_5m,
            "atr_pct": atr_pct,
            "liq": True if liq_1m else False,
            "ml": True if ml_breakout else False,
            "ob": True if ob_zone else False,
            "div": True if div_1m else False
        }
        struct_to_show = f"{struct_1h[:4]}/{struct_15m[:4]}" if "INTRA" in signal else f"{struct_5m[:4]}/{struct_1m[:4]}"
        dir_to_show = dir_intra if "INTRA" in signal else direction
            
        return {
            "sym": symbol.replace("USDT", ""), 
            "price": f"{price:,.4f}", 
            "limit": f"{limit_price:,.4f}",
            "sig": signal, 
            "score": score_to_show,
            "struct": struct_to_show,
            "dir": dir_to_show, 
            "dir_15m": dir_intra,
            "ai": ai_brain,
            "regime": regime,
            "cvd": cvd_ratio
        }
    except Exception as e: return None
