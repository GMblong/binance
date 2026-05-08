import time
import math
from utils.config import ACCOUNT_RISK_PERCENT, MAX_LEVERAGE, EXIT_ON_REVERSAL, GLOBAL_BTC_EXIT, DAILY_LOSS_LIMIT_PCT, DAILY_PROFIT_TARGET_PCT
from utils.state import bot_state, market_data
from utils.helpers import round_step
from engine.api import binance_request, get_symbol_precision
from utils.logger import log_error
from utils.intelligence import calculate_kelly_risk, update_feature_weights
from engine.ml_engine import ml_predictor
from strategies.analyzer import MarketAnalyzer

async def open_position_async(client, symbol, side, signal_type, ai_brain):
    try:
        prec = await get_symbol_precision(client, symbol)
        
        # Use live price from market_data if possible
        curr_price = market_data.prices.get(symbol, ai_brain["limit_price"])
        # CRITICAL: Round limit price to TICK SIZE
        limit_price = round_step(ai_brain["limit_price"], prec["tick"])
        
        # Risk Management: Calculate Quantity
        balance = bot_state["balance"]
        base_risk = balance * ACCOUNT_RISK_PERCENT
        
        # --- DYNAMIC POSITION SIZING (KELLY + ML) ---
        ml_prob = ai_brain.get("ml_prob", 0.5)
        
        # Get historical win rate from ML performance
        perf = ml_predictor.performance.get(symbol, [])
        win_rate = sum(perf) / len(perf) if len(perf) >= 5 else 0.5
        
        kelly_mult = calculate_kelly_risk(symbol, win_rate=win_rate, rr=2.0)
        
        if ml_prob > 0.75 or ml_prob < 0.25:
            risk_amt = base_risk * 1.2 * kelly_mult
        elif ml_prob >= 0.6 or ml_prob <= 0.4:
            risk_amt = base_risk * kelly_mult
        else:
            risk_amt = base_risk * 0.5 * kelly_mult

        sl_pct = max(0.1, ai_brain.get("sl", 1.0)) # Safety: Min 0.1% SL
        
        # Prevent division by zero if limit_price is too small and rounded to 0
        if limit_price <= 0:
            return False
            
        # Quantity = Risk Amount / (Price * SL%)
        quantity = risk_amt / (limit_price * (sl_pct / 100))
        quantity = round_step(quantity, prec["step"])
        
        if quantity <= 0: return False

        # Apply Leverage
        notional = quantity * limit_price
        max_notional = balance * MAX_LEVERAGE
        if notional > max_notional:
            quantity = round_step(max_notional / limit_price * 0.9, prec["step"])
            notional = quantity * limit_price
        
        if quantity <= 0: return False
        
        # Minimum notional check (Binance requires $5-$20 depending on pair)
        if notional < 5.5:
            return False

        limit_price_str = f"{limit_price:.{prec['p_prec']}f}"

        # --- SET LEVERAGE EXPLICITLY DYNAMICALLY ---
        # Use cached tier if available, otherwise try from MAX down
        cached_lev = bot_state.get("_lev_cache", {}).get(symbol)
        lev_targets = [cached_lev] if cached_lev else [MAX_LEVERAGE, 25, 20, 15, 10, 5, 2, 1]
        
        for target_lev in lev_targets:
            try:
                lev_res = await binance_request(client, 'POST', '/fapi/v1/leverage', {"symbol": symbol, "leverage": target_lev})
                if lev_res and lev_res.status_code == 200:
                    max_notional_tier = float(lev_res.json().get("maxNotionalValue", max_notional))
                    # Cache successful leverage for this symbol
                    bot_state.setdefault("_lev_cache", {})[symbol] = target_lev
                    if notional > max_notional_tier:
                        quantity = round_step(max_notional_tier / limit_price * 0.9, prec["step"])
                        notional = quantity * limit_price
                    break
                elif lev_res and lev_res.status_code == 400:
                    continue
            except: pass
            
        if quantity <= 0: return False
        
        # Also attempt to set margin type to ISOLATED to avoid cross margin issues
        try:
            await binance_request(client, 'POST', '/fapi/v1/marginType', {"symbol": symbol, "marginType": "ISOLATED"})
        except: pass

        if quantity <= 0: return False

        # --- ORDER EXECUTION (MARKET vs LIMIT) ---
        is_market = ai_brain.get("is_market", False)
        order_type = "MARKET" if is_market else "LIMIT"
        
        order_params = {
            "symbol": symbol, "side": side, "type": order_type,
            "quantity": f"{quantity:.{prec['q_prec']}f}"
        }
        
        if not is_market:
            order_params["price"] = limit_price_str
            order_params["timeInForce"] = "GTC"
        
        res = await binance_request(client, 'POST', '/fapi/v1/order', order_params)
        
        if res and res.status_code == 200:
            order_data = res.json()
            # For market orders, use the actual fill price if available, else curr_price
            execution_price = float(order_data.get("avgPrice", 0)) or curr_price
            if is_market:
                bot_state["last_log"] = f"[bold magenta]MARKET {side} {symbol} at {execution_price:.{prec['p_prec']}f}[/]"
            else:
                bot_state["last_log"] = f"[bold cyan]LIMIT {side} {symbol} at {limit_price_str}[/]"
            
            # --- PLACE SERVER-SIDE SL & TP (only for MARKET orders - limit orders get SL/TP after fill) ---
            if is_market:
                try:
                    ref_price = execution_price
                    sl_mult = (1 - (ai_brain["sl"]/100)) if side == "BUY" else (1 + (ai_brain["sl"]/100))
                    sl_price = round_step(ref_price * sl_mult, prec["tick"])
                    sl_price_str = f"{sl_price:.{prec['p_prec']}f}"

                    tp_mult = (1 + (ai_brain["tp"]/100)) if side == "BUY" else (1 - (ai_brain["tp"]/100))
                    tp_price = round_step(ref_price * tp_mult, prec["tick"])
                    tp_price_str = f"{tp_price:.{prec['p_prec']}f}"

                    sl_res = await binance_request(client, 'POST', '/fapi/v1/order', {
                        "symbol": symbol, "side": "SELL" if side == "BUY" else "BUY", 
                        "type": "STOP_MARKET", "stopPrice": sl_price_str, 
                        "closePosition": "true", "workingType": "MARK_PRICE"
                    })

                    tp_res = await binance_request(client, 'POST', '/fapi/v1/order', {
                        "symbol": symbol, "side": "SELL" if side == "BUY" else "BUY", 
                        "type": "TAKE_PROFIT_MARKET", "stopPrice": tp_price_str, 
                        "closePosition": "true", "workingType": "MARK_PRICE"
                    })
                    
                    if sl_res and sl_res.status_code != 200:
                        err_msg = sl_res.json().get("msg", "")
                        if "existing" not in err_msg.lower():
                            log_error(f"SL Placement Fail for {symbol}: {err_msg}")
                except Exception as e:
                    log_error(f"SL/TP Placement Exception for {symbol}: {str(e)}")
            
            if not is_market:
                bot_state["limit_orders"][symbol] = {
                    "orderId": order_data["orderId"],
                    "side": side,
                    "price": limit_price,
                    "quantity": quantity,
                    "ai": ai_brain,
                    "timestamp": time.time()
                }
            else:
                # For Market orders, we sync to trades immediately so it doesn't show as "pending" in UI
                bot_state["trades"][symbol] = {
                    "peak": execution_price, 
                    "tp": ai_brain.get("tp", 2.0), 
                    "sl": ai_brain.get("sl", 1.0), 
                    "ts_act": ai_brain.get("ts_act", 0.8), 
                    "ts_cb": ai_brain.get("ts_cb", 0.25), 
                    "side": "LONG" if side == "BUY" else "SHORT", 
                    "entry_time": time.time(),
                    "type": ai_brain.get("type", "INTRA"), 
                    "regime": ai_brain.get("regime", "TRENDING"), 
                    "atr_pct": ai_brain.get("atr_pct", 0.5),
                    "active_features": list(ai_brain.get("active_features", [])),
                }
            return True
        else:
            msg = res.json().get("msg", "Unknown") if res else "No Response"
            bot_state["last_log"] = f"[red]Limit Order Fail {symbol}: {msg}[/]"
            log_error(f"Limit Order Fail {symbol}: {msg}")
            return False
            
    except Exception as e:
        log_error(f"Limit Order Exception ({symbol}): {str(e)}")
        return False

async def close_position_async(client, symbol, side, amount, reason, pnl=0.0):
    try:
        prec = await get_symbol_precision(client, symbol)
        close_side = "SELL" if side == "LONG" else "BUY"
        
        # 1. Cancel all open orders for this symbol first
        await binance_request(client, 'DELETE', '/fapi/v1/allOpenOrders', {"symbol": symbol})
        
        # 2. Place Market Close Order
        res = await binance_request(client, 'POST', '/fapi/v1/order', {
            "symbol": symbol, "side": close_side, "type": "MARKET",
            "quantity": f"{abs(amount):.{prec['q_prec']}f}", "reduceOnly": "true"
        })
        
        if res and res.status_code == 200:
            col = "green" if pnl > 0 else "red"
            bot_state["last_log"] = f"[bold {col}]CLOSED {symbol} ({reason}) | Est. PnL: {pnl:+.2f}%[/]"
            
            # Mark as recently closed so WebSocket doesn't double-count
            bot_state.setdefault("_recently_closed", {})[symbol] = time.time()

            # Update ML Predictor Performance (single source of truth)
            is_win = pnl > 0
            ml_predictor.update_performance(symbol, is_win)

            # Feedback loop: credit/penalize the features that fired at entry
            trade_meta = bot_state.get("trades", {}).get(symbol, {})
            active_feats = trade_meta.get("active_features") or []
            if active_feats:
                update_feature_weights(active_feats, is_win)

            if is_win:
                bot_state["wins"] = bot_state.get("wins", 0) + 1
            else:
                bot_state["losses"] = bot_state.get("losses", 0) + 1
            
            # Update Symbol Performance
            if symbol not in bot_state["sym_perf"]: bot_state["sym_perf"][symbol] = {'w':0, 'l':0, 'c':0}
            if is_win:
                bot_state["sym_perf"][symbol]['w'] += 1
                bot_state["sym_perf"][symbol]['c'] = 0
            else:
                bot_state["sym_perf"][symbol]['l'] += 1
                bot_state["sym_perf"][symbol]['c'] += 1
                bot_state["sym_perf"][symbol]['last_loss_time'] = time.time()
            
            if symbol in bot_state["trades"]: del bot_state["trades"][symbol]
            return True
        return False
    except Exception as e:
        log_error(f"Close Position Exception ({symbol}): {str(e)}")
        return False

async def manage_limit_orders(client, all_signals):
    try:
        now = time.time()
        for symbol, lo in list(bot_state["limit_orders"].items()):
            age = now - lo["timestamp"]
            # Cancel stale limit orders after 3 minutes (sniper needs time for pullback)
            if age > 180:
                await binance_request(client, 'DELETE', '/fapi/v1/order', {"symbol": symbol, "orderId": lo["orderId"]})
                del bot_state["limit_orders"][symbol]
                bot_state["last_log"] = f"[dim]Cancelled stale limit for {symbol}[/]"
            # Also cancel if signal flipped direction
            elif all_signals:
                sym_short = symbol.replace("USDT", "")
                curr_sig = next((s for s in all_signals if s["sym"] == sym_short), None)
                if curr_sig and curr_sig.get("dir"):
                    lo_dir = 1 if lo["side"] == "BUY" else -1
                    if curr_sig["dir"] != lo_dir:
                        await binance_request(client, 'DELETE', '/fapi/v1/order', {"symbol": symbol, "orderId": lo["orderId"]})
                        del bot_state["limit_orders"][symbol]
                        bot_state["last_log"] = f"[dim]Cancelled reversed limit for {symbol}[/]"
    except Exception as e:
        log_error(f"Manage Limit Orders Exception: {str(e)}")

last_exit_check = {}

async def partial_close_async(client, symbol, side, full_amt, pct, reason, pnl):
    """Close a fraction of position. Returns True if executed."""
    try:
        prec = await get_symbol_precision(client, symbol)
        close_qty = round_step(abs(full_amt) * pct, prec["step"])
        if close_qty <= 0:
            return False
        close_side = "SELL" if side == "LONG" else "BUY"
        res = await binance_request(client, 'POST', '/fapi/v1/order', {
            "symbol": symbol, "side": close_side, "type": "MARKET",
            "quantity": f"{close_qty:.{prec['q_prec']}f}", "reduceOnly": "true"
        })
        if res and res.status_code == 200:
            bot_state["last_log"] = f"[bold green]PARTIAL {int(pct*100)}% {symbol} ({reason}) PnL:{pnl:+.2f}%[/]"
            return True
        return False
    except:
        return False

async def check_and_execute_exits(client, symbol, current_price, all_signals=[]):
    """
    ML-Adaptive Exit Engine with:
    1. Structure-based trailing (not fixed %)
    2. Partial TP at key levels
    3. Time decay for stuck trades
    4. Real-time ML re-evaluation
    5. Momentum + VSA confluence exit
    """
    global last_exit_check
    now = time.time()
    
    if now - last_exit_check.get(symbol, 0) < 1.0:
        return False
    last_exit_check[symbol] = now
    
    try:
        ai = bot_state["trades"].get(symbol, {})
        if not ai or ai.get("exit_pending"):
            return False
        
        pos = next((p for p in bot_state.get("active_positions", []) if p['symbol'] == symbol), None)
        if not pos:
            return False
        
        entry_price = float(pos['entryPrice'])
        side = "LONG" if float(pos['positionAmt']) > 0 else "SHORT"
        amt = float(pos['positionAmt'])
        direction = 1 if side == "LONG" else -1
        pnl_pct = ((current_price - entry_price) / entry_price) * 100 * direction
        
        # Update Peak
        if side == "LONG":
            ai["peak"] = max(ai.get("peak", current_price), current_price)
        else:
            ai["peak"] = min(ai.get("peak", current_price), current_price)
        peak_pnl = abs((ai["peak"] - entry_price) / entry_price * 100)
        
        # Get market context
        k1m = market_data.klines.get(symbol, {}).get("1m")
        k15m = market_data.klines.get(symbol, {}).get("15m")
        regime = ai.get("regime", "TRENDING")
        initial_sl = ai.get("sl", 1.0)
        initial_tp = ai.get("tp", 2.0)
        entry_time = ai.get("entry_time", now)
        hold_minutes = (now - entry_time) / 60
        
        reason = None
        
        # === 1. PARTIAL TAKE PROFIT (50% at TP1) ===
        if not ai.get("partial_done") and pnl_pct >= max(initial_sl * 1.0, 0.6):
            # Hit RR 1:1 (min 0.6% to cover fees) -> close 50% and let rest run
            did_partial = await partial_close_async(client, symbol, side, amt, 0.5, "TP1 (RR1:1)", pnl_pct)
            if did_partial:
                ai["partial_done"] = True
                ai["be_active"] = True  # Activate breakeven for remainder
                return False  # Don't full-close, let runner continue
        
        # === 2. STRUCTURE-BASED ADAPTIVE TRAILING ===
        trail_act = ai.get("ts_act", 0.8)
        base_cb = ai.get("ts_cb", 0.25)
        market_vol = bot_state.get("market_vol", 1.0)
        
        # ML-adjusted callback: tighter when ML says momentum dying
        ml_adj = 1.0
        sym_short = symbol.replace("USDT", "")
        curr_sig = next((s for s in all_signals if s["sym"] == sym_short), None) if all_signals else None
        if curr_sig and "ai" in curr_sig:
            ml_prob = curr_sig["ai"].get("ml_prob", 0.5)
            # If ML says momentum weakening, tighten trail
            if side == "LONG" and ml_prob < 0.45:
                ml_adj = 0.6
            elif side == "SHORT" and ml_prob > 0.55:
                ml_adj = 0.6
            # If ML says strong continuation, widen trail
            elif (side == "LONG" and ml_prob > 0.7) or (side == "SHORT" and ml_prob < 0.3):
                ml_adj = 1.4
        
        # Structure-aware callback: use ATR from live data
        struct_cb = base_cb
        if k1m is not None and len(k1m) >= 14:
            live_atr = MarketAnalyzer.get_atr(k1m, 14).iloc[-1]
            live_atr_pct = (live_atr / current_price) * 100
            # Callback = 0.5x current ATR, adjusted by ML and volatility
            struct_cb = max(0.15, live_atr_pct * 0.5 * ml_adj * min(max(market_vol, 0.7), 1.5))
        else:
            struct_cb = base_cb * min(max(market_vol, 0.5), 2.0) * ml_adj
        
        # Progressive tightening as profit grows
        if peak_pnl >= trail_act * 3:
            struct_cb *= 0.4
        elif peak_pnl >= trail_act * 2:
            struct_cb *= 0.6
        elif peak_pnl >= trail_act * 1.5:
            struct_cb *= 0.8
        
        if peak_pnl >= trail_act and (peak_pnl - pnl_pct) >= struct_cb:
            reason = f"AI-TRAIL {pnl_pct:.2f}% (CB:{struct_cb:.2f}%)"
        
        # === 3. BREAKEVEN PROTECTION (after partial or RR 1:1) ===
        if not reason and ai.get("be_active"):
            be_buffer = initial_sl * 0.5  # Give runner room to breathe
            if pnl_pct <= be_buffer:
                reason = "SMART-BE (Protecting profit)"
        elif not reason and peak_pnl >= initial_sl and pnl_pct <= 0.1:
            reason = "SMART-BE (Hit RR1:1 dropped)"
        
        # === 4. TIME DECAY - Exit stuck trades ===
        if not reason:
            # Regime-based max hold time
            max_hold = 30 if regime == "RANGING" else 60 if regime == "TRENDING" else 45
            if hold_minutes > max_hold:
                if pnl_pct > 0.1:
                    reason = f"TIME-TP (Held {int(hold_minutes)}m, PnL:{pnl_pct:.2f}%)"
                elif pnl_pct < -0.3 and hold_minutes > max_hold * 1.5:
                    reason = f"TIME-SL (Stuck {int(hold_minutes)}m, cutting loss)"
            # Moderate decay: if barely moving after 15 min, reduce tolerance
            elif hold_minutes > 15 and abs(pnl_pct) < 0.2:
                # Trade going nowhere - if ML now disagrees, exit
                if curr_sig and "ai" in curr_sig:
                    ml_p = curr_sig["ai"].get("ml_prob", 0.5)
                    if (side == "LONG" and ml_p < 0.4) or (side == "SHORT" and ml_p > 0.6):
                        reason = f"DECAY-EXIT (Flat {int(hold_minutes)}m, ML disagrees)"
        
        # === 5. ML + VSA + STRUCTURE REVERSAL ===
        if EXIT_ON_REVERSAL and not reason and curr_sig and "ai" in curr_sig:
            sig_dir = curr_sig["dir"]
            curr_score = curr_sig.get("score", 0)
            ml_prob = curr_sig["ai"].get("ml_prob", 0.5)
            
            vsa_sig = 0
            struct_break = False
            if k1m is not None and len(k1m) >= 15:
                vsa_sig = MarketAnalyzer.detect_vsa_signals(k1m)
                # Check if structure broke against us
                struct_dir, _, _, _ = MarketAnalyzer.detect_structure(k1m)
                if side == "LONG" and struct_dir == "BEARISH":
                    struct_break = True
                elif side == "SHORT" and struct_dir == "BULLISH":
                    struct_break = True
            
            # Strong reversal: ML + direction flip + structure break
            if side == "LONG" and sig_dir == -1 and ml_prob < 0.38 and struct_break:
                reason = "AI-REVERSAL (ML+STRUCT BEARISH)"
            elif side == "SHORT" and sig_dir == 1 and ml_prob > 0.62 and struct_break:
                reason = "AI-REVERSAL (ML+STRUCT BULLISH)"
            # Moderate reversal: ML disagrees strongly
            elif side == "LONG" and ml_prob < 0.3 and pnl_pct < 0.5:
                reason = "AI-REVERSAL (ML STRONG BEARISH)"
            elif side == "SHORT" and ml_prob > 0.7 and pnl_pct < 0.5:
                reason = "AI-REVERSAL (ML STRONG BULLISH)"
            # Momentum dead with profit
            elif pnl_pct > 0.3:
                momentum_dead = False
                if side == "LONG" and curr_score < 25 and ml_prob < 0.48:
                    momentum_dead = True
                elif side == "SHORT" and curr_score < 25 and ml_prob > 0.52:
                    momentum_dead = True
                if momentum_dead:
                    if vsa_sig == -direction:
                        reason = "SMART-TP (DEAD + VSA CONTRA)"
                    elif struct_break:
                        reason = "SMART-TP (DEAD + STRUCT BREAK)"
                    else:
                        reason = "SMART-TP (MOMENTUM DEAD)"

        if reason:
            ai["exit_pending"] = True
            return await close_position_async(client, symbol, side, amt, reason, pnl_pct)
        return False
    except Exception as e:
        log_error(f"Fast Exit Exception ({symbol}): {str(e)}")
        return False

async def manage_active_positions(client, all_signals):
    try:
        # 1. Get Positions (Weight 5)
        res = await binance_request(client, 'GET', '/fapi/v2/positionRisk')
        if res is None or res.status_code != 200: return []
        
        pos_data = res.json()
        positions = [p for p in pos_data if float(p.get('positionAmt', 0)) != 0]
        
        if not positions:
            bot_state["logged_secure"] = []
            return []
        
        for p in positions:
            symbol, amt = p['symbol'], float(p['positionAmt'])
            entry_price, mark_price = float(p['entryPrice']), float(p['markPrice'])
            side = "LONG" if amt > 0 else "SHORT"
            pnl_pct = ((mark_price - entry_price) / entry_price) * 100 * (1 if side == "LONG" else -1)
            
            # TRADE STATE SYNC
            if symbol not in bot_state["trades"]:
                lo_data = bot_state["limit_orders"].get(symbol)
                if lo_data:
                    ai = lo_data["ai"]
                    bot_state["trades"][symbol] = {
                        "peak": mark_price, "tp": ai.get("tp", 2.0), "sl": ai.get("sl", 1.0), 
                        "ts_act": ai.get("ts_act", 0.8), "ts_cb": ai.get("ts_cb", 0.25), 
                        "side": side, "entry_time": lo_data.get("timestamp", time.time()),
                        "type": ai.get("type", "INTRA"), "regime": ai.get("regime", "TRENDING"), 
                        "atr_pct": ai.get("atr_pct", 0.5),
                        "active_features": list(ai.get("active_features", [])),
                    }
                else:
                    bot_state["trades"][symbol] = {
                        "peak": mark_price, "tp": 2.0, "sl": 1.0, "ts_act": 0.8, "ts_cb": 0.25, 
                        "side": side, "entry_time": time.time(), "type": "INTRA",
                        "active_features": [],
                    }
            
            # CRITICAL: Always remove from limit_orders if we found an active position for this symbol
            if symbol in bot_state["limit_orders"]:
                del bot_state["limit_orders"][symbol]
            
            ai = bot_state["trades"].get(symbol, {})
            if side == "LONG": ai["peak"] = max(ai.get("peak", mark_price), mark_price)
            else: ai["peak"] = min(ai.get("peak", mark_price), mark_price)
            peak_pnl = abs((ai["peak"] - entry_price) / entry_price * 100)

            # 2. ORDER PROTECTION (Symbol-specific check, Weight 1)
            orders_res = await binance_request(client, 'GET', '/fapi/v1/openOrders', {"symbol": symbol})
            symbol_orders = orders_res.json() if orders_res and orders_res.status_code == 200 else []
            
            has_sl = any(o['type'] in ['STOP_MARKET', 'STOP'] for o in symbol_orders)
            has_tp = any(o['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT'] for o in symbol_orders)

            # --- STRUCTURAL SL PROTECTION ---
            # Periodically check for new Order Blocks to move SL behind them
            k1m = market_data.klines.get(symbol, {}).get("1m")
            struct_sl = None
            if k1m is not None:
                if side == "LONG":
                    ob = MarketAnalyzer.find_nearest_order_block(k1m, mark_price, 1)
                    if ob and ob["bottom"] > entry_price:
                        struct_sl = ob["bottom"]
                else:
                    ob = MarketAnalyzer.find_nearest_order_block(k1m, mark_price, -1)
                    if ob and ob["top"] < entry_price:
                        struct_sl = ob["top"]

            if not has_sl or not has_tp or struct_sl:
                prec = await get_symbol_precision(client, symbol)
                if not has_sl or struct_sl:
                    if struct_sl:
                        # Move SL to structure but keep it safe (buffer)
                        sl_price = round_step(struct_sl, prec["tick"])
                    else:
                        sl_mult = (1 - (ai.get("sl", 1.0)/100)) if side == "LONG" else (1 + (ai.get("sl", 1.0)/100))
                        sl_price = round_step(entry_price * sl_mult, prec["tick"])
                    
                    # Update or Create SL
                    if has_sl: # Cancel existing SL first if we are moving it
                        sl_order = next(o for o in symbol_orders if o['type'] in ['STOP_MARKET', 'STOP'])
                        await binance_request(client, 'DELETE', '/fapi/v1/order', {"symbol": symbol, "orderId": sl_order["orderId"]})
                    
                    await binance_request(client, 'POST', '/fapi/v1/order', {
                        "symbol": symbol, "side": "SELL" if side == "LONG" else "BUY", 
                        "type": "STOP_MARKET", "stopPrice": f"{sl_price:.{prec['p_prec']}f}",
                        "closePosition": "true", "workingType": "MARK_PRICE"
                    })

            # 3. EXIT LOGIC (Now handled by Fast Exit Engine)
            all_valid = all_signals # Pass context
            exit_triggered = await check_and_execute_exits(client, symbol, mark_price, all_valid)
            
            if not exit_triggered:
                # Still check for BTC-DANGER which is a global factor
                if bot_state["btc_state"] == "DANGER" and GLOBAL_BTC_EXIT:
                    await close_position_async(client, symbol, side, amt, "BTC-DANGER", pnl_pct)

        return positions
    except Exception as e:
        log_error(f"Manage Positions Exception: {str(e)}")
        return []
