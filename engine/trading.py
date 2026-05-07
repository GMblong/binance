import time
import math
from utils.config import ACCOUNT_RISK_PERCENT, MAX_LEVERAGE, EXIT_ON_REVERSAL, GLOBAL_BTC_EXIT, DAILY_LOSS_LIMIT_PCT, DAILY_PROFIT_TARGET_PCT
from utils.state import bot_state, market_data
from utils.helpers import round_step
from engine.api import binance_request, get_symbol_precision
from utils.logger import log_error
from utils.intelligence import calculate_kelly_risk
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

        limit_price_str = f"{limit_price:.{prec['p_prec']}f}"

        # --- SET LEVERAGE EXPLICITLY DYNAMICALLY ---
        # Attempt to set leverage starting from MAX_LEVERAGE down to 1.
        for target_lev in [MAX_LEVERAGE, 25, 20, 15, 10, 5, 2, 1]:
            try:
                lev_res = await binance_request(client, 'POST', '/fapi/v1/leverage', {"symbol": symbol, "leverage": target_lev})
                if lev_res and lev_res.status_code == 200:
                    max_notional_tier = float(lev_res.json().get("maxNotionalValue", max_notional))
                    # Safely cap quantity if it still exceeds the tier's max notional
                    if notional > max_notional_tier:
                        quantity = round_step(max_notional_tier / limit_price * 0.9, prec["step"])
                        notional = quantity * limit_price
                    break # Success, exit loop
                elif lev_res and lev_res.status_code == 400:
                    # If Binance says this specific leverage is invalid, try the next one in the loop
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
            
            # --- PLACE SERVER-SIDE SL & TP ---
            try:
                # Use execution price for SL/TP calculation
                ref_price = execution_price if is_market else limit_price
                
                sl_mult = (1 - (ai_brain["sl"]/100)) if side == "BUY" else (1 + (ai_brain["sl"]/100))
                sl_price = round_step(ref_price * sl_mult, prec["tick"])
                sl_price_str = f"{sl_price:.{prec['p_prec']}f}"

                tp_mult = (1 + (ai_brain["tp"]/100)) if side == "BUY" else (1 - (ai_brain["tp"]/100))
                tp_price = round_step(ref_price * tp_mult, prec["tick"])
                tp_price_str = f"{tp_price:.{prec['p_prec']}f}"

                # Use STOP_MARKET / TAKE_PROFIT_MARKET but with precise formatting
                sl_res = await binance_request(client, 'POST', '/fapi/v1/algoOrder', {
                    "symbol": symbol, "side": "SELL" if side == "BUY" else "BUY", 
                    "type": "STOP_MARKET", "algoType": "CONDITIONAL", "triggerPrice": sl_price_str, 
                    "closePosition": "true", "workingType": "MARK_PRICE"
                })

                tp_res = await binance_request(client, 'POST', '/fapi/v1/algoOrder', {
                    "symbol": symbol, "side": "SELL" if side == "BUY" else "BUY", 
                    "type": "TAKE_PROFIT_MARKET", "algoType": "CONDITIONAL", "triggerPrice": tp_price_str, 
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
                    "atr_pct": ai_brain.get("atr_pct", 0.5)
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
            
            # Update ML Predictor Performance
            ml_predictor.update_performance(symbol, pnl > 0)
            
            # W/L and Exact Daily PNL is now tracked via WebSocket's Realized Profit (rp) event
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
            # Timeout limit orders after 1 minute (60 seconds)
            if (now - lo["timestamp"]) > 60:
                await binance_request(client, 'DELETE', '/fapi/v1/order', {"symbol": symbol, "orderId": lo["orderId"]})
                del bot_state["limit_orders"][symbol]
                bot_state["last_log"] = f"[dim]Cancelled stale limit for {symbol}[/]"
    except Exception as e:
        log_error(f"Manage Limit Orders Exception: {str(e)}")

async def check_and_execute_exits(client, symbol, current_price, all_signals=[]):
    """
    Fast Execution Engine: Checks SL/TP/Trailing/Reversal for a single symbol.
    Triggered by WebSocket or Main Loop.
    """
    try:
        # Get position details from local state for speed
        pos = next((p for p in bot_state.get("active_positions", []) if p['symbol'] == symbol), None)
        if not pos: return False
        
        entry_price = float(pos['entryPrice'])
        side = "LONG" if float(pos['positionAmt']) > 0 else "SHORT"
        amt = float(pos['positionAmt'])
        pnl_pct = ((current_price - entry_price) / entry_price) * 100 * (1 if side == "LONG" else -1)
        
        ai = bot_state["trades"].get(symbol, {})
        if not ai: return False
        
        # Update Peak for Trailing Stop
        if side == "LONG": ai["peak"] = max(ai.get("peak", current_price), current_price)
        else: ai["peak"] = min(ai.get("peak", current_price), current_price)
        peak_pnl = abs((ai["peak"] - entry_price) / entry_price * 100)

        reason = None
        
        # 1. Trailing Stop Logic (Fast Check)
        trail_act = ai.get("ts_act", 0.8)
        base_cb = ai.get("ts_cb", 0.25)
        market_vol = bot_state.get("market_vol", 1.0)
        vol_multiplier = min(max(market_vol, 0.5), 2.0)
        dynamic_cb = base_cb * vol_multiplier
        
        current_cb = dynamic_cb
        if peak_pnl >= trail_act * 3: current_cb = dynamic_cb * 0.3
        elif peak_pnl >= trail_act * 2: current_cb = dynamic_cb * 0.6
        elif peak_pnl >= trail_act * 1.5: current_cb = dynamic_cb * 0.8
            
        if peak_pnl >= trail_act and (peak_pnl - pnl_pct) >= current_cb:
            reason = f"AI-TRAIL {pnl_pct:.2f}% (CB: {current_cb:.2f}%)"

        # 2. Smart Reversal & Momentum Check
        if EXIT_ON_REVERSAL and not reason and all_signals:
            sym_short = symbol.replace("USDT", "")
            curr_sig = next((s for s in all_signals if s["sym"] == sym_short), None)
            if curr_sig and "ai" in curr_sig:
                sig_dir = curr_sig["dir"]
                curr_score = curr_sig.get("score", 0)
                ml_prob = curr_sig["ai"].get("ml_prob", 0.5)
                
                if side == "LONG" and sig_dir == -1 and ml_prob < 0.4:
                    reason = "AI-REVERSAL (BEARISH)"
                elif side == "SHORT" and sig_dir == 1 and ml_prob > 0.6:
                    reason = "AI-REVERSAL (BULLISH)"
                elif pnl_pct > 0.3:
                    if side == "LONG" and curr_score < 30 and ml_prob < 0.5:
                        reason = "SMART-TP (MOMENTUM DEAD)"
                    elif side == "SHORT" and curr_score < 30 and ml_prob > 0.5:
                        reason = "SMART-TP (MOMENTUM DEAD)"

        if reason:
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
                        "atr_pct": ai.get("atr_pct", 0.5)
                    }
                else:
                    bot_state["trades"][symbol] = {
                        "peak": mark_price, "tp": 2.0, "sl": 1.0, "ts_act": 0.8, "ts_cb": 0.25, 
                        "side": side, "entry_time": time.time(), "type": "INTRA"
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
                    
                    await binance_request(client, 'POST', '/fapi/v1/algoOrder', {
                        "symbol": symbol, "side": "SELL" if side == "LONG" else "BUY", 
                        "type": "STOP_MARKET", "algoType": "CONDITIONAL", "triggerPrice": f"{sl_price:.{prec['p_prec']}f}",
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
