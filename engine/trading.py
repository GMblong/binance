import time
import math
from utils.config import ACCOUNT_RISK_PERCENT, MAX_LEVERAGE, EXIT_ON_REVERSAL, GLOBAL_BTC_EXIT, DAILY_LOSS_LIMIT_PCT, DAILY_PROFIT_TARGET_PCT
from utils.state import bot_state, market_data
from utils.helpers import round_step
from engine.api import binance_request, get_symbol_precision
from utils.logger import log_error

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
        
        # --- DYNAMIC POSITION SIZING ---
        ml_prob = ai_brain.get("ml_prob", 0.5)
        if ml_prob > 0.75 or ml_prob < 0.25:
            risk_amt = base_risk * 1.2 # 120% of normal risk for high conviction
        elif ml_prob >= 0.6 or ml_prob <= 0.4:
            risk_amt = base_risk # 100% of normal risk
        else:
            risk_amt = base_risk * 0.5 # 50% of normal risk for low conviction

        sl_pct = max(0.1, ai_brain.get("sl", 1.0)) # Safety: Min 0.1% SL
        
        # Quantity = Risk Amount / (Price * SL%)
        quantity = risk_amt / (limit_price * (sl_pct / 100))
        quantity = round_step(quantity, prec["step"])
        
        if quantity <= 0: return False

        # Apply Leverage
        notional = quantity * limit_price
        max_notional = balance * MAX_LEVERAGE
        if notional > max_notional:
            quantity = round_step(max_notional / limit_price * 0.9, prec["step"])
        
        if quantity <= 0: return False

        limit_price_str = f"{limit_price:.{prec['p_prec']}f}"

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
            
            bot_state["limit_orders"][symbol] = {
                "orderId": order_data["orderId"],
                "side": side,
                "price": limit_price,
                "quantity": quantity,
                "ai": ai_brain,
                "timestamp": time.time()
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
                    if symbol in bot_state["limit_orders"]: del bot_state["limit_orders"][symbol]
                else:
                    bot_state["trades"][symbol] = {
                        "peak": mark_price, "tp": 2.0, "sl": 1.0, "ts_act": 0.8, "ts_cb": 0.25, 
                        "side": side, "entry_time": time.time(), "type": "INTRA"
                    }
            
            ai = bot_state["trades"].get(symbol, {})
            if side == "LONG": ai["peak"] = max(ai.get("peak", mark_price), mark_price)
            else: ai["peak"] = min(ai.get("peak", mark_price), mark_price)
            peak_pnl = abs((ai["peak"] - entry_price) / entry_price * 100)

            # 2. ORDER PROTECTION (Symbol-specific check, Weight 1)
            orders_res = await binance_request(client, 'GET', '/fapi/v1/openOrders', {"symbol": symbol})
            symbol_orders = orders_res.json() if orders_res and orders_res.status_code == 200 else []
            
            has_sl = any(o['type'] in ['STOP_MARKET', 'STOP'] for o in symbol_orders)
            has_tp = any(o['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT'] for o in symbol_orders)

            if not has_sl or not has_tp:
                prec = await get_symbol_precision(client, symbol)
                if not has_sl:
                    sl_mult = (1 - (ai.get("sl", 1.0)/100)) if side == "LONG" else (1 + (ai.get("sl", 1.0)/100))
                    sl_price = round_step(entry_price * sl_mult, prec["tick"])
                    await binance_request(client, 'POST', '/fapi/v1/algoOrder', {
                        "symbol": symbol, "side": "SELL" if side == "LONG" else "BUY", 
                        "type": "STOP_MARKET", "algoType": "CONDITIONAL", "triggerPrice": f"{sl_price:.{prec['p_prec']}f}",
                        "closePosition": "true", "workingType": "MARK_PRICE"
                    })
                if not has_tp:
                    tp_mult = (1 + (ai.get("tp", 2.0)/100)) if side == "LONG" else (1 - (ai.get("tp", 2.0)/100))
                    tp_price = round_step(entry_price * tp_mult, prec["tick"])
                    await binance_request(client, 'POST', '/fapi/v1/algoOrder', {
                        "symbol": symbol, "side": "SELL" if side == "LONG" else "BUY", 
                        "type": "TAKE_PROFIT_MARKET", "algoType": "CONDITIONAL", "triggerPrice": f"{tp_price:.{prec['p_prec']}f}",
                        "closePosition": "true", "workingType": "MARK_PRICE"
                    })

            # 3. EXIT LOGIC
            reason = None
            if bot_state["btc_state"] == "DANGER" and GLOBAL_BTC_EXIT: reason = "BTC-DANGER"
            
            trail_act = ai.get("ts_act", 0.8)
            base_cb = ai.get("ts_cb", 0.25)
            
            # --- VOLATILITY-ADJUSTED DYNAMIC TRAILING STOP ---
            # Adjust callback based on global market volatility to avoid being stopped out by noise
            market_vol = bot_state.get("market_vol", 1.0)
            vol_multiplier = min(max(market_vol, 0.5), 2.0) # Cap multiplier between 0.5x and 2x
            
            # Base callback is expanded in high volatility, tightened in low volatility
            dynamic_cb = base_cb * vol_multiplier
            
            # As profit grows larger, tighten the callback to lock in profit (Tiered + Volatility aware)
            current_cb = dynamic_cb
            if peak_pnl >= trail_act * 3:
                current_cb = dynamic_cb * 0.3  # Super tight (30%) if 3x target
            elif peak_pnl >= trail_act * 2:
                current_cb = dynamic_cb * 0.6  # Medium tight (60%) if 2x target
            elif peak_pnl >= trail_act * 1.5:
                current_cb = dynamic_cb * 0.8  # Slightly tight if 1.5x target
                
            if peak_pnl >= trail_act and (peak_pnl - pnl_pct) >= current_cb:
                reason = f"AI-TRAIL {pnl_pct:.2f}% (CB: {current_cb:.2f}%)"
                
            # Smart Exit / Reversal Detection
            if EXIT_ON_REVERSAL and not reason:
                sym_short = symbol.replace("USDT", "")
                curr_sig = next((s for s in all_signals if s["sym"] == sym_short), None)
                if curr_sig and "ai" in curr_sig:
                    sig_dir = curr_sig["dir"]
                    curr_score = curr_sig.get("score", 0)
                    ml_prob = curr_sig["ai"].get("ml_prob", 0.5)
                    
                    # 1. Reversal Detection (Cutting Losses Early)
                    if side == "LONG" and sig_dir == -1 and ml_prob < 0.4:
                        reason = "AI-REVERSAL (BEARISH)"
                    elif side == "SHORT" and sig_dir == 1 and ml_prob > 0.6:
                        reason = "AI-REVERSAL (BULLISH)"
                    
                    # 2. Smart Take Profit (Momentum Exhaustion)
                    # If we are in profit, but the trend has completely lost momentum (score dropped below 30)
                    elif pnl_pct > 0.3: # Only trigger if we have at least 0.3% profit
                        if side == "LONG" and curr_score < 30 and ml_prob < 0.5:
                            reason = "SMART-TP (MOMENTUM DEAD)"
                        elif side == "SHORT" and curr_score < 30 and ml_prob > 0.5:
                            reason = "SMART-TP (MOMENTUM DEAD)"
            
            if reason:
                await close_position_async(client, symbol, side, amt, reason, pnl_pct)

        return positions
    except Exception as e:
        log_error(f"Manage Positions Exception: {str(e)}")
        return []
