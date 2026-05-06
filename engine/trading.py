import time
import math
from utils.config import ACCOUNT_RISK_PERCENT, MAX_LEVERAGE, EXIT_ON_REVERSAL, GLOBAL_BTC_EXIT, DAILY_LOSS_LIMIT_PCT, DAILY_PROFIT_TARGET_PCT
from utils.state import bot_state, market_data
from utils.helpers import round_step
from engine.api import binance_request, get_symbol_precision, get_balance_async
from utils.database import save_state_to_db
from utils.logger import log_error

async def open_position_async(client, symbol, side, signal_type, ai_brain):
    try:
        # --- EQ Module Checks ---
        if bot_state["state"] == "LOCKDOWN":
            bot_state["last_log"] = "[red]LOCKDOWN: Daily loss limit reached. Trading locked.[/]"
            return
        if bot_state["state"] == "DONE":
            bot_state["last_log"] = "[green]DONE: Daily profit target reached.[/]"
            return
            
        risk_pct = ACCOUNT_RISK_PERCENT
        # --- Dynamic Position Sizing 2.1 (Ultra-Smart) ---
        mv = bot_state.get("market_vol", 1.0)
        
        if mv > 2.0: # Extreme Market Turbulence
            risk_pct *= 0.3
        elif mv > 1.5 or ai_brain.get("regime") == "VOLATILE":
            risk_pct *= 0.5 
        elif mv < 0.7: # Quiet Market, can be slightly more aggressive
            risk_pct *= 1.1
            
        if ai_brain.get("score", 0) >= 85 and ai_brain.get("regime") == "TRENDING" and mv < 1.2:
            risk_pct *= 1.2 # Bonus risk only in healthy trends
            
        if bot_state["state"] == "FRAGILE":
            elapsed = time.time() - bot_state["last_loss_time"]
            if elapsed < 15 * 60:
                bot_state["last_log"] = f"[yellow]FRAGILE: Cooling off... ({int(15 - elapsed/60)}m left)[/]"
                return
            else:
                risk_pct = ACCOUNT_RISK_PERCENT * 0.5 

        balance = await get_balance_async(client)
        if balance < 10: 
            bot_state["last_log"] = f"[yellow]Balance too low for {symbol} ($ {balance:.2f})[/]"
            return
            
        ticker_res = await binance_request(client, 'GET', '/fapi/v1/ticker/price', {'symbol': symbol})
        if ticker_res is None or ticker_res.status_code != 200: 
            bot_state["last_log"] = f"[red]Price Fetch Fail for {symbol}[/]"
            return
            
        try:
            ticker_data = ticker_res.json()
            if 'price' not in ticker_data:
                bot_state["last_log"] = f"[red]Invalid Price Data for {symbol}[/]"
                return
            price = float(ticker_data['price'])
        except Exception:
            bot_state["last_log"] = f"[red]Price Parse Fail for {symbol}[/]"
            return

        prec = await get_symbol_precision(client, symbol)
        
        calculated_leverage = max(10, min(ai_brain.get("lev_max", MAX_LEVERAGE), 20, int(20 / ai_brain["sl"])))
        
        await binance_request(client, 'POST', '/fapi/v1/leverage', {'symbol': symbol, 'leverage': calculated_leverage})
        
        limit_price = ai_brain.get("limit_price", price)
        # Ensure limit_price follows the tick size (precision) required by Binance
        limit_price = round_step(limit_price, prec["tick"])
        
        risk_amount_usdt = balance * risk_pct
        trade_value_usdt = risk_amount_usdt / (ai_brain["sl"] / 100)
        
        # Calculate quantity based on limit_price
        quantity = round_step(trade_value_usdt / limit_price, prec["step"])
        
        # --- Binance Minimum Notional Check ($5.0) ---
        # CRITICAL: We MUST use limit_price here because that's what Binance uses to validate the order.
        if (quantity * limit_price) < 5.1:
            # Force quantity up to meet minimum $5.2 notional at the limit price
            raw_qty = 5.2 / limit_price
            quantity = math.ceil(raw_qty / prec["step"]) * prec["step"]
            quantity = round(quantity, 8) 
            
        if quantity <= 0: return

        limit_price_str = f"{limit_price:.{prec['p_prec']}f}"
        
        # --- GUARANTEED SINGLE LIMIT ORDER POLICY ---
        # Fetch current open orders from Binance directly to ensure no duplicates
        check_res = await binance_request(client, 'GET', '/fapi/v1/openOrders', {"symbol": symbol})
        if check_res and check_res.status_code == 200:
            existing_orders = check_res.json()
            if isinstance(existing_orders, list) and len(existing_orders) > 0:
                # Cancel ALL existing limit orders for this symbol first
                for old_ord in existing_orders:
                    if old_ord.get("type") == "LIMIT":
                        await binance_request(client, 'DELETE', '/fapi/v1/order', {
                            "symbol": symbol, "orderId": old_ord["orderId"]
                        })
                bot_state["last_log"] = f"[yellow]CLEANED & UPDATING Limit for {symbol}[/]"

        # Now place the new single limit order
        res = await binance_request(client, 'POST', '/fapi/v1/order', {
            "symbol": symbol, "side": side, "type": "LIMIT", 
            "quantity": quantity, "price": limit_price_str, "timeInForce": "GTC"
        })
        
        if res and res.status_code == 200:
            order_data = res.json()
            bot_state["last_log"] = f"[bold cyan]LIMIT {side} {symbol} at {limit_price_str}[/]"
            
            bot_state["limit_orders"][symbol] = {
                "orderId": order_data["orderId"],
                "side": side,
                "price": limit_price,
                "quantity": quantity,
                "sig": signal_type,
                "ai": ai_brain,
                "timestamp": time.time()
            }
        else:
            msg = res.json().get('msg', 'Error') if res else "Timeout"
            bot_state["last_log"] = f"[red]Limit Order Fail {symbol}: {msg}[/]"
            log_error(f"Limit Order Fail {symbol}: {msg}")
    except Exception as e: 
        bot_state["last_log"] = f"[red]Limit Order Err: {str(e)[:20]}[/]"
        log_error(f"Limit Order Exception ({symbol}): {str(e)}")

async def close_position_async(client, symbol, side, amt, reason, pnl_val=0.0):
    try:
        prec = await get_symbol_precision(client, symbol)
        q_close = abs(round_step(amt, prec["step"]))
        await binance_request(client, 'POST', '/fapi/v1/order', {"symbol": symbol, "side": "SELL" if side == "LONG" else "BUY", "type": "MARKET", "quantity": q_close, "reduceOnly": "true"})
        await binance_request(client, 'DELETE', '/fapi/v1/allOpenOrders', {"symbol": symbol})
        await binance_request(client, 'DELETE', '/fapi/v1/algoOpenOrders', {"symbol": symbol})
        
        # --- AI FEEDBACK LOOP (Singularity) ---
        ai_data = bot_state["trades"].get(symbol, {})
        if ai_data:
            # 1. Symbol Performance & Blacklist
            sp = bot_state["sym_perf"].setdefault(symbol, {'w':0, 'l':0, 'c':0})
            if pnl_val > 0:
                sp['w'] += 1
                sp['c'] = 0
            else:
                sp['l'] += 1
                sp['c'] += 1
                if sp['c'] >= 2: # 2 consecutive losses = Blacklist
                    bot_state["blacklist"][symbol] = time.time() + (4 * 3600)
            
            # 2. Strategy Attribution (Context-Aware)
            regime = ai_data.get("regime", "TRENDING")
            for feat in ["liq", "ml", "ob", "div"]:
                if ai_data.get(feat):
                    res_idx = 0 if pnl_val > 0 else 1
                    key = f"{regime}:{feat}"
                    # Initialize if missing (safety)
                    if key not in bot_state["strat_perf"]:
                        bot_state["strat_perf"][key] = [0, 0]
                    bot_state["strat_perf"][key][res_idx] += 1
            
            # 3. Neural Weight Recalculation (EMA-based Adaptation)
            # This allows the bot to "learn" more smoothly which strategies are working
            for key, counts in bot_state["strat_perf"].items():
                w, l = counts[0], counts[1]
                total = w + l
                if total >= 2:
                    wr = w / total
                    current_w = bot_state["neural_weights"].get(key, 1.0)
                    
                    # Target weight based on performance tiers
                    if wr >= 0.75: target = 1.4    # Exceptional performance
                    elif wr >= 0.60: target = 1.2  # Good performance
                    elif wr <= 0.30: target = 0.4  # Very poor performance
                    elif wr <= 0.45: target = 0.7  # Underperforming
                    else: target = 1.0             # Neutral/Stable
                    
                    # Smooth adaptation (Alpha = 0.3)
                    # New Weight = (Old * 0.7) + (Target * 0.3)
                    bot_state["neural_weights"][key] = round((current_w * 0.7) + (target * 0.3), 2)
                    
                    # Reset counts occasionally to allow for fresh learning in new market regimes
                    if total >= 20:
                        bot_state["strat_perf"][key] = [int(w * 0.5), int(l * 0.5)] # Decay memory

        bot_state["last_log"] = f"[bold orange3]EXIT {symbol} | {reason} | PNL: {pnl_val:+.2f}[/]"
        if symbol in bot_state["trades"]: del bot_state["trades"][symbol]
        
        # --- EQ Module Update ---
        bot_state["daily_pnl"] += pnl_val
        if pnl_val < 0:
            bot_state["losses"] += 1
            bot_state["consec_losses"] += 1
            bot_state["last_loss_time"] = time.time()
            if bot_state["consec_losses"] >= 4 and bot_state["state"] not in ["LOCKDOWN", "DONE"]:
                bot_state["state"] = "FRAGILE"
        elif pnl_val > 0:
            bot_state["wins"] += 1
            bot_state["consec_losses"] = 0
            if bot_state["state"] == "FRAGILE":
                bot_state["state"] = "NORMAL"

        if bot_state["start_balance"] > 0:
            pnl_pct = bot_state["daily_pnl"] / bot_state["start_balance"]
            if pnl_pct <= -DAILY_LOSS_LIMIT_PCT:
                bot_state["state"] = "LOCKDOWN"
            elif pnl_pct >= DAILY_PROFIT_TARGET_PCT:
                bot_state["state"] = "DONE"

        save_state_to_db()
        await get_balance_async(client) # Force instant balance update

    except Exception as e:
        bot_state["last_log"] = f"[red]Close Pos Err ({symbol}): {str(e)[:40]}[/]"
        log_error(f"Close Position Exception ({symbol}): {str(e)}")


async def manage_limit_orders(client, all_signals):
    try:
        res = await binance_request(client, 'GET', '/fapi/v1/openOrders')
        if res is None or res.status_code != 200: return
        
        open_orders_data = res.json()
        if not isinstance(open_orders_data, list): return
        
        open_order_symbols = [o['symbol'] for o in open_orders_data if o['type'] == 'LIMIT']
        
        symbols_to_remove = []
        now = time.time()
        for symbol, data in bot_state["limit_orders"].items():
            # Grace period: 10 seconds for API to sync
            if (now - data.get("timestamp", 0)) < 10:
                continue
                
            if symbol not in open_order_symbols:
                symbols_to_remove.append(symbol)
                
        for s in symbols_to_remove:
            # Only remove if it's not about to be processed as a new trade
            # (manage_active_positions will handle the transfer if it's filled)
            if s not in bot_state["trades"]:
                del bot_state["limit_orders"][s]
            
        for symbol, data in list(bot_state["limit_orders"].items()):
            ana = next((s for s in all_signals if (s['sym'] + "USDT") == symbol), None)
            
            if not ana or "WAIT" in ana["sig"]:
                bot_state["last_log"] = f"[yellow]CANCEL {symbol}: Signal Expired[/]"
                await binance_request(client, 'DELETE', '/fapi/v1/order', {"symbol": symbol, "orderId": data["orderId"]})
                if symbol in bot_state["limit_orders"]: del bot_state["limit_orders"][symbol]
                continue
                
            # --- LIMIT CHASING (v9.0 Supreme) ---
            # If momentum trade and very close but not filled, chase it!
            if data.get("ai", {}).get("entry_mode") == "MOMENTUM":
                curr_p = market_data.prices.get(symbol)
                if curr_p:
                    dist_pct = abs(curr_p - data["price"]) / curr_p * 100
                    elapsed = time.time() - data.get("timestamp", 0)
                    
                    # If within 0.15% and has been waiting more than 10 seconds
                    if dist_pct < 0.15 and elapsed > 10:
                        prec = await get_symbol_precision(client, symbol)
                        # Shift limit slightly closer to market price
                        new_price = round_step(curr_p * (1.0002 if data["side"] == "BUY" else 0.9998), prec["tick"])
                        
                        bot_state["last_log"] = f"[cyan]CHASING {symbol}: Shifting Limit to {new_price}[/]"
                        
                        # Cancel old and place new
                        await binance_request(client, 'DELETE', '/fapi/v1/order', {"symbol": symbol, "orderId": data["orderId"]})
                        
                        new_res = await binance_request(client, 'POST', '/fapi/v1/order', {
                            "symbol": symbol, "side": data["side"], "type": "LIMIT", 
                            "quantity": data["quantity"], "price": f"{new_price:.{prec['p_prec']}f}", "timeInForce": "GTC"
                        })
                        
                        if new_res and new_res.status_code == 200:
                            order_data = new_res.json()
                            bot_state["limit_orders"][symbol].update({
                                "orderId": order_data["orderId"],
                                "price": new_price,
                                "timestamp": time.time()
                            })
    except Exception as e:
        log_error(f"Manage Limit Orders Exception: {str(e)}")

async def manage_active_positions(client, all_signals):
    try:
        res = await binance_request(client, 'GET', '/fapi/v2/positionRisk')
        if res is None or res.status_code != 200: return []
        
        pos_data = res.json()
        if not isinstance(pos_data, list): return []
        
        positions = [p for p in pos_data if float(p.get('positionAmt', 0)) != 0]
        
        if not positions:
            if "WS" not in bot_state["last_log"] and bot_state["state"] == "NORMAL":
                bot_state["last_log"] = "[dim]Scanning for signals...[/]"
            bot_state["logged_secure"] = []
            return []
        
        orders_res = await binance_request(client, 'GET', '/fapi/v1/openOrders')
        all_open_orders = []
        if orders_res and orders_res.status_code == 200:
            ord_data = orders_res.json()
            if isinstance(ord_data, list): all_open_orders = ord_data
        
        algo_orders_res = await binance_request(client, 'GET', '/fapi/v1/openAlgoOrders')
        all_algo_orders = []
        if algo_orders_res and algo_orders_res.status_code == 200:
            a_ord_data = algo_orders_res.json()
            if isinstance(a_ord_data, list): all_algo_orders = a_ord_data
        
        active_symbols = []
        for p in positions:
            symbol, amt = p['symbol'], float(p['positionAmt'])
            active_symbols.append(symbol)
            entry_price, mark_price = float(p['entryPrice']), float(p['markPrice'])
            side = "LONG" if amt > 0 else "SHORT"
            pnl_pct = ((mark_price - entry_price) / entry_price) * 100 * (1 if side == "LONG" else -1)
            
            if symbol not in bot_state["trades"]:
                # Use data from limit_orders if available
                lo_data = bot_state["limit_orders"].get(symbol)
                if lo_data and lo_data["side"] == ("BUY" if side == "LONG" else "SELL"):
                    ai = lo_data["ai"]
                    bot_state["trades"][symbol] = {
                        "peak": mark_price, 
                        "tp": ai["tp"], 
                        "sl": ai["sl"], 
                        "ts_act": ai["ts_act"], 
                        "ts_cb": ai["ts_cb"], 
                        "side": side,
                        "entry_time": lo_data.get("timestamp", time.time()),
                        "type": ai["type"],
                        "regime": ai.get("regime", "TRENDING"), # Store regime for context-learning
                        "liq": ai.get("liq"),
                        "ml": ai.get("ml"),
                        "ob": ai.get("ob"),
                        "div": ai.get("div"),
                        "atr_pct": ai.get("atr_pct", 0.5),
                        "trail_struct": ai.get("trail_struct")
                    }
                    if symbol in bot_state["limit_orders"]: del bot_state["limit_orders"][symbol]
                else:
                    bot_state["trades"][symbol] = {
                        "peak": mark_price, 
                        "tp": 2.0, 
                        "sl": 1.0, 
                        "ts_act": 0.8, 
                        "ts_cb": 0.25, 
                        "side": side,
                        "entry_time": time.time(),
                        "type": "INTRA"
                    }
            
            ai = bot_state["trades"][symbol]
            elapsed_min = (time.time() - ai.get("entry_time", time.time())) / 60
            
            # --- BTC DANGER GUARD ---
            if bot_state["btc_state"] == "DANGER" and GLOBAL_BTC_EXIT:
                await close_position_async(client, symbol, side, amt, "BTC-DANGER", pnl_pct)
                continue

            # --- SMART BREAKEVEN & SCALING OUT ---
            if pnl_pct >= (ai["tp"] * 0.5):
                # 1. Scaling Out (Secure 50% Profit)
                if not ai.get("scaled_out", False):
                    prec = await get_symbol_precision(client, symbol)
                    q_partial = round_step(abs(amt) * 0.5, prec["step"])
                    if q_partial > 0:
                        res = await binance_request(client, 'POST', '/fapi/v1/order', {
                            "symbol": symbol, "side": "SELL" if side == "LONG" else "BUY", 
                            "type": "MARKET", "quantity": q_partial, "reduceOnly": "true"
                        })
                        if res and res.status_code == 200:
                            ai["scaled_out"] = True
                            bot_state["last_log"] = f"[bold gold1]SCALED OUT: 50% Secured for {symbol}[/]"

                # 2. Smart Breakeven (Risk-Free)
                if not ai.get("sl_be", False):
                    prec = await get_symbol_precision(client, symbol)
                    be_mult = 1.0005 if side == "LONG" else 0.9995
                    be_price = round_step(entry_price * be_mult, prec["tick"])
                    await binance_request(client, 'DELETE', '/fapi/v1/algoOpenOrders', {"symbol": symbol})
                    sl_res = await binance_request(client, 'POST', '/fapi/v1/algoOrder', {
                        "symbol": symbol, "side": "SELL" if side == "LONG" else "BUY", 
                        "type": "STOP_MARKET", "algoType": "CONDITIONAL", 
                        "triggerPrice": f"{be_price:.{prec['p_prec']}f}",
                        "closePosition": "true", "workingType": "MARK_PRICE"
                    })
                    if sl_res and sl_res.status_code == 200:
                        ai["sl_be"] = True
                        bot_state["last_log"] = f"[bold green]SMART BE: {symbol} is now RISK-FREE[/]"

            # --- MARKET STRUCTURE TRAILING STOP ---
            if ai.get("scaled_out", False) and ai.get("trail_struct"):
                ts_level = ai["trail_struct"]
                if (side == "LONG" and mark_price < ts_level) or (side == "SHORT" and mark_price > ts_level):
                    await close_position_async(client, symbol, side, amt, "TRAIL-STRUCT", pnl_pct)
                    continue

            symbol_orders = [o for o in all_open_orders if o['symbol'] == symbol]
            symbol_algo_orders = [o for o in all_algo_orders if o['symbol'] == symbol]
            
            has_sl = any(o['type'] in ['STOP', 'STOP_MARKET'] and (
                (side == "LONG" and float(o.get('stopPrice', 0)) < entry_price) or 
                (side == "SHORT" and float(o.get('stopPrice', 0)) > entry_price)
            ) for o in symbol_orders) or any(o['orderType'] in ['STOP', 'STOP_MARKET'] and (
                (side == "LONG" and float(o.get('triggerPrice', 0)) < entry_price) or 
                (side == "SHORT" and float(o.get('triggerPrice', 0)) > entry_price)
            ) for o in symbol_algo_orders)
            
            # NOTE: has_tp logic removed for Infinite Runner mode
            
            if not has_sl:
                prec = await get_symbol_precision(client, symbol)
                close_side = "SELL" if side == "LONG" else "BUY"
                sl_mult = (1 - (ai["sl"]/100)) if side == "LONG" else (1 + (ai["sl"]/100))
                sl_price = round_step(entry_price * sl_mult, prec["tick"])
                
                sl_res = await binance_request(client, 'POST', '/fapi/v1/algoOrder', {
                    "symbol": symbol, "side": close_side, "type": "STOP_MARKET", 
                    "algoType": "CONDITIONAL", "triggerPrice": f"{sl_price:.{prec['p_prec']}f}",
                    "closePosition": "true", "workingType": "MARK_PRICE"
                })
                
                if sl_res and sl_res.status_code == 200:
                    bot_state["last_log"] = f"[cyan]Updated SL for {symbol}[/]"
            else:
                if symbol not in bot_state.get("logged_secure", []):
                    bot_state["last_log"] = f"[green]POSITIONS GUARDED for {symbol}[/]"
                    bot_state.setdefault("logged_secure", []).append(symbol)

            if side == "LONG":
                ai["peak"] = max(ai["peak"], mark_price)
                peak_pnl = ((ai["peak"] - entry_price) / entry_price) * 100
            else:
                ai["peak"] = min(ai["peak"], mark_price)
                peak_pnl = ((entry_price - ai["peak"]) / entry_price) * 100

            # --- SNIPER EXIT LOGIC (Real-time) ---
            reason = None
            curr_p = market_data.prices.get(symbol, mark_price)
            d1m = market_data.klines.get(symbol, {}).get("1m")
            
            if d1m is not None and not d1m.empty:
                # 1. Momentum Decay (Volume Exhaustion)
                vol_ma = d1m["v"].rolling(10).mean().iloc[-1]
                curr_vol = d1m["v"].iloc[-1]
                if pnl_pct > (ai["tp"] * 0.7) and curr_vol < vol_ma * 0.35:
                    reason = f"SNIPER-EXIT (Vol Decay) {pnl_pct:.2f}%"

                # 2. Real-time Reversal (Hard Stop on sharp bounce)
                # Loosened from 0.15% to 0.4% to allow for normal market noise
                if not reason and pnl_pct > 0.4:
                    if side == "LONG":
                        if curr_p < mark_price * 0.996: 
                            reason = f"SNIPER-EXIT (Flash Rev) {pnl_pct:.2f}%"
                        elif d1m["c"].iloc[-1] < d1m["l"].iloc[-2] and pnl_pct > 1.0:
                            reason = f"SNIPER-EXIT (MS Shift) {pnl_pct:.2f}%"
                    else:
                        if curr_p > mark_price * 1.004:
                            reason = f"SNIPER-EXIT (Flash Rev) {pnl_pct:.2f}%"
                        elif d1m["c"].iloc[-1] > d1m["h"].iloc[-2] and pnl_pct > 1.0:
                            reason = f"SNIPER-EXIT (MS Shift) {pnl_pct:.2f}%"

            # 3. Dynamic Profit Lock (Omniscient Trailing Stop)
            if not reason:
                trail_dist = max(0.5, ai.get("atr_pct", 0.5) * 2.0)
                
                if peak_pnl >= (ai["tp"] * 0.5) and (peak_pnl - pnl_pct) >= trail_dist:
                    reason = f"OMNI-TRAIL (ATR) {pnl_pct:.2f}%"
                elif peak_pnl >= ai["ts_act"] and (peak_pnl - pnl_pct) >= ai["ts_cb"]:
                    reason = f"AI-TRAIL {pnl_pct:.2f}%"
                elif elapsed_min > 60: # Increased from 20m to 60m
                    reason = f"TIME-OUT ({int(elapsed_min)}m)"
                elif EXIT_ON_REVERSAL:
                    ana = next((s for s in all_signals if s['sym'] + "USDT" == symbol), None)
                    if ana:
                        if (side == "LONG" and ana["dir"] == -1) or (side == "SHORT" and ana["dir"] == 1):
                            reason = f"REVERSAL ({ana['sig']})"
            
            if reason: 
                # --- PnL ACCURACY FIX (Fee & Slippage Estimation) ---
                # Estimated round-trip fee + slippage = 0.1% of position size
                raw_pnl = float(p.get('unRealizedProfit', 0.0))
                position_value = abs(amt) * entry_price
                estimated_cost = position_value * 0.001 # 0.1%
                pnl_val = raw_pnl - estimated_cost
                
                await close_position_async(client, symbol, side, amt, reason, pnl_val)

        bot_state["trades"] = {k: v for k, v in bot_state["trades"].items() if k in active_symbols}
        return positions
    except Exception as e: 
        err_msg = str(e)[:40]
        bot_state["last_log"] = f"[red]Manage Pos Err: {err_msg}[/]"
        log_error(f"Manage Positions Exception: {err_msg}")
        return []
