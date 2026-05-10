import time
import math
import asyncio
import httpx
from typing import Optional
from utils.config import ACCOUNT_RISK_PERCENT, MAX_LEVERAGE, EXIT_ON_REVERSAL, GLOBAL_BTC_EXIT, DAILY_LOSS_LIMIT_PCT, DAILY_PROFIT_TARGET_PCT, MIN_NOTIONAL_USD
from utils.state import bot_state, market_data
from utils.helpers import round_step
from engine.api import binance_request, get_symbol_precision
from utils.logger import log_error
from utils.telegram import alert_open_position, alert_close_position, alert_partial_close, alert_limit_filled
from utils.intelligence import calculate_kelly_risk, update_feature_weights
from engine.ml_engine import ml_predictor
from strategies.analyzer import MarketAnalyzer

async def open_position_async(client: httpx.AsyncClient, symbol: str, side: str, signal_type: str, ai_brain: dict) -> bool:
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

        # --- CONFIDENCE DECAY: reduce size on recent losing streak ---
        recent = perf[-5:] if len(perf) >= 3 else []
        if recent:
            recent_wr = sum(recent) / len(recent)
            if recent_wr <= 0.2:
                risk_amt *= 0.3
            elif recent_wr <= 0.4:
                risk_amt *= 0.6

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
        if notional < MIN_NOTIONAL_USD:
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
            except Exception as e:
                log_error(f"Leverage set failed for {symbol}: {e}", include_traceback=False)
            
        if quantity <= 0: return False
        
        # Also attempt to set margin type to ISOLATED to avoid cross margin issues
        try:
            await binance_request(client, 'POST', '/fapi/v1/marginType', {"symbol": symbol, "marginType": "ISOLATED"})
        except Exception:
            pass  # Expected to fail if already ISOLATED

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

                    sl_res = await binance_request(client, 'POST', '/fapi/v1/algoOrder', {
                        "algoType": "CONDITIONAL",
                        "symbol": symbol, "side": "SELL" if side == "BUY" else "BUY", 
                        "type": "STOP_MARKET", "triggerPrice": sl_price_str, 
                        "closePosition": "true", "workingType": "MARK_PRICE"
                    })

                    tp_res = await binance_request(client, 'POST', '/fapi/v1/algoOrder', {
                        "algoType": "CONDITIONAL",
                        "symbol": symbol, "side": "SELL" if side == "BUY" else "BUY", 
                        "type": "TAKE_PROFIT_MARKET", "triggerPrice": tp_price_str, 
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
                    "brain_signals": list(ai_brain.get("brain_signals", [])),
                }
            asyncio.create_task(alert_open_position(
                symbol, side, quantity,
                execution_price if is_market else limit_price,
                bot_state.get("_lev_cache", {}).get(symbol, MAX_LEVERAGE),
                f"{'MARKET' if is_market else 'LIMIT'} | {signal_type}",
                sl=f"{ai_brain.get('sl', 1.0):.2f}%",
                tp=f"{ai_brain.get('tp', 2.0):.2f}%",
            ))
            return True
        else:
            msg = res.json().get("msg", "Unknown") if res else "No Response"
            bot_state["last_log"] = f"[red]Limit Order Fail {symbol}: {msg}[/]"
            log_error(f"Limit Order Fail {symbol}: {msg}")
            return False
            
    except Exception as e:
        log_error(f"Limit Order Exception ({symbol}): {str(e)}")
        return False

async def close_position_async(client: httpx.AsyncClient, symbol: str, side: str, amount: float, reason: str, pnl: float = 0.0) -> bool:
    try:
        prec = await get_symbol_precision(client, symbol)
        close_side = "SELL" if side == "LONG" else "BUY"
        
        # 1. Cancel all open orders (regular + algo) for this symbol
        await binance_request(client, 'DELETE', '/fapi/v1/allOpenOrders', {"symbol": symbol})
        # Cancel algo orders (SL/TP) - must cancel individually by algoId
        algo_res = await binance_request(client, 'GET', '/fapi/v1/openAlgoOrders', {"symbol": symbol})
        if algo_res and algo_res.status_code == 200:
            algo_data = algo_res.json()
            algo_list = algo_data.get('orders', algo_data) if isinstance(algo_data, dict) else algo_data
            for ao in (algo_list if isinstance(algo_list, list) else []):
                algo_id = ao.get('algoId')
                if algo_id:
                    await binance_request(client, 'DELETE', '/fapi/v1/algoOrder', {"symbol": symbol, "algoId": str(algo_id)})
        
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

            # Scalping Brain accuracy feedback
            from engine.scalping_brain import scalping_brain
            brain_signals = trade_meta.get("brain_signals") or []
            if brain_signals:
                scalping_brain.update_accuracy(symbol, brain_signals, is_win)

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
            
            asyncio.create_task(alert_close_position(symbol, side, pnl, pnl, reason))
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
            # Cancel stale limit orders after 90 seconds (scalping needs fast entry)
            if age > 90:
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
            asyncio.create_task(alert_partial_close(symbol, pct, pnl, reason))
            return True
        return False
    except Exception as e:
        log_error(f"Partial close failed for {symbol}: {e}")
        return False

async def check_and_execute_exits(client, symbol, current_price, all_signals=[]):
    """
    ML-Adaptive Exit Engine - optimized for sub-second response.
    """
    global last_exit_check
    now = time.time()
    
    # Reduced throttle: 0.2s instead of 1.0s for faster exits
    if now - last_exit_check.get(symbol, 0) < 0.2:
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
        
        # === 1. PARTIAL TAKE PROFIT (40% at 0.6R for scalping) ===
        if not ai.get("partial_done") and pnl_pct >= max(initial_sl * 0.6, 0.3):
            did_partial = await partial_close_async(client, symbol, side, amt, 0.4, "TP1-SCALP", pnl_pct)
            if did_partial:
                ai["partial_done"] = True
                ai["be_active"] = True
                return False
        
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
            # If ML says momentum weakening, tighten trail aggressively
            if side == "LONG" and ml_prob < 0.45:
                ml_adj = 0.4  # Very tight - protect profit
            elif side == "SHORT" and ml_prob > 0.55:
                ml_adj = 0.4
            # If ML says strong continuation, widen trail slightly
            elif (side == "LONG" and ml_prob > 0.7) or (side == "SHORT" and ml_prob < 0.3):
                ml_adj = 1.3
        
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
        # Account for slippage+fees (~0.15%) when deciding breakeven
        slippage_buffer = 0.15
        if not reason and ai.get("be_active"):
            be_buffer = max(0.1, initial_sl * 0.15)  # Tight: protect at 0.1% above entry
            if pnl_pct <= be_buffer:
                reason = "SMART-BE (Protecting profit)"
        elif not reason and peak_pnl >= initial_sl * 0.7 and pnl_pct <= slippage_buffer:
            reason = "SMART-BE (Profit evaporated)"
        
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
        
        # --- RECONCILIATION: Detect positions closed by exchange (TP/SL hit) ---
        # Compare bot_state["trades"] with actual Binance positions
        active_symbols = {p['symbol'] for p in positions}
        orphaned = [s for s in list(bot_state.get("trades", {}).keys()) if s not in active_symbols]
        
        for symbol in orphaned:
            # Skip if recently closed by bot itself (avoid double-counting)
            recently_closed = bot_state.get("_recently_closed", {})
            if recently_closed.get(symbol, 0) > time.time() - 10:
                continue
            
            trade_meta = bot_state["trades"][symbol]
            entry_time = trade_meta.get("entry_time", 0)
            side = trade_meta.get("side", "LONG")
            
            # Estimate PnL from last known mark price
            last_price = market_data.prices.get(symbol, 0)
            entry_price = trade_meta.get("peak", last_price)  # peak is set to entry on open for market orders
            
            # Try to get actual fill from recent trades API
            pnl_pct = 0.0
            try:
                trades_res = await binance_request(client, 'GET', '/fapi/v1/userTrades', 
                    {"symbol": symbol, "limit": 5})
                if trades_res and trades_res.status_code == 200:
                    user_trades = trades_res.json()
                    # Find the closing trade (reduceOnly or opposite side)
                    close_trades = [t for t in user_trades if float(t.get('realizedPnl', 0)) != 0]
                    if close_trades:
                        # Sum realized PnL from recent closing trades
                        realized = sum(float(t['realizedPnl']) for t in close_trades 
                                      if float(t['time'])/1000 > entry_time)
                        if realized != 0:
                            # Calculate PnL% from realized USDT
                            qty = abs(float(close_trades[-1].get('qty', 1)))
                            ep = float(close_trades[-1].get('price', last_price))
                            notional = qty * ep
                            pnl_pct = (realized / notional) * 100 if notional > 0 else 0
            except Exception:
                pass
            
            is_win = pnl_pct > 0
            
            # Update ML performance
            ml_predictor.update_performance(symbol, is_win)
            
            # Feedback loop: features + brain signals
            active_feats = trade_meta.get("active_features") or []
            if active_feats:
                update_feature_weights(active_feats, is_win)
            
            from engine.scalping_brain import scalping_brain
            brain_signals = trade_meta.get("brain_signals") or []
            if brain_signals:
                scalping_brain.update_accuracy(symbol, brain_signals, is_win)
            
            # Update win/loss counters
            if is_win:
                bot_state["wins"] = bot_state.get("wins", 0) + 1
            else:
                bot_state["losses"] = bot_state.get("losses", 0) + 1
            
            if symbol not in bot_state.get("sym_perf", {}):
                bot_state.setdefault("sym_perf", {})[symbol] = {'w': 0, 'l': 0, 'c': 0}
            if is_win:
                bot_state["sym_perf"][symbol]['w'] += 1
                bot_state["sym_perf"][symbol]['c'] = 0
            else:
                bot_state["sym_perf"][symbol]['l'] += 1
                bot_state["sym_perf"][symbol]['c'] = bot_state["sym_perf"][symbol].get('c', 0) + 1
                bot_state["sym_perf"][symbol]['last_loss_time'] = time.time()
            
            reason = "EXCHANGE-TP/SL"
            col = "green" if is_win else "red"
            bot_state["last_log"] = f"[bold {col}]DETECTED CLOSE {symbol} ({reason}) | PnL: {pnl_pct:+.2f}%[/]"
            asyncio.create_task(alert_close_position(symbol, side, pnl_pct, pnl_pct, reason))
            
            # Cleanup
            del bot_state["trades"][symbol]
            bot_state.setdefault("_recently_closed", {})[symbol] = time.time()
            log_error(f"Reconciled orphan position {symbol}: closed by exchange (PnL: {pnl_pct:+.2f}%)", include_traceback=False)
        
        if not positions:
            bot_state["logged_secure"] = []
            return []
        
        # Batch fetch ALL open orders once (instead of per-symbol)
        all_orders_res = await binance_request(client, 'GET', '/fapi/v1/openOrders')
        all_open_orders = all_orders_res.json() if all_orders_res and all_orders_res.status_code == 200 else []
        # Also fetch algo orders (SL/TP are now placed via algo API)
        algo_orders_res = await binance_request(client, 'GET', '/fapi/v1/openAlgoOrders')
        all_algo_orders = []
        if algo_orders_res and algo_orders_res.status_code == 200:
            algo_data = algo_orders_res.json()
            all_algo_orders = algo_data.get('orders', algo_data) if isinstance(algo_data, dict) else algo_data
        # Index by symbol for O(1) lookup
        orders_by_symbol = {}
        for o in all_open_orders:
            orders_by_symbol.setdefault(o['symbol'], []).append(o)
        # Index algo orders by symbol
        algo_by_symbol = {}
        for o in (all_algo_orders if isinstance(all_algo_orders, list) else []):
            algo_by_symbol.setdefault(o.get('symbol', ''), []).append(o)
        
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
                        "brain_signals": list(ai.get("brain_signals", [])),
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

            # 2. ORDER PROTECTION - use pre-fetched orders (no extra API call)
            symbol_orders = orders_by_symbol.get(symbol, [])
            symbol_algo = algo_by_symbol.get(symbol, [])
            has_sl = (any(o['type'] in ['STOP_MARKET', 'STOP'] for o in symbol_orders) or
                      any(o.get('orderType', o.get('type', '')) in ['STOP_MARKET', 'STOP'] for o in symbol_algo))
            has_tp = (any(o['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT'] for o in symbol_orders) or
                      any(o.get('orderType', o.get('type', '')) in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT'] for o in symbol_algo))

            # --- STRUCTURAL SL PROTECTION ---
            k1m = market_data.klines.get(symbol, {}).get("1m")
            struct_sl = None
            if k1m is not None and len(k1m) >= 20:
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
                        sl_price = round_step(struct_sl, prec["tick"])
                    else:
                        sl_mult = (1 - (ai.get("sl", 1.0)/100)) if side == "LONG" else (1 + (ai.get("sl", 1.0)/100))
                        sl_price = round_step(entry_price * sl_mult, prec["tick"])
                    
                    if has_sl:
                        sl_order = next(o for o in symbol_orders if o['type'] in ['STOP_MARKET', 'STOP'])
                        await binance_request(client, 'DELETE', '/fapi/v1/order', {"symbol": symbol, "orderId": sl_order["orderId"]})
                    
                    await binance_request(client, 'POST', '/fapi/v1/algoOrder', {
                        "algoType": "CONDITIONAL",
                        "symbol": symbol, "side": "SELL" if side == "LONG" else "BUY", 
                        "type": "STOP_MARKET", "triggerPrice": f"{sl_price:.{prec['p_prec']}f}",
                        "closePosition": "true", "workingType": "MARK_PRICE"
                    })

                if not has_tp:
                    tp_mult = (1 + (ai.get("tp", 2.0)/100)) if side == "LONG" else (1 - (ai.get("tp", 2.0)/100))
                    tp_price = round_step(entry_price * tp_mult, prec["tick"])
                    await binance_request(client, 'POST', '/fapi/v1/algoOrder', {
                        "algoType": "CONDITIONAL",
                        "symbol": symbol, "side": "SELL" if side == "LONG" else "BUY",
                        "type": "TAKE_PROFIT_MARKET", "triggerPrice": f"{tp_price:.{prec['p_prec']}f}",
                        "closePosition": "true", "workingType": "MARK_PRICE"
                    })

            # 3. EXIT LOGIC (Fast Exit Engine)
            all_valid = all_signals
            exit_triggered = await check_and_execute_exits(client, symbol, mark_price, all_valid)
            
            if not exit_triggered:
                if bot_state["btc_state"] == "DANGER" and GLOBAL_BTC_EXIT:
                    await close_position_async(client, symbol, side, amt, "BTC-DANGER", pnl_pct)

        return positions
    except Exception as e:
        log_error(f"Manage Positions Exception: {str(e)}")
        return []
