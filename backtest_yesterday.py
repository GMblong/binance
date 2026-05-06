import asyncio
import httpx
import pandas as pd
import numpy as np
from datetime import datetime
import time
from rich.console import Console
from rich.table import Table

from strategies.analyzer import MarketAnalyzer
from engine.ml_engine import ml_predictor
from utils.config import API_URL

console = Console()

async def get_top_movers(client, limit=10):
    res = await client.get(f"{API_URL}/fapi/v1/ticker/24hr")
    data = res.json()
    filtered = [t for t in data if t["symbol"].endswith("USDT") and float(t["quoteVolume"]) > 10_000_000]
    for t in filtered:
        t['cp'] = float(t['priceChangePercent'])
    movers = sorted(filtered, key=lambda x: abs(x['cp']), reverse=True)[:limit]
    return [m["symbol"] for m in movers]

async def fetch_klines(client, symbol, interval, limit=1500, end_time=None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time:
        params["endTime"] = end_time
    res = await client.get(f"{API_URL}/fapi/v1/klines", params=params)
    df = pd.DataFrame(res.json()).iloc[:, [0, 1, 2, 3, 4, 5]]
    df.columns = ["ot", "o", "h", "l", "c", "v"]
    return df.astype(float)

async def run_single_iteration(client, symbols, reference_time, iteration):
    starting_balance = 100.0
    balance = starting_balance
    risk_percent = 0.02 # 2% risk as per live config
    MAX_POSITIONS = 3
    
    console.print(f"\n[bold magenta]=== RUN {iteration} (QUANTUM SIMULATION) ===[/]")
    
    # 1. PRE-FETCH DATA & TRAIN AI
    all_data = {}
    end_t = int((reference_time - 86400) * 1000)
    
    for symbol in symbols:
        # Train only once per session to save time
        if symbol not in ml_predictor.models:
            console.print(f"   [dim]Training {symbol} AI...[/dim]")
            await ml_predictor.train_model(client, symbol, end_time=end_t)
        else:
            console.print(f"   [dim]Using pre-trained AI for {symbol}...[/dim]")
            
        # Use ML engine's fetcher to get OI and Funding Rate
        d15m_all = await ml_predictor.fetch_historical_data(client, symbol, interval="15m", limit=1500, end_time=end_t)
        
        d1m = await fetch_klines(client, symbol, "1m", limit=1500, end_time=end_t)
        # Extract 200 candles before end_t for the 15m context
        d15m = d15m_all.tail(200)
        d1h = await fetch_klines(client, symbol, "1h", limit=100, end_time=end_t)
        
        if not d1m.empty and not d15m.empty and not d1h.empty:
            all_data[symbol] = {
                "1m": d1m, "15m": d15m, "1h": d1h,
                "atr": MarketAnalyzer.get_atr(d1m, 14),
                "full_15m": d15m_all # Keep full for feature engineering
            }

    active_positions = []
    pending_orders = [] # Move initialization here
    trade_history = []
    sim_length = 1440
    
    console.print(f"   [bold yellow]Simulating Global Timeline for 10 symbols...[/bold yellow]")
    
    for i in range(sim_length):
        # 1. SIMULATE PENDING LIMIT ORDERS FILL ---
        for order in pending_orders[:]:
            symbol = order['symbol']
            actual_idx = len(all_data[symbol]["1m"]) - sim_length + i
            curr_row = all_data[symbol]["1m"].iloc[actual_idx]
            
            filled = False
            if order['side'] == 'BUY' and curr_row['l'] <= order['price']: filled = True
            elif order['side'] == 'SELL' and curr_row['h'] >= order['price']: filled = True
            
            if filled:
                active_positions.append({
                    'symbol': symbol,
                    'side': 'LONG' if order['side'] == 'BUY' else 'SHORT',
                    'entry': order['price'], 'size': order['size'],
                    'sl': order['sl'], 'tp': order['tp'],
                    'ts_act': order['ts_act'], 'ts_cb': order['ts_cb'],
                    'peak': order['price']
                })
                pending_orders.remove(order)
            elif i - order['created_at'] > 60:
                pending_orders.remove(order)

        # 2. CHECK EXITS FOR ALL ACTIVE POSITIONS
        for pos in active_positions[:]:
            symbol = pos['symbol']
            actual_idx = len(all_data[symbol]["1m"]) - sim_length + i
            curr_row = all_data[symbol]["1m"].iloc[actual_idx]
            curr_price = curr_row['c']
            
            exit_price = None
            reason = ""
            
            if pos['side'] == 'LONG':
                pos['peak'] = max(pos['peak'], curr_row['h'])
                if curr_row['l'] <= pos['sl']: exit_price, reason = pos['sl'], "Stop Loss/BE"
                elif curr_row['h'] >= pos['tp']: exit_price, reason = pos['tp'], "Take Profit"
                else:
                    peak_pnl = (pos['peak'] - pos['entry']) / pos['entry'] * 100
                    curr_pnl = (curr_price - pos['entry']) / pos['entry'] * 100
                    if peak_pnl > pos['ts_act'] and (peak_pnl - curr_pnl) > pos['ts_cb']:
                        exit_price, reason = curr_price, "AI Trailing Stop"
            else:
                pos['peak'] = min(pos['peak'], curr_row['l'])
                if curr_row['h'] >= pos['sl']: exit_price, reason = pos['sl'], "Stop Loss/BE"
                elif curr_row['l'] <= pos['tp']: exit_price, reason = pos['tp'], "Take Profit"
                else:
                    peak_pnl = (pos['entry'] - pos['peak']) / pos['entry'] * 100
                    curr_pnl = (pos['entry'] - curr_price) / pos['entry'] * 100
                    if peak_pnl > pos['ts_act'] and (peak_pnl - curr_pnl) > pos['ts_cb']:
                        exit_price, reason = curr_price, "AI Trailing Stop"
                        
            if exit_price:
                pnl_pct = (exit_price - pos['entry']) / pos['entry'] if pos['side'] == 'LONG' else (pos['entry'] - exit_price) / pos['entry']
                usd_pnl = pnl_pct * pos['size'] - (pos['size'] * 0.0008)
                balance += usd_pnl
                trade_history.append({"symbol": symbol, "pnl": usd_pnl, "reason": reason})
                active_positions.remove(pos)

        # 3. GENERATE NEW SIGNALS FOR ALL SYMBOLS
        if i % 15 == 0 and len(active_positions) < MAX_POSITIONS:
            signals = []
            for symbol, data in all_data.items():
                if any(p['symbol'] == symbol for p in active_positions): continue
                
                actual_idx = len(data["1m"]) - sim_length + i
                curr_price = data["1m"]["c"].iloc[actual_idx]
                
                d15_idx = int(actual_idx / 15)
                if d15_idx >= len(data["15m"]): d15_idx = len(data["15m"]) - 1
                d1h_idx = int(actual_idx / 60)
                if d1h_idx >= len(data["1h"]): d1h_idx = len(data["1h"]) - 1
                
                sub_15m = data["15m"].iloc[:d15_idx+1]
                sub_1m = data["1m"].iloc[:actual_idx+1]
                sub_1h = data["1h"].iloc[:d1h_idx+1]
                
                if len(sub_15m) < 30 or len(sub_1h) < 10: continue
                
                ema20_1h = MarketAnalyzer.get_ema(sub_1h['c'], 20).iloc[-1]
                htf_dir = 1 if sub_1h['c'].iloc[-1] > ema20_1h else -1
                ema9_15m = MarketAnalyzer.get_ema(sub_15m['c'], 9).iloc[-1]
                ema21_15m = MarketAnalyzer.get_ema(sub_15m['c'], 21).iloc[-1]
                direction = 1 if ema9_15m > ema21_15m else -1
                
                score = MarketAnalyzer.calculate_score(sub_1m, sub_15m, direction, 1.0, 0.0)
                if direction != htf_dir: score -= 40
                
                ml_prob = 0.5
                if ml_predictor.models.get(symbol):
                    try:
                        # Get historical 15m data up to this point from full_15m (which has OI/Funding)
                        curr_ts = sub_15m['ot'].iloc[-1]
                        feats = data["full_15m"][data["full_15m"]['ot'] <= curr_ts].copy()
                        feats = ml_predictor.feature_engineering(feats)
                        model = ml_predictor.models[symbol]
                        f_cols = model.feature_name_
                        X = feats[f_cols].iloc[[-1]]
                        ml_prob = model.predict_proba(X)[0][1]
                    except Exception as e:
                        pass
                
                if direction == 1 and ml_prob > 0.6: score += int((ml_prob - 0.5) * 40)
                elif direction == -1 and ml_prob < 0.4: score += int((0.5 - ml_prob) * 40)
                score = min(max(score, 0), 100)
                
                if score >= 75:
                    # Breakout detection for Aggressive Entry
                    prev_5_high = sub_1m['h'].iloc[-6:-1].max()
                    prev_5_low = sub_1m['l'].iloc[-6:-1].min()
                    is_breakout = (direction == 1 and curr_price > prev_5_high) or (direction == -1 and curr_price < prev_5_low)
                    
                    fvg = MarketAnalyzer.get_nearest_fvg(sub_1m)
                    ob = MarketAnalyzer.find_nearest_order_block(sub_1m, curr_price, direction)
                    ema9_1m = MarketAnalyzer.get_ema(sub_1m['c'], 9).iloc[-1]
                    
                    limit_p = curr_price
                    if direction == 1:
                        if ob and ob["type"] == "BULLISH": limit_p = ob["top"] * 1.0005 # Front-run offset
                        elif fvg and fvg["type"] == "BULLISH": limit_p = fvg["top"] * 1.0005
                        else: limit_p = (curr_price + ema9_1m) / 2
                    else:
                        if ob and ob["type"] == "BEARISH": limit_p = ob["bottom"] * 0.9995 # Front-run offset
                        elif fvg and fvg["type"] == "BEARISH": limit_p = fvg["bottom"] * 0.9995
                        else: limit_p = (curr_price + ema9_1m) / 2
                    
                    dist = abs(curr_price - limit_p) / curr_price * 100
                    
                    # QUANTUM LOGIC:
                    # 1. Mode Agresif: Score >= 85
                    # 2. Mode Sabar: Score 75-84 (Limit Order up to 0.3% away)
                    use_market = (score >= 85) or dist <= 0.1
                    
                    if use_market or dist <= 0.3:
                        atr = data["atr"].iloc[actual_idx]
                        sl_pct = max(0.7, min(3.0, (atr * 2 / curr_price * 100)))
                        
                        # Adaptive RR: Use conservative 1.5x unless EXTREMELY sure
                        rr_mult = 2.0 if ml_prob > 0.9 or ml_prob < 0.1 else 1.5
                        tp_pct = sl_pct * rr_mult
                        
                        risk_mult = 1.2 if (ml_prob > 0.75 or ml_prob < 0.25) else (1.0 if 0.4<=ml_prob<=0.6 else 0.5)
                        trade_size_usd = (balance * risk_percent * risk_mult) / (sl_pct / 100)
                        
                        if use_market: # Immediate Fill
                            active_positions.append({
                                'symbol': symbol, 'side': 'LONG' if direction == 1 else 'SHORT',
                                'entry': curr_price, 'size': trade_size_usd,
                                'sl': curr_price * (1 - sl_pct/100) if direction == 1 else curr_price * (1 + sl_pct/100),
                                'tp': curr_price * (1 + tp_pct/100) if direction == 1 else curr_price * (1 - tp_pct/100),
                                'ts_act': sl_pct * 0.8, 'ts_cb': sl_pct * 0.3, 'peak': curr_price
                            })
                        else: # Pending Limit
                            pending_orders.append({
                                'symbol': symbol, 'side': 'BUY' if direction == 1 else 'SELL',
                                'price': limit_p, 'size': trade_size_usd,
                                'sl': limit_p * (1 - sl_pct/100) if direction == 1 else limit_p * (1 + sl_pct/100),
                                'tp': limit_p * (1 + tp_pct/100) if direction == 1 else limit_p * (1 - tp_pct/100),
                                'ts_act': sl_pct * 0.8, 'ts_cb': sl_pct * 0.3,
                                'created_at': i
                            })

    # CLEANUP END OF DAY
    for pos in active_positions:
        symbol = pos['symbol']
        curr_price = all_data[symbol]["1m"]["c"].iloc[-1]
        pnl_pct = (curr_price - pos['entry']) / pos['entry'] if pos['side'] == 'LONG' else (pos['entry'] - curr_price) / pos['entry']
        usd_pnl = pnl_pct * pos['size'] - (pos['size'] * 0.0008)
        balance += usd_pnl
        trade_history.append({"symbol": symbol, "pnl": usd_pnl, "reason": "End of Day Close"})

    win = len([t for t in trade_history if t['pnl'] > 0])
    loss = len([t for t in trade_history if t['pnl'] <= 0])
    net_profit = balance - 100
    console.print(f"   [bold cyan]Stats: {win} Wins / {loss} Losses | Profit: ${net_profit:+.2f}[/]")
    return net_profit

async def run_multiple_backtests(iterations=2):
    console.print(f"[bold green]=== Backtest Simulator (PORTFOLIO: 5 COINS, 3 SLOTS) ===[/]")
    all_results = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Use a fixed set of high-volume symbols for consistent comparison
        symbols = ['LABUSDT', 'IOUSDT', 'ZECUSDT', 'TONUSDT', 'JTOUSDT']
        console.print(f"Simulating Portfolio for: {symbols}")
        reference_time = int(time.time() / 3600) * 3600
        for it in range(1, iterations + 1):
            net = await run_single_iteration(client, symbols, reference_time, it)
            all_results.append(net)
    avg_net = sum(all_results) / len(all_results)
    console.print(f"\n[bold green]AVERAGE PORTFOLIO PROFIT: ${avg_net:+.2f}[/]")

if __name__ == "__main__":
    asyncio.run(run_multiple_backtests(2))