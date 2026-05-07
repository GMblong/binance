import asyncio
import httpx
import pandas as pd
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table

from strategies.analyzer import MarketAnalyzer
from engine.ml_engine import ml_predictor
from utils.config import API_URL
from utils.state import bot_state
from utils.database import init_db, load_state_from_db
from datetime import time as dt_time

console = Console()
init_db()
load_state_from_db()

def get_session_from_ts(ts):
    now = datetime.utcfromtimestamp(ts / 1000).time()
    if dt_time(13, 0) <= now <= dt_time(22, 0): return "NEW_YORK"
    if dt_time(8, 0) <= now <= dt_time(17, 0): return "LONDON"
    if dt_time(0, 0) <= now <= dt_time(9, 0): return "ASIA"
    return "QUIET"

def detect_lead_lag_sim(sub_sym, sub_lead):
    if len(sub_sym) < 5 or len(sub_lead) < 5: return 0
    sym_ret = (sub_sym['c'].iloc[-1] - sub_sym['c'].iloc[-5]) / sub_sym['c'].iloc[-5] * 100
    lead_ret = (sub_lead['c'].iloc[-1] - sub_lead['c'].iloc[-5]) / sub_lead['c'].iloc[-5] * 100
    if lead_ret > 0.3 and sym_ret < 0.1: return 1
    if lead_ret < -0.3 and sym_ret > -0.1: return -1
    return 0

async def get_top_movers(client, limit=5):
    res = await client.get(f"{API_URL}/fapi/v1/ticker/24hr")
    data = res.json()
    filtered = [t for t in data if t["symbol"].endswith("USDT") and float(t["quoteVolume"]) > 10_000_000]
    for t in filtered:
        t['cp'] = float(t['priceChangePercent'])
    # Top gainers/losers by volatility
    movers = sorted(filtered, key=lambda x: abs(x['cp']), reverse=True)[:limit]
    return [m["symbol"] for m in movers]

async def fetch_klines(client, symbol, interval, limit=1500):
    res = await client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": limit})
    df = pd.DataFrame(res.json()).iloc[:, [0, 1, 2, 3, 4, 5]]
    df.columns = ["ot", "o", "h", "l", "c", "v"]
    return df.astype(float)

async def run_backtest():
    starting_balance = 100.0
    balance = starting_balance
    risk_percent = 0.02 # Updated to 2%
    
    console.print(f"[bold green]=== Backtest Simulator (Last 24 Hours) ===[/]")
    console.print(f"Starting Balance: ${balance:.2f}\n")
    console.print(f"Using Neural Weights: {bot_state.get('neural_weights')}\n")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        symbols = await get_top_movers(client, limit=5)
        console.print(f"Selected Top Movers for Simulation: {symbols}\n")
        
        # Pre-fetch BTC data for lead-lag simulation
        df_btc_1m = await fetch_klines(client, "BTCUSDT", "1m", limit=1500)
        
        for symbol in symbols:
            console.print(f"[bold cyan]=> Analyzing {symbol}...[/]")
            
            # 1. Train AI
            console.print("   [dim]Training LightGBM AI Model with historical data...[/dim]")
            await ml_predictor.train_model(client, symbol)
            
            # 2. Fetch Data
            df_1m = await fetch_klines(client, symbol, "1m", limit=1500)
            df_15m = await fetch_klines(client, symbol, "15m", limit=200)
            if df_1m.empty or df_15m.empty: continue
            
            # Ensure we simulate exactly 1 day (1440 minutes)
            sim_data = df_1m.iloc[-1440:].copy().reset_index(drop=True)
            
            positions = []
            pending_orders = []
            trade_history = []
            
            atr_1m_all = MarketAnalyzer.get_atr(df_1m, 14)
            
            console.print(f"   [dim]Simulating 1440 minutes (Sniper Mode - Limit Fills)...[/dim]")
            
            for i in range(1440):
                actual_idx = len(df_1m) - 1440 + i
                curr_row = df_1m.iloc[actual_idx]
                curr_price = curr_row['c']
                curr_high = curr_row['h']
                curr_low = curr_row['l']
                curr_ts = curr_row['ot']
                
                # A. Check Pending Orders (Limit Fills)
                for order in pending_orders[:]:
                    filled = False
                    if order['side'] == 'LONG' and curr_low <= order['price']:
                        filled = True
                    elif order['side'] == 'SHORT' and curr_high >= order['price']:
                        filled = True
                    
                    if filled:
                        positions.append({
                            'side': order['side'],
                            'entry': order['price'],
                            'size': order['size'],
                            'sl': order['sl'],
                            'tp': order['tp'],
                            'ts_act': order['ts_act'],
                            'ts_cb': order['ts_cb'],
                            'peak': order['price']
                        })
                        pending_orders.remove(order)
                    elif i - order['ts'] > 15: # Cancel limit order after 15 mins if not filled
                        pending_orders.remove(order)

                # B. Check Exits (Active Positions)
                for pos in positions[:]:
                    exit_price = None
                    reason = ""
                    
                    if pos['side'] == 'LONG':
                        pos['peak'] = max(pos['peak'], curr_high)
                        if curr_low <= pos['sl']: exit_price, reason = pos['sl'], "Stop Loss"
                        elif curr_high >= pos['tp']: exit_price, reason = pos['tp'], "Take Profit"
                        else:
                            peak_pnl = (pos['peak'] - pos['entry']) / pos['entry'] * 100
                            curr_pnl = (curr_price - pos['entry']) / pos['entry'] * 100
                            if peak_pnl > pos['ts_act'] and (peak_pnl - curr_pnl) > pos['ts_cb']:
                                exit_price, reason = curr_price, "AI Trailing Stop"
                    else:
                        pos['peak'] = min(pos['peak'], curr_low)
                        if curr_high >= pos['sl']: exit_price, reason = pos['sl'], "Stop Loss"
                        elif curr_low <= pos['tp']: exit_price, reason = pos['tp'], "Take Profit"
                        else:
                            peak_pnl = (pos['entry'] - pos['peak']) / pos['entry'] * 100
                            curr_pnl = (pos['entry'] - curr_price) / pos['entry'] * 100
                            if peak_pnl > pos['ts_act'] and (peak_pnl - curr_pnl) > pos['ts_cb']:
                                exit_price, reason = curr_price, "AI Trailing Stop"
                                
                    if exit_price:
                        pnl_pct = (exit_price - pos['entry']) / pos['entry'] if pos['side'] == 'LONG' else (pos['entry'] - exit_price) / pos['entry']
                        usd_pnl = pnl_pct * pos['size'] - (pos['size'] * 0.0008)
                        balance += usd_pnl
                        trade_history.append({"side": pos['side'], "pnl": usd_pnl, "reason": reason})
                        positions.remove(pos)

                # C. Generate Sniper Signals (Limit Orders)
                if i % 5 == 0 and len(positions) == 0 and len(pending_orders) == 0:
                    d15_idx = int(actual_idx / 15)
                    sub_15m = df_15m.iloc[:d15_idx+1]
                    sub_1m = df_1m.iloc[:actual_idx+1]
                    if len(sub_15m) < 30: continue
                    
                    ema9_15m = MarketAnalyzer.get_ema(sub_15m['c'], 9).iloc[-1]
                    ema21_15m = MarketAnalyzer.get_ema(sub_15m['c'], 21).iloc[-1]
                    direction = 1 if ema9_15m > ema21_15m else -1
                    regime = MarketAnalyzer.detect_regime(sub_15m)
                    
                    # Simulation context
                    session = get_session_from_ts(curr_ts)
                    sub_btc = df_btc_1m.iloc[:actual_idx+1]
                    lead_lag = detect_lead_lag_sim(sub_1m, sub_btc)

                    s_weights = bot_state.get("neural_weights")
                    score = MarketAnalyzer.calculate_score(sub_1m, sub_15m, direction, 1.0, 0.0, regime=regime, neural_weights=s_weights, session=session, lead_lag=lead_lag)
                    
                    # AI ML Boost
                    ml_prob = 0.5
                    if ml_predictor.models.get(symbol):
                        try:
                            f_cols = ml_predictor.models[symbol].feature_name_
                            feats = ml_predictor.feature_engineering(sub_1m.copy())
                            if not feats.empty:
                                X = feats[f_cols].iloc[[-1]]
                                ml_prob = ml_predictor.models[symbol].predict_proba(X)[0][1]
                        except: pass
                    
                    if direction == 1:
                        if ml_prob >= 0.65: score += int((ml_prob - 0.6) * 50)
                        elif ml_prob < 0.40: score -= int((0.4 - ml_prob) * 50)
                    else:
                        if ml_prob <= 0.35: score += int((0.4 - ml_prob) * 50)
                        elif ml_prob > 0.60: score -= int((ml_prob - 0.6) * 50)
                    score = min(max(score, 0), 100)

                    # SNIPER ENTRY LOGIC (EMA/FVG)
                    ema9_1m = MarketAnalyzer.get_ema(sub_1m['c'], 9).iloc[-1]
                    limit_p = ema9_1m
                    fvg = MarketAnalyzer.get_nearest_fvg(sub_1m)
                    if direction == 1 and fvg and fvg["type"] == "BULLISH": limit_p = fvg["top"]
                    elif direction == -1 and fvg and fvg["type"] == "BEARISH": limit_p = fvg["bottom"]

                    # Threshold check
                    is_breakout = False
                    if actual_idx >= 6:
                        prev_5_h = df_1m['h'].iloc[actual_idx-6:actual_idx-1].max()
                        prev_5_l = df_1m['l'].iloc[actual_idx-6:actual_idx-1].min()
                        is_breakout = (direction == 1 and curr_price > prev_5_h) or (direction == -1 and curr_price < prev_5_l)

                    threshold = 85 if regime == "RANGING" else (70 if is_breakout else 80)
                    
                    if score >= threshold:
                        dist = abs(curr_price - limit_p) / curr_price * 100
                        if dist <= 0.5:
                            atr = atr_1m_all.iloc[actual_idx]
                            sl_pct = max(0.7, min(3.0, (atr * 2.97 / curr_price * 100)))
                            tp_pct = sl_pct * 2.08
                            
                            risk_mult = 1.2 if (ml_prob > 0.75 or ml_prob < 0.25) else (1.0 if 0.4<=ml_prob<=0.6 else 0.5)
                            trade_size_usd = (balance * risk_percent * risk_mult) / (sl_pct / 100)
                            
                            pending_orders.append({
                                'side': 'LONG' if direction == 1 else 'SHORT',
                                'price': limit_p, 'size': trade_size_usd,
                                'sl': limit_p * (1 - sl_pct/100) if direction == 1 else limit_p * (1 + sl_pct/100),
                                'tp': limit_p * (1 + tp_pct/100) if direction == 1 else limit_p * (1 - tp_pct/100),
                                'ts_act': sl_pct * 0.8, 'ts_cb': sl_pct * 0.3,
                                'ts': i
                            })
            
            # Close pending open positions at end of day
            for pos in positions:
                pnl_pct = (curr_price - pos['entry']) / pos['entry'] if pos['side'] == 'LONG' else (pos['entry'] - curr_price) / pos['entry']
                usd_pnl = pnl_pct * pos['size'] - (pos['size'] * 0.0008)
                balance += usd_pnl
                trade_history.append({"side": pos['side'], "pnl": usd_pnl, "reason": "End of Day Close"})
                
            win = len([t for t in trade_history if t['pnl'] > 0])
            loss = len([t for t in trade_history if t['pnl'] <= 0])
            total_pnl = sum([t['pnl'] for t in trade_history])
            col = "green" if total_pnl > 0 else "red"
            console.print(f"   [bold {col}]Result {symbol}: {win}W / {loss}L | PnL: ${total_pnl:+.2f}[/]\n")
            
    col = "green" if balance >= 100 else "red"
    console.print(f"[bold {col}]=== FINAL BALANCE AFTER 24H: ${balance:.2f} (Net: ${balance-100:+.2f}) ===[/]")

if __name__ == "__main__":
    asyncio.run(run_backtest())