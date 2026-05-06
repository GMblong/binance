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

console = Console()

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
    risk_percent = 0.05 # 5% account risk per trade
    
    console.print(f"[bold green]=== Backtest Simulator (Last 24 Hours) ===[/]")
    console.print(f"Starting Balance: ${balance:.2f}\n")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        symbols = await get_top_movers(client, limit=5)
        console.print(f"Selected Top Movers for Simulation: {symbols}\n")
        
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
            trade_history = []
            
            atr_1m_all = MarketAnalyzer.get_atr(df_1m, 14)
            
            console.print(f"   [dim]Simulating 1440 minutes (tick-by-tick proxy)...[/dim]")
            
            for i in range(1440):
                actual_idx = len(df_1m) - 1440 + i
                curr_row = df_1m.iloc[actual_idx]
                curr_price = curr_row['c']
                
                # Check Exits
                for pos in positions[:]:
                    exit_price = None
                    reason = ""
                    
                    if pos['side'] == 'LONG':
                        pos['peak'] = max(pos['peak'], curr_row['h'])
                        if curr_row['l'] <= pos['sl']: exit_price, reason = pos['sl'], "Stop Loss"
                        elif curr_row['h'] >= pos['tp']: exit_price, reason = pos['tp'], "Take Profit"
                        else:
                            peak_pnl = (pos['peak'] - pos['entry']) / pos['entry'] * 100
                            curr_pnl = (curr_price - pos['entry']) / pos['entry'] * 100
                            if peak_pnl > pos['ts_act'] and (peak_pnl - curr_pnl) > pos['ts_cb']:
                                exit_price, reason = curr_price, "AI Trailing Stop"
                    else:
                        pos['peak'] = min(pos['peak'], curr_row['l'])
                        if curr_row['h'] >= pos['sl']: exit_price, reason = pos['sl'], "Stop Loss"
                        elif curr_row['l'] <= pos['tp']: exit_price, reason = pos['tp'], "Take Profit"
                        else:
                            peak_pnl = (pos['entry'] - pos['peak']) / pos['entry'] * 100
                            curr_pnl = (pos['entry'] - curr_price) / pos['entry'] * 100
                            if peak_pnl > pos['ts_act'] and (peak_pnl - curr_pnl) > pos['ts_cb']:
                                exit_price, reason = curr_price, "AI Trailing Stop"
                                
                    if exit_price:
                        # Realized PnL Calculation (Leveraged Position Size)
                        pnl_pct = (exit_price - pos['entry']) / pos['entry'] if pos['side'] == 'LONG' else (pos['entry'] - exit_price) / pos['entry']
                        usd_pnl = pnl_pct * pos['size']
                        
                        # Deduct generic trading fee (0.04% per side)
                        fee = pos['size'] * 0.0008
                        usd_pnl -= fee
                        
                        balance += usd_pnl
                        trade_history.append({"side": pos['side'], "pnl": usd_pnl, "reason": reason})
                        positions.remove(pos)

                # Generate Signals every 15 minutes if no active positions
                if i % 15 == 0 and len(positions) == 0:
                    d15_idx = int(actual_idx / 15)
                    sub_15m = df_15m.iloc[:d15_idx+1]
                    sub_1m = df_1m.iloc[:actual_idx+1]
                    
                    if len(sub_15m) < 30: continue
                    
                    ema9_15m = MarketAnalyzer.get_ema(sub_15m['c'], 9).iloc[-1]
                    ema21_15m = MarketAnalyzer.get_ema(sub_15m['c'], 21).iloc[-1]
                    direction = 1 if ema9_15m > ema21_15m else -1
                    
                    score = MarketAnalyzer.calculate_score(sub_1m, sub_15m, direction, 1.0, 0.0)
                    
                    # AI Inference
                    ml_prob = 0.5
                    if ml_predictor.models.get(symbol):
                        try:
                            feats = sub_15m.copy()
                            feats = ml_predictor.feature_engineering(feats)
                            model = ml_predictor.models[symbol]
                            f_cols = model.feature_name_
                            if not feats.empty and all(c in feats.columns for c in f_cols):
                                X = feats[f_cols].iloc[[-1]]
                                ml_prob = model.predict_proba(X)[0][1]
                        except: pass
                    
                    if direction == 1 and ml_prob > 0.6: score += int((ml_prob - 0.5) * 40)
                    elif direction == -1 and ml_prob < 0.4: score += int((0.5 - ml_prob) * 40)
                    score = min(max(score, 0), 100)
                    
                    if score >= 75:
                        atr = atr_1m_all.iloc[actual_idx]
                        sl_pct = max(0.7, min(3.0, (atr * 2 / curr_price * 100)))
                        tp_pct = sl_pct * 1.5
                        
                        risk_mult = 1.2 if (ml_prob > 0.75 or ml_prob < 0.25) else (1.0 if 0.4<=ml_prob<=0.6 else 0.5)
                        trade_risk_usd = balance * risk_percent * risk_mult
                        trade_size_usd = trade_risk_usd / (sl_pct / 100)
                        
                        entry_price = curr_price
                        sl_price = entry_price * (1 - sl_pct/100) if direction == 1 else entry_price * (1 + sl_pct/100)
                        tp_price = entry_price * (1 + tp_pct/100) if direction == 1 else entry_price * (1 - tp_pct/100)
                        
                        positions.append({
                            'side': 'LONG' if direction == 1 else 'SHORT',
                            'entry': entry_price, 'size': trade_size_usd,
                            'sl': sl_price, 'tp': tp_price,
                            'ts_act': sl_pct * 0.8, 'ts_cb': sl_pct * 0.3,
                            'peak': entry_price
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