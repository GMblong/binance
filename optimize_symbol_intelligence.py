import optuna
import pandas as pd
import numpy as np
import asyncio
import httpx
import sqlite3
import os
from strategies.analyzer import MarketAnalyzer
from engine.ml_engine import ml_predictor
from utils.config import API_URL

async def fetch_klines(client, symbol, interval, limit=1000):
    try:
        res = await client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": limit})
        if res.status_code != 200: return pd.DataFrame()
        df = pd.DataFrame(res.json()).iloc[:, [0, 1, 2, 3, 4, 5]]
        df.columns = ["ot", "o", "h", "l", "c", "v"]
        return df.astype(float)
    except: return pd.DataFrame()

def simulate_sync(symbol, weights, df_1m, df_15m):
    balance = 100.0
    risk_percent = 0.02 
    atr_1m = MarketAnalyzer.get_atr(df_1m, 14)
    positions = []
    
    # 24h simulation (1440m)
    lookback = min(1440, len(df_1m) - 150)
    start_idx = len(df_1m) - lookback
    
    for i in range(lookback):
        actual_idx = start_idx + i
        curr_row = df_1m.iloc[actual_idx]
        curr_price = curr_row['c']
        
        for pos in positions[:]:
            exit_price = None
            if pos['side'] == 'LONG':
                if curr_row['l'] <= pos['sl']: exit_price = pos['sl']
                elif curr_row['h'] >= pos['tp']: exit_price = pos['tp']
            else:
                if curr_row['h'] >= pos['sl']: exit_price = pos['sl']
                elif curr_row['l'] <= pos['tp']: exit_price = pos['tp']
                
            if exit_price:
                pnl_pct = (exit_price - pos['entry']) / pos['entry'] if pos['side'] == 'LONG' else (pos['entry'] - exit_price) / pos['entry']
                balance += pnl_pct * pos['size'] - (pos['size'] * 0.0008)
                positions.remove(pos)

        if i % 15 == 0 and len(positions) == 0:
            d15_idx = int(actual_idx / 15)
            sub_15m = df_15m.iloc[:d15_idx+1]
            sub_1m = df_1m.iloc[:actual_idx+1]
            if len(sub_15m) < 30: continue
            
            ema9_15m = MarketAnalyzer.get_ema(sub_15m['c'], 9).iloc[-1]
            ema21_15m = MarketAnalyzer.get_ema(sub_15m['c'], 21).iloc[-1]
            direction = 1 if ema9_15m > ema21_15m else -1
            
            regime = MarketAnalyzer.detect_regime(sub_15m)
            score = MarketAnalyzer.calculate_score(sub_1m, sub_15m, direction, 1.0, 0.0, regime=regime, neural_weights=weights)
            
            # Simple ML boost simulation
            ml_w = weights.get(f"{regime}:ml", 1.0)
            score = min(100, score + 10 * ml_w) 

            if score >= 75:
                atr = atr_1m.iloc[actual_idx]
                sl_pct = max(0.7, min(3.0, (atr * 2.97 / curr_price * 100)))
                tp_pct = sl_pct * 2.08
                trade_size_usd = balance * risk_percent / (sl_pct / 100)
                entry_price = curr_price
                sl_price = entry_price * (1 - sl_pct/100) if direction == 1 else entry_price * (1 + sl_pct/100)
                tp_price = entry_price * (1 + tp_pct/100) if direction == 1 else entry_price * (1 - tp_pct/100)
                positions.append({'side': 'LONG' if direction == 1 else 'SHORT', 'entry': entry_price, 'size': trade_size_usd, 'sl': sl_price, 'tp': tp_price})
    return balance

async def optimize_symbol(client, symbol):
    print(f"🧠 Intelligence: Fetching data for {symbol}...")
    df_1m = await fetch_klines(client, symbol, "1m", 1500)
    df_15m = await fetch_klines(client, symbol, "15m", 500)
    
    if df_1m.empty or df_15m.empty: return None

    def run_study():
        print(f"🚀 Intelligence: Starting parallel optimization for {symbol}...")
        def objective(trial):
            weights = {}
            for regime in ["RANGING", "TRENDING", "VOLATILE"]:
                weights[f"{regime}:liq"] = trial.suggest_float(f"{regime}_liq", 0.5, 4.0)
                weights[f"{regime}:ob"] = trial.suggest_float(f"{regime}_ob", 0.5, 4.0)
                weights[f"{regime}:div"] = trial.suggest_float(f"{regime}_div", 0.5, 4.0)
                weights[f"{regime}:ml"] = trial.suggest_float(f"{regime}_ml", 0.5, 4.0)
            return simulate_sync(symbol, weights, df_1m, df_15m)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=60)
        return study.best_params

    return await asyncio.to_thread(run_study)

def save_to_db(symbol, best_params):
    conn = sqlite3.connect("bot_data.db")
    cursor = conn.cursor()
    for k, v in best_params.items():
        parts = k.split("_")
        regime = parts[0]
        feat = parts[1]
        feature_key = f"{regime}:{feat}"
        cursor.execute("INSERT OR REPLACE INTO sym_weights (symbol, feature, weight) VALUES (?, ?, ?)", (symbol, feature_key, v))
    conn.commit()
    conn.close()

async def get_all_symbols(client):
    try:
        res = await client.get(f"{API_URL}/fapi/v1/ticker/24hr")
        if res.status_code == 200:
            data = res.json()
            # Filter: Only USDT pairs with at least 5M volume to ensure data quality
            symbols = [t["symbol"] for t in data if t["symbol"].endswith("USDT") and float(t["quoteVolume"]) > 5_000_000]
            return sorted(symbols)
    except: pass
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

async def main():
    async with httpx.AsyncClient() as client:
        symbols = await get_all_symbols(client)
        print(f"🔥 Starting Parallel Intelligence Training for ALL {len(symbols)} symbols...")
        
        # Process in batches of 10 to prevent system overload
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            print(f"\n📦 Processing Batch {i//batch_size + 1}: {batch}")
            
            tasks = [optimize_symbol(client, sym) for sym in batch]
            results = await asyncio.gather(*tasks)
            
            for j, best in enumerate(results):
                sym = batch[j]
                if best:
                    save_to_db(sym, best)
                    print(f"✅ Symbol Intelligence Updated for {sym}")
                else:
                    print(f"❌ Failed to optimize {sym}")

if __name__ == "__main__":
    asyncio.run(main())
