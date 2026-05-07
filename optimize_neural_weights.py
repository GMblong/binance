import optuna
import pandas as pd
import numpy as np
import asyncio
import httpx
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
    
    lookback = 500
    start_idx = len(df_1m) - lookback
    
    for i in range(lookback):
        actual_idx = start_idx + i
        if actual_idx >= len(df_1m): break
        
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
            # Simplified sub-selection for speed
            sub_15m = df_15m.iloc[:d15_idx+1]
            sub_1m = df_1m.iloc[:actual_idx+1]
            
            if len(sub_15m) < 30: continue
            
            ema9_15m = MarketAnalyzer.get_ema(sub_15m['c'], 9).iloc[-1]
            ema21_15m = MarketAnalyzer.get_ema(sub_15m['c'], 21).iloc[-1]
            direction = 1 if ema9_15m > ema21_15m else -1
            
            regime = MarketAnalyzer.detect_regime(sub_15m)
            # Use calculate_score without ml_prob boost for basic weight optimization
            score = MarketAnalyzer.calculate_score(sub_1m, sub_15m, direction, 1.0, 0.0, regime=regime, neural_weights=weights)
            
            if score >= 75:
                atr = atr_1m.iloc[actual_idx]
                sl_pct = max(0.7, min(3.0, (atr * 2.97 / curr_price * 100)))
                tp_pct = sl_pct * 2.08
                
                trade_size_usd = balance * risk_percent / (sl_pct / 100)
                entry_price = curr_price
                sl_price = entry_price * (1 - sl_pct/100) if direction == 1 else entry_price * (1 + sl_pct/100)
                tp_price = entry_price * (1 + tp_pct/100) if direction == 1 else entry_price * (1 - tp_pct/100)
                
                positions.append({
                    'side': 'LONG' if direction == 1 else 'SHORT',
                    'entry': entry_price, 'size': trade_size_usd,
                    'sl': sl_price, 'tp': tp_price
                })
    return balance

async def run_optimization():
    async with httpx.AsyncClient() as client:
        symbols = ["ZEREBROUSDT", "PLAYUSDT"]
        print(f"Fetching market data for {symbols}...")
        
        data_map = {}
        for symbol in symbols:
            df_1m = await fetch_klines(client, symbol, "1m", 1000)
            df_15m = await fetch_klines(client, symbol, "15m", 500)
            if not df_1m.empty and not df_15m.empty:
                data_map[symbol] = (df_1m, df_15m)
        
        if not data_map:
            print("Failed to fetch data for any symbol.")
            return

        def objective(trial):
            weights = {
                "RANGING:liq": trial.suggest_float("liq_w", 0.1, 4.0),
                "RANGING:ob": trial.suggest_float("ob_w", 0.1, 4.0),
                "RANGING:div": trial.suggest_float("div_w", 0.1, 4.0),
                "TRENDING:liq": trial.suggest_float("t_liq_w", 0.1, 4.0),
                "TRENDING:ob": trial.suggest_float("t_ob_w", 0.1, 4.0),
                "TRENDING:div": trial.suggest_float("t_div_w", 0.1, 4.0),
            }
            
            total_balance = 0
            for symbol, (df_1m, df_15m) in data_map.items():
                total_balance += simulate_sync(symbol, weights, df_1m, df_15m)
            
            return total_balance / len(data_map) # Optimize average balance

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=150) # Increased trials for multi-symbol
        
        print("\nOptimization Complete!")
        print(f"Best Avg Balance: ${study.best_value:.2f}")
        print("Best Weights:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    asyncio.run(run_optimization())
