import optuna
import pandas as pd
import requests
from engine.ml_engine import ml_predictor
from strategies.analyzer import MarketAnalyzer
from utils.config import API_URL

# Sync version of simulation
def simulate_sync(symbol, sl_mult, tp_mult):
    res = requests.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "1m", "limit": 1000})
    df = pd.DataFrame(res.json()).iloc[:, [0, 1, 2, 3, 4, 5]]
    df.columns = ["ot", "o", "h", "l", "c", "v"]
    df = df.astype(float)
    
    atr = MarketAnalyzer.get_atr(df, 14)
    balance = 100.0
    
    for i in range(14, 900):
        curr_price = df.iloc[i]['c']
        curr_atr = atr.iloc[i]
        ema9 = df['c'].iloc[i-9:i].mean()
        ema21 = df['c'].iloc[i-21:i].mean()
        
        if ema9 > ema21:
            sl = curr_price - (curr_atr * sl_mult)
            tp = curr_price + (curr_atr * tp_mult)
            for j in range(1, 11):
                if i+j >= len(df): break
                high = df.iloc[i+j]['h']
                low = df.iloc[i+j]['l']
                if low <= sl:
                    balance -= 1.0
                    break
                if high >= tp:
                    balance += 1.5
                    break
    return balance

def objective(trial):
    sl_mult = trial.suggest_float("sl_mult", 1.0, 3.0)
    tp_mult = trial.suggest_float("tp_mult", 1.5, 5.0)
    return simulate_sync("BTCUSDT", sl_mult, tp_mult)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
