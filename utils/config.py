import os

API_URL = "https://fapi.binance.com"
ACCOUNT_RISK_PERCENT = 0.02  
MAX_POSITIONS = 5            
MAX_LEVERAGE = 20
            
USE_BTC_FILTER = False
EXIT_ON_REVERSAL = True 
GLOBAL_BTC_EXIT = False 

DAILY_PROFIT_TARGET_PCT = 0.10  # 10% Profit Target
DAILY_LOSS_LIMIT_PCT = 0.05    # 5% Loss Limit

def load_env():
    keys = {}
    try:
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        keys[k] = v
    except: pass
    return keys

env = load_env()
API_KEY = env.get("BINANCE_API_KEY")
API_SECRET = env.get("BINANCE_API_SECRET")
