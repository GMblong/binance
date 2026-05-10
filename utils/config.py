import os

# --- Trading Constants ---
MIN_NOTIONAL_USD = 5.5          # Binance minimum order value
MIN_VOLUME_FILTER = 5_000_000   # Minimum 24h quote volume for scanning
MIN_VOLUME_SCREENER = 10_000_000  # Minimum volume for coin screener
TAKER_FEE_PCT = 0.08  # Round-trip taker fee (0.04% open + 0.04% close)
CONSEC_LOSS_COOLDOWN_SEC = 3600   # 60 min cooldown after 3 consecutive losses
MAX_CONSEC_LOSSES = 3             # Trigger cooldown after N losses
ML_RETRAIN_INTERVAL_SEC = 7200    # 2h between ML model retrains (scalping needs fresh models)
API_BAN_SLEEP_SEC = 300           # Sleep duration on 418 hard ban
KLINE_MAX_CANDLES = 300           # Max candles to keep in memory
DB_SAVE_INTERVAL_SEC = 30         # Batched DB write interval

def load_env():
    keys = {}
    try:
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        keys[k] = v
    except (IOError, OSError):
        pass
    return keys

env = load_env()
API_KEY = env.get("BINANCE_API_KEY") or env.get("API_KEY")
API_SECRET = env.get("BINANCE_API_SECRET") or env.get("API_SECRET")
API_URL = env.get("API_URL", "https://fapi.binance.com")

MAX_POSITIONS = int(env.get("MAX_POSITIONS", 5))
USE_BTC_FILTER = env.get("USE_BTC_FILTER", "False").lower() == "true"
ACCOUNT_RISK_PERCENT = float(env.get("ACCOUNT_RISK_PERCENT", 0.02))
MAX_LEVERAGE = int(env.get("MAX_LEVERAGE", 20))
EXIT_ON_REVERSAL = env.get("EXIT_ON_REVERSAL", "True").lower() == "true"
GLOBAL_BTC_EXIT = env.get("GLOBAL_BTC_EXIT", "False").lower() == "true"
DAILY_LOSS_LIMIT_PCT = float(env.get("DAILY_LOSS_LIMIT_PCT", 0.05))
DAILY_PROFIT_TARGET_PCT = float(env.get("DAILY_PROFIT_TARGET_PCT", 0.10))
RETRAIN_ON_STARTUP = env.get("RETRAIN_ON_STARTUP", "True").lower() == "true"
