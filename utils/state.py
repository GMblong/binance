import asyncio
import pandas as pd
import time

bot_state = {
    "balance": 0.0,
    "btc_trend": 0,
    "btc_state": "INITIALIZING",
    "btc_dir": 0, # 1: Long Only, -1: Short Only, 0: Both, -99: Locked
    "last_log": "Initializing...",
    "trades": {},
    "ws_online": False,
    "daily_pnl": 0.0,
    "wins": 0,
    "losses": 0,
    "consec_losses": 0,
    "last_loss_time": 0,
    "state": "NORMAL", # NORMAL, FRAGILE, LOCKDOWN, DONE
    "start_balance": 0.0,
    "api_err_logged": False,
    "logged_secure": [],
    "alt_breadth": 0, # Percentage of bullish coins in scan list
    "btc_dom": 50.0, # Bitcoin Dominance %
    "ai_confidence": 1.0, # 1.0: Normal, >1.0: Conservative, <1.0: Aggressive
    "liq_map": {}, # Store predicted liquidation clusters per symbol
    "blacklist": {}, # {symbol: expiry_timestamp}
    "active_positions": [], # Cached from API
    "sym_perf": {}, # {symbol: {'w':0, 'l':0, 'c':0}} c=consec_loss
    "ws_msg_count": 0,
    "ws_last_msg": 0,
    # Context-Aware Performance: { "REGIME:FEATURE": [wins, losses] }
    "strat_perf": {}, 
    # Context-Aware Weights: { "REGIME:FEATURE": weight }
    "neural_weights": {},
    "heartbeat": 0,
    "limit_orders": {}, # {symbol: {orderId: ..., price: ..., side: ..., ai: ...}}
    "is_passive": False, # If True, don't open new trades
    "market_vol": 1.0, # Market-wide volatility index
    "directional_bias": 0, # Net direction of current positions
    "sym_weights": {}, # {symbol: {regime:feature: weight}}
    "ui_active": False, # If True, keyboard listener is paused
    "api_err_count": 0, # API Circuit Breaker counters
    "api_req_count": 0,
    "api_health_status": "OK", # OK, DEGRADED, BLOCKED
    "last_db_save": 0, # Batched DB save timestamp
}

symbol_info_cache = {}

class MarketData:
    def __init__(self):
        self.klines = {} # {symbol: {interval: DataFrame}}
        self.prices = {} # {symbol: price}
        self.tickers = [] # List of top tickers
        self.funding = {} # {symbol: rate}
        self.oi = {} # {symbol: open_interest}
        self.prev_oi = {} # {symbol: previous_oi}
        self.imbalance = {} # {symbol: ratio}
        self.last_prime = {} # {symbol: timestamp}
        self.current_scan_list = []
        self.lock = asyncio.Lock()

    async def update_kline(self, symbol, interval, new_candle):
        async with self.lock:
            if symbol not in self.klines: self.klines[symbol] = {}
            if interval not in self.klines[symbol]: return 
            
            df = self.klines[symbol][interval]
            if df.empty: return

            last_idx = df.index[-1]
            last_ot = df.at[last_idx, 'ot']
            
            if new_candle['ot'] == last_ot:
                # Optimized update: directly set values on the existing row
                for col in ["o", "h", "l", "c", "v", "tbv"]:
                    try:
                        df.iat[-1, df.columns.get_loc(col)] = float(new_candle[col])
                    except ValueError:
                        # If dtype conflict (e.g. int64 vs float), force cast the whole column
                        df[col] = df[col].astype(float)
                        df.iat[-1, df.columns.get_loc(col)] = float(new_candle[col])
            elif new_candle['ot'] > last_ot:
                # New candle rollover: minimal concat
                new_row = pd.DataFrame([new_candle])
                # Ensure new row has correct types before concat
                for col in ["o", "h", "l", "c", "v", "tbv"]:
                    new_row[col] = new_row[col].astype(float)
                
                df = pd.concat([df, new_row], ignore_index=True)
                if len(df) > 300:
                    df = df.iloc[-300:]
                self.klines[symbol][interval] = df.reset_index(drop=True)
                
            self.last_prime[symbol] = time.time()

market_data = MarketData()
