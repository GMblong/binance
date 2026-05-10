import asyncio
import pandas as pd
import numpy as np
import time
import threading
from typing import Optional, List, Tuple
from collections import deque


class BotState(dict):
    """Thread-safe dict wrapper for bot state with default values.
    
    Provides safe access via .get() with defaults, and prevents KeyError
    on missing keys by returning sensible defaults.
    """
    _lock = threading.Lock()

    _DEFAULTS = {
        "balance": 0.0,
        "btc_trend": 0,
        "btc_state": "INITIALIZING",
        "btc_dir": 0,
        "last_log": "Initializing...",
        "trades": {},
        "ws_online": False,
        "daily_pnl": 0.0,
        "wins": 0,
        "losses": 0,
        "consec_losses": 0,
        "last_loss_time": 0,
        "state": "NORMAL",
        "start_balance": 0.0,
        "api_err_logged": False,
        "logged_secure": [],
        "alt_breadth": 0,
        "btc_dom": 50.0,
        "ai_confidence": 1.0,
        "liq_map": {},
        "blacklist": {},
        "active_positions": [],
        "sym_perf": {},
        "ws_msg_count": 0,
        "ws_last_msg": 0,
        "strat_perf": {},
        "neural_weights": {},
        "heartbeat": 0,
        "limit_orders": {},
        "is_passive": False,
        "market_vol": 1.0,
        "directional_bias": 0,
        "sym_weights": {},
        "ui_active": False,
        "api_err_count": 0,
        "api_req_count": 0,
        "api_health_status": "OK",
        "last_db_save": 0,
    }

    def __init__(self):
        super().__init__(self._DEFAULTS)

    def __missing__(self, key):
        """Return None for missing keys instead of raising KeyError."""
        return None

    def safe_increment(self, key: str, amount: int = 1):
        """Thread-safe increment for counters."""
        with self._lock:
            self[key] = self.get(key, 0) + amount

    def snapshot_for_db(self) -> dict:
        """Thread-safe snapshot of keys needed for DB save."""
        with self._lock:
            return {
                "daily_pnl": self.get("daily_pnl", 0.0),
                "wins": self.get("wins", 0),
                "losses": self.get("losses", 0),
                "ai_confidence": self.get("ai_confidence", 1.0),
                "start_balance": self.get("start_balance", 0.0),
                "_last_day": self.get("_last_day", ""),
                "sym_perf": dict(self.get("sym_perf", {})),
                "strat_perf": dict(self.get("strat_perf", {})),
                "neural_weights": dict(self.get("neural_weights", {})),
                "sym_weights": dict(self.get("sym_weights", {})),
                "blacklist": dict(self.get("blacklist", {})),
                "trades": dict(self.get("trades", {})),
            }

    def apply_db_load(self, data: dict):
        """Thread-safe bulk update from DB load results."""
        with self._lock:
            for k, v in data.items():
                self[k] = v


bot_state = BotState()

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
        # Live CVD buffer from @aggTrade (tick-level).
        # Each entry: (timestamp_sec, signed_quote_volume)
        # signed = +qty*price if aggressive BUY (m=False), -qty*price if aggressive SELL (m=True)
        self.cvd_buf = {}  # {symbol: deque[(ts, signed_qvol)]}
        # Tick-level trade buffer (richer than cvd_buf) for microstructure alpha:
        # (ts, price, qty, sign) where sign=+1 aggressive BUY, -1 aggressive SELL
        self.tick_buf = {}  # {symbol: deque[(ts, price, qty, sign)]}
        # Orderflow microstructure: depth snapshots for velocity & spoofing detection
        self.depth_history = {}  # {symbol: deque[(ts, bid_total, ask_total, top_bid_qty, top_ask_qty)]}
        # Best bid/ask + sizes for microprice (weighted fair value)
        self.best_quote = {}  # {symbol: (ts, best_bid, best_bid_qty, best_ask, best_ask_qty)}
        # Microstructure alpha cache (computed periodically, read by strategy)
        self.micro_alpha = {}  # {symbol: {features_dict, ts}}
        self.lock = asyncio.Lock()

    def push_agg_trade(self, symbol: str, ts_sec: float, qty: float, price: float, is_buyer_maker: bool, max_keep: int = 600):
        """Append a signed quote-volume delta from an aggTrade event.

        Also populates tick_buf (richer info: price + qty + sign) used by the
        microstructure alpha engine (VPIN, Kyle's lambda, whale prints, etc.).
        """
        buf = self.cvd_buf.get(symbol)
        if buf is None:
            buf = deque(maxlen=max_keep)
            self.cvd_buf[symbol] = buf
        sign = -1.0 if is_buyer_maker else 1.0
        signed = qty * price * sign
        buf.append((ts_sec, signed))

        # Tick-level buffer for microstructure calculations
        tbuf = self.tick_buf.get(symbol)
        if tbuf is None:
            tbuf = deque(maxlen=max_keep * 3)  # keep more ticks (~1800)
            self.tick_buf[symbol] = tbuf
        tbuf.append((ts_sec, price, qty, sign))

    def get_trades(self, symbol: str, window_sec: int = 60, max_items: int = 600) -> List[Tuple[float, float, float, float]]:
        """Return list of tick-level trades in window: [(ts, price, qty, sign), ...]

        Most-recent-first iteration is converted to chronological order.
        """
        buf = self.tick_buf.get(symbol)
        if not buf:
            return []
        cutoff = time.time() - window_sec
        out = []
        for row in reversed(buf):
            if row[0] < cutoff or len(out) >= max_items:
                break
            out.append(row)
        out.reverse()
        return out

    def get_live_cvd(self, symbol: str, window_sec: int = 60) -> Tuple[float, int]:
        """Return (cvd_sum_usd, n_trades) over the last `window_sec` seconds."""
        buf = self.cvd_buf.get(symbol)
        if not buf:
            return 0.0, 0
        cutoff = time.time() - window_sec
        total = 0.0
        n = 0
        for ts, delta in reversed(buf):
            if ts < cutoff:
                break
            total += delta
            n += 1
        return total, n

    def push_depth_snapshot(self, symbol: str, bid_total: float, ask_total: float, top_bid_qty: float, top_ask_qty: float):
        """Store orderbook depth snapshot for velocity analysis."""
        buf = self.depth_history.get(symbol)
        if buf is None:
            buf = deque(maxlen=60)
            self.depth_history[symbol] = buf
        buf.append((time.time(), bid_total, ask_total, top_bid_qty, top_ask_qty))

    def push_best_quote(self, symbol: str, best_bid: float, best_bid_qty: float,
                        best_ask: float, best_ask_qty: float):
        """Store best bid/ask + sizes for microprice calculation."""
        self.best_quote[symbol] = (time.time(), best_bid, best_bid_qty, best_ask, best_ask_qty)

    def get_microprice(self, symbol: str) -> Optional[float]:
        """Size-weighted fair value: microprice = (bid*ask_qty + ask*bid_qty) / (bid_qty + ask_qty).

        More predictive than mid-price because it reflects the side with LESS
        liquidity (where price is likely to move next).
        Returns None if no recent quote.
        """
        q = self.best_quote.get(symbol)
        if not q:
            return None
        ts, bb, bbq, ba, baq = q
        if time.time() - ts > 5.0 or (bbq + baq) <= 0 or bb <= 0 or ba <= 0:
            return None
        return (bb * baq + ba * bbq) / (bbq + baq)

    def get_depth_velocity(self, symbol: str, window_sec: int = 10) -> Tuple[float, float, float]:
        """Calculate rate of change in orderbook depth.
        
        Returns: (bid_velocity, ask_velocity, spoof_score)
        - bid_velocity > 0: bids increasing (buying pressure building)
        - ask_velocity > 0: asks increasing (selling pressure building)
        - spoof_score > 0.5: likely spoofing detected (large orders appearing/disappearing)
        """
        buf = self.depth_history.get(symbol)
        if not buf or len(buf) < 3:
            return 0.0, 0.0, 0.0
        
        cutoff = time.time() - window_sec
        recent = [(ts, bt, at, tbq, taq) for ts, bt, at, tbq, taq in buf if ts >= cutoff]
        if len(recent) < 3:
            return 0.0, 0.0, 0.0
        
        # Velocity = (latest - earliest) / time_diff
        t_diff = recent[-1][0] - recent[0][0]
        if t_diff == 0:
            return 0.0, 0.0, 0.0
        
        bid_vel = (recent[-1][1] - recent[0][1]) / t_diff
        ask_vel = (recent[-1][2] - recent[0][2]) / t_diff
        
        # Spoofing detection: large top-of-book qty that appears then vanishes
        top_bid_changes = []
        top_ask_changes = []
        for i in range(1, len(recent)):
            top_bid_changes.append(abs(recent[i][3] - recent[i-1][3]))
            top_ask_changes.append(abs(recent[i][4] - recent[i-1][4]))
        
        avg_top_bid = sum(recent[i][3] for i in range(len(recent))) / len(recent) if recent else 1
        avg_top_ask = sum(recent[i][4] for i in range(len(recent))) / len(recent) if recent else 1
        
        # High churn at top of book relative to average size = spoofing
        bid_churn = (sum(top_bid_changes) / len(top_bid_changes)) / (avg_top_bid + 1e-8) if top_bid_changes else 0
        ask_churn = (sum(top_ask_changes) / len(top_ask_changes)) / (avg_top_ask + 1e-8) if top_ask_changes else 0
        spoof_score = min(1.0, max(bid_churn, ask_churn))
        
        return bid_vel, ask_vel, spoof_score

    def detect_iceberg(self, symbol: str, window_sec: int = 30) -> Tuple[bool, bool]:
        """Detect iceberg orders: large hidden orders that refill at same price level.
        
        Pattern: top-of-book qty stays constant despite aggressive trades eating into it.
        If aggTrade volume consumed > displayed qty but displayed qty barely changes → iceberg.
        
        Returns: (iceberg_bid: bool, iceberg_ask: bool)
        """
        buf = self.depth_history.get(symbol)
        agg_buf = self.cvd_buf.get(symbol)
        if not buf or len(buf) < 6 or not agg_buf:
            return False, False
        
        cutoff = time.time() - window_sec
        recent_depth = [(ts, bt, at, tbq, taq) for ts, bt, at, tbq, taq in buf if ts >= cutoff]
        if len(recent_depth) < 6:
            return False, False
        
        # Calculate total aggressive volume consumed in window
        buy_vol = 0.0
        sell_vol = 0.0
        for ts, delta in reversed(agg_buf):
            if ts < cutoff:
                break
            if delta > 0:
                buy_vol += delta
            else:
                sell_vol += abs(delta)
        
        # Check bid side: if lots of selling (aggressor sells into bids) but top bid barely moves
        top_bid_values = [d[3] for d in recent_depth]
        top_ask_values = [d[4] for d in recent_depth]
        
        bid_std = np.std(top_bid_values) if len(top_bid_values) > 1 else 999
        ask_std = np.std(top_ask_values) if len(top_ask_values) > 1 else 999
        avg_top_bid = np.mean(top_bid_values)
        avg_top_ask = np.mean(top_ask_values)
        
        # Iceberg bid: heavy selling into bids, but bid qty stays stable (refilling)
        iceberg_bid = (sell_vol > avg_top_bid * 3) and (bid_std / (avg_top_bid + 1e-8) < 0.3)
        # Iceberg ask: heavy buying into asks, but ask qty stays stable (refilling)
        iceberg_ask = (buy_vol > avg_top_ask * 3) and (ask_std / (avg_top_ask + 1e-8) < 0.3)
        
        return iceberg_bid, iceberg_ask

    async def update_kline(self, symbol, interval, new_candle):
        """Optimized kline update - minimal lock time, direct numpy operations."""
        if symbol not in self.klines or interval not in self.klines[symbol]:
            return
        
        df = self.klines[symbol][interval]
        if df.empty:
            return

        last_ot = df.iat[-1, df.columns.get_loc('ot')]
        new_ot = new_candle['ot']
        
        if new_ot == last_ot:
            # Update existing candle in-place (no lock needed for atomic writes)
            idx = len(df) - 1
            df.iat[idx, df.columns.get_loc('o')] = new_candle['o']
            df.iat[idx, df.columns.get_loc('h')] = new_candle['h']
            df.iat[idx, df.columns.get_loc('l')] = new_candle['l']
            df.iat[idx, df.columns.get_loc('c')] = new_candle['c']
            df.iat[idx, df.columns.get_loc('v')] = new_candle['v']
            df.iat[idx, df.columns.get_loc('tbv')] = new_candle['tbv']
        elif new_ot > last_ot:
            # New candle - need lock for structural change
            async with self.lock:
                new_row = pd.DataFrame([new_candle])
                for col in ["o", "h", "l", "c", "v", "tbv"]:
                    new_row[col] = new_row[col].astype(float)
                df = pd.concat([df, new_row], ignore_index=True)
                if len(df) > 300:
                    df = df.iloc[-300:]
                self.klines[symbol][interval] = df.reset_index(drop=True)
                
        self.last_prime[symbol] = time.time()

market_data = MarketData()
