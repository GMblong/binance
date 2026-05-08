import httpx
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from utils.config import API_URL
import pandas_ta as ta
import asyncio
import time

class MLPredictor:
    def __init__(self):
        self.models = {}  # Store models per symbol
        self.last_trained = {} # Store last training time per symbol
        self.performance = {} # Store win rate tracking (symbol: [list of bool])
        self.feature_importance = {} # Store feature importance per symbol
        self.feature_cache = {} # Cache for pre-calculated features {symbol: df}
        self.retrain_sem = asyncio.Semaphore(5) # Allow 5 concurrent retrains for fast startup
        self.startup_done = False
        # Track running retrains so we don't dispatch the same symbol twice.
        self._retraining = set()
        # Drift-based retrain bookkeeping.
        self.trades_since_train = {}  # {symbol: int}
        self.last_drift_check = {}    # {symbol: ts}

    def update_performance(self, symbol, is_win):
        if symbol not in self.performance: self.performance[symbol] = []
        self.performance[symbol].append(is_win)
        if len(self.performance[symbol]) > 50: self.performance[symbol].pop(0)
        # Increment trades-since-train counter for drift detection
        self.trades_since_train[symbol] = self.trades_since_train.get(symbol, 0) + 1

    def recent_win_rate(self, symbol, window: int = 15):
        """Win rate over the last `window` trades, or None if insufficient."""
        perf = self.performance.get(symbol, [])
        if len(perf) < window:
            return None
        tail = perf[-window:]
        return sum(tail) / len(tail)

    def should_retrain(self, symbol, now=None) -> bool:
        """Decide if a symbol's model is stale enough to warrant retraining.

        Triggers (any one is enough):
        - No model yet.
        - Time-based: trained >12h ago (legacy behaviour).
        - Performance drift: recent win rate <45% over last 15 trades AND
          at least 15 trades have happened since last train.
        - Trade count: 40+ closed trades since last train (guaranteed refresh).
        """
        import time as _t
        now = now or _t.time()
        if symbol not in self.models:
            return True
        last = self.last_trained.get(symbol, 0)
        if (now - last) > 43200:  # 12h
            return True
        tst = self.trades_since_train.get(symbol, 0)
        if tst >= 40:
            return True
        wr = self.recent_win_rate(symbol, window=15)
        if wr is not None and wr < 0.45 and tst >= 15:
            return True
        return False

    async def maybe_retrain(self, client, symbol):
        """Dispatch a background retrain if needed + not already running."""
        if symbol in self._retraining:
            return False
        if not self.should_retrain(symbol):
            return False
        self._retraining.add(symbol)

        async def _go():
            try:
                async with self.retrain_sem:
                    ok = await self.train_model(client, symbol)
                if ok:
                    self.trades_since_train[symbol] = 0
            finally:
                self._retraining.discard(symbol)

        asyncio.create_task(_go())
        return True

    def _get_feature_list(self, df_cols):
        features = [
            'ema_9', 'ema_21', 'rsi', 'atr', 
            'roc_c_1', 'roc_c_5', 'roc_v_1', 
            'volatility', 'body_size', 'upper_wick', 'lower_wick',
            'dist_ema9', 'dist_ema21', 'dist_vwap', 'cvd_roc',
            'oi_roc', 'funding'
        ]
        features.extend([col for col in df_cols if 'MACD' in col])
        return features

    def feature_engineering(self, df):
        """Optimized feature engineering with local caching."""
        # Use only the last 100 rows for inference to speed up, or 1500 for training
        is_training = len(df) > 200
        
        # 1. Base Technical Indicators
        df['ema_9'] = ta.ema(df['c'], length=9)
        df['ema_21'] = ta.ema(df['c'], length=21)
        df['rsi'] = ta.rsi(df['c'], length=14)
        df['atr'] = ta.atr(df['h'], df['l'], df['c'], length=14)
        
        # Fast MACD calculation
        exp1 = df['c'].ewm(span=12, adjust=False).mean()
        exp2 = df['c'].ewm(span=26, adjust=False).mean()
        df['MACD_12_26_9'] = exp1 - exp2
        df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
        df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
        
        # Higher Time Frame Proxies (using 1m data)
        df['ema_60'] = ta.ema(df['c'], length=60)
            
        # Institutional Flow Proxies (VWAP & CVD)
        df['typical_price'] = (df['h'] + df['l'] + df['c']) / 3
        df['vol_price'] = df['typical_price'] * df['v']
        df['vwap'] = df['vol_price'].rolling(window=100).sum() / df['v'].rolling(window=100).sum()
        df['dist_vwap'] = (df['c'] - df['vwap']) / df['vwap']
        df.drop(columns=['vol_price'], inplace=True, errors='ignore')
        
        # Advanced CVD Proxy (Wick-based Volume Delta)
        range_diff = df['h'] - df['l'] + 1e-8
        buy_vol = df['v'] * ((df['c'] - df['l']) / range_diff)
        sell_vol = df['v'] * ((df['h'] - df['c']) / range_diff)
        df['cvd'] = (buy_vol - sell_vol).cumsum()
        df['cvd_roc'] = df['cvd'].pct_change(3).fillna(0)
        
        # 2. Institutional Flow (OI & Funding)
        if 'oi' not in df.columns: df['oi'] = 0
        df['oi_roc'] = df['oi'].pct_change(3).fillna(0)
        if 'funding' not in df.columns: df['funding'] = 0

        # 3. Momentum & Volatility
        df['roc_c_1'] = df['c'].pct_change(1)
        df['roc_c_5'] = df['c'].pct_change(5)
        df['roc_v_1'] = df['v'].pct_change(1)
        df['volatility'] = (df['h'] - df['l']) / df['c']
        df['body_size'] = abs(df['c'] - df['o']) / df['c']
        df['upper_wick'] = (df['h'] - df[['o', 'c']].max(axis=1)) / df['c']
        df['lower_wick'] = (df[['o', 'c']].min(axis=1) - df['l']) / df['c']
        
        # 4. Trend Strength
        df['dist_ema9'] = (df['c'] - df['ema_9']) / df['ema_9']
        df['dist_ema21'] = (df['c'] - df['ema_21']) / df['ema_21']
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Drop only necessary temporary columns
        df.drop(columns=['typical_price', 'dir_vol', 'cvd'], inplace=True, errors='ignore')
        if not is_training:
            return df.tail(1) # For prediction, we only need the last row
        
        features = self._get_feature_list(df.columns)
        df.dropna(subset=features, inplace=True)
        return df

    def apply_triple_barrier(self, df, pt_mult=1.5, sl_mult=1.0, lookahead=12):
        targets = []
        prices, atrs, highs, lows = df['c'].values, df['atr'].values, df['h'].values, df['l'].values
        
        # Adaptive Multipliers based on recent median ATR to normalize volatility regimes
        recent_median_atr = df['atr'].rolling(100).median().bfill().values
        
        for i in range(len(df)):
            if i + lookahead >= len(df):
                targets.append(np.nan)
                continue
            entry_price, current_atr, base_atr = prices[i], atrs[i], recent_median_atr[i]
            
            # Scale the multipliers: if current ATR is very high compared to base, widen the barrier
            vol_ratio = current_atr / (base_atr + 1e-8)
            dyn_pt_mult = pt_mult * min(max(vol_ratio, 0.8), 2.0)
            dyn_sl_mult = sl_mult * min(max(vol_ratio, 0.8), 2.0)
            
            pt_price, sl_price = entry_price + (current_atr * dyn_pt_mult), entry_price - (current_atr * dyn_sl_mult)
            label = 0 # 1 = Hit PT, 0 = Hit SL or Timeout (Chop)
            for j in range(1, lookahead + 1):
                idx = i + j
                if highs[idx] >= pt_price: label = 1; break
                elif lows[idx] <= sl_price: label = 0; break
            targets.append(label)
        df['target'] = targets
        df.dropna(subset=['target'], inplace=True)
        return df

    async def fetch_historical_data(self, client, symbol, interval="15m", limit=300, end_time=None):
        try:
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            if end_time: params["endTime"] = end_time
            res = await client.get(f"{API_URL}/fapi/v1/klines", params=params, timeout=15)
            if res.status_code != 200: return None
            df = pd.DataFrame(res.json()).iloc[:, [0, 1, 2, 3, 4, 5]]
            df.columns = ["ot", "o", "h", "l", "c", "v"]
            for col in ["ot", "o", "h", "l", "c", "v"]: df[col] = df[col].astype(float)
            return df
        except: return None

    async def fetch_extended_data(self, client, symbol, interval="15m", total=1500):
        """Fetch extended historical data by paginating backwards."""
        all_dfs = []
        end_time = None
        per_req = 1500
        remaining = total
        while remaining > 0:
            limit = min(per_req, remaining)
            df = await self.fetch_historical_data(client, symbol, interval, limit, end_time)
            if df is None or df.empty:
                break
            all_dfs.insert(0, df)
            remaining -= len(df)
            if len(df) < limit:
                break
            end_time = int(df['ot'].iloc[0]) - 1
            await asyncio.sleep(0.1)
        if not all_dfs:
            return None
        result = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=['ot']).sort_values('ot').reset_index(drop=True)
        return result

    async def _inject_funding_oi(self, client, symbol, df):
        """Inject real funding rate and OI data into training dataframe."""
        from utils.state import market_data
        try:
            # Fetch funding rate history
            res = await client.get(f"{API_URL}/fapi/v1/fundingRate", params={"symbol": symbol, "limit": 100}, timeout=10)
            funding_map = {}
            if res and res.status_code == 200:
                for f in res.json():
                    funding_map[int(f['fundingTime'])] = float(f['fundingRate'])
            
            # Assign funding to nearest candle
            if funding_map:
                fund_times = sorted(funding_map.keys())
                df['funding'] = 0.0
                for ft in fund_times:
                    mask = df['ot'] <= ft
                    if mask.any():
                        idx = mask[mask].index[-1]
                        df.loc[idx:, 'funding'] = funding_map[ft]
            
            # Inject OI from live state or fetch
            live_oi = market_data.oi.get(symbol, 0)
            if live_oi > 0:
                df['oi'] = live_oi  # Static fill (best effort for training)
            else:
                df['oi'] = 0
            
            # Fetch OI history if available
            try:
                oi_res = await client.get(f"{API_URL}/futures/data/openInterestHist", 
                    params={"symbol": symbol, "period": "15m", "limit": 100}, timeout=10)
                if oi_res and oi_res.status_code == 200:
                    oi_data = oi_res.json()
                    oi_map = {int(x['timestamp']): float(x['sumOpenInterest']) for x in oi_data}
                    if oi_map:
                        oi_times = sorted(oi_map.keys())
                        for ot_val in oi_times:
                            mask = (df['ot'] >= ot_val - 900000) & (df['ot'] <= ot_val + 900000)
                            if mask.any():
                                df.loc[mask, 'oi'] = oi_map[ot_val]
            except:
                pass
        except:
            if 'funding' not in df.columns: df['funding'] = 0.0
            if 'oi' not in df.columns: df['oi'] = 0.0
        return df

    async def train_model(self, client, symbol, end_time=None):
        # Train on 1m data since predict() receives 1m candles — must match distribution
        df = await self.fetch_extended_data(client, symbol, interval="1m", total=1500)
        lookahead = 15  # 15 candles forward (~15 min)
        if df is None or len(df) < 500:
            # Fallback: 15m with adjusted lookahead
            df = await self.fetch_extended_data(client, symbol, interval="15m", total=1500)
            lookahead = 10
            if df is None or len(df) < 300: return False
        
        # Inject real funding & OI data
        df = await self._inject_funding_oi(client, symbol, df)
        
        def _train_sync():
            df_sync = self.feature_engineering(df.copy())
            df_sync = self.apply_triple_barrier(df_sync, pt_mult=1.5, sl_mult=1.2, lookahead=lookahead)
            if len(df_sync) < 100: return False
            features = self._get_feature_list(df_sync.columns)
            X, y = df_sync[features], df_sync['target']
            
            # Walk-forward validation: 3 folds, train on past, test on future
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            last_model = None
            for train_idx, test_idx in tscv.split(X):
                m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, num_leaves=25, subsample=0.8, colsample_bytree=0.8, random_state=42, class_weight='balanced', n_jobs=-1, verbose=-1)
                m.fit(X.iloc[train_idx], y.iloc[train_idx])
                acc = (m.predict(X.iloc[test_idx]) == y.iloc[test_idx]).mean()
                scores.append(acc)
                last_model = m
            
            avg_acc = np.mean(scores)
            if avg_acc < 0.52 or last_model is None: return False  # Stricter: must beat 52%
            self.models[symbol], self.last_trained[symbol] = last_model, time.time()
            return True
        return await asyncio.to_thread(_train_sync)

    async def batch_pretrain(self, client, symbols):
        """Pre-train all models in parallel at startup for instant predictions."""
        async def _train_one(sym):
            async with self.retrain_sem:
                await self.train_model(client, sym)
        await asyncio.gather(*[_train_one(s) for s in symbols], return_exceptions=True)
        self.startup_done = True

    async def predict(self, client, symbol, current_df):
        now = time.time()
        if symbol not in self.models:
            if not await self.train_model(client, symbol): return 0.5
            self.trades_since_train[symbol] = 0
        else:
            # Non-blocking check: dispatch retrain if time/drift triggers fire.
            asyncio.create_task(self.maybe_retrain(client, symbol))
        model = self.models.get(symbol)
        if model is None: return 0.5
        df = self.feature_engineering(current_df.tail(100).copy())
        if df.empty: return 0.5
        features = self._get_feature_list(df.columns)
        for f in features:
            if f not in df.columns: return 0.5
        X_live = df[features].iloc[[-1]]
        return model.predict_proba(X_live)[0][1]

ml_predictor = MLPredictor()
