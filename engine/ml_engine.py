import httpx
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
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

    def update_performance(self, symbol, is_win):
        if symbol not in self.performance: self.performance[symbol] = []
        self.performance[symbol].append(is_win)
        if len(self.performance[symbol]) > 50: self.performance[symbol].pop(0)

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

    async def train_model(self, client, symbol, end_time=None):
        # 15m is more predictable (less noise), use as primary training interval
        df = await self.fetch_historical_data(client, symbol, interval="15m", limit=500, end_time=end_time)
        lookahead = 8  # ~2 hours forward
        if df is None or len(df) < 200:
            # Fallback to 1m with 1500 candles
            df = await self.fetch_historical_data(client, symbol, interval="1m", limit=1500, end_time=end_time)
            lookahead = 15
            if df is None or len(df) < 500: return False
        def _train_sync():
            df_sync = self.feature_engineering(df.copy())
            df_sync = self.apply_triple_barrier(df_sync, pt_mult=1.5, sl_mult=1.2, lookahead=lookahead)
            if len(df_sync) < 100: return False
            features = self._get_feature_list(df_sync.columns)
            X, y = df_sync[features], df_sync['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.06, max_depth=5, num_leaves=20, subsample=0.8, colsample_bytree=0.8, random_state=42, class_weight='balanced', n_jobs=-1, verbose=-1)
            model.fit(X_train, y_train)
            # Validate before deploy - reject bad models
            test_acc = (model.predict(X_test) == y_test).mean()
            if test_acc < 0.50: return False  # Model worse than random
            self.models[symbol], self.last_trained[symbol] = model, time.time()
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
        if (now - self.last_trained.get(symbol, 0)) > 43200:
            # Throttle retraining with semaphore (M6)
            async def _retrain_with_sem():
                async with self.retrain_sem:
                    await self.train_model(client, symbol)
            asyncio.create_task(_retrain_with_sem())
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
