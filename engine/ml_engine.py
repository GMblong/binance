import httpx
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import optuna
from utils.config import API_URL
import pandas_ta as ta
import asyncio
import time

optuna.logging.set_verbosity(optuna.logging.WARNING)

class MLPredictor:
    def __init__(self):
        self.models = {}  # Store models per symbol
        self.last_trained = {} # Store last training time per symbol
        self.performance = {} # Store win rate tracking (symbol: [list of bool])
        self.feature_importance = {} # Store feature importance per symbol
        self.feature_cache = {} # Cache for pre-calculated features {symbol: df}

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
            
        # Institutional Flow Proxies (VWAP & CVD)
        df['typical_price'] = (df['h'] + df['l'] + df['c']) / 3
        df['vwap'] = (df['typical_price'] * df['v']).cumsum() / df['v'].cumsum()
        df['dist_vwap'] = (df['c'] - df['vwap']) / df['vwap']
        
        # CVD Proxy
        df['dir_vol'] = np.where(df['c'] > df['o'], df['v'], -df['v'])
        df['cvd'] = df['dir_vol'].cumsum()
        df['cvd_roc'] = df['cvd'].pct_change(3)
        
        # 2. Institutional Flow (OI & Funding)
        if 'oi' not in df.columns: df['oi'] = 0
        df['oi_roc'] = df['oi'].pct_change(3)
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
        
        df.dropna(inplace=True)
        return df

    def apply_triple_barrier(self, df, pt_mult=1.5, sl_mult=1.0, lookahead=12):
        targets = []
        prices, atrs, highs, lows = df['c'].values, df['atr'].values, df['h'].values, df['l'].values
        for i in range(len(df)):
            if i + lookahead >= len(df):
                targets.append(np.nan)
                continue
            entry_price, current_atr = prices[i], atrs[i]
            pt_price, sl_price = entry_price + (current_atr * pt_mult), entry_price - (current_atr * sl_mult)
            label = np.nan
            for j in range(1, lookahead + 1):
                idx = i + j
                if highs[idx] >= pt_price: label = 1; break
                elif lows[idx] <= sl_price: label = 0; break
            targets.append(label)
        df['target'] = targets
        df.dropna(subset=['target'], inplace=True)
        return df

    async def fetch_historical_data(self, client, symbol, interval="15m", limit=1500, end_time=None):
        try:
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            if end_time: params["endTime"] = end_time
            res = await client.get(f"{API_URL}/fapi/v1/klines", params=params, timeout=10)
            if res.status_code != 200: return None
            df = pd.DataFrame(res.json()).iloc[:, [0, 1, 2, 3, 4, 5]]
            df.columns = ["ot", "o", "h", "l", "c", "v"]
            for col in ["ot", "o", "h", "l", "c", "v"]: df[col] = df[col].astype(float)
            if limit > 200:
                res_oi = await client.get(f"{API_URL}/fapi/v1/openInterestHist", params={"symbol": symbol, "period": "15m", "limit": 500}, timeout=10)
                if res_oi.status_code == 200:
                    oi_df = pd.DataFrame(res_oi.json())
                    if not oi_df.empty:
                        oi_df['ot'], oi_df['oi'] = oi_df['timestamp'].astype(float), oi_df['sumOpenInterest'].astype(float)
                        df = pd.merge(df, oi_df[['ot', 'oi']], on='ot', how='left').ffill()
            return df
        except: return None

    async def train_model(self, client, symbol, end_time=None):
        df = await self.fetch_historical_data(client, symbol, interval="1m", limit=1500, end_time=end_time)
        if df is None or len(df) < 500: return False
        def _train_sync():
            df_sync = self.feature_engineering(df.copy())
            df_sync = self.apply_triple_barrier(df_sync, pt_mult=1.5, sl_mult=1.2, lookahead=15)
            if len(df_sync) < 200: return False
            features = self._get_feature_list(df_sync.columns)
            X, y = df_sync[features], df_sync['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
            model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, num_leaves=15, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2, verbose=-1)
            model.fit(X_train, y_train)
            self.models[symbol], self.last_trained[symbol] = model, time.time()
            return True
        return await asyncio.to_thread(_train_sync)

    async def predict(self, client, symbol, current_df):
        now = time.time()
        if symbol not in self.models:
            if not await self.train_model(client, symbol): return 0.5
        if (now - self.last_trained.get(symbol, 0)) > 43200:
            asyncio.create_task(self.train_model(client, symbol))
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
