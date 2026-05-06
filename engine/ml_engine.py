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

    async def fetch_historical_data(self, client, symbol, interval="15m", limit=1500, end_time=None):
        try:
            # 1. Fetch Klines
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            if end_time:
                params["endTime"] = end_time
            res = await client.get(
                f"{API_URL}/fapi/v1/klines",
                params=params,
                timeout=15
            )
            if res.status_code != 200:
                return None
            
            df = pd.DataFrame(res.json()).iloc[:, [0, 1, 2, 3, 4, 5]]
            df.columns = ["ot", "o", "h", "l", "c", "v"]
            for col in ["ot", "o", "h", "l", "c", "v"]:
                df[col] = df[col].astype(float)
            
            # 2. Fetch Historical Open Interest
            # Map kline interval to OI period (15m, 1h, etc.)
            oi_period = interval if interval in ["5m", "15m", "30m", "1h"] else "15m"
            res_oi = await client.get(
                f"{API_URL}/fapi/v1/openInterestHist",
                params={"symbol": symbol, "period": oi_period, "limit": limit},
                timeout=15
            )
            if res_oi.status_code == 200:
                oi_df = pd.DataFrame(res_oi.json())
                if not oi_df.empty:
                    oi_df['ot'] = oi_df['timestamp'].astype(float)
                    oi_df['oi'] = oi_df['sumOpenInterest'].astype(float)
                    oi_df = oi_df[['ot', 'oi']]
                    df = pd.merge(df, oi_df, on='ot', how='left')
                    df['oi'] = df['oi'].ffill()

            # 3. Fetch Funding Rate
            res_fr = await client.get(
                f"{API_URL}/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": 100}, # Funding is every 8h, 100 is enough
                timeout=15
            )
            if res_fr.status_code == 200:
                fr_df = pd.DataFrame(res_fr.json())
                if not fr_df.empty:
                    fr_df['ot'] = fr_df['fundingTime'].astype(float)
                    fr_df['funding'] = fr_df['fundingRate'].astype(float)
                    fr_df = fr_df[['ot', 'funding']]
                    # Use merge_asof for funding since it's infrequent
                    df = pd.merge_asof(df.sort_values('ot'), fr_df.sort_values('ot'), on='ot', direction='backward')
                    df['funding'] = df['funding'].fillna(0)

            return df
        except Exception as e:
            print(f"Error fetching data for ML {symbol}: {e}")
            return None

    def feature_engineering(self, df):
        # 1. Base Technical Indicators
        df['ema_9'] = ta.ema(df['c'], length=9)
        df['ema_21'] = ta.ema(df['c'], length=21)
        df['rsi'] = ta.rsi(df['c'], length=14)
        df['atr'] = ta.atr(df['h'], df['l'], df['c'], length=14)
        
        macd = ta.macd(df['c'], fast=12, slow=26, signal=9)
        if macd is not None:
            df = pd.concat([df, macd], axis=1)
            
        # Institutional Flow Proxies (VWAP & CVD)
        df['typical_price'] = (df['h'] + df['l'] + df['c']) / 3
        df['vwap'] = (df['typical_price'] * df['v']).cumsum() / df['v'].cumsum()
        df['dist_vwap'] = (df['c'] - df['vwap']) / df['vwap']
        
        # CVD Proxy: Volume multiplied by direction
        df['dir_vol'] = np.where(df['c'] > df['o'], df['v'], -df['v'])
        df['cvd'] = df['dir_vol'].cumsum()
        df['cvd_roc'] = df['cvd'].pct_change(3)
        
        # 2. Institutional Flow (OI & Funding)
        if 'oi' in df.columns:
            df['oi_roc'] = df['oi'].pct_change(3)
        else:
            df['oi_roc'] = 0
            
        if 'funding' not in df.columns:
            df['funding'] = 0

        # 3. Rates of Change (Momentum)
        df['roc_c_1'] = df['c'].pct_change(1)
        df['roc_c_5'] = df['c'].pct_change(5)
        df['roc_v_1'] = df['v'].pct_change(1)
        
        # 4. Volatility & Micro-Structure
        df['volatility'] = (df['h'] - df['l']) / df['c']
        df['body_size'] = abs(df['c'] - df['o']) / df['c']
        df['upper_wick'] = (df['h'] - df[['o', 'c']].max(axis=1)) / df['c']
        df['lower_wick'] = (df[['o', 'c']].min(axis=1) - df['l']) / df['c']
        
        # 5. Distance to EMAs (Mean Reversion / Trend Strength)
        df['dist_ema9'] = (df['c'] - df['ema_9']) / df['ema_9']
        df['dist_ema21'] = (df['c'] - df['ema_21']) / df['ema_21']
        
        # Clean up temporary columns and NaNs
        df.drop(columns=['typical_price', 'dir_vol', 'cvd'], inplace=True, errors='ignore')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df

    def apply_triple_barrier(self, df, pt_mult=1.5, sl_mult=1.0, lookahead=12):
        """
        Triple Barrier Method for labeling.
        Label 1: Hit Profit Target (Upper Barrier) first.
        Label 0: Hit Stop Loss (Lower Barrier) first.
        Time Expiry: Dropped (np.nan) so the model only learns clear trends.
        """
        targets = []
        prices = df['c'].values
        atrs = df['atr'].values
        highs = df['h'].values
        lows = df['l'].values
        
        for i in range(len(df)):
            if i + lookahead >= len(df):
                targets.append(np.nan)
                continue
                
            entry_price = prices[i]
            current_atr = atrs[i]
            
            # Dynamic barriers based on ATR
            pt_price = entry_price + (current_atr * pt_mult)
            sl_price = entry_price - (current_atr * sl_mult)
            
            label = np.nan # Default: Time Expiry (Will be dropped)
            
            # Walk forward 'lookahead' steps
            for j in range(1, lookahead + 1):
                idx = i + j
                if highs[idx] >= pt_price:
                    label = 1
                    break # Hit PT first
                elif lows[idx] <= sl_price:
                    label = 0
                    break # Hit SL first
                    
            targets.append(label)
            
        df['target'] = targets
        df.dropna(subset=['target'], inplace=True)
        return df

    async def train_model(self, client, symbol, end_time=None):
        # Increased limit for better training data
        df = await self.fetch_historical_data(client, symbol, interval="15m", limit=1500, end_time=end_time)
        if df is None or df.empty:
            return False

        def _train_sync():
            df_sync = df.copy()
            df_sync = self.feature_engineering(df_sync)
            df_sync = self.apply_triple_barrier(df_sync, pt_mult=1.5, sl_mult=1.0, lookahead=15)

            if len(df_sync) < 150: # Lowered threshold slightly since we drop more rows now
                return False

            # Prepare features (X) and target (y)
            features = [
                'ema_9', 'ema_21', 'rsi', 'atr', 
                'roc_c_1', 'roc_c_5', 'roc_v_1', 
                'volatility', 'body_size', 'upper_wick', 'lower_wick',
                'dist_ema9', 'dist_ema21', 'dist_vwap', 'cvd_roc',
                'oi_roc', 'funding'
            ]
            features.extend([col for col in df_sync.columns if 'MACD' in col])
            
            X = df_sync[features]
            y = df_sync['target']

            # Walk-forward / chronological split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            def objective(trial):
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'max_depth': trial.suggest_int('max_depth', 3, 6),
                    'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                    'random_state': 42,
                    'n_jobs': 1,
                    'verbose': -1
                }
                model = lgb.LGBMClassifier(**param)
                model.fit(X_train, y_train)
                
                # Predict probabilities and calculate log_loss on the test set
                preds = model.predict_proba(X_test)
                try:
                    loss = log_loss(y_test, preds)
                except ValueError:
                    loss = 10.0 # Penalty if log_loss fails (e.g., single class predicted)
                return loss

            # Run Optuna optimization
            study = optuna.create_study(direction='minimize')
            # Keep trials relatively low to ensure fast live execution
            study.optimize(objective, n_trials=10, timeout=15)
            
            best_params = study.best_params
            best_params['random_state'] = 42
            best_params['n_jobs'] = 1
            best_params['verbose'] = -1

            # Train the final model using the best hyperparameters
            model = lgb.LGBMClassifier(**best_params)
            model.fit(X_train, y_train)
            
            self.models[symbol] = model
            self.last_trained[symbol] = time.time()
            return True

        return await asyncio.to_thread(_train_sync)

    async def predict(self, client, symbol, current_df):
        now = time.time()
        # Retrain if model is missing or older than 12 hours
        if symbol not in self.models or (now - self.last_trained.get(symbol, 0)) > 43200:
            success = await self.train_model(client, symbol)
            if not success:
                return 0.5 

        model = self.models.get(symbol)
        if model is None:
            return 0.5

        df = current_df.copy()
        df = self.feature_engineering(df)
        
        if df.empty:
            return 0.5

        features = [
            'ema_9', 'ema_21', 'rsi', 'atr', 
            'roc_c_1', 'roc_c_5', 'roc_v_1', 
            'volatility', 'body_size', 'upper_wick', 'lower_wick',
            'dist_ema9', 'dist_ema21', 'dist_vwap', 'cvd_roc',
            'oi_roc', 'funding'
        ]
        features.extend([col for col in df.columns if 'MACD' in col])
        
        # Ensure all columns exist before predicting
        for f in features:
            if f not in df.columns: return 0.5

        X_live = df[features].iloc[[-1]]
        
        # Predict probability of hitting PT before SL (UP)
        prob = model.predict_proba(X_live)[0][1]
        return prob

ml_predictor = MLPredictor()
