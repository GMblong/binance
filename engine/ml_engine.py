import httpx
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from utils.config import API_URL, ML_RETRAIN_INTERVAL_SEC
import pandas_ta as ta
import asyncio
import time
import os
import joblib
from pathlib import Path
from strategies.analyzer import MarketAnalyzer

# Limit ML training threads to avoid CPU overload
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


class MLPredictor:
    def __init__(self):
        self.models = {}
        self.last_trained = {}
        self.performance = {}
        self.feature_importance = {}
        self.feature_cache = {}
        self.retrain_sem = asyncio.Semaphore(4)
        self.startup_done = False
        self._retraining = set()
        self.trades_since_train = {}
        self.last_drift_check = {}
        self._pred_cache = {}
        self._pred_cache_ttl = 5.0
        self._load_all_models()

    def _load_all_models(self):
        """Load persisted models on startup."""
        for path in MODEL_DIR.glob("*_ensemble.joblib"):
            symbol = path.stem.replace("_ensemble", "")
            try:
                data = joblib.load(path)
                self.models[symbol] = data["models"]
                self.last_trained[symbol] = data.get("last_trained", 0)
                self.performance[symbol] = data.get("performance", [])
            except Exception:
                pass

    def _save_model(self, symbol):
        """Persist model to disk after training."""
        try:
            joblib.dump({
                "models": self.models[symbol],
                "last_trained": self.last_trained.get(symbol, time.time()),
                "performance": self.performance.get(symbol, []),
            }, MODEL_DIR / f"{symbol}_ensemble.joblib")
        except Exception:
            pass

    def update_performance(self, symbol, is_win):
        if symbol not in self.performance:
            self.performance[symbol] = []
        self.performance[symbol].append(is_win)
        if len(self.performance[symbol]) > 50:
            self.performance[symbol].pop(0)
        self.trades_since_train[symbol] = self.trades_since_train.get(symbol, 0) + 1

    def recent_win_rate(self, symbol, window=15):
        perf = self.performance.get(symbol, [])
        if len(perf) < window:
            return None
        return sum(perf[-window:]) / window

    def should_retrain(self, symbol, now=None):
        now = now or time.time()
        if symbol not in self.models:
            return True
        last = self.last_trained.get(symbol, 0)
        if (now - last) > ML_RETRAIN_INTERVAL_SEC:
            return True
        tst = self.trades_since_train.get(symbol, 0)
        if tst >= 40:
            return True
        wr = self.recent_win_rate(symbol, window=15)
        if wr is not None and wr < 0.45 and tst >= 15:
            return True
        return False

    async def maybe_retrain(self, client, symbol):
        """Background retrain for existing models (time/drift/trade-based)."""
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
            'oi_roc', 'funding',
            'sell_buy_ratio', 'rsi_slope', 'below_emas',
            # Trading analysis signals
            'structure', 'divergence', 'vsa', 'liq_sweep',
            'ob_near', 'fvg', 'wyckoff', 'vol_anomaly',
            'vol_breakout', 'adx', 'ob_imbalance',
        ]
        features.extend([col for col in df_cols if 'MACD' in col])
        return features

    def feature_engineering(self, df):
        is_training = len(df) > 200

        df['ema_9'] = ta.ema(df['c'], length=9)
        df['ema_21'] = ta.ema(df['c'], length=21)
        df['rsi'] = ta.rsi(df['c'], length=14)
        df['atr'] = ta.atr(df['h'], df['l'], df['c'], length=14)

        exp1 = df['c'].ewm(span=12, adjust=False).mean()
        exp2 = df['c'].ewm(span=26, adjust=False).mean()
        df['MACD_12_26_9'] = exp1 - exp2
        df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
        df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']

        if is_training:
            df['ema_60'] = ta.ema(df['c'], length=60)

        # VWAP (use smaller window for prediction to reduce computation)
        vwap_window = 100 if is_training else 60
        df['typical_price'] = (df['h'] + df['l'] + df['c']) / 3
        df['vol_price'] = df['typical_price'] * df['v']
        df['vwap'] = df['vol_price'].rolling(window=vwap_window, min_periods=1).sum() / df['v'].rolling(window=vwap_window, min_periods=1).sum()
        df['dist_vwap'] = (df['c'] - df['vwap']) / (df['vwap'] + 1e-8)
        df.drop(columns=['vol_price'], inplace=True, errors='ignore')

        range_diff = df['h'] - df['l'] + 1e-8
        buy_vol = df['v'] * ((df['c'] - df['l']) / range_diff)
        sell_vol = df['v'] * ((df['h'] - df['c']) / range_diff)
        df['cvd'] = (buy_vol - sell_vol).cumsum()
        df['cvd_roc'] = df['cvd'].pct_change(3).fillna(0)

        if 'oi' not in df.columns:
            df['oi'] = 0
        df['oi_roc'] = df['oi'].pct_change(3).fillna(0)
        if 'funding' not in df.columns:
            df['funding'] = 0

        df['roc_c_1'] = df['c'].pct_change(1)
        df['roc_c_5'] = df['c'].pct_change(5)
        df['roc_v_1'] = df['v'].pct_change(1)
        df['volatility'] = (df['h'] - df['l']) / df['c']
        df['body_size'] = abs(df['c'] - df['o']) / df['c']
        df['upper_wick'] = (df['h'] - df[['o', 'c']].max(axis=1)) / df['c']
        df['lower_wick'] = (df[['o', 'c']].min(axis=1) - df['l']) / df['c']

        df['dist_ema9'] = (df['c'] - df['ema_9']) / (df['ema_9'] + 1e-8)
        df['dist_ema21'] = (df['c'] - df['ema_21']) / (df['ema_21'] + 1e-8)

        # --- FILTER-SYNCED FEATURES (ML learns same signals as entry filters) ---
        # Sell/Buy volume ratio: >1 = sellers dominate, <1 = buyers dominate
        _range = df['h'] - df['l'] + 1e-8
        _buy_v = df['v'] * ((df['c'] - df['l']) / _range)
        _sell_v = df['v'] * ((df['h'] - df['c']) / _range)
        df['sell_buy_ratio'] = (_sell_v.rolling(10).sum() / (_buy_v.rolling(10).sum() + 1e-8)).fillna(1.0)
        
        # RSI slope: negative = declining momentum
        df['rsi_slope'] = (df['rsi'] - df['rsi'].shift(3)).fillna(0)
        
        # Price position vs EMAs: -1 = below both, +1 = above both, 0 = between
        df['below_emas'] = np.where(
            (df['c'] < df['ema_9']) & (df['c'] < df['ema_21']), -1.0,
            np.where((df['c'] > df['ema_9']) & (df['c'] > df['ema_21']), 1.0, 0.0)
        )

        # --- TRADING ANALYSIS SIGNALS AS ML FEATURES ---
        # Structure: detect HH/HL (bullish=1) vs LH/LL (bearish=-1)
        struct_dir, _, _, _ = MarketAnalyzer.detect_structure(df)
        df['structure'] = 1.0 if struct_dir == "BULLISH" else (-1.0 if struct_dir == "BEARISH" else 0.0)

        # RSI Divergence: bullish=1, bearish=-1, none=0
        df['divergence'] = float(MarketAnalyzer.detect_rsi_divergence(df))

        # VSA (Volume Spread Analysis): buy=1, sell=-1, neutral=0
        df['vsa'] = float(MarketAnalyzer.detect_vsa_signals(df))

        # Liquidity sweep: swept lows=1 (bullish), swept highs=-1 (bearish)
        df['liq_sweep'] = float(MarketAnalyzer.detect_liquidity_sweep(df))

        # Order block proximity: 1 if near OB in direction, 0 otherwise
        last_price = df['c'].iloc[-1]
        ob_bull = MarketAnalyzer.find_nearest_order_block(df, last_price, 1)
        ob_bear = MarketAnalyzer.find_nearest_order_block(df, last_price, -1)
        df['ob_near'] = 1.0 if ob_bull else (-1.0 if ob_bear else 0.0)

        # FVG (Fair Value Gap): bullish=1, bearish=-1, none=0
        fvg = MarketAnalyzer.get_nearest_fvg(df)
        df['fvg'] = (1.0 if fvg and fvg["type"] == "BULLISH" else
                     (-1.0 if fvg and fvg["type"] == "BEARISH" else 0.0))

        # Wyckoff phase: ACCUMULATION=1, MARKUP=2, DISTRIBUTION=-1, MARKDOWN=-2
        wyckoff = MarketAnalyzer.detect_wyckoff_phase(df)
        wyckoff_map = {"ACCUMULATION": 1.0, "MARKUP": 2.0, "DISTRIBUTION": -1.0, "MARKDOWN": -2.0}
        df['wyckoff'] = wyckoff_map.get(wyckoff, 0.0)

        # Volume anomaly: 1 if abnormal volume detected
        df['vol_anomaly'] = 1.0 if MarketAnalyzer.detect_volume_anomaly(df) else 0.0

        # Volatility breakout: 1 if breaking out
        df['vol_breakout'] = 1.0 if MarketAnalyzer.detect_volatility_breakout(df) else 0.0

        # ADX (trend strength)
        df['adx'] = MarketAnalyzer.get_adx(df, 14)

        # Orderbook imbalance from market_data (if available)
        from utils.state import market_data
        df['ob_imbalance'] = market_data.imbalance.get(df.attrs.get('symbol', ''), 1.0)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.drop(columns=['typical_price', 'dir_vol', 'cvd'], inplace=True, errors='ignore')

        if not is_training:
            return df.tail(1)

        features = self._get_feature_list(df.columns)
        df.dropna(subset=features, inplace=True)
        return df

    def apply_triple_barrier(self, df, pt_mult=1.5, sl_mult=1.0, lookahead=12):
        targets = []
        prices, atrs, highs, lows = df['c'].values, df['atr'].values, df['h'].values, df['l'].values
        recent_median_atr = df['atr'].rolling(100).median().bfill().values

        for i in range(len(df)):
            if i + lookahead >= len(df):
                targets.append(np.nan)
                continue
            entry_price, current_atr, base_atr = prices[i], atrs[i], recent_median_atr[i]
            vol_ratio = current_atr / (base_atr + 1e-8)
            dyn_pt_mult = pt_mult * min(max(vol_ratio, 0.8), 2.0)
            dyn_sl_mult = sl_mult * min(max(vol_ratio, 0.8), 2.0)
            pt_price = entry_price + (current_atr * dyn_pt_mult)
            sl_price = entry_price - (current_atr * dyn_sl_mult)
            label = 0
            for j in range(1, lookahead + 1):
                idx = i + j
                if highs[idx] >= pt_price:
                    label = 1
                    break
                elif lows[idx] <= sl_price:
                    label = 0
                    break
            targets.append(label)
        df['target'] = targets
        df.dropna(subset=['target'], inplace=True)
        return df

    async def fetch_historical_data(self, client, symbol, interval="15m", limit=300, end_time=None):
        try:
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            if end_time:
                params["endTime"] = end_time
            res = await client.get(f"{API_URL}/fapi/v1/klines", params=params, timeout=15)
            if res.status_code != 200:
                return None
            df = pd.DataFrame(res.json()).iloc[:, [0, 1, 2, 3, 4, 5]]
            df.columns = ["ot", "o", "h", "l", "c", "v"]
            for col in ["ot", "o", "h", "l", "c", "v"]:
                df[col] = df[col].astype(float)
            return df
        except Exception:
            return None

    async def fetch_extended_data(self, client, symbol, interval="15m", total=1500):
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
        return pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=['ot']).sort_values('ot').reset_index(drop=True)

    async def _inject_funding_oi(self, client, symbol, df):
        try:
            res = await client.get(f"{API_URL}/fapi/v1/fundingRate", params={"symbol": symbol, "limit": 100}, timeout=10)
            funding_map = {}
            if res and res.status_code == 200:
                for f in res.json():
                    funding_map[int(f['fundingTime'])] = float(f['fundingRate'])
            if funding_map:
                fund_times = sorted(funding_map.keys())
                df['funding'] = 0.0
                for ft in fund_times:
                    mask = df['ot'] <= ft
                    if mask.any():
                        idx = mask[mask].index[-1]
                        df.loc[idx:, 'funding'] = funding_map[ft]

            from utils.state import market_data
            live_oi = market_data.oi.get(symbol, 0)
            df['oi'] = live_oi if live_oi > 0 else 0

            try:
                oi_res = await client.get(f"{API_URL}/futures/data/openInterestHist",
                    params={"symbol": symbol, "period": "15m", "limit": 100}, timeout=10)
                if oi_res and oi_res.status_code == 200:
                    oi_data = oi_res.json()
                    oi_map = {int(x['timestamp']): float(x['sumOpenInterest']) for x in oi_data}
                    if oi_map:
                        for ot_val in sorted(oi_map.keys()):
                            mask = (df['ot'] >= ot_val - 900000) & (df['ot'] <= ot_val + 900000)
                            if mask.any():
                                df.loc[mask, 'oi'] = oi_map[ot_val]
            except Exception:
                pass
        except Exception:
            if 'funding' not in df.columns:
                df['funding'] = 0.0
            if 'oi' not in df.columns:
                df['oi'] = 0.0
        return df

    async def train_model(self, client, symbol, end_time=None):
        # Try 1m first (more data points, better for scalping)
        df = await self.fetch_extended_data(client, symbol, interval="1m", total=1500)
        lookahead = 15
        if df is not None and len(df) >= 200:
            df = await self._inject_funding_oi(client, symbol, df)
            ok = await asyncio.to_thread(self._train_sync, df, symbol, lookahead)
            if ok:
                return True
        # Fallback to 15m (smoother data, better patterns for major coins)
        df = await self.fetch_extended_data(client, symbol, interval="15m", total=1500)
        lookahead = 10
        if df is None or len(df) < 100:
            return False
        df = await self._inject_funding_oi(client, symbol, df)
        return await asyncio.to_thread(self._train_sync, df, symbol, lookahead)

    def _train_sync(self, df, symbol, lookahead):
        df_sync = self.feature_engineering(df.copy())
        df_sync = self.apply_triple_barrier(df_sync, pt_mult=1.5, sl_mult=1.2, lookahead=lookahead)
        if len(df_sync) < 60:
            return False

        features = self._get_feature_list(df_sync.columns)
        X, y = df_sync[features], df_sync['target']

        tscv = TimeSeriesSplit(n_splits=2)
        scores_lgb, scores_xgb, scores_mlp, scores_ens = [], [], [], []
        last_lgb, last_xgb, last_mlp, last_scaler = None, None, None, None

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            # --- LightGBM ---
            m_lgb = lgb.LGBMClassifier(
                n_estimators=80, learning_rate=0.08, max_depth=5,
                num_leaves=20, subsample=0.8, colsample_bytree=0.8,
                class_weight='balanced', n_jobs=2, verbose=-1, random_state=42
            )
            m_lgb.fit(X_tr, y_tr)
            acc_lgb = (m_lgb.predict(X_te) == y_te).mean()
            scores_lgb.append(acc_lgb)

            # --- XGBoost ---
            m_xgb = xgb.XGBClassifier(
                n_estimators=80, learning_rate=0.08, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
                use_label_encoder=False, eval_metric='logloss',
                n_jobs=2, verbosity=0, random_state=42
            )
            m_xgb.fit(X_tr, y_tr)
            acc_xgb = (m_xgb.predict(X_te) == y_te).mean()
            scores_xgb.append(acc_xgb)

            # --- MLP (Neural Net) ---
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            m_mlp = MLPClassifier(
                hidden_layer_sizes=(32, 16), max_iter=150,
                learning_rate='adaptive', early_stopping=True,
                validation_fraction=0.15, random_state=42
            )
            m_mlp.fit(X_tr_s, y_tr)
            acc_mlp = (m_mlp.predict(X_te_s) == y_te).mean()
            scores_mlp.append(acc_mlp)

            # --- Ensemble Vote ---
            p_lgb = m_lgb.predict_proba(X_te)[:, 1]
            p_xgb = m_xgb.predict_proba(X_te)[:, 1]
            p_mlp = m_mlp.predict_proba(X_te_s)[:, 1]
            p_ens = (p_lgb + p_xgb + p_mlp) / 3.0
            acc_ens = ((p_ens >= 0.5).astype(int) == y_te).mean()
            scores_ens.append(acc_ens)

            last_lgb, last_xgb, last_mlp, last_scaler = m_lgb, m_xgb, m_mlp, scaler

        avg_ens = np.mean(scores_ens)
        if avg_ens < 0.48 or last_lgb is None:
            return False

        self.models[symbol] = {
            'lgb': last_lgb,
            'xgb': last_xgb,
            'mlp': last_mlp,
            'scaler': last_scaler
        }
        self.last_trained[symbol] = time.time()
        self._save_model(symbol)
        return True

    async def batch_pretrain(self, client, symbols):
        async def _train_one(sym):
            async with self.retrain_sem:
                await self.train_model(client, sym)
        await asyncio.gather(*[_train_one(s) for s in symbols], return_exceptions=True)
        self.startup_done = True

    async def predict(self, client, symbol, current_df):
        # Check prediction cache first
        if not current_df.empty:
            last_ot = float(current_df.iloc[-1]['ot']) if 'ot' in current_df.columns else 0
            cached = self._pred_cache.get(symbol)
            if cached and cached[0] == last_ot and (time.time() - cached[2]) < self._pred_cache_ttl:
                return cached[1]
        
        if symbol not in self.models:
            # Train inline only if semaphore is available (non-blocking check)
            if symbol not in self._retraining:
                if self.retrain_sem._value > 0:  # Semaphore has capacity
                    self._retraining.add(symbol)
                    try:
                        async with self.retrain_sem:
                            await self.train_model(client, symbol)
                        self.trades_since_train[symbol] = 0
                    finally:
                        self._retraining.discard(symbol)
                else:
                    # Semaphore full (batch retrain running), queue background
                    asyncio.create_task(self.maybe_retrain(client, symbol))
            # If still no model after training attempt, return neutral
            if symbol not in self.models:
                return 0.5
        else:
            asyncio.create_task(self.maybe_retrain(client, symbol))

        ensemble = self.models.get(symbol)
        if ensemble is None:
            return 0.5

        df = self.feature_engineering(current_df.tail(100).copy())
        if df.empty:
            return 0.5

        features = self._get_feature_list(df.columns)
        for f in features:
            if f not in df.columns:
                return 0.5

        X_live = df[features].iloc[[-1]]

        # --- Ensemble Prediction (Weighted Average) ---
        try:
            p_lgb = ensemble['lgb'].predict_proba(X_live)[0][1]
        except Exception:
            p_lgb = 0.5
        try:
            p_xgb = ensemble['xgb'].predict_proba(X_live)[0][1]
        except Exception:
            p_xgb = 0.5
        try:
            X_scaled = ensemble['scaler'].transform(X_live)
            p_mlp = ensemble['mlp'].predict_proba(X_scaled)[0][1]
        except Exception:
            p_mlp = 0.5

        # Weighted vote: LGB 0.4, XGB 0.35, MLP 0.25
        final_prob = (p_lgb * 0.4) + (p_xgb * 0.35) + (p_mlp * 0.25)

        # Consensus bonus: if all 3 agree strongly, boost confidence
        all_probs = [p_lgb, p_xgb, p_mlp]
        agreement = 1.0 - np.std(all_probs) * 2
        if agreement > 0.85:
            final_prob = 0.5 + (final_prob - 0.5) * 1.15

        result = max(0.0, min(1.0, final_prob))
        
        # Cache the prediction
        last_ot = float(current_df.iloc[-1]['ot']) if not current_df.empty and 'ot' in current_df.columns else 0
        self._pred_cache[symbol] = (last_ot, result, time.time())
        return result


ml_predictor = MLPredictor()
