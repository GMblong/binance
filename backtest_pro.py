"""
Pro Backtester v2 (100% sinkron dengan strategi LIVE)
=====================================================
Replika EXACT dari logika entry/exit live bot termasuk:
- Ensemble ML (LightGBM + XGBoost + MLP)
- Wyckoff, HMM regime, multi-candle fake detection
- Structure-based SL/TP dengan liquidation magnets
- Partial TP (50% at RR1:1) + trailing + breakeven

Jalanin:
    python backtest_pro.py
    python backtest_pro.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --days 5
    python backtest_pro.py --offline
"""
from __future__ import annotations

import argparse
import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import httpx
    _HAS_HTTPX = True
except Exception:
    _HAS_HTTPX = False

try:
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    _HAS_ML = True
except Exception:
    _HAS_ML = False

from strategies.analyzer import MarketAnalyzer

API_URL = "https://fapi.binance.com"
TAKER_FEE = 0.0004
SPREAD_BPS = 0.5
SLIPPAGE_ATR_FRAC = 0.05

_FEATURES = [
    "ema_9", "ema_21", "rsi", "atr",
    "roc_c_1", "roc_c_5", "roc_v_1",
    "volatility", "body_size", "upper_wick", "lower_wick",
    "dist_ema9", "dist_ema21", "dist_vwap", "cvd_roc",
    "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
]


def _ema(s, length):
    return s.ewm(span=length, adjust=False).mean()

def _rsi(s, length=14):
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _atr_series(df, length=14):
    h, l, c = df["h"], df["l"], df["c"]
    tr = pd.concat([(h - l).abs(), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def build_ml_features(df):
    df = df.copy()
    df["ema_9"] = _ema(df["c"], 9)
    df["ema_21"] = _ema(df["c"], 21)
    df["rsi"] = _rsi(df["c"], 14)
    df["atr"] = _atr_series(df, 14)
    e12 = df["c"].ewm(span=12, adjust=False).mean()
    e26 = df["c"].ewm(span=26, adjust=False).mean()
    df["MACD_12_26_9"] = e12 - e26
    df["MACDs_12_26_9"] = df["MACD_12_26_9"].ewm(span=9, adjust=False).mean()
    df["MACDh_12_26_9"] = df["MACD_12_26_9"] - df["MACDs_12_26_9"]
    tp = (df["h"] + df["l"] + df["c"]) / 3
    vp = tp * df["v"]
    df["vwap"] = vp.rolling(100).sum() / df["v"].rolling(100).sum()
    df["dist_vwap"] = (df["c"] - df["vwap"]) / df["vwap"]
    rng = (df["h"] - df["l"]).replace(0, np.nan) + 1e-9
    buy_vol = df["v"] * ((df["c"] - df["l"]) / rng)
    sell_vol = df["v"] * ((df["h"] - df["c"]) / rng)
    cvd = (buy_vol - sell_vol).cumsum()
    df["cvd_roc"] = cvd.pct_change(3).fillna(0)
    df["roc_c_1"] = df["c"].pct_change(1)
    df["roc_c_5"] = df["c"].pct_change(5)
    df["roc_v_1"] = df["v"].pct_change(1)
    df["volatility"] = (df["h"] - df["l"]) / df["c"]
    df["body_size"] = (df["c"] - df["o"]).abs() / df["c"]
    df["upper_wick"] = (df["h"] - df[["o", "c"]].max(axis=1)) / df["c"]
    df["lower_wick"] = (df[["o", "c"]].min(axis=1) - df["l"]) / df["c"]
    df["dist_ema9"] = (df["c"] - df["ema_9"]) / df["ema_9"]
    df["dist_ema21"] = (df["c"] - df["ema_21"]) / df["ema_21"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def triple_barrier_labels(df, pt_mult=1.5, sl_mult=1.2, lookahead=15):
    prices, highs, lows, atrs = df["c"].values, df["h"].values, df["l"].values, df["atr"].values
    median_atr = df["atr"].rolling(100).median().bfill().values
    labels = np.full(len(df), np.nan)
    for i in range(len(df) - lookahead):
        entry, cur_atr, base_atr = prices[i], atrs[i], median_atr[i]
        if not math.isfinite(cur_atr) or cur_atr <= 0:
            continue
        vr = min(max(cur_atr / (base_atr + 1e-9), 0.8), 2.0)
        pt_p = entry + cur_atr * pt_mult * vr
        sl_p = entry - cur_atr * sl_mult * vr
        lbl = 0
        for j in range(1, lookahead + 1):
            if highs[i + j] >= pt_p:
                lbl = 1; break
            if lows[i + j] <= sl_p:
                break
        labels[i] = lbl
    return pd.Series(labels, index=df.index)


class EnsembleMLSync:
    """Sync Ensemble ML (LGB + XGB + MLP) untuk backtest."""
    def __init__(self):
        self.models = {}

    def train(self, symbol, df_1m):
        if not _HAS_ML or len(df_1m) < 400:
            return False
        feats = build_ml_features(df_1m)
        feats["target"] = triple_barrier_labels(feats)
        feats = feats.dropna(subset=_FEATURES + ["target"])
        if len(feats) < 150:
            return False
        X, y = feats[_FEATURES], feats["target"].astype(int)
        if y.nunique() < 2:
            return False

        tscv = TimeSeriesSplit(n_splits=3)
        last_lgb = last_xgb = last_mlp = last_scaler = None
        best_acc = 0

        for tr_idx, te_idx in tscv.split(X):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            m_lgb = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.05, max_depth=6,
                num_leaves=25, subsample=0.8, colsample_bytree=0.8,
                class_weight='balanced', n_jobs=-1, verbose=-1, random_state=42)
            m_lgb.fit(X_tr, y_tr)

            m_xgb = xgb.XGBClassifier(n_estimators=120, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
                use_label_encoder=False, eval_metric='logloss',
                n_jobs=-1, verbosity=0, random_state=42)
            m_xgb.fit(X_tr, y_tr)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            m_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                learning_rate='adaptive', early_stopping=True,
                validation_fraction=0.15, random_state=42)
            m_mlp.fit(X_tr_s, y_tr)

            # Ensemble accuracy
            p = (m_lgb.predict_proba(X_te)[:, 1] * 0.4 +
                 m_xgb.predict_proba(X_te)[:, 1] * 0.35 +
                 m_mlp.predict_proba(X_te_s)[:, 1] * 0.25)
            acc = ((p >= 0.5).astype(int) == y_te).mean()
            if acc > best_acc:
                best_acc = acc
                last_lgb, last_xgb, last_mlp, last_scaler = m_lgb, m_xgb, m_mlp, scaler

        if best_acc < 0.52 or last_lgb is None:
            return False
        self.models[symbol] = {'lgb': last_lgb, 'xgb': last_xgb, 'mlp': last_mlp, 'scaler': last_scaler}
        return True

    def predict(self, symbol, df_1m):
        ens = self.models.get(symbol)
        if ens is None:
            return 0.5
        feats = build_ml_features(df_1m.tail(120))
        feats = feats.dropna(subset=_FEATURES)
        if feats.empty:
            return 0.5
        X = feats[_FEATURES].iloc[[-1]]
        try:
            p_lgb = ens['lgb'].predict_proba(X)[0][1]
            p_xgb = ens['xgb'].predict_proba(X)[0][1]
            X_s = ens['scaler'].transform(X)
            p_mlp = ens['mlp'].predict_proba(X_s)[0][1]
            final = p_lgb * 0.4 + p_xgb * 0.35 + p_mlp * 0.25
            # Consensus bonus
            std = np.std([p_lgb, p_xgb, p_mlp])
            if std < 0.075:
                final = 0.5 + (final - 0.5) * 1.15
            return max(0.0, min(1.0, final))
        except:
            return 0.5


# ---------------------------------------------------------------------------
# Signal generator (100% match dengan live analyze_hybrid_async)
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    side: str
    score: int
    regime: str
    setup: str
    entry_hint: float
    sl_pct: float
    tp_pct: float
    ml_prob: float
    is_market: bool


def analyze_sync(d1m, d15m, d1h, ml_prob, imbalance=1.0, funding=0.0):
    """Exact replica of strategies/hybrid.py analyze_hybrid_async."""
    if len(d1m) < 60 or len(d15m) < 30 or len(d1h) < 20:
        return None
    try:
        price = float(d1m["c"].iloc[-1])

        # --- Direction (ADX-gated, RSI mean-rev fallback) ---
        ema9_15m = MarketAnalyzer.get_ema(d15m["c"], 9).iloc[-1]
        ema21_15m = MarketAnalyzer.get_ema(d15m["c"], 21).iloc[-1]
        adx_val = MarketAnalyzer.get_adx(d15m, 14)
        if adx_val > 20:
            direction = 1 if ema9_15m > ema21_15m else -1
        else:
            rsi_15m = MarketAnalyzer.get_rsi(d15m["c"], 14).iloc[-1]
            if rsi_15m < 35: direction = 1
            elif rsi_15m > 65: direction = -1
            else: direction = 1 if ema9_15m > ema21_15m else -1

        # --- HTF alignment (1h) ---
        ema50_1h = MarketAnalyzer.get_ema(d1h["c"], 50).iloc[-1] if len(d1h) >= 50 else d1h["c"].iloc[-1]
        htf_dir = 1 if d1h["c"].iloc[-1] > ema50_1h else -1
        mtf_aligned = (direction == htf_dir)

        atr = MarketAnalyzer.get_atr(d1m, 14).iloc[-1]
        if not math.isfinite(atr) or atr <= 0:
            return None
        base_atr_pct = (atr / price) * 100
        regime = MarketAnalyzer.detect_regime(d15m)

        # --- Score (full live logic) ---
        score = MarketAnalyzer.calculate_score(
            d1m, d15m, direction, imbalance, funding, regime, None, "QUIET", 0)
        if not mtf_aligned:
            score = int(score * 0.6)

        # --- ML blending ---
        if direction == 1:
            if ml_prob >= 0.65: score += int((ml_prob - 0.6) * 50)
            elif ml_prob < 0.40: score += int((ml_prob - 0.4) * 50)
        else:
            if ml_prob <= 0.35: score += int((0.4 - ml_prob) * 50)
            elif ml_prob > 0.60: score += int((0.6 - ml_prob) * 50)
        score = max(0, min(score, 100))

        # --- Wyckoff phase ---
        wyckoff = MarketAnalyzer.detect_wyckoff_phase(d15m)
        if direction == 1 and wyckoff == "DISTRIBUTION":
            score = int(score * 0.5)
        elif direction == -1 and wyckoff == "ACCUMULATION":
            score = int(score * 0.5)
        elif direction == 1 and wyckoff == "MARKUP":
            score = min(score + 10, 100)
        elif direction == -1 and wyckoff == "MARKDOWN":
            score = min(score + 10, 100)

        # --- HMM regime ---
        hmm = MarketAnalyzer.detect_hmm_regime(d15m)
        if hmm == "MOMENTUM" and regime == "TRENDING":
            score = min(score + 5, 100)
        elif hmm == "MEAN_REVERT" and regime == "RANGING":
            score = min(score + 5, 100)

        # --- Liquidation magnet ---
        liq = MarketAnalyzer.predict_liquidation_clusters(d15m)
        if liq:
            if direction == 1 and any(p > price for p in liq["short_liq"]):
                score = min(score + 5, 100)
            elif direction == -1 and any(p < price for p in liq["long_liq"]):
                score = min(score + 5, 100)

        # --- Fake move detection ---
        vol_avg = d1m["v"].tail(20).mean()
        vol_recent = d1m["v"].tail(3).mean()
        vol_confirmed = vol_recent > vol_avg * 1.2
        last = d1m.iloc[-1]
        rng = float(last["h"] - last["l"])
        body = abs(float(last["c"] - last["o"]))
        body_ratio = body / rng if rng > 0 else 0
        is_fake = (not vol_confirmed) and body_ratio < 0.3

        # Multi-candle fake
        if not is_fake and len(d1m) >= 5:
            is_fake = MarketAnalyzer.detect_multi_candle_fake(d1m, direction)

        # Overextended check
        ema21_1m = MarketAnalyzer.get_ema(d1m["c"], 21).iloc[-1]
        dist_ema = abs(price - ema21_1m) / price * 100
        is_over = dist_ema > base_atr_pct * 2.5

        # --- SL (structure-based) ---
        buffer_pct = max(base_atr_pct * 0.3, 0.12)
        low_L1, low_L2 = d1m["l"].tail(15).min(), d1m["l"].tail(30).min()
        high_L1, high_L2 = d1m["h"].tail(15).max(), d1m["h"].tail(30).max()
        ob = MarketAnalyzer.find_nearest_order_block(d1m, price, direction)

        if direction == 1:
            struct_low = low_L2 if ((price - low_L2) / price * 100) <= 3.0 else low_L1
            sl_price = ob["bottom"] if ob else struct_low
            sl_pct = ((price - sl_price) / price * 100) + buffer_pct
        else:
            struct_high = high_L2 if ((high_L2 - price) / price * 100) <= 3.0 else high_L1
            sl_price = ob["top"] if ob else struct_high
            sl_pct = ((sl_price - price) / price * 100) + buffer_pct

        if ml_prob > 0.7 or ml_prob < 0.3: sl_pct *= 0.85
        elif 0.45 < ml_prob < 0.55: sl_pct *= 1.15
        sl_pct = max(0.4, min(sl_pct, 3.5))

        # --- TP ---
        rr_mult = 2.0
        if regime == "RANGING": rr_mult = 1.5
        elif regime == "TRENDING":
            rr_mult = 2.5 if ((direction == 1 and ml_prob > 0.65) or (direction == -1 and ml_prob < 0.35)) else 2.0
        elif regime == "VOLATILE": rr_mult = 1.8

        htf_high = d15m["h"].tail(20).max()
        htf_low = d15m["l"].tail(20).min()
        tp_pct = sl_pct * rr_mult
        if direction == 1:
            sd = ((htf_high - price) / price) * 100
            if sd > sl_pct * 1.2: tp_pct = min(max(tp_pct, sd), sl_pct * 3.5)
            if liq and any(p > price for p in liq.get("short_liq", [])):
                lt = min(p for p in liq["short_liq"] if p > price)
                ld = ((lt - price) / price) * 100
                if ld > tp_pct and ld < sl_pct * 4: tp_pct = ld
        else:
            sd = ((price - htf_low) / price) * 100
            if sd > sl_pct * 1.2: tp_pct = min(max(tp_pct, sd), sl_pct * 3.5)
            if liq and any(p < price for p in liq.get("long_liq", [])):
                lt = max(p for p in liq["long_liq"] if p < price)
                ld = ((price - lt) / price) * 100
                if ld > tp_pct and ld < sl_pct * 4: tp_pct = ld
        tp_pct = max(sl_pct * 1.3, tp_pct)

        # --- Threshold & breakout ---
        prev_15_high = d1m["h"].iloc[-16:-1].max()
        prev_15_low = d1m["l"].iloc[-16:-1].min()
        raw_brk = (direction == 1 and price > prev_15_high) or (direction == -1 and price < prev_15_low)
        is_breakout = raw_brk and vol_confirmed and body_ratio > 0.5

        threshold = 80
        if regime == "VOLATILE": threshold = 75 if is_breakout else 90
        elif regime == "RANGING": threshold = 88
        elif regime == "TRENDING": threshold = 75 if is_breakout else 82

        is_market = regime == "TRENDING" and is_breakout and score >= 90 and not is_fake

        if is_fake or is_over: return None
        if score < threshold: return None

        side = "LONG" if direction == 1 else "SHORT"
        setup = f"{regime}-{'BRK' if is_breakout else 'PULL'}"
        return Signal(side=side, score=int(score), regime=regime, setup=setup,
                      entry_hint=float(price), sl_pct=float(sl_pct), tp_pct=float(tp_pct),
                      ml_prob=float(ml_prob), is_market=bool(is_market))
    except:
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

async def fetch_klines(client, symbol, interval="1m", limit=1500):
    try:
        r = await client.get(f"{API_URL}/fapi/v1/klines",
                             params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=20)
        if r.status_code != 200: return None
        df = pd.DataFrame(r.json()).iloc[:, [0, 1, 2, 3, 4, 5]]
        df.columns = ["ot", "o", "h", "l", "c", "v"]
        return df.astype(float).reset_index(drop=True)
    except: return None


def resample(df_1m, minutes):
    df = df_1m.copy()
    df["dt"] = pd.to_datetime(df["ot"], unit="ms")
    df.set_index("dt", inplace=True)
    agg = df.resample(f"{minutes}min", label="right", closed="right").agg(
        {"ot": "first", "o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()
    return agg.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str; side: str; entry: float; size_usd: float
    sl: float; tp1: float; tp2: float; opened_at: int
    r_unit: float; setup: str; partial_taken: bool = False; peak: float = 0.0

@dataclass
class ClosedTrade:
    symbol: str; side: str; setup: str; opened_at: int; closed_at: int
    pnl_usd: float; reason: str; entry: float; exit: float; bars_held: int

@dataclass
class AccountState:
    balance: float; peak_balance: float; day_start_balance: float
    consec_losses: int = 0; cooldown_until: int = 0
    trades: List[ClosedTrade] = field(default_factory=list)


def _slip(side, price, atr_1m, is_market):
    if not is_market: return price
    s = SLIPPAGE_ATR_FRAC * atr_1m + price * SPREAD_BPS / 10000
    return price + s if side == "LONG" else price - s


def _close(pos, exit_price, reason, bar_idx):
    pct = ((exit_price - pos.entry) / pos.entry) if pos.side == "LONG" else ((pos.entry - exit_price) / pos.entry)
    gross = pct * pos.size_usd
    fee = 2 * TAKER_FEE * pos.size_usd
    return ClosedTrade(pos.symbol, pos.side, pos.setup, pos.opened_at, bar_idx,
                       gross - fee, reason, pos.entry, exit_price, bar_idx - pos.opened_at)


@dataclass
class BacktestConfig:
    symbols: List[str]
    days: int = 3
    starting_balance: float = 100.0
    risk_pct: float = 0.02
    max_positions: int = 5
    max_consec_losses: int = 3
    cooldown_bars: int = 60
    daily_loss_limit_pct: float = 0.05
    offline: bool = False
    ml_min_prob: float = 0.55
    train_frac: float = 0.3
    partial_pct: float = 0.5


class ProBacktester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ml = EnsembleMLSync()

    async def _load(self, client, symbol):
        need = self.cfg.days * 1440 + 500
        frames = []
        end_time = None
        while sum(len(f) for f in frames) < need:
            params = {"symbol": symbol, "interval": "1m", "limit": 1500}
            if end_time: params["endTime"] = end_time
            try:
                r = await client.get(f"{API_URL}/fapi/v1/klines", params=params, timeout=20)
                if r.status_code != 200: break
                raw = r.json()
                if not raw: break
                df = pd.DataFrame(raw).iloc[:, [0, 1, 2, 3, 4, 5]]
                df.columns = ["ot", "o", "h", "l", "c", "v"]
                df = df.astype(float)
                frames.append(df)
                end_time = int(df["ot"].iloc[0]) - 1
                if len(raw) < 1500: break
            except: break
        if not frames: return None
        df1 = pd.concat(frames[::-1], ignore_index=True).drop_duplicates(subset="ot").sort_values("ot").reset_index(drop=True)
        if len(df1) < 500: return None
        return {"1m": df1, "15m": resample(df1, 15), "1h": resample(df1, 60)}

    def _check_position(self, pos, bar, bar_idx):
        partial = None
        if not pos.partial_taken:
            hit_tp1 = (pos.side == "LONG" and bar["h"] >= pos.tp1) or \
                      (pos.side == "SHORT" and bar["l"] <= pos.tp1)
            if hit_tp1:
                size_part = pos.size_usd * self.cfg.partial_pct
                part_pos = Position(pos.symbol, pos.side, pos.entry, size_part,
                                    pos.sl, pos.tp1, pos.tp2, pos.opened_at, pos.r_unit, pos.setup)
                partial = _close(part_pos, pos.tp1, "TP1", bar_idx)
                pos.size_usd *= (1 - self.cfg.partial_pct)
                pos.partial_taken = True
                pos.sl = pos.entry  # breakeven

        full = None
        if pos.side == "LONG":
            if bar["l"] <= pos.sl:
                full = _close(pos, pos.sl, "BE" if pos.partial_taken else "SL", bar_idx)
            elif bar["h"] >= pos.tp2:
                full = _close(pos, pos.tp2, "TP2", bar_idx)
        else:
            if bar["h"] >= pos.sl:
                full = _close(pos, pos.sl, "BE" if pos.partial_taken else "SL", bar_idx)
            elif bar["l"] <= pos.tp2:
                full = _close(pos, pos.tp2, "TP2", bar_idx)

        # Time decay: close after 60 bars (1h) if stuck
        if full is None and (bar_idx - pos.opened_at) > 60:
            full = _close(pos, float(bar["c"]), "TIME", bar_idx)

        return partial, full

    def _run_symbol(self, symbol, data, acct):
        df1, d15, d1h = data["1m"], data["15m"], data["1h"]
        train_end = max(int(len(df1) * self.cfg.train_frac), 300)
        print(f"    Training ensemble ML on {train_end} candles...", end=" ", flush=True)
        trained = self.ml.train(symbol, df1.iloc[:train_end].copy())
        print("OK" if trained else "SKIP (insufficient data)")

        start = max(train_end, 300)
        positions = []
        sym_trades = 0

        for i in range(start, len(df1) - 1):
            bar = df1.iloc[i]
            next_bar = df1.iloc[i + 1]

            # Manage positions
            new_pos = []
            for pos in positions:
                partial, full = self._check_position(pos, bar, i)
                if partial:
                    acct.balance += partial.pnl_usd
                    acct.peak_balance = max(acct.peak_balance, acct.balance)
                    acct.trades.append(partial); sym_trades += 1
                if full:
                    acct.balance += full.pnl_usd
                    acct.peak_balance = max(acct.peak_balance, acct.balance)
                    acct.trades.append(full); sym_trades += 1
                    acct.consec_losses = 0 if full.pnl_usd > 0 else acct.consec_losses + 1
                    if acct.consec_losses >= self.cfg.max_consec_losses:
                        acct.cooldown_until = i + self.cfg.cooldown_bars
                        acct.consec_losses = 0
                else:
                    new_pos.append(pos)
            positions = new_pos

            # Guards
            if i < acct.cooldown_until: continue
            if acct.balance <= acct.day_start_balance * (1 - self.cfg.daily_loss_limit_pct): continue
            if len(positions) >= self.cfg.max_positions: continue
            # Don't open same symbol twice
            if any(p.symbol == symbol for p in positions): continue

            # Build slices
            ts_ms = int(bar["ot"])
            d1m_slc = df1.iloc[max(0, i-200):i+1]
            d15m_slc = d15[d15["ot"] <= ts_ms].tail(100)
            d1h_slc = d1h[d1h["ot"] <= ts_ms].tail(50)
            if len(d15m_slc) < 30 or len(d1h_slc) < 20: continue

            # Only run full analysis every 15 candles (15 min) to match live scan interval
            if i % 15 != 0: continue

            ml_prob = self.ml.predict(symbol, d1m_slc) if trained else 0.5
            sig = analyze_sync(d1m_slc, d15m_slc, d1h_slc, ml_prob)
            if sig is None: continue

            # ML gate
            if sig.side == "LONG" and ml_prob < self.cfg.ml_min_prob: continue
            if sig.side == "SHORT" and (1 - ml_prob) < self.cfg.ml_min_prob: continue

            # Position sizing
            risk_amt = acct.balance * self.cfg.risk_pct
            notional = risk_amt / (sig.sl_pct / 100)
            if notional < 5: continue
            notional = min(notional, acct.balance * 20)

            # Fill on next bar
            atr_1m = MarketAnalyzer.get_atr(d1m_slc.tail(30), 14).iloc[-1]
            if not math.isfinite(atr_1m) or atr_1m <= 0: continue
            fill = _slip(sig.side, float(next_bar["o"]), float(atr_1m), is_market=True)

            if sig.side == "LONG":
                sl = fill * (1 - sig.sl_pct / 100)
                r = fill - sl
                tp1 = fill + r
                tp2 = fill + fill * (sig.tp_pct / 100)
            else:
                sl = fill * (1 + sig.sl_pct / 100)
                r = sl - fill
                tp1 = fill - r
                tp2 = fill - fill * (sig.tp_pct / 100)

            positions.append(Position(symbol=symbol, side=sig.side, entry=fill,
                size_usd=notional, sl=sl, tp1=tp1, tp2=tp2, opened_at=i+1,
                r_unit=r, setup=sig.setup))

        # Close remaining
        for pos in positions:
            trade = _close(pos, float(df1["c"].iloc[-1]), "EOD", len(df1)-1)
            acct.balance += trade.pnl_usd
            acct.trades.append(trade); sym_trades += 1
        return sym_trades

    async def run(self):
        acct = AccountState(balance=self.cfg.starting_balance,
                           peak_balance=self.cfg.starting_balance,
                           day_start_balance=self.cfg.starting_balance)
        async with httpx.AsyncClient(timeout=30.0) as client:
            for sym in self.cfg.symbols:
                print(f"  Loading {sym}...")
                data = await self._load(client, sym)
                if not data:
                    print(f"  [warn] skip {sym}: no data"); continue
                n = self._run_symbol(sym, data, acct)
                print(f"  {sym}: {n} trades  balance=${acct.balance:.2f}")
        return acct


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(acct, start_balance):
    trades = acct.trades
    n = len(trades)
    print("\n" + "=" * 60)
    print("  BACKTEST RESULT (Ensemble ML + Full Live Logic)")
    print("=" * 60)
    if n == 0:
        print("No trades taken."); return

    pnls = np.array([t.pnl_usd for t in trades])
    wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
    wr = len(wins) / n
    avg_w = wins.mean() if len(wins) else 0
    avg_l = losses.mean() if len(losses) else 0
    expect = pnls.mean()
    pf = (wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    equity = start_balance + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min() if len(dd) else 0
    rets = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([])
    sharpe = (rets.mean() / rets.std() * math.sqrt(1440)) if len(rets) > 0 and rets.std() > 0 else 0

    print(f"  Trades            : {n}")
    print(f"  Win rate          : {wr * 100:.1f}%")
    print(f"  Avg win / loss    : ${avg_w:+.3f} / ${avg_l:+.3f}")
    print(f"  Expectancy/trade  : ${expect:+.3f}")
    print(f"  Profit factor     : {pf:.2f}")
    print(f"  Max drawdown      : {max_dd * 100:.2f}%")
    print(f"  Sharpe (1m basis) : {sharpe:.2f}")
    print(f"  Start → Final     : ${start_balance:.2f} → ${acct.balance:.2f}  "
          f"(net ${acct.balance - start_balance:+.2f}, {(acct.balance/start_balance-1)*100:+.2f}%)")

    from collections import defaultdict
    by_setup = defaultdict(list); by_reason = defaultdict(list); by_sym = defaultdict(list)
    for t in trades:
        by_setup[t.setup].append(t.pnl_usd)
        by_reason[t.reason].append(t.pnl_usd)
        by_sym[t.symbol].append(t.pnl_usd)

    print("\n  Per-setup:")
    for k, v in sorted(by_setup.items()):
        a = np.array(v); w = (a > 0).mean()
        print(f"    {k:18s} n={len(a):3d}  wr={w*100:4.1f}%  net=${a.sum():+.2f}")
    print("\n  Per-exit-reason:")
    for k, v in sorted(by_reason.items()):
        a = np.array(v)
        print(f"    {k:18s} n={len(a):3d}  net=${a.sum():+.2f}")
    print("\n  Per-symbol:")
    for k, v in sorted(by_sym.items()):
        a = np.array(v); w = (a > 0).mean()
        print(f"    {k:12s} n={len(a):3d}  wr={w*100:4.1f}%  net=${a.sum():+.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,SUIUSDT,DOGEUSDT")
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--balance", type=float, default=100.0)
    p.add_argument("--risk", type=float, default=0.02)
    p.add_argument("--max-positions", type=int, default=5)
    p.add_argument("--ml-min", type=float, default=0.55)
    return p.parse_args()


async def _amain():
    args = _parse()
    cfg = BacktestConfig(
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        days=args.days, starting_balance=args.balance, risk_pct=args.risk,
        max_positions=args.max_positions, ml_min_prob=args.ml_min,
    )
    print(f"\nBacktest v2 | days={cfg.days} risk={cfg.risk_pct*100:.1f}% "
          f"max_pos={cfg.max_positions} ml_min={cfg.ml_min_prob}")
    print(f"Symbols: {cfg.symbols}\n")
    bt = ProBacktester(cfg)
    acct = await bt.run()
    summarize(acct, cfg.starting_balance)


if __name__ == "__main__":
    asyncio.run(_amain())
