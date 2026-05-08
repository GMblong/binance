"""
Pro Backtester (sinkron dengan strategi LIVE)
=============================================
Replika logika entry & exit dari `strategies/hybrid.py` + `strategies/analyzer.py`
+ `engine/ml_engine.py` — tanpa async, tanpa dependency ke httpx/websocket.
Gunanya: memvalidasi strategi bot live pada data historis.

Jalanin:
    python backtest_pro.py                                  # default: BTC/ETH/SOL/BNB/XRP, 3 hari
    python backtest_pro.py --symbols BTCUSDT,ETHUSDT --days 5
    python backtest_pro.py --offline                        # data sintetik GBM (kalau API tak bisa)

Output: win rate, expectancy, profit factor, max drawdown, Sharpe, per-setup breakdown.
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
    from sklearn.model_selection import train_test_split
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

from strategies.analyzer import MarketAnalyzer


API_URL = "https://fapi.binance.com"

# Cost model (matching live trading realistic fills)
TAKER_FEE = 0.0004            # 0.04% per side
SPREAD_BPS = 0.5              # 0.5 bps baseline
SLIPPAGE_ATR_FRAC = 0.05      # 5% dari ATR 1m sebagai slippage untuk MARKET

# ---------------------------------------------------------------------------
# ML engine (sync clone dari engine/ml_engine.py, tanpa httpx)
# ---------------------------------------------------------------------------

_FEATURES = [
    "ema_9", "ema_21", "rsi", "atr",
    "roc_c_1", "roc_c_5", "roc_v_1",
    "volatility", "body_size", "upper_wick", "lower_wick",
    "dist_ema9", "dist_ema21", "dist_vwap", "cvd_roc",
    "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
]


def _ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()


def _rsi(s: pd.Series, length: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr_series(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["h"], df["l"], df["c"]
    tr = pd.concat([(h - l).abs(), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering — sama dengan `MLPredictor.feature_engineering` tapi sync."""
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

    # VWAP
    tp = (df["h"] + df["l"] + df["c"]) / 3
    vp = tp * df["v"]
    df["vwap"] = vp.rolling(100).sum() / df["v"].rolling(100).sum()
    df["dist_vwap"] = (df["c"] - df["vwap"]) / df["vwap"]

    # CVD proxy
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


def triple_barrier_labels(df: pd.DataFrame, pt_mult: float = 1.5, sl_mult: float = 1.2, lookahead: int = 15) -> pd.Series:
    """Label 1 kalau TP hit duluan, 0 kalau SL hit/timeout."""
    prices = df["c"].values
    highs = df["h"].values
    lows = df["l"].values
    atrs = df["atr"].values
    median_atr = df["atr"].rolling(100).median().bfill().values

    labels = np.full(len(df), np.nan)
    for i in range(len(df) - lookahead):
        entry = prices[i]
        cur_atr = atrs[i]
        base_atr = median_atr[i]
        if not math.isfinite(cur_atr) or cur_atr <= 0 or not math.isfinite(base_atr):
            continue
        vol_ratio = cur_atr / (base_atr + 1e-9)
        dyn_pt = pt_mult * min(max(vol_ratio, 0.8), 2.0)
        dyn_sl = sl_mult * min(max(vol_ratio, 0.8), 2.0)
        pt_price = entry + cur_atr * dyn_pt
        sl_price = entry - cur_atr * dyn_sl
        lbl = 0
        for j in range(1, lookahead + 1):
            if highs[i + j] >= pt_price:
                lbl = 1
                break
            if lows[i + j] <= sl_price:
                lbl = 0
                break
        labels[i] = lbl
    return pd.Series(labels, index=df.index)


class SyncMLPredictor:
    """Versi sync dari `MLPredictor`. Dilatih sekali di awal backtest per simbol."""

    def __init__(self):
        self.models: Dict[str, object] = {}

    def train(self, symbol: str, df_1m: pd.DataFrame) -> bool:
        if not _HAS_LGBM or len(df_1m) < 300:
            return False
        feats = build_ml_features(df_1m)
        feats["target"] = triple_barrier_labels(feats, pt_mult=1.5, sl_mult=1.2, lookahead=15)
        feats = feats.dropna(subset=_FEATURES + ["target"])
        if len(feats) < 120:
            return False
        X = feats[_FEATURES]
        y = feats["target"].astype(int)
        if y.nunique() < 2:
            return False
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.06, max_depth=5, num_leaves=20,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            class_weight="balanced", n_jobs=-1, verbose=-1,
        )
        model.fit(X_tr, y_tr)
        acc = (model.predict(X_te) == y_te).mean()
        if acc < 0.50:
            return False
        self.models[symbol] = model
        return True

    def predict(self, symbol: str, df_1m: pd.DataFrame) -> float:
        model = self.models.get(symbol)
        if model is None:
            return 0.5
        feats = build_ml_features(df_1m.tail(120))
        feats = feats.dropna(subset=_FEATURES)
        if feats.empty:
            return 0.5
        X = feats[_FEATURES].iloc[[-1]]
        try:
            return float(model.predict_proba(X)[0][1])
        except Exception:
            return 0.5


# ---------------------------------------------------------------------------
# Signal generator (sync clone dari `analyze_hybrid_async`)
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    side: str          # "LONG" / "SHORT"
    score: int
    regime: str
    setup: str         # label untuk breakdown report
    entry_hint: float  # harga limit rekomendasi (fallback = close)
    sl_pct: float
    tp_pct: float
    ml_prob: float
    is_market: bool


def analyze_sync(
    d1m: pd.DataFrame,
    d15m: pd.DataFrame,
    d1h: pd.DataFrame,
    ml_prob: float,
    imbalance: float = 1.0,
    funding: float = 0.0,
) -> Optional[Signal]:
    """Ekuivalen sinkron dari `strategies.hybrid.analyze_hybrid_async`.

    Catatan: fitur live-only (lead-lag BTC, session heuristic) disederhanakan.
    Fokusnya mereplika *entry gate* + *score* + *SL/TP sizing* supaya backtest
    mencerminkan strategi nyata.
    """
    if len(d1m) < 60 or len(d15m) < 30 or len(d1h) < 20:
        return None

    try:
        price = float(d1m["c"].iloc[-1])

        # --- Direction (ADX-gated EMA cross, RSI mean-rev fallback) ---
        ema9_15m = MarketAnalyzer.get_ema(d15m["c"], 9).iloc[-1]
        ema21_15m = MarketAnalyzer.get_ema(d15m["c"], 21).iloc[-1]
        adx_val = MarketAnalyzer.get_adx(d15m, 14)
        if adx_val > 20:
            direction = 1 if ema9_15m > ema21_15m else -1
        else:
            rsi_15m = MarketAnalyzer.get_rsi(d15m["c"], 14).iloc[-1]
            if rsi_15m < 35:
                direction = 1
            elif rsi_15m > 65:
                direction = -1
            else:
                direction = 1 if ema9_15m > ema21_15m else -1

        # --- HTF alignment (1h) ---
        ema50_1h = MarketAnalyzer.get_ema(d1h["c"], 50).iloc[-1] if len(d1h) >= 50 else d1h["c"].iloc[-1]
        htf_dir = 1 if d1h["c"].iloc[-1] > ema50_1h else -1
        mtf_aligned = (direction == htf_dir)

        atr = MarketAnalyzer.get_atr(d1m, 14).iloc[-1]
        if not math.isfinite(atr) or atr <= 0:
            return None
        base_atr_pct = (atr / price) * 100

        regime = MarketAnalyzer.detect_regime(d15m)

        # --- Score (hybrid live logic) ---
        score = MarketAnalyzer.calculate_score(
            d1m, d15m, direction, imbalance=imbalance, funding=funding,
            regime=regime, neural_weights=None, session="QUIET", lead_lag=0,
        )
        if not mtf_aligned:
            score = int(score * 0.6)

        # --- ML blending (matching hybrid.py) ---
        if direction == 1:
            if ml_prob >= 0.65:
                score += int((ml_prob - 0.6) * 50)
            elif ml_prob < 0.40:
                score += int((ml_prob - 0.4) * 50)
        else:
            if ml_prob <= 0.35:
                score += int((0.4 - ml_prob) * 50)
            elif ml_prob > 0.60:
                score += int((0.6 - ml_prob) * 50)
        score = max(0, min(score, 100))

        # --- Liquidation magnet bonus ---
        liq = MarketAnalyzer.predict_liquidation_clusters(d15m)
        if liq:
            if direction == 1 and any(p > price for p in liq["short_liq"]):
                score = min(score + 5, 100)
            elif direction == -1 and any(p < price for p in liq["long_liq"]):
                score = min(score + 5, 100)

        # --- Fake move / overextended rejection ---
        vol_avg = d1m["v"].tail(20).mean()
        vol_recent = d1m["v"].tail(3).mean()
        vol_confirmed = vol_recent > vol_avg * 1.2
        last = d1m.iloc[-1]
        rng = float(last["h"] - last["l"])
        body = abs(float(last["c"] - last["o"]))
        body_ratio = body / rng if rng > 0 else 0
        is_fake = (not vol_confirmed) and body_ratio < 0.3

        ema21_1m = MarketAnalyzer.get_ema(d1m["c"], 21).iloc[-1]
        dist_ema = abs(price - ema21_1m) / price * 100
        is_over = dist_ema > base_atr_pct * 2.5

        # --- SL (structure-based with ML modulation) ---
        buffer_pct = max(base_atr_pct * 0.3, 0.12)
        low_L1 = d1m["l"].tail(15).min()
        low_L2 = d1m["l"].tail(30).min()
        high_L1 = d1m["h"].tail(15).max()
        high_L2 = d1m["h"].tail(30).max()
        ob = MarketAnalyzer.find_nearest_order_block(d1m, price, direction)
        if direction == 1:
            struct_low = low_L2 if ((price - low_L2) / price * 100) <= 3.0 else low_L1
            sl_price = ob["bottom"] if ob else (struct_low if struct_low else price * 0.99)
            sl_pct = ((price - sl_price) / price * 100) + buffer_pct
        else:
            struct_high = high_L2 if ((high_L2 - price) / price * 100) <= 3.0 else high_L1
            sl_price = ob["top"] if ob else (struct_high if struct_high else price * 1.01)
            sl_pct = ((sl_price - price) / price * 100) + buffer_pct
        if ml_prob > 0.7 or ml_prob < 0.3:
            sl_pct *= 0.85
        elif 0.45 < ml_prob < 0.55:
            sl_pct *= 1.15
        sl_pct = max(0.4, min(sl_pct, 3.5))

        # --- TP (regime + liq magnet) ---
        rr_mult = 2.0
        if regime == "RANGING":
            rr_mult = 1.5
        elif regime == "TRENDING":
            rr_mult = 2.5 if ((direction == 1 and ml_prob > 0.65) or (direction == -1 and ml_prob < 0.35)) else 2.0
        elif regime == "VOLATILE":
            rr_mult = 1.8

        htf_high = d15m["h"].tail(20).max() if len(d15m) >= 20 else high_L2
        htf_low = d15m["l"].tail(20).min() if len(d15m) >= 20 else low_L2
        tp_pct = sl_pct * rr_mult
        if direction == 1:
            struct_dist = ((htf_high - price) / price) * 100
            if struct_dist > sl_pct * 1.2:
                tp_pct = min(max(tp_pct, struct_dist), sl_pct * 3.5)
            if liq and any(p > price for p in liq.get("short_liq", [])):
                liq_target = min(p for p in liq["short_liq"] if p > price)
                liq_dist = ((liq_target - price) / price) * 100
                if liq_dist > tp_pct and liq_dist < sl_pct * 4:
                    tp_pct = liq_dist
        else:
            struct_dist = ((price - htf_low) / price) * 100
            if struct_dist > sl_pct * 1.2:
                tp_pct = min(max(tp_pct, struct_dist), sl_pct * 3.5)
            if liq and any(p < price for p in liq.get("long_liq", [])):
                liq_target = max(p for p in liq["long_liq"] if p < price)
                liq_dist = ((price - liq_target) / price) * 100
                if liq_dist > tp_pct and liq_dist < sl_pct * 4:
                    tp_pct = liq_dist
        tp_pct = max(sl_pct * 1.3, tp_pct)

        # --- Threshold & breakout ---
        prev_15_high = d1m["h"].iloc[-16:-1].max()
        prev_15_low = d1m["l"].iloc[-16:-1].min()
        raw_breakout = (direction == 1 and price > prev_15_high) or (direction == -1 and price < prev_15_low)
        is_breakout = raw_breakout and vol_confirmed and body_ratio > 0.5

        threshold = 80
        if regime == "VOLATILE":
            threshold = 75 if is_breakout else 90
        elif regime == "RANGING":
            threshold = 88
        elif regime == "TRENDING":
            threshold = 75 if is_breakout else 82

        is_market = regime == "TRENDING" and is_breakout and score >= 90 and not is_fake

        if is_fake or is_over:
            return None
        if score < threshold:
            return None

        side = "LONG" if direction == 1 else "SHORT"
        setup = f"{regime}-{'BRK' if is_breakout else 'PULL'}"
        return Signal(
            side=side, score=int(score), regime=regime, setup=setup,
            entry_hint=float(price), sl_pct=float(sl_pct), tp_pct=float(tp_pct),
            ml_prob=float(ml_prob), is_market=bool(is_market),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

async def fetch_klines(client, symbol: str, interval: str, limit: int = 1500) -> Optional[pd.DataFrame]:
    try:
        r = await client.get(f"{API_URL}/fapi/v1/klines",
                             params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=20)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    raw = r.json()
    if not raw:
        return None
    df = pd.DataFrame(raw).iloc[:, [0, 1, 2, 3, 4, 5]]
    df.columns = ["ot", "o", "h", "l", "c", "v"]
    return df.astype(float).reset_index(drop=True)


def synth_klines(n: int, seed: int, start_price: float = 30000.0, phases: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    phase_len = n // phases
    mu_pool = [0.00005, -0.00005, 0.0, 0.00015, -0.00015]
    sg_pool = [0.0006, 0.0012, 0.0020]
    prices = [start_price]
    for _ in range(phases):
        mu = rng.choice(mu_pool); sg = rng.choice(sg_pool)
        for _ in range(phase_len):
            prices.append(max(prices[-1] * (1 + rng.normal(mu, sg)), 1e-6))
    close = np.array(prices[1:])
    wick = np.abs(rng.normal(0, close.std() / close.mean() * 0.3, size=len(close)))
    high = close * (1 + wick * 0.5); low = close * (1 - wick * 0.5)
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    ret = np.concatenate([[0.0], np.diff(close) / close[:-1]])
    v = (500 + np.abs(ret) * 300000) * (1 + rng.normal(0, 0.2, size=len(close))).clip(0.1)
    now_ms = int(datetime.utcnow().timestamp() * 1000)
    ot = now_ms - (len(close) - np.arange(len(close))) * 60_000
    return pd.DataFrame({"ot": ot, "o": open_, "h": high, "l": low, "c": close, "v": v}).reset_index(drop=True)


def resample(df_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    df = df_1m.copy()
    df["dt"] = pd.to_datetime(df["ot"], unit="ms")
    df.set_index("dt", inplace=True)
    agg = df.resample(f"{minutes}min", label="right", closed="right").agg(
        {"ot": "first", "o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}
    ).dropna()
    return agg.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    side: str
    entry: float
    size_usd: float
    sl: float
    tp1: float
    tp2: float
    opened_at: int
    r_unit: float
    setup: str
    partial_taken: bool = False
    peak: float = 0.0


@dataclass
class ClosedTrade:
    symbol: str; side: str; setup: str
    opened_at: int; closed_at: int
    pnl_usd: float; reason: str
    entry: float; exit: float; bars_held: int


@dataclass
class AccountState:
    balance: float
    peak_balance: float
    day_start_balance: float
    consec_losses: int = 0
    cooldown_until: int = 0
    trades: List[ClosedTrade] = field(default_factory=list)


def _slip(side: str, price: float, atr_1m: float, is_market: bool) -> float:
    if not is_market:
        return price
    s = SLIPPAGE_ATR_FRAC * atr_1m + price * SPREAD_BPS / 10000
    return price + s if side == "LONG" else price - s


def _close(pos: Position, exit_price: float, reason: str, bar_idx: int) -> ClosedTrade:
    if pos.side == "LONG":
        pct = (exit_price - pos.entry) / pos.entry
    else:
        pct = (pos.entry - exit_price) / pos.entry
    gross = pct * pos.size_usd
    fee = 2 * TAKER_FEE * pos.size_usd
    return ClosedTrade(pos.symbol, pos.side, pos.setup, pos.opened_at, bar_idx,
                       gross - fee, reason, pos.entry, exit_price, bar_idx - pos.opened_at)


@dataclass
class BacktestConfig:
    symbols: List[str]
    days: int = 3
    starting_balance: float = 100.0
    risk_pct: float = 0.01
    max_positions: int = 3
    max_consec_losses: int = 3
    cooldown_bars: int = 60
    daily_loss_limit_pct: float = 0.05
    offline: bool = False
    ml_min_prob: float = 0.50
    train_frac: float = 0.5
    partial_pct: float = 0.5          # 50% close di TP1 (RR 1:1)


class ProBacktester:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.ml = SyncMLPredictor()

    async def _load(self, client, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
        if self.cfg.offline:
            seed = abs(hash(symbol)) % (2**31)
            n = self.cfg.days * 1440 + 500
            df1 = synth_klines(n, seed=seed, start_price=1000.0 * (1 + (seed % 50) / 50))
        else:
            # Binance max limit per req = 1500; paginate backward until we have enough
            need = self.cfg.days * 1440 + 500
            frames: List[pd.DataFrame] = []
            end_time: Optional[int] = None
            while sum(len(f) for f in frames) < need:
                batch = 1500
                params = {"symbol": symbol, "interval": "1m", "limit": batch}
                if end_time:
                    params["endTime"] = end_time
                try:
                    r = await client.get(f"{API_URL}/fapi/v1/klines", params=params, timeout=20)
                except Exception:
                    break
                if r.status_code != 200:
                    break
                raw = r.json()
                if not raw:
                    break
                df = pd.DataFrame(raw).iloc[:, [0, 1, 2, 3, 4, 5]]
                df.columns = ["ot", "o", "h", "l", "c", "v"]
                df = df.astype(float)
                frames.append(df)
                end_time = int(df["ot"].iloc[0]) - 1
                if len(raw) < batch:
                    break
            if not frames:
                return None
            df1 = pd.concat(frames[::-1], ignore_index=True).drop_duplicates(subset="ot").sort_values("ot").reset_index(drop=True)
        if df1 is None or len(df1) < 500:
            return None
        return {"1m": df1, "15m": resample(df1, 15), "1h": resample(df1, 60)}

    def _train_split_idx(self, df1: pd.DataFrame) -> int:
        train_end = int(len(df1) * self.cfg.train_frac)
        return max(train_end, 300)

    def _htf_slice(self, htf: pd.DataFrame, ts_ms: int) -> pd.DataFrame:
        return htf[htf["ot"] <= ts_ms]

    def _check_position(self, pos: Position, bar: pd.Series, bar_idx: int
                        ) -> Tuple[Optional[ClosedTrade], Optional[ClosedTrade]]:
        partial: Optional[ClosedTrade] = None
        if not pos.partial_taken:
            hit_tp1 = (pos.side == "LONG" and bar["h"] >= pos.tp1) or \
                      (pos.side == "SHORT" and bar["l"] <= pos.tp1)
            if hit_tp1:
                size_part = pos.size_usd * self.cfg.partial_pct
                part_pos = Position(pos.symbol, pos.side, pos.entry, size_part,
                                    pos.sl, pos.tp1, pos.tp2, pos.opened_at,
                                    pos.r_unit, pos.setup)
                partial = _close(part_pos, pos.tp1, "TP1", bar_idx)
                pos.size_usd *= (1 - self.cfg.partial_pct)
                pos.partial_taken = True
                pos.sl = pos.entry  # breakeven

        full: Optional[ClosedTrade] = None
        if pos.side == "LONG":
            if bar["l"] <= pos.sl:
                reason = "BE" if pos.partial_taken else "SL"
                full = _close(pos, pos.sl, reason, bar_idx)
            elif bar["h"] >= pos.tp2:
                full = _close(pos, pos.tp2, "TP2", bar_idx)
        else:
            if bar["h"] >= pos.sl:
                reason = "BE" if pos.partial_taken else "SL"
                full = _close(pos, pos.sl, reason, bar_idx)
            elif bar["l"] <= pos.tp2:
                full = _close(pos, pos.tp2, "TP2", bar_idx)
        return partial, full

    def _run_symbol(self, symbol: str, data: Dict[str, pd.DataFrame], acct: AccountState) -> int:
        df1 = data["1m"]
        d15 = data["15m"]
        d1h = data["1h"]

        # Train ML on first half only (walk-forward)
        train_end = int(len(df1) * self.cfg.train_frac)
        trained = self.ml.train(symbol, df1.iloc[:train_end].copy())

        start = max(train_end, 300)
        positions: List[Position] = []
        sym_trades = 0

        for i in range(start, len(df1) - 1):
            bar = df1.iloc[i]
            next_bar = df1.iloc[i + 1]

            # Update open positions
            new_positions: List[Position] = []
            for pos in positions:
                partial, full = self._check_position(pos, bar, i)
                if partial:
                    acct.balance += partial.pnl_usd
                    acct.peak_balance = max(acct.peak_balance, acct.balance)
                    acct.trades.append(partial)
                    sym_trades += 1
                if full:
                    acct.balance += full.pnl_usd
                    acct.peak_balance = max(acct.peak_balance, acct.balance)
                    acct.trades.append(full)
                    sym_trades += 1
                    is_win = full.pnl_usd > 0
                    acct.consec_losses = 0 if is_win else acct.consec_losses + 1
                    if acct.consec_losses >= self.cfg.max_consec_losses:
                        acct.cooldown_until = i + self.cfg.cooldown_bars
                        acct.consec_losses = 0
                else:
                    new_positions.append(pos)
            positions = new_positions

            # Guards
            if i < acct.cooldown_until:
                continue
            if acct.balance <= acct.day_start_balance * (1 - self.cfg.daily_loss_limit_pct):
                continue
            if len(positions) >= self.cfg.max_positions:
                continue

            # Build slices (no lookahead)
            ts_ms = int(bar["ot"])
            d1m_slc = df1.iloc[: i + 1]
            d15m_slc = self._htf_slice(d15, ts_ms)
            d1h_slc = self._htf_slice(d1h, ts_ms)
            if len(d15m_slc) < 30 or len(d1h_slc) < 20:
                continue

            # ML prob
            ml_prob = self.ml.predict(symbol, d1m_slc) if trained else 0.5

            sig = analyze_sync(d1m_slc, d15m_slc, d1h_slc, ml_prob)
            if sig is None:
                continue

            # ML gate (additional guard)
            if sig.side == "LONG" and ml_prob < self.cfg.ml_min_prob:
                continue
            if sig.side == "SHORT" and ml_prob > (1 - self.cfg.ml_min_prob):
                continue

            # Size: R-based
            # risk = cfg.risk_pct * balance; notional = risk / sl_pct
            risk_amt = acct.balance * self.cfg.risk_pct
            sl_pct = sig.sl_pct
            notional = risk_amt / (sl_pct / 100)
            if notional < 5:
                continue
            notional = min(notional, acct.balance * 20)  # max 20x leverage

            # Enter on NEXT bar open with slippage
            atr_1m = MarketAnalyzer.get_atr(d1m_slc.tail(30), 14).iloc[-1]
            if not math.isfinite(atr_1m) or atr_1m <= 0:
                continue
            fill = _slip(sig.side, float(next_bar["o"]), float(atr_1m), is_market=True)

            if sig.side == "LONG":
                sl = fill * (1 - sl_pct / 100)
                r = fill - sl
                tp1 = fill + r * 1.0
                tp2 = fill + fill * (sig.tp_pct / 100)
            else:
                sl = fill * (1 + sl_pct / 100)
                r = sl - fill
                tp1 = fill - r * 1.0
                tp2 = fill - fill * (sig.tp_pct / 100)

            positions.append(Position(
                symbol=symbol, side=sig.side, entry=fill, size_usd=notional,
                sl=sl, tp1=tp1, tp2=tp2, opened_at=i + 1, r_unit=r, setup=sig.setup,
            ))

        # Close remaining at last bar
        last_idx = len(df1) - 1
        last_close = float(df1["c"].iloc[-1])
        for pos in positions:
            trade = _close(pos, last_close, "EOD", last_idx)
            acct.balance += trade.pnl_usd
            acct.trades.append(trade)
            sym_trades += 1
        return sym_trades

    async def run(self) -> AccountState:
        acct = AccountState(
            balance=self.cfg.starting_balance,
            peak_balance=self.cfg.starting_balance,
            day_start_balance=self.cfg.starting_balance,
        )
        if self.cfg.offline or not _HAS_HTTPX:
            for sym in self.cfg.symbols:
                data = await self._load(None, sym)
                if not data:
                    print(f"[warn] skip {sym}: no data")
                    continue
                n = self._run_symbol(sym, data, acct)
                print(f"  {sym}: {n} trades  balance=${acct.balance:.2f}")
        else:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for sym in self.cfg.symbols:
                    data = await self._load(client, sym)
                    if not data:
                        print(f"[warn] skip {sym}: no data")
                        continue
                    n = self._run_symbol(sym, data, acct)
                    print(f"  {sym}: {n} trades  balance=${acct.balance:.2f}")
        return acct


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(acct: AccountState, start_balance: float) -> None:
    trades = acct.trades
    n = len(trades)
    print("\n" + "=" * 54)
    print("BACKTEST RESULT (strategi LIVE: hybrid + analyzer + ML)")
    print("=" * 54)
    if n == 0:
        print("No trades taken.")
        print(f"Final balance: ${acct.balance:.2f}")
        return

    pnls = np.array([t.pnl_usd for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    wr = len(wins) / n
    avg_w = wins.mean() if len(wins) else 0.0
    avg_l = losses.mean() if len(losses) else 0.0
    expect = pnls.mean()
    pf = (wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    equity = start_balance + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min() if len(dd) else 0.0
    rets = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([])
    sharpe = (rets.mean() / rets.std() * math.sqrt(1440)) if rets.std() > 0 else 0.0

    print(f"Trades            : {n}")
    print(f"Win rate          : {wr * 100:.1f}%")
    print(f"Avg win / loss    : ${avg_w:+.3f} / ${avg_l:+.3f}")
    print(f"Expectancy/trade  : ${expect:+.3f}")
    print(f"Profit factor     : {pf:.2f}")
    print(f"Max drawdown      : {max_dd * 100:.2f}%")
    print(f"Sharpe (1m basis) : {sharpe:.2f}")
    print(f"Start → Final     : ${start_balance:.2f} → ${acct.balance:.2f}  "
          f"(net ${acct.balance - start_balance:+.2f}, {(acct.balance/start_balance-1)*100:+.2f}%)")

    from collections import defaultdict
    by_setup = defaultdict(list); by_reason = defaultdict(list); by_sym = defaultdict(list)
    for t in trades:
        by_setup[t.setup].append(t.pnl_usd)
        by_reason[t.reason].append(t.pnl_usd)
        by_sym[t.symbol].append(t.pnl_usd)
    print("\nPer-setup:")
    for k, v in sorted(by_setup.items()):
        a = np.array(v)
        w = (a > 0).mean() if len(a) else 0
        print(f"  {k:18s} n={len(a):3d}  wr={w*100:4.1f}%  net=${a.sum():+.2f}")
    print("\nPer-exit-reason:")
    for k, v in sorted(by_reason.items()):
        a = np.array(v)
        print(f"  {k:18s} n={len(a):3d}  net=${a.sum():+.2f}")
    print("\nPer-symbol:")
    for k, v in sorted(by_sym.items()):
        a = np.array(v)
        w = (a > 0).mean() if len(a) else 0
        print(f"  {k:12s} n={len(a):3d}  wr={w*100:4.1f}%  net=${a.sum():+.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT")
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--balance", type=float, default=100.0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--ml-min", type=float, default=0.50)
    return p.parse_args()


async def _amain():
    args = _parse()
    cfg = BacktestConfig(
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        days=args.days, starting_balance=args.balance, risk_pct=args.risk,
        max_positions=args.max_positions, offline=args.offline, ml_min_prob=args.ml_min,
    )
    print(f"Backtest | offline={cfg.offline} days={cfg.days} risk={cfg.risk_pct*100:.1f}% "
          f"max_pos={cfg.max_positions} ml_min={cfg.ml_min_prob}")
    print(f"Symbols: {cfg.symbols}")
    bt = ProBacktester(cfg)
    acct = await bt.run()
    summarize(acct, cfg.starting_balance)


if __name__ == "__main__":
    asyncio.run(_amain())
