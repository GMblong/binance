import time
import pandas as pd
import asyncio
from datetime import datetime
from utils.config import API_URL, API_KEY
from utils.state import bot_state, market_data
from strategies.analyzer import MarketAnalyzer
from engine.ml_engine import ml_predictor
from engine.api import get_orderbook_imbalance
from utils.intelligence import get_current_session, detect_lead_lag, calculate_market_volatility
from utils.logger import log_error
from engine.multi_exchange import aggregate_flow
from engine.depth_predictor import depth_predictor
from engine.sentiment import sentiment_filter
from engine.microstructure import micro_engine
from engine.superhuman import superhuman
from engine.scalping_brain import scalping_brain

api_sem = asyncio.Semaphore(15)  # Higher concurrency for parallel analysis
busy_symbols = set()

# Analysis result cache - avoid re-analyzing if data hasn't changed
_analysis_cache = {}  # {symbol: (last_kline_ot, result, ts)}
_indicator_cache = {}  # {(symbol, interval): (last_ot, indicators_dict)}
_IND_TTL = 4.0  # seconds — reuse indicators if candle hasn't changed


def is_near_funding_settlement(buffer_minutes=10):
    """Block/penalize entries near funding rate settlement (00:00, 08:00, 16:00 UTC)."""
    now = datetime.utcnow()
    minutes_in_day = now.hour * 60 + now.minute
    for h in (0, 480, 960):  # 0:00, 8:00, 16:00 in minutes
        diff = abs(minutes_in_day - h)
        if diff > 720:
            diff = 1440 - diff
        if diff <= buffer_minutes:
            return True
    return False


def is_spread_too_wide(symbol, max_spread_bps=15):
    """Block entry if bid-ask spread > threshold (basis points)."""
    quote = market_data.best_quote.get(symbol)
    if not quote:
        return False
    _, best_bid, _, best_ask, _ = quote
    if best_bid <= 0:
        return False
    spread_bps = ((best_ask - best_bid) / best_bid) * 10000
    return spread_bps > max_spread_bps


def get_oi_delta_signal(symbol):
    """OI Delta Momentum: detect smart money positioning from OI + price change."""
    oi_current = market_data.oi.get(symbol, 0)
    oi_prev = market_data.prev_oi.get(symbol, 0)
    if oi_prev == 0 or oi_current == 0:
        return 0
    oi_delta = (oi_current - oi_prev) / oi_prev
    k = market_data.klines.get(symbol, {}).get("1m")
    if k is None or len(k) < 5:
        return 0
    price_delta = (k['c'].iloc[-1] - k['c'].iloc[-5]) / k['c'].iloc[-5]
    if oi_delta > 0.01 and price_delta > 0.001:
        return 1   # New longs (bullish)
    elif oi_delta > 0.01 and price_delta < -0.001:
        return -1  # New shorts (bearish)
    elif oi_delta < -0.01 and price_delta > 0.001:
        return 0   # Short covering (neutral-ish)
    elif oi_delta < -0.01 and price_delta < -0.001:
        return 0   # Long liquidation (neutral-ish)
    return 0

async def get_btc_trend(client):
    try:
        headers = {'X-MBX-APIKEY': API_KEY, 'User-Agent': 'Mozilla/5.0'}
        res = await client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": "BTCUSDT", "interval": "1m", "limit": 100}, headers=headers, timeout=10)
        if res.status_code == 200:
            df = pd.DataFrame(res.json()).iloc[:, [0, 1, 2, 3, 4, 5]].astype(float)
            df.columns = ["ot", "o", "h", "l", "c", "v"]
            ema = MarketAnalyzer.get_ema(df["c"], 20).iloc[-1]
            bot_state["btc_state"] = "BULLISH" if df["c"].iloc[-1] > ema else "BEARISH"
            bot_state["btc_dir"] = 1 if df["c"].iloc[-1] > ema else -1
    except Exception as e:
        log_error(f"BTC trend fetch error: {e}", include_traceback=False)

async def analyze_hybrid_async(client, symbol):
    if symbol in busy_symbols: 
        return _analysis_cache.get(symbol, (0, None))[1]
    try:
        now = time.time()
        if symbol not in market_data.klines: market_data.klines[symbol] = {}
        
        # Check cache - if kline data hasn't changed, return cached result
        k = market_data.klines.get(symbol, {})
        if "1m" in k and not k["1m"].empty:
            last_ot = float(k["1m"].iloc[-1]['ot'])
            cached = _analysis_cache.get(symbol)
            if cached and cached[0] == last_ot and (now - cached[2]) < 3.0:
                return cached[1]
        
        # Optimasi Sinkronisasi: Gunakan data WS jika tersedia dan segar
        k = market_data.klines.get(symbol, {})
        has_data = "1m" in k and "15m" in k and "1h" in k
        
        # Cek kesegaran data 1m (prioritas WS)
        is_fresh = False
        if "1m" in k and not k["1m"].empty:
            last_ts = k["1m"].iloc[-1]["ot"] / 1000.0
            if (now - last_ts) < 120: # Data kurang dari 120 detik dianggap cukup segar untuk scalping
                is_fresh = True

        # Force sync via REST hanya jika data hilang atau basi (> 2 menit)
        if not has_data or not is_fresh:
            last_p = market_data.last_prime.get(symbol, 0)
            # Throttle REST sync agar tidak spam (min 30s antar fetch per koin)
            if (now - last_p) > 30:
                busy_symbols.add(symbol)
                try:
                    async with api_sem:
                        headers = {'User-Agent': 'Mozilla/5.0'}
                        if API_KEY: headers['X-MBX-APIKEY'] = API_KEY
                        
                        res1, res15, res1h = await asyncio.gather(
                            client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "1m", "limit": 200}, headers=headers, timeout=15),
                            client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "15m", "limit": 100}, headers=headers, timeout=15),
                            client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "1h", "limit": 50}, headers=headers, timeout=15),
                        )
                        
                        if res1.status_code == 200 and res15.status_code == 200 and res1h.status_code == 200:
                            data1, data15, data1h = res1.json(), res15.json(), res1h.json()

                            def proc(data):
                                df = pd.DataFrame(data).iloc[:, [0, 1, 2, 3, 4, 5, 9]]
                                df.columns = ["ot", "o", "h", "l", "c", "v", "tbv"]
                                for col in ["o", "h", "l", "c", "v", "tbv"]: df[col] = df[col].astype(float)
                                return df
                            market_data.klines[symbol]["1m"] = proc(data1)
                            market_data.klines[symbol]["15m"] = proc(data15)
                            market_data.klines[symbol]["1h"] = proc(data1h)
                            market_data.last_prime[symbol] = now

                        else:
                            # SET LAST_PRIME TO AVOID SPAMMING ON FAILURE (Backoff 30 seconds)
                            market_data.last_prime[symbol] = now - 30 
                finally:
                    busy_symbols.discard(symbol)

        k = market_data.klines.get(symbol, {})
        # Re-check cache *after* sync attempt
        if "1m" not in k or "15m" not in k or "1h" not in k: 
            return None

        d1m, d15m, d1h = k["1m"], k["15m"], k["1h"]
        

        try:
            price = d1m["c"].iloc[-1]
            
            # --- SMART INDICATORS (compute once, reuse everywhere) ---
            # Check indicator cache first
            _ind_key_1m = (symbol, "1m")
            _ind_key_15m = (symbol, "15m")
            _last_ot_1m = float(d1m.iloc[-1]['ot'])
            _last_ot_15m = float(d15m.iloc[-1]['ot'])
            _cached_1m = _indicator_cache.get(_ind_key_1m)
            _cached_15m = _indicator_cache.get(_ind_key_15m)

            if _cached_1m and _cached_1m[0] == _last_ot_1m and (now - _cached_1m[2]) < _IND_TTL:
                ind1m = _cached_1m[1]
            else:
                _ema9_1m_s  = MarketAnalyzer.get_ema(d1m["c"], 9)
                _ema21_1m_s = MarketAnalyzer.get_ema(d1m["c"], 21)
                _rsi_1m_s   = MarketAnalyzer.get_rsi(d1m["c"], 14)
                ind1m = {
                    "ema9":  _ema9_1m_s.iloc[-1],
                    "ema21": _ema21_1m_s.iloc[-1],
                    "rsi":   _rsi_1m_s.iloc[-1],
                    "atr":   MarketAnalyzer.get_atr(d1m, 14).iloc[-1],
                }
                _indicator_cache[_ind_key_1m] = (_last_ot_1m, ind1m, now)

            if _cached_15m and _cached_15m[0] == _last_ot_15m and (now - _cached_15m[2]) < _IND_TTL:
                ind15m = _cached_15m[1]
            else:
                _rsi_15m_s = MarketAnalyzer.get_rsi(d15m["c"], 14)
                ind15m = {
                    "ema9":  MarketAnalyzer.get_ema(d15m["c"], 9).iloc[-1],
                    "ema21": MarketAnalyzer.get_ema(d15m["c"], 21).iloc[-1],
                    "rsi":   _rsi_15m_s.iloc[-1],
                    "rsi_prev": _rsi_15m_s.iloc[-3] if len(_rsi_15m_s) >= 3 else _rsi_15m_s.iloc[-1],
                    "adx":   MarketAnalyzer.get_adx(d15m, 14),
                    "atr":   MarketAnalyzer.get_atr(d15m, 14).iloc[-1] if len(d15m) >= 14 else ind1m["atr"],
                }
                _indicator_cache[_ind_key_15m] = (_last_ot_15m, ind15m, now)

            # Unpack — single source of truth, no recomputation below
            ema9_1m   = ind1m["ema9"]
            ema21_1m  = ind1m["ema21"]
            rsi_1m    = ind1m["rsi"]
            atr       = ind1m["atr"]
            ema9_15m  = ind15m["ema9"]
            ema21_15m = ind15m["ema21"]
            rsi_15m   = ind15m["rsi"]
            adx_val   = ind15m["adx"]
            atr_15m   = ind15m["atr"]
            is_reversal_trade = False
            if adx_val > 20:
                ema_dir = 1 if ema9_15m > ema21_15m else -1
                
                # EXHAUSTION REVERSAL: trend sudah di ujung, ambil posisi reversal
                if ema_dir == -1 and (rsi_15m < 35 or rsi_1m < 30):
                    direction = 1   # Trend bearish exhausted → LONG (bounce)
                    is_reversal_trade = True
                elif ema_dir == 1 and (rsi_15m > 65 or rsi_1m > 70):
                    direction = -1  # Trend bullish exhausted → SHORT (drop)
                    is_reversal_trade = True
                else:
                    direction = ema_dir
            else:
                # Weak trend: use RSI mean-reversion logic
                if rsi_15m < 35:
                    direction = 1  # Oversold = buy
                elif rsi_15m > 65:
                    direction = -1  # Overbought = sell
                else:
                    direction = 1 if ema9_15m > ema21_15m else -1
            
            
        except Exception as e:
            return None
        
        # --- HIGHER TIMEFRAME (HTF) BIAS ---
        # 1h EMA confirmation: don't trade against the hourly trend
        ema20_1h = MarketAnalyzer.get_ema(d1h["c"], 20).iloc[-1] if len(d1h) >= 20 else price
        ema50_1h = MarketAnalyzer.get_ema(d1h["c"], 50).iloc[-1] if len(d1h) >= 50 else ema20_1h
        htf_direction = 1 if d1h["c"].iloc[-1] > ema50_1h else -1
        # Stricter MTF: check 1m RSI alignment too (reuse cached rsi_1m)
        m1_aligned = (direction == 1 and rsi_1m > 45) or (direction == -1 and rsi_1m < 55)
        mtf_aligned = (direction == htf_direction) and m1_aligned
        
        # --- INSTITUTIONAL DATA GATHERING (parallel async) ---
        _gather_results = await asyncio.gather(
            get_orderbook_imbalance(client, symbol),
            micro_engine.compute(symbol, window_sec=60, client=client),
            scalping_brain.compute(symbol, direction, MarketAnalyzer.detect_regime(d15m), d1m, d15m, d1h, client=client),
            return_exceptions=True,
        )
        imbalance = _gather_results[0] if not isinstance(_gather_results[0], Exception) else 1.0
        micro = _gather_results[1] if not isinstance(_gather_results[1], Exception) and _gather_results[1] else {
            "vpin": 0.5, "ofi": 0.0, "hurst": 0.5, "entropy": 0.5, "microprice_skew": 0.0,
            "absorption": 0.0, "whale_prints": 0, "quote_stuffing": 0.0, "hawkes_intensity": 0.0,
            "vol_compression": 0.0, "skewness": 0.0, "kurtosis": 3.0,
        }
        brain = _gather_results[2] if not isinstance(_gather_results[2], Exception) and _gather_results[2] else {
            "score_boost": 0, "n_signals": 0, "entry_quality": 0.0, "signals_fired": [],
        }
        regime = MarketAnalyzer.detect_regime(d15m)
        funding = market_data.funding.get(symbol, 0.0)
        session = get_current_session()
        lead_lag = detect_lead_lag(symbol)
        
        # --- PER-SYMBOL DYNAMIC WEIGHTS ---
        # Try to get specific weights for this symbol, fallback to global weights
        s_weights = bot_state.get("sym_weights", {}).get(symbol)
        if not s_weights:
            s_weights = bot_state.get("neural_weights")
            
        score, active_features = MarketAnalyzer.calculate_score(
            d1m, d15m, direction, imbalance, funding, regime, s_weights,
            session, lead_lag, return_features=True,
        )
        struct, _, recent_high, recent_low = MarketAnalyzer.detect_structure(d1m)
        
        # Penalize if 15m and 1h disagree (counter-trend = high risk)
        if not mtf_aligned:
            score = int(score * 0.75)  # 25% penalty for fighting HTF
        
        # --- FUNDING RATE TIMING FILTER ---
        if is_near_funding_settlement(10):
            score = int(score * 0.75)  # Penalty near funding settlement
        
        # --- SPREAD FILTER ---
        if is_spread_too_wide(symbol, max_spread_bps=15):
            score = int(score * 0.6)  # Very wide spread = bad liquidity
        
        # --- OI DELTA MOMENTUM ---
        oi_signal = get_oi_delta_signal(symbol)
        if oi_signal == direction:
            score = min(score + 10, 100)  # Smart money agrees
        elif oi_signal == -direction:
            score = max(score - 12, 0)  # Smart money disagrees
        
        # --- MACHINE LEARNING INTEGRATION ---
        # Predict UP probability (0.0 to 1.0)
        ml_prob = await ml_predictor.predict(client, symbol, d1m)
        ml_has_model = symbol in ml_predictor.models
        ml_score_boost = 0
        ml_w = s_weights.get(f"{regime}:ml", 1.0) if s_weights else 1.0

        # Logika Baru (Scaled Confidence): 
        # Hanya boost jika sangat yakin (>60% atau <40%), dan penalti berskala jika tidak setuju
        if direction == 1:
            if ml_prob >= 0.60:
                ml_score_boost = int((ml_prob - 0.55) * 50 * ml_w)
            elif ml_prob < 0.40:
                ml_score_boost = int((ml_prob - 0.4) * 50 * ml_w) # Hasil negatif
        elif direction == -1:
            if ml_prob <= 0.40:
                ml_score_boost = int((0.45 - ml_prob) * 50 * ml_w)
            elif ml_prob > 0.60:
                ml_score_boost = int((0.6 - ml_prob) * 50 * ml_w) # Hasil negatif
        
        score = min(max(score + ml_score_boost, 0), 100)
        
        # Penalti jika ML model belum tersedia — cap score agar tidak bisa entry
        if not ml_has_model:
            score = min(score, 75)  # Max 75 tanpa ML
        
        # --- LIQUIDATION MAGNET (M2) ---
        liq = MarketAnalyzer.predict_liquidation_clusters(d15m)
        if liq:
            if "liq_map" not in bot_state: bot_state["liq_map"] = {}
            bot_state["liq_map"][symbol] = liq
            
            # If LONG, price is drawn up to short_liq (liquidating shorters)
            if direction == 1 and any(p > price for p in liq["short_liq"]):
                score = min(score + 5, 100)
            # If SHORT, price is drawn down to long_liq (liquidating longers)
            elif direction == -1 and any(p < price for p in liq["long_liq"]):
                score = min(score + 5, 100)

        # --- REAL CVD FROM aggTrade (tick-level aggressive flow) ---
        # Falls back to 0 if buffer empty (symbol not in top-N subscription).
        # CVD is *normalized* by recent dollar volume so the threshold is
        # comparable across coins.
        cvd_60s, cvd_n = market_data.get_live_cvd(symbol, window_sec=60)
        cvd_300s, _ = market_data.get_live_cvd(symbol, window_sec=300)
        if cvd_n >= 10:  # Need at least 10 trades in the window
            # Normalize by the 1m candle's typical dollar volume
            try:
                last_candle = d1m.iloc[-1]
                norm_dv = float(last_candle['v'] * last_candle['c']) + 1.0
                cvd_ratio_60 = cvd_60s / norm_dv     # ~[-1, +1]
                cvd_ratio_300 = cvd_300s / (norm_dv * 5.0)
            except Exception:
                cvd_ratio_60 = cvd_ratio_300 = 0.0

            # Strong aggressive flow in our direction → bonus
            if direction == 1:
                if cvd_ratio_60 > 0.25 and cvd_ratio_300 > 0.15:
                    score = min(score + 10, 100)
                    active_features.append(f"{regime}:cvd_live")
                elif cvd_ratio_60 < -0.25:  # Strong sells against LONG → penalty
                    score = max(score - 15, 0)
            else:  # SHORT
                if cvd_ratio_60 < -0.25 and cvd_ratio_300 < -0.15:
                    score = min(score + 10, 100)
                    active_features.append(f"{regime}:cvd_live")
                elif cvd_ratio_60 > 0.25:
                    score = max(score - 15, 0)
        
        # --- ENTRY LOGIC (Smart Sniper with Fake-Move Detection) ---
        
        # Pre-calculate needed values
        base_atr_pct = (atr / price) * 100
        ob = MarketAnalyzer.find_nearest_order_block(d1m, price, direction)
        fvg = MarketAnalyzer.get_nearest_fvg(d1m)
        
        # 1. Validate move is REAL (not fake/manipulation)
        is_fake_move = False
        
        # Check volume confirmation: real moves have above-average volume
        vol_avg = d1m['v'].tail(20).mean()
        vol_recent = d1m['v'].tail(3).mean()
        vol_confirmed = vol_recent > vol_avg * 1.2
        
        # Check candle body quality: fake moves have long wicks, small bodies
        last_candle = d1m.iloc[-1]
        candle_range = last_candle['h'] - last_candle['l']
        candle_body = abs(last_candle['c'] - last_candle['o'])
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        
        # Fake move indicators:
        # - Low volume + big wick = stop hunt / manipulation
        # - Price spike but close near open = rejection
        if not vol_confirmed and body_ratio < 0.3:
            is_fake_move = True
        
        # Check if price is overextended from EMA (chasing)
        dist_from_ema21 = abs(price - MarketAnalyzer.get_ema(d1m["c"], 21).iloc[-1]) / price * 100
        is_overextended = dist_from_ema21 > base_atr_pct * 2.5
        
        # --- MULTI-CANDLE FAKE MOVE DETECTION (3-candle sequence) ---
        if not is_fake_move and len(d1m) >= 5:
            is_fake_move = MarketAnalyzer.detect_multi_candle_fake(d1m, direction)
        
        # --- IMMEDIATE EMA POSITION PENALTY ---
        if not is_reversal_trade:
            if direction == 1 and price < ema9_1m and price < ema21_1m:
                score = int(score * 0.85)
            elif direction == -1 and price > ema9_1m and price > ema21_1m:
                score = int(score * 0.85)
        
        # --- TREND EXHAUSTION PENALTY ---
        if direction == -1 and rsi_1m < 30:
            score = int(score * 0.7)
        elif direction == 1 and rsi_1m > 70:
            score = int(score * 0.7)
        elif direction == -1 and rsi_15m < 35:
            score = int(score * 0.75)
        elif direction == 1 and rsi_15m > 65:
            score = int(score * 0.75)
        
        # --- WYCKOFF PHASE CONTEXT ---
        wyckoff = MarketAnalyzer.detect_wyckoff_phase(d15m)
        # Block entries against Wyckoff phase
        if direction == 1 and wyckoff == "DISTRIBUTION":
            score = int(score * 0.7)  # Penalty: buying into distribution
        elif direction == -1 and wyckoff == "ACCUMULATION":
            score = int(score * 0.7)  # Penalty: shorting into accumulation
        elif direction == 1 and wyckoff == "MARKUP":
            score = min(score + 10, 100)  # Bonus: buying in markup phase
            active_features.append(f"{regime}:wyckoff")
        elif direction == -1 and wyckoff == "MARKDOWN":
            score = min(score + 10, 100)  # Bonus: shorting in markdown phase
            active_features.append(f"{regime}:wyckoff")
        
        # --- ORDERFLOW MICROSTRUCTURE ---
        bid_vel, ask_vel, spoof_score = market_data.get_depth_velocity(symbol)
        # If spoofing detected, treat as fake move
        if spoof_score > 0.6:
            is_fake_move = True
        # Depth velocity confirmation
        if direction == 1 and bid_vel > 0 and ask_vel < 0:
            score = min(score + 5, 100)  # Bids growing, asks shrinking = bullish
            active_features.append(f"{regime}:depth_vel")
        elif direction == -1 and ask_vel > 0 and bid_vel < 0:
            score = min(score + 5, 100)  # Asks growing, bids shrinking = bearish
            active_features.append(f"{regime}:depth_vel")
        
        # --- HMM REGIME CONTEXT ---
        hmm_regime = MarketAnalyzer.detect_hmm_regime(d15m)
        if hmm_regime == "MOMENTUM" and regime == "TRENDING":
            score = min(score + 5, 100)  # Double confirmation: rule-based + HMM agree
            active_features.append(f"{regime}:hmm")
        elif hmm_regime == "MEAN_REVERT" and regime == "RANGING":
            score = min(score + 5, 100)  # Both say mean-revert
            active_features.append(f"{regime}:hmm")
        
        # --- FRACTAL DIMENSION & VARIANCE RATIO (Market Quality Filter) ---
        fd = MarketAnalyzer.fractal_dimension(d1m, n=50)
        vr = MarketAnalyzer.variance_ratio_test(d15m, period=5)
        # Only trade structured markets (FD < 1.5 = not pure noise)
        if fd > 1.6:
            score = int(score * 0.6)  # Heavy penalty: market is noise/chop
        elif fd < 1.3:
            score = min(score + 5, 100)  # Clean structure
            active_features.append(f"{regime}:fractal")
        # VR confirms regime: >1 = momentum, <1 = mean-revert
        if regime == "TRENDING" and vr > 1.2:
            score = min(score + 5, 100)
            active_features.append(f"{regime}:var_ratio")
        elif regime == "RANGING" and vr < 0.8:
            score = min(score + 5, 100)
            active_features.append(f"{regime}:var_ratio")
        
        # --- ICEBERG ORDER DETECTION ---
        iceberg_bid, iceberg_ask = market_data.detect_iceberg(symbol)
        if direction == 1 and iceberg_bid:
            score = min(score + 10, 100)  # Hidden buyer supporting price
            active_features.append(f"{regime}:iceberg")
        elif direction == -1 and iceberg_ask:
            score = min(score + 10, 100)  # Hidden seller capping price
            active_features.append(f"{regime}:iceberg")
        elif direction == 1 and iceberg_ask:
            score = max(score - 10, 0)  # Hidden seller blocks our long
        elif direction == -1 and iceberg_bid:
            score = max(score - 10, 0)  # Hidden buyer blocks our short
        
        # --- MULTI-EXCHANGE DIVERGENCE (Aggregate: Binance + Bybit + OKX) ---
        avg_div, cvd_bias = aggregate_flow.get_cross_exchange_signal(symbol)
        if direction == 1 and avg_div > 0.03:
            score = min(score + 8, 100)
            active_features.append(f"{regime}:xchange")
        elif direction == -1 and avg_div < -0.03:
            score = min(score + 8, 100)
            active_features.append(f"{regime}:xchange")
        # Cross-exchange CVD alignment
        if direction == 1 and cvd_bias > 0.3:
            score = min(score + 5, 100)
            active_features.append(f"{regime}:xcvd")
        elif direction == -1 and cvd_bias < -0.3:
            score = min(score + 5, 100)
            active_features.append(f"{regime}:xcvd")
        
        # --- PREDICTIVE DEPTH (Real vs Fake Wall) ---
        bid_real, ask_real = depth_predictor.predict(symbol)
        if direction == 1 and bid_real > 0.7:
            score = min(score + 5, 100)  # Real bid wall = support confirmed
            active_features.append(f"{regime}:depth_pred")
        elif direction == -1 and ask_real > 0.7:
            score = min(score + 5, 100)  # Real ask wall = resistance confirmed
            active_features.append(f"{regime}:depth_pred")
        elif direction == 1 and ask_real > 0.8:
            score = max(score - 8, 0)  # Real ask wall blocks our long
        elif direction == -1 and bid_real > 0.8:
            score = max(score - 8, 0)  # Real bid wall blocks our short
        
        # --- SENTIMENT FILTER ---
        sentiment = sentiment_filter.get_sentiment(symbol)
        if sentiment < -0.5 and direction == 1:
            score = max(score - 20, 0)  # Bearish event, don't go long
        elif sentiment > 0.5 and direction == -1:
            score = max(score - 20, 0)  # Bullish event, don't go short
        liq_bias = sentiment_filter.get_liq_bias(symbol)
        if liq_bias == direction:
            score = min(score + 5, 100)  # Liquidation cascade in our direction
            active_features.append(f"{regime}:liq_cascade")
        
        # --- MICROSTRUCTURE ALPHA ENGINE (Superhuman Signals) ---
        # VPIN: smart money detected (>0.6 = informed trading active)
        if micro["vpin"] > 0.6:
            # High VPIN + OFI alignment = strong institutional flow
            if (direction == 1 and micro["ofi"] > 0.2) or (direction == -1 and micro["ofi"] < -0.2):
                score = min(score + 12, 100)
                active_features.append(f"{regime}:vpin")
            elif (direction == 1 and micro["ofi"] < -0.2) or (direction == -1 and micro["ofi"] > 0.2):
                score = max(score - 15, 0)  # Informed flow AGAINST us

        # Hurst: trend persistence confirmation
        if micro["hurst"] > 0.6 and regime == "TRENDING":
            score = min(score + 8, 100)  # Trend is persistent (not random)
            active_features.append(f"{regime}:hurst")
        elif micro["hurst"] < 0.4 and regime == "RANGING":
            score = min(score + 5, 100)  # Mean-reverting confirmed
            active_features.append(f"{regime}:hurst")

        # Entropy: low entropy = predictable (good for us)
        if micro["entropy"] < 0.6:
            score = min(score + 5, 100)
            active_features.append(f"{regime}:entropy")

        # Microprice skew: fair value directional bias
        if (direction == 1 and micro["microprice_skew"] > 0.3) or \
           (direction == -1 and micro["microprice_skew"] < -0.3):
            score = min(score + 8, 100)
            active_features.append(f"{regime}:microprice")
        elif (direction == 1 and micro["microprice_skew"] < -0.4) or \
             (direction == -1 and micro["microprice_skew"] > 0.4):
            score = max(score - 10, 0)  # Fair value against us

        # Absorption: whale accumulating silently
        if micro["absorption"] > 0.7:
            score = min(score + 8, 100)
            active_features.append(f"{regime}:absorption")

        # Whale prints: large outlier trades in our direction
        if micro["whale_prints"] >= 2 and abs(micro["ofi"]) > 0.15:
            if (direction == 1 and micro["ofi"] > 0) or (direction == -1 and micro["ofi"] < 0):
                score = min(score + 10, 100)
                active_features.append(f"{regime}:whale")

        # Quote stuffing: if high, treat as manipulation (fake move)
        if micro["quote_stuffing"] > 0.7:
            is_fake_move = True

        # Hawkes intensity: burst of activity = momentum building
        if micro["hawkes_intensity"] > 0.6:
            score = min(score + 5, 100)
            active_features.append(f"{regime}:hawkes")

        # Vol compression: coiled spring about to explode
        if micro["vol_compression"] > 0.7:
            score = min(score + 8, 100)
            active_features.append(f"{regime}:vol_compress")

        # Higher-order stats: tail risk / distribution shift
        if direction == 1 and micro["skewness"] > 0.5:
            score = min(score + 5, 100)  # Positive skew = upside potential
        elif direction == -1 and micro["skewness"] < -0.5:
            score = min(score + 5, 100)  # Negative skew = downside potential
        if micro["kurtosis"] > 5.0:
            # Fat tails = extreme move likely, tighten SL later
            pass

        # --- SUPERHUMAN SIGNAL DETECTION (Invisible to humans) ---
        sh = superhuman.compute(symbol, d1m, d15m, d1h)

        # Tick Imbalance Bars: informed flow detected
        if sh["tib_signal"] == direction:
            score = min(score + 10, 100)
            active_features.append(f"{regime}:tib")
        elif sh["tib_signal"] == -direction:
            score = max(score - 8, 0)

        # Flow Toxicity: smart money active before big move
        if sh["toxicity"] > 0.7:
            if (direction == 1 and sh["smart_money"] > 0.2) or (direction == -1 and sh["smart_money"] < -0.2):
                score = min(score + 12, 100)
                active_features.append(f"{regime}:toxicity")
            elif (direction == 1 and sh["smart_money"] < -0.3) or (direction == -1 and sh["smart_money"] > 0.3):
                score = max(score - 12, 0)

        # Entropy Regime Shift: regime changing in our favor
        if sh["entropy_shift"] > 0.3 and regime == "RANGING":
            score = min(score + 8, 100)  # Shifting to trending = momentum opportunity
            active_features.append(f"{regime}:entropy_shift")
        elif sh["entropy_shift"] < -0.3 and regime == "TRENDING":
            score = max(score - 8, 0)  # Trend dying

        # Hidden Divergence: multi-TF invisible divergence
        if sh["hidden_div"] == direction:
            score = min(score + 10, 100)
            active_features.append(f"{regime}:hidden_div")
        elif sh["hidden_div"] == -direction:
            score = max(score - 10, 0)

        # Information Asymmetry: insiders trading
        if sh["info_asymmetry"] > 0.6 and sh["tib_signal"] == direction:
            score = min(score + 8, 100)
            active_features.append(f"{regime}:info_asym")

        # Gamma Proxy: squeeze potential
        if direction == 1 and sh["gamma_proxy"] > 0.3:
            score = min(score + 5, 100)
            active_features.append(f"{regime}:gamma")
        elif direction == -1 and sh["gamma_proxy"] < -0.3:
            score = min(score + 5, 100)
            active_features.append(f"{regime}:gamma")

        # Temporal Alpha: time-based edge
        if sh["temporal_alpha"] > 0.3:
            score = min(score + 5, 100)
            active_features.append(f"{regime}:temporal")
        elif sh["temporal_alpha"] < -0.3:
            score = max(score - 5, 0)

        # Autocorrelation Decay: momentum exhaustion
        if sh["autocorr_decay"] > 0.7:
            score = max(score - 10, 0)  # Momentum dying, don't enter
            is_fake_move = True

        # Micro Momentum Shift: tick-level trend change
        if sh["micro_momentum"] == direction:
            score = min(score + 8, 100)
            active_features.append(f"{regime}:micro_shift")
        elif sh["micro_momentum"] == -direction:
            score = max(score - 8, 0)

        # Price Discovery: Binance lagging other exchanges
        if (direction == 1 and sh["price_discovery"] > 0.3) or \
           (direction == -1 and sh["price_discovery"] < -0.3):
            score = min(score + 8, 100)
            active_features.append(f"{regime}:price_disc")

        # --- SCALPING BRAIN META-INTELLIGENCE (Bayesian fusion of ALL invisible signals) ---
        score = min(max(score + brain["score_boost"], 0), 100)
        if brain["n_signals"] >= 5:
            active_features.append(f"{regime}:brain_confluence")
        if brain["entry_quality"] > 0.7:
            active_features.append(f"{regime}:brain_hq")

        # 2. Smart limit price placement
        # Priority: Order Block > FVG > VWAP-area > Volume Profile POC > EMA21 pullback
        vwap_price = None
        if len(d1m) >= 60:
            tp = (d1m['h'] + d1m['l'] + d1m['c']) / 3
            vwap_price = (tp * d1m['v']).tail(60).sum() / d1m['v'].tail(60).sum()
        
        # Volume Profile POC from 15m data (stronger level)
        vpoc = MarketAnalyzer.get_volume_profile(d15m, bins=30) if len(d15m) >= 20 else None
        
        if direction == 1:
            # LONG: place limit at support levels (below current price)
            if ob and ob["top"] < price and ob["top"] > price * 0.97:
                limit_p = ob["top"]  # Order block top = strong support
            elif fvg and fvg["type"] == "BULLISH" and fvg["top"] < price:
                limit_p = fvg["top"]  # FVG fill level
            elif vwap_price and vwap_price < price and vwap_price > price * 0.98:
                limit_p = vwap_price  # VWAP as dynamic support
            elif vpoc and vpoc["poc"] < price and vpoc["poc"] > price * 0.97:
                limit_p = vpoc["poc"]  # Volume POC as magnet support
            else:
                # EMA21 pullback (deeper than EMA9 = better entry) — use cached
                limit_p = ema21_1m if ema21_1m < price else ema9_1m
        else:
            # SHORT: place limit at resistance levels (above current price)
            if ob and ob["bottom"] > price and ob["bottom"] < price * 1.03:
                limit_p = ob["bottom"]  # Order block bottom = strong resistance
            elif fvg and fvg["type"] == "BEARISH" and fvg["bottom"] > price:
                limit_p = fvg["bottom"]  # FVG fill level
            elif vwap_price and vwap_price > price and vwap_price < price * 1.02:
                limit_p = vwap_price
            elif vpoc and vpoc["poc"] > price and vpoc["poc"] < price * 1.03:
                limit_p = vpoc["poc"]  # Volume POC as magnet resistance
            else:
                limit_p = ema21_1m if ema21_1m > price else ema9_1m

        # --- SMART MARKET STRUCTURE SL & TP (H6) ---
        atr_15m_pct = (atr_15m / price) * 100
        buffer_pct = max(base_atr_pct * 0.3, 0.12)
        
        # Multi-timeframe swing levels
        recent_low_L1 = d1m['l'].tail(15).min()
        recent_low_L2 = d1m['l'].tail(30).min()
        recent_high_L1 = d1m['h'].tail(15).max()
        recent_high_L2 = d1m['h'].tail(30).max()
        # 15m structure for TP targets
        htf_high = d15m['h'].tail(20).max() if len(d15m) >= 20 else recent_high_L2
        htf_low = d15m['l'].tail(20).min() if len(d15m) >= 20 else recent_low_L2
        
        if direction == 1:
            struct_low = recent_low_L2 if ((price - recent_low_L2) / price * 100) <= 3.0 else recent_low_L1
            sl_price = ob["bottom"] if ob else (struct_low if struct_low else price * (1 - 0.01))
            sl_pct = ((price - sl_price) / price * 100) + buffer_pct
        else:
            struct_high = recent_high_L2 if ((recent_high_L2 - price) / price * 100) <= 3.0 else recent_high_L1
            sl_price = ob["top"] if ob else (struct_high if struct_high else price * (1 + 0.01))
            sl_pct = ((sl_price - price) / price * 100) + buffer_pct
        
        # ML-adjusted SL: tighter when high confidence, wider when uncertain
        if ml_prob > 0.7 or ml_prob < 0.3:
            sl_pct *= 0.85  # High confidence = tighter SL (less risk per trade)
        elif 0.45 < ml_prob < 0.55:
            sl_pct *= 1.15  # Uncertain = wider SL (give room)
        
        # Fat-tail protection: widen SL when kurtosis is extreme (distribution has fat tails)
        if micro["kurtosis"] > 5.0:
            sl_pct *= 1.1
        
        sl_pct = max(0.6, min(sl_pct, 3.5))  # Min 0.6% SL to avoid premature stops
        
        # --- SMART TP: Structure-based targets + Liquidation Magnets ---
        # Base RR from regime
        rr_mult = 2.0
        if regime == "RANGING":
            rr_mult = 1.5
        elif regime == "TRENDING":
            rr_mult = 2.5 if ((direction == 1 and ml_prob > 0.65) or (direction == -1 and ml_prob < 0.35)) else 2.0
        elif regime == "VOLATILE":
            rr_mult = 1.8  # Quick exits in volatile
        
        # Try to use structural TP target (next resistance/support)
        struct_tp_pct = sl_pct * rr_mult  # Default
        if direction == 1:
            # TP target = next HTF resistance or liquidation cluster
            struct_dist = ((htf_high - price) / price) * 100
            if struct_dist > sl_pct * 1.2:  # Only if it gives at least 1.2R
                struct_tp_pct = min(struct_dist, sl_pct * 3.5)
            # Check liquidation magnet as extended target
            if liq and any(p > price for p in liq.get("short_liq", [])):
                liq_target = min(p for p in liq["short_liq"] if p > price)
                liq_dist = ((liq_target - price) / price) * 100
                if liq_dist > struct_tp_pct and liq_dist < sl_pct * 4:
                    struct_tp_pct = liq_dist  # Extend TP to liquidation magnet
        else:
            struct_dist = ((price - htf_low) / price) * 100
            if struct_dist > sl_pct * 1.2:
                struct_tp_pct = min(struct_dist, sl_pct * 3.5)
            if liq and any(p < price for p in liq.get("long_liq", [])):
                liq_target = max(p for p in liq["long_liq"] if p < price)
                liq_dist = ((price - liq_target) / price) * 100
                if liq_dist > struct_tp_pct and liq_dist < sl_pct * 4:
                    struct_tp_pct = liq_dist
        
        tp_pct = max(sl_pct * 1.3, struct_tp_pct)  # Minimum 1.3R
        
        # --- TRAILING STOP PARAMETERS (ML-adaptive) ---
        # Activation: earlier in trending, later in ranging
        ts_act = sl_pct * 0.4 if regime == "TRENDING" else sl_pct * 0.5
        ts_act = max(0.2, ts_act)  # Min 0.2% activation
        # Callback: tighter when ML confident
        ts_cb = base_atr_pct * 0.5 if (ml_prob > 0.65 or ml_prob < 0.35) else base_atr_pct * 0.7
        ts_cb = max(0.15, min(ts_cb, sl_pct * 0.4))
        
        # Breakout detection (validated with volume + body)
        prev_15_high = d1m['h'].iloc[-16:-1].max()
        prev_15_low = d1m['l'].iloc[-16:-1].min()
        raw_breakout = (direction == 1 and price > prev_15_high) or (direction == -1 and price < prev_15_low)
        # Real breakout = structure break + volume + strong body
        is_breakout = raw_breakout and vol_confirmed and body_ratio > 0.5
        
        # Logika Baru: Adaptive Threshold berdasarkan Regime & Breakout
        threshold = 62
        if regime == "VOLATILE":
            threshold = 58 if is_breakout else 72
        elif regime == "RANGING":
            threshold = 68
        elif regime == "TRENDING":
            threshold = 60 if is_breakout else 65
        
        # Reversal scalps need lower threshold (by definition counter-trend)
        if is_reversal_trade:
            threshold = max(threshold - 20, 48)
            
        # --- ADAPTIVE EXECUTION ENGINE ---
        is_market_order = False
        
        # Check score momentum (rising score = momentum building, don't miss)
        prev_cached = _analysis_cache.get(symbol)
        prev_score = prev_cached[1]["score"] if prev_cached and prev_cached[1] else 0
        score_rising = (score - prev_score) >= 8 and prev_score >= 55  # Jumped 8+ from already-decent
        
        # --- PULLBACK PROBABILITY ---
        _pb_score = 0
        if dist_from_ema21 > base_atr_pct * 1.5: _pb_score += 2
        if (direction == 1 and rsi_1m > 70) or (direction == -1 and rsi_1m < 30): _pb_score += 2
        if not vol_confirmed: _pb_score += 1
        # Consecutive candles check (compute inline)
        _last5c = d1m['c'].tail(5).values
        _last5o = d1m['o'].tail(5).values
        _greens = sum(1 for i in range(5) if _last5c[i] > _last5o[i])
        _reds = sum(1 for i in range(5) if _last5c[i] < _last5o[i])
        if (direction == 1 and _greens >= 4) or (direction == -1 and _reds >= 4): _pb_score += 2
        if vol_confirmed and body_ratio > 0.6: _pb_score -= 2
        if adx_val > 30: _pb_score -= 1
        
        pullback_likely = _pb_score >= 3  # High prob of pullback → use limit
        no_pullback = _pb_score <= 0      # Low prob → use market, don't miss
        
        # Market orders for high-conviction signals that pass ALL filters
        # 1. Classic breakout with extreme conviction
        if regime == "TRENDING" and is_breakout and score >= 72 and not is_fake_move:
            is_market_order = True
            limit_p = price
        # 2. All filters pass + very high score = market order (don't miss the move)
        elif score >= 80 and not is_fake_move and not is_overextended and vol_confirmed:
            is_market_order = True
            limit_p = price
        # 3. MTF fully aligned + trending + strong score = market order
        elif score >= 75 and mtf_aligned and regime == "TRENDING" and not is_fake_move and not is_overextended:
            is_market_order = True
            limit_p = price
        # 4. Score rising fast (momentum building) + trending = market order
        elif score_rising and score >= 72 and regime == "TRENDING" and not is_fake_move:
            is_market_order = True
            limit_p = price
        # 5. No pullback expected + good score = market order (don't miss the move)
        elif no_pullback and score >= 68 and not is_fake_move and not is_overextended:
            is_market_order = True
            limit_p = price
        # Everything else = limit order (sniper mode)

        signal = "WAIT"
        dist = abs(price - limit_p) / price * 100
        
        # --- CANDLE FRESHNESS CHECK ---
        # Only generate signal if last 1m candle is >45s old (data nearly final)
        # Prevents acting on incomplete candle that may reverse before close
        last_candle_age = (now * 1000 - d1m['ot'].iloc[-1]) / 1000  # seconds
        candle_too_young = last_candle_age < 8  # First 8s of candle = unreliable
        
        # --- MOMENTUM CONFIRMATION (Anti-Falling-Knife) ---
        # Don't BUY into consecutive red candles, don't SHORT into consecutive green
        last5_closes = d1m['c'].tail(5).values
        last5_opens = d1m['o'].tail(5).values
        consec_red = sum(1 for i in range(len(last5_closes)) if last5_closes[i] < last5_opens[i])
        consec_green = sum(1 for i in range(len(last5_closes)) if last5_closes[i] > last5_opens[i])
        falling_knife = (direction == 1 and consec_red >= 5)  # 5/5 red candles = falling
        rising_knife = (direction == -1 and consec_green >= 5)  # 5/5 green = pumping
        
        # For reversal trades: knife/cascading is CONFIRMATION not blocker
        if is_reversal_trade:
            falling_knife = False
            rising_knife = False
        
        # Also check: is price making lower lows (for long) or higher highs (for short)?
        last3_lows = d1m['l'].tail(3).values
        last3_highs = d1m['h'].tail(3).values
        # Cascading: only block if no strong reversal signal (VSA or liquidity sweep)
        _vsa_check = MarketAnalyzer.detect_vsa_signals(d1m)
        _sweep_check = MarketAnalyzer.detect_liquidity_sweep(d1m)
        _has_reversal_signal = (_vsa_check == direction) or (_sweep_check == direction)
        cascading_down = (last3_lows[2] < last3_lows[1] < last3_lows[0]) and direction == 1 and not is_reversal_trade and not _has_reversal_signal
        cascading_up = (last3_highs[2] > last3_highs[1] > last3_highs[0]) and direction == -1 and not is_reversal_trade and not _has_reversal_signal
        
        # --- SELL/BUY PRESSURE FILTER ---
        # Block LONG if sellers dominate, block SHORT if buyers dominate
        last10 = d1m.tail(10)
        _sell_vol = last10[last10['c'] < last10['o']]['v'].sum()
        _buy_vol = last10[last10['c'] >= last10['o']]['v'].sum()
        sell_pressure = (direction == 1 and _sell_vol > _buy_vol * 1.5) and not is_reversal_trade
        buy_pressure = (direction == -1 and _buy_vol > _sell_vol * 1.5) and not is_reversal_trade
        
        # --- RSI MOMENTUM DECLINING FILTER ---
        _rsi_now = rsi_15m
        _rsi_prev = ind15m["rsi_prev"]
        rsi_declining_long = (direction == 1 and _rsi_now < _rsi_prev and _rsi_now < 42) and not is_reversal_trade
        rsi_rising_short = (direction == -1 and _rsi_now > _rsi_prev and _rsi_now > 58) and not is_reversal_trade
        
        # --- EMA POSITION FILTER (reuse cached values) ---
        price_below_emas = (direction == 1 and price < ema9_1m and price < ema21_1m)
        price_above_emas = (direction == -1 and price > ema9_1m and price > ema21_1m)
        
        # Block entry if fake move or overextended
        if is_fake_move or is_overextended or candle_too_young:
            signal = "WAIT"
        elif falling_knife or rising_knife:
            signal = "WAIT"  # Don't catch falling knives
        elif cascading_down or cascading_up:
            signal = "WAIT"  # Price cascading against our direction
        elif sell_pressure or buy_pressure:
            signal = "WAIT"  # Opposing volume dominates
        elif rsi_declining_long or rsi_rising_short:
            signal = "WAIT"  # Momentum fading against our direction
        # ML VETO: block entry if ML strongly disagrees with direction
        elif (direction == 1 and ml_prob < 0.28) or (direction == -1 and ml_prob > 0.72):
            signal = "WAIT"  # ML confidence >28% against direction = veto
        elif score >= threshold:
            # Limit order must be at a meaningful distance (not just chasing)
            if is_market_order:
                signal = "SCALP-LONG" if direction == 1 else "SCALP-SHORT"
            elif dist >= 0.05 and dist <= 2.5:
                signal = "SCALP-LONG" if direction == 1 else "SCALP-SHORT"
            elif dist < 0.05:
                # Very close to level — use market order for all cases
                is_market_order = True
                limit_p = price
                signal = "SCALP-LONG" if direction == 1 else "SCALP-SHORT"
        
        result = {
            "sym": symbol.replace("USDT", ""), "price": f"{price:,.4f}", "limit": float(limit_p),
            "sig": signal, "score": int(score), "struct": struct[:4], "dir": int(direction), "dir_15m": int(direction),
            "regime": regime, "ai": {
                "tp": float(tp_pct), "sl": float(sl_pct), "limit_price": float(limit_p),
                "ts_act": float(ts_act), "ts_cb": float(ts_cb),
                "type": "SCALP", "regime": regime, "score": int(score), "ml_prob": float(round(ml_prob, 2)),
                "is_market": is_market_order, "atr_pct": float(base_atr_pct),
                "active_features": list(active_features),
                "brain_signals": brain.get("signals_fired", []),
            }
        }
        # Cache result keyed by last kline open time
        cache_ot = float(d1m.iloc[-1]['ot']) if not d1m.empty else 0
        _analysis_cache[symbol] = (cache_ot, result, time.time())
        return result
    except Exception as e:
        if symbol in busy_symbols: busy_symbols.discard(symbol)
        log_error(f"Analysis Crash ({symbol}): {str(e)}")
        return None
