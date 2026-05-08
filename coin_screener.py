"""
Smart Coin Screener
-------------------
Menggantikan logika "top movers by abs(price_change%)" yang naif.

Masalah lama: Memilih coin hanya karena sudah bergerak banyak = sering terlambat.
Solusi: Multi-factor scoring yang mengidentifikasi coin SEBELUM breakout besar.

Faktor scoring:
1. Volume Surge (anomaly) - Volume spike relatif terhadap rata-rata = institutional interest
2. Volatility Expansion - ATR expanding = market sedang "waking up"  
3. Momentum Quality - Bukan hanya arah, tapi KUALITAS momentum (smooth vs choppy)
4. Relative Strength - Coin outperform BTC = ada demand spesifik
5. Sector Rotation - Sektor yang sedang "hot" mendapat bonus
6. Liquidity Score - Volume cukup untuk entry/exit tanpa slippage
"""
import numpy as np
import pandas as pd
from utils.state import market_data, bot_state
from utils.intelligence import SECTORS


def screen_coins(tickers: list, top_n: int = 10) -> list:
    """
    Mengembalikan list symbol terbaik berdasarkan multi-factor scoring.
    
    Args:
        tickers: List of ticker dicts [{s, q, c, o, cp}, ...]
        top_n: Jumlah coin yang dikembalikan
    Returns:
        List of symbol strings sorted by score (best first)
    """
    if not tickers:
        return []

    # Filter: minimum $10M volume for liquidity
    candidates = [
        t for t in tickers 
        if t.get("q", 0) > 10_000_000 
        and t["s"].endswith("USDT")
    ]
    if not candidates:
        return [t["s"] for t in tickers[:top_n]]

    scored = []
    btc_ret = _get_btc_return()

    # Pre-calculate sector heat
    sector_heat = _calc_sector_heat(candidates)

    for t in candidates:
        symbol = t["s"]
        score = _score_coin(t, btc_ret, sector_heat)
        if score > 0:
            scored.append((symbol, score))

    # Sort by score descending, return top_n
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s, _ in zip(scored, range(top_n))]


def _score_coin(ticker: dict, btc_ret: float, sector_heat: dict) -> float:
    """Multi-factor scoring untuk satu coin."""
    symbol = ticker["s"]
    score = 0.0

    cp = ticker.get("cp", 0)  # 24h price change %
    quote_vol = ticker.get("q", 0)

    # --- 1. Volume Surge Score (0-30) ---
    # Bandingkan volume saat ini vs historical dari klines jika tersedia
    k = market_data.klines.get(symbol, {})
    df_15m = k.get("15m")

    if df_15m is not None and len(df_15m) >= 20:
        recent_vol = df_15m["v"].tail(4).sum()  # Volume 1 jam terakhir
        avg_vol = df_15m["v"].tail(20).mean() * 4  # Rata-rata volume 1 jam
        if avg_vol > 0:
            vol_ratio = recent_vol / avg_vol
            # Scoring: ratio 1.5x = 10pts, 2x = 20pts, 3x+ = 30pts
            vol_score = min(30, max(0, (vol_ratio - 1.0) * 30))
            score += vol_score
    else:
        # Fallback: gunakan quote volume relatif terhadap median
        score += min(15, (quote_vol / 50_000_000) * 10)

    # --- 2. Volatility Expansion (0-20) ---
    if df_15m is not None and len(df_15m) >= 30:
        h, l, c = df_15m["h"], df_15m["l"], df_15m["c"]
        tr = pd.concat([(h - l).abs(), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr_recent = tr.tail(4).mean()
        atr_baseline = tr.tail(20).mean()
        if atr_baseline > 0:
            expansion = atr_recent / atr_baseline
            # ATR expanding = market waking up, good for trading
            if expansion > 1.2:
                score += min(20, (expansion - 1.0) * 20)

    # --- 3. Momentum Quality (0-25) ---
    # Bukan hanya magnitude, tapi smoothness dan recency
    if df_15m is not None and len(df_15m) >= 10:
        closes = df_15m["c"].tail(10)
        returns = closes.pct_change().dropna()
        if len(returns) >= 5:
            # Directional consistency: berapa banyak candle searah
            pos_count = (returns > 0).sum()
            neg_count = (returns < 0).sum()
            consistency = max(pos_count, neg_count) / len(returns)

            # Magnitude of recent move
            move_5 = abs((closes.iloc[-1] - closes.iloc[-5]) / closes.iloc[-5]) * 100

            # Sweet spot: moderate move (0.5-3%) with high consistency
            if 0.3 < move_5 < 5.0 and consistency >= 0.6:
                momentum_score = min(25, move_5 * 5 * consistency)
                score += momentum_score
            elif move_5 >= 5.0:
                # Sudah terlalu jauh = penalti (late entry risk)
                score += 5
    else:
        # Fallback: gunakan 24h change tapi dengan diminishing returns
        abs_cp = abs(cp)
        if 0.5 < abs_cp < 5.0:
            score += min(15, abs_cp * 3)
        elif abs_cp >= 5.0:
            score += 5  # Penalti untuk yang sudah terlalu jauh

    # --- 4. Relative Strength vs BTC (0-15) ---
    if btc_ret is not None and cp != 0:
        # Coin yang outperform BTC = ada demand spesifik
        excess_return = abs(cp) - abs(btc_ret)
        if excess_return > 0.5:
            score += min(15, excess_return * 5)

    # --- 5. Sector Rotation Bonus (0-10) ---
    sector = SECTORS.get(symbol, "UNKNOWN")
    if sector != "UNKNOWN" and sector in sector_heat:
        heat = sector_heat[sector]
        if heat > 0.6:  # Sektor sedang hot
            score += min(10, heat * 10)

    return score


def _get_btc_return() -> float:
    """Ambil return BTC 24h dari tickers."""
    for t in market_data.tickers:
        if t["s"] == "BTCUSDT":
            return t.get("cp", 0)
    return 0.0


def _calc_sector_heat(candidates: list) -> dict:
    """
    Hitung 'heat' per sektor: rata-rata abs return dari coin di sektor tersebut.
    Sektor dengan banyak coin bergerak = sedang ada rotasi ke sana.
    """
    sector_returns = {}
    for t in candidates:
        sector = SECTORS.get(t["s"], "UNKNOWN")
        if sector == "UNKNOWN":
            continue
        if sector not in sector_returns:
            sector_returns[sector] = []
        sector_returns[sector].append(abs(t.get("cp", 0)))

    sector_heat = {}
    for sector, rets in sector_returns.items():
        if len(rets) >= 2:
            avg = sum(rets) / len(rets)
            # Normalize: 2% avg move = heat 1.0
            sector_heat[sector] = min(1.0, avg / 2.0)
        else:
            sector_heat[sector] = 0.3  # Default low heat

    return sector_heat
