import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time
from utils.state import market_data, bot_state

def get_current_session():
    """Returns the current global trading session."""
    now = datetime.utcnow().time()
    
    # Session Windows (UTC)
    # Asia: 00:00 - 09:00
    # London: 08:00 - 17:00
    # New York: 13:00 - 22:00
    
    if dt_time(13, 0) <= now <= dt_time(22, 0): return "NEW_YORK"
    if dt_time(8, 0) <= now <= dt_time(17, 0): return "LONDON"
    if dt_time(0, 0) <= now <= dt_time(9, 0): return "ASIA"
    return "QUIET"

def calculate_market_volatility():
    """Calculates a global volatility index based on top symbols."""
    vols = []
    for symbol in market_data.current_scan_list:
        df = market_data.klines.get(symbol, {}).get("15m")
        if df is not None and not df.empty:
            # 15m ATR / Price as a percentage
            high_low = df['h'] - df['l']
            avg_price = df['c'].rolling(14).mean()
            vol = (high_low.rolling(14).mean() / avg_price) * 100
            if not pd.isna(vol.iloc[-1]):
                vols.append(vol.iloc[-1])
    
    if vols:
        avg_vol = sum(vols) / len(vols)
        # Normalize: 1.0 is "normal", > 1.5 is high, < 0.7 is low
        bot_state["market_vol"] = round(avg_vol / 1.0, 2) # Assuming 1.0% is baseline
    return bot_state["market_vol"]

# Base sector mapping (fallback when clustering hasn't run yet)
_BASE_SECTORS = {
    "BTCUSDT": "L1", "ETHUSDT": "L1", "SOLUSDT": "L1", "ADAUSDT": "L1",
    "DOGEUSDT": "MEME", "SHIBUSDT": "MEME", "PEPEUSDT": "MEME", "WIFUSDT": "MEME", "FLOKIUSDT": "MEME",
    "LINKUSDT": "L1", "AVAXUSDT": "L1", "DOTUSDT": "L1", "MATICUSDT": "L1", "SUIUSDT": "L2", "ARBUSDT": "L2",
    "UNIUSDT": "DEFI", "AAVEUSDT": "DEFI", "MKRUSDT": "DEFI",
    "FETUSDT": "AI", "AGIXUSDT": "AI", "OCEANUSDT": "AI", "WLDUSDT": "AI", "RENDERUSDT": "AI"
}

# Dynamic clusters (updated by DynamicClusterer)
SECTORS = dict(_BASE_SECTORS)


class DynamicClusterer:
    """Auto-cluster coins by return correlation every 24h."""

    def __init__(self):
        self.last_run = 0
        self.interval = 86400  # 24h

    def maybe_recluster(self):
        """Run clustering if enough time has passed."""
        import time
        now = time.time()
        if now - self.last_run < self.interval:
            return
        self._run_clustering()
        self.last_run = now

    def _run_clustering(self):
        """Cluster symbols based on 15m return correlation."""
        global SECTORS
        try:
            # Collect returns from all symbols with 15m data
            returns_dict = {}
            for sym, klines in market_data.klines.items():
                df = klines.get("15m")
                if df is not None and len(df) >= 30:
                    ret = df['c'].pct_change().tail(30).dropna().values
                    if len(ret) >= 20:
                        returns_dict[sym] = ret

            if len(returns_dict) < 6:
                return  # Not enough data

            symbols = list(returns_dict.keys())
            # Build correlation matrix
            n = len(symbols)
            min_len = min(len(v) for v in returns_dict.values())
            matrix = np.array([returns_dict[s][-min_len:] for s in symbols])

            # Correlation-based distance
            corr = np.corrcoef(matrix)
            corr = np.nan_to_num(corr, nan=0.0)
            distance = 1 - corr

            # Simple agglomerative clustering (no scipy needed)
            n_clusters = min(6, n // 2)
            labels = self._simple_cluster(distance, n_clusters)

            # Assign cluster labels
            cluster_names = ["C0", "C1", "C2", "C3", "C4", "C5"]
            new_sectors = dict(_BASE_SECTORS)  # Start with base
            for i, sym in enumerate(symbols):
                cluster_id = labels[i]
                new_sectors[sym] = cluster_names[cluster_id % len(cluster_names)]

            SECTORS.update(new_sectors)
            bot_state["last_log"] = f"[cyan]CLUSTER: Regrouped {len(symbols)} coins into {n_clusters} clusters[/]"

        except Exception:
            pass  # Silently fail, keep old sectors

    def _simple_cluster(self, distance_matrix, n_clusters):
        """Simple greedy clustering without scipy."""
        n = len(distance_matrix)
        labels = list(range(n))  # Each point starts as its own cluster

        # Merge closest pairs until we have n_clusters
        while len(set(labels)) > n_clusters:
            min_dist = float('inf')
            merge_i, merge_j = 0, 1
            unique_labels = list(set(labels))

            for idx_a in range(len(unique_labels)):
                for idx_b in range(idx_a + 1, len(unique_labels)):
                    la, lb = unique_labels[idx_a], unique_labels[idx_b]
                    # Average linkage
                    points_a = [i for i, l in enumerate(labels) if l == la]
                    points_b = [i for i, l in enumerate(labels) if l == lb]
                    avg_dist = np.mean([distance_matrix[i][j] for i in points_a for j in points_b])
                    if avg_dist < min_dist:
                        min_dist = avg_dist
                        merge_i, merge_j = la, lb

            # Merge cluster merge_j into merge_i
            for i in range(n):
                if labels[i] == merge_j:
                    labels[i] = merge_i

        # Renumber labels to 0..n_clusters-1
        unique = list(set(labels))
        remap = {old: new for new, old in enumerate(unique)}
        return [remap[l] for l in labels]


dynamic_clusterer = DynamicClusterer()


def get_sector(symbol):
    return SECTORS.get(symbol, "UNKNOWN")

def get_symbol_correlation(sym1, sym2):
    """Calculates correlation between two symbols based on 15m klines."""
    try:
        df1 = market_data.klines.get(sym1, {}).get("15m")
        df2 = market_data.klines.get(sym2, {}).get("15m")
        
        if df1 is None or df2 is None or df1.empty or df2.empty:
            return 0.5 # Neutral if no data
            
        # Use returns for correlation
        ret1 = df1['c'].pct_change().tail(20)
        ret2 = df2['c'].pct_change().tail(20)
        
        corr = ret1.corr(ret2)
        return corr if not pd.isna(corr) else 0.5
    except:
        return 0.5

def is_correlated_exposure(new_symbol, new_side):
    """Checks if opening a new position adds too much correlation (H7)."""
    active_positions = bot_state.get("active_positions", [])
    if not active_positions:
        return False
        
    new_sector = get_sector(new_symbol)
    sector_count = 0
    
    for pos in active_positions:
        pos_sym = pos['symbol']
        pos_side = "LONG" if float(pos['positionAmt']) > 0 else "SHORT"
        
        # If same direction, check correlation and sector
        if pos_side == new_side:
            # 1. Check direct pair-wise correlation
            corr = get_symbol_correlation(new_symbol, pos_sym)
            if corr > 0.85: # Highly correlated
                return True
                
            # 2. Check sector exposure (max 2 per sector)
            if get_sector(pos_sym) == new_sector and new_sector != "UNKNOWN":
                sector_count += 1
                if sector_count >= 2:
                    return True
                
    return False

def calculate_kelly_risk(symbol, win_rate=0.5, rr=2.0):
    """Calculates a fractional Kelly multiplier based on historical performance."""
    try:
        # K = W - ((1 - W) / R)
        kelly = win_rate - ((1 - win_rate) / rr)
        # Use Half-Kelly for safety and cap it between 0.5x and 1.5x of base risk
        multiplier = max(0.5, min(1.5, (kelly * 0.5) / 0.125)) # 0.125 is half-kelly for 50% WR/2RR
        return round(multiplier, 2)
    except:
        return 1.0

def detect_lead_lag(symbol, leader="BTCUSDT"):
    """
    Robust lead-lag using correlation and return gap (H8).
    Enhanced with cascade chain: BTC → ETH → sector leaders → alts
    """
    try:
        if symbol == leader: return 0
        
        # Multi-level cascade: check ETH as intermediate leader for altcoins
        cascade_leaders = ["BTCUSDT", "ETHUSDT"]
        best_signal = 0
        
        for lead in cascade_leaders:
            if symbol == lead:
                continue
            df_sym = market_data.klines.get(symbol, {}).get("1m")
            df_lead = market_data.klines.get(lead, {}).get("1m")
            
            if df_sym is None or df_lead is None or len(df_sym) < 30 or len(df_lead) < 30:
                continue
                
            sym_ret = df_sym['c'].pct_change().tail(30).fillna(0)
            lead_ret = df_lead['c'].pct_change().tail(30).fillna(0)
            
            # Calculate Beta = Cov(sym, lead) / Var(lead)
            cov = sym_ret.cov(lead_ret)
            var_lead = lead_ret.var()
            beta = cov / var_lead if var_lead != 0 else 0
            
            # Only valid if they generally move together (beta > 0.5 for cascade)
            if beta > 0.5:
                # Check the gap over the last 3 candles (faster detection)
                sym_recent = (df_sym['c'].iloc[-1] - df_sym['c'].iloc[-4]) / df_sym['c'].iloc[-4] * 100
                lead_recent = (df_lead['c'].iloc[-1] - df_lead['c'].iloc[-4]) / df_lead['c'].iloc[-4] * 100
                
                std_gap = (lead_ret - sym_ret).std() * 100
                if std_gap == 0: continue
                
                # Cascade signal: leader moved, follower hasn't caught up
                gap = lead_recent - sym_recent
                z_score = gap / std_gap
                
                if z_score > 1.5:
                    signal = 1  # Bullish lag (leader up, alt hasn't followed)
                elif z_score < -1.5:
                    signal = -1  # Bearish lag
                else:
                    signal = 0
                
                # Stronger signal from BTC > ETH
                weight = 1.0 if lead == "BTCUSDT" else 0.7
                if abs(signal * weight) > abs(best_signal):
                    best_signal = signal
        
        return best_signal
    except:
        return 0


# ---------------------------------------------------------------------------
# Neural-weights feedback loop (Bayesian credit assignment)
# ---------------------------------------------------------------------------

def update_feature_weights(active_features, is_win, min_samples: int = 8,
                           min_weight: float = 0.4, max_weight: float = 1.6):
    """Credit/penalize each active feature based on trade outcome.

    - `strat_perf[feat] = [wins, losses]` is incremented.
    - `neural_weights[feat]` is recomputed from win-rate with a trust factor
      that grows with sample size (so cold features stay at 1.0).

    Weight formula (Laplace-smoothed win rate, centered at 0.5):
        wr      = (wins + 1) / (wins + losses + 2)
        trust   = min(1.0, (wins + losses) / 30)
        weight  = clip(1.0 + (wr - 0.5) * 2 * trust, min_weight, max_weight)

    So a feature that wins 70% over 30+ trades gets ~1.4; one that wins 30%
    gets ~0.6; fresh features stay at 1.0.
    """
    if not active_features:
        return
    perf = bot_state.setdefault("strat_perf", {})
    weights = bot_state.setdefault("neural_weights", {})
    for feat in active_features:
        if feat not in perf:
            perf[feat] = [0, 0]
        if is_win:
            perf[feat][0] += 1
        else:
            perf[feat][1] += 1
        w, l = perf[feat]
        total = w + l
        if total < min_samples:
            # Not enough data — keep neutral weight
            weights.setdefault(feat, 1.0)
            continue
        wr = (w + 1) / (total + 2)  # Laplace smoothing
        trust = min(1.0, total / 30.0)
        weight = 1.0 + (wr - 0.5) * 2.0 * trust
        weights[feat] = max(min_weight, min(max_weight, round(weight, 3)))
