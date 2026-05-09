"""
Auto-Parameter Optimization
============================
Runs backtest periodically (1x/day during quiet hours) and feeds optimal
parameters back into live bot config (threshold, SL/TP multipliers, etc).
"""
import asyncio
import time
import numpy as np
from utils.state import bot_state
from utils.logger import log_error


class AutoOptimizer:
    def __init__(self):
        self.last_run = 0
        self.interval = 86400  # 24h
        self.best_params = {}
        self.running = False

    async def run_optimization(self, client):
        """Run backtest with param sweep and update live config."""
        if self.running:
            return
        self.running = True
        try:
            result = await asyncio.to_thread(self._optimize_sync)
            if result:
                self.best_params = result
                self._apply_to_live(result)
                bot_state["last_log"] = f"[bold green]AUTO-OPT: Updated params (WR:{result.get('win_rate',0)*100:.0f}%)[/]"
            self.last_run = time.time()
        except Exception as e:
            log_error(f"AutoOptimizer Error: {str(e)}")
        finally:
            self.running = False

    def _optimize_sync(self):
        """Synchronous param sweep using backtest logic."""
        from strategies.analyzer import MarketAnalyzer
        import pandas as pd

        # Use cached kline data from market_data for quick backtest
        from utils.state import market_data

        # Collect available 15m data for top symbols
        test_data = {}
        for sym, klines in market_data.klines.items():
            df = klines.get("15m")
            if df is not None and len(df) >= 60:
                test_data[sym] = df.copy()

        if len(test_data) < 3:
            return None

        # Parameter grid (small, focused)
        param_grid = [
            {"threshold": 75, "sl_mult": 1.0, "tp_mult": 2.0, "trail_act": 0.7},
            {"threshold": 80, "sl_mult": 1.0, "tp_mult": 2.5, "trail_act": 0.8},
            {"threshold": 82, "sl_mult": 1.2, "tp_mult": 2.0, "trail_act": 0.8},
            {"threshold": 85, "sl_mult": 1.2, "tp_mult": 2.5, "trail_act": 0.9},
            {"threshold": 88, "sl_mult": 1.5, "tp_mult": 3.0, "trail_act": 1.0},
        ]

        best_score = -999
        best_params = param_grid[2]  # Default middle

        for params in param_grid:
            wins, losses = 0, 0
            total_pnl = 0.0

            for sym, df in test_data.items():
                if len(df) < 40:
                    continue
                atr = MarketAnalyzer.get_atr(df, 14)
                closes = df['c'].values
                highs = df['h'].values
                lows = df['l'].values

                for i in range(30, len(df) - 10):
                    # Simulate entry check
                    d_slice = df.iloc[i-30:i+1]
                    regime = MarketAnalyzer.detect_regime(d_slice)
                    struct, _, _, _ = MarketAnalyzer.detect_structure(d_slice)

                    # Simple score proxy
                    ema9 = closes[i-9:i+1].mean()
                    ema21 = closes[i-21:i+1].mean()
                    direction = 1 if ema9 > ema21 else -1

                    # Skip if no clear direction
                    if struct == "CHOP":
                        continue

                    score = 50
                    if (direction == 1 and struct == "BULLISH") or (direction == -1 and struct == "BEARISH"):
                        score += 30
                    if score < params["threshold"]:
                        continue

                    # Simulate trade outcome
                    entry = closes[i]
                    curr_atr = float(atr.iloc[i]) if i < len(atr) and not np.isnan(atr.iloc[i]) else entry * 0.01
                    sl_dist = curr_atr * params["sl_mult"]
                    tp_dist = curr_atr * params["tp_mult"]

                    hit_tp = False
                    for j in range(i+1, min(i+10, len(df))):
                        if direction == 1:
                            if highs[j] >= entry + tp_dist:
                                hit_tp = True
                                break
                            if lows[j] <= entry - sl_dist:
                                break
                        else:
                            if lows[j] <= entry - tp_dist:
                                hit_tp = True
                                break
                            if highs[j] >= entry + sl_dist:
                                break

                    if hit_tp:
                        wins += 1
                        total_pnl += tp_dist / entry * 100
                    else:
                        losses += 1
                        total_pnl -= sl_dist / entry * 100

            total = wins + losses
            if total < 10:
                continue
            wr = wins / total
            # Score = expectancy-weighted (win_rate * avg_win - loss_rate * avg_loss)
            expectancy = total_pnl / total
            score = expectancy * np.sqrt(total)  # Reward more trades too

            if score > best_score:
                best_score = score
                best_params = {**params, "win_rate": wr, "expectancy": expectancy, "n_trades": total}

        return best_params if best_score > 0 else None

    def _apply_to_live(self, params):
        """Apply optimized params to bot_state for live use."""
        bot_state["opt_params"] = {
            "threshold_adj": params.get("threshold", 80),
            "sl_mult": params.get("sl_mult", 1.0),
            "tp_mult": params.get("tp_mult", 2.0),
            "trail_act_mult": params.get("trail_act", 0.8),
        }

    async def maybe_run(self, client):
        """Check if it's time to run (quiet hours, once per day)."""
        now = time.time()
        if now - self.last_run < self.interval:
            return
        # Run during Asia session (quieter)
        from utils.intelligence import get_current_session
        if get_current_session() in ["ASIA", "QUIET"]:
            asyncio.create_task(self.run_optimization(client))


auto_optimizer = AutoOptimizer()
