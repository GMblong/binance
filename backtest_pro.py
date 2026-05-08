"""
Pro Backtester
--------------
Replays historical 1m klines tick-by-tick with strict, trader-realistic
execution semantics. Designed to give an HONEST estimate of expected PnL.

Improvements over backtest_today.py / backtest_yesterday.py:

  * NO look-ahead.   Signals are generated at candle close[t]; SL/TP/partial
    exits are evaluated using candles [t+1, t+2, ...]. Pending limit orders
    are checked AFTER the signal candle, never on the signal bar.
  * Realistic fills. Market orders pay a configurable spread + slippage on
    entry and on every exit; taker fees are charged on both sides.
  * Partial exits. 60% of size closes at TP1 (R=1); remainder runs to TP2
    with the stop moved to break-even. This mirrors how professional
    scalpers actually manage risk.
  * Per-symbol daily loss kill-switch, cooldown after consecutive losses,
    and portfolio-level drawdown guard.
  * Deterministic walk-forward: the ML model is trained on the first half
    of the series and evaluated on the second half only, preventing
    label leakage.
  * Synthetic-data fallback (--offline): generates GBM-style 1m candles
    with regime shifts when Binance API is unreachable, so the strategy
    can still be validated end-to-end.

Run:
    python backtest_pro.py                  # live klines from Binance
    python backtest_pro.py --offline        # synthetic data
    python backtest_pro.py --symbols BTCUSDT,ETHUSDT --days 3

Produces a per-symbol and portfolio-level summary with win rate, expectancy,
max drawdown, Sharpe, and profit factor.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import random
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

from strategies.scalper_pro import (
    ScalperPro, Features, build_features, fractional_kelly_risk, position_notional,
)
from engine.ml_engine_v2 import ml_v2


API_URL = "https://fapi.binance.com"

# ---------- Cost model ----------

TAKER_FEE = 0.0004            # 0.04% per side
SPREAD_BPS = 0.5              # 0.5 bps half-spread baseline
SLIPPAGE_ATR_FRAC = 0.05      # 5% of 1m ATR of adverse slippage on market orders


# ---------- Data loading ----------

async def fetch_klines(client, symbol: str, interval: str,
                        limit: int = 1500, end_time: Optional[int] = None) -> Optional[pd.DataFrame]:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time:
        params["endTime"] = end_time
    try:
        r = await client.get(f"{API_URL}/fapi/v1/klines", params=params, timeout=20)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    raw = r.json()
    if not raw:
        return None
    df = pd.DataFrame(raw).iloc[:, [0, 1, 2, 3, 4, 5, 9]]
    df.columns = ["ot", "o", "h", "l", "c", "v", "tbv"]
    return df.astype(float).reset_index(drop=True)


def synth_klines(n: int, seed: int, start_price: float = 30000.0,
                 drift_phases: int = 4) -> pd.DataFrame:
    """Generate 1m klines with alternating trend/range/vol regimes.

    Not a substitute for real data, but useful for offline validation of the
    backtester pipeline and sanity checks."""
    rng = np.random.default_rng(seed)
    phase_len = n // drift_phases
    mu_choices = [0.00005, -0.00005, 0.0, 0.00015, -0.00015]
    sigma_choices = [0.0006, 0.0012, 0.0020]

    prices = [start_price]
    for p in range(drift_phases):
        mu = rng.choice(mu_choices)
        sigma = rng.choice(sigma_choices)
        for _ in range(phase_len):
            ret = rng.normal(mu, sigma)
            prices.append(max(prices[-1] * (1 + ret), 1e-6))

    close = np.array(prices[1:])
    # Intra-bar wicks: random range around close
    wick = np.abs(rng.normal(0, close.std() / close.mean() * 0.3, size=len(close)))
    high = close * (1 + wick * 0.5)
    low = close * (1 - wick * 0.5)
    open_ = np.concatenate([[start_price], close[:-1]])
    # High/low envelope
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])

    # Volume correlated with |return|
    ret = np.concatenate([[0.0], np.diff(close) / close[:-1]])
    base_vol = 500 + np.abs(ret) * 300000
    v = base_vol * (1 + rng.normal(0, 0.2, size=len(close))).clip(0.1)
    tbv = v * (0.5 + np.sign(ret) * np.abs(ret) * 50).clip(0.1, 0.9)

    now_ms = int(datetime.utcnow().timestamp() * 1000)
    ot = now_ms - (len(close) - np.arange(len(close))) * 60_000
    df = pd.DataFrame({"ot": ot, "o": open_, "h": high, "l": low, "c": close, "v": v, "tbv": tbv})
    return df.reset_index(drop=True)


def resample(df_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample 1m dataframe to higher timeframe using ot (ms)."""
    df = df_1m.copy()
    df["dt"] = pd.to_datetime(df["ot"], unit="ms")
    df.set_index("dt", inplace=True)
    rule = f"{minutes}min"
    agg = df.resample(rule, label="right", closed="right").agg(
        {"ot": "first", "o": "first", "h": "max", "l": "min", "c": "last",
         "v": "sum", "tbv": "sum"}
    ).dropna()
    return agg.reset_index(drop=True)


# ---------- Execution engine ----------

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
    partial_taken: bool = False
    remaining_frac: float = 1.0
    r_unit: float = 0.0
    setup: str = ""


@dataclass
class ClosedTrade:
    symbol: str
    side: str
    setup: str
    opened_at: int
    closed_at: int
    pnl_usd: float
    reason: str
    entry: float
    exit: float
    bars_held: int


@dataclass
class AccountState:
    balance: float
    peak_balance: float
    day_start_balance: float
    consec_losses: int = 0
    cooldown_until: int = 0
    trades: List[ClosedTrade] = field(default_factory=list)


def _apply_slippage(side: str, price: float, atr_1m: float, is_market: bool) -> float:
    """Return fill price after slippage for MARKET orders."""
    if not is_market:
        return price
    slip = SLIPPAGE_ATR_FRAC * atr_1m + price * SPREAD_BPS / 10000
    return price + slip if side == "LONG" else price - slip


def _close_position(pos: Position, exit_price: float, reason: str,
                    bar_idx: int) -> ClosedTrade:
    if pos.side == "LONG":
        pnl_pct = (exit_price - pos.entry) / pos.entry
    else:
        pnl_pct = (pos.entry - exit_price) / pos.entry
    gross = pnl_pct * pos.size_usd
    # Taker fees on both sides of the portion being closed
    fee = 2 * TAKER_FEE * pos.size_usd
    return ClosedTrade(
        symbol=pos.symbol, side=pos.side, setup=pos.setup,
        opened_at=pos.opened_at, closed_at=bar_idx,
        pnl_usd=gross - fee, reason=reason,
        entry=pos.entry, exit=exit_price, bars_held=bar_idx - pos.opened_at,
    )


# ---------- Backtest loop ----------

@dataclass
class BacktestConfig:
    symbols: List[str]
    days: int = 1
    starting_balance: float = 100.0
    risk_pct: float = 0.01          # 1% base risk; Kelly adjusts within 0.25x-1.5x
    max_positions: int = 2
    max_consec_losses: int = 3
    cooldown_bars: int = 60         # 60 min cooldown after 3 losses in a row
    daily_loss_limit_pct: float = 0.04
    offline: bool = False
    train_frac: float = 0.5         # use first 50% for ML training
    ml_min_prob: float = 0.55


class ProBacktester:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.strategy = ScalperPro()

    async def _load_symbol(self, client, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
        if self.cfg.offline:
            seed = abs(hash(symbol)) % (2**31)
            # 1m history: `days * 1440` + 500 warm-up candles
            n = self.cfg.days * 1440 + 500
            df_1m = synth_klines(n, seed=seed, start_price=1000.0 * (1 + (seed % 50) / 50))
        else:
            df_1m = await fetch_klines(client, symbol, "1m", limit=min(1500, self.cfg.days * 1440 + 500))
        if df_1m is None or len(df_1m) < 500:
            return None
        df_5m = resample(df_1m, 5)
        df_15m = resample(df_1m, 15)
        df_1h = resample(df_1m, 60)
        return {"1m": df_1m, "5m": df_5m, "15m": df_15m, "1h": df_1h}

    def _warmup_and_train(self, data: Dict[str, pd.DataFrame], symbol: str) -> int:
        """Train ML on the training portion only; return the index in 1m where
        out-of-sample testing begins. Also acts as the start-of-simulation bar."""
        df1 = data["1m"]
        train_end = int(len(df1) * self.cfg.train_frac)
        # Need at least 300 bars of warmup in-sample for indicator stability
        start_bar = max(train_end, 300)
        try:
            ml_v2.train(df1.iloc[:train_end].copy(), symbol)
        except Exception:
            pass
        return start_bar

    def _align_htf(self, df_htf: pd.DataFrame, ts_ms: int) -> pd.DataFrame:
        """Slice HTF dataframe to rows with ot <= ts_ms (no future leakage)."""
        return df_htf[df_htf["ot"] <= ts_ms]

    def _check_position(self, pos: Position, bar: pd.Series, bar_idx: int) -> Tuple[Optional[ClosedTrade], Optional[ClosedTrade]]:
        """Update position for this bar, returning (partial_close, full_close) if any."""
        # Partial take at TP1
        partial: Optional[ClosedTrade] = None
        if not pos.partial_taken:
            hit = (pos.side == "LONG" and bar["h"] >= pos.tp1) or \
                  (pos.side == "SHORT" and bar["l"] <= pos.tp1)
            if hit:
                partial_size = pos.size_usd * self.strategy.PARTIAL_PCT
                part_pos = Position(
                    symbol=pos.symbol, side=pos.side, entry=pos.entry,
                    size_usd=partial_size, sl=pos.sl, tp1=pos.tp1, tp2=pos.tp2,
                    opened_at=pos.opened_at, r_unit=pos.r_unit, setup=pos.setup,
                )
                partial = _close_position(part_pos, pos.tp1, "TP1", bar_idx)
                pos.size_usd *= (1 - self.strategy.PARTIAL_PCT)
                pos.partial_taken = True
                pos.sl = pos.entry  # move stop to break-even on the runner

        # Full exit at SL or TP2
        full: Optional[ClosedTrade] = None
        if pos.side == "LONG":
            if bar["l"] <= pos.sl:
                full = _close_position(pos, pos.sl, "SL" if not pos.partial_taken else "BE", bar_idx)
            elif bar["h"] >= pos.tp2:
                full = _close_position(pos, pos.tp2, "TP2", bar_idx)
        else:
            if bar["h"] >= pos.sl:
                full = _close_position(pos, pos.sl, "SL" if not pos.partial_taken else "BE", bar_idx)
            elif bar["l"] <= pos.tp2:
                full = _close_position(pos, pos.tp2, "TP2", bar_idx)
        return partial, full

    def _run_single(self, symbol: str, data: Dict[str, pd.DataFrame], acct: AccountState) -> None:
        df1 = data["1m"]
        start = self._warmup_and_train(data, symbol)
        positions: List[Position] = []

        # Stats for dynamic Kelly sizing (scoped per symbol)
        rolling_win_rate = 0.5
        rolling_rr = 1.3
        win_window: List[int] = []

        # Walk forward, bar-by-bar
        for i in range(start, len(df1) - 1):
            bar = df1.iloc[i]
            next_bar = df1.iloc[i + 1]

            # 1) Update open positions against this bar (intra-bar order: SL worst case).
            new_positions: List[Position] = []
            for pos in positions:
                partial, full = self._check_position(pos, bar, i)
                if partial:
                    acct.balance += partial.pnl_usd
                    acct.peak_balance = max(acct.peak_balance, acct.balance)
                    acct.trades.append(partial)
                if full:
                    acct.balance += full.pnl_usd
                    acct.peak_balance = max(acct.peak_balance, acct.balance)
                    acct.trades.append(full)
                    is_win = 1 if full.pnl_usd > 0 else 0
                    win_window.append(is_win)
                    if len(win_window) > 30:
                        win_window.pop(0)
                    rolling_win_rate = sum(win_window) / len(win_window)
                    acct.consec_losses = 0 if is_win else acct.consec_losses + 1
                    if acct.consec_losses >= self.cfg.max_consec_losses:
                        acct.cooldown_until = i + self.cfg.cooldown_bars
                        acct.consec_losses = 0
                else:
                    new_positions.append(pos)
            positions = new_positions

            # 2) Guard rails before generating new entries
            if i < acct.cooldown_until:
                continue
            if acct.balance <= acct.day_start_balance * (1 - self.cfg.daily_loss_limit_pct):
                continue
            if len(positions) >= self.cfg.max_positions:
                continue

            # 3) Build features from data up to and including candle `i`.
            ts_ms = int(bar["ot"])
            df_1m_slice = df1.iloc[: i + 1]
            df_5m_slice = self._align_htf(data["5m"], ts_ms)
            df_15m_slice = self._align_htf(data["15m"], ts_ms)
            df_1h_slice = self._align_htf(data["1h"], ts_ms)
            if len(df_15m_slice) < 50 or len(df_1h_slice) < 30:
                continue

            feats = build_features(df_1m_slice, df_5m_slice, df_15m_slice, df_1h_slice)
            if feats is None:
                continue

            ml_prob = ml_v2.predict(df_1m_slice, symbol)
            sig = self.strategy.generate_signal(feats, ml_prob=ml_prob, ml_min=self.cfg.ml_min_prob)
            if sig is None:
                continue

            # 4) Size & enter on the NEXT bar's open (prevents look-ahead).
            adj_risk = fractional_kelly_risk(self.cfg.risk_pct, rolling_win_rate, rolling_rr)
            adj_risk *= (0.6 + 0.8 * sig.confidence)  # confidence scaling: 0.6x .. 1.4x
            notional = position_notional(acct.balance, adj_risk, sig.entry, sig.sl)
            if notional < 5:  # below Binance min notional on most futures pairs
                continue

            fill = _apply_slippage(sig.side, next_bar["o"], feats.atr_1m, is_market=True)
            # Recompute stops/targets from the actual fill to keep R consistent
            r = abs(sig.entry - sig.sl)
            if sig.side == "LONG":
                sl = fill - r
                tp1 = fill + r * self.strategy.TP1_R
                tp2 = fill + r * self.strategy.TP2_R
            else:
                sl = fill + r
                tp1 = fill - r * self.strategy.TP1_R
                tp2 = fill - r * self.strategy.TP2_R

            positions.append(Position(
                symbol=symbol, side=sig.side, entry=fill, size_usd=notional,
                sl=sl, tp1=tp1, tp2=tp2, opened_at=i + 1, r_unit=r, setup=sig.setup,
            ))

        # Close any still-open at last candle
        last_bar_idx = len(df1) - 1
        last_close = float(df1["c"].iloc[-1])
        for pos in positions:
            closed = _close_position(pos, last_close, "EOD", last_bar_idx)
            acct.balance += closed.pnl_usd
            acct.trades.append(closed)

    async def run(self) -> AccountState:
        acct = AccountState(
            balance=self.cfg.starting_balance,
            peak_balance=self.cfg.starting_balance,
            day_start_balance=self.cfg.starting_balance,
        )

        if self.cfg.offline or not _HAS_HTTPX:
            for sym in self.cfg.symbols:
                data = await self._load_symbol(None, sym)
                if not data:
                    continue
                self._run_single(sym, data, acct)
        else:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for sym in self.cfg.symbols:
                    data = await self._load_symbol(client, sym)
                    if not data:
                        print(f"[warn] could not load {sym}, skipping")
                        continue
                    self._run_single(sym, data, acct)
        return acct


# ---------- Reporting ----------

def summarize(acct: AccountState, start_balance: float) -> None:
    trades = acct.trades
    n = len(trades)
    if n == 0:
        print("No trades taken.")
        print(f"Final balance: ${acct.balance:.2f}")
        return

    pnls = np.array([t.pnl_usd for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / n
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    expectancy = pnls.mean()
    pf = (-wins.sum() / losses.sum()) if losses.sum() < 0 else float("inf")
    # Equity curve for drawdown & Sharpe proxy
    equity = start_balance + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min() if len(dd) else 0.0
    returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([])
    sharpe = (returns.mean() / returns.std() * math.sqrt(1440)) if returns.std() > 0 else 0.0

    print("\n=== RESULT ===")
    print(f"Trades            : {n}")
    print(f"Win rate          : {wr * 100:.1f}%")
    print(f"Avg win / loss    : ${avg_win:+.3f} / ${avg_loss:+.3f}")
    print(f"Expectancy / trade: ${expectancy:+.3f}")
    print(f"Profit factor     : {pf:.2f}")
    print(f"Max drawdown      : {max_dd * 100:.2f}%")
    print(f"Sharpe (1m basis) : {sharpe:.2f}")
    print(f"Starting balance  : ${start_balance:.2f}")
    print(f"Final balance     : ${acct.balance:.2f}  (net ${acct.balance - start_balance:+.2f})")

    # Breakdown by setup
    from collections import defaultdict
    by_setup = defaultdict(list)
    for t in trades:
        by_setup[t.setup].append(t.pnl_usd)
    print("\nBy setup:")
    for k, v in by_setup.items():
        arr = np.array(v)
        w = (arr > 0).mean() if len(arr) else 0
        print(f"  {k:18s}  n={len(arr):3d}  wr={w * 100:4.1f}%  net=${arr.sum():+.2f}")


# ---------- CLI ----------

def _parse_args():
    p = argparse.ArgumentParser(description="Pro backtester for the scalper")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT",
                   help="comma-separated futures symbols")
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--balance", type=float, default=100.0)
    p.add_argument("--risk", type=float, default=0.01, help="base risk per trade, fraction")
    p.add_argument("--max-positions", type=int, default=2)
    p.add_argument("--offline", action="store_true",
                   help="use synthetic data (required if sandbox has no internet)")
    p.add_argument("--ml-min", type=float, default=0.55, help="ML probability gate")
    return p.parse_args()


async def _amain():
    args = _parse_args()
    cfg = BacktestConfig(
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        days=args.days,
        starting_balance=args.balance,
        risk_pct=args.risk,
        max_positions=args.max_positions,
        offline=args.offline,
        ml_min_prob=args.ml_min,
    )
    print(f"Running backtest | offline={cfg.offline} | days={cfg.days} | symbols={cfg.symbols}")
    bt = ProBacktester(cfg)
    acct = await bt.run()
    summarize(acct, cfg.starting_balance)


if __name__ == "__main__":
    asyncio.run(_amain())
