"""Honest, look-ahead-free backtester.

Usage:
    python backtest_honest.py             # default: yesterday, 5 symbols
    python backtest_honest.py --today     # simulate the last 24h (needs
                                          # train cutoff < sim start)
    python backtest_honest.py --symbols BTCUSDT,ETHUSDT,SOLUSDT

Compared with the legacy backtest_today.py / backtest_yesterday.py this
script:

  1. Trains the ML model strictly before the simulation window (no
     look-ahead).
  2. Uses realistic cost (taker + maker + slippage) and fill (limit fills
     only on penetration) models.
  3. Enforces a hard multi-timeframe gate and an EV gate on every trade.
  4. Caps leverage adaptively so a whipsaw cannot liquidate the account
     before SL is hit.
  5. Reports Sharpe, Sortino, Profit Factor, Max DD, fill rate, etc.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime
from datetime import time as dt_time

import httpx
import pandas as pd
from rich.console import Console

from backtest import EventDrivenBacktester, CostModel, FillModel, print_report
from backtest.ml_helper import WalkForwardPredictor
from backtest.signal_adapter import make_signal_fn
from backtest.universe import select_liquid_trending
from engine.ml_engine import ml_predictor
from strategies.analyzer import MarketAnalyzer
from utils.config import API_URL
from utils.database import init_db, load_state_from_db

console = Console()
init_db()
load_state_from_db()


def session_from_ts(ts_ms: int) -> str:
    t = datetime.utcfromtimestamp(ts_ms / 1000).time()
    if dt_time(13, 0) <= t <= dt_time(22, 0):
        return "NEW_YORK"
    if dt_time(8, 0) <= t <= dt_time(17, 0):
        return "LONDON"
    if dt_time(0, 0) <= t <= dt_time(9, 0):
        return "ASIA"
    return "QUIET"


def lead_lag_sim(sub_sym: pd.DataFrame, sub_lead: pd.DataFrame) -> int:
    if len(sub_sym) < 5 or len(sub_lead) < 5:
        return 0
    sym_ret = (sub_sym["c"].iloc[-1] - sub_sym["c"].iloc[-5]) / sub_sym["c"].iloc[-5] * 100
    lead_ret = (sub_lead["c"].iloc[-1] - sub_lead["c"].iloc[-5]) / sub_lead["c"].iloc[-5] * 100
    if lead_ret > 0.3 and sym_ret < 0.1:
        return 1
    if lead_ret < -0.3 and sym_ret > -0.1:
        return -1
    return 0


async def fetch_klines(client, symbol, interval, limit=1500, end_time=None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time:
        params["endTime"] = end_time
    res = await client.get(f"{API_URL}/fapi/v1/klines", params=params)
    df = pd.DataFrame(res.json()).iloc[:, [0, 1, 2, 3, 4, 5]]
    df.columns = ["ot", "o", "h", "l", "c", "v"]
    return df.astype(float)


async def fetch_funding_history(client, symbol, end_time=None):
    try:
        params = {"symbol": symbol, "limit": 500}
        if end_time:
            params["endTime"] = end_time
        res = await client.get(f"{API_URL}/fapi/v1/fundingRate", params=params)
        rows = res.json()
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["fundingTime"] = df["fundingTime"].astype("int64")
        df["fundingRate"] = df["fundingRate"].astype(float)
        s = pd.Series(df["fundingRate"].values, index=df["fundingTime"].values).sort_index()
        return s
    except Exception:
        return None


async def run(symbols, sim_window_hours=24, train_lookback_hours=24,
              use_today=False, max_positions=3, risk_percent=0.01,
              min_ev_pct=0.10, max_leverage=10, starting_balance=100.0):
    """Single backtest iteration.

    If `use_today`, we simulate the *past* `sim_window_hours` hours ending
    now, and train on the `train_lookback_hours` window ending just before
    the sim start (so NO look-ahead).

    If not `use_today` (default), we simulate yesterday (i.e. 24h ending
    at "now - sim_window_hours"), and train on the 24h before that.
    """
    now_ms = int(time.time() * 1000)
    if use_today:
        sim_end_ms = now_ms
    else:
        sim_end_ms = now_ms - sim_window_hours * 3600 * 1000
    sim_start_ms = sim_end_ms - sim_window_hours * 3600 * 1000
    # Train strictly BEFORE sim_start_ms.
    train_end_ms = sim_start_ms  # ml fetches kandle end_time <= train_end_ms

    console.print(
        f"[bold]Sim window[/]: {datetime.utcfromtimestamp(sim_start_ms/1000)} UTC "
        f"-> {datetime.utcfromtimestamp(sim_end_ms/1000)} UTC"
    )
    console.print(
        f"[bold]Train cutoff[/]: {datetime.utcfromtimestamp(train_end_ms/1000)} UTC "
        f"(ALL ML training data ends here)"
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        if not symbols:
            symbols = await select_liquid_trending(client, API_URL, limit=5)
            console.print(f"Liquid-trending picker selected: {symbols}")
        else:
            console.print(f"Using symbols: {symbols}")

        # Train ML BEFORE loading sim data.
        helper = WalkForwardPredictor(client, ml_predictor, train_end_ms)
        console.print("[dim]Training ML models (walk-forward, strict cutoff)...[/dim]")
        await helper.train(symbols)

        # Fetch sim-window data for each symbol.
        # We need 1500 1m candles ending at sim_end_ms so the first sim bar
        # has enough context (240 ema etc.).
        data = {}
        btc_1m = await fetch_klines(client, "BTCUSDT", "1m", limit=1500, end_time=sim_end_ms)

        for symbol in symbols:
            d1m = await fetch_klines(client, symbol, "1m", limit=1500, end_time=sim_end_ms)
            d15m = await fetch_klines(client, symbol, "15m", limit=500, end_time=sim_end_ms)
            d1h = await fetch_klines(client, symbol, "1h", limit=200, end_time=sim_end_ms)
            if d1m.empty or d15m.empty or d1h.empty:
                console.print(f"[yellow]Skip {symbol}: empty data[/]")
                continue
            funding = await fetch_funding_history(client, symbol, end_time=sim_end_ms)
            data[symbol] = {
                "1m": d1m,
                "15m": d15m,
                "1h": d1h,
                "funding": funding,
                "tick": d1m["c"].iloc[-1] * 1e-4,
            }

        if not data:
            console.print("[red]No data fetched, aborting.[/]")
            return

        cost = CostModel()
        fill = FillModel(require_penetration=True, penetration_ticks=1)

        signal_fn = make_signal_fn(
            ml_prob_fn=helper.predict_prob,
            risk_percent=risk_percent,
            min_ev_pct=min_ev_pct,
            max_leverage=max_leverage,
            cost_model=cost,
            session_fn=session_from_ts,
            lead_lag_fn=lead_lag_sim,
            btc_1m=btc_1m,
        )

        bt = EventDrivenBacktester(
            data=data,
            signal_fn=signal_fn,
            starting_balance=starting_balance,
            cost_model=cost,
            fill_model=fill,
            max_positions=max_positions,
            sim_bars=sim_window_hours * 60,
            signal_every_n_bars=5,
        )
        console.print("[cyan]Running event-driven simulation...[/cyan]")
        metrics = bt.run()

        extra = {
            "Fill rate": f"{metrics['fill_rate']*100:.1f}%",
            "Signals gen": metrics["signals"],
            "Limit placed": metrics["limit_placed"],
            "Limit filled": metrics["limit_filled"],
            "Limit expired": metrics["limit_expired"],
            "Start balance": f"${metrics['start_balance']:.2f}",
            "Final balance": f"${metrics['final_balance']:.2f}",
        }
        title = "BACKTEST (HONEST, {} SYMBOLS)".format(len(data))
        print_report(console, title, metrics, extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", action="store_true")
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--ev", type=float, default=0.10)
    ap.add_argument("--max-lev", type=int, default=10)
    ap.add_argument("--balance", type=float, default=100.0)
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    asyncio.run(
        run(
            symbols=symbols,
            sim_window_hours=args.hours,
            use_today=args.today,
            risk_percent=args.risk,
            min_ev_pct=args.ev,
            max_leverage=args.max_lev,
            starting_balance=args.balance,
        )
    )


if __name__ == "__main__":
    main()
