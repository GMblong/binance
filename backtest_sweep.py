"""Multi-day walk-forward sweep of the honest backtester.

For each day D in [today - N days, today - 1 day]:
  1. Train ML on the 24h ending at midnight(D) UTC.
  2. Simulate the 24h of day D with the event-driven engine.
  3. Record per-day metrics.

Finally prints an aggregated report across all days so we can tell if
the edge is consistent or if the single-day result was luck.

Usage:
    python backtest_sweep.py                       # last 7 days, picker-selected
    python backtest_sweep.py --days 14
    python backtest_sweep.py --days 7 --symbols BTCUSDT,ETHUSDT,SOLUSDT
    python backtest_sweep.py --days 7 --ev 0.05    # relax EV gate
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from datetime import datetime, timezone

import httpx
import pandas as pd
from rich.console import Console
from rich.table import Table

from backtest import EventDrivenBacktester, CostModel, FillModel
from backtest.ml_helper import WalkForwardPredictor
from backtest.signal_adapter import make_signal_fn
from backtest.universe import select_liquid_trending
from backtest_honest import (
    fetch_klines,
    fetch_funding_history,
    session_from_ts,
    lead_lag_sim,
)
from engine.ml_engine import ml_predictor
from utils.config import API_URL
from utils.database import init_db, load_state_from_db

console = Console()
init_db()
load_state_from_db()


async def simulate_one_day(
    client, symbols, sim_end_ms, risk_percent, min_ev_pct, max_leverage,
    max_positions, starting_balance,
):
    sim_start_ms = sim_end_ms - 24 * 3600 * 1000
    train_end_ms = sim_start_ms

    # Reset ML models so each day gets its own freshly trained model (strict
    # walk-forward). Without this the predictor would keep stale weights
    # from earlier days.
    ml_predictor.models = {}
    helper = WalkForwardPredictor(client, ml_predictor, train_end_ms)
    await helper.train(symbols)

    btc_1m = await fetch_klines(client, "BTCUSDT", "1m", limit=1500, end_time=sim_end_ms)

    data = {}
    for symbol in symbols:
        d1m = await fetch_klines(client, symbol, "1m", limit=1500, end_time=sim_end_ms)
        d15m = await fetch_klines(client, symbol, "15m", limit=500, end_time=sim_end_ms)
        d1h = await fetch_klines(client, symbol, "1h", limit=200, end_time=sim_end_ms)
        if d1m.empty or d15m.empty or d1h.empty:
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
        return None

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
        sim_bars=24 * 60,
        signal_every_n_bars=5,
    )
    metrics = bt.run()
    return metrics, len(data)


async def run(days, symbols, risk_percent, min_ev_pct, max_leverage,
              max_positions, starting_balance):
    now_ms = int(time.time() * 1000)

    async with httpx.AsyncClient(timeout=30.0) as client:
        if not symbols:
            try:
                symbols = await select_liquid_trending(
                    client, API_URL, limit=5, debug=True
                )
            except Exception as exc:
                console.print(f"[yellow]Picker error: {exc}[/]")
                symbols = []
            if not symbols:
                symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
                console.print(
                    f"[yellow]Picker empty, fallback: {symbols}[/]"
                )
            else:
                console.print(f"Picker selected: {symbols}")
        else:
            console.print(f"Using symbols: {symbols}")

        per_day = []
        for i in range(days, 0, -1):
            sim_end_ms = now_ms - (i - 1) * 24 * 3600 * 1000
            date_str = datetime.fromtimestamp(sim_end_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            console.print(f"\n[bold cyan]-- Day {days - i + 1}/{days} ending {date_str} UTC --[/]")
            result = await simulate_one_day(
                client, symbols, sim_end_ms,
                risk_percent, min_ev_pct, max_leverage,
                max_positions, starting_balance,
            )
            if result is None:
                console.print("  [yellow]No data, skip.[/]")
                continue
            metrics, n_syms = result
            pf = metrics["profit_factor"]
            pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
            console.print(
                f"  trades={metrics['trades']}, WR={metrics['win_rate']*100:.1f}%, "
                f"PF={pf_s}, "
                f"NET=${metrics['net_pnl']:+.2f}, "
                f"fill={metrics['fill_rate']*100:.0f}%, symbols={n_syms}"
            )
            per_day.append({"date": date_str, **metrics})

    if not per_day:
        console.print("[red]No days produced metrics; aborting.[/]")
        return

    # Aggregate
    total_trades = sum(d["trades"] for d in per_day)
    total_wins = sum(d["trades"] * d["win_rate"] for d in per_day)
    total_pnl = sum(d["net_pnl"] for d in per_day)
    total_fees = sum(d["total_fees"] for d in per_day)
    daily_pnls = [d["net_pnl"] for d in per_day]
    profit_days = sum(1 for p in daily_pnls if p > 0)
    loss_days = sum(1 for p in daily_pnls if p < 0)

    mean_daily = statistics.mean(daily_pnls) if daily_pnls else 0.0
    stdev_daily = statistics.pstdev(daily_pnls) if len(daily_pnls) > 1 else 0.0
    t_stat = (mean_daily / (stdev_daily / (len(daily_pnls) ** 0.5))) if stdev_daily > 0 else 0.0

    daily_rets = [p / starting_balance for p in daily_pnls]
    mean_ret = statistics.mean(daily_rets) if daily_rets else 0.0
    stdev_ret = statistics.pstdev(daily_rets) if len(daily_rets) > 1 else 0.0
    sharpe_daily = (mean_ret / stdev_ret) * (365 ** 0.5) if stdev_ret > 0 else 0.0

    tbl = Table(title=f"Per-day results ({len(per_day)} days)")
    tbl.add_column("Date")
    tbl.add_column("Trades", justify="right")
    tbl.add_column("WR%", justify="right")
    tbl.add_column("PF", justify="right")
    tbl.add_column("NET $", justify="right")
    tbl.add_column("Fees $", justify="right")
    tbl.add_column("Fill%", justify="right")
    for d in per_day:
        color = "green" if d["net_pnl"] > 0 else ("red" if d["net_pnl"] < 0 else "white")
        pf = d["profit_factor"]
        pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        tbl.add_row(
            d["date"], str(d["trades"]), f"{d['win_rate']*100:.1f}",
            pf_s, f"[{color}]{d['net_pnl']:+.2f}[/]",
            f"{d['total_fees']:.2f}", f"{d['fill_rate']*100:.0f}",
        )
    console.print(tbl)

    overall_wr = (total_wins / total_trades * 100) if total_trades else 0.0

    color = "green" if total_pnl > 0 else "red"
    console.print("\n[bold]===== SWEEP AGGREGATE =====[/]")
    console.print(f"Days simulated : {len(per_day)}")
    console.print(f"Total trades   : {total_trades}")
    console.print(f"Overall WR     : {overall_wr:.1f}%")
    console.print(f"Profit days    : {profit_days}")
    console.print(f"Loss days      : {loss_days}")
    console.print(f"Mean daily PnL : ${mean_daily:+.3f}")
    console.print(f"Stdev daily    : ${stdev_daily:.3f}")
    console.print(f"t-stat vs 0    : {t_stat:.2f}  (|t|>2 ~ 95% significant)")
    console.print(f"Sharpe (daily->ann): {sharpe_daily:.2f}")
    console.print(f"Total fees     : ${total_fees:.2f}")
    console.print(f"[bold {color}]TOTAL NET PnL  : ${total_pnl:+.2f} "
                  f"({total_pnl / starting_balance * 100:+.2f}% of ${starting_balance:.0f})[/]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--ev", type=float, default=0.10)
    ap.add_argument("--max-lev", type=int, default=10)
    ap.add_argument("--balance", type=float, default=100.0)
    ap.add_argument("--max-positions", type=int, default=3)
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    asyncio.run(
        run(
            days=args.days,
            symbols=symbols,
            risk_percent=args.risk,
            min_ev_pct=args.ev,
            max_leverage=args.max_lev,
            max_positions=args.max_positions,
            starting_balance=args.balance,
        )
    )


if __name__ == "__main__":
    main()
