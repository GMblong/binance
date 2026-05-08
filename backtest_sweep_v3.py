"""Multi-day walk-forward sweep for v3 (fusion + online learning + maker-TP).

Unlike `backtest_sweep_v2.py`, the OnlineLearner state is created ONCE
at the start of the sweep and carried across all days. This matches
the live-deployment scenario where the online head keeps learning from
yesterday's trades when it runs today. If you want the no-memory
baseline instead, pass `--reset-online on` (useful for isolating the
per-day contribution of online learning).

Each day:
  1. Re-trains the frozen LightGBM (ml_v2) on the most recent
     `train_chunks * 1500` bars ending at that day's 00:00.
  2. Runs the 24h simulation with the persistent OnlineLearner.
  3. Reports per-day metrics including online AUC evolution so we can
     watch whether the live head is actually getting smarter.

Usage:
    python backtest_sweep_v3.py --tf 5m --days 7
    python backtest_sweep_v3.py --tf 5m --days 14 --ev 0.05
    python backtest_sweep_v3.py --tf 5m --days 7 --maker-tp off --reset-online on
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
from backtest.signal_adapter_v3 import make_signal_fn_v3
from backtest.universe import select_liquid_trending
from backtest.tf_config import get as get_tf_config, TFConfig
from backtest_honest import fetch_klines, fetch_funding_history, session_from_ts
from backtest_v2 import fetch_training_bars, build_cross_section, _bar_ms

from ml_v2.trainer import train_pooled
from ml_v2.predictor import V2Predictor
from ml_v2.online import OnlineLearner
from ml_v2.features import FEATURE_COLUMNS

from utils.config import API_URL
from utils.database import init_db, load_state_from_db

console = Console()
init_db()
load_state_from_db()


async def simulate_one_day_v3(
    client,
    symbols,
    sim_end_ms,
    risk_percent,
    min_ev_pct,
    max_leverage,
    max_positions,
    starting_balance,
    train_chunks,
    fusion_weights,
    tf: TFConfig,
    online_learner: OnlineLearner,
    maker_tp: bool,
    online_blend_weight: float,
    adaptive_ev: bool,
):
    """Run one day's sim, reusing the persistent online_learner."""
    sim_start_ms = sim_end_ms - 24 * 3600 * 1000
    train_end_ms = sim_start_ms
    bar_ms = _bar_ms(tf)

    # --- Train the frozen LightGBM for this day ---
    train_per_symbol = {}
    for s in symbols:
        df = await fetch_training_bars(
            client, s, train_end_ms,
            interval=tf.base_interval, bar_ms=bar_ms, chunks=train_chunks,
        )
        if df is not None and not df.empty:
            train_per_symbol[s] = df.reset_index(drop=True)
    if not train_per_symbol:
        return None

    btc_train = await fetch_training_bars(
        client, "BTCUSDT", train_end_ms,
        interval=tf.base_interval, bar_ms=bar_ms, chunks=train_chunks,
    )
    if btc_train is None or btc_train.empty:
        return None
    btc_train_close = btc_train.set_index("ot")["c"]
    all_ot = sorted(
        set().union(*(df["ot"].tolist() for df in train_per_symbol.values()))
    )
    btc_aligned = btc_train_close.reindex(all_ot).ffill()

    funding_train = {}
    for s in symbols:
        fs = await fetch_funding_history(client, s, end_time=train_end_ms)
        if fs is not None:
            funding_train[s] = fs

    bundle = train_pooled(
        per_symbol=train_per_symbol,
        btc_close=btc_aligned,
        funding_per_symbol=funding_train,
        cross_section=None,
        horizon=tf.horizon_bars,
        regime_quantile=tf.regime_atr_quantile,
    )
    predictor = V2Predictor(bundle)
    ml_regime_auc = {}
    for reg, m in bundle.get("metrics", {}).get("per_regime", {}).items():
        if m.get("skipped"):
            ml_regime_auc[reg] = None
        else:
            aucs = m.get("fold_auc", [])
            ml_regime_auc[reg] = sum(aucs) / len(aucs) if aucs else None

    # --- Simulation window data ---
    btc_1m = await fetch_klines(
        client, "BTCUSDT", tf.base_interval, limit=1500, end_time=sim_end_ms
    )
    data = {}
    for symbol in symbols:
        d1m = await fetch_klines(
            client, symbol, tf.base_interval, limit=1500, end_time=sim_end_ms
        )
        d15m = await fetch_klines(
            client, symbol, tf.mtf_interval, limit=500, end_time=sim_end_ms
        )
        d1h = await fetch_klines(
            client, symbol, tf.htf_interval, limit=200, end_time=sim_end_ms
        )
        if d1m.empty or d15m.empty or d1h.empty:
            continue
        funding = await fetch_funding_history(client, symbol, end_time=sim_end_ms)
        data[symbol] = {
            "1m": d1m, "15m": d15m, "1h": d1h,
            "funding": funding,
            "tick": d1m["c"].iloc[-1] * 1e-4,
        }
    if not data:
        return None

    cross_section = await build_cross_section(client, list(data.keys()), sim_end_ms)
    cost = CostModel(maker_tp=maker_tp)
    fill = FillModel(require_penetration=True, penetration_ticks=1)
    funding_sim = {
        s: data[s].get("funding")
        for s in data if data[s].get("funding") is not None
    }

    if tf.name == "5m":
        struct_short, struct_long, breakout_window = 10, 20, 10
    elif tf.name == "15m":
        struct_short, struct_long, breakout_window = 6, 12, 8
    else:
        struct_short, struct_long, breakout_window = 15, 30, 16

    # AUC-aware ML weight scaling per day (same policy as v3 runner).
    fw = dict(fusion_weights) if fusion_weights else None
    if fw is not None:
        if not predictor.is_ready():
            fw["ml"] = 0.0
        else:
            best_auc = predictor.best_auc()
            if best_auc < 0.51:
                fw["ml"] = 0.0
            elif best_auc < 0.55:
                fw["ml"] = fw.get("ml", 1.0) * (best_auc - 0.51) / 0.04

    signal_fn, online_learner = make_signal_fn_v3(
        v2_predictor=predictor,
        online_learner=online_learner,
        btc_1m=btc_1m,
        funding_per_symbol=funding_sim,
        cross_section=cross_section,
        risk_percent=risk_percent,
        min_ev_pct=min_ev_pct,
        max_leverage=max_leverage,
        cost_model=cost,
        fusion_weights=fw,
        session_fn=session_from_ts,
        mtf_ratio=tf.mtf_ratio,
        htf_ratio=tf.htf_ratio,
        struct_short=struct_short,
        struct_long=struct_long,
        breakout_window=breakout_window,
        trail_ttl_bars=tf.trail_ttl_bars,
        min_1m_warmup=max(60, struct_long * 2),
        sym_id_map=predictor.sym_id_map,
        adaptive_ev=adaptive_ev,
        online_blend_weight=online_blend_weight,
    )

    def _on_close(trade, meta):
        sig_id = meta.get("sig_id")
        if not sig_id:
            return
        try:
            online_learner.record_outcome(
                sig_id, 1 if trade.pnl > 0 else 0
            )
        except Exception:
            pass

    mins_per_bar = bar_ms // 60_000
    sim_bars = (24 * 60) // max(1, mins_per_bar)

    bt = EventDrivenBacktester(
        data=data,
        signal_fn=signal_fn,
        starting_balance=starting_balance,
        cost_model=cost,
        fill_model=fill,
        max_positions=max_positions,
        sim_bars=sim_bars,
        signal_every_n_bars=tf.signal_every_n_bars,
        on_trade_close=_on_close,
    )
    metrics = bt.run()
    metrics["_n_symbols"] = len(data)
    metrics["_ml_regime_auc"] = ml_regime_auc
    metrics["_ml_ready"] = predictor.is_ready()
    metrics["_online_auc"] = online_learner.recent_auc()
    metrics["_online_updates"] = online_learner._n_updates
    metrics["_online_ready"] = online_learner.is_ready()
    return metrics


async def run_sweep(
    days, symbols, risk_percent, min_ev_pct, max_leverage,
    max_positions, starting_balance, train_chunks, fusion_weights, tf: TFConfig,
    maker_tp: bool, online_blend_weight: float, adaptive_ev: bool,
    reset_online_daily: bool,
):
    chunks = train_chunks if train_chunks is not None else tf.default_train_chunks
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
            console.print(f"Symbols: {symbols}")
        else:
            console.print(f"Using symbols: {symbols}")

        console.print(
            f"[bold]TF base[/]: {tf.name} "
            f"(mtf={tf.mtf_interval} htf={tf.htf_interval}) "
            f"train_chunks={chunks}"
        )
        console.print(
            f"[bold]Maker-TP[/]: {'ON' if maker_tp else 'OFF'}   "
            f"[bold]Online blend[/]: {online_blend_weight}   "
            f"[bold]Adaptive EV[/]: {'ON' if adaptive_ev else 'OFF'}   "
            f"[bold]Reset online daily[/]: {'YES' if reset_online_daily else 'NO'}"
        )

        # Single OnlineLearner persists across all days unless
        # `reset_online_daily=True`. Persistence is what makes this a
        # "live learning" simulation instead of 7 disjoint backtests.
        persistent_online = OnlineLearner(feature_cols=list(FEATURE_COLUMNS))

        per_day = []
        for i in range(days, 0, -1):
            sim_end_ms = now_ms - (i - 1) * 24 * 3600 * 1000
            date_str = datetime.fromtimestamp(
                sim_end_ms / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d")
            console.print(
                f"\n[bold cyan]-- Day {days - i + 1}/{days} ending "
                f"{date_str} UTC --[/]"
            )

            day_learner = (
                OnlineLearner(feature_cols=list(FEATURE_COLUMNS))
                if reset_online_daily
                else persistent_online
            )

            result = await simulate_one_day_v3(
                client, symbols, sim_end_ms,
                risk_percent, min_ev_pct, max_leverage,
                max_positions, starting_balance, chunks, fusion_weights, tf,
                day_learner, maker_tp, online_blend_weight, adaptive_ev,
            )
            if result is None:
                console.print("  [yellow]No data, skip.[/]")
                continue
            m = result
            pf = m["profit_factor"]
            pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
            auc_str = " | AUC: "
            for reg, a in m["_ml_regime_auc"].items():
                auc_str += (
                    f"{reg[:3]}={a:.2f} " if a is not None else f"{reg[:3]}=-- "
                )
            ml_tag = "[green]ML-ON[/]" if m["_ml_ready"] else "[red]ML-OFF[/]"
            online_tag = (
                f"[green]ONLINE({m['_online_auc']:.2f}, "
                f"n={m['_online_updates']})[/]"
                if m["_online_ready"]
                else f"[yellow]ONLINE-WARMUP(n={m['_online_updates']})[/]"
            )
            console.print(
                f"  {ml_tag} {online_tag} trades={m['trades']}, "
                f"WR={m['win_rate']*100:.1f}%, PF={pf_s}, "
                f"NET=${m['net_pnl']:+.2f}, "
                f"fill={m['fill_rate']*100:.0f}%{auc_str}"
            )
            per_day.append({"date": date_str, **m})

    if not per_day:
        console.print("[red]No days produced metrics.[/]")
        return

    total_trades = sum(d["trades"] for d in per_day)
    total_wins = sum(d["trades"] * d["win_rate"] for d in per_day)
    total_pnl = sum(d["net_pnl"] for d in per_day)
    total_fees = sum(d["total_fees"] for d in per_day)
    daily_pnls = [d["net_pnl"] for d in per_day]
    profit_days = sum(1 for p in daily_pnls if p > 0)
    loss_days = sum(1 for p in daily_pnls if p < 0)

    mean_daily = statistics.mean(daily_pnls) if daily_pnls else 0.0
    stdev_daily = statistics.pstdev(daily_pnls) if len(daily_pnls) > 1 else 0.0
    t_stat = (
        (mean_daily / (stdev_daily / (len(daily_pnls) ** 0.5)))
        if stdev_daily > 0 else 0.0
    )
    daily_rets = [p / starting_balance for p in daily_pnls]
    mean_ret = statistics.mean(daily_rets) if daily_rets else 0.0
    stdev_ret = statistics.pstdev(daily_rets) if len(daily_rets) > 1 else 0.0
    sharpe_daily = (mean_ret / stdev_ret) * (365 ** 0.5) if stdev_ret > 0 else 0.0

    tbl = Table(
        title=f"V3 per-day results TF={tf.name} ({len(per_day)} days)"
    )
    tbl.add_column("Date")
    tbl.add_column("ML", justify="center")
    tbl.add_column("Onl", justify="center")
    tbl.add_column("Trades", justify="right")
    tbl.add_column("WR%", justify="right")
    tbl.add_column("PF", justify="right")
    tbl.add_column("NET $", justify="right")
    tbl.add_column("AUC T/V/R", justify="right")
    tbl.add_column("OnlAUC", justify="right")
    for d in per_day:
        color = (
            "green" if d["net_pnl"] > 0
            else ("red" if d["net_pnl"] < 0 else "white")
        )
        pf = d["profit_factor"]
        pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        aucs = d["_ml_regime_auc"]
        auc_cell = "/".join(
            f"{aucs.get(r):.2f}" if aucs.get(r) is not None else "--"
            for r in ("TRENDING", "VOLATILE", "RANGING")
        )
        tbl.add_row(
            d["date"],
            "ON" if d["_ml_ready"] else "OFF",
            "ON" if d["_online_ready"] else "warm",
            str(d["trades"]),
            f"{d['win_rate']*100:.1f}",
            pf_s,
            f"[{color}]{d['net_pnl']:+.2f}[/]",
            auc_cell,
            f"{d['_online_auc']:.2f}",
        )
    console.print(tbl)

    overall_wr = (total_wins / total_trades * 100) if total_trades else 0.0
    color = "green" if total_pnl > 0 else "red"
    console.print(f"\n[bold]===== V3 SWEEP AGGREGATE (TF={tf.name}) =====[/]")
    console.print(f"Days simulated : {len(per_day)}")
    console.print(f"Total trades   : {total_trades}")
    console.print(f"Overall WR     : {overall_wr:.1f}%")
    console.print(f"Profit days    : {profit_days}")
    console.print(f"Loss days      : {loss_days}")
    console.print(f"Mean daily PnL : ${mean_daily:+.3f}")
    console.print(f"Stdev daily    : ${stdev_daily:.3f}")
    console.print(f"t-stat vs 0    : {t_stat:.2f}  (|t|>2 ~ 95% significant)")
    console.print(f"Sharpe (d->ann): {sharpe_daily:.2f}")
    console.print(f"Total fees     : ${total_fees:.2f}")
    console.print(
        f"[bold {color}]TOTAL NET PnL  : ${total_pnl:+.2f} "
        f"({total_pnl / starting_balance * 100:+.2f}% of ${starting_balance:.0f})[/]"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tf", type=str, default="5m",
        choices=["1m", "5m", "15m"],
        help="Base timeframe (default 5m)",
    )
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--ev", type=float, default=0.05)
    ap.add_argument("--max-lev", type=int, default=10)
    ap.add_argument("--balance", type=float, default=100.0)
    ap.add_argument("--max-positions", type=int, default=3)
    ap.add_argument("--train-chunks", type=int, default=None)
    ap.add_argument("--w-tech", type=float, default=0.6)
    ap.add_argument("--w-ml", type=float, default=1.0)
    ap.add_argument("--w-flow", type=float, default=0.5)
    ap.add_argument("--w-online", type=float, default=0.3)
    ap.add_argument("--maker-tp", type=str, default="on", choices=["on", "off"])
    ap.add_argument("--adaptive-ev", type=str, default="on", choices=["on", "off"])
    ap.add_argument(
        "--reset-online", type=str, default="off", choices=["on", "off"],
        help="If 'on', reset OnlineLearner each day (no cross-day learning)",
    )
    args = ap.parse_args()

    tf = get_tf_config(args.tf)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    fw = {"tech": args.w_tech, "ml": args.w_ml, "flow": args.w_flow}
    asyncio.run(
        run_sweep(
            days=args.days, symbols=symbols,
            risk_percent=args.risk, min_ev_pct=args.ev,
            max_leverage=args.max_lev, max_positions=args.max_positions,
            starting_balance=args.balance,
            train_chunks=args.train_chunks, fusion_weights=fw,
            tf=tf,
            maker_tp=(args.maker_tp == "on"),
            online_blend_weight=args.w_online,
            adaptive_ev=(args.adaptive_ev == "on"),
            reset_online_daily=(args.reset_online == "on"),
        )
    )


if __name__ == "__main__":
    main()
