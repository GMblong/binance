"""Backtest v3 — v2 fusion + online learning + maker-maker exits.

v3 = v2 + three surgical upgrades designed to unlock positive EV at
the 1m/5m timeframe where v2 landed at -$6 to -$21 over 7 days:

  1. Maker-maker exits: CostModel.maker_tp=True so winners pay maker
     fee on BOTH legs. Round-trip fee drops from ~0.08% (taker*2) to
     ~0.04% (maker*2) for a winner and ~0.06% for a loser. This is
     often the difference between -0.05% and +0.05% per trade.

  2. Online learning: an SGDClassifier is updated after every trade
     closes. The frozen LightGBM is still the primary predictor, but
     the online head adapts to regime drift between retrains. Labels
     are delayed (we only learn from resolved trades) so there is no
     look-ahead leakage into the signal generator.

  3. Adaptive EV gate: minimum EV threshold scales inversely with the
     best classifier AUC. When AUC<=0.50, demand 2.5x base EV (only
     tech+flow setups fire). When AUC>=0.55, relax to 0.8x base EV.

Walk-forward contract is identical to v2: ml_v2 trains on bars that
end STRICTLY BEFORE sim_start_ms. The online learner starts EMPTY on
the sim window and accumulates labels during simulation (like live).

Usage:
    python backtest_v3.py --tf 5m --today
    python backtest_v3.py --tf 5m --today --maker-tp off
    python backtest_v3.py --tf 5m --symbols BTCUSDT,ETHUSDT,SOLUSDT
    python backtest_v3.py --tf 5m --ev 0.05 --w-online 0.4
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime

import httpx
import pandas as pd
from rich.console import Console

from backtest import (
    EventDrivenBacktester,
    CostModel,
    FillModel,
    print_report,
)
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


async def run_v3(
    symbols,
    use_today: bool,
    sim_window_hours: int,
    risk_percent: float,
    min_ev_pct: float,
    max_leverage: int,
    starting_balance: float,
    fusion_weights: dict | None,
    tf: TFConfig,
    train_chunks: int | None = None,
    max_positions: int = 3,
    maker_tp: bool = True,
    online_blend_weight: float = 0.3,
    adaptive_ev: bool = True,
):
    chunks = train_chunks if train_chunks is not None else tf.default_train_chunks
    bar_ms = _bar_ms(tf)

    now_ms = int(time.time() * 1000)
    sim_end_ms = now_ms if use_today else now_ms - sim_window_hours * 3600 * 1000
    sim_start_ms = sim_end_ms - sim_window_hours * 3600 * 1000
    train_end_ms = sim_start_ms

    console.print(
        f"[bold]TF base[/]: {tf.name} (mtf={tf.mtf_interval} htf={tf.htf_interval})"
    )
    console.print(
        f"[bold]Maker-TP[/]: {'ON' if maker_tp else 'OFF'}   "
        f"[bold]Online blend[/]: {online_blend_weight}   "
        f"[bold]Adaptive EV[/]: {'ON' if adaptive_ev else 'OFF'}"
    )
    console.print(
        f"[bold]Sim window[/]: "
        f"{datetime.utcfromtimestamp(sim_start_ms/1000)} UTC -> "
        f"{datetime.utcfromtimestamp(sim_end_ms/1000)} UTC"
    )
    console.print(
        f"[bold]Train cutoff[/]: "
        f"{datetime.utcfromtimestamp(train_end_ms/1000)} UTC"
    )

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
                console.print(f"[yellow]Picker empty, fallback: {symbols}[/]")
            else:
                console.print(f"Picker selected: {symbols}")
        else:
            console.print(f"Using symbols: {symbols}")

        # -------- Training corpus (ends BEFORE sim_start_ms) --------
        console.print(
            f"[dim]Fetching training corpus "
            f"({chunks} chunks of 1500 {tf.base_interval} bars/sym)...[/dim]"
        )
        train_per_symbol = {}
        for s in symbols:
            df = await fetch_training_bars(
                client, s, train_end_ms,
                interval=tf.base_interval, bar_ms=bar_ms, chunks=chunks,
            )
            if df is not None and not df.empty:
                train_per_symbol[s] = df
                console.print(f"  [dim]{s}: {len(df)} bars[/dim]")
        btc_train = await fetch_training_bars(
            client, "BTCUSDT", train_end_ms,
            interval=tf.base_interval, bar_ms=bar_ms, chunks=chunks,
        )
        if btc_train is None or btc_train.empty:
            console.print("[red]No BTC training data; aborting.[/]")
            return
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

        console.print("[cyan]Training ml_v2 (pooled, regime-conditional)...[/cyan]")
        for s in list(train_per_symbol.keys()):
            train_per_symbol[s] = train_per_symbol[s].reset_index(drop=True)
        bundle = train_pooled(
            per_symbol=train_per_symbol,
            btc_close=btc_aligned,
            funding_per_symbol=funding_train,
            cross_section=None,
            horizon=tf.horizon_bars,
            regime_quantile=tf.regime_atr_quantile,
        )
        predictor = V2Predictor(bundle)
        if bundle.get("regime_thresholds"):
            rt = bundle["regime_thresholds"]
            console.print(
                f"  [dim]Regime bands: vol_hi={rt['atr_vol_hi']:.3f}% "
                f"trend_lo={rt['atr_trend_lo']:.3f}% "
                f"dist_lo={rt['dist_trend_lo']:.4f}[/dim]"
            )
        if not predictor.is_ready():
            console.print(
                "[yellow]ml_v2 could not train any regime model -- "
                "falling back to neutral P_ml=0.5. Online + micro + "
                "technical still active.[/]"
            )
            total_rows = bundle.get("metrics", {}).get("total_rows", 0)
            console.print(
                f"  [dim]Total trainable rows: {total_rows}[/dim]"
            )
            if fusion_weights is not None:
                fusion_weights = dict(fusion_weights)
                fusion_weights["ml"] = 0.0
        else:
            pr = bundle.get("metrics", {}).get("per_regime", {})
            for reg, m in pr.items():
                if m.get("skipped"):
                    console.print(
                        f"  [dim]regime={reg} skipped (n={m.get('n')})[/dim]"
                    )
                else:
                    aucs = m.get("fold_auc", [])
                    if aucs:
                        console.print(
                            f"  [dim]regime={reg}: n={m.get('n')}, "
                            f"fold AUC mean={sum(aucs)/len(aucs):.3f}[/dim]"
                        )
            if fusion_weights is not None:
                fusion_weights = dict(fusion_weights)
                best_auc = predictor.best_auc()
                if best_auc < 0.51:
                    fusion_weights["ml"] = 0.0
                    console.print(
                        f"  [yellow]Best AUC {best_auc:.3f} < 0.51 -> "
                        f"ML weight set to 0[/]"
                    )
                elif best_auc < 0.55:
                    scale = (best_auc - 0.51) / 0.04
                    fusion_weights["ml"] = fusion_weights["ml"] * scale
                    console.print(
                        f"  [dim]Best AUC {best_auc:.3f} -> ML weight scaled "
                        f"to {fusion_weights['ml']:.2f}[/dim]"
                    )
                else:
                    console.print(
                        f"  [green]Best AUC {best_auc:.3f} -> ML weight "
                        f"{fusion_weights['ml']:.2f} (full)[/]"
                    )

        # -------- Simulation data (AFTER train_end_ms) --------
        console.print("[dim]Fetching simulation window data...[/dim]")
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
                console.print(f"[yellow]Skip {symbol}: empty sim data[/]")
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
            console.print("[red]No sim data fetched; aborting.[/]")
            return

        cross_section = await build_cross_section(
            client, list(data.keys()), sim_end_ms
        )

        # -------- Engine wiring --------
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

        # Fresh online learner per run. In live deployment the learner
        # would persist across sessions; in backtest we start empty so
        # the first ~30 trades are used purely for warmup.
        online_learner = OnlineLearner(feature_cols=list(FEATURE_COLUMNS))

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
            fusion_weights=fusion_weights,
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

        # Engine close-hook feeds the online learner. pnl > 0 => label 1
        # (trade won). The learner already knows the direction from
        # record_signal(), so it translates to UP-probability space.
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
        sim_bars = (sim_window_hours * 60) // max(1, mins_per_bar)

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
        console.print("[cyan]Running event-driven simulation (v3)...[/cyan]")
        metrics = bt.run()

        extra = {
            "TF": tf.name,
            "Sim bars": sim_bars,
            "Fill rate": f"{metrics['fill_rate']*100:.1f}%",
            "Signals gen": metrics["signals"],
            "Limit placed": metrics["limit_placed"],
            "Limit filled": metrics["limit_filled"],
            "Limit expired": metrics["limit_expired"],
            "Start balance": f"${metrics['start_balance']:.2f}",
            "Final balance": f"${metrics['final_balance']:.2f}",
            "Maker-TP": "ON" if maker_tp else "OFF",
            "Online AUC": f"{online_learner.recent_auc():.3f}",
            "Online updates": online_learner._n_updates,
        }
        print_report(
            console,
            f"BACKTEST V3 (FUSION+ONLINE, {len(data)} SYMBOLS, TF={tf.name})",
            metrics,
            extra,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tf", type=str, default="5m",
        choices=["1m", "5m", "15m"],
        help="Base timeframe (default 5m)",
    )
    ap.add_argument("--today", action="store_true")
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--ev", type=float, default=0.05)
    ap.add_argument("--max-lev", type=int, default=10)
    ap.add_argument("--balance", type=float, default=100.0)
    ap.add_argument("--w-tech", type=float, default=0.6)
    ap.add_argument("--w-ml", type=float, default=1.0)
    ap.add_argument("--w-flow", type=float, default=0.5)
    ap.add_argument("--w-online", type=float, default=0.3,
                    help="Blend weight of the online learner (0..1, default 0.3)")
    ap.add_argument("--maker-tp", type=str, default="on",
                    choices=["on", "off"],
                    help="Route TP exits as maker limit (halves win-side fee)")
    ap.add_argument("--adaptive-ev", type=str, default="on",
                    choices=["on", "off"],
                    help="Scale min EV threshold by recent classifier AUC")
    ap.add_argument(
        "--train-chunks", type=int, default=None,
        help="Override default chunks for the chosen TF",
    )
    args = ap.parse_args()

    tf = get_tf_config(args.tf)
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    fw = {"tech": args.w_tech, "ml": args.w_ml, "flow": args.w_flow}
    asyncio.run(
        run_v3(
            symbols=syms,
            use_today=args.today,
            sim_window_hours=args.hours,
            risk_percent=args.risk,
            min_ev_pct=args.ev,
            max_leverage=args.max_lev,
            starting_balance=args.balance,
            fusion_weights=fw,
            tf=tf,
            train_chunks=args.train_chunks,
            maker_tp=(args.maker_tp == "on"),
            online_blend_weight=args.w_online,
            adaptive_ev=(args.adaptive_ev == "on"),
        )
    )


if __name__ == "__main__":
    main()
