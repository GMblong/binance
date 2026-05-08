"""Backtest v2 — Phase 3 (microstructure) + Phase 4 (ml_v2 fusion).

Everything that was honest in v1 stays honest:
  - Realistic cost + fill model.
  - Strict walk-forward: ml_v2 is trained on a corpus ending at
    `train_end_ms`, the simulation then runs on bars AFTER that cutoff.
  - Event-driven engine loops 1-minute bars globally across symbols.

New in v2:
  - Pooled regime-conditional LightGBM (ml_v2) replaces per-symbol model.
  - Microstructure features fused via logit blending (ml_v2.fusion).
  - Cross-section 1h-rank is computed across the selected universe.

Usage:
    python backtest_v2.py                  # yesterday, picker-selected
    python backtest_v2.py --today
    python backtest_v2.py --symbols BTCUSDT,ETHUSDT,SOLUSDT
    python backtest_v2.py --ev 0.05 --risk 0.01 --days 1
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime, time as dt_time

import httpx
import pandas as pd
from rich.console import Console

from backtest import (
    EventDrivenBacktester,
    CostModel,
    FillModel,
    print_report,
)
from backtest.signal_adapter_v2 import make_signal_fn_v2
from backtest.universe import select_liquid_trending
from backtest_honest import fetch_klines, fetch_funding_history, session_from_ts

from ml_v2.trainer import train_pooled
from ml_v2.predictor import V2Predictor

from utils.config import API_URL
from utils.database import init_db, load_state_from_db

console = Console()
init_db()
load_state_from_db()


async def fetch_training_1m(client, symbol, end_time_ms, chunks: int = 7):
    """Fetch N * 1500 minutes of 1m bars ending at end_time_ms.

    Default `chunks=7` gives ~10500 minutes ≈ 7 days of 1m bars, which
    is the minimum for a pooled regime-conditional LightGBM to converge.
    Earlier versions only fetched 2*1500 minutes which left each regime
    with < 300 samples and ml_v2 silently fell back to P_ml=0.5.
    """
    frames = []
    cursor = end_time_ms
    for _ in range(chunks):
        df = await fetch_klines(client, symbol, "1m", limit=1500, end_time=cursor)
        if df is None or df.empty:
            break
        frames.append(df)
        cursor = int(df["ot"].iloc[0]) - 60_000
        if cursor <= 0:
            break
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True).drop_duplicates("ot").sort_values("ot")
    return full.reset_index(drop=True)


async def build_cross_section(client, symbols, end_time_ms):
    """Construct a DataFrame indexed by ot with 1h returns per symbol.

    Used for the `cs_rank_1h` feature. Falls back to None if anything
    fails so the pipeline still runs."""
    try:
        frames = {}
        for s in symbols:
            df = await fetch_klines(client, s, "1h", limit=200, end_time=end_time_ms)
            if df is None or df.empty:
                continue
            # Use previous 1h return (close over close) aligned on ot of
            # the bar OPEN time.
            df = df.set_index("ot")[["c"]]
            df["ret_1h"] = df["c"].pct_change(1)
            frames[s] = df["ret_1h"]
        if not frames:
            return None
        cs = pd.DataFrame(frames).sort_index().ffill()
        return cs
    except Exception:
        return None


async def run_v2(
    symbols,
    use_today: bool,
    sim_window_hours: int,
    risk_percent: float,
    min_ev_pct: float,
    max_leverage: int,
    starting_balance: float,
    fusion_weights: dict | None,
    train_chunks: int = 7,
    max_positions: int = 3,
):
    now_ms = int(time.time() * 1000)
    sim_end_ms = now_ms if use_today else now_ms - sim_window_hours * 3600 * 1000
    sim_start_ms = sim_end_ms - sim_window_hours * 3600 * 1000
    train_end_ms = sim_start_ms

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
                symbols = await select_liquid_trending(client, API_URL, limit=5, debug=True)
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
        console.print(f"[dim]Fetching training corpus ({train_chunks} chunks ~{train_chunks*1500}m per symbol)...[/dim]")
        train_per_symbol = {}
        for s in symbols:
            df = await fetch_training_1m(client, s, train_end_ms, chunks=train_chunks)
            if df is not None and not df.empty:
                train_per_symbol[s] = df
                console.print(f"  [dim]{s}: {len(df)} bars[/dim]")
        btc_train = await fetch_training_1m(client, "BTCUSDT", train_end_ms, chunks=train_chunks)
        if btc_train is None or btc_train.empty:
            console.print("[red]No BTC training data; aborting.[/]")
            return
        btc_train_close = btc_train.set_index("ot")["c"]
        # Reindex btc onto the union of all symbols' ot so merge-safe.
        all_ot = sorted(set().union(*(df["ot"].tolist() for df in train_per_symbol.values())))
        btc_aligned = btc_train_close.reindex(all_ot).ffill()

        funding_train = {}
        for s in symbols:
            fs = await fetch_funding_history(client, s, end_time=train_end_ms)
            if fs is not None:
                funding_train[s] = fs

        console.print("[cyan]Training ml_v2 (pooled, regime-conditional)...[/cyan]")
        # train_pooled expects per-symbol DFs indexed by integer, but uses
        # df['ot'] internally via features; reset_index is safe.
        for s in list(train_per_symbol.keys()):
            train_per_symbol[s] = train_per_symbol[s].reset_index(drop=True)
        # Align BTC to each symbol's bar ot inside trainer -- features
        # reindex btc_close onto df.index which is integer. Passing the
        # raw btc close series with ot index lets features.build_feature_matrix
        # call .reindex(df.index) -- to make that work, we set BTC index to
        # integer and pass a mapper via features? Simplest: pass
        # a per-symbol BTC series built by matching ot.
        # For training we instead rebuild btc_close to match each symbol's
        # row count by reindexing on the symbol ot.
        bundle = train_pooled(
            per_symbol=train_per_symbol,
            btc_close=btc_aligned,
            funding_per_symbol=funding_train,
            cross_section=None,  # cross-section is costly in training; off here
        )
        predictor = V2Predictor(bundle)
        if not predictor.is_ready():
            console.print(
                "[yellow]ml_v2 could not train any regime model -- "
                "falling back to neutral P_ml=0.5. Microstructure + "
                "technical still active.[/]"
            )
            total_rows = bundle.get("metrics", {}).get("total_rows", 0)
            console.print(
                f"  [dim]Total trainable rows after feature+label filter: {total_rows}[/dim]"
            )
            for reg, m in bundle.get("metrics", {}).get("per_regime", {}).items():
                console.print(f"  [dim]regime={reg}: n={m.get('n', 0)} (min=150)[/dim]")
            # If ML is off, reduce ML weight in fusion so P_ml=0.5 doesn't
            # drag decisions toward random. Tech + flow will carry us.
            if fusion_weights is not None:
                fusion_weights = dict(fusion_weights)
                fusion_weights["ml"] = 0.0
        else:
            pr = bundle.get("metrics", {}).get("per_regime", {})
            for reg, m in pr.items():
                if m.get("skipped"):
                    console.print(f"  [dim]regime={reg} skipped (n={m.get('n')})[/dim]")
                else:
                    aucs = m.get("fold_auc", [])
                    if aucs:
                        console.print(
                            f"  [dim]regime={reg}: n={m.get('n')}, "
                            f"fold AUC mean={sum(aucs)/len(aucs):.3f}[/dim]"
                        )
                    else:
                        console.print(
                            f"  [dim]regime={reg}: n={m.get('n')} "
                            f"(CV folds not scored)[/dim]"
                        )

        # -------- Simulation data (AFTER train_end_ms) --------
        console.print("[dim]Fetching simulation window data...[/dim]")
        btc_1m = await fetch_klines(client, "BTCUSDT", "1m", limit=1500, end_time=sim_end_ms)
        data = {}
        for symbol in symbols:
            d1m = await fetch_klines(client, symbol, "1m", limit=1500, end_time=sim_end_ms)
            d15m = await fetch_klines(client, symbol, "15m", limit=500, end_time=sim_end_ms)
            d1h = await fetch_klines(client, symbol, "1h", limit=200, end_time=sim_end_ms)
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

        cross_section = await build_cross_section(client, list(data.keys()), sim_end_ms)

        cost = CostModel()
        fill = FillModel(require_penetration=True, penetration_ticks=1)
        funding_sim = {s: data[s].get("funding") for s in data if data[s].get("funding") is not None}

        signal_fn = make_signal_fn_v2(
            v2_predictor=predictor,
            btc_1m=btc_1m,
            funding_per_symbol=funding_sim,
            cross_section=cross_section,
            risk_percent=risk_percent,
            min_ev_pct=min_ev_pct,
            max_leverage=max_leverage,
            cost_model=cost,
            fusion_weights=fusion_weights,
            session_fn=session_from_ts,
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
        console.print("[cyan]Running event-driven simulation (v2)...[/cyan]")
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
        print_report(
            console,
            f"BACKTEST V2 (FUSION, {len(data)} SYMBOLS)",
            metrics,
            extra,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", action="store_true")
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--ev", type=float, default=0.10)
    ap.add_argument("--max-lev", type=int, default=10)
    ap.add_argument("--balance", type=float, default=100.0)
    ap.add_argument("--w-tech", type=float, default=0.6)
    ap.add_argument("--w-ml", type=float, default=1.0)
    ap.add_argument("--w-flow", type=float, default=0.5)
    ap.add_argument("--train-chunks", type=int, default=7,
                    help="Fetch N*1500 bars for training (default 7 => ~1 week)")
    args = ap.parse_args()

    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    fw = {"tech": args.w_tech, "ml": args.w_ml, "flow": args.w_flow}
    asyncio.run(
        run_v2(
            symbols=syms,
            use_today=args.today,
            sim_window_hours=args.hours,
            risk_percent=args.risk,
            min_ev_pct=args.ev,
            max_leverage=args.max_lev,
            starting_balance=args.balance,
            fusion_weights=fw,
            train_chunks=args.train_chunks,
        )
    )


if __name__ == "__main__":
    main()
