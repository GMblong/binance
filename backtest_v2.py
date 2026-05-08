"""Backtest v2 — Phase 3 (microstructure) + Phase 4 (ml_v2 fusion).

Now fully TF-aware: pass `--tf 5m` or `--tf 15m` to escape 1m HFT noise.
All intervals, horizon, chunks, and signal spacing come from TF_CONFIGS.

Everything that was honest in v1 stays honest:
  - Realistic cost + fill model.
  - Strict walk-forward: ml_v2 is trained on a corpus ending at
    `train_end_ms`, the simulation then runs on bars AFTER that cutoff.
  - Event-driven engine loops base bars globally across symbols.

New in v2:
  - Pooled regime-conditional LightGBM (ml_v2) replaces per-symbol model.
  - Microstructure features fused via logit blending (ml_v2.fusion).
  - Cross-section 1h-rank is computed across the selected universe.

Usage:
    python backtest_v2.py                        # yesterday, 1m (legacy)
    python backtest_v2.py --tf 5m --today        # 5m base bars
    python backtest_v2.py --tf 15m --today       # 15m base bars
    python backtest_v2.py --tf 5m --symbols BTCUSDT,ETHUSDT,SOLUSDT
    python backtest_v2.py --tf 5m --ev 0.05 --risk 0.01
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
from backtest.tf_config import get as get_tf_config, TFConfig
from backtest_honest import fetch_klines, fetch_funding_history, session_from_ts

from ml_v2.trainer import train_pooled
from ml_v2.predictor import V2Predictor

from utils.config import API_URL
from utils.database import init_db, load_state_from_db

console = Console()
init_db()
load_state_from_db()


async def fetch_training_bars(
    client,
    symbol,
    end_time_ms,
    interval: str,
    bar_ms: int,
    chunks: int,
):
    """Fetch N chunks of `interval` bars ending at end_time_ms.

    Each chunk is 1500 bars (Binance kline limit). Previous versions of
    this helper were hardcoded to 1m which made TF-agnostic training
    impossible.
    """
    frames = []
    cursor = end_time_ms
    for _ in range(chunks):
        df = await fetch_klines(
            client, symbol, interval, limit=1500, end_time=cursor
        )
        if df is None or df.empty:
            break
        frames.append(df)
        cursor = int(df["ot"].iloc[0]) - bar_ms
        if cursor <= 0:
            break
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True).drop_duplicates("ot").sort_values("ot")
    return full.reset_index(drop=True)


# Back-compat alias used by backtest_sweep_v2 and earlier callers.
async def fetch_training_1m(client, symbol, end_time_ms, chunks: int = 7):
    """Legacy 1m-only helper. Prefer fetch_training_bars()."""
    return await fetch_training_bars(
        client, symbol, end_time_ms,
        interval="1m", bar_ms=60_000, chunks=chunks,
    )


async def build_cross_section(client, symbols, end_time_ms):
    """Construct a DataFrame indexed by ot with 1h returns per symbol."""
    try:
        frames = {}
        for s in symbols:
            df = await fetch_klines(client, s, "1h", limit=200, end_time=end_time_ms)
            if df is None or df.empty:
                continue
            df = df.set_index("ot")[["c"]]
            df["ret_1h"] = df["c"].pct_change(1)
            frames[s] = df["ret_1h"]
        if not frames:
            return None
        cs = pd.DataFrame(frames).sort_index().ffill()
        return cs
    except Exception:
        return None


def _bar_ms(tf: TFConfig) -> int:
    """Convert TFConfig base_interval to milliseconds."""
    mapping = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
    }
    return mapping.get(tf.base_interval, 60_000)


async def run_v2(
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
                "falling back to neutral P_ml=0.5. Microstructure + "
                "technical still active.[/]"
            )
            total_rows = bundle.get("metrics", {}).get("total_rows", 0)
            console.print(
                f"  [dim]Total trainable rows after feature+label filter: "
                f"{total_rows}[/dim]"
            )
            for reg, m in bundle.get("metrics", {}).get("per_regime", {}).items():
                console.print(
                    f"  [dim]regime={reg}: n={m.get('n', 0)} (min=150)[/dim]"
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
                    else:
                        console.print(
                            f"  [dim]regime={reg}: n={m.get('n')} "
                            f"(CV folds not scored)[/dim]"
                        )
            if fusion_weights is not None:
                fusion_weights = dict(fusion_weights)
                best_auc = predictor.best_auc()
                if best_auc < 0.51:
                    fusion_weights["ml"] = 0.0
                    console.print(
                        f"  [yellow]Best AUC {best_auc:.3f} < 0.51 -> ML weight set to 0[/]"
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
                "1m": d1m,   # key name kept for engine compat -- holds base TF bars
                "15m": d15m, # key name kept -- holds MTF bars
                "1h": d1h,   # key name kept -- holds HTF bars
                "funding": funding,
                "tick": d1m["c"].iloc[-1] * 1e-4,
            }
        if not data:
            console.print("[red]No sim data fetched; aborting.[/]")
            return

        cross_section = await build_cross_section(client, list(data.keys()), sim_end_ms)

        cost = CostModel()
        fill = FillModel(require_penetration=True, penetration_ticks=1)
        funding_sim = {
            s: data[s].get("funding")
            for s in data if data[s].get("funding") is not None
        }

        # TF-aware structural windows: shorter on higher TFs because a
        # 15-bar window on 5m already covers 75 minutes of structure.
        if tf.name == "5m":
            struct_short, struct_long, breakout_window = 10, 20, 10
        elif tf.name == "15m":
            struct_short, struct_long, breakout_window = 6, 12, 8
        else:
            struct_short, struct_long, breakout_window = 15, 30, 16

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
            mtf_ratio=tf.mtf_ratio,
            htf_ratio=tf.htf_ratio,
            struct_short=struct_short,
            struct_long=struct_long,
            breakout_window=breakout_window,
            trail_ttl_bars=tf.trail_ttl_bars,
            min_1m_warmup=max(60, struct_long * 2),
        )

        # How many base bars fit in the sim window?
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
        )
        console.print("[cyan]Running event-driven simulation (v2)...[/cyan]")
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
        }
        print_report(
            console,
            f"BACKTEST V2 (FUSION, {len(data)} SYMBOLS, TF={tf.name})",
            metrics,
            extra,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tf", type=str, default="1m",
        choices=["1m", "5m", "15m"],
        help="Base timeframe (default 1m; strongly recommend 5m for crypto scalping)",
    )
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
    ap.add_argument(
        "--train-chunks", type=int, default=None,
        help="Override default chunks for the chosen TF",
    )
    args = ap.parse_args()

    tf = get_tf_config(args.tf)
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
            tf=tf,
            train_chunks=args.train_chunks,
        )
    )


if __name__ == "__main__":
    main()
