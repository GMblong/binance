"""Performance metrics for backtests.

We intentionally compute metrics from realized PnL of closed trades plus the
equity-curve derived metrics. Nothing in this module uses future data.
"""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import List, Dict, Any


def compute_metrics(
    trades: List[Dict[str, Any]], equity_curve: List[float]
) -> Dict[str, float]:
    """Compute the standard trading metrics.

    trades: list of dicts with keys {pnl, hold_bars, reason, fees, side}
    equity_curve: list of balance snapshots, 1 per minute
    """
    n = len(trades)
    if n == 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd_pct": 0.0,
            "avg_hold_bars": 0.0,
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "total_fees": 0.0,
        }

    pnls = [t["pnl"] for t in trades]
    fees = [t.get("fees", 0.0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / n
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    expectancy = sum(pnls) / n
    avg_hold = mean(t.get("hold_bars", 0) for t in trades) if trades else 0.0

    # Equity-curve stats (per-minute returns).
    if len(equity_curve) >= 2:
        rets = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i - 1]
            cur = equity_curve[i]
            if prev > 0:
                rets.append((cur - prev) / prev)
        if rets:
            mu = mean(rets)
            sd = pstdev(rets) if len(rets) > 1 else 0.0
            # Annualize for 1m bars: sqrt(365 * 1440)
            ann = math.sqrt(365 * 1440)
            sharpe = (mu / sd) * ann if sd > 0 else 0.0
            neg = [r for r in rets if r < 0]
            dsd = pstdev(neg) if len(neg) > 1 else 0.0
            sortino = (mu / dsd) * ann if dsd > 0 else 0.0
        else:
            sharpe = sortino = 0.0

        peak = equity_curve[0]
        max_dd = 0.0
        for v in equity_curve:
            if v > peak:
                peak = v
            if peak > 0:
                dd = (peak - v) / peak
                if dd > max_dd:
                    max_dd = dd
        max_dd_pct = max_dd * 100
    else:
        sharpe = sortino = max_dd_pct = 0.0

    return {
        "trades": n,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd_pct": max_dd_pct,
        "avg_hold_bars": avg_hold,
        "net_pnl": sum(pnls),
        "gross_pnl": sum(pnls) + sum(fees),
        "total_fees": sum(fees),
    }


def print_report(
    console, title: str, metrics: Dict[str, float], extra: Dict[str, Any] = None
) -> None:
    color = "green" if metrics.get("net_pnl", 0) > 0 else "red"
    console.print(f"\n[bold {color}]=== {title} ===[/]")
    console.print(f"  Trades        : {metrics['trades']}")
    console.print(f"  Win Rate      : {metrics['win_rate']*100:.1f}%")
    pf = metrics['profit_factor']
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    console.print(f"  Profit Factor : {pf_s}")
    console.print(f"  Expectancy    : ${metrics['expectancy']:+.3f} / trade")
    console.print(f"  Sharpe (ann.) : {metrics['sharpe']:.2f}")
    console.print(f"  Sortino (ann.): {metrics['sortino']:.2f}")
    console.print(f"  Max Drawdown  : {metrics['max_dd_pct']:.2f}%")
    console.print(f"  Avg Hold      : {metrics['avg_hold_bars']:.1f} bars")
    console.print(
        f"  Gross / Fees  : ${metrics['gross_pnl']:+.2f} / ${metrics['total_fees']:.2f}"
    )
    console.print(
        f"  [bold {color}]NET PnL       : ${metrics['net_pnl']:+.2f}[/]"
    )
    if extra:
        for k, v in extra.items():
            console.print(f"  {k:<14}: {v}")
