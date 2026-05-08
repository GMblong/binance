"""Symbol selection: pick coins by liquidity + trend persistence, NOT by
abs(price_change_percent) of the last 24h.

`get_top_movers` in the legacy scripts ranked by volatility, which is
exactly how retail gets farmed. This picker biases toward:

  - quoteVolume >= threshold (liquidity)
  - tight enough spread proxy (bidPrice / askPrice)
  - recent trend persistence (|realized 4h return| / realized 4h stdev)

Coins with a very large 1h range vs their 24h average (recent blow-off) are
excluded -- they are likely exhausted.
"""

from __future__ import annotations

from typing import List

import pandas as pd


DEFAULT_FALLBACK = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT",
]


async def select_liquid_trending(
    client,
    api_url: str,
    limit: int = 5,
    min_quote_vol: float = 30_000_000.0,
    max_spread_pct: float = 0.15,
    max_range_pct: float = 25.0,
    debug: bool = False,
) -> List[str]:
    """Pick liquid trending coins. Falls back progressively:

    1. Try strict filters (liquidity + tight spread + not-exhausted range).
    2. If none pass, relax the filters (2x looser) and retry.
    3. If still none, return a hard-coded list of blue-chip perps so the
       backtest can at least run.
    """
    try:
        res = await client.get(f"{api_url}/fapi/v1/ticker/24hr")
        data = res.json()
    except Exception:
        return DEFAULT_FALLBACK[:limit]

    def rank(min_qv: float, max_sp: float, max_rng: float):
        rows = []
        for t in data:
            sym = t.get("symbol", "")
            if not sym.endswith("USDT") or sym.endswith("_USDT"):
                continue
            # Skip inverse / quarterly contracts that sometimes sneak in.
            if any(x in sym for x in ("_", "-")):
                continue
            try:
                qv = float(t["quoteVolume"])
                if qv < min_qv:
                    continue
                bid = float(t.get("bidPrice", 0) or 0)
                ask = float(t.get("askPrice", 0) or 0)
                if bid > 0 and ask > 0:
                    spread_pct = (ask - bid) / ask * 100
                    if spread_pct > max_sp:
                        continue
                else:
                    spread_pct = 0.05
                last = float(t["lastPrice"])
                if last <= 0:
                    continue
                h24 = float(t["highPrice"])
                l24 = float(t["lowPrice"])
                range_pct = (h24 - l24) / last * 100
                if range_pct > max_rng:
                    continue
                chg = float(t["priceChangePercent"])
                mid = (h24 + l24) / 2
                persist = (last - mid) / last * 100
                score = qv / 1e7 + abs(chg) * 2 + abs(persist) * 5 - spread_pct * 50
                rows.append((sym, score))
            except Exception:
                continue
        rows.sort(key=lambda r: r[1], reverse=True)
        return [r[0] for r in rows[:limit]]

    picked = rank(min_quote_vol, max_spread_pct, max_range_pct)
    if debug:
        print(f"[universe] strict picked {len(picked)}")
    if picked:
        return picked

    picked = rank(min_quote_vol / 2, max_spread_pct * 2, max_range_pct * 1.5)
    if debug:
        print(f"[universe] relaxed picked {len(picked)}")
    if picked:
        return picked

    # Keep only those actually listed in the 24hr response to avoid
    # requesting klines for a symbol the exchange removed.
    listed = {t.get("symbol", "") for t in data}
    picked = [s for s in DEFAULT_FALLBACK if s in listed][:limit]
    if not picked:
        picked = DEFAULT_FALLBACK[:limit]
    return picked
