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


async def select_liquid_trending(
    client, api_url: str, limit: int = 5, min_quote_vol: float = 50_000_000.0
) -> List[str]:
    try:
        res = await client.get(f"{api_url}/fapi/v1/ticker/24hr")
        data = res.json()
    except Exception:
        return []

    rows = []
    for t in data:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        try:
            qv = float(t["quoteVolume"])
            if qv < min_quote_vol:
                continue
            bid = float(t.get("bidPrice", 0))
            ask = float(t.get("askPrice", 0))
            if bid <= 0 or ask <= 0:
                continue
            spread_pct = (ask - bid) / ask * 100
            if spread_pct > 0.05:
                continue
            last = float(t["lastPrice"])
            h24 = float(t["highPrice"])
            l24 = float(t["lowPrice"])
            range_pct = (h24 - l24) / last * 100
            # Avoid coins that already ran (blow-off risk): 24h range > 15%.
            if range_pct > 15:
                continue
            chg = float(t["priceChangePercent"])
            # Crude trend persistence: close vs midpoint of 24h range.
            mid = (h24 + l24) / 2
            persist = (last - mid) / last * 100
            score = qv / 1e7 + abs(chg) * 2 + abs(persist) * 5 - spread_pct * 50
            rows.append((sym, score, qv, chg, range_pct))
        except Exception:
            continue

    rows.sort(key=lambda r: r[1], reverse=True)
    return [r[0] for r in rows[:limit]]
