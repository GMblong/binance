"""Order-book microstructure.

Two signals are extracted:

1. L5 imbalance: volume-weighted ratio of top-5 bid vs ask levels. A sharp
   asymmetry predicts short-term direction at a >50% base rate on liquid
   alt-USDT perps.

2. Liquidity void: looks 5-20 levels deep to find gaps in the book.
   A void above the ask is a "magnet" for upward moves (price can sweep
   through because there is nothing to fill against it). Symmetrical on
   the bid side.

Nothing here is live-websocket-specific: `ob_l5_imbalance` and
`detect_liquidity_void` take already-fetched bids/asks arrays, so they
can be called from either the live bot or a backtest replay of depth
snapshots.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any


def ob_l5_imbalance(bids: List[List[float]], asks: List[List[float]]) -> float:
    """Return signed imbalance in [-1, 1]. Positive = more bids.

    `bids`/`asks` are lists of [price, size]. We use top-5 volume
    weighted by distance-to-mid (closer levels count more).
    """
    if not bids or not asks:
        return 0.0
    try:
        mid = (float(bids[0][0]) + float(asks[0][0])) / 2.0
    except Exception:
        return 0.0

    def weighted(side):
        tot = 0.0
        for px, sz in side[:5]:
            px = float(px)
            sz = float(sz)
            dist_bps = abs(px - mid) / mid * 1e4 if mid > 0 else 1.0
            w = 1.0 / (1.0 + dist_bps)
            tot += sz * w
        return tot

    b = weighted(bids)
    a = weighted(asks)
    if b + a <= 0:
        return 0.0
    return (b - a) / (b + a)


def detect_liquidity_void(
    bids: List[List[float]],
    asks: List[List[float]],
    depth: int = 20,
    gap_threshold_bps: float = 5.0,
) -> Dict[str, Any]:
    """Detect gaps in the first `depth` levels of the book.

    A "void" is a price region with notional-size far below the median of
    its side, followed immediately by a large level. In practice the fast
    path for a market move.

    Returns dict with:
      - up_void: price above ask[0] that is a magnet, or None
      - down_void: price below bid[0] that is a magnet, or None
      - up_strength / down_strength: 0..1 confidence
    """
    res: Dict[str, Any] = {
        "up_void": None, "down_void": None,
        "up_strength": 0.0, "down_strength": 0.0,
    }
    try:
        if len(asks) < depth or len(bids) < depth:
            return res

        ask_sizes = [float(a[1]) for a in asks[:depth]]
        bid_sizes = [float(b[1]) for b in bids[:depth]]
        if not ask_sizes or not bid_sizes:
            return res
        med_ask = sorted(ask_sizes)[len(ask_sizes) // 2]
        med_bid = sorted(bid_sizes)[len(bid_sizes) // 2]

        # Look for first level (beyond index 1) whose size is < 0.2x median
        # AND where the NEXT level is > 2x median. That is a structural
        # void followed by a wall, i.e. "price flies then stops".
        for i in range(1, depth - 1):
            if ask_sizes[i] < 0.2 * med_ask and ask_sizes[i + 1] > 2.0 * med_ask:
                dist_bps = (float(asks[i + 1][0]) - float(asks[0][0])) / float(asks[0][0]) * 1e4
                if dist_bps >= gap_threshold_bps:
                    res["up_void"] = float(asks[i + 1][0])
                    res["up_strength"] = min(1.0, ask_sizes[i + 1] / (med_ask + 1e-8) / 4.0)
                    break
        for i in range(1, depth - 1):
            if bid_sizes[i] < 0.2 * med_bid and bid_sizes[i + 1] > 2.0 * med_bid:
                dist_bps = (float(bids[0][0]) - float(bids[i + 1][0])) / float(bids[0][0]) * 1e4
                if dist_bps >= gap_threshold_bps:
                    res["down_void"] = float(bids[i + 1][0])
                    res["down_strength"] = min(1.0, bid_sizes[i + 1] / (med_bid + 1e-8) / 4.0)
                    break
    except Exception:
        pass
    return res


async def fetch_depth_snapshot(client, api_url: str, symbol: str, limit: int = 50):
    """Fetch a depth snapshot. Helper so callers don't repeat URL logic.

    Returns (bids, asks) where each is a list of [price_str, size_str].
    If the request fails, returns ([], []).
    """
    try:
        res = await client.get(
            f"{api_url}/fapi/v1/depth",
            params={"symbol": symbol, "limit": limit},
            timeout=5,
        )
        if res.status_code != 200:
            return [], []
        j = res.json()
        return j.get("bids", []), j.get("asks", [])
    except Exception:
        return [], []
