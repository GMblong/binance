"""
Pre-train ML models for ALL tradeable coins.
Run this before starting the bot to ensure full ML coverage.

Usage:
    python pretrain_all.py              # Train all coins with >$10M volume
    python pretrain_all.py --top 50     # Train top 50 by volume
    python pretrain_all.py --force      # Retrain even if model exists
"""
import asyncio
import sys
import time
import httpx
from pathlib import Path
from engine.ml_engine import ml_predictor, MODEL_DIR
from utils.config import API_URL, API_KEY, API_SECRET

async def main():
    top_n = 100
    force = "--force" in sys.argv
    for arg in sys.argv:
        if arg.startswith("--top"):
            idx = sys.argv.index(arg)
            if idx + 1 < len(sys.argv):
                top_n = int(sys.argv[idx + 1])

    async with httpx.AsyncClient(headers={"X-MBX-APIKEY": API_KEY}) as client:
        # Fetch all futures tickers
        res = await client.get(f"{API_URL}/fapi/v1/ticker/24hr")
        if res.status_code != 200:
            print(f"Failed to fetch tickers: {res.status_code}")
            return

        tickers = [
            {"s": t["symbol"], "q": float(t["quoteVolume"])}
            for t in res.json()
            if t["symbol"].endswith("USDT") and float(t["quoteVolume"]) > 10_000_000
        ]
        tickers.sort(key=lambda x: x["q"], reverse=True)
        symbols = [t["s"] for t in tickers[:top_n]]

        # Filter already trained (unless --force)
        if not force:
            existing = {p.stem.replace("_ensemble", "") for p in MODEL_DIR.glob("*_ensemble.joblib")}
            symbols = [s for s in symbols if s not in existing]

        total = len(symbols)
        if total == 0:
            print("All models already trained. Use --force to retrain.")
            return

        print(f"Training {total} models (concurrency=4)...")
        done = 0
        failed = []

        sem = asyncio.Semaphore(4)

        async def train_one(sym):
            nonlocal done
            async with sem:
                try:
                    ok = await ml_predictor.train_model(client, sym)
                    done += 1
                    status = "✓" if ok else "✗ (insufficient data)"
                    print(f"  [{done}/{total}] {sym} {status}")
                    if not ok:
                        failed.append(sym)
                except Exception as e:
                    done += 1
                    failed.append(sym)
                    print(f"  [{done}/{total}] {sym} ✗ ({e})")

        start = time.time()
        await asyncio.gather(*[train_one(s) for s in symbols])
        elapsed = time.time() - start

        print(f"\nDone in {elapsed:.0f}s")
        print(f"Success: {total - len(failed)}/{total}")
        if failed:
            print(f"Failed: {', '.join(failed)}")

if __name__ == "__main__":
    asyncio.run(main())
