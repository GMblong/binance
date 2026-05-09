"""Benchmark WITH ML models pre-trained (realistic live scenario)."""
import asyncio, time, sys, os
import numpy as np
import httpx, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import API_URL, API_KEY
from utils.state import market_data, bot_state
from strategies.hybrid import analyze_hybrid_async, get_btc_trend
from engine.ml_engine import ml_predictor

DURATION = 120

async def run():
    print("=" * 60)
    print("  BENCHMARK WITH ML (Full Pipeline + Trained Models)")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=15.0) as client:
        headers = {'X-MBX-APIKEY': API_KEY}
        
        # 1. Fetch tickers
        res = await client.get(f"{API_URL}/fapi/v1/ticker/24hr", headers=headers)
        tkr_data = res.json()
        tickers = [{"s": t["symbol"], "q": float(t["quoteVolume"]), "c": float(t["lastPrice"]), "o": float(t["openPrice"])}
                   for t in tkr_data if t["symbol"].endswith("USDT") and float(t["quoteVolume"]) > 10_000_000]
        tickers.sort(key=lambda x: x["q"], reverse=True)
        top_symbols = [t["s"] for t in tickers[:10]]  # Top 10 for ML training
        for t in tickers: market_data.prices[t["s"]] = t["c"]
        market_data.tickers = tickers
        await get_btc_trend(client)
        
        # 2. Fetch klines for top symbols
        print(f"\n[1] Fetching klines for {len(top_symbols)} symbols...")
        for symbol in top_symbols:
            res1, res15, res1h = await asyncio.gather(
                client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "1m", "limit": 200}, headers=headers),
                client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "15m", "limit": 100}, headers=headers),
                client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "1h", "limit": 50}, headers=headers),
            )
            def proc(data):
                df = pd.DataFrame(data).iloc[:, [0,1,2,3,4,5,9]]
                df.columns = ["ot","o","h","l","c","v","tbv"]
                for col in ["o","h","l","c","v","tbv"]: df[col] = df[col].astype(float)
                return df
            market_data.klines[symbol] = {"1m": proc(res1.json()), "15m": proc(res15.json()), "1h": proc(res1h.json())}
            market_data.last_prime[symbol] = time.time()

        # 3. PRE-TRAIN ML MODELS (this is the expensive part)
        print(f"[2] Training ML models (LightGBM + XGBoost + MLP) for {len(top_symbols)} symbols...")
        t_train = time.time()
        await ml_predictor.batch_pretrain(client, top_symbols)
        train_time = time.time() - t_train
        trained = [s for s in top_symbols if s in ml_predictor.models]
        print(f"    Trained: {len(trained)}/{len(top_symbols)} models in {train_time:.1f}s")
        print(f"    Models: LightGBM + XGBoost + MLP (ensemble)")

        # 4. Run benchmark WITH ML
        print(f"\n[3] Running 2-minute benchmark WITH ML inference...")
        stats = {"analyses": 0, "signals": 0, "times": [], "ml_times": []}
        start = time.time()
        cycle = 0

        while time.time() - start < DURATION:
            cycle += 1
            for symbol in top_symbols:
                if time.time() - start >= DURATION:
                    break
                t0 = time.time()
                
                # Measure ML prediction time separately
                t_ml = time.time()
                ml_prob = await ml_predictor.predict(client, symbol, market_data.klines[symbol]["1m"])
                ml_lat = (time.time() - t_ml) * 1000
                stats["ml_times"].append(ml_lat)
                
                # Full analysis (includes ML inside)
                result = await analyze_hybrid_async(client, symbol)
                lat = (time.time() - t0) * 1000
                stats["times"].append(lat)
                stats["analyses"] += 1
                if result and "WAIT" not in result.get("sig", "WAIT"):
                    stats["signals"] += 1

            elapsed = time.time() - start
            rate = stats["analyses"] / elapsed
            print(f"\r    Cycle {cycle} | Analyses: {stats['analyses']} | Signals: {stats['signals']} | Rate: {rate:.1f}/sec   ", end="", flush=True)
            await asyncio.sleep(0.05)

        total = time.time() - start
        at = np.array(stats["times"])
        mt = np.array(stats["ml_times"])

        print(f"\n\n{'='*60}")
        print(f"  RESULTS (WITH ML INFERENCE)")
        print(f"{'='*60}")
        print(f"  Duration:            {total:.1f}s")
        print(f"  Total Analyses:      {stats['analyses']}")
        print(f"  Analyses/minute:     {stats['analyses']/(total/60):.0f}")
        print(f"  Analyses/second:     {stats['analyses']/total:.1f}")
        print(f"  Signals Generated:   {stats['signals']}")
        print(f"  ML Models Active:    {len(trained)} (LGB+XGB+MLP ensemble)")
        print()
        print(f"  LATENCY (ms):")
        print(f"    Full Analysis:  avg={at.mean():.1f}  p50={np.median(at):.1f}  p95={np.percentile(at,95):.1f}")
        print(f"    ML Inference:   avg={mt.mean():.2f}  p50={np.median(mt):.2f}  p95={np.percentile(mt,95):.2f}")
        print(f"    ML Training:    {train_time:.1f}s total ({train_time/len(trained):.1f}s per model)")
        print()
        print(f"  COMPARISON:")
        print(f"    Human trader:   ~1-2 analyses/min")
        print(f"    This bot:       {stats['analyses']/(total/60):.0f} analyses/min")
        print(f"    Speed ratio:    {stats['analyses']/(total/60)/1.5:.0f}x faster than human")
        print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(run())
