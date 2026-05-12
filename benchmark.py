"""
Benchmark: Full Pipeline Speed Test (2 minutes)
================================================
Mengukur kecepatan bot dalam:
1. Fetch data (klines + orderbook)
2. Analisis teknikal + microstructure + superhuman signals
3. Generate signal (LONG/SHORT/WAIT)

Output: jumlah analisis per menit, latency per coin, total signals generated.
"""
import asyncio
import time
import httpx
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import API_URL, API_KEY
from utils.state import market_data, bot_state
from strategies.hybrid import analyze_hybrid_async, get_btc_trend
from engine.microstructure import micro_engine
from engine.superhuman import superhuman

DURATION = 120  # 2 minutes

async def run_benchmark():
    print("=" * 60)
    print("  BENCHMARK: Full Trading Pipeline Speed Test")
    print("  Duration: 2 minutes")
    print("=" * 60)
    print()

    stats = {
        "total_analyses": 0,
        "total_signals": 0,
        "fetch_times": [],
        "analysis_times": [],
        "superhuman_times": [],
        "micro_times": [],
        "symbols_analyzed": set(),
        "signals_generated": [],  # (symbol, signal, score, latency)
        "errors": 0,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        # 1. Initial setup: fetch tickers
        print("[1/4] Fetching market tickers...")
        t0 = time.time()
        headers = {'X-MBX-APIKEY': API_KEY}
        res = await client.get(f"{API_URL}/fapi/v1/ticker/24hr", headers=headers)
        if res.status_code != 200:
            print(f"ERROR: Cannot fetch tickers (status {res.status_code})")
            return
        
        tkr_data = res.json()
        tickers = [{"s": t["symbol"], "q": float(t["quoteVolume"]), "c": float(t["lastPrice"]), "o": float(t["openPrice"])}
                   for t in tkr_data if t["symbol"].endswith("USDT") and float(t["quoteVolume"]) > 10_000_000]
        tickers.sort(key=lambda x: x["q"], reverse=True)
        top_symbols = [t["s"] for t in tickers[:20]]
        for t in tickers:
            market_data.prices[t["s"]] = t["c"]
        market_data.tickers = tickers
        
        print(f"    Found {len(tickers)} liquid pairs, testing top 20")
        print(f"    Ticker fetch: {(time.time()-t0)*1000:.0f}ms")
        print()

        # 2. BTC Trend
        print("[2/4] Getting BTC trend...")
        await get_btc_trend(client)
        print(f"    BTC State: {bot_state.get('btc_state', 'UNKNOWN')}")
        print()

        # 3. Run benchmark loop
        print("[3/4] Running full pipeline benchmark for 2 minutes...")
        print(f"    Symbols: {', '.join([s.replace('USDT','') for s in top_symbols[:10]])}...")
        print()
        
        start_time = time.time()
        cycle = 0
        
        while time.time() - start_time < DURATION:
            cycle += 1
            cycle_start = time.time()
            
            # Analyze batch of symbols
            for symbol in top_symbols:
                if time.time() - start_time >= DURATION:
                    break
                
                t_fetch = time.time()
                
                # Fetch klines if not cached
                if symbol not in market_data.klines or "1m" not in market_data.klines.get(symbol, {}):
                    try:
                        res1, res15, res1h = await asyncio.gather(
                            client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "1m", "limit": 200}, headers=headers),
                            client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "15m", "limit": 100}, headers=headers),
                            client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": "1h", "limit": 50}, headers=headers),
                        )
                        if res1.status_code == 200 and res15.status_code == 200 and res1h.status_code == 200:
                            def proc(data):
                                df = pd.DataFrame(data).iloc[:, [0,1,2,3,4,5,9]]
                                df.columns = ["ot","o","h","l","c","v","tbv"]
                                for col in ["o","h","l","c","v","tbv"]: df[col] = df[col].astype(float)
                                return df
                            if symbol not in market_data.klines:
                                market_data.klines[symbol] = {}
                            market_data.klines[symbol]["1m"] = proc(res1.json())
                            market_data.klines[symbol]["15m"] = proc(res15.json())
                            market_data.klines[symbol]["1h"] = proc(res1h.json())
                            market_data.last_prime[symbol] = time.time()
                    except:
                        stats["errors"] += 1
                        continue
                
                fetch_latency = (time.time() - t_fetch) * 1000
                stats["fetch_times"].append(fetch_latency)
                
                # Run full analysis
                t_analysis = time.time()
                try:
                    result = await analyze_hybrid_async(client, symbol)
                    analysis_latency = (time.time() - t_analysis) * 1000
                    stats["analysis_times"].append(analysis_latency)
                    stats["total_analyses"] += 1
                    stats["symbols_analyzed"].add(symbol)
                    
                    if result and result.get("sig") and "WAIT" not in result["sig"]:
                        stats["total_signals"] += 1
                        stats["signals_generated"].append((
                            result["sym"], result["sig"], result.get("score", 0),
                            analysis_latency
                        ))
                except Exception as e:
                    stats["errors"] += 1
                
                # Benchmark superhuman signals standalone
                k = market_data.klines.get(symbol, {})
                if "1m" in k and "15m" in k:
                    t_sh = time.time()
                    superhuman.compute(symbol, k["1m"], k["15m"], k.get("1h"))
                    stats["superhuman_times"].append((time.time() - t_sh) * 1000)
                    
                    t_mi = time.time()
                    await micro_engine.compute(symbol, client=client)
                    stats["micro_times"].append((time.time() - t_mi) * 1000)
            
            # Progress
            elapsed = time.time() - start_time
            rate = stats["total_analyses"] / elapsed if elapsed > 0 else 0
            print(f"\r    Cycle {cycle} | Analyses: {stats['total_analyses']} | Signals: {stats['total_signals']} | Rate: {rate:.1f}/sec | Errors: {stats['errors']}   ", end="", flush=True)
            
            # Small delay to avoid rate limits
            await asyncio.sleep(0.1)
        
        total_time = time.time() - start_time
        print("\n")
        
        # 4. Results
        print("[4/4] BENCHMARK RESULTS")
        print("=" * 60)
        print(f"  Total Duration:          {total_time:.1f}s")
        print(f"  Total Analyses:          {stats['total_analyses']}")
        print(f"  Total Signals Generated: {stats['total_signals']}")
        print(f"  Unique Symbols:          {len(stats['symbols_analyzed'])}")
        print(f"  Errors:                  {stats['errors']}")
        print()
        print("  THROUGHPUT:")
        print(f"    Analyses/minute:       {stats['total_analyses'] / (total_time/60):.1f}")
        print(f"    Analyses/second:       {stats['total_analyses'] / total_time:.2f}")
        print(f"    Signals/minute:        {stats['total_signals'] / (total_time/60):.1f}")
        print()
        
        if stats["fetch_times"]:
            import numpy as np
            ft = np.array(stats["fetch_times"])
            at = np.array(stats["analysis_times"]) if stats["analysis_times"] else np.array([0])
            sh = np.array(stats["superhuman_times"]) if stats["superhuman_times"] else np.array([0])
            mi = np.array(stats["micro_times"]) if stats["micro_times"] else np.array([0])
            
            print("  LATENCY (ms):")
            print(f"    Data Fetch:     avg={ft.mean():.1f}  p50={np.median(ft):.1f}  p95={np.percentile(ft,95):.1f}  p99={np.percentile(ft,99):.1f}")
            print(f"    Full Analysis:  avg={at.mean():.1f}  p50={np.median(at):.1f}  p95={np.percentile(at,95):.1f}  p99={np.percentile(at,99):.1f}")
            print(f"    Superhuman:     avg={sh.mean():.2f}  p50={np.median(sh):.2f}  p95={np.percentile(sh,95):.2f}")
            print(f"    Microstructure: avg={mi.mean():.2f}  p50={np.median(mi):.2f}  p95={np.percentile(mi,95):.2f}")
            print()
            print(f"    Total per coin (fetch+analysis): avg={ft.mean()+at.mean():.1f}ms")
        
        if stats["signals_generated"]:
            print()
            print("  SIGNALS GENERATED:")
            print(f"    {'Symbol':<10} {'Signal':<14} {'Score':<6} {'Latency'}")
            print(f"    {'-'*10} {'-'*14} {'-'*6} {'-'*10}")
            for sym, sig, score, lat in stats["signals_generated"][:15]:
                print(f"    {sym:<10} {sig:<14} {score:<6} {lat:.1f}ms")
            if len(stats["signals_generated"]) > 15:
                print(f"    ... and {len(stats['signals_generated'])-15} more")
        
        print()
        print("=" * 60)
        print("  VERDICT:")
        if stats["total_analyses"] > 0:
            apm = stats['total_analyses'] / (total_time/60)
            if apm > 200:
                print("  🚀 EXCELLENT - Superhuman speed (>200 analyses/min)")
            elif apm > 100:
                print("  ✅ GOOD - Professional grade (>100 analyses/min)")
            elif apm > 50:
                print("  ⚡ DECENT - Acceptable performance (>50 analyses/min)")
            else:
                print("  ⚠️  SLOW - Needs optimization (<50 analyses/min)")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
