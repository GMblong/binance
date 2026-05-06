import asyncio
import httpx
import pandas as pd
import pandas_ta as ta
import time
from utils.config import API_KEY, API_SECRET, API_URL
from utils.state import market_data, bot_state
from strategies.analyzer import MarketAnalyzer

async def test_step_by_step(symbol="BTCUSDT"):
    async with httpx.AsyncClient(timeout=30.0) as client:
        print(f"1. Fetching data for {symbol}...")
        headers = {'X-MBX-APIKEY': API_KEY} if API_KEY else {}
        
        async def fetch(interval):
            res = await client.get(f"{API_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": 250}, headers=headers)
            df = pd.DataFrame(res.json()).iloc[:, [0, 1, 2, 3, 4, 5, 9]]
            df.columns = ["ot", "o", "h", "l", "c", "v", "tbv"]
            df[["o", "h", "l", "c", "v", "tbv"]] = df[["o", "h", "l", "c", "v", "tbv"]].astype(float)
            return df

        d1m = await fetch("1m")
        d15m = await fetch("15m")
        print(f"   Done. 1m: {len(d1m)}, 15m: {len(d15m)}")

        print("2. Testing Indicators...")
        try:
            ema9 = ta.ema(d1m["c"], 9)
            print(f"   EMA9 (1m): {ema9.iloc[-1] if ema9 is not None else 'FAIL'}")
            
            regime = MarketAnalyzer.detect_regime(d15m)
            print(f"   Regime: {regime}")
            
            struct, fvg = MarketAnalyzer.detect_structure(d1m)
            print(f"   Structure (1m): {struct}, FVG: {fvg}")
            
            print("3. Testing Scoring...")
            # Mock current weights
            weights = {"liq": 1.0, "ml": 1.0, "ob": 1.0, "div": 1.0}
            score = MarketAnalyzer.calculate_score(d1m, d15m, 1, neural_weights=weights)
            print(f"   Score (Bullish): {score}")
            
            print("✅ All steps completed!")
        except Exception as e:
            print(f"❌ Failed at step: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_step_by_step("BTCUSDT"))
