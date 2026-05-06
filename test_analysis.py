import asyncio
import httpx
import pandas as pd
import pandas_ta as ta
import time
from utils.config import API_KEY, API_SECRET, API_URL
from strategies.hybrid import analyze_hybrid_async
from utils.state import market_data, bot_state
from utils.logger import init_logger

async def test_full_analysis(symbol="BTCUSDT"):
    print(f"--- Testing Analysis for {symbol} ---")
    async with httpx.AsyncClient(timeout=30.0) as client:
        # We need to mock the bot_state for the analysis to work
        bot_state["ws_last_msg"] = time.time()
        
        print("Running analyze_hybrid_async...")
        try:
            result = await analyze_hybrid_async(client, symbol)
            if result:
                print("✅ Analysis SUCCESS!")
                print(f"   Signal: {result['sig']}")
                print(f"   Score: {result['score']}")
                print(f"   Price: {result['price']}")
            else:
                print("❌ Analysis returned NONE.")
                # Let's check why by inspecting market_data
                print(f"   Market Data Klines for {symbol}: {list(market_data.klines.get(symbol, {}).keys())}")
                for interval, df in market_data.klines.get(symbol, {}).items():
                    print(f"   - {interval}: {len(df)} rows")
        except Exception as e:
            print(f"🔥 Analysis CRASHED: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    init_logger()
    asyncio.run(test_full_analysis("BTCUSDT"))
