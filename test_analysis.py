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
    
    # Check if we are in a venv
    import sys
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️ WARNING: It seems you are NOT running this in a virtual environment (venv).")
        print("   If you see 'ModuleNotFoundError', please run: ./venv/bin/python3 test_analysis.py")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # We need to mock the bot_state for the analysis to work
        bot_state["ws_last_msg"] = time.time()
        
        print("Running analyze_hybrid_async...")
        try:
            start_t = time.time()
            result = await analyze_hybrid_async(client, symbol)
            end_t = time.time()
            
            if result:
                print(f"✅ Analysis SUCCESS! (Took {end_t - start_t:.2f}s)")
                print(f"   Signal: {result['sig']}")
                print(f"   Score: {result['score']}")
                print(f"   Price: {result['price']}")
                print(f"   Limit Price: {result.get('limit', 'N/A')}")
                print(f"   Structure: {result.get('struct', 'N/A')}")
                print(f"   Regime: {result.get('regime', 'N/A')}")
                print(f"   AI Brain (Clean): {result.get('ai', 'N/A')}")
                
                # Verify types
                if isinstance(result.get('score'), (int, float)):
                    pass
                else:
                    print(f"   ⚠️ Type Warning: Score is {type(result.get('score'))}")
            else:
                print("❌ Analysis returned NONE.")
                # Let's check why by inspecting market_data
                k = market_data.klines.get(symbol, {})
                print(f"   Market Data Klines for {symbol}: {list(k.keys())}")
                if not k:
                    print("   ⚠️ No kline data found. Check sync_debug.log for API errors.")
                for interval, df in k.items():
                    print(f"   - {interval}: {len(df)} rows")
        except Exception as e:
            print(f"🔥 Analysis CRASHED: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    init_logger()
    asyncio.run(test_full_analysis("BTCUSDT"))
