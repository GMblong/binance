import asyncio
import httpx
import time
from utils.config import API_KEY, API_SECRET, API_URL

async def test_data():
    print(f"Testing connection to: {API_URL}")
    print(f"API_KEY present: {'Yes' if API_KEY else 'No'}")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Test Public Ticker
        print("\n1. Testing Public Ticker (24hr)...")
        try:
            res = await client.get(f"{API_URL}/fapi/v1/ticker/24hr")
            if res.status_code == 200:
                data = res.json()
                print(f"✅ Success! Received {len(data)} tickers.")
                # Show top 3 by volume
                usdt_tickers = [t for t in data if t['symbol'].endswith('USDT')]
                top_3 = sorted(usdt_tickers, key=lambda x: float(x['quoteVolume']), reverse=True)[:3]
                for t in top_3:
                    print(f"   - {t['symbol']}: Vol {float(t['quoteVolume']):,.0f} USDT")
            else:
                print(f"❌ Failed! Status Code: {res.status_code}")
                print(f"   Response: {res.text}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

        # 2. Test Kline Fetching (with API Key header)
        print("\n2. Testing Kline Fetching (BTCUSDT 15m)...")
        try:
            headers = {'X-MBX-APIKEY': API_KEY} if API_KEY else {}
            res = await client.get(
                f"{API_URL}/fapi/v1/klines", 
                params={"symbol": "BTCUSDT", "interval": "15m", "limit": 5},
                headers=headers
            )
            if res.status_code == 200:
                print(f"✅ Success! Received {len(res.json())} candles.")
            else:
                print(f"❌ Failed! Status Code: {res.status_code}")
                print(f"   Response: {res.text}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

        # 3. Test Authenticated Request (Balance)
        if API_KEY and API_SECRET:
            print("\n3. Testing Authenticated Balance Request...")
            from engine.api import get_balance_async
            try:
                balance = await get_balance_async(client)
                print(f"✅ Success! Balance: {balance} USDT")
            except Exception as e:
                print(f"❌ Authenticated Error: {str(e)}")
        else:
            print("\n3. Skipping Authenticated Test (Keys missing)")

if __name__ == "__main__":
    asyncio.run(test_data())
