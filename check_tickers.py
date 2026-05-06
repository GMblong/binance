import asyncio
import httpx
from utils.config import API_URL

async def check():
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{API_URL}/fapi/v1/ticker/24hr")
        if res.status_code == 200:
            data = res.json()
            usdt_tickers = [t for t in data if t['symbol'].endswith('USDT')]
            print(f"Total USDT Tickers: {len(usdt_tickers)}")
            
            vol_50m = [t for t in usdt_tickers if float(t['quoteVolume']) > 50_000_000]
            print(f"Tickers with > 50M Volume: {len(vol_50m)}")
            
            vol_20m = [t for t in usdt_tickers if float(t['quoteVolume']) > 20_000_000]
            print(f"Tickers with > 20M Volume: {len(vol_20m)}")
            
            if vol_50m:
                print("\nTop 5 Movers (> 50M):")
                sorted_movers = sorted(vol_50m, key=lambda x: abs(float(x['priceChangePercent'])), reverse=True)[:5]
                for t in sorted_movers:
                    print(f"- {t['symbol']}: {t['priceChangePercent']}% | Vol: {float(t['quoteVolume']):,.0f}")
            else:
                print("\nNo tickers found with > 50M volume.")

if __name__ == "__main__":
    asyncio.run(check())
