import asyncio
import httpx
from strategies.hybrid import analyze_hybrid_async
from utils.config import API_URL

async def test():
    async with httpx.AsyncClient() as client:
        result = await analyze_hybrid_async(client, "BTCUSDT")
        print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(test())
