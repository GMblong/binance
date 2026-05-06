import asyncio
import time
import httpx
from engine.api import binance_request
from utils.config import API_KEY, API_SECRET, API_URL

async def test():
    async with httpx.AsyncClient() as client:
        res = await binance_request(client, 'POST', '/fapi/v1/order', {
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "STOP_MARKET",
            "stopPrice": "60000.00",
            "closePosition": "true",
            "workingType": "MARK_PRICE"
        })
        if res:
            print(res.status_code)
            print(res.text)

asyncio.run(test())
