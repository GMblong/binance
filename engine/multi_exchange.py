"""
Multi-Exchange Price Feed (Bybit Public WebSocket)
No API key required — public market data only.
Used as leading indicator: if Bybit price diverges from Binance → signal.
"""
import ssl
import json
import time
import asyncio
import websockets
from websockets.asyncio.client import connect
from utils.state import market_data
from utils.logger import log_error


class BybitFeed:
    def __init__(self):
        self.uri = "wss://stream.bybit.com/v5/public/linear"
        self.prices = {}  # {symbol: price}
        self.running = True
        self.connected = False

    def get_divergence(self, symbol: str):
        """Calculate price divergence between Bybit and Binance.
        
        Returns: float (positive = Bybit higher, negative = Bybit lower)
        As percentage: +0.05 means Bybit is 0.05% higher than Binance.
        """
        bybit_price = self.prices.get(symbol, 0)
        binance_price = market_data.prices.get(symbol, 0)
        if bybit_price == 0 or binance_price == 0:
            return 0.0
        return ((bybit_price - binance_price) / binance_price) * 100

    async def start(self, symbols=None):
        """Connect to Bybit public WS and stream prices."""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        while self.running:
            try:
                async with connect(self.uri, ssl=ssl_context, ping_interval=20) as ws:
                    self.connected = True
                    # Subscribe to top coins tickers
                    top_syms = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                                           "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT"]
                    args = [f"tickers.{s}" for s in top_syms]
                    await ws.send(json.dumps({"op": "subscribe", "args": args}))

                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                            data = json.loads(msg)
                            if data.get("topic", "").startswith("tickers."):
                                d = data.get("data", {})
                                sym = d.get("symbol", "")
                                last_price = d.get("lastPrice")
                                if sym and last_price:
                                    self.prices[sym] = float(last_price)
                        except asyncio.TimeoutError:
                            break
                        except Exception:
                            break
            except Exception as e:
                self.connected = False
                log_error(f"Bybit WS Error: {str(e)[:50]}")
                await asyncio.sleep(10)


bybit_feed = BybitFeed()
