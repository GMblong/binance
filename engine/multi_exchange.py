"""
Multi-Exchange Price Feed (Bybit + OKX Public WebSocket)
No API key required — public market data only.
Used as leading indicator: cross-exchange divergence + aggregate CVD.
"""
import ssl
import json
import time
import asyncio
import websockets
from collections import deque
from websockets.asyncio.client import connect
from utils.state import market_data
from utils.logger import log_error


class BybitFeed:
    def __init__(self):
        self.uri = "wss://stream.bybit.com/v5/public/linear"
        self.prices = {}
        self.cvd_buf = {}  # {symbol: deque[(ts, signed_vol)]}
        self.running = True
        self.connected = False

    def get_divergence(self, symbol: str):
        bybit_price = self.prices.get(symbol, 0)
        binance_price = market_data.prices.get(symbol, 0)
        if bybit_price == 0 or binance_price == 0:
            return 0.0
        return ((bybit_price - binance_price) / binance_price) * 100

    def get_cvd(self, symbol: str, window_sec=60):
        buf = self.cvd_buf.get(symbol)
        if not buf:
            return 0.0
        cutoff = time.time() - window_sec
        return sum(d for ts, d in buf if ts >= cutoff)

    async def start(self, symbols=None):
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        while self.running:
            try:
                async with connect(self.uri, ssl=ssl_context, ping_interval=20) as ws:
                    self.connected = True
                    top_syms = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                                           "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT"]
                    args = [f"tickers.{s}" for s in top_syms]
                    args += [f"publicTrade.{s}" for s in top_syms[:5]]
                    await ws.send(json.dumps({"op": "subscribe", "args": args}))

                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                            data = json.loads(msg)
                            topic = data.get("topic", "")
                            if topic.startswith("tickers."):
                                d = data.get("data", {})
                                sym = d.get("symbol", "")
                                last_price = d.get("lastPrice")
                                if sym and last_price:
                                    self.prices[sym] = float(last_price)
                            elif topic.startswith("publicTrade."):
                                for t in data.get("data", []):
                                    sym = t.get("symbol", "")
                                    qty = float(t.get("size", 0))
                                    price = float(t.get("price", 0))
                                    side = t.get("side", "")
                                    signed = qty * price * (1.0 if side == "Buy" else -1.0)
                                    if sym not in self.cvd_buf:
                                        self.cvd_buf[sym] = deque(maxlen=500)
                                    self.cvd_buf[sym].append((time.time(), signed))
                        except asyncio.TimeoutError:
                            break
                        except Exception:
                            break
            except Exception as e:
                self.connected = False
                log_error(f"Bybit WS Error: {str(e)[:50]}")
                await asyncio.sleep(10)


class OKXFeed:
    def __init__(self):
        self.uri = "wss://ws.okx.com:8443/ws/v5/public"
        self.prices = {}
        self.cvd_buf = {}
        self.running = True
        self.connected = False

    def _to_binance_sym(self, okx_id: str) -> str:
        """Convert OKX instId (BTC-USDT-SWAP) to Binance format (BTCUSDT)."""
        parts = okx_id.split("-")
        if len(parts) >= 2:
            return parts[0] + parts[1]
        return okx_id

    def get_divergence(self, symbol: str):
        okx_price = self.prices.get(symbol, 0)
        binance_price = market_data.prices.get(symbol, 0)
        if okx_price == 0 or binance_price == 0:
            return 0.0
        return ((okx_price - binance_price) / binance_price) * 100

    def get_cvd(self, symbol: str, window_sec=60):
        buf = self.cvd_buf.get(symbol)
        if not buf:
            return 0.0
        cutoff = time.time() - window_sec
        return sum(d for ts, d in buf if ts >= cutoff)

    async def start(self):
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        top_pairs = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
                     "BNB-USDT-SWAP", "XRP-USDT-SWAP", "DOGE-USDT-SWAP",
                     "ADA-USDT-SWAP", "AVAX-USDT-SWAP", "LINK-USDT-SWAP", "SUI-USDT-SWAP"]

        while self.running:
            try:
                async with connect(self.uri, ssl=ssl_context, ping_interval=20) as ws:
                    self.connected = True
                    # Subscribe tickers + trades
                    args = [{"channel": "tickers", "instId": p} for p in top_pairs]
                    args += [{"channel": "trades", "instId": p} for p in top_pairs[:5]]
                    await ws.send(json.dumps({"op": "subscribe", "args": args}))

                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                            data = json.loads(msg)
                            if "data" not in data:
                                continue
                            ch = data.get("arg", {}).get("channel", "")
                            for d in data["data"]:
                                inst_id = d.get("instId", "")
                                sym = self._to_binance_sym(inst_id)
                                if ch == "tickers":
                                    last = d.get("last")
                                    if last:
                                        self.prices[sym] = float(last)
                                elif ch == "trades":
                                    qty = float(d.get("sz", 0))
                                    price = float(d.get("px", 0))
                                    side = d.get("side", "")
                                    signed = qty * price * (1.0 if side == "buy" else -1.0)
                                    if sym not in self.cvd_buf:
                                        self.cvd_buf[sym] = deque(maxlen=500)
                                    self.cvd_buf[sym].append((time.time(), signed))
                        except asyncio.TimeoutError:
                            break
                        except Exception:
                            break
            except Exception as e:
                self.connected = False
                log_error(f"OKX WS Error: {str(e)[:50]}")
                await asyncio.sleep(10)


class AggregateOrderflow:
    """Aggregate CVD and divergence across Binance + Bybit + OKX."""

    def __init__(self, bybit: BybitFeed, okx: OKXFeed):
        self.bybit = bybit
        self.okx = okx

    def get_cross_exchange_signal(self, symbol: str):
        """Returns: (aggregate_divergence_pct, aggregate_cvd_bias)
        
        divergence > 0: other exchanges higher (bullish for Binance)
        cvd_bias > 0: net buying across exchanges
        """
        bybit_div = self.bybit.get_divergence(symbol)
        okx_div = self.okx.get_divergence(symbol)
        # Average divergence (weighted: Bybit slightly more liquid)
        avg_div = bybit_div * 0.55 + okx_div * 0.45 if (bybit_div != 0 or okx_div != 0) else 0.0

        # Aggregate CVD from all sources
        binance_cvd, _ = market_data.get_live_cvd(symbol, window_sec=60)
        bybit_cvd = self.bybit.get_cvd(symbol, 60)
        okx_cvd = self.okx.get_cvd(symbol, 60)

        total_cvd = binance_cvd + bybit_cvd + okx_cvd
        # Normalize to a -1 to +1 scale roughly
        # Use Binance 1m candle volume as reference
        norm = 1.0
        k = market_data.klines.get(symbol, {}).get("1m")
        if k is not None and len(k) > 0:
            last_vol = float(k.iloc[-1]['v'] * k.iloc[-1]['c'])
            if last_vol > 0:
                norm = last_vol
        cvd_bias = total_cvd / (norm + 1.0)
        cvd_bias = max(-1.0, min(1.0, cvd_bias))

        return avg_div, cvd_bias


bybit_feed = BybitFeed()
okx_feed = OKXFeed()
aggregate_flow = AggregateOrderflow(bybit_feed, okx_feed)
