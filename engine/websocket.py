import ssl
import json
import time
import asyncio
import websockets
from websockets.asyncio.client import connect
from websockets.protocol import State
from utils.state import bot_state, market_data
from utils.logger import log_error
import pandas as pd

class WebSocketManager:
    def __init__(self):
        # Base URI for targeted streams
        self.base_uri = "wss://fstream.binance.com"
        self.running = True
        self.last_msg_time = 0
        self.msg_count = 0
        self.reconnect_count = 0
        self.active_streams = set()
        self.ws = None
        self.listen_key = None
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    async def update_subscriptions(self, symbols):
        """Dynamically update WS subscriptions based on scan list."""
        if not self.ws or self.ws.state != State.OPEN: return
        
        # Desired streams: targeted ticker + 4 kline intervals for each symbol
        desired = set()
        for s in symbols:
            s_low = s.lower()
            desired.add(f"{s_low}@ticker") # Targeted ticker instead of global array
            for interval in ["1m", "5m", "15m", "1h"]:
                desired.add(f"{s_low}@kline_{interval}")
        
        to_subscribe = list(desired - self.active_streams)
        to_unsubscribe = list(self.active_streams - desired)
        
        if to_subscribe:
            try:
                # Batch subscriptions
                for i in range(0, len(to_subscribe), 20):
                    batch = to_subscribe[i:i+20]
                    await self.ws.send(json.dumps({
                        "method": "SUBSCRIBE",
                        "params": batch,
                        "id": int(time.time() * 1000) % 100000
                    }))
                self.active_streams.update(to_subscribe)
            except: pass
            
        if to_unsubscribe:
            try:
                for i in range(0, len(to_unsubscribe), 20):
                    batch = to_unsubscribe[i:i+20]
                    await self.ws.send(json.dumps({
                        "method": "UNSUBSCRIBE",
                        "params": batch,
                        "id": int(time.time() * 1000) % 100000
                    }))
                for s in to_unsubscribe: self.active_streams.discard(s)
            except: pass

    async def start(self, client):
        from engine.api import get_listen_key, keep_alive_listen_key
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        while self.running:
            try:
                self.listen_key = await get_listen_key(client)
                if not self.listen_key:
                    await asyncio.sleep(5)
                    continue

                # Combine listenKey with market streams for single connection
                uri = f"{self.base_uri}/stream?streams={self.listen_key}"
                
                bot_state["last_log"] = f"[bold yellow]WS Connecting...[/]"
                async with connect(
                    uri,
                    ssl=ssl_context,
                    additional_headers=self.headers,
                    ping_interval=None, 
                    max_size=2**22 
                ) as ws:
                    self.ws = ws
                    self.reconnect_count = 0
                    bot_state["ws_online"] = True
                    bot_state["last_log"] = "[bold cyan]WS Connected (User + Market Data)[/]"
                    
                    # Start ListenKey Keep-Alive Task
                    async def keep_alive_task():
                        while self.ws and self.ws.state == State.OPEN:
                            await asyncio.sleep(1800) # 30 minutes
                            await keep_alive_listen_key(client, self.listen_key)
                    
                    ka_task = asyncio.create_task(keep_alive_task())

                    # Re-subscribe to previously active market streams
                    if self.active_streams:
                        streams = list(self.active_streams)
                        for i in range(0, len(streams), 20):
                            batch = streams[i:i+20]
                            await ws.send(json.dumps({
                                "method": "SUBSCRIBE",
                                "params": batch,
                                "id": 1
                            }))

                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=60)
                            self.last_msg_time = time.time()
                            self.msg_count += 1
                            bot_state["ws_msg_count"] = self.msg_count
                            bot_state["ws_last_msg"] = self.last_msg_time
                            
                            raw_data = json.loads(msg)
                            
                            if "stream" in raw_data:
                                stream_name = raw_data["stream"]
                                data = raw_data["data"]
                                
                                # 1. User Data (ListenKey Stream)
                                if stream_name == self.listen_key:
                                    event_type = data.get("e")
                                    if event_type == "ACCOUNT_UPDATE":
                                        for asset in data.get("a", {}).get("B", []):
                                            if asset["a"] == "USDT":
                                                bot_state["balance"] = float(asset["wb"])
                                    elif event_type == "ORDER_TRADE_UPDATE":
                                        o = data["o"]
                                        if o["X"] == "FILLED":
                                            bot_state["last_log"] = f"[bold green]WS: {o['s']} Filled at {o['L']}[/]"
                                            rp = float(o.get("rp", 0))
                                            if rp != 0:
                                                bot_state["daily_pnl"] = bot_state.get("daily_pnl", 0.0) + rp
                                                if rp > 0:
                                                    bot_state["wins"] = bot_state.get("wins", 0) + 1
                                                elif rp < 0:
                                                    bot_state["losses"] = bot_state.get("losses", 0) + 1

                                # 2. Market Data
                                elif "@ticker" in stream_name:
                                    market_data.prices[data['s']] = float(data['c'])
                                elif "@kline_" in stream_name:
                                    k = data['k']
                                    await market_data.update_kline(data['s'], k['i'], {
                                        "ot": k['t'], "o": float(k['o']), "h": float(k['h']),
                                        "l": float(k['l']), "c": float(k['c']),
                                        "v": float(k['v']), "tbv": float(k['V'])
                                    })

                            if self.msg_count % 100 == 0:
                                bot_state["last_log"] = f"[bold green]WS Live: {self.msg_count} pkts | {len(self.active_streams)} streams[/]"

                        except (asyncio.TimeoutError, websockets.ConnectionClosed):
                            ka_task.cancel()
                            break 
                        except Exception as e:
                            log_error(f"WS Message Error: {str(e)}")
                            await asyncio.sleep(1)
                            break
            except Exception as e:
                bot_state["ws_online"] = False
                self.reconnect_count += 1
                log_error(f"WS Connection Error ({self.reconnect_count}): {str(e)}")
                await asyncio.sleep(min(15, 2 * self.reconnect_count))

ws_manager = WebSocketManager()
