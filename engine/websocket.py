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
        self.base_uri = "wss://fstream.binancefuture.com/stream"
        self.running = True
        self.last_msg_time = 0
        self.msg_count = 0
        self.reconnect_count = 0
        self.active_streams = set()
        self.ws = None
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

    async def start(self):
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        while self.running:
            try:
                bot_state["last_log"] = f"[bold yellow]WS Connecting...[/]"
                async with connect(
                    self.base_uri,
                    ssl=ssl_context,
                    additional_headers=self.headers,
                    # Disable internal ping for Termux stability, let the server or TCP handle it
                    ping_interval=None, 
                    max_size=2**22 # 4MB buffer is plenty for targeted streams
                ) as ws:
                    self.ws = ws
                    self.reconnect_count = 0
                    bot_state["ws_online"] = True
                    bot_state["last_log"] = "[bold cyan]WS Connected (Targeted Mode)[/]"
                    
                    # Re-subscribe to previously active streams if any
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
                            stream_name = raw_data.get("stream", "")
                            data = raw_data.get("data")
                            if not data: continue

                            # 1. Handle Targeted Ticker
                            if "@ticker" in stream_name:
                                s = data['s']
                                p = float(data['c'])
                                market_data.prices[s] = p
                                # Note: global market_data.tickers list will be maintained by REST fallback
                                # to ensure we always have the top mover context.
                            
                            # 2. Handle Real-time Klines
                            elif "@kline_" in stream_name:
                                s = data['s']
                                k = data['k']
                                await market_data.update_kline(s, k['i'], {
                                    "ot": k['t'], "o": float(k['o']), "h": float(k['h']),
                                    "l": float(k['l']), "c": float(k['c']),
                                    "v": float(k['v']), "tbv": float(k['V'])
                                })

                            if self.msg_count % 100 == 0:
                                bot_state["last_log"] = f"[bold green]WS Live: {self.msg_count} pkts | {len(self.active_streams)} streams[/]"

                        except (asyncio.TimeoutError, websockets.ConnectionClosed):
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
