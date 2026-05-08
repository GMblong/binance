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
                                        sym = o['s']
                                        oid = o['i']
                                        status = o['X']
                                        
                                        # Update active_positions cache on Fill/Partial
                                        if status in ["FILLED", "PARTIALLY_FILLED"]:
                                            # We trigger a background refresh to be 100% sure of the state
                                            from engine.api import binance_request
                                            async def refresh_pos():
                                                res = await binance_request(client, 'GET', '/fapi/v2/positionRisk')
                                                if res and res.status_code == 200:
                                                    bot_state["active_positions"] = [p for p in res.json() if float(p['positionAmt']) != 0]
                                            asyncio.create_task(refresh_pos())

                                        # 1. Update Daily PnL on every trade execution
                                        if o.get("x") == "TRADE":
                                            trade_rp = float(o.get("rp", 0))
                                            if trade_rp != 0:
                                                bot_state["daily_pnl"] = bot_state.get("daily_pnl", 0.0) + trade_rp
                                        
                                        # 2. IMMEDIATELY CLEAN UP LIMIT ORDERS ON FILL/CANCEL
                                        if status in ["FILLED", "PARTIALLY_FILLED", "CANCELED", "EXPIRED"]:
                                            if sym in bot_state.get("limit_orders", {}):
                                                # If it's the same order ID, remove it
                                                if bot_state["limit_orders"][sym].get("orderId") == oid:
                                                    del bot_state["limit_orders"][sym]
                                        
                                        # 3. Handle W/L and Logging on FILLED or significant TRADE
                                        if status == "FILLED":
                                            bot_state["last_log"] = f"[bold green]WS: {sym} Filled at {o['L']}[/]"
                                            
                                            # Avoid double counting W/L for the same order
                                            if not hasattr(self, '_processed_orders'): self._processed_orders = set()
                                            if oid not in self._processed_orders:
                                                rp = float(o.get("rp", 0))
                                                # If rp is 0 on the final fill, it might have been in partials, 
                                                # but for simplicity we count it if non-zero.
                                                if rp != 0:
                                                    self._processed_orders.add(oid)
                                                    if len(self._processed_orders) > 200: self._processed_orders.pop() # Keep it small
                                                    
                                                    if rp > 0:
                                                        ml_predictor.update_performance(sym, True)
                                                        bot_state["wins"] = bot_state.get("wins", 0) + 1
                                                        if sym not in bot_state["sym_perf"]: bot_state["sym_perf"][sym] = {'w':0, 'l':0, 'c':0}
                                                        bot_state["sym_perf"][sym]['w'] += 1
                                                        bot_state["sym_perf"][sym]['c'] = 0
                                                    else:
                                                        ml_predictor.update_performance(sym, False)
                                                        bot_state["losses"] = bot_state.get("losses", 0) + 1
                                                        if sym not in bot_state["sym_perf"]: bot_state["sym_perf"][sym] = {'w':0, 'l':0, 'c':0}
                                                        bot_state["sym_perf"][sym]['l'] += 1
                                                        bot_state["sym_perf"][sym]['c'] += 1
                                                        bot_state["sym_perf"][sym]['last_loss_time'] = time.time()

                                # 2. Market Data
                                elif "@ticker" in stream_name:
                                    symbol = data['s']
                                    price = float(data['c'])
                                    market_data.prices[symbol] = price
                                    
                                    # FAST EXIT CHECK: If we have an active position, check exits immediately
                                    active_symbols = [p['symbol'] for p in bot_state.get("active_positions", [])]
                                    if symbol in active_symbols:
                                        # Trigger fast exit check using current analysis context
                                        from engine.trading import check_and_execute_exits
                                        all_valid = bot_state.get("last_scan_results", [])
                                        asyncio.create_task(check_and_execute_exits(client, symbol, price, all_valid))
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
