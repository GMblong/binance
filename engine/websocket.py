import ssl
import json
import time
import asyncio
import websockets
from websockets.asyncio.client import connect
from websockets.protocol import State
from utils.state import bot_state, market_data
from utils.logger import log_error
from utils.intelligence import update_feature_weights
from engine.ml_engine import ml_predictor
from engine.trading import check_and_execute_exits
from engine.sentiment import sentiment_filter
from collections import deque

# Try to use orjson for faster JSON parsing (falls back to stdlib json)
try:
    import orjson
    def _loads(data):
        return orjson.loads(data)
except ImportError:
    _loads = json.loads

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
        
        # Desired streams: targeted ticker + 4 kline intervals for each symbol.
        # @aggTrade is expensive (hundreds of msg/sec for BTC) so we limit it
        # to the top-N coins where live CVD actually matters for entries.
        # @depth@500ms for top-N coins for real-time orderflow microstructure.
        AGG_TOP_N = 30
        DEPTH_TOP_N = 30
        agg_symbols = set(symbols[:AGG_TOP_N])
        depth_symbols = set(symbols[:DEPTH_TOP_N])
        desired = set()
        desired.add("!forceOrder@arr")  # Global liquidation stream
        for s in symbols:
            s_low = s.lower()
            desired.add(f"{s_low}@ticker")
            # Only 1m and 15m via WS (5m and 1h are updated via REST to save streams)
            for interval in ["1m", "15m"]:
                desired.add(f"{s_low}@kline_{interval}")
            if s in agg_symbols:
                desired.add(f"{s_low}@aggTrade")
            if s in depth_symbols:
                desired.add(f"{s_low}@depth@500ms")
        
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
                    await asyncio.sleep(0.1)  # Rate limit safety
                self.active_streams.update(to_subscribe)
            except Exception:
                pass  # WS subscription failures are retried on next cycle
            
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
            except Exception:
                pass  # WS unsubscription failures are non-critical

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
                            
                            raw_data = _loads(msg)
                            
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
                                            # Alert limit order fills
                                            if status == "FILLED" and o.get("o") == "LIMIT":
                                                from utils.telegram import alert_limit_filled
                                                asyncio.create_task(alert_limit_filled(sym, o.get("S", ""), o.get("L", o.get("p", ""))))
                                            # We trigger a background refresh to be 100% sure of the state
                                            async def refresh_pos():
                                                from engine.api import binance_request
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
                                            if not hasattr(self, '_processed_orders'):
                                                self._processed_orders = deque(maxlen=200)
                                            if oid not in self._processed_orders:
                                                rp = float(o.get("rp", 0))
                                                # Skip if already counted by close_position_async
                                                recently_closed = bot_state.get("_recently_closed", {})
                                                already_counted = sym in recently_closed and (time.time() - recently_closed[sym]) < 10
                                                
                                                if rp != 0 and not already_counted:
                                                    self._processed_orders.append(oid)
                                                    
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

                                                    # Feature-level feedback (only if we still have the entry snapshot)
                                                    trade_meta = bot_state.get("trades", {}).get(sym, {})
                                                    active_feats = trade_meta.get("active_features") or []
                                                    if active_feats:
                                                        update_feature_weights(active_feats, rp > 0)
                                                    # Clean stale trade snapshot so we don't double-credit
                                                    if sym in bot_state.get("trades", {}):
                                                        del bot_state["trades"][sym]
                                                elif rp != 0 and already_counted:
                                                    # Clean up the recently_closed marker
                                                    recently_closed.pop(sym, None)

                                # 2. Market Data
                                elif "@ticker" in stream_name:
                                    symbol = data['s']
                                    price = float(data['c'])
                                    market_data.prices[symbol] = price
                                    
                                    # FAST EXIT CHECK: If we have an active position, check exits immediately
                                    active_pos = bot_state.get("active_positions", [])
                                    if active_pos and any(p['symbol'] == symbol for p in active_pos):
                                        all_valid = bot_state.get("last_scan_results", [])
                                        asyncio.create_task(check_and_execute_exits(client, symbol, price, all_valid))
                                elif "@kline_" in stream_name:
                                    k = data['k']
                                    await market_data.update_kline(data['s'], k['i'], {
                                        "ot": k['t'], "o": float(k['o']), "h": float(k['h']),
                                        "l": float(k['l']), "c": float(k['c']),
                                        "v": float(k['v']), "tbv": float(k['V'])
                                    })
                                elif "@aggTrade" in stream_name:
                                    # Binance aggTrade payload: s=symbol, p=price, q=qty, m=isBuyerMaker, T=trade_time_ms
                                    try:
                                        sym_a = data['s']
                                        price_a = float(data['p'])
                                        qty_a = float(data['q'])
                                        is_bm = bool(data.get('m', False))
                                        ts_a = float(data.get('T', time.time() * 1000)) / 1000.0
                                        market_data.push_agg_trade(sym_a, ts_a, qty_a, price_a, is_bm)
                                    except Exception:
                                        pass
                                elif "@depth" in stream_name:
                                    # Binance depth stream: b=bids[[price,qty],...], a=asks[[price,qty],...]
                                    try:
                                        sym_d = stream_name.split("@")[0].upper()
                                        bids = data.get('b', [])
                                        asks = data.get('a', [])
                                        bid_total = sum(float(b[1]) for b in bids[:10])
                                        ask_total = sum(float(a[1]) for a in asks[:10])
                                        top_bid_qty = float(bids[0][1]) if bids else 0
                                        top_ask_qty = float(asks[0][1]) if asks else 0
                                        top_bid_px = float(bids[0][0]) if bids else 0
                                        top_ask_px = float(asks[0][0]) if asks else 0
                                        # Update imbalance + depth microstructure
                                        if ask_total > 0:
                                            market_data.imbalance[sym_d] = bid_total / ask_total
                                        market_data.push_depth_snapshot(sym_d, bid_total, ask_total, top_bid_qty, top_ask_qty)
                                        # Best quote for microprice (size-weighted fair value)
                                        if top_bid_px > 0 and top_ask_px > 0:
                                            market_data.push_best_quote(sym_d, top_bid_px, top_bid_qty, top_ask_px, top_ask_qty)
                                    except Exception:
                                        pass
                                elif "forceOrder" in stream_name:
                                    try:
                                        sentiment_filter.process_force_order(data)
                                        # Alert large liquidations
                                        from utils.telegram import alert_liquidation
                                        o = data.get("o", {})
                                        liq_sym = o.get("s", "")
                                        liq_side = o.get("S", "")
                                        liq_qty = float(o.get("q", 0))
                                        liq_price = float(o.get("p", 0))
                                        asyncio.create_task(alert_liquidation(liq_sym, liq_side, liq_qty, liq_price))
                                    except Exception:
                                        pass

                            if self.msg_count % 500 == 0:
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
