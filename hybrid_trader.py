import os
import sys
import time
import httpx
import asyncio
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.align import Align

from utils.config import API_KEY, API_SECRET, API_URL, MAX_POSITIONS, USE_BTC_FILTER
from utils.state import bot_state, market_data
from engine.websocket import ws_manager
from engine.api import binance_request, get_balance_async, get_market_depth_data
from engine.trading import close_position_async, manage_active_positions, open_position_async, manage_limit_orders
from ui.dashboard import generate_dashboard_async
from strategies.hybrid import get_btc_trend, analyze_hybrid_async
from utils.database import init_db, load_state_from_db, save_state_to_db
from utils.intelligence import calculate_market_volatility, is_correlated_exposure
from utils.logger import init_logger, log_error

console = Console()
init_db()
load_state_from_db()

if not API_KEY or not API_SECRET or API_KEY == "YOUR_API_KEY":
    console.print("[bold red]ERROR: API_KEY or API_SECRET missing in .env file![/]")
    exit(1)

async def trading_loop(client):
    """Dedicated loop for trading logic to ensure separation from UI."""
    # Initial Sync
    await get_balance_async(client)
    
    while True:
        try:
            # 1. Update Core Data
            await get_btc_trend(client)
            
            # 2. Use Tickers from WS (Updated in real-time)
            # FALLBACK: If WS is offline or data is older than 5 seconds, poll REST
            is_ws_stale = (time.time() - bot_state.get("ws_last_msg", 0)) > 5
            
            # Periodically fetch Funding and OI for top symbols
            if bot_state["heartbeat"] % 10 == 0:
                # We do this for current top symbols
                asyncio.create_task(get_market_depth_data(client, market_data.current_scan_list))

            if not market_data.tickers or is_ws_stale:
                if is_ws_stale and bot_state.get("ws_msg_count", 0) > 0:
                    bot_state["last_log"] = "[yellow]WS Lagging: Falling back to REST...[/]"
                
                ticker_res = await client.get(f"{API_URL}/fapi/v1/ticker/24hr")
                if ticker_res.status_code == 200:
                    tkr_data = ticker_res.json()
                    tkr = []
                    for t in tkr_data:
                        symbol = t["symbol"]
                        if not symbol.endswith("USDT"): continue
                        price = float(t["lastPrice"])
                        tkr.append({
                            "s": symbol, 
                            "q": float(t["quoteVolume"]), 
                            "c": price, 
                            "o": float(t["openPrice"])
                        })
                        # CRITICAL: Sync prices for UI
                        market_data.prices[symbol] = price
                        
                    market_data.tickers = tkr
            
            if not market_data.tickers:
                await asyncio.sleep(1)
                continue

            # Sort by volume and calculate change from open (approx 24h change)
            for t in market_data.tickers:
                if t.get("o") and float(t["o"]) > 0:
                    t["cp"] = ((float(t["c"]) - float(t["o"])) / float(t["o"])) * 100
                else: t["cp"] = 0

            filtered = [t for t in market_data.tickers if float(t.get("q", 0)) > 20_000_000]
            top_movers = sorted(filtered, key=lambda x: abs(float(x.get("cp", 0))), reverse=True)[:10]
            top_symbols = [t["s"] for t in top_movers]
            
            # Sync WS Subscriptions with current top symbols
            await ws_manager.update_subscriptions(top_symbols)
            
            # 3. Analyze Symbols in Parallel
            # We fetch active positions once here and store in bot_state for UI and logic
            active_pos = await manage_active_positions(client, []) # Empty signals for now to get initial list
            bot_state["active_positions"] = active_pos
            active_pos_symbols = [p['symbol'] for p in active_pos]
            limit_symbols = list(bot_state.get("limit_orders", {}).keys())
            
            # Update Intelligence Metrics
            calculate_market_volatility()
            net_bias = sum([1 if float(p['positionAmt']) > 0 else -1 for p in active_pos])
            bot_state["directional_bias"] = net_bias

            final_symbols = list(set(top_symbols + active_pos_symbols + limit_symbols))
            market_data.current_scan_list = final_symbols
            
            # Use asyncio.gather for parallel analysis
            tasks = [analyze_hybrid_async(client, s) for s in final_symbols]
            all_results = await asyncio.gather(*tasks)
            
            all_valid = [r for r in all_results if r]
            
            # 4. Manage & Execute
            # Re-run manage with actual signals for reversal logic
            active_pos = await manage_active_positions(client, all_valid)
            await manage_limit_orders(client, all_valid)
            bot_state["active_positions"] = active_pos
            
            # --- MULTI-ORDER EXECUTION LOGIC ---
            limit_orders = bot_state.get("limit_orders", {})
            current_slots_occupied = len(active_pos) + len(limit_orders)
            
            if current_slots_occupied < MAX_POSITIONS and not bot_state.get("is_passive", False):
                for s in all_valid:
                    if "WAIT" in s["sig"] or s["ai"] is None: continue
                    sym_full = s["sym"] + "USDT"
                    
                    # 1. STRICT SINGLE POSITION POLICY
                    # Skip if symbol is already an active position
                    if any(p['symbol'] == sym_full for p in active_pos): continue
                    
                    # 2. Check if we have room for NEW symbols
                    is_update = sym_full in limit_orders
                    if is_update:
                        existing = limit_orders[sym_full]
                        # Throttle Updates: Only update if signal changed or limit price moved > 0.1%
                        price_diff = abs(s["ai"]["limit_price"] - float(existing["price"])) / float(existing["price"]) * 100
                        if s["sig"] == existing.get("sig") and price_diff < 0.1:
                            continue
                    elif current_slots_occupied >= MAX_POSITIONS:
                        # Skip new symbols if all slots are filled
                        continue
                    
                    sig_side = "LONG" if s["dir"] == 1 else "SHORT"
                    
                    # --- ULTRA-SMART FILTERS ---
                    # 1. Correlation Check: Don't add more exposure to same-direction correlated coins
                    if is_correlated_exposure(sym_full, sig_side):
                        if not is_update: # Only skip if it's a new trade attempt
                            bot_state["last_log"] = f"[yellow]SKIP {s['sym']}: High Correlation[/]"
                            continue
                        
                    # 2. Directional Bias Check: Limit bias in choppy markets
                    if bot_state["market_vol"] > 1.5 or "RANGING" in s.get("regime", ""):
                        if abs(net_bias) >= 2 and ( (net_bias > 0 and sig_side == "LONG") or (net_bias < 0 and sig_side == "SHORT") ):
                            if not is_update:
                                bot_state["last_log"] = f"[yellow]SKIP {s['sym']}: Bias Limit ({net_bias})[/]"
                                continue

                    can_open = False
                    if not USE_BTC_FILTER or bot_state["btc_dir"] == 0:
                        can_open = True
                    elif sig_side == "LONG" and bot_state["btc_dir"] == 1:
                        can_open = True
                    elif sig_side == "SHORT" and bot_state["btc_dir"] == -1:
                        can_open = True
                    
                    if can_open:
                        await open_position_async(client, sym_full, "BUY" if sig_side == "LONG" else "SELL", s["sig"], s["ai"])
                        if not is_update:
                            current_slots_occupied += 1

                # Update balance once after potentially opening multiple positions
                await get_balance_async(client)

            # Store results for UI
            bot_state["last_scan_results"] = all_valid
            bot_state["heartbeat"] += 1
            save_state_to_db() # Persist state every cycle
            
            # If WS is lagging, we want to loop faster to simulate real-time via REST
            loop_sleep = 0.5 if is_ws_stale else 1.0
            await asyncio.sleep(loop_sleep)
        except Exception as e:
            err_msg = str(e)[:50]
            bot_state["last_log"] = f"[red]Trading Loop Err: {err_msg}[/]"
            log_error(f"Trading Loop Exception: {err_msg}")
            await asyncio.sleep(2)

async def cleanup_and_exit(client):
    """Clean up all pending orders before exiting."""
    bot_state["last_log"] = "[bold red]EXITING: Cleaning up all limit orders...[/]"
    limit_orders = list(bot_state.get("limit_orders", {}).keys())
    if limit_orders:
        for sym in limit_orders:
            try:
                await binance_request(client, 'DELETE', '/fapi/v1/allOpenOrders', {"symbol": sym})
            except: pass
    os._exit(0)

async def main():
    init_logger() # Reset log file
    console.clear()
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Start background tasks
        asyncio.create_task(ws_manager.start())
        asyncio.create_task(trading_loop(client))
        
        with Live(None, refresh_per_second=2, screen=False) as live:
            # Task for keyboard shortcuts
            async def handle_keys():
                while True:
                    try:
                        key = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                        key_clean = key.strip().lower()
                        if key_clean == 'c':
                            bot_state["last_log"] = "[bold red]EMERGENCY: Closing All Positions...[/]"
                            res = await binance_request(client, 'GET', '/fapi/v2/positionRisk')
                            if res and res.status_code == 200:
                                positions = [p for p in res.json() if float(p['positionAmt']) != 0]
                                for p in positions:
                                    symbol, amt = p['symbol'], float(p['positionAmt'])
                                    side = "LONG" if amt > 0 else "SHORT"
                                    await close_position_async(client, symbol, side, amt, "MANUAL-POS-ONLY")
                        elif key_clean == 'k':
                            bot_state["last_log"] = "[bold yellow]EMERGENCY: Canceling All Limit Orders...[/]"
                            limit_orders = list(bot_state.get("limit_orders", {}).keys())
                            for sym in limit_orders:
                                await binance_request(client, 'DELETE', '/fapi/v1/allOpenOrders', {"symbol": sym})
                            bot_state["limit_orders"] = {}
                        elif key_clean == 'a':
                            bot_state["last_log"] = "[bold red]EMERGENCY: CLEARING EVERYTHING...[/]"
                            # 1. Cancel Orders
                            limit_orders = list(bot_state.get("limit_orders", {}).keys())
                            for sym in limit_orders:
                                await binance_request(client, 'DELETE', '/fapi/v1/allOpenOrders', {"symbol": sym})
                            bot_state["limit_orders"] = {}
                            # 2. Close Positions
                            res = await binance_request(client, 'GET', '/fapi/v2/positionRisk')
                            if res and res.status_code == 200:
                                positions = [p for p in res.json() if float(p['positionAmt']) != 0]
                                for p in positions:
                                    symbol, amt = p['symbol'], float(p['positionAmt'])
                                    side = "LONG" if amt > 0 else "SHORT"
                                    await close_position_async(client, symbol, side, amt, "MANUAL-ALL")
                        elif key_clean == 'p':
                            bot_state["is_passive"] = not bot_state.get("is_passive", False)
                            status = "ENABLED" if bot_state["is_passive"] else "DISABLED"
                            bot_state["last_log"] = f"[bold yellow]PASSIVE MODE: {status}[/]"
                        elif key_clean == 'x':
                            await cleanup_and_exit(client)
                    except Exception as e:
                        log_error(f"Keyboard Task Err: {str(e)}")
                        await asyncio.sleep(1)
            
            asyncio.create_task(handle_keys())

            while True:
                try:
                    dashboard = await generate_dashboard_async(client)
                    live.update(dashboard)
                    await asyncio.sleep(0.5) # Smoother UI updates
                except Exception as e:
                    log_error(f"UI Update Err: {str(e)}")
                    await asyncio.sleep(1) # Smoother UI updates

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt:
        # For KeyboardInterrupt, we'd need a way to run async cleanup.
        # Given the sync nature of os._exit, we'll suggest users use 'x' for safe exit.
        print("\nForce Exiting...")
        os._exit(0)
