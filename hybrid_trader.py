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
    """Deeply optimized loop to prevent Rate Limits."""
    # Initial setup
    await get_balance_async(client)
    last_ticker_refresh = 0
    loop_count = 0
    
    while True:
        try:
            now = time.time()
            loop_count += 1
            
            # 1. Slow Cycle: BTC Trend and Full Ticker Refresh (Every 5 mins)
            is_ws_stale = (now - bot_state.get("ws_last_msg", 0)) > 30
            
            # Periodically sync balance (every ~7.5 seconds) to ensure real-time UI match
            if loop_count % 5 == 0:
                await get_balance_async(client)

            if loop_count % 300 == 0 or not market_data.tickers or is_ws_stale:
                await get_btc_trend(client)
                ticker_res = await client.get(f"{API_URL}/fapi/v1/ticker/24hr")
                if ticker_res.status_code == 200:
                    last_ticker_refresh = now
                    tkr_data = ticker_res.json()
                    tkr = []
                    for t in tkr_data:
                        symbol = t["symbol"]
                        if not symbol.endswith("USDT"): continue
                        tkr.append({
                            "s": symbol, "q": float(t["quoteVolume"]), 
                            "c": float(t["lastPrice"]), "o": float(t["openPrice"])
                        })
                        market_data.prices[symbol] = float(t["lastPrice"])
                    market_data.tickers = tkr

            if not market_data.tickers:
                bot_state["last_log"] = "[yellow]Initial Sync...[/]"
                await asyncio.sleep(2); continue

            # 2. Process Prices (Fast, mostly from memory/WS)
            for t in market_data.tickers:
                if t["s"] in market_data.prices: t["c"] = market_data.prices[t["s"]]
                if t.get("o") and t["o"] > 0: t["cp"] = ((t["c"] - t["o"]) / t["o"]) * 100
                else: t["cp"] = 0

            # 3. Top Movers Selection
            filtered = [t for t in market_data.tickers if t["q"] > 5_000_000]
            top_movers = sorted(filtered, key=lambda x: abs(x["cp"]), reverse=True)[:10]
            top_symbols = [t["s"] for t in top_movers]
            
            # TURBO UI Placeholder
            placeholder_results = []
            for t in top_movers[:5]: # Display top 5 in UI
                sym_full = t["s"]
                k = market_data.klines.get(sym_full, {})
                status = "SYNC (1m)" if "1m" not in k else "READY"
                placeholder_results.append({
                    "sym": t["s"].replace("USDT", ""), "price": f"{t['c']:,.4f}",
                    "sig": status, "score": 0, "dir": 1 if t["cp"] > 0 else -1,
                    "struct": "WAIT", "regime": "SCANNING"
                })
            if not bot_state.get("last_scan_results"): bot_state["last_scan_results"] = placeholder_results

            await ws_manager.update_subscriptions(top_symbols)
            
            # 4. Critical Management (Once per loop)
            # Use real analysis results from previous cycle for management
            all_valid = bot_state.get("last_scan_results", [])
            active_pos = await manage_active_positions(client, all_valid)
            await manage_limit_orders(client, all_valid)
            bot_state["active_positions"] = active_pos
            
            # Intelligence Metrics
            calculate_market_volatility()
            net_bias = sum([1 if float(p['positionAmt']) > 0 else -1 for p in active_pos])
            bot_state["directional_bias"] = net_bias

            # 5. Analysis Phase (Concurrent and Smart)
            final_symbols = list(set(top_symbols[:5] + [p['symbol'] for p in active_pos] + list(bot_state.get("limit_orders", {}).keys())))
            market_data.current_scan_list = final_symbols
            
            # Fetch institutional data (OI & Funding) periodically
            if loop_count % 5 == 0:
                await get_market_depth_data(client, final_symbols)
            
            # Concurrently analyze symbols
            tasks = [analyze_hybrid_async(client, s) for s in final_symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            analysis_results = []
            for res in results:
                if isinstance(res, Exception):
                    log_error(f"Concurrent Analysis Error: {str(res)}")
                elif res:
                    analysis_results.append(res)
            
            if analysis_results:
                # Merge and Sort: Add placeholder only if the symbol isn't already successfully analyzed
                existing_syms = [r["sym"] for r in analysis_results]
                for p in placeholder_results:
                    if p["sym"] not in existing_syms: 
                        analysis_results.append(p)
                bot_state["last_scan_results"] = sorted(analysis_results, key=lambda x: x.get('score', -1), reverse=True)
            
            # 6. Execution Phase
            if not bot_state.get("is_passive", False):
                for s in analysis_results:
                    if "WAIT" in s["sig"] or s.get("ai") is None: continue
                    sym_full = s["sym"] + "USDT"
                    if any(p['symbol'] == sym_full for p in active_pos): continue
                    
                    sig_side = "LONG" if s["dir"] == 1 else "SHORT"
                    if is_correlated_exposure(sym_full, sig_side): continue
                    
                    can_open = False
                    if not USE_BTC_FILTER or bot_state["btc_dir"] == 0: can_open = True
                    elif sig_side == "LONG" and bot_state["btc_dir"] == 1: can_open = True
                    elif sig_side == "SHORT" and bot_state["btc_dir"] == -1: can_open = True
                    
                    if can_open:
                        is_replacement = sym_full in bot_state.get("limit_orders", {})
                        current_slots = len(active_pos) + len(bot_state.get("limit_orders", {}))
                        
                        if current_slots < MAX_POSITIONS or is_replacement:
                            if is_replacement:
                                old_lo = bot_state["limit_orders"][sym_full]
                                try:
                                    await binance_request(client, 'DELETE', '/fapi/v1/order', {"symbol": sym_full, "orderId": old_lo["orderId"]})
                                    # Also delete associated SL/TP algo orders
                                    await binance_request(client, 'DELETE', '/fapi/v1/allOpenOrders', {"symbol": sym_full})
                                    del bot_state["limit_orders"][sym_full]
                                except: pass
                            
                            await open_position_async(client, sym_full, "BUY" if sig_side == "LONG" else "SELL", s["sig"], s["ai"])

            bot_state["heartbeat"] += 1
            await asyncio.to_thread(save_state_to_db)
            await asyncio.sleep(1.5) # Relaxed loop
        except Exception as e:
            bot_state["last_log"] = f"[red]Loop Err: {str(e)[:40]}[/]"
            await asyncio.sleep(5)

async def cleanup_and_exit(client):
    """Clean up all pending orders before exiting."""
    bot_state["last_log"] = "[bold red]EXITING: Cleaning up...[/]"
    limit_orders = list(bot_state.get("limit_orders", {}).keys())
    for sym in limit_orders:
        try: await binance_request(client, 'DELETE', '/fapi/v1/allOpenOrders', {"symbol": sym})
        except: pass
    os._exit(0)

async def main():
    init_logger()
    console.clear()
    async with httpx.AsyncClient(timeout=30.0) as client:
        asyncio.create_task(ws_manager.start(client))
        asyncio.create_task(trading_loop(client))
        
        with Live(None, refresh_per_second=2, screen=False) as live:
            async def handle_keys():
                while True:
                    try:
                        key = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                        k = key.strip().lower()
                        if k == 'c':
                            res = await binance_request(client, 'GET', '/fapi/v2/positionRisk')
                            if res and res.status_code == 200:
                                for p in [x for x in res.json() if float(x['positionAmt']) != 0]:
                                    await close_position_async(client, p['symbol'], "LONG" if float(p['positionAmt'])>0 else "SHORT", float(p['positionAmt']), "MANUAL")
                        elif k == 'k':
                            for sym in list(bot_state.get("limit_orders", {}).keys()):
                                await binance_request(client, 'DELETE', '/fapi/v1/allOpenOrders', {"symbol": sym})
                            bot_state["limit_orders"] = {}
                        elif k == 'p':
                            bot_state["is_passive"] = not bot_state.get("is_passive", False)
                        elif k == 'x': await cleanup_and_exit(client)
                    except: await asyncio.sleep(1)
            
            asyncio.create_task(handle_keys())
            while True:
                try:
                    dashboard = await generate_dashboard_async(client)
                    live.update(dashboard)
                    await asyncio.sleep(0.5)
                except: await asyncio.sleep(1)

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: os._exit(0)
