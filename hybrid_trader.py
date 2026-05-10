import os
import sys
import time
import httpx
import asyncio
import questionary
import select
import tty
import termios
from rich.console import Console

# Disable CPR warning and potential hangs in certain terminals
os.environ["PROMPT_TOOLKIT_NO_CPR"] = "1"

from rich.live import Live
from rich.panel import Panel
from rich.align import Align

from utils.config import API_KEY, API_SECRET, API_URL, MAX_POSITIONS, USE_BTC_FILTER, DAILY_LOSS_LIMIT_PCT, DAILY_PROFIT_TARGET_PCT, MIN_VOLUME_FILTER, CONSEC_LOSS_COOLDOWN_SEC, MAX_CONSEC_LOSSES, DB_SAVE_INTERVAL_SEC
from utils.state import bot_state, market_data
from engine.websocket import ws_manager
from engine.api import binance_request, get_balance_async, get_market_depth_data
from engine.trading import close_position_async, manage_active_positions, open_position_async, manage_limit_orders
from ui.dashboard import generate_dashboard_async
from strategies.hybrid import get_btc_trend, analyze_hybrid_async
from utils.database import init_db, load_state_from_db, save_state_to_db
from utils.intelligence import calculate_market_volatility, is_correlated_exposure
from coin_screener import screen_coins
from engine.multi_exchange import bybit_feed, okx_feed, aggregate_flow
from engine.auto_optimizer import auto_optimizer
from engine.depth_predictor import depth_predictor
from engine.sentiment import sentiment_filter
from utils.logger import init_logger, log_error
from utils.telegram import send_telegram, alert_kill_switch, alert_circuit_breaker, alert_sentiment_pause, alert_startup, alert_shutdown, alert_error, alert_daily_summary, alert_cooldown, command_loop

console = Console()
init_db()
load_state_from_db()

if not API_KEY or not API_SECRET or API_KEY == "YOUR_API_KEY":
    console.print("[bold red]ERROR: API_KEY or API_SECRET missing in .env file![/]")
    exit(1)

shutdown_event = asyncio.Event()

async def trading_loop(client):
    """Deeply optimized loop to prevent Rate Limits."""
    # Initial setup
    await get_balance_async(client)
    asyncio.create_task(alert_startup(bot_state.get("balance", 0), len(bot_state.get("active_positions", []))))
    last_ticker_refresh = 0
    loop_count = 0
    
    # --- EAGER ML PRE-TRAINING ---
    # Fetch tickers first, then pre-train ML on top coins immediately
    from engine.ml_engine import ml_predictor
    await get_btc_trend(client)  # Immediate BTC trend detection
    ticker_res = await client.get(f"{API_URL}/fapi/v1/ticker/24hr")
    if ticker_res.status_code == 200:
        tkr_data = ticker_res.json()
        tkr = [{"s": t["symbol"], "q": float(t["quoteVolume"]), "c": float(t["lastPrice"]), "o": float(t["openPrice"])} for t in tkr_data if t["symbol"].endswith("USDT")]
        market_data.tickers = tkr
        for t in tkr: market_data.prices[t["s"]] = t["c"]
        last_ticker_refresh = time.time()
        # Pre-train top 15 coins by volume
        top_by_vol = sorted(tkr, key=lambda x: x["q"], reverse=True)[:15]
        pretrain_symbols = [t["s"] for t in top_by_vol]
        asyncio.create_task(ml_predictor.batch_pretrain(client, pretrain_symbols))
        bot_state["last_log"] = f"[bold cyan]ML Pre-training {len(pretrain_symbols)} models...[/]"
    
    while not shutdown_event.is_set():
        try:
            now = time.time()
            loop_count += 1
            
            # --- DAILY RESET (00:00 UTC) ---
            from datetime import datetime, timezone
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if bot_state.get("_last_day") != today_str:
                # New day: reset start_balance, wins/losses, and re-enable trading
                current_bal = bot_state.get("balance", 0)
                if current_bal > 0:
                    bot_state["start_balance"] = current_bal
                bot_state["wins"] = 0
                bot_state["losses"] = 0
                bot_state["_last_day"] = today_str
                if bot_state.get("is_passive") and not bot_state.get("_manual_passive"):
                    bot_state["is_passive"] = False  # Re-enable if kill-switch paused yesterday
                bot_state["last_log"] = f"[bold cyan]NEW DAY: Start balance reset to ${current_bal:.2f}[/]"
            
            # --- KILL-SWITCH HARIAN (C1) ---
            if bot_state.get("start_balance", 0) > 0:
                daily_pnl_pct = ((bot_state.get("balance", 0) - bot_state["start_balance"]) / bot_state["start_balance"]) * 100
                if daily_pnl_pct <= -(DAILY_LOSS_LIMIT_PCT * 100) or daily_pnl_pct >= (DAILY_PROFIT_TARGET_PCT * 100):
                    if not bot_state.get("is_passive", False):
                        bot_state["is_passive"] = True
                        bot_state["last_log"] = f"[bold red]KILL-SWITCH ACTIVATED (PnL: {daily_pnl_pct:.2f}%)![/]"
                        asyncio.create_task(alert_kill_switch(daily_pnl_pct, bot_state.get('balance', 0)))
                        # Clean up positions and orders
                        active_pos = bot_state.get("active_positions", [])
                        for p in active_pos:
                            await close_position_async(client, p['symbol'], "LONG" if float(p['positionAmt'])>0 else "SHORT", float(p['positionAmt']), "KILL-SWITCH")
                        for sym in list(bot_state.get("limit_orders", {}).keys()):
                            await binance_request(client, 'DELETE', '/fapi/v1/allOpenOrders', {"symbol": sym})
                        bot_state["limit_orders"] = {}
                        
            # 1. Slow Cycle: BTC Trend and Full Ticker Refresh
            is_ws_stale = (now - bot_state.get("ws_last_msg", 0)) > 30
            
            # Periodically sync balance (every ~7.5 seconds) to ensure real-time UI match
            if loop_count % 5 == 0:
                await get_balance_async(client)

            # Throttle the heavy ticker endpoint to at most once per 60 seconds
            if loop_count % 40 == 0 or not market_data.tickers or (is_ws_stale and now - last_ticker_refresh > 60):
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

            # 3. Smart Coin Selection (Multi-Factor Screener)
            filtered = [t for t in market_data.tickers if t["q"] > MIN_VOLUME_FILTER]
            top_symbols = screen_coins(filtered, top_n=30)
            
            # TURBO UI Placeholder
            placeholder_results = []
            for sym in top_symbols[:20]:
                k = market_data.klines.get(sym, {})
                status = "SYNC (1m)" if "1m" not in k else "READY"
                price_val = market_data.prices.get(sym, 0)
                placeholder_results.append({
                    "sym": sym.replace("USDT", ""), "price": f"{price_val:,.4f}",
                    "sig": status, "score": 0, "dir": 0,
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

            # --- NEW: Auto-Optimizer, Sentiment, Depth Predictor, Clustering ---
            if loop_count % 200 == 0:
                await auto_optimizer.maybe_run(client)
            if loop_count % 40 == 0:
                from utils.intelligence import dynamic_clusterer
                dynamic_clusterer.maybe_recluster()
            # Sentiment check (background, non-blocking)
            if sentiment_filter.should_pause():
                if not bot_state.get("is_passive", False):
                    asyncio.create_task(alert_sentiment_pause())
                bot_state["is_passive"] = True
                bot_state["last_log"] = "[bold red]SENTIMENT PAUSE: High-impact event detected[/]"

            # Auto-reload DB weights periodically
            if loop_count % 100 == 0:
                await asyncio.to_thread(load_state_from_db)

            # 5. Analysis Phase (Concurrent and Smart - Parallel)
            top_20 = top_symbols[:20]  # Analisa paralel 20 koin
            
            # Scan top 20 plus any active positions and limit orders in parallel
            scan_targets = list(set(top_20 + [p['symbol'] for p in active_pos] + list(bot_state.get("limit_orders", {}).keys())))
                
            market_data.current_scan_list = scan_targets
            
            # Depth predictor observation (label walls for training)
            for s in scan_targets[:10]:
                depth_predictor.observe_and_label(s)
            
            # Aggressive ML training: train any scan target that has no model yet
            if loop_count % 10 == 1:
                untrained = [s for s in scan_targets if s not in ml_predictor.models and s not in ml_predictor._retraining]
                if untrained:
                    asyncio.create_task(ml_predictor.batch_pretrain(client, untrained[:6]))
            
            # Fetch institutional data (OI & Funding) periodically (Setiap ~30 detik untuk menghemat rate limit)
            if loop_count % 20 == 0:
                await get_market_depth_data(client, scan_targets)
            
            # Concurrently analyze ALL targeted symbols in parallel (no batching delay)
            tasks = [analyze_hybrid_async(client, s) for s in scan_targets]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            all_results = list(zip(scan_targets, batch_results))
            
            # Merge results into state
            current_results_map = {r["sym"]: r for r in bot_state.get("last_scan_results", []) if r.get("regime") != "SCANNING"}
            
            for sym, res in all_results:
                if isinstance(res, Exception):
                    log_error(f"Concurrent Analysis Error: {str(res)}")
                elif res:
                    current_results_map[res["sym"]] = res
            
            # Rebuild list for UI
            analysis_results = []
            for t in top_20:
                sym_short = t.replace("USDT", "")
                if sym_short in current_results_map:
                    analysis_results.append(current_results_map[sym_short])
                else:
                    # Add placeholder for unscanned top symbols
                    price_val = market_data.prices.get(t, 0)
                    k = market_data.klines.get(t, {})
                    status = "SYNC" if "1m" not in k else "WAIT"
                    analysis_results.append({
                        "sym": sym_short, "price": f"{price_val:,.4f}",
                        "sig": status, "score": 0, "dir": 0,
                        "struct": "WAIT", "regime": "SCANNING"
                    })
                    
            for s in scan_targets:
                sym_short = s.replace("USDT", "")
                if sym_short not in [r["sym"] for r in analysis_results] and sym_short in current_results_map:
                    analysis_results.append(current_results_map[sym_short])
            
            if analysis_results:
                bot_state["last_scan_results"] = sorted(analysis_results, key=lambda x: x.get('score', -1), reverse=True)
            
            # 6. Execution Phase
            if not bot_state.get("is_passive", False):
                for s in analysis_results:
                    if "WAIT" in s["sig"] or s.get("ai") is None: continue
                    sym_full = s["sym"] + "USDT"
                    
                    # PREVENT DUPLICATES: Skip if already in active position OR already has a pending limit order
                    if any(p['symbol'] == sym_full for p in active_pos): continue
                    
                    # --- BLACKLIST & COOLDOWN FILTER (C2 & C3) ---
                    if bot_state.get("blacklist", {}).get(sym_full, 0) > time.time(): continue
                    
                    sym_perf = bot_state.get("sym_perf", {}).get(sym_full, {})
                    if sym_perf.get("c", 0) >= MAX_CONSEC_LOSSES:
                        last_loss = sym_perf.get("last_loss_time", 0)
                        if time.time() - last_loss < CONSEC_LOSS_COOLDOWN_SEC:
                            bot_state.setdefault("blacklist", {})[sym_full] = time.time() + CONSEC_LOSS_COOLDOWN_SEC
                            bot_state["sym_perf"][sym_full]["c"] = 0 # Reset biar tidak infinite loop
                            asyncio.create_task(alert_cooldown(sym_full, MAX_CONSEC_LOSSES))
                            continue
                            
                    is_replacement = sym_full in bot_state.get("limit_orders", {})
                    # If already has a limit order and it's NOT a replacement (same direction), skip it
                    if is_replacement:
                        old_lo = bot_state["limit_orders"][sym_full]
                        sig_side = "LONG" if s["dir"] == 1 else "SHORT"
                        old_side = "LONG" if old_lo["side"] == "BUY" else "SHORT"
                        
                        # If same side, only continue if BOTH are limit orders. 
                        # If the new signal is MARKET, we want to replace the old limit order.
                        new_is_market = s.get("ai", {}).get("is_market", False)
                        if sig_side == old_side and not new_is_market: continue
                    
                    sig_side = "LONG" if s["dir"] == 1 else "SHORT"
                    if is_correlated_exposure(sym_full, sig_side): continue
                    
                    can_open = False
                    if not USE_BTC_FILTER or bot_state["btc_dir"] == 0: can_open = True
                    elif sig_side == "LONG" and bot_state["btc_dir"] == 1: can_open = True
                    elif sig_side == "SHORT" and bot_state["btc_dir"] == -1: can_open = True
                    
                    if can_open:
                        current_slots = len(active_pos) + len(bot_state.get("limit_orders", {}))
                        
                        if current_slots < MAX_POSITIONS or is_replacement:
                            if is_replacement:
                                old_lo = bot_state["limit_orders"][sym_full]
                                try:
                                    await binance_request(client, 'DELETE', '/fapi/v1/order', {"symbol": sym_full, "orderId": old_lo["orderId"]})
                                    # Also delete associated SL/TP algo orders
                                    await binance_request(client, 'DELETE', '/fapi/v1/allOpenOrders', {"symbol": sym_full})
                                    del bot_state["limit_orders"][sym_full]
                                except Exception as e:
                                    log_error(f"Order replacement cancel failed for {sym_full}: {e}", include_traceback=False)
                            
                            await open_position_async(client, sym_full, "BUY" if sig_side == "LONG" else "SELL", s["sig"], s["ai"])

            bot_state["heartbeat"] += 1
            
            # --- API CIRCUIT BREAKER (M5) & BATCHED DB WRITES (M3) ---
            if now - bot_state.get("last_db_save", 0) > DB_SAVE_INTERVAL_SEC:
                # 1. API Circuit Breaker Eval
                reqs = bot_state.get("api_req_count", 0)
                errs = bot_state.get("api_err_count", 0)
                if reqs > 10:
                    err_rate = errs / reqs
                    if err_rate > 0.3: # >30% error rate
                        bot_state["api_health_status"] = "BLOCKED"
                        bot_state["last_log"] = f"[bold white on red]API CIRCUIT BREAKER! Rate: {err_rate*100:.1f}%[/]"
                        asyncio.create_task(alert_circuit_breaker(err_rate))
                    elif err_rate > 0.1:
                        bot_state["api_health_status"] = "DEGRADED"
                    else:
                        bot_state["api_health_status"] = "OK"
                        
                bot_state["api_req_count"] = 0
                bot_state["api_err_count"] = 0
                
                # Auto-recovery after 60s of silence
                if bot_state.get("api_health_status") == "BLOCKED" and errs == 0:
                     bot_state["api_health_status"] = "OK"
                     
                # 2. Batched DB Writes
                await asyncio.to_thread(save_state_to_db)
                bot_state["last_db_save"] = now

            # 3. Hourly Summary to Telegram
            if loop_count % 2400 == 0:  # ~every hour (2400 loops * ~1.5s avg)
                balance = bot_state.get("balance", 0)
                start = bot_state.get("start_balance", 0)
                pnl = balance - start if start > 0 else 0
                asyncio.create_task(alert_daily_summary(balance, pnl, bot_state.get("wins", 0), bot_state.get("losses", 0)))

            await asyncio.sleep(max(0.05, min(0.5, 0.3 / max(bot_state.get("market_vol", 1.0), 0.5))))  # Ultra-fast: 50ms-500ms adaptive
        except Exception as e:
            bot_state["last_log"] = f"[red]Loop Err: {str(e)[:40]}[/]"
            asyncio.create_task(alert_error(str(e)))
            await asyncio.sleep(5)


async def cleanup_and_exit(client):
    """Clean up all pending orders and signal shutdown."""
    bot_state["last_log"] = "[bold red]EXITING: Cleaning up...[/]"
    limit_orders = list(bot_state.get("limit_orders", {}).keys())
    for sym in limit_orders:
        try: await binance_request(client, 'DELETE', '/fapi/v1/allOpenOrders', {"symbol": sym})
        except Exception:
            pass
    ws_manager.running = False
    save_state_to_db()
    await alert_shutdown()
    shutdown_event.set()

async def main():
    init_logger()
    console.clear()
    async with httpx.AsyncClient(timeout=30.0) as client:
        asyncio.create_task(ws_manager.start(client))
        asyncio.create_task(trading_loop(client))
        asyncio.create_task(bybit_feed.start())
        asyncio.create_task(okx_feed.start())
        asyncio.create_task(sentiment_filter.run_loop(client))
        
        # Telegram command handler
        tg_action_queue = asyncio.Queue()
        asyncio.create_task(command_loop(bot_state, market_data, tg_action_queue))

        async def process_tg_actions():
            while not shutdown_event.is_set():
                try:
                    action = await asyncio.wait_for(tg_action_queue.get(), timeout=1)
                    if action == "CLOSE_ALL":
                        res = await binance_request(client, 'GET', '/fapi/v2/positionRisk')
                        if res and res.status_code == 200:
                            for p in [x for x in res.json() if float(x['positionAmt']) != 0]:
                                await close_position_async(client, p['symbol'], "LONG" if float(p['positionAmt'])>0 else "SHORT", float(p['positionAmt']), "TELEGRAM")
                    elif action == "CANCEL_ORDERS":
                        for sym in list(bot_state.get("limit_orders", {}).keys()):
                            await binance_request(client, 'DELETE', '/fapi/v1/allOpenOrders', {"symbol": sym})
                        bot_state["limit_orders"] = {}
                    elif action == "SHUTDOWN":
                        await cleanup_and_exit(client)
                    elif action.startswith("CLOSE:"):
                        symbol = action.split(":", 1)[1]
                        pos = next((p for p in bot_state.get("active_positions", []) if p['symbol'] == symbol), None)
                        if pos:
                            await close_position_async(client, pos['symbol'], "LONG" if float(pos['positionAmt'])>0 else "SHORT", float(pos['positionAmt']), "TELEGRAM")
                        else:
                            await send_telegram(f"❌ No position found for {symbol}")
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    await asyncio.sleep(1)

        asyncio.create_task(process_tg_actions())
        
        ui_queue = asyncio.Queue()

        with Live(None, refresh_per_second=4, screen=True) as live:
            async def handle_keys():
                def get_char():
                    if not os.isatty(sys.stdin.fileno()):
                        return None
                    old_settings = termios.tcgetattr(sys.stdin)
                    try:
                        tty.setcbreak(sys.stdin.fileno())
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            return sys.stdin.read(1)
                    except (OSError, termios.error):
                        return None
                    finally:
                        termios.tcsetattr(sys.stdin, termios.TCSANOW, old_settings)
                    return None

                while not shutdown_event.is_set():
                    try:
                        if bot_state.get("ui_active", False):
                            await asyncio.sleep(0.2)
                            continue

                        char = await asyncio.get_event_loop().run_in_executor(None, get_char)
                        if not char:
                            continue
                        
                        k = char.lower()
                        if k == 'c':
                            res = await binance_request(client, 'GET', '/fapi/v2/positionRisk')
                            if res and res.status_code == 200:
                                for p in [x for x in res.json() if float(x['positionAmt']) != 0]:
                                    await close_position_async(client, p['symbol'], "LONG" if float(p['positionAmt'])>0 else "SHORT", float(p['positionAmt']), "MANUAL")
                            bot_state["last_log"] = "[bold red]Closing all positions...[/]"
                        elif k == '\t' or k == 'm': # Tab or 'm' for Menu
                            await ui_queue.put("CLOSE_SELECT")
                        elif k == 'k':
                            for sym in list(bot_state.get("limit_orders", {}).keys()):
                                await binance_request(client, 'DELETE', '/fapi/v1/allOpenOrders', {"symbol": sym})
                            bot_state["limit_orders"] = {}
                            bot_state["last_log"] = "[bold cyan]All limit orders cancelled.[/]"
                        elif k == 'p':
                            bot_state["is_passive"] = not bot_state.get("is_passive", False)
                            status = "PASSIVE" if bot_state["is_passive"] else "ACTIVE"
                            bot_state["last_log"] = f"[bold]Bot mode set to {status}[/]"
                        elif k == 'r':
                            load_state_from_db()
                            bot_state["last_log"] = "[bold green]Intelligence Weights Reloaded![/]"
                        elif k == 'x' or k == 'q': await cleanup_and_exit(client)
                    except Exception as e:
                        bot_state["last_log"] = f"[red]Key Error: {str(e)[:40]}[/]"
                        await asyncio.sleep(1)
            
            asyncio.create_task(handle_keys())
            while not shutdown_event.is_set():
                try:
                    if not ui_queue.empty():
                        action = await ui_queue.get()
                        if action == "CLOSE_SELECT":
                            bot_state["ui_active"] = True
                            live.stop()
                            try:
                                await asyncio.sleep(0.2)
                                active_pos = bot_state.get("active_positions", [])
                                if not active_pos:
                                    console.print("[yellow]No active positions to close.[/]")
                                    await asyncio.sleep(1)
                                else:
                                    choices = [f"{p['symbol']} ({'LONG' if float(p['positionAmt'])>0 else 'SHORT'})" for p in active_pos]
                                    choices.append("Cancel")
                                    selected = await questionary.select("Select coin to close:", choices=choices).ask_async()
                                    if selected and selected != "Cancel":
                                        symbol = selected.split(" ")[0]
                                        p = next(x for x in active_pos if x['symbol'] == symbol)
                                        await close_position_async(client, p['symbol'], "LONG" if float(p['positionAmt'])>0 else "SHORT", float(p['positionAmt']), "MANUAL")
                                        bot_state["last_log"] = f"[bold yellow]Manual Close: {p['symbol']}[/]"
                            finally:
                                bot_state["ui_active"] = False
                                live.start()
                        ui_queue.task_done()

                    dashboard = await generate_dashboard_async(client)
                    live.update(dashboard)
                    await asyncio.sleep(0.25)
                except Exception as e:
                    bot_state["ui_active"] = False
                    try: live.start()
                    except Exception: pass
                    await asyncio.sleep(1)

if __name__ == "__main__":
    try: asyncio.run(main())
    except (KeyboardInterrupt, SystemExit, RuntimeError):
        pass
    finally:
        from utils.database import save_state_to_db
        save_state_to_db()
        sys.exit(0)
