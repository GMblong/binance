import time
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.align import Align
from rich.padding import Padding
from rich import box

from utils.config import USE_BTC_FILTER, MAX_POSITIONS, API_URL, DAILY_PROFIT_TARGET_PCT
from utils.state import bot_state, market_data
from utils.intelligence import get_current_session, calculate_kelly_risk
from engine.api import get_balance_async, binance_request, get_market_depth_data
from engine.ml_engine import ml_predictor
from engine.websocket import ws_manager
from engine.trading import manage_active_positions, open_position_async
from strategies.hybrid import get_btc_trend, analyze_hybrid_async
from strategies.analyzer import MarketAnalyzer
from engine.multi_exchange import bybit_feed, okx_feed

ticker_cache = {} # {symbol: last_data}

# Dashboard render cache - avoid recalculating static elements
_vp_cache = {}  # {symbol: (last_ot, vp_result)}
_last_layout_cache = {"ts": 0, "layout": None}

def generate_sparkline(prices):
    if not prices or len(prices) < 2: return ""
    chars = "  ▂▃▄▅▆▇█"
    min_p, max_p = min(prices), max(prices)
    rng = max_p - min_p
    if rng == 0: return chars[4] * len(prices)
    res = ""
    for p in prices:
        idx = int(((p - min_p) / rng) * (len(chars) - 1))
        res += chars[idx]
    return res

async def generate_dashboard_async(client):
    try:
        # Use results from the background trading loop
        all_valid = bot_state.get("last_scan_results", [])
        
        # --- SMART FLOW BREADTH (Risk-On/Off) ---
        sector_status = "[dim]NORMAL[/]"
        if all_valid:
            bull_count = sum(1 for r in all_valid if r["dir"] == 1 and r.get("score", 0) >= 70)
            bear_count = sum(1 for r in all_valid if r["dir"] == -1 and r.get("score", 0) >= 70)
            bot_state["alt_breadth"] = (bull_count / len(all_valid)) * 100
            
            if bull_count >= 4: sector_status = "[bold green]BULL PUMP[/]"
            elif bear_count >= 4: sector_status = "[bold red]BEAR DUMP[/]"
        
        # Use cached positions from trading_loop (stored in bot_state)
        active_pos = bot_state.get("active_positions", [])

        def create_combined_table(title, data):
            table = Table(title=f" {title} ", expand=True, box=box.ROUNDED, border_style="cyan", header_style="bold cyan", title_style="bold cyan", title_justify="left")
            table.add_column("SYMBOL", justify="left", style="bold white")
            table.add_column("TREND", justify="center")
            table.add_column("ML PROB", justify="center")
            table.add_column("POC DIST", justify="center")
            table.add_column("SCORE", justify="center")
            table.add_column("REGIME", justify="center")
            table.add_column("SIGNAL", justify="right")
            
            if not data:
                table.add_row("", "", "", "[dim italic]Scanning market data...[/]", "", "", "")
                return table
            
            for r in data: 
                score_val = int(r.get("score", 0))
                bar_length = score_val // 10
                
                # Trend Direction Indicator
                is_bull = r.get("dir", 0) == 1
                trend_icon = "▲" if is_bull else ("▼" if r.get("dir", 0) == -1 else "—")
                trend_col = "bold green" if is_bull else ("bold red" if r.get("dir", 0) == -1 else "dim white")
                
                # ML Probability with consensus indicator
                ml_prob = r.get("ai", {}).get("ml_prob", 0.5)
                ml_col = "bold green" if ml_prob > 0.65 else ("bold red" if ml_prob < 0.35 else "white")
                # Consensus: if prob is far from 0.5, models likely agree
                consensus = "◆" if (ml_prob > 0.7 or ml_prob < 0.3) else ("◇" if (ml_prob > 0.6 or ml_prob < 0.4) else " ")
                ml_str = f"[{ml_col}]{ml_prob:.2f}{consensus}[/]"
                
                # POC Distance (cached)
                poc_dist_str = "[dim]-[/]"
                sym_full = r['sym'] + "USDT"
                if sym_full in market_data.klines and "15m" in market_data.klines[sym_full]:
                    d15 = market_data.klines[sym_full]["15m"]
                    vp_key = float(d15.iloc[-1]['ot']) if not d15.empty else 0
                    cached_vp = _vp_cache.get(sym_full)
                    if cached_vp and cached_vp[0] == vp_key:
                        vp = cached_vp[1]
                    else:
                        vp = MarketAnalyzer.get_volume_profile(d15)
                        _vp_cache[sym_full] = (vp_key, vp)
                    if vp:
                        curr_p = float(r['price'].replace(',', ''))
                        dist = (curr_p - vp['poc']) / vp['poc'] * 100
                        dist_col = "green" if dist > 0 else "red"
                        poc_dist_str = f"[{dist_col}]{dist:+.2f}%[/]"

                if score_val >= 80:
                    score_col = "bold bright_green" if is_bull else "bold bright_red"
                    bar_char = "█"
                    bg_char = "░"
                    bg_col = "green" if is_bull else "red"
                elif score_val >= 60:
                    score_col = "bold yellow"
                    bar_char = "▓"
                    bg_char = "░"
                    bg_col = "yellow"
                else:
                    score_col = "bold white"
                    bar_char = "▒"
                    bg_char = " "
                    bg_col = "white"

                score_bar = f"[{score_col}]{bar_char * bar_length}[/][dim {bg_col}]{bg_char * (10 - bar_length)}[/]"
                
                sig = r['sig']
                curr_price_live = market_data.prices.get(sym_full, r.get('price', '0.00'))
                
                if "SCALP" in sig or "SQZ" in sig: sig_style = "bold blink yellow"
                elif "INTRA" in sig: sig_style = "bold blink cyan"
                else: sig_style = "dim white"
                
                table.add_row(
                    r["sym"], 
                    f"[{trend_col}]{trend_icon}[/]",
                    ml_str,
                    poc_dist_str,
                    f"{score_bar} [bold]{score_val}%[/]", 
                    f"[{'cyan' if r.get('regime','')=='TRENDING' else ('yellow' if r.get('regime','')=='RANGING' else ('red' if r.get('regime','')=='VOLATILE' else 'dim'))}]{r.get('regime', '...')[:5]}[/]", 
                    f"[{sig_style}]{sig}[/]\n[dim]{curr_price_live}[/]"
                )
            return table

        # --- AI INSIGHT GENERATOR ---
        insight_text = "[dim italic]Scanning for institutional footprints...[/]"
        best_pick = None
        if all_valid:
            valid_no_pos = [r for r in all_valid if not any(p['symbol'] == (r['sym']+"USDT") for p in active_pos)]
            if valid_no_pos:
                best_pick = sorted(valid_no_pos, key=lambda x: x['score'], reverse=True)[0]
                
                reasons = []
                if best_pick.get('sig') == "LIQ-HUNT": reasons.append("[bold red]LIQUIDATION TRAP DETECTED[/]")
                if best_pick.get('sig') == "ML-BO": reasons.append("[bold cyan]PREDICTIVE VOLATILITY EXPLOSION[/]")
                
                # CVD Insight
                cvd = best_pick.get('cvd', 0.5)
                if cvd > 0.65: reasons.append(f"[bold green]CVD BUYERS {cvd*100:.0f}%[/]")
                elif cvd < 0.35: reasons.append(f"[bold red]CVD SELLERS {(1-cvd)*100:.0f}%[/]")
                
                if best_pick.get('div'): reasons.append(f"[bold yellow]{best_pick['div']}[/]")
                if best_pick.get('sweep'): reasons.append(f"[bold cyan]{best_pick['sweep']}[/]")
                if best_pick.get('ob'): reasons.append("[bold magenta]ORDER BLOCK REACTION[/]")
                if best_pick.get('sync'): reasons.append("[bold green]PERFECT TF SYNC[/]")
                
                # Volume Profile & VSA Insights
                sym_full = best_pick['sym'] + "USDT"
                if sym_full in market_data.klines and "15m" in market_data.klines[sym_full]:
                    cached_vp = _vp_cache.get(sym_full)
                    vp = cached_vp[1] if cached_vp else None
                    if vp:
                        curr_p = float(best_pick['price'].replace(',', ''))
                        if curr_p > vp['poc']: reasons.append("[bold green]ABOVE POC[/]")
                        else: reasons.append("[bold red]BELOW POC[/]")
                    
                    if "1m" in market_data.klines[sym_full]:
                        vsa_sig = MarketAnalyzer.detect_vsa_signals(market_data.klines[sym_full]["1m"])
                        if vsa_sig != 0:
                            reasons.append(f"[bold yellow]VSA ({'BULL' if vsa_sig==1 else 'BEAR'})[/]")

                reason_str = " + ".join(reasons) if reasons else "STRONG MOMENTUM"
                regime_str = best_pick.get("regime", "SCANNING")
                insight_text = f"Top Pick: [bold white]{best_pick['sym']}[/] ({best_pick['score']}%). {reason_str}. Regime: [bold]{regime_str}[/]."

        # --- PORTFOLIO & PENDING ORDERS ---
        pos_table = Table(title=" 💼 ACTIVE POSITIONS ", expand=True, box=box.ROUNDED, header_style="bold magenta", border_style="magenta", title_style="bold yellow", title_justify="left")
        pos_table.add_column("SYMBOL"); pos_table.add_column("SIDE"); pos_table.add_column("KELLY", justify="center")
        pos_table.add_column("ENTRY"); pos_table.add_column("MARK")
        pos_table.add_column("TP/SL (BOT)", justify="center"); pos_table.add_column("PNL", justify="right")
        
        for p in active_pos:
            symbol = p['symbol']
            side = "LONG" if float(p['positionAmt']) > 0 else "SHORT"
            side_col = "green" if side == "LONG" else "red"
            pnl_raw = float(p['unRealizedProfit'])
            notional = abs(float(p['positionAmt'])) * float(p['entryPrice'])
            pnl = pnl_raw
            pnl_pct = (pnl / notional * 100) if notional > 0 else 0
            pnl_col = "bold white on green" if pnl > 0 else "bold white on red"
            
            tp_sl_str = "[dim]-[/]"
            kelly_str = "[dim]1.0x[/]"
            if symbol in bot_state["trades"]:
                ai_data = bot_state["trades"][symbol]
                tp_val = ai_data.get("tp", 0)
                sl_val = ai_data.get("sl", 0)
                tp_sl_str = f"[green]{tp_val:.1f}%[/]/[red]{sl_val:.1f}%[/]"
                
                # Dynamic Kelly Multiplier
                perf = ml_predictor.performance.get(symbol, [])
                wr = sum(perf) / len(perf) if len(perf) >= 5 else 0.5
                km = calculate_kelly_risk(symbol, win_rate=wr)
                kelly_str = f"[bold cyan]{km:.1f}x[/]"

            pos_table.add_row(
                f"[bold white]{symbol}[/]",
                f"[{side_col}]{side}[/]",
                kelly_str,
                f"{float(p['entryPrice']):,.4f}",
                f"{float(p['markPrice']):,.4f}",
                tp_sl_str,
                f"[{pnl_col}] {pnl:+.2f} USDT [/]"
            )

        # --- PENDING LIMIT ORDERS TABLE ---
        limit_orders = bot_state.get("limit_orders", {})
        limit_table = Table(title=" 🎯 PENDING LIMIT ORDERS (SNIPER) ", expand=True, box=box.ROUNDED, header_style="bold cyan", border_style="cyan", title_style="bold cyan", title_justify="left")
        limit_table.add_column("SYMBOL"); limit_table.add_column("SIDE")
        limit_table.add_column("MODE", justify="center")
        limit_table.add_column("MARKET PRICE", justify="right")
        limit_table.add_column("LIMIT PRICE", justify="right")
        limit_table.add_column("DISTANCE", justify="center")
        limit_table.add_column("SPEC (SL/TP)"); limit_table.add_column("SCORE")

        for symbol, lo in limit_orders.items():
            # Safety: Skip if already showing in active positions
            if any(p['symbol'] == symbol for p in active_pos):
                continue

            ai = lo.get("ai", {})
            side_col = "green" if lo['side'] == "BUY" else "red"
            side_str = "LONG" if lo['side'] == "BUY" else "SHORT"
            
            mode_str = ai.get("entry_mode", "PULL")
            mode_col = "bold magenta" if mode_str == "MOMENTUM" else "dim white"
            
            # Distance from current price
            curr_p = market_data.prices.get(symbol, lo['price'])
            dist = abs(curr_p - lo['price']) / curr_p * 100
            
            limit_table.add_row(
                f"[bold white]{symbol}[/]",
                f"[{side_col}]{side_str}[/]",
                f"[{mode_col}]{mode_str[:4]}[/]",
                f"{curr_p:,.4f}",
                f"[bold cyan]{lo['price']:,.4f}[/]",
                f"[yellow]{dist:+.2f}%[/]",
                f"[red]{ai.get('sl', 0):.1f}%[/]/[green]{ai.get('tp', 0):.1f}%[/]",
                f"[bold]{ai.get('score', 0)}%[/]"
            )

        if not active_pos and not limit_orders:
            pos_element = Panel(Align.center("[dim italic]No Active Positions or Pending Orders...[/]", vertical="middle"), 
                              title=" 💼 PORTFOLIO ", border_style="yellow", box=box.ROUNDED)
        else:
            # Combine tables in a layout or split
            pos_element = Layout()
            pos_element.split_column(
                Layout(Padding(pos_table, (0,0,1,0)), ratio=1 if active_pos else 0),
                Layout(limit_table, ratio=1 if limit_orders else 0)
            )
            # If one is empty, the other takes space
            if not active_pos: pos_element = limit_table
            elif not limit_orders: pos_element = pos_table

        # --- BTC STATUS 2.0 ---
        btc_status = bot_state.get("btc_state", "UNKNOWN")
        
        if btc_status == "BULLISH": btc_style = "bold green"
        elif btc_status == "BEARISH": btc_style = "bold red"
        elif btc_status == "DANGER": btc_style = "bold white on red blink"
        elif "TRAP" in btc_status: btc_style = "bold orange3"
        elif btc_status == "SQUEEZE": btc_style = "bold cyan"
        else: btc_style = "bold yellow"
        
        # Calculate real-time equity and PnL
        total_unrealized_pnl = sum(float(p['unRealizedProfit']) for p in active_pos)
        
        # Live Balance from bot_state (totalMarginBalance already includes unrealized PnL)
        curr_bal = bot_state.get("balance", 0.0)
        display_bal = curr_bal
        
        # Breadth
        br_val = bot_state["alt_breadth"]
        br_status = "RISK-ON" if br_val > 65 else ("RISK-OFF" if br_val < 35 else "NEUTRAL")

        # Daily PnL Styling
        daily_pnl = bot_state.get("daily_pnl", 0.0)
        display_pnl = daily_pnl + total_unrealized_pnl
        start_bal = bot_state.get("start_balance", curr_bal)
        target_amt = start_bal * DAILY_PROFIT_TARGET_PCT
        
        # UI Fallback for initial sync
        bal_display = f"{display_bal:,.2f}" if curr_bal > 0 else "SYNCING..."
        target_display = f"{target_amt:.2f}" if target_amt > 0 else "---"
        
        pnl_col = "bold bright_green" if display_pnl > 0 else ("bold bright_red" if display_pnl < 0 else "white")

        # Heartbeat logic
        hb = bot_state.get("heartbeat", 0)
        hb_char = "●" if hb % 2 == 0 else "○"
        hb_style = "bold green" if hb % 2 == 0 else "dim green"
        
        # WS Metrics
        ws_pkts = bot_state.get("ws_msg_count", 0)
        last_ws = time.time() - bot_state.get("ws_last_msg", 0)
        ws_status = f"[bold green]{ws_pkts}[/]" if last_ws < 2 else f"[bold red]{ws_pkts} (LAG)[/]"

        # Passive Mode Status
        is_p = bot_state.get("is_passive", False)
        mode_status = "[bold yellow]PASSIVE[/]" if is_p else "[bold green]ACTIVE[/]"

        # --- UI COMPONENTS ---
        session = get_current_session()
        sess_col = "cyan" if session == "LONDON" else ("green" if session == "NEW_YORK" else ("magenta" if session == "ASIA" else "yellow"))
        
        mv = bot_state.get("market_vol", 1.0)
        mv_col = "green" if mv < 1.0 else ("red" if mv > 1.5 else "yellow")
        
        bias = bot_state.get("directional_bias", 0)
        bias_str = f"+{bias}" if bias > 0 else str(bias)
        bias_col = "green" if bias > 0 else ("red" if bias < 0 else "white")
        
        pnl_col = "bold bright_green" if display_pnl > 0 else ("bold bright_red" if display_pnl < 0 else "white")

        # Cross-exchange status
        xch_parts = []
        if bybit_feed.connected: xch_parts.append("[green]BY✓[/]")
        else: xch_parts.append("[red]BY✗[/]")
        if okx_feed.connected: xch_parts.append("[green]OKX✓[/]")
        else: xch_parts.append("[red]OKX✗[/]")
        xch_str = "|".join(xch_parts)

        # Card 1: Main Status
        opt_status = "[bold green]ONLINE[/]" if (time.time() % 60 < 5) else "[dim green]STANDBY[/]"
        n_models = len(ml_predictor.models)
        status_card = Panel(
            Align.center(Text.from_markup(
                f"[{hb_style}] {hb_char} [/][bold white on bright_magenta] SUPREME v10 [/]\n"
                f"[dim]SES:[/][{sess_col}]{session}[/] [dim]|[/] [dim]ML:[/][cyan]{n_models}[/] [dim]|[/] {xch_str}"
            )), border_style="bright_magenta", box=box.ROUNDED
        )
        
        # Card 2: Market Context
        market_card = Panel(
            Align.center(Text.from_markup(
                f"[dim]VOL:[/][{mv_col}]{mv:.1f}[/] [dim]|[/] [dim]BIAS:[/][{bias_col}]{bias_str}[/]\n"
                f"[dim]BTC:[/][{btc_style}]{btc_status}[/]"
            )), title="[dim]MARKET[/]", border_style="cyan", box=box.ROUNDED
        )
        
        # Card 3: Balance
        bal_card = Panel(
            Align.center(Text.from_markup(
                f"[bold green]${bal_display}[/]\n"
                f"[{pnl_col}]{display_pnl:+.2f} USDT[/]"
            )), title="[dim]BALANCE[/]", border_style="green", box=box.ROUNDED
        )
        
        # Card 4: Performance
        total_trades = bot_state['wins'] + bot_state['losses']
        wr_val = (bot_state['wins'] / total_trades * 100) if total_trades > 0 else 0
        wr_col = "green" if wr_val >= 55 else ("yellow" if wr_val >= 45 else "red")
        perf_card = Panel(
            Align.center(Text.from_markup(
                f"[bold green]{bot_state['wins']}W[/] [dim]-[/] [bold red]{bot_state['losses']}L[/]\n"
                f"[dim]WR:[/][{wr_col}]{wr_val:.0f}%[/] [dim]|[/] [dim]PF:[/][bold]{(bot_state['wins']/max(bot_state['losses'],1)):.1f}[/]"
            )), title="[dim]PERFORMANCE[/]", border_style="yellow", box=box.ROUNDED
        )

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="main", ratio=2), # Increased ratio for market radar
            Layout(Panel(insight_text, title=" 🧠 AI ANALYST INSIGHT ", border_style="magenta", padding=(0, 1), box=box.ROUNDED), size=3),
            Layout(pos_element, ratio=1, minimum_size=4),
            Layout(Panel(
                Text.from_markup(
                    f"{bot_state['last_log']}\n"
                    f"[dim] [P] Passive | [C] Close All | [M] Menu | [K] Cancel Limit | [X] Exit | ⚡{ws_pkts} | {datetime.now().strftime('%H:%M:%S')}[/]"
                ), 
                title=" SYSTEM ", 
                border_style="dim", box=box.ROUNDED
            ), size=4)
        )
        
        layout["header"].split_row(
            Layout(status_card, ratio=20),
            Layout(market_card, ratio=15),
            Layout(bal_card, ratio=12),
            Layout(perf_card, ratio=12)
        )
        
        # Consolidate and sort movers by score (top 15)
        combined_list = sorted(all_valid, key=lambda x: x.get('score', 0), reverse=True)[:12]
        layout["main"].update(create_combined_table("🔥 MARKET RADAR", combined_list))
        
        return layout
    except Exception as e: return Panel(f"Error: {str(e)}")