import httpx
import asyncio
import time
import json
from utils.config import env

TELEGRAM_BOT_TOKEN = env.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = env.get("TELEGRAM_CHAT_ID", "")
_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=10)
    return _client


# ============================================================
# LOW-LEVEL SEND HELPERS
# ============================================================

async def send_telegram(message: str, reply_markup=None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return None
    try:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message,
                   "parse_mode": "HTML", "disable_web_page_preview": True}
        # Always attach persistent reply keyboard unless an inline keyboard is specified
        if reply_markup:
            payload["reply_markup"] = reply_markup
        else:
            payload["reply_markup"] = _REPLY_KEYBOARD
        r = await _get_client().post(f"{_API}/sendMessage", json=payload)
        if r.status_code == 200:
            return r.json().get("result", {}).get("message_id")
    except Exception:
        pass
    return None


async def _edit_message(message_id, text, reply_markup=None):
    if not message_id:
        return
    try:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "message_id": message_id,
                   "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        if reply_markup:
            payload["reply_markup"] = reply_markup
        await _get_client().post(f"{_API}/editMessageText", json=payload)
    except Exception:
        pass


async def _answer_callback(callback_id, text=""):
    try:
        await _get_client().post(f"{_API}/answerCallbackQuery",
            json={"callback_query_id": callback_id, "text": text})
    except Exception:
        pass


def _kb(buttons):
    """Build InlineKeyboardMarkup from list of rows. Each row = list of (text, callback_data)."""
    return {"inline_keyboard": [[{"text": t, "callback_data": d} for t, d in row] for row in buttons]}


def _box(title: str, rows: list[str], icon: str = "") -> str:
    prefix = f"{icon} " if icon else ""
    parsed = []
    for r in rows:
        if "|" in r:
            label, value = r.split("|", 1)
            parsed.append((label.strip(), value.strip()))
        elif r == "---":
            parsed.append(None)
        else:
            parsed.append(("", r.strip()))
    labels = [p[0] for p in parsed if p is not None and p[0]]
    values = [p[1] for p in parsed if p is not None]
    lw = max((len(l) for l in labels), default=6)
    vw = max((len(v) for v in values), default=6)
    inner = lw + 3 + vw
    lines = [f"{prefix}<b>{title}</b>\n<code>"]
    lines.append(f"+{'-' * (inner + 2)}+")
    for p in parsed:
        if p is None:
            lines.append(f"+{'-' * (inner + 2)}+")
        elif p[0]:
            lines.append(f"| {p[0]:<{lw}} : {p[1]:>{vw}} |")
        else:
            lines.append(f"| {p[1]:^{inner}} |")
    lines.append(f"+{'-' * (inner + 2)}+")
    lines.append("</code>")
    return "\n".join(lines)


# ============================================================
# PERSISTENT REPLY KEYBOARD (always visible at bottom)
# ============================================================

_REPLY_KEYBOARD = {
    "keyboard": [
        ["📊 Status", "💰 PnL", "📂 Positions"],
        ["🎯 Signals", "🌍 Market", "⚠️ Risk"],
        ["⏸ Passive", "▶️ Active", "❌ Close All"],
        ["🟢 Start Bot", "🛑 Stop Bot"],
    ],
    "resize_keyboard": True,
    "is_persistent": True,
}

# Inline keyboard for sub-menus / confirmations
_INLINE_BACK = _kb([[("🔙 Menu", "menu")]])


async def send_control_panel():
    """Send welcome message that activates the persistent reply keyboard."""
    text = "🤖 <b>TRADING BOT</b> — Online\n\nUse the menu below 👇"
    await send_telegram(text)


# ============================================================
# ALERT FUNCTIONS (push notifications)
# ============================================================

async def alert_open_position(symbol, side, qty, price, leverage, signal, sl=None, tp=None, details=None):
    """Enhanced open alert with filter breakdown."""
    coin = symbol.replace("USDT", "")
    direction = "LONG" if side == "BUY" else "SHORT"
    icon = "🟢" if side == "BUY" else "🔴"
    rows = [
        f"Price|{price}",
        f"Size|{qty}",
        f"Leverage|{leverage}x",
        f"Signal|{signal}",
    ]
    if sl: rows.append(f"SL|{sl}")
    if tp: rows.append(f"TP|{tp}")

    if details:
        rows.append("---")
        if details.get("ml_prob"):
            rows.append(f"ML Conf|{details['ml_prob']:.0%}")
        if details.get("score"):
            rows.append(f"Score|{details['score']}/100")
        if details.get("regime"):
            rows.append(f"Regime|{details['regime']}")
        if details.get("atr_pct"):
            rows.append(f"ATR|{details['atr_pct']:.2f}%")
        # Filters passed
        filters = details.get("filters_passed", [])
        if filters:
            rows.append("---")
            rows.append(f"✅ {' | '.join(filters)}")
        # Brain signals
        brain_sigs = details.get("brain_signals", [])
        if brain_sigs:
            rows.append(f"🧠 {' | '.join(brain_sigs[:4])}")

    msg = _box(f"{direction}  {coin}/USDT", rows, icon)
    await send_telegram(msg)


async def alert_close_position(symbol, side, pnl, pnl_pct, reason, details=None):
    """Enhanced close alert with specific analysis reason."""
    coin = symbol.replace("USDT", "")
    icon = "💚" if pnl >= 0 else "❤️"
    label = "WIN" if pnl >= 0 else "LOSS"
    rows = [
        f"PnL|{pnl_pct:+.2f}% (${pnl:+.2f})",
        f"Reason|{reason}",
    ]
    if details:
        rows.append("---")
        if details.get("duration"):
            rows.append(f"Duration|{details['duration']}")
        if details.get("max_pnl"):
            rows.append(f"Max PnL|{details['max_pnl']:+.2f}%")
        if details.get("exit_trigger"):
            rows.append(f"Trigger|{details['exit_trigger']}")
        if details.get("ml_exit_prob"):
            rows.append(f"ML Exit|{details['ml_exit_prob']:.0%}")
        if details.get("analysis"):
            rows.append(f"Analysis|{details['analysis']}")
    msg = _box(f"{label}  {coin}/USDT  {side}", rows, icon)
    await send_telegram(msg)


async def alert_kill_switch(daily_pnl_pct, balance):
    msg = _box("KILL-SWITCH", [f"Daily PnL|{daily_pnl_pct:+.2f}%", f"Balance|${balance:.2f}", "---", "Trading halted"], "🚨")
    await send_telegram(msg)


async def alert_circuit_breaker(err_rate):
    msg = _box("CIRCUIT BREAKER", [f"Error Rate|{err_rate*100:.0f}%", "Trading paused"], "⚡")
    await send_telegram(msg)


async def alert_sentiment_pause(event=""):
    rows = ["High-impact event", "Entries paused"]
    if event: rows.append(event)
    msg = _box("SENTIMENT PAUSE", rows, "📰")
    await send_telegram(msg)


async def alert_startup(balance, positions):
    msg = _box("BOT ONLINE", [f"Balance|${balance:.2f}", f"Positions|{positions}", f"Time|{time.strftime('%H:%M UTC')}"], "🟢")
    await send_telegram(msg)
    await send_control_panel()


async def alert_shutdown():
    msg = _box("BOT OFFLINE", [f"State|Saved", f"Time|{time.strftime('%H:%M UTC')}"], "🔴")
    await send_telegram(msg)


async def alert_error(error):
    await send_telegram(f"⚠️ <b>ERROR</b>\n<code>{error[:200]}</code>")


async def alert_partial_close(symbol, pct, pnl_pct, reason):
    coin = symbol.replace("USDT", "")
    msg = _box(f"PARTIAL {int(pct*100)}%  {coin}", [f"PnL|{pnl_pct:+.2f}%", f"Reason|{reason}"], "✂️")
    await send_telegram(msg)


async def alert_limit_filled(symbol, side, price):
    coin = symbol.replace("USDT", "")
    direction = "LONG" if side == "BUY" else "SHORT"
    icon = "🟢" if side == "BUY" else "🔴"
    await send_telegram(f"{icon} <b>Limit Filled</b> — {coin} {direction} @ <code>{price}</code>")


async def alert_cooldown(symbol, losses):
    coin = symbol.replace("USDT", "")
    await send_telegram(f"🧊 <b>Cooldown</b> — {coin} ({losses} consecutive losses, 60min pause)")


async def alert_liquidation(symbol, side, qty, price):
    coin = symbol.replace("USDT", "")
    usd = qty * price
    if usd < 50000: return
    msg = _box(f"LIQUIDATION  {coin}", [f"Side|{'LONG' if side == 'SELL' else 'SHORT'}", f"Size|${usd:,.0f}", f"Price|{price}"], "💀")
    await send_telegram(msg)


async def alert_ml_retrain(symbol, win_rate):
    coin = symbol.replace("USDT", "")
    await send_telegram(f"🧠 <b>ML Retrained</b> — {coin} (WR: {win_rate:.0f}%)")


async def alert_blacklist_hit(symbol):
    coin = symbol.replace("USDT", "")
    await send_telegram(f"⛔ <b>Blacklisted</b> — {coin} (consecutive losses)")


async def alert_daily_summary(balance, pnl, wins, losses):
    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0
    icon = "📈" if pnl >= 0 else "📉"
    msg = _box("HOURLY REPORT", [f"Balance|${balance:.2f}", f"PnL|${pnl:+.2f}", f"Trades|{wins}W / {losses}L", f"Win Rate|{wr:.0f}%"], icon)
    await send_telegram(msg)


# ============================================================
# PROACTIVE ALERTS (background push without user asking)
# ============================================================

_proactive_state = {
    "last_drawdown_alert": 0,
    "last_regime": None,
    "last_milestone": 0,
    "streak_alerted": 0,
}


async def proactive_alert_check(bot_state, market_data):
    """Called every ~30s from main loop. Push alerts for important events."""
    now = time.time()
    balance = bot_state.get("balance", 0)
    start = bot_state.get("start_balance", 0)
    if start <= 0:
        return

    daily_pct = (balance - start) / start * 100

    # --- Drawdown Warning (tiered) ---
    if daily_pct < -2 and now - _proactive_state["last_drawdown_alert"] > 300:
        if daily_pct < -3.5:
            level = "🚨 CRITICAL"
        elif daily_pct < -2:
            level = "⚠️ WARNING"
        msg = _box(f"DRAWDOWN {level}", [
            f"Daily|{daily_pct:+.2f}%",
            f"Balance|${balance:.2f}",
            f"Kill-switch|at -5%",
        ], "📉")
        await send_telegram(msg)
        _proactive_state["last_drawdown_alert"] = now

    # --- Regime Change ---
    current_regime = bot_state.get("market_regime", "UNKNOWN")
    if current_regime != _proactive_state["last_regime"] and _proactive_state["last_regime"] is not None:
        await send_telegram(f"🔄 <b>Regime Change</b>\n{_proactive_state['last_regime']} → <b>{current_regime}</b>")
    _proactive_state["last_regime"] = current_regime

    # --- Winning Streak / Milestone ---
    wins = bot_state.get("wins", 0)
    losses = bot_state.get("losses", 0)
    streak = bot_state.get("win_streak", 0)
    if streak >= 5 and streak > _proactive_state["streak_alerted"]:
        await send_telegram(f"🔥 <b>{streak} Wins in a Row!</b>")
        _proactive_state["streak_alerted"] = streak
    if streak < 3:
        _proactive_state["streak_alerted"] = 0

    # PnL milestones ($25 increments)
    pnl = balance - start
    milestone = int(pnl / 25) * 25
    if milestone > 0 and milestone > _proactive_state["last_milestone"]:
        await send_telegram(f"💰 <b>Milestone!</b> Daily PnL reached <b>+${milestone}</b>")
        _proactive_state["last_milestone"] = milestone
    if pnl < _proactive_state["last_milestone"]:
        _proactive_state["last_milestone"] = max(0, milestone)

    # --- High-Confidence Missed Signal ---
    missed = bot_state.get("missed_signals", [])
    for sig in missed[-3:]:  # max 3 per check
        await send_telegram(
            f"👁 <b>Missed Signal</b> — {sig['symbol'].replace('USDT','')}\n"
            f"Dir: {'LONG' if sig['dir']==1 else 'SHORT'} | Score: {sig['score']}\n"
            f"Reason: {sig.get('reason','max positions')}")
    if missed:
        bot_state["missed_signals"] = []


# ============================================================
# CALLBACK QUERY HANDLERS (inline button responses)
# ============================================================

async def _handle_status(bot_state):
    balance = bot_state.get("balance", 0)
    mode = "⏸ Passive" if bot_state.get("is_passive") else "▶️ Active"
    positions = len(bot_state.get("active_positions", []))
    pending = len(bot_state.get("limit_orders", {}))
    api = bot_state.get("api_health_status", "OK")
    wins = bot_state.get("wins", 0)
    losses = bot_state.get("losses", 0)
    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0
    start_bal = bot_state.get("start_balance", 0)
    daily_pnl = ((balance - start_bal) / start_bal * 100) if start_bal > 0 else 0
    text = _box("STATUS", [
        f"Mode|{mode}", f"Balance|${balance:.2f}", f"Daily|{daily_pnl:+.2f}%",
        f"Positions|{positions} open, {pending} pending",
        f"Record|{wins}W / {losses}L ({wr:.0f}%)", f"API|{api}",
    ], "📊")
    kb = _kb([[("🔙 Menu", "menu")]])
    return text, kb


async def _handle_pnl(bot_state):
    balance = bot_state.get("balance", 0)
    start = bot_state.get("start_balance", 0)
    wins = bot_state.get("wins", 0)
    losses = bot_state.get("losses", 0)
    daily_pnl = balance - start if start > 0 else 0
    daily_pct = (daily_pnl / start * 100) if start > 0 else 0
    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0
    icon = "📈" if daily_pnl >= 0 else "📉"
    text = _box("PnL TODAY", [
        f"Start|${start:.2f}", f"Current|${balance:.2f}",
        f"PnL|${daily_pnl:+.2f} ({daily_pct:+.2f}%)",
        f"Trades|{total} ({wins}W {losses}L, {wr:.0f}%)",
    ], icon)
    kb = _kb([[("🔙 Menu", "menu")]])
    return text, kb


async def _handle_positions(bot_state):
    positions = bot_state.get("active_positions", [])
    if not positions:
        text = "📂 <b>No active positions</b>"
        kb = _kb([[("🔙 Menu", "menu")]])
        return text, kb

    rows = []
    buttons = []
    for p in positions:
        sym = p["symbol"]
        coin = sym.replace("USDT", "")
        amt = float(p["positionAmt"])
        entry = float(p["entryPrice"])
        side = "L" if amt > 0 else "S"
        upnl = float(p.get("unRealizedProfit", 0))
        notional = abs(amt) * entry
        pnl_pct = (upnl / notional * 100) if notional > 0 else 0
        emoji = "🟢" if pnl_pct >= 0 else "🔴"
        rows.append(f"{coin} {side}|{pnl_pct:+.2f}% (${upnl:+.2f})")
        buttons.append((f"❌ {coin}", f"close:{sym}"))

    text = _box(f"POSITIONS ({len(positions)})", rows, "📊")
    # Build button rows (2 per row)
    btn_rows = [buttons[i:i+2] for i in range(0, len(buttons), 2)]
    btn_rows.append([("🔙 Menu", "menu")])
    kb = _kb(btn_rows)
    return text, kb


async def _handle_market(bot_state, market_data):
    from utils.intelligence import get_current_session, calculate_market_volatility
    session = get_current_session()
    vol = bot_state.get("market_vol", 1.0)
    btc_state = bot_state.get("btc_state", "UNKNOWN")
    regime = bot_state.get("market_regime", "UNKNOWN")
    breadth = bot_state.get("alt_breadth", 50)
    n_pos = len(bot_state.get("active_positions", []))

    vol_label = "🔥 HIGH" if vol > 1.5 else ("❄️ LOW" if vol < 0.7 else "NORMAL")
    breadth_label = "🟢 RISK-ON" if breadth > 60 else ("🔴 RISK-OFF" if breadth < 40 else "NEUTRAL")

    text = _box("MARKET INTEL", [
        f"Session|{session}",
        f"BTC|{btc_state}",
        f"Regime|{regime}",
        f"Volatility|{vol:.2f}x ({vol_label})",
        f"Alt Breadth|{breadth:.0f}% ({breadth_label})",
        f"Positions|{n_pos}",
    ], "🌍")
    kb = _kb([[("🔙 Menu", "menu")]])
    return text, kb


async def _handle_signals(bot_state):
    results = bot_state.get("last_scan_results", [])
    if not results:
        text = "🎯 <b>No signals available</b>\n(waiting for next scan cycle)"
        kb = _kb([[("🔙 Menu", "menu")]])
        return text, kb

    # Top 5 by score
    top = sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:5]
    rows = []
    for r in top:
        coin = r.get("sym", "?")
        d = "⬆️" if r.get("dir", 0) == 1 else "⬇️"
        score = r.get("score", 0)
        ai = r.get("ai", {})
        ml = ai.get("ml_prob", 0.5)
        sig = r.get("sig", "")[:8]
        rows.append(f"{d} {coin}|{score}pts ML:{ml:.0%} {sig}")

    text = _box("TOP SIGNALS", rows, "🎯")
    kb = _kb([[("🔙 Menu", "menu")]])
    return text, kb


async def _handle_risk(bot_state):
    balance = bot_state.get("balance", 0)
    start = bot_state.get("start_balance", 0)
    positions = bot_state.get("active_positions", [])

    total_exposure = sum(abs(float(p["positionAmt"])) * float(p["markPrice"]) for p in positions)
    margin_pct = (total_exposure / balance * 100) if balance > 0 else 0
    daily_pnl = ((balance - start) / start * 100) if start > 0 else 0
    max_dd = bot_state.get("max_drawdown_pct", 0)
    n_sectors = len(set(bot_state.get("position_sectors", {}).values()))

    risk_level = "🟢 LOW" if margin_pct < 30 else ("🟡 MED" if margin_pct < 60 else "🔴 HIGH")

    text = _box("RISK DASHBOARD", [
        f"Exposure|${total_exposure:,.0f}",
        f"Margin Used|{margin_pct:.1f}% ({risk_level})",
        f"Daily PnL|{daily_pnl:+.2f}%",
        f"Max Drawdown|{max_dd:.2f}%",
        f"Sectors|{n_sectors} diversified",
        f"Open|{len(positions)} / {bot_state.get('max_positions', 5)}",
    ], "⚠️")
    kb = _kb([[("🔙 Menu", "menu")]])
    return text, kb


# ============================================================
# MAIN COMMAND LOOP (handles both /commands and callback queries)
# ============================================================

_last_update_id = 0


async def _get_updates():
    global _last_update_id
    try:
        res = await _get_client().get(f"{_API}/getUpdates",
            params={"offset": _last_update_id + 1, "timeout": 5})
        if res.status_code == 200:
            return res.json().get("result", [])
    except Exception:
        pass
    return []


async def command_loop(bot_state, market_data, action_queue):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    global _last_update_id
    await _set_bot_commands()
    await send_control_panel()

    while True:
        try:
            updates = await _get_updates()
            for u in updates:
                _last_update_id = u["update_id"]

                # --- Handle Callback Queries (inline button clicks) ---
                cb = u.get("callback_query")
                if cb:
                    cb_id = cb["id"]
                    chat_id = str(cb.get("message", {}).get("chat", {}).get("id", ""))
                    msg_id = cb.get("message", {}).get("message_id")
                    data = cb.get("data", "")

                    if chat_id != TELEGRAM_CHAT_ID:
                        await _answer_callback(cb_id)
                        continue

                    text, kb = None, None

                    if data == "menu" or data == "refresh":
                        text = "🤖 <b>TRADING BOT</b>\n\nUse the keyboard below 👇"
                        kb = None
                    elif data == "status":
                        text, kb = await _handle_status(bot_state)
                    elif data == "pnl":
                        text, kb = await _handle_pnl(bot_state)
                    elif data == "positions":
                        text, kb = await _handle_positions(bot_state)
                    elif data == "market":
                        text, kb = await _handle_market(bot_state, market_data)
                    elif data == "signals":
                        text, kb = await _handle_signals(bot_state)
                    elif data == "risk":
                        text, kb = await _handle_risk(bot_state)
                    elif data == "passive":
                        bot_state["is_passive"] = True
                        text = "⏸ <b>PASSIVE MODE</b>\nNo new entries. Existing positions managed."
                        kb = _kb([[("▶️ Resume Trading", "active"), ("🔙 Menu", "menu")]])
                    elif data == "active":
                        bot_state["is_passive"] = False
                        text = "▶️ <b>ACTIVE MODE</b>\nTrading resumed."
                        kb = _kb([[("⏸ Pause", "passive"), ("🔙 Menu", "menu")]])
                    elif data == "closeall_confirm":
                        text = "⚠️ <b>Close ALL positions?</b>\nThis cannot be undone."
                        kb = _kb([[("✅ Yes, Close All", "closeall_exec"), ("❌ Cancel", "menu")]])
                    elif data == "closeall_exec":
                        await action_queue.put("CLOSE_ALL")
                        text = "⚡ <b>Closing all positions...</b>"
                        kb = _kb([[("🔙 Menu", "menu")]])
                    elif data == "cancel_confirm":
                        text = "🗑 <b>Cancel all limit orders?</b>"
                        kb = _kb([[("✅ Yes", "cancel_exec"), ("❌ No", "menu")]])
                    elif data == "cancel_exec":
                        await action_queue.put("CANCEL_ORDERS")
                        text = "🗑 <b>Orders cancelled</b>"
                        kb = _kb([[("🔙 Menu", "menu")]])
                    elif data == "stop_confirm":
                        text = "🛑 <b>Stop the bot?</b>\nWill cancel orders and save state."
                        kb = _kb([[("✅ Yes, Stop", "stop_exec"), ("❌ Cancel", "menu")]])
                    elif data == "stop_exec":
                        await action_queue.put("SHUTDOWN")
                        text = "🔴 <b>Stopping bot...</b>"
                        kb = None
                    elif data.startswith("close:"):
                        symbol = data.split(":", 1)[1]
                        await action_queue.put(f"CLOSE:{symbol}")
                        text = f"⚡ Closing <b>{symbol.replace('USDT','')}</b>..."
                        kb = _kb([[("🔙 Positions", "positions"), ("🔙 Menu", "menu")]])

                    if text:
                        await _edit_message(msg_id, text, kb)
                    await _answer_callback(cb_id)
                    continue

                # --- Handle Text Commands + Reply Keyboard Buttons ---
                msg = u.get("message", {})
                chat_id = str(msg.get("chat", {}).get("id", ""))
                text = msg.get("text", "").strip()

                if chat_id != TELEGRAM_CHAT_ID:
                    continue

                # Strip emoji variant selectors for reliable matching
                clean = text.replace("\ufe0f", "")

                # Map reply keyboard button text to actions
                if clean in ("/start", "/menu", "/panel", "/help"):
                    await send_control_panel()
                elif "Status" in clean or clean == "/status":
                    t, k = await _handle_status(bot_state)
                    await send_telegram(t, k)
                elif "Positions" in clean or clean in ("/positions", "/pos"):
                    t, k = await _handle_positions(bot_state)
                    await send_telegram(t, k)
                elif "PnL" in clean or clean == "/pnl":
                    t, k = await _handle_pnl(bot_state)
                    await send_telegram(t, k)
                elif "Market" in clean or clean == "/market":
                    t, k = await _handle_market(bot_state, market_data)
                    await send_telegram(t, k)
                elif "Signals" in clean or clean == "/signals":
                    t, k = await _handle_signals(bot_state)
                    await send_telegram(t, k)
                elif "Risk" in clean or clean == "/risk":
                    t, k = await _handle_risk(bot_state)
                    await send_telegram(t, k)
                elif "Passive" in clean or clean == "/passive":
                    bot_state["is_passive"] = True
                    await send_telegram("⏸ <b>PASSIVE MODE</b>\nNo new entries. Existing positions managed.")
                elif "Active" in clean or clean == "/active":
                    bot_state["is_passive"] = False
                    await send_telegram("▶️ <b>ACTIVE MODE</b>\nTrading resumed.")
                elif "Close All" in clean or clean == "/closeall":
                    await send_telegram("⚠️ <b>Close ALL positions?</b>", _kb([[("✅ Yes, Close All", "closeall_exec"), ("❌ Cancel", "menu")]]))
                elif "Stop Bot" in clean or clean == "/stopbot":
                    await send_telegram("🛑 Stop bot?", _kb([[("✅ Yes", "stop_exec"), ("❌ No", "menu")]]))
                elif "Start Bot" in clean or clean == "/startbot":
                    import subprocess
                    result = subprocess.run(["systemctl", "is-active", "trading-bot"], capture_output=True, text=True)
                    if result.stdout.strip() == "active":
                        await send_telegram("⚠️ Bot already running")
                    else:
                        subprocess.run(["systemctl", "start", "trading-bot"])
                        await asyncio.sleep(3)
                        result = subprocess.run(["systemctl", "is-active", "trading-bot"], capture_output=True, text=True)
                        if result.stdout.strip() == "active":
                            await send_telegram("🟢 <b>Bot Started</b>")
                        else:
                            await send_telegram("❌ Start failed — check journalctl")
                elif clean.startswith("/close "):
                    symbol = clean.split(" ", 1)[1].upper()
                    if not symbol.endswith("USDT"): symbol += "USDT"
                    await action_queue.put(f"CLOSE:{symbol}")
                    await send_telegram(f"⚡ Closing {symbol}...")

            await asyncio.sleep(1)
        except Exception as e:
            from utils.logger import log_error
            log_error(f"Telegram command_loop error: {e}")
            await asyncio.sleep(5)


async def _set_bot_commands():
    commands = [
        {"command": "menu", "description": "Control panel"},
        {"command": "status", "description": "Bot overview"},
        {"command": "positions", "description": "Open trades"},
        {"command": "pnl", "description": "Today's PnL"},
        {"command": "market", "description": "Market intelligence"},
        {"command": "signals", "description": "Top signals"},
        {"command": "risk", "description": "Risk dashboard"},
        {"command": "passive", "description": "Pause entries"},
        {"command": "active", "description": "Resume trading"},
        {"command": "closeall", "description": "Close all positions"},
        {"command": "stopbot", "description": "Stop bot"},
    ]
    try:
        await _get_client().post(f"{_API}/setMyCommands", json={"commands": commands})
    except Exception:
        pass
