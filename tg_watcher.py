"""
Telegram Watcher - Always-on service.
- When trading-bot is OFFLINE: handles ALL commands (start/stop/status)
- When trading-bot is ONLINE: only handles /stopbot, defers rest to main bot
"""
import httpx
import asyncio
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import env

TOKEN = env.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = env.get("TELEGRAM_CHAT_ID", "")
API = f"https://api.telegram.org/bot{TOKEN}"


async def send(text):
    kb = {"keyboard": [
        ["📊 Status", "💰 PnL", "📂 Positions"],
        ["🎯 Signals", "🌍 Market", "⚠️ Risk"],
        ["⏸ Passive", "▶️ Active", "❌ Close All"],
        ["🟢 Start Bot", "🛑 Stop Bot"],
    ], "resize_keyboard": True, "is_persistent": True}
    async with httpx.AsyncClient(timeout=10) as c:
        await c.post(f"{API}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML", "reply_markup": kb})


def bot_is_running():
    r = subprocess.run(["systemctl", "is-active", "trading-bot"], capture_output=True, text=True)
    return r.stdout.strip() == "active"


async def main():
    if not TOKEN or not CHAT_ID:
        print("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
        return

    last_update_id = 0

    while True:
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                # If bot is running, main bot handles commands — we just idle
                if bot_is_running():
                    await asyncio.sleep(5)
                    continue

                res = await c.get(f"{API}/getUpdates",
                    params={"offset": last_update_id + 1, "timeout": 10})
                if res.status_code != 200:
                    await asyncio.sleep(5)
                    continue

                for u in res.json().get("result", []):
                    last_update_id = u["update_id"]
                    msg = u.get("message", {})
                    chat_id = str(msg.get("chat", {}).get("id", ""))
                    text = msg.get("text", "").strip()

                    if chat_id != CHAT_ID:
                        continue

                    if text == "/startbot" or "Start Bot" in text:
                        subprocess.run(["systemctl", "start", "trading-bot"])
                        await asyncio.sleep(3)
                        if bot_is_running():
                            await send("🟢 <b>Bot Started</b>")
                        else:
                            await send("❌ Start failed\njournalctl -u trading-bot")

                    elif text == "/stopbot" or "Stop Bot" in text:
                        await send("Bot is not running.")

                    elif text in ("/status", "/start"):
                        await send("<code>+----------------+\n| Status : Offline |\n+----------------+</code>\n\nUse /startbot to start.")

                    else:
                        await send("🔴 Bot is offline.\n\nUse /startbot to start.")

        except Exception:
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
