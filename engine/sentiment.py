"""
News & Sentiment Filter
========================
Monitors Binance announcements + large liquidation events.
Auto-pauses or adjusts bot behavior during high-impact events.
"""
import asyncio
import time
import re
from utils.state import bot_state
from utils.logger import log_error


class SentimentFilter:
    def __init__(self):
        self.last_check = 0
        self.check_interval = 300  # 5 min
        self.active_events = []  # [{type, symbol, severity, expires}]
        self.liq_threshold = 1_000_000  # $1M liquidation = significant
        self.recent_liqs = {}  # {symbol: {ts, total_usd, direction}}

    async def check_announcements(self, client):
        """Check Binance announcements for listing/delisting/maintenance."""
        now = time.time()
        if now - self.last_check < self.check_interval:
            return
        self.last_check = now

        try:
            # Binance public announcement endpoint
            res = await client.get(
                "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query",
                params={"type": 1, "pageNo": 1, "pageSize": 5},
                timeout=5
            )
            if res.status_code != 200:
                return

            data = res.json()
            articles = data.get("data", {}).get("catalogs", [{}])[0].get("articles", []) if data.get("data") else []

            # Clear expired events
            self.active_events = [e for e in self.active_events if e['expires'] > now]

            for article in articles[:5]:
                title = article.get("title", "").upper()
                release_ts = article.get("releaseDate", 0) / 1000

                # Skip old news (>2h)
                if now - release_ts > 7200:
                    continue

                # Detect high-impact events
                if any(kw in title for kw in ["DELIST", "REMOVE", "SUSPEND"]):
                    symbols = self._extract_symbols(title)
                    for sym in symbols:
                        self.active_events.append({
                            'type': 'DELIST', 'symbol': sym,
                            'severity': 'HIGH', 'expires': now + 3600
                        })
                        bot_state.setdefault("blacklist", {})[sym] = now + 3600

                elif any(kw in title for kw in ["LIST", "NEW LISTING", "WILL LIST"]):
                    symbols = self._extract_symbols(title)
                    for sym in symbols:
                        self.active_events.append({
                            'type': 'LISTING', 'symbol': sym,
                            'severity': 'MEDIUM', 'expires': now + 1800
                        })

                elif any(kw in title for kw in ["MAINTENANCE", "UPGRADE", "SYSTEM"]):
                    self.active_events.append({
                        'type': 'MAINTENANCE', 'symbol': 'ALL',
                        'severity': 'HIGH', 'expires': now + 3600
                    })

        except Exception as e:
            log_error(f"Announcement check error: {str(e)[:50]}")

    async def check_liquidations(self, client):
        """Monitor large liquidation events.
        
        allForceOrders REST endpoint deprecated by Binance.
        Use OI drop detection as proxy for liquidation cascades.
        Real-time liquidations come via WS !forceOrder stream (subscribed separately).
        """
        try:
            from utils.state import market_data
            now = time.time()
            
            # Detect liquidation cascades from OI drops
            for sym, curr_oi in market_data.oi.items():
                prev_oi = market_data.prev_oi.get(sym, 0)
                if prev_oi == 0:
                    market_data.prev_oi[sym] = curr_oi
                    continue
                
                oi_drop_pct = (prev_oi - curr_oi) / (prev_oi + 1e-8) * 100
                market_data.prev_oi[sym] = curr_oi
                
                # OI dropped >5% in one check = likely liquidation cascade
                if oi_drop_pct > 5:
                    k = market_data.klines.get(sym, {}).get("1m")
                    if k is not None and len(k) >= 3:
                        price_change = (float(k.iloc[-1]['c']) - float(k.iloc[-3]['c'])) / float(k.iloc[-3]['c']) * 100
                        if sym not in self.recent_liqs:
                            self.recent_liqs[sym] = {'ts': now, 'total_usd': 0, 'buys': 0, 'sells': 0}
                        liq = self.recent_liqs[sym]
                        if now - liq['ts'] > 300:
                            liq = {'ts': now, 'total_usd': 0, 'buys': 0, 'sells': 0}
                        liq['ts'] = now
                        est_usd = curr_oi * abs(oi_drop_pct) / 100
                        liq['total_usd'] += est_usd
                        if price_change > 0:
                            liq['buys'] += est_usd
                        else:
                            liq['sells'] += est_usd
                        self.recent_liqs[sym] = liq
                        
                        if liq['total_usd'] > 5_000_000:
                            self.active_events.append({
                                'type': 'LIQ_CASCADE', 'symbol': sym,
                                'severity': 'HIGH', 'expires': now + 600
                            })
        except Exception as e:
            log_error(f"Liquidation check error: {str(e)[:50]}")

    def process_force_order(self, data: dict):
        """Process real-time liquidation from WS !forceOrder stream."""
        try:
            now = time.time()
            o = data.get("o", {})
            sym = o.get("s", "")
            qty = float(o.get("q", 0))
            price = float(o.get("p", 0))
            side = o.get("S", "")
            usd_val = qty * price

            if usd_val < self.liq_threshold:
                return

            if sym not in self.recent_liqs:
                self.recent_liqs[sym] = {'ts': now, 'total_usd': 0, 'buys': 0, 'sells': 0}

            liq = self.recent_liqs[sym]
            if now - liq['ts'] > 300:
                liq = {'ts': now, 'total_usd': 0, 'buys': 0, 'sells': 0}

            liq['total_usd'] += usd_val
            if side == "BUY":
                liq['buys'] += usd_val
            else:
                liq['sells'] += usd_val
            liq['ts'] = now
            self.recent_liqs[sym] = liq

            if liq['total_usd'] > 5_000_000:
                self.active_events.append({
                    'type': 'LIQ_CASCADE', 'symbol': sym,
                    'severity': 'HIGH', 'expires': now + 600
                })
                bot_state["last_log"] = f"[bold red]⚠ LIQ CASCADE: {sym} ${liq['total_usd']/1e6:.1f}M[/]"
        except Exception:
            pass

    def get_sentiment(self, symbol: str):
        """Get sentiment score for a symbol.
        
        Returns: float (-1 to +1)
        -1 = extremely bearish event (delist, long cascade liq)
        +1 = bullish event (new listing hype)
        0 = neutral
        """
        now = time.time()
        score = 0.0

        for event in self.active_events:
            if event['expires'] < now:
                continue
            if event['symbol'] != symbol and event['symbol'] != 'ALL':
                continue

            if event['type'] == 'DELIST':
                score -= 1.0
            elif event['type'] == 'LISTING':
                score += 0.5
            elif event['type'] == 'MAINTENANCE':
                score -= 0.3
            elif event['type'] == 'LIQ_CASCADE':
                # Direction depends on which side got liquidated
                liq = self.recent_liqs.get(symbol, {})
                if liq.get('sells', 0) > liq.get('buys', 0):
                    score -= 0.5  # Longs getting rekt = bearish
                else:
                    score += 0.5  # Shorts getting rekt = bullish

        return max(-1.0, min(1.0, score))

    def should_pause(self):
        """Check if bot should pause all trading due to system-wide event."""
        now = time.time()
        for event in self.active_events:
            if event['expires'] < now:
                continue
            if event['symbol'] == 'ALL' and event['severity'] == 'HIGH':
                return True
        return False

    def get_liq_bias(self, symbol: str):
        """Get liquidation-based directional bias.
        
        If shorts are getting liquidated heavily → price likely continues up.
        If longs are getting liquidated heavily → price likely continues down.
        """
        liq = self.recent_liqs.get(symbol)
        if not liq or time.time() - liq['ts'] > 300:
            return 0
        total = liq.get('buys', 0) + liq.get('sells', 0)
        if total < self.liq_threshold:
            return 0
        # Net direction: positive = shorts liquidated (bullish), negative = longs liquidated
        net = (liq.get('buys', 0) - liq.get('sells', 0)) / total
        if net > 0.3:
            return 1
        elif net < -0.3:
            return -1
        return 0

    def _extract_symbols(self, text: str):
        """Extract trading symbols from announcement text."""
        # Common patterns: "BTC", "ETHUSDT", etc.
        tokens = re.findall(r'\b([A-Z]{2,10})(?:USDT)?\b', text)
        # Filter to likely crypto symbols
        ignore = {"THE", "AND", "FOR", "NEW", "ALL", "WILL", "HAS", "ARE", "NOT",
                  "FROM", "WITH", "THIS", "THAT", "HAVE", "BEEN", "BINANCE", "USD",
                  "USDT", "FUTURES", "MARGIN", "SPOT", "TRADING", "PAIR"}
        symbols = []
        for t in tokens:
            if t not in ignore and len(t) >= 3:
                sym = t + "USDT" if not t.endswith("USDT") else t
                symbols.append(sym)
        return symbols[:3]  # Max 3 symbols per announcement

    async def run_loop(self, client):
        """Background loop to periodically check news + liquidations."""
        while True:
            try:
                await self.check_announcements(client)
                await asyncio.sleep(30)
                await self.check_liquidations(client)
                await asyncio.sleep(270)  # Total ~5 min cycle
            except Exception as e:
                log_error(f"SentimentFilter loop error: {str(e)[:50]}")
                await asyncio.sleep(60)


sentiment_filter = SentimentFilter()
