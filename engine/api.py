import time
import httpx
import asyncio
import urllib.parse
from utils.config import API_URL, API_KEY, API_SECRET
from utils.state import bot_state, symbol_info_cache
from utils.helpers import get_signature
from utils.logger import log_error

async def binance_request(client, method, endpoint, params=None):
    params = params or {}
    params['timestamp'] = int(time.time() * 1000)
    query_string = urllib.parse.urlencode(params)
    signature = get_signature(query_string, API_SECRET)
    full_params = {**params, "signature": signature}
    headers = {'X-MBX-APIKEY': API_KEY}
    
    url = f"{API_URL}{endpoint}"
    for attempt in range(3):
        try:
            if method == 'POST':
                res = await client.post(url, data=full_params, headers=headers, timeout=10)
            elif method == 'DELETE':
                res = await client.delete(url, params=full_params, headers=headers, timeout=10)
            else:
                res = await client.get(url, params=full_params, headers=headers, timeout=10)
            
            if res.status_code >= 500:
                raise httpx.HTTPStatusError(f"Server Error {res.status_code}", request=None, response=res)
            return res
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if attempt == 2: 
                log_error(f"API Request Final Fail ({endpoint}): {str(e)}")
                return None 
            await asyncio.sleep(1 * (attempt + 1))
    return None

async def get_symbol_precision(client, symbol):
    if symbol in symbol_info_cache: return symbol_info_cache[symbol]
    try:
        res = await client.get(f"{API_URL}/fapi/v1/exchangeInfo")
        data = res.json()
        for s in data['symbols']:
            if s['symbol'] == symbol:
                tick_size = float(next(f['tickSize'] for f in s['filters'] if f['filterType'] == 'PRICE_FILTER'))
                step_size = float(next(f['stepSize'] for f in s['filters'] if f['filterType'] == 'LOT_SIZE'))
                prec = {
                    "tick": tick_size,
                    "step": step_size,
                    "p_prec": int(s['pricePrecision']),
                    "q_prec": int(s['quantityPrecision'])
                }
                symbol_info_cache[symbol] = prec
                return prec
    except: pass
    return {"tick": 0.01, "step": 0.01, "p_prec": 2, "q_prec": 2}

async def get_balance_async(client):
    try:
        res = await binance_request(client, 'GET', '/fapi/v2/account')
        if res is None or res.status_code != 200:
            return bot_state["balance"]
            
        data = res.json()
        assets = data.get('assets', [])
        
        # Find USDT with case-insensitive check
        usdt_asset = next((a for a in assets if str(a['asset']).upper() == 'USDT'), None)
        if usdt_asset:
            # Try multiple possible keys returned by different Binance API versions
            val_str = usdt_asset.get('walletBalance') or usdt_asset.get('availableBalance') or usdt_asset.get('balance', '0')
            val = float(val_str)
            
            if val > 0:
                bot_state["balance"] = val
                # Critical: Ensure start_balance is never 0 if we have a real balance
                if bot_state.get("start_balance", 0) <= 0:
                    bot_state["start_balance"] = val
            return val
        return bot_state["balance"]
    except Exception as e:
        bot_state["last_log"] = f"[red]Bal Parser Err: {str(e)[:15]}[/]"
        log_error(f"Balance Parser Exception: {str(e)}")
        return bot_state["balance"]

async def get_market_depth_data(client, symbols):
    try:
        # Fetch Funding Rates
        res = await client.get(f"{API_URL}/fapi/v1/premiumIndex")
        if res.status_code == 200:
            data = res.json()
            for item in data:
                s = item['symbol']
                if s in symbols:
                    market_data.funding[s] = float(item['lastFundingRate'])
        
        # Fetch Open Interest for top symbols
        for s in symbols:
            res_oi = await client.get(f"{API_URL}/fapi/v1/openInterest", params={"symbol": s})
            if res_oi.status_code == 200:
                market_data.oi[s] = float(res_oi.json()['openInterest'])
    except: pass

async def get_orderbook_imbalance(client, symbol):
    try:
        res = await client.get(f"{API_URL}/fapi/v1/depth", params={"symbol": symbol, "limit": 20})
        if res.status_code == 200:
            data = res.json()
            bids = sum(float(b[1]) for b in data['bids'])
            asks = sum(float(a[1]) for b in data['asks'])
            ratio = bids / asks if asks > 0 else 1.0
            market_data.imbalance[symbol] = ratio
            return ratio
    except: pass
    return 1.0

async def get_btc_dominance(client):
    try:
        # Using BTCDOMUSDT contract as a proxy for dominance index
        res = await client.get(f"{API_URL}/fapi/v1/ticker/price", params={"symbol": "BTCDOMUSDT"})
        if res.status_code == 200:
            # The value is usually scaled, e.g., 1200 means ~12% dominance premium or similar
            # We just need the direction/relative value
            val = float(res.json()['price'])
            bot_state["btc_dom"] = val
    except: pass
