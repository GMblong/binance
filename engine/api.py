import time
import httpx
import asyncio
import urllib.parse
from utils.config import API_URL, API_KEY, API_SECRET
from utils.state import bot_state, symbol_info_cache, market_data
from utils.helpers import get_signature
from utils.logger import log_error

async def binance_request(client, method, endpoint, params=None):
    if bot_state.get("api_health_status") == "BLOCKED":
        return None # Circuit breaker active
        
    bot_state["api_req_count"] = bot_state.get("api_req_count", 0) + 1
    
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
            
            if res.status_code == 200:
                return res
            elif res.status_code == 429:
                bot_state["api_err_count"] = bot_state.get("api_err_count", 0) + 1
                bot_state["last_log"] = "[bold red]RATE LIMIT (429)[/]"
                await asyncio.sleep(min(30, 5 * (attempt + 1)))
            elif res.status_code == 418:
                bot_state["api_err_count"] = bot_state.get("api_err_count", 0) + 1
                bot_state["last_log"] = "[bold white on red] HARD BAN (418) - STOPPING 5 MIN [/]"
                await asyncio.sleep(300)
                return None
            elif res.status_code >= 500:
                bot_state["api_err_count"] = bot_state.get("api_err_count", 0) + 1
                raise httpx.HTTPStatusError(f"Server Error {res.status_code}", request=None, response=res)
            else:
                if res.status_code == 400:
                    try:
                        err_data = res.json()
                        if err_data.get("code") == -4130:
                            return res # Silent return for existing SL/TP
                        bot_state["api_err_count"] = bot_state.get("api_err_count", 0) + 1
                        log_error(f"API 400 Error for {endpoint}: {res.text}", include_traceback=False)
                    except:
                        bot_state["api_err_count"] = bot_state.get("api_err_count", 0) + 1
                        log_error(f"API 400 Error for {endpoint}: {res.text}", include_traceback=False)
                else:
                    bot_state["api_err_count"] = bot_state.get("api_err_count", 0) + 1
                    log_error(f"API Error {res.status_code} for {endpoint}", include_traceback=False)
                return res
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if attempt == 2: 
                bot_state["api_err_count"] = bot_state.get("api_err_count", 0) + 1
                log_error(f"API Request Final Fail ({endpoint}): {str(e)}")
                return None 
            await asyncio.sleep(2 * (attempt + 1))
    return None

async def get_symbol_precision(client, symbol):
    if symbol in symbol_info_cache: return symbol_info_cache[symbol]
    try:
        res = await client.get(f"{API_URL}/fapi/v1/exchangeInfo")
        if res.status_code == 200:
            data = res.json()
            for s in data['symbols']:
                if s['symbol'] == symbol:
                    tick_size = float(next(f['tickSize'] for f in s['filters'] if f['filterType'] == 'PRICE_FILTER'))
                    step_size = float(next(f['stepSize'] for f in s['filters'] if f['filterType'] == 'LOT_SIZE'))
                    
                    # Calculate precision based on tick/step size if p_prec/q_prec not accurate
                    p_prec = int(s.get('pricePrecision', 2))
                    q_prec = int(s.get('quantityPrecision', 2))
                    
                    prec = {
                        "tick": tick_size,
                        "step": step_size,
                        "p_prec": p_prec,
                        "q_prec": q_prec
                    }
                    symbol_info_cache[symbol] = prec
                    return prec
    except Exception as e:
        log_error(f"Precision Fetch Error for {symbol}: {str(e)}")
    
    # Smarter fallback: BTC/ETH usually have more precision than others
    if "BTC" in symbol or "ETH" in symbol:
        return {"tick": 0.01, "step": 0.001, "p_prec": 2, "q_prec": 3}
    return {"tick": 0.0001, "step": 0.1, "p_prec": 4, "q_prec": 1}

async def get_listen_key(client):
    try:
        res = await client.post(f"{API_URL}/fapi/v1/listenKey", headers={'X-MBX-APIKEY': API_KEY})
        if res.status_code == 200:
            return res.json()['listenKey']
    except Exception as e:
        log_error(f"ListenKey Fetch Error: {str(e)}")
    return None

async def keep_alive_listen_key(client, listen_key):
    try:
        await client.put(f"{API_URL}/fapi/v1/listenKey", headers={'X-MBX-APIKEY': API_KEY})
    except Exception as e:
        log_error(f"ListenKey KeepAlive Error: {str(e)}")

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
        
        # Fetch Open Interest concurrently (C7)
        sem = asyncio.Semaphore(5)
        async def fetch_oi(s):
            async with sem:
                try:
                    res_oi = await client.get(f"{API_URL}/fapi/v1/openInterest", params={"symbol": s})
                    if res_oi.status_code == 200:
                        market_data.oi[s] = float(res_oi.json()['openInterest'])
                except: pass
                
        tasks = [fetch_oi(s) for s in symbols]
        if tasks:
            await asyncio.gather(*tasks)
    except: pass

last_imbalance_fetch = {}

async def get_orderbook_imbalance(client, symbol):
    import time
    global last_imbalance_fetch
    now = time.time()
    
    # Throttle requests to once every 10 seconds per symbol to prevent 429
    if now - last_imbalance_fetch.get(symbol, 0) < 10:
        return market_data.imbalance.get(symbol, 1.0)
        
    try:
        res = await client.get(f"{API_URL}/fapi/v1/depth", params={"symbol": symbol, "limit": 20})
        if res.status_code == 200:
            data = res.json()
            bids = sum(float(b[1]) for b in data['bids'])
            asks = sum(float(a[1]) for a in data['asks'])
            
            # Calculate imbalance ratio (Bids / Asks)
            # > 1 means more bids (buying pressure), < 1 means more asks (selling pressure)
            if asks == 0: asks = 1 # Prevent division by zero
            imbalance = bids / asks
            market_data.imbalance[symbol] = imbalance
            last_imbalance_fetch[symbol] = now
            return imbalance
    except Exception as e:
        log_error(f"Orderbook Imbalance Error ({symbol}): {str(e)}")
    
    return market_data.imbalance.get(symbol, 1.0)

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
