import hmac
import hashlib
import math

def get_signature(query_string, api_secret):
    return hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()

def round_step(value, step):
    return round(math.floor(value / step) * step, 8)
