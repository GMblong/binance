import hmac
import hashlib
import math

def get_signature(query_string, api_secret):
    return hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()

def round_step(value, step):
    if step == 0: return value
    # Calculate precision based on the number of decimals in the step string
    step_str = format(step, 'f').rstrip('0')
    precision = len(step_str.split('.')[1]) if '.' in step_str else 0
    # Use Decimal-like precision to avoid floating point drift
    factor = 1 / step
    return round(math.floor((value + 1e-12) * factor) / factor, precision)
