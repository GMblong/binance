import hmac
import hashlib
import math
from typing import Union


def get_signature(query_string: str, api_secret: str) -> str:
    """Generate HMAC-SHA256 signature for Binance API authentication."""
    return hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()


def round_step(value: float, step: float) -> float:
    """Round value down to the nearest step size (for Binance quantity/price precision)."""
    if not step or step == 0 or not math.isfinite(value):
        return value
    step_str = format(step, 'f').rstrip('0')
    precision = len(step_str.split('.')[1]) if '.' in step_str else 0
    factor = 1 / step
    try:
        res = math.floor((value + 1e-12) * factor) / factor
        return round(res, precision)
    except (OverflowError, ValueError):
        return value
