# Threshold Adjustment Guide

## Current Thresholds (Conservative)
```python
# strategies/hybrid.py line 733-740
threshold = 75
if regime == "VOLATILE":
    threshold = 70 if is_breakout else 85
elif regime == "RANGING":
    threshold = 82
elif regime == "TRENDING":
    threshold = 68 if is_breakout else 75
```

## Recommended Aggressive Thresholds
```python
threshold = 65
if regime == "VOLATILE":
    threshold = 60 if is_breakout else 75
elif regime == "RANGING":
    threshold = 70
elif regime == "TRENDING":
    threshold = 58 if is_breakout else 65
```

## Trade-offs
- **Lower threshold** = More signals, more trades, potentially more losses
- **Higher threshold** = Fewer signals, higher quality, miss some opportunities

## How to Apply
Edit `/root/binance/strategies/hybrid.py` lines 733-740 with the aggressive values above.

## Alternative: Wait for Better Market Conditions
- Asia session (00:00-08:00 UTC): Low volume, sideways
- Europe session (08:00-16:00 UTC): Medium volume
- US session (13:30-20:00 UTC): High volume, best for scalping
