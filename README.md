# Binance Futures Trading Bot

Automated cryptocurrency futures trading bot for Binance with ML ensemble, microstructure analysis, and multi-exchange data feeds.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    hybrid_trader.py                       │
│              (Main Loop & Orchestrator)                   │
├─────────────┬──────────────┬────────────┬───────────────┤
│  engine/    │ strategies/  │   utils/   │     ui/       │
├─────────────┼──────────────┼────────────┼───────────────┤
│ api.py      │ hybrid.py    │ config.py  │ dashboard.py  │
│ trading.py  │ analyzer.py  │ state.py   │               │
│ websocket.py│              │ database.py│               │
│ ml_engine.py│              │ helpers.py │               │
│ superhuman  │              │ intelligence│              │
│ microstructure│            │ logger.py  │               │
│ sentiment.py│              │            │               │
│ depth_predictor│           │            │               │
│ multi_exchange│            │            │               │
│ auto_optimizer│            │            │               │
└─────────────┴──────────────┴────────────┴───────────────┘
```

## Data Flow

```
Binance WS ──┐
Bybit WS ────┼──► MarketData (state.py) ──► analyze_hybrid_async()
OKX WS ──────┘         │                         │
                        │                    ┌────┴────┐
                        ▼                    ▼         ▼
                  coin_screener.py     ML Ensemble  Technical
                        │              (LGB+XGB+MLP) Analysis
                        ▼                    │         │
                  top_symbols ◄──────────────┴────┬────┘
                        │                         ▼
                        └──────────────► Signal Generation
                                              │
                                              ▼
                                     open_position_async()
                                     manage_active_positions()
```

## Modules

### engine/
- **api.py** — Binance REST API wrapper with retry, rate limit, circuit breaker
- **trading.py** — Position open/close, SL/TP management, partial exits
- **websocket.py** — Real-time kline, ticker, aggTrade, depth streams
- **ml_engine.py** — LightGBM + XGBoost + MLP ensemble with online retraining
- **superhuman.py** — 12 signals invisible to humans (VPIN, entropy, tick imbalance)
- **microstructure.py** — Kyle's lambda, Hurst exponent, whale prints, absorption
- **sentiment.py** — News/announcement filter + liquidation cascade detection
- **depth_predictor.py** — Orderbook wall spoofing detection (online learning)
- **multi_exchange.py** — Bybit + OKX price feeds for cross-exchange divergence
- **auto_optimizer.py** — Daily parameter sweep via backtest

### strategies/
- **hybrid.py** — Main signal generator combining all engines
- **analyzer.py** — Technical indicators (Numba-accelerated EMA, RSI, ATR, ADX, HMM)

### utils/
- **config.py** — All configuration constants and .env loading
- **state.py** — BotState (thread-safe dict) + MarketData (klines, ticks, depth)
- **database.py** — SQLite persistence for sym_perf, strat_perf, neural_weights
- **intelligence.py** — Kelly criterion, sector correlation, dynamic clustering
- **helpers.py** — HMAC signature, price/qty rounding
- **logger.py** — Error logging to file

## Setup

```bash
# 1. Clone and install
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your Binance API keys

# 3. Run
python hybrid_trader.py
```

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| BINANCE_API_KEY | — | Binance Futures API key |
| BINANCE_API_SECRET | — | Binance Futures API secret |
| MAX_POSITIONS | 5 | Max concurrent positions |
| ACCOUNT_RISK_PERCENT | 0.02 | Risk per trade (2%) |
| MAX_LEVERAGE | 20 | Maximum leverage |
| USE_BTC_FILTER | False | Only trade with BTC trend |
| DAILY_LOSS_LIMIT_PCT | 0.05 | Kill-switch at 5% daily loss |
| DAILY_PROFIT_TARGET_PCT | 0.10 | Kill-switch at 10% daily profit |

## Keyboard Shortcuts (Runtime)

| Key | Action |
|-----|--------|
| `p` | Toggle passive mode (stop new entries) |
| `c` | Close ALL positions (market) |
| `m`/`Tab` | Select individual position to close |
| `k` | Cancel all limit orders |
| `r` | Reload intelligence weights from DB |
| `q`/`x` | Graceful exit (cancel orders, save state) |

## Risk Management

- **Kill-switch**: Auto-stops trading at daily loss/profit limits
- **Circuit breaker**: Pauses API calls when error rate > 30%
- **Consecutive loss cooldown**: 60min blacklist after 3 losses on same symbol
- **Correlation filter**: Prevents overexposure to same sector
- **Sentiment pause**: Halts trading during high-impact news events
- **Dynamic position sizing**: Kelly criterion × ML confidence

## Testing

```bash
# Unit tests
python -m pytest tests/ -v

# Benchmark (2 min speed test)
python benchmark.py

# Benchmark with ML models
python benchmark_ml.py

# Backtest
python backtest_pro.py --symbols BTCUSDT,ETHUSDT --days 7
```
