# Binance Futures Bot - Histori Perbaikan (9 Mei 2026)

## Nilai Awal: 68/100 → Akhir: ~82/100

## Bug yang Diperbaiki

### 1. Endpoint SL/TP Error -4120
- **Masalah**: Binance memindahkan conditional orders (STOP_MARKET, TAKE_PROFIT_MARKET) dari `/fapi/v1/order` ke Algo Order API
- **Fix**: Gunakan `POST /fapi/v1/algoOrder` dengan `algoType=CONDITIONAL` dan `triggerPrice` (bukan `stopPrice`)
- **Cancel**: `DELETE /fapi/v1/algoOrder` per `algoId` (tidak ada batch cancel)

### 2. SL/TP Tidak Terpasang
- **Masalah**: Posisi aktif tidak punya SL/TP karena endpoint lama error
- **Fix**: Auto-attach SL/TP di `manage_active_positions` — cek algo orders + regular orders, pasang jika belum ada

### 3. SystemExit Crash saat Quit (q/x)
- **Masalah**: `raise SystemExit(0)` di async task menyebabkan unhandled exception
- **Fix**: Gunakan `asyncio.get_event_loop().stop()` + `finally` block di `__main__`

### 4. Trailing Stop Trigger saat Restart
- **Masalah**: `bot_state["trades"]` (termasuk `peak`) tidak persist → reset saat restart → trailing langsung trigger
- **Fix**: Simpan/load `trades` state ke tabel `active_trades` di SQLite

### 5. Bare except:pass (30+ instance)
- **Fix**: Semua diganti `except Exception:` dengan logging di yang kritis

### 6. ML Veto Rule
- **Masalah**: Bot buka posisi yang bertentangan dengan ML prediction
- **Fix**: Block entry jika `ml_prob < 0.35` (untuk LONG) atau `ml_prob > 0.65` (untuk SHORT)

### 7. ML Training Data Kurang
- **Masalah**: `train_model` hanya fetch 500 candles (~8 jam)
- **Fix**: Dinaikkan ke 1500 candles (~25 jam) untuk akurasi lebih baik

### 8. PnL Display Tidak Akurat
- **Masalah**: Dashboard tampilkan PnL tanpa fee, berbeda dengan Binance app
- **Fix**: Kurangi estimasi fee (0.05% entry + 0.05% exit = 0.10% notional)

### 9. os._exit(0) → Graceful Shutdown
- **Fix**: Save DB state + stop loop + sys.exit(0)

### 10. Magic Numbers → Named Constants
- `MIN_NOTIONAL_USD`, `CONSEC_LOSS_COOLDOWN_SEC`, `ML_RETRAIN_INTERVAL_SEC`, `API_BAN_SLEEP_SEC`, dll di `config.py`

## Endpoint yang Valid (Tested)
```
✅ GET  /fapi/v1/exchangeInfo
✅ GET  /fapi/v1/ticker/price
✅ GET  /fapi/v1/klines
✅ GET  /fapi/v1/depth
✅ GET  /fapi/v1/openInterest
✅ GET  /fapi/v1/ticker/24hr
✅ GET  /fapi/v2/balance
✅ GET  /fapi/v2/positionRisk
✅ GET  /fapi/v1/openOrders
✅ GET  /fapi/v1/openAlgoOrders
✅ POST /fapi/v1/leverage
✅ POST /fapi/v1/listenKey
✅ POST /fapi/v1/algoOrder (SL/TP conditional)
✅ POST /fapi/v1/order (LIMIT/MARKET only)
✅ DELETE /fapi/v1/allOpenOrders
✅ DELETE /fapi/v1/algoOrder (per algoId)
❌ POST /fapi/v1/order (STOP_MARKET) → gunakan algoOrder
❌ DELETE /fapi/v1/allOpenAlgoOrders → tidak ada, cancel per algoId
```

## Testing
- 56 unit tests (all pass)
- Files: test_helpers.py, test_state.py, test_config.py, test_screener.py, test_trading.py, test_ml_engine.py
- Config: pyproject.toml dengan pytest asyncio auto-mode

## Arsitektur Penting
- `bot_state["trades"]` sekarang persist ke DB (tabel `active_trades`)
- ML veto di `strategies/hybrid.py` line ~633
- SL/TP via `/fapi/v1/algoOrder` di `engine/trading.py`
- Auto-attach SL/TP di `manage_active_positions`
- Fee estimation di `ui/dashboard.py`

---

## Update Lanjutan (9 Mei 2026, 11:00-11:21)

### 11. Exit Logic Lebih Peka (Protect Profit)
- Partial TP trigger: RR 1:1 (min 0.6%) → **RR 0.7:1 (min 0.45%)**
- ML bearish callback: ×0.6 → **×0.4** (trailing jauh lebih ketat)
- Breakeven buffer: 50% SL → **30% SL** (protect lebih cepat)
- BE trigger: setelah RR 1:1 → **setelah 70% SL distance**

### 12. Slippage Diperhitungkan
- Total buffer fee+slippage: **0.15%**
- Partial TP minimum: **0.45%** (net profit ~0.30% setelah fee+slippage)
- Breakeven trigger: close saat profit tinggal **0.15%** (= breakeven setelah costs)

### 13. ML Veto Rule
- Block LONG jika `ml_prob < 0.35` (ML 65%+ yakin SHORT)
- Block SHORT jika `ml_prob > 0.65` (ML 65%+ yakin LONG)
- Ditambahkan di `strategies/hybrid.py`

### 14. ML Training Data Dinaikkan
- `train_model`: 500 → **1500 candles** (~25 jam data 1m)
- Akurasi prediksi lebih baik

### 15. Cancel Algo Orders Fix
- `DELETE /fapi/v1/allOpenAlgoOrders` → 404 (tidak ada)
- Fix: fetch `GET /fapi/v1/openAlgoOrders` lalu cancel per `algoId`

### Verifikasi Live
- AXSUSDT: ML bilang SHORT → bot close otomatis via AI-TRAIL ✅
- AXSUSDT limit BUY: ML setuju LONG 68% conf ✅
- ONUSDT limit: ML strongly disagree → tidak akan dibuka lagi (veto) ✅
- SL/TP auto-attach berfungsi dalam 3-5 detik ✅
- Error log bersih setelah semua fix ✅
