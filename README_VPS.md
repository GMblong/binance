# Panduan Migrasi ke VPS

Berikut adalah langkah-langkah untuk menjalankan bot trading ini di VPS Anda:

## 1. Persiapan di VPS (Ubuntu/Debian)
Update sistem dan install Python serta pip:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv tmux -y
```

## 2. Transfer File
Anda bisa menggunakan `git` (direkomendasikan) atau `scp` untuk memindahkan folder `trading/` ke VPS.

## 3. Setup Virtual Environment
Masuk ke folder project dan buat environment terisolasi:
```bash
cd trading
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 4. Konfigurasi API
Pastikan file `.env` sudah terisi dengan API Key dan Secret Binance Anda:
```bash
nano .env
```

## 5. Menjalankan Bot dengan TMUX
Gunakan `tmux` agar bot tetap jalan meskipun SSH terputus:
```bash
tmux new -s trading_bot
source venv/bin/activate
python3 hybrid_trader.py
```
*   Untuk keluar dari tampilan tmux (tanpa mematikan bot): Tekan `Ctrl+B` lalu `D`.
*   Untuk masuk kembali ke tampilan bot: Ketik `tmux attach -t trading_bot`.

## 6. Keluar dari Bot
Di dalam tampilan bot, Anda bisa menekan tombol `x` untuk keluar secara aman (membersihkan semua limit order yang menggantung).
