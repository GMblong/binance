import asyncio
import httpx
import time
import sys
import os
from rich.console import Console
from rich.table import Table
from websockets.asyncio.client import connect

sys.path.append(os.getcwd())

from strategies.hybrid import analyze_hybrid_async
from utils.state import market_data
from engine.websocket import _loads

console = Console()

async def public_ws_listener(symbol):
    """Public WS listener to populate market_data for testing."""
    s_low = symbol.lower()
    streams = [
        f"{s_low}@ticker",
        f"{s_low}@depth@500ms",
        f"{s_low}@aggTrade"
    ]
    uri = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"
    try:
        async with connect(uri) as ws:
            while True:
                msg = await ws.recv()
                raw = _loads(msg)
                stream = raw.get("stream", "")
                data = raw.get("data", {})
                if not data: continue
                
                if "aggTrade" in stream:
                    market_data.push_agg_trade(
                        data['s'], float(data['T'])/1000.0, 
                        float(data['q']), float(data['p']), data['m']
                    )
                elif "depth" in stream:
                    bids = sum(float(b[1]) for b in data['b'])
                    asks = sum(float(a[1]) for a in data['a'])
                    top_bid_q = float(data['b'][0][1]) if data['b'] else 0
                    top_ask_q = float(data['a'][0][1]) if data['a'] else 0
                    market_data.push_depth_snapshot(data['s'], bids, asks, top_bid_q, top_ask_q)
                elif "ticker" in stream:
                    market_data.prices[data['s']] = float(data['c'])
    except Exception as e:
        console.print(f"[dim red]WS Background Error: {e}[/]")

async def run_final_test(symbol="SOLUSDT"):
    console.print(f"\n[bold cyan]Menjalankan Analisis Sistem untuk {symbol}...[/]")
    
    # Jalankan listener di background agar data tick masuk sambil kita analisis
    listener_task = asyncio.create_task(public_ws_listener(symbol))
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Jalankan analisa hybrid (Ini akan memicu REST Fallback jika buffer masih kosong)
        console.print("[yellow]Menghitung sinyal (ML + Microstructure + Indicators)...[/]")
        t0 = time.time()
        result = await analyze_hybrid_async(client, symbol)
        latency = (time.time() - t0) * 1000
        
        if result:
            # --- TABEL HASIL UTAMA ---
            table = Table(title=f"Hasil Analisis: {symbol}")
            table.add_column("Parameter", style="cyan")
            table.add_column("Nilai", style="magenta")
            
            table.add_row("Harga Sekarang", f"${result['price']}")
            table.add_row("Rekomendasi", f"[bold]{result['sig']}[/]")
            table.add_row("Skor Kepercayaan", f"{result['score']}/100")
            table.add_row("Regime Pasar", result['regime'])
            table.add_row("Latensi Analisis", f"{latency:.0f}ms")
            
            ai = result.get("ai", {})
            table.add_row("Probabilitas ML (Naik)", f"{ai.get('ml_prob', 0)*100:.1f}%")
            table.add_row("Target Profit (TP)", f"{ai.get('tp', 0):.2f}%")
            table.add_row("Stop Loss (SL)", f"{ai.get('sl', 0):.2f}%")
            
            console.print(table)
            
            # --- TABEL MICROSTRUCTURE (Data 'Invisible') ---
            micro = market_data.micro_alpha.get(symbol, {})
            if micro:
                m_table = Table(title="Sinyal Institusi (Microstructure)")
                m_table.add_column("Sinyal", style="green")
                m_table.add_column("Nilai", style="yellow")
                m_table.add_column("Arti", style="dim")
                
                m_table.add_row("VPIN", f"{micro.get('vpin', 0):.4f}", "Deteksi Smart Money")
                m_table.add_row("OFI", f"{micro.get('ofi', 0):.4f}", "Ketidakseimbangan Order")
                m_table.add_row("KYLE LAMBDA", f"{micro.get('kyle_lambda', 0):.6f}", "Dampak Harga")
                m_table.add_row("HURST", f"{micro.get('hurst', 0):.2f}", ">0.5 Trending, <0.5 Reversion")
                m_table.add_row("WHALE PRINTS", str(micro.get("whale_prints", 0)), "Jumlah Transaksi Gajah")
                
                console.print(m_table)
            else:
                console.print("[red]Gagal mendapatkan data Microstructure.[/]")
        else:
            console.print("[bold red]Analisis gagal. Pastikan koneksi internet stabil.[/]")

    listener_task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(run_final_test("SOLUSDT"))
    except KeyboardInterrupt:
        pass
