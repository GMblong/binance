import sqlite3
import time
import json
from utils.state import bot_state

DB_PATH = "bot_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table for general state
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bot_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    
    # Table for symbol performance
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sym_perf (
            symbol TEXT PRIMARY KEY,
            wins INTEGER,
            losses INTEGER,
            consec_losses INTEGER,
            last_loss_time REAL
        )
    """)
    
    # Table for strategy performance
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS strat_perf (
            feature TEXT PRIMARY KEY,
            wins INTEGER,
            losses INTEGER,
            weight REAL
        )
    """)
    
    # Table for blacklists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS blacklist (
            symbol TEXT PRIMARY KEY,
            expiry REAL
        )
    """)

    # Table for per-symbol neural weights
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sym_weights (
            symbol TEXT,
            feature TEXT,
            weight REAL,
            PRIMARY KEY (symbol, feature)
        )
    """)
    
    conn.commit()
    conn.close()

def save_state_to_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Save Meta
        meta = {
            "daily_pnl": bot_state["daily_pnl"],
            "wins": bot_state["wins"],
            "losses": bot_state["losses"],
            "ai_confidence": bot_state["ai_confidence"]
        }
        for k, v in meta.items():
            cursor.execute("INSERT OR REPLACE INTO bot_meta (key, value) VALUES (?, ?)", (k, str(v)))
            
        # Save Symbol Perf
        for sym, data in bot_state["sym_perf"].items():
            cursor.execute("""
                INSERT OR REPLACE INTO sym_perf (symbol, wins, losses, consec_losses, last_loss_time)
                VALUES (?, ?, ?, ?, ?)
            """, (sym, data['w'], data['l'], data['c'], data.get('last_loss_time', 0)))
            
        # Save Strat Perf & Weights
        for feat, counts in bot_state["strat_perf"].items():
            weight = bot_state["neural_weights"].get(feat, 1.0)
            cursor.execute("""
                INSERT OR REPLACE INTO strat_perf (feature, wins, losses, weight)
                VALUES (?, ?, ?, ?)
            """, (feat, counts[0], counts[1], weight))

        # Save Per-Symbol Weights
        for sym, weights in bot_state["sym_weights"].items():
            for feat, weight in weights.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO sym_weights (symbol, feature, weight)
                    VALUES (?, ?, ?)
                """, (sym, feat, weight))
            
        # Save Blacklist
        for sym, expiry in bot_state["blacklist"].items():
            cursor.execute("INSERT OR REPLACE INTO blacklist (symbol, expiry) VALUES (?, ?)", (sym, expiry))
            
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Save Error: {e}")

def load_state_from_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Load Meta
        cursor.execute("SELECT key, value FROM bot_meta")
        for k, v in cursor.fetchall():
            if k in ["daily_pnl", "ai_confidence"]: bot_state[k] = float(v)
            elif k in ["wins", "losses"]: bot_state[k] = int(v)
            
        # Load Symbol Perf
        cursor.execute("SELECT * FROM sym_perf")
        for row in cursor.fetchall():
            bot_state["sym_perf"][row[0]] = {'w': row[1], 'l': row[2], 'c': row[3], 'last_loss_time': row[4]}
            
        # Load Strat Perf
        cursor.execute("SELECT * FROM strat_perf")
        rows = cursor.fetchall()
        for row in rows:
            bot_state["strat_perf"][row[0]] = [row[1], row[2]]
            bot_state["neural_weights"][row[0]] = row[3]

        # Load Per-Symbol Weights
        cursor.execute("SELECT * FROM sym_weights")
        for row in cursor.fetchall():
            sym, feat, weight = row
            if sym not in bot_state["sym_weights"]: bot_state["sym_weights"][sym] = {}
            bot_state["sym_weights"][sym][feat] = weight
            
        # Initialize defaults for new regimes if not present
        for regime in ["TRENDING", "RANGING", "VOLATILE", "EXHAUSTION", "SQUEEZE"]:
            for feat in ["liq", "ml", "ob", "div"]:
                key = f"{regime}:{feat}"
                if key not in bot_state["neural_weights"]:
                    bot_state["neural_weights"][key] = 1.0
                    bot_state["strat_perf"][key] = [0, 0]
            
        # Load Blacklist
        now = time.time()
        cursor.execute("SELECT * FROM blacklist WHERE expiry > ?", (now,))
        for row in cursor.fetchall():
            bot_state["blacklist"][row[0]] = row[1]
            
        conn.close()
    except Exception as e:
        print(f"DB Load Error: {e}")
