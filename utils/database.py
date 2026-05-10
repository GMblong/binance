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

    # Table for active trade state (persists across restarts)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS active_trades (
            symbol TEXT PRIMARY KEY,
            data TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def save_state_to_db():
    try:
        snap = bot_state.snapshot_for_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Save Meta
        meta = {
            "daily_pnl": snap["daily_pnl"],
            "wins": snap["wins"],
            "losses": snap["losses"],
            "ai_confidence": snap["ai_confidence"],
            "start_balance": snap.get("start_balance", 0),
            "_last_day": snap.get("_last_day", ""),
        }
        for k, v in meta.items():
            cursor.execute("INSERT OR REPLACE INTO bot_meta (key, value) VALUES (?, ?)", (k, str(v)))
            
        # Save Symbol Perf
        for sym, data in snap["sym_perf"].items():
            cursor.execute("""
                INSERT OR REPLACE INTO sym_perf (symbol, wins, losses, consec_losses, last_loss_time)
                VALUES (?, ?, ?, ?, ?)
            """, (sym, data['w'], data['l'], data['c'], data.get('last_loss_time', 0)))
            
        # Save Strat Perf & Weights
        for feat, counts in snap["strat_perf"].items():
            weight = snap["neural_weights"].get(feat, 1.0)
            cursor.execute("""
                INSERT OR REPLACE INTO strat_perf (feature, wins, losses, weight)
                VALUES (?, ?, ?, ?)
            """, (feat, counts[0], counts[1], weight))

        # Save Per-Symbol Weights
        for sym, weights in snap["sym_weights"].items():
            for feat, weight in weights.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO sym_weights (symbol, feature, weight)
                    VALUES (?, ?, ?)
                """, (sym, feat, weight))
            
        # Save Blacklist
        for sym, expiry in snap["blacklist"].items():
            cursor.execute("INSERT OR REPLACE INTO blacklist (symbol, expiry) VALUES (?, ?)", (sym, expiry))

        # Save Active Trades
        cursor.execute("DELETE FROM active_trades")
        for sym, data in snap["trades"].items():
            cursor.execute("INSERT OR REPLACE INTO active_trades (symbol, data) VALUES (?, ?)",
                           (sym, json.dumps(data)))
            
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Save Error: {e}")

def load_state_from_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        updates = {}
        
        # Load Meta
        cursor.execute("SELECT key, value FROM bot_meta")
        for k, v in cursor.fetchall():
            if k in ["daily_pnl", "ai_confidence", "start_balance"]: updates[k] = float(v)
            elif k in ["wins", "losses"]: updates[k] = int(v)
            elif k == "_last_day": updates[k] = v
            
        # Load Symbol Perf
        sym_perf = {}
        cursor.execute("SELECT * FROM sym_perf")
        for row in cursor.fetchall():
            sym_perf[row[0]] = {'w': row[1], 'l': row[2], 'c': row[3], 'last_loss_time': row[4]}
        updates["sym_perf"] = sym_perf
            
        # Load Strat Perf
        strat_perf = {}
        neural_weights = {}
        cursor.execute("SELECT * FROM strat_perf")
        for row in cursor.fetchall():
            strat_perf[row[0]] = [row[1], row[2]]
            neural_weights[row[0]] = row[3]

        # Initialize defaults for new regimes if not present
        for regime in ["TRENDING", "RANGING", "VOLATILE", "EXHAUSTION", "SQUEEZE"]:
            for feat in ["liq", "ml", "ob", "div"]:
                key = f"{regime}:{feat}"
                if key not in neural_weights:
                    neural_weights[key] = 1.0
                    strat_perf[key] = [0, 0]

        updates["strat_perf"] = strat_perf
        updates["neural_weights"] = neural_weights

        # Load Per-Symbol Weights
        sym_weights = {}
        cursor.execute("SELECT * FROM sym_weights")
        for row in cursor.fetchall():
            sym, feat, weight = row
            if sym not in sym_weights: sym_weights[sym] = {}
            sym_weights[sym][feat] = weight
        updates["sym_weights"] = sym_weights
            
        # Load Blacklist
        now = time.time()
        blacklist = {}
        cursor.execute("SELECT * FROM blacklist WHERE expiry > ?", (now,))
        for row in cursor.fetchall():
            blacklist[row[0]] = row[1]
        updates["blacklist"] = blacklist

        # Load Active Trades
        try:
            trades = {}
            cursor.execute("SELECT symbol, data FROM active_trades")
            for sym, data_str in cursor.fetchall():
                trades[sym] = json.loads(data_str)
            updates["trades"] = trades
        except Exception:
            pass
            
        conn.close()
        bot_state.apply_db_load(updates)
    except Exception as e:
        print(f"DB Load Error: {e}")
