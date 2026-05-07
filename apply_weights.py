import sqlite3
import os

DB_PATH = "bot_data.db"

weights = {
    "RANGING:liq": 2.0155,
    "RANGING:ob": 3.2712,
    "RANGING:div": 1.0664,
    "TRENDING:liq": 1.9642,
    "TRENDING:ob": 3.6968,
    "TRENDING:div": 2.5768
}

def update_db():
    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for feat, weight in weights.items():
        cursor.execute("SELECT feature FROM strat_perf WHERE feature = ?", (feat,))
        if cursor.fetchone():
            cursor.execute("UPDATE strat_perf SET weight = ? WHERE feature = ?", (weight, feat))
            print(f"Updated {feat} to {weight}")
        else:
            cursor.execute("INSERT INTO strat_perf (feature, wins, losses, weight) VALUES (?, 0, 0, ?)", (feat, weight))
            print(f"Inserted {feat} with weight {weight}")
            
    conn.commit()
    conn.close()
    print("Database update complete.")

if __name__ == "__main__":
    update_db()
