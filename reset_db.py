import os
import sqlite3
from utils.database import init_db
from utils.state import bot_state

DB_PATH = "bot_data.db"

def reset_database():
    print(f"Stopping and resetting database: {DB_PATH}...")
    
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            print("Old database file removed successfully.")
        except Exception as e:
            print(f"Error removing database file: {e}")
            return

    # Re-initialize the database structure
    try:
        init_db()
        print("Database initialized with clean tables.")
    except Exception as e:
        print(f"Error initializing database: {e}")
        return

    print("\nDatabase reset complete!")
    print("Please RESTART the bot now to ensure the new state is loaded correctly.")

if __name__ == "__main__":
    reset_database()
