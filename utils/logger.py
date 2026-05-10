import os
import traceback
from datetime import datetime

LOG_FILE = "error.log"

def init_logger():
    """Clears the log file at the start of each session."""
    try:
        with open(LOG_FILE, "w") as f:
            f.write(f"=== Bot Session Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    except Exception: pass

def log_error(msg, include_traceback=True):
    """Logs an error message with timestamp and optional traceback."""
    try:
        with open(LOG_FILE, "a") as f:
            timestamp = datetime.now().strftime("%H:%M:%S")
            f.write(f"[{timestamp}] ERROR: {msg}\n")
            if include_traceback:
                f.write(traceback.format_exc())
                f.write("-" * 50 + "\n")
    except Exception: pass
