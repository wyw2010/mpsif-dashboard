"""
Single entry point for Render deployment.
Launches the FastAPI dashboard and a background scheduler that
refreshes sub-fund data every 15 minutes.

Usage:
    python start.py                     # Local: http://localhost:8000
    PORT=8000 python start.py           # Custom port
"""

import os
import threading
import time
import logging

import schedule
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", "8000"))


def scheduler_loop():
    """Background thread: initial data load + periodic refresh."""
    import data_cache

    log.info("[Start] Initial data load beginning...")
    try:
        data_cache.refresh()
        log.info("[Start] Initial data load complete")
    except Exception as e:
        log.error(f"[Start] Initial data load failed: {e}")

    schedule.every(15).minutes.do(_safe_refresh)
    log.info("[Start] Scheduled data refresh every 15 minutes")

    while True:
        schedule.run_pending()
        time.sleep(30)


def _safe_refresh():
    """Wrapper to catch exceptions so the scheduler doesn't die."""
    import data_cache
    try:
        data_cache.refresh()
    except Exception as e:
        log.error(f"[Start] Scheduled refresh failed: {e}")


def main():
    # Start scheduler in background thread
    t = threading.Thread(target=scheduler_loop, daemon=True)
    t.start()

    # Start dashboard (blocking)
    log.info(f"[Start] Starting dashboard on port {PORT}")
    import uvicorn
    from dashboard import app
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
