"""
Run the dashboard as a standalone process.

Usage:
    .venv/bin/python scripts/run_dashboard.py

Then open http://127.0.0.1:5000 in your browser.
The bot does not need to be running — the dashboard reads the database directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.config import load_config
from src.dashboard.server import run

config = load_config()

config_summary = {
    "trading_env":          config.binance.trading_env.value,
    "symbol":               config.strategy.symbol,
    "timeframe":            config.strategy.timeframe,
    "ema_period":           config.strategy.ema_period,
    "rsi_period":           config.strategy.rsi_period,
    "atr_stop_multiplier":  config.strategy.atr_stop_multiplier,
    "atr_trail_multiplier": config.strategy.atr_trail_multiplier,
    "risk_pct":             config.risk.risk_per_trade_pct * 100,
}

print(f"Dashboard running at http://127.0.0.1:5000")
print(f"Press Ctrl+C to stop.")
run(db_path=config.database.path, config_summary=config_summary)
