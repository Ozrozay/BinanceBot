"""
CLI entry point for running a backtest.

Usage:
    .venv/bin/python scripts/run_backtest.py
    .venv/bin/python scripts/run_backtest.py --days 365 --equity 10000 --no-cache
"""

import argparse
import sys
from pathlib import Path

# Make src importable from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.config import load_config
from src.logging.structured import setup_logging
from src.api.client import BinanceClient
from src.strategy.trend_following import TrendFollowingStrategy
from src.backtest.runner import run_backtest


def main():
    parser = argparse.ArgumentParser(description="Run trend-following backtest")
    parser.add_argument("--days", type=int, default=180, help="Days of history (default 180)")
    parser.add_argument("--equity", type=float, default=10_000.0, help="Starting equity in USDT")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh data fetch")
    args = parser.parse_args()

    setup_logging()

    config = load_config()
    client = BinanceClient(config)
    strategy = TrendFollowingStrategy(config.strategy)

    print(f"\nRunning backtest: {args.days} days | ${args.equity:,.0f} starting equity")
    print(f"Symbol: {config.strategy.symbol} | Timeframe: {config.strategy.timeframe}")
    print(f"Strategy: {strategy.name()}\n")

    report = run_backtest(
        config=config,
        strategy=strategy,
        client=client,
        days=args.days,
        start_equity=args.equity,
        use_cache=not args.no_cache,
    )

    report.print()


if __name__ == "__main__":
    main()
