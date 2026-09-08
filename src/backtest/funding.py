"""
Funding rate loader for backtesting cost modeling.

Binance perpetual futures charge funding every 8 hours (00:00, 08:00, 16:00 UTC).
If we hold a long position and the rate is positive, we pay; if negative, we receive.
The opposite applies for short positions.

Skipping this in backtests overstates returns for strategies that hold positions
across multiple funding periods — which trend-following absolutely does.
"""

import logging
import time
from pathlib import Path

import pandas as pd

from src.api.client import BinanceClient

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data_cache")
FUNDING_INTERVAL_HOURS = 8


def fetch_funding_rates(
    client: BinanceClient,
    symbol: str,
    days: int = 180,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch funding rate history for `symbol` over the last `days` days.

    Returns a DataFrame with columns:
      timestamp (UTC datetime), funding_rate (float, e.g. 0.0001 = 0.01%)
    Sorted ascending.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    slug = symbol.replace("/", "-").replace(":", "-")
    cache_file = CACHE_DIR / f"funding_{slug}_{days}d.parquet"

    if use_cache and cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24:
            logger.info("Loading funding rates from cache: %s", cache_file)
            return pd.read_parquet(cache_file)

    logger.info("Fetching funding rate history: %s (%d days)...", symbol, days)

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 3600 * 1000

    all_rates: list[dict] = []
    since_ms = start_ms

    while True:
        batch = client.fetch_funding_rate_history(symbol, since=since_ms, limit=1000)
        if not batch:
            break
        all_rates.extend(batch)
        last_ts = batch[-1]["timestamp"]
        if last_ts <= since_ms or len(batch) < 1000:
            break
        since_ms = last_ts + 1
        time.sleep(0.2)

    if not all_rates:
        logger.warning("No funding rate history returned for %s — funding costs will be zero", symbol)
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    df = pd.DataFrame(all_rates)
    df = df[["timestamp", "fundingRate"]].rename(columns={"fundingRate": "funding_rate"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    logger.info("Fetched %d funding rate records", len(df))
    df.to_parquet(cache_file)
    return df


def compute_funding_cost(
    funding_df: pd.DataFrame,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    direction: str,       # "long" or "short"
    position_notional: float,  # size * avg_price in USDT
) -> float:
    """
    Calculate total funding cost (positive = cost paid, negative = received)
    for a position held from entry_time to exit_time.

    For a long: you pay when rate > 0, receive when rate < 0.
    For a short: the opposite.
    """
    if funding_df.empty:
        return 0.0

    mask = (funding_df["timestamp"] > entry_time) & (funding_df["timestamp"] <= exit_time)
    applicable = funding_df.loc[mask, "funding_rate"]

    total_rate = applicable.sum()

    if direction == "short":
        total_rate = -total_rate  # short pays when rate is negative

    return position_notional * total_rate
