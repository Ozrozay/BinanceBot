"""
Backtest runner and metrics reporter.

Runs the backtest via backtesting.py and prints the full quality-bar report:
  - Total trades, win rate
  - Sharpe ratio
  - Profit factor
  - Max drawdown
  - CAGR
  - Avg win vs avg loss (red flag check)
  - Funding-cost-adjusted equity
"""

import logging
from dataclasses import dataclass

import pandas as pd
from backtesting import Backtest

from src.backtest.funding import compute_funding_cost, fetch_funding_rates
from src.backtest.wrapper import BacktestStrategyWrapper
from src.config import AppConfig
from src.data.historical import fetch_historical_range
from src.strategy.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestReport:
    total_trades: int
    win_rate: float           # 0–1
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float   # 0–1
    cagr: float               # annualised return 0–1
    avg_win_usd: float
    avg_loss_usd: float
    total_return_pct: float
    funding_cost_usd: float
    funding_adjusted_return_pct: float
    start_equity: float
    end_equity: float

    def print(self) -> None:
        sep = "=" * 60
        print(sep)
        print("  BACKTEST RESULTS")
        print(sep)
        print(f"  Total trades      : {self.total_trades}")
        print(f"  Win rate          : {self.win_rate*100:.1f}%  {'⚠ below 40%' if self.win_rate < 0.4 else '✓'}")
        print(f"  Profit factor     : {self.profit_factor:.2f}  {'⚠ <1.0' if self.profit_factor < 1.0 else '✓'}")
        print(f"  Sharpe ratio      : {self.sharpe_ratio:.2f}")
        print(f"  Max drawdown      : {self.max_drawdown_pct*100:.1f}%")
        print(f"  CAGR              : {self.cagr*100:.1f}%")
        print(f"  Avg win           : ${self.avg_win_usd:.2f}")
        print(f"  Avg loss          : ${self.avg_loss_usd:.2f}  {'⚠ loss > win' if abs(self.avg_loss_usd) > self.avg_win_usd else '✓'}")
        print(f"  Total return      : {self.total_return_pct*100:.1f}%")
        print(f"  Funding cost      : ${self.funding_cost_usd:.2f}")
        print(f"  Funding-adj return: {self.funding_adjusted_return_pct*100:.1f}%")
        print(f"  Equity: ${self.start_equity:.2f} → ${self.end_equity:.2f}")
        print(sep)

        if self.total_trades < 50:
            print(f"  ⚠  Only {self.total_trades} trades — need 50+ for statistical significance")
        if self.win_rate < 0.4:
            print("  ⚠  Win rate below 40% — review strategy before proceeding")
        if abs(self.avg_loss_usd) > self.avg_win_usd:
            print("  ⚠  Average loss exceeds average win — poor expectancy")
        if self.total_trades >= 50 and self.win_rate >= 0.4 and self.profit_factor >= 1.0:
            print("  ✓  Passes quality bar — ready for paper trading review")
        print(sep)


def run_backtest(
    config: AppConfig,
    strategy: Strategy,
    client,
    days: int = 180,
    start_equity: float = 10_000.0,
    use_cache: bool = True,
    cache_max_age_hours: int = 24,
) -> BacktestReport:
    """
    Fetch data, compute indicators, run the backtest, return a report.
    """
    from src.data.historical import fetch_historical_range

    # 1. Fetch historical OHLCV
    df = fetch_historical_range(
        client,
        symbol=config.strategy.symbol,
        timeframe=config.strategy.timeframe,
        days=days,
        use_cache=use_cache,
        cache_max_age_hours=cache_max_age_hours,
    )

    logger.info("Data range: %s → %s (%d bars)", df["timestamp"].iloc[0], df["timestamp"].iloc[-1], len(df))

    # 2. Pre-compute indicators on the full dataset
    df_with_indicators = strategy.compute_indicators(df)

    # 3. Fetch funding rates
    try:
        funding_df = fetch_funding_rates(client, config.strategy.symbol, days=days, use_cache=use_cache)
    except Exception as e:
        logger.warning("Could not fetch funding rates (%s) — funding cost will be 0", e)
        funding_df = pd.DataFrame(columns=["timestamp", "funding_rate"])

    # 4. Reformat for backtesting.py (needs Open/High/Low/Close/Volume with DatetimeIndex)
    bt_df = df_with_indicators.copy()
    bt_df = bt_df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    bt_df = bt_df.set_index("timestamp")
    bt_df.index = pd.DatetimeIndex(bt_df.index)

    # 5. Configure the wrapper with our strategy and risk params
    # source_df keeps the lowercase + indicator version for the wrapper's next() slicing
    BacktestStrategyWrapper.source_df = df_with_indicators.reset_index(drop=True)
    BacktestStrategyWrapper.our_strategy = strategy
    BacktestStrategyWrapper.risk_per_trade_pct = config.risk.risk_per_trade_pct
    BacktestStrategyWrapper.max_leverage = float(config.risk.max_leverage)
    BacktestStrategyWrapper.atr_stop_multiplier = config.strategy.atr_stop_multiplier
    BacktestStrategyWrapper.atr_trail_multiplier = config.strategy.atr_trail_multiplier
    BacktestStrategyWrapper.funding_df = funding_df

    # 6. Run
    # FractionalBacktest: scales OHLCV prices by 1/100e6 internally so we can
    # trade in satoshi units (whole integers) while still holding <1 BTC.
    # The wrapper scales sl/tp by the same factor before passing to buy()/sell().
    from backtesting.lib import FractionalBacktest
    bt = FractionalBacktest(
        bt_df,
        BacktestStrategyWrapper,
        cash=start_equity,
        commission=0.0004,        # Binance futures taker fee: 0.04% per side
        exclusive_orders=True,
        finalize_trades=True,     # close any open trades at end so they count in stats
    )

    stats = bt.run()
    logger.info("Backtest complete.")

    # 7. Extract metrics
    trades = stats["_trades"]
    total_trades = len(trades)

    if total_trades == 0:
        logger.warning("No trades generated — strategy produced no signals in this period")
        return BacktestReport(
            total_trades=0, win_rate=0, profit_factor=0, sharpe_ratio=0,
            max_drawdown_pct=0, cagr=0, avg_win_usd=0, avg_loss_usd=0,
            total_return_pct=0, funding_cost_usd=0, funding_adjusted_return_pct=0,
            start_equity=start_equity, end_equity=start_equity,
        )

    wins = trades[trades["PnL"] > 0]["PnL"]
    losses = trades[trades["PnL"] <= 0]["PnL"]

    win_rate = len(wins) / total_trades
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    sharpe = float(stats.get("Sharpe Ratio", 0) or 0)
    max_dd = abs(float(stats.get("Max. Drawdown [%]", 0) or 0)) / 100
    cagr = float(stats.get("Return (Ann.) [%]", 0) or 0) / 100
    total_return_pct = float(stats.get("Return [%]", 0) or 0) / 100
    end_equity = start_equity * (1 + total_return_pct)

    # 8. Funding cost (accumulated in wrapper)
    # backtesting.py doesn't expose the wrapper instance post-run cleanly,
    # so we recompute funding cost from the trades DataFrame directly.
    funding_cost_total = _compute_total_funding_cost(trades, funding_df)
    funding_adj_end_equity = end_equity - funding_cost_total
    funding_adj_return_pct = (funding_adj_end_equity - start_equity) / start_equity

    return BacktestReport(
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        cagr=cagr,
        avg_win_usd=avg_win,
        avg_loss_usd=avg_loss,
        total_return_pct=total_return_pct,
        funding_cost_usd=funding_cost_total,
        funding_adjusted_return_pct=funding_adj_return_pct,
        start_equity=start_equity,
        end_equity=end_equity,
    )


def _compute_total_funding_cost(trades: pd.DataFrame, funding_df: pd.DataFrame) -> float:
    """
    Recompute funding cost from the trades DataFrame after the backtest completes.
    trades has columns: EntryTime, ExitTime, Size, EntryPrice, ExitPrice, Direction
    """
    if funding_df.empty or trades.empty:
        return 0.0

    total = 0.0
    for _, trade in trades.iterrows():
        direction = "long" if trade.get("Size", 0) > 0 else "short"
        notional = abs(float(trade.get("Size", 0))) * float(trade.get("EntryPrice", 0))
        entry_time = pd.Timestamp(trade["EntryTime"])
        exit_time = pd.Timestamp(trade["ExitTime"])
        cost = compute_funding_cost(funding_df, entry_time, exit_time, direction, notional)
        total += cost

    return total
