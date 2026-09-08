"""
Parameter sweep across key strategy settings.
Tests combinations of RSI threshold, ATR multipliers, and timeframe.
Prints a ranked table so we can pick the best config before live trading.
"""

import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.config import load_config, StrategyConfig
from src.logging.structured import setup_logging
from src.api.client import BinanceClient
from src.strategy.trend_following import TrendFollowingStrategy
from src.backtest.runner import run_backtest, BacktestReport

# Suppress INFO noise during sweep — only show the table
import logging
logging.disable(logging.INFO)

PARAM_GRID = {
    "timeframe":            ["1h", "4h"],
    "rsi_entry_threshold":  [45, 50],       # RSI cross level for entry
    "atr_stop_multiplier":  [1.5, 2.0],
    "atr_trail_multiplier": [2.0, 3.0, 4.0],
}

DAYS = 365   # use a full year for sweep so we have more trades per config


class TrendFollowingStrategyParam(TrendFollowingStrategy):
    """Subclass that supports a configurable RSI entry threshold."""

    def __init__(self, config: StrategyConfig, rsi_threshold: float = 50.0) -> None:
        super().__init__(config)
        self.rsi_threshold = rsi_threshold

    def generate_signal(self, data):
        import pandas as pd
        from src.signals.signal import Direction, Signal, SignalType

        ema_col = f"ema_{self.config.ema_period}"
        rsi_col = f"rsi_{self.config.rsi_period}"
        atr_col = f"atr_{self.config.atr_period}"

        if len(data) < 2:
            return self._hold(data, "not enough bars")

        curr = data.iloc[-1]
        prev = data.iloc[-2]

        for col in (ema_col, rsi_col, atr_col):
            if pd.isna(curr[col]):
                return self._hold(data, f"{col} NaN")

        close = curr["close"]
        ema   = curr[ema_col]
        rsi_now  = curr[rsi_col]
        rsi_prev = prev[rsi_col]
        atr   = curr[atr_col]
        ts    = self._timestamp(curr)
        sym   = self.config.symbol
        thr   = self.rsi_threshold

        # Exit signals
        if pd.notna(prev[ema_col]) and prev["close"] > prev[ema_col] and close <= ema:
            return Signal(ts, sym, SignalType.EXIT, Direction.FLAT,
                          reason="price crossed below EMA")
        if pd.notna(prev[ema_col]) and prev["close"] < prev[ema_col] and close >= ema:
            return Signal(ts, sym, SignalType.EXIT, Direction.FLAT,
                          reason="price crossed above EMA")

        # Entry signals with configurable threshold
        if close > ema and rsi_now > thr and pd.notna(rsi_prev) and rsi_prev <= thr:
            stop = close - atr * self.config.atr_stop_multiplier
            tp   = close + atr * self.config.atr_trail_multiplier
            return Signal(ts, sym, SignalType.ENTRY, Direction.LONG,
                          entry_price=close, suggested_stop=stop, suggested_tp=tp,
                          atr=atr, reason=f"RSI>{thr}")

        if close < ema and rsi_now < thr and pd.notna(rsi_prev) and rsi_prev >= thr:
            stop = close + atr * self.config.atr_stop_multiplier
            tp   = close - atr * self.config.atr_trail_multiplier
            return Signal(ts, sym, SignalType.ENTRY, Direction.SHORT,
                          entry_price=close, suggested_stop=stop, suggested_tp=tp,
                          atr=atr, reason=f"RSI<{thr}")

        return self._hold(data, "no trigger")


def sweep():
    config = load_config()
    client = BinanceClient(config)

    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    total = len(combos)

    print(f"\nRunning {total} parameter combinations over {DAYS} days of BTC/USDT data...\n")

    results = []
    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        tf     = params["timeframe"]
        rsi_t  = params["rsi_entry_threshold"]
        sl_m   = params["atr_stop_multiplier"]
        tp_m   = params["atr_trail_multiplier"]

        print(f"[{i:>2}/{total}] tf={tf} rsi_thr={rsi_t} sl={sl_m}x tp={tp_m}x ...", end=" ", flush=True)

        # Build a config with this timeframe + ATR multipliers
        import dataclasses
        sc = dataclasses.replace(
            config.strategy,
            timeframe=tf,
            atr_stop_multiplier=sl_m,
            atr_trail_multiplier=tp_m,
        )
        import dataclasses as dc
        cfg = dc.replace(config, strategy=sc)

        strategy = TrendFollowingStrategyParam(sc, rsi_threshold=rsi_t)

        try:
            report = run_backtest(
                config=cfg,
                strategy=strategy,
                client=client,
                days=DAYS,
                start_equity=10_000.0,
                use_cache=True,
            )
            score = _score(report)
            results.append((score, params, report))
            print(f"trades={report.total_trades} wr={report.win_rate*100:.0f}% "
                  f"pf={report.profit_factor:.2f} sharpe={report.sharpe_ratio:.2f} "
                  f"ret={report.funding_adjusted_return_pct*100:.1f}%")
        except Exception as e:
            print(f"ERROR: {e}")

    # Rank by score
    results.sort(key=lambda x: x[0], reverse=True)

    print("\n" + "=" * 90)
    print(f"  TOP RESULTS (ranked by composite score)")
    print("=" * 90)
    hdr = f"{'#':>3}  {'tf':>4}  {'rsi':>4}  {'sl':>4}  {'tp':>4}  "
    hdr += f"{'trades':>6}  {'wr%':>5}  {'PF':>5}  {'sharpe':>6}  {'dd%':>5}  {'ret%':>6}  {'score':>6}"
    print(hdr)
    print("-" * 90)
    for rank, (score, params, r) in enumerate(results[:10], 1):
        print(
            f"{rank:>3}  {params['timeframe']:>4}  {params['rsi_entry_threshold']:>4}  "
            f"{params['atr_stop_multiplier']:>4}  {params['atr_trail_multiplier']:>4}  "
            f"{r.total_trades:>6}  {r.win_rate*100:>5.1f}  {r.profit_factor:>5.2f}  "
            f"{r.sharpe_ratio:>6.2f}  {r.max_drawdown_pct*100:>5.1f}  "
            f"{r.funding_adjusted_return_pct*100:>6.1f}  {score:>6.2f}"
        )
    print("=" * 90)

    if results:
        best_score, best_params, best_report = results[0]
        print(f"\nBEST CONFIG:")
        for k, v in best_params.items():
            print(f"  {k} = {v}")
        print()
        best_report.print()

        print("\nTo use this config, update .env:")
        print(f"  TIMEFRAME={best_params['timeframe']}")
        print(f"  ATR_STOP_MULTIPLIER={best_params['atr_stop_multiplier']}")
        print(f"  ATR_TRAIL_MULTIPLIER={best_params['atr_trail_multiplier']}")
        print(f"  # RSI threshold requires code change in .env — see StrategyConfig")


def _score(r: BacktestReport) -> float:
    """
    Composite score that rewards all quality-bar dimensions:
    - Profit factor (most important — does the strategy make money?)
    - Sharpe ratio (risk-adjusted)
    - Win rate (above 40% preferred)
    - Trade count (penalise configs that barely trigger)
    - Drawdown penalty
    """
    if r.total_trades < 10:
        return -999.0
    trade_bonus = min(r.total_trades / 50, 1.0)   # max out at 50 trades
    wr_bonus    = max(0, r.win_rate - 0.4) * 2    # bonus for win rate above 40%
    dd_penalty  = r.max_drawdown_pct * 2
    return (
        r.profit_factor * 3
        + r.sharpe_ratio * 2
        + wr_bonus
        + trade_bonus
        - dd_penalty
        + r.funding_adjusted_return_pct * 10
    )


if __name__ == "__main__":
    sweep()
