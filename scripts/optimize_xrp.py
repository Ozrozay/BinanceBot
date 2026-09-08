"""
XRP-specific parameter sweep — two staged passes.

Stage 1: Find best timeframe + RSI + ATR stop/trail (fast, 36 combos)
Stage 2: Take top 5 from Stage 1, add ADX + volume filter (90 combos)

Uses trailing stop (no fixed TP) throughout.
All other architecture unchanged.
"""

import itertools
import sys
import dataclasses
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.disable(logging.INFO)

import numpy as np
import pandas as pd
from ta.trend import ADXIndicator

from src.config import load_config, StrategyConfig
from src.api.client import BinanceClient
from src.strategy.trend_following import TrendFollowingStrategy
from src.signals.signal import Direction, Signal, SignalType
from src.backtest.runner import run_backtest, BacktestReport
from src.backtest.wrapper import BacktestStrategyWrapper

DAYS = 365


# ------------------------------------------------------------------
# Extended strategy: adds ADX + volume filter on top of base strategy
# ------------------------------------------------------------------

class XRPStrategy(TrendFollowingStrategy):
    def __init__(
        self,
        config: StrategyConfig,
        rsi_threshold: float = 45.0,
        adx_threshold: float = 0.0,
        volume_filter: float = 0.0,
    ) -> None:
        super().__init__(config)
        self.rsi_threshold = rsi_threshold
        self.adx_threshold = adx_threshold
        self.volume_filter = volume_filter

    def compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = super().compute_indicators(data)
        min_rows = max(self.config.ema_period, 30) + 5

        if len(df) >= min_rows:
            # ADX
            try:
                adx_ind = ADXIndicator(df["high"], df["low"], df["close"], window=14)
                df["adx"] = adx_ind.adx()
            except Exception:
                df["adx"] = np.nan

            # Relative volume (current / 20-bar mean)
            df["rel_volume"] = df["volume"] / df["volume"].rolling(20).mean()
        else:
            df["adx"] = np.nan
            df["rel_volume"] = np.nan

        return df

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        ema_col = f"ema_{self.config.ema_period}"
        rsi_col = f"rsi_{self.config.rsi_period}"
        atr_col = f"atr_{self.config.atr_period}"

        if len(data) < 2:
            return self._hold(data, "not enough bars")

        curr = data.iloc[-1]
        prev = data.iloc[-2]

        for col in (ema_col, rsi_col, atr_col):
            if col not in data.columns or pd.isna(curr[col]):
                return self._hold(data, f"{col} NaN")

        close    = curr["close"]
        ema      = curr[ema_col]
        rsi_now  = curr[rsi_col]
        rsi_prev = prev[rsi_col]
        atr      = curr[atr_col]
        ts       = self._timestamp(curr)
        sym      = self.config.symbol
        thr      = self.rsi_threshold

        # ADX filter
        if self.adx_threshold > 0 and "adx" in data.columns:
            adx_val = curr.get("adx", np.nan)
            if pd.isna(adx_val) or adx_val < self.adx_threshold:
                return self._hold(data, f"ADX {adx_val:.1f} < {self.adx_threshold}")

        # Volume filter
        if self.volume_filter > 0 and "rel_volume" in data.columns:
            rv = curr.get("rel_volume", np.nan)
            if pd.isna(rv) or rv < self.volume_filter:
                return self._hold(data, f"low volume {rv:.2f}x")

        # Exit signals
        if pd.notna(prev[ema_col]) and prev["close"] > prev[ema_col] and close <= ema:
            return Signal(ts, sym, SignalType.EXIT, Direction.FLAT,
                          reason="price crossed below EMA")
        if pd.notna(prev[ema_col]) and prev["close"] < prev[ema_col] and close >= ema:
            return Signal(ts, sym, SignalType.EXIT, Direction.FLAT,
                          reason="price crossed above EMA")

        # Entry signals
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


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def score(r: BacktestReport) -> float:
    if r.total_trades < 10:
        return -999.0
    trade_bonus = min(r.total_trades / 50, 1.0)
    wr_bonus    = max(0, r.win_rate - 0.4) * 2
    dd_penalty  = r.max_drawdown_pct * 2
    return (
        r.profit_factor * 3
        + r.sharpe_ratio * 2
        + wr_bonus
        + trade_bonus
        - dd_penalty
        + r.funding_adjusted_return_pct * 10
    )


def run_combo(config, client, params: dict) -> tuple[float, dict, BacktestReport] | None:
    tf      = params["timeframe"]
    rsi_t   = params["rsi_threshold"]
    sl_m    = params["atr_stop"]
    trail_m = params["atr_trail"]
    adx_t   = params.get("adx_threshold", 0)
    vol_f   = params.get("volume_filter", 0)

    sc = dataclasses.replace(
        config.strategy,
        timeframe=tf,
        atr_stop_multiplier=sl_m,
        atr_trail_multiplier=trail_m,
    )
    cfg = dataclasses.replace(config, strategy=sc)
    strategy = XRPStrategy(sc, rsi_threshold=rsi_t, adx_threshold=adx_t, volume_filter=vol_f)

    BacktestStrategyWrapper.use_trailing_stop = True

    try:
        report = run_backtest(cfg, strategy, client, days=DAYS, use_cache=True)
        s = score(report)
        return s, params, report
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def print_table(results: list, title: str) -> None:
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    hdr = f"{'#':>3}  {'tf':>4}  {'rsi':>4}  {'sl':>4}  {'trail':>5}  {'adx':>4}  {'vol':>4}  "
    hdr += f"{'trades':>6}  {'wr%':>5}  {'PF':>5}  {'sharpe':>6}  {'dd%':>5}  {'ret%':>6}  {'score':>6}"
    print(hdr)
    print("-" * 100)
    for rank, (s, p, r) in enumerate(results[:10], 1):
        print(
            f"{rank:>3}  {p['timeframe']:>4}  {p['rsi_threshold']:>4}  "
            f"{p['atr_stop']:>4}  {p['atr_trail']:>5}  "
            f"{p.get('adx_threshold',0):>4}  {p.get('volume_filter',0):>4}  "
            f"{r.total_trades:>6}  {r.win_rate*100:>5.1f}  {r.profit_factor:>5.2f}  "
            f"{r.sharpe_ratio:>6.2f}  {r.max_drawdown_pct*100:>5.1f}  "
            f"{r.funding_adjusted_return_pct*100:>6.1f}  {s:>6.2f}"
        )
    print("=" * 100)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def sweep():
    config = load_config()
    client = BinanceClient(config)

    # ---- Stage 1: base params ----
    grid1 = {
        "timeframe":     ["1h", "4h"],
        "rsi_threshold": [40, 45, 50],
        "atr_stop":      [1.5, 2.0],
        "atr_trail":     [1.5, 2.0, 3.0],
    }
    combos1 = [dict(zip(grid1.keys(), v)) for v in itertools.product(*grid1.values())]
    print(f"\nStage 1: {len(combos1)} combinations (base params on XRP) — trailing stop ON\n")

    results1 = []
    for i, p in enumerate(combos1, 1):
        print(f"[{i:>2}/{len(combos1)}] tf={p['timeframe']} rsi={p['rsi_threshold']} "
              f"sl={p['atr_stop']}x trail={p['atr_trail']}x ...", end=" ", flush=True)
        out = run_combo(config, client, p)
        if out:
            s, params, r = out
            results1.append(out)
            print(f"trades={r.total_trades} wr={r.win_rate*100:.0f}% "
                  f"pf={r.profit_factor:.2f} sharpe={r.sharpe_ratio:.2f} "
                  f"ret={r.funding_adjusted_return_pct*100:.1f}%")
        else:
            print("skipped")

    results1.sort(key=lambda x: x[0], reverse=True)
    print_table(results1, "STAGE 1 RESULTS — Base Params")

    # ---- Stage 2: ADX + volume on top 5 ----
    top5_params = [p for _, p, _ in results1[:5]]
    grid2 = {
        "adx_threshold": [0, 20, 25],
        "volume_filter": [0, 1.2, 1.5],
    }
    stage2_combos = []
    for base in top5_params:
        for v in itertools.product(*grid2.values()):
            extra = dict(zip(grid2.keys(), v))
            stage2_combos.append({**base, **extra})

    # Skip combos where both filters are off (already tested in stage 1)
    stage2_combos = [c for c in stage2_combos if not (c["adx_threshold"] == 0 and c["volume_filter"] == 0)]

    print(f"\nStage 2: {len(stage2_combos)} combinations (ADX + volume on top 5)\n")

    results2 = []
    for i, p in enumerate(stage2_combos, 1):
        print(f"[{i:>2}/{len(stage2_combos)}] tf={p['timeframe']} rsi={p['rsi_threshold']} "
              f"sl={p['atr_stop']}x trail={p['atr_trail']}x "
              f"adx={p['adx_threshold']} vol={p['volume_filter']}x ...", end=" ", flush=True)
        out = run_combo(config, client, p)
        if out:
            s, params, r = out
            results2.append(out)
            print(f"trades={r.total_trades} wr={r.win_rate*100:.0f}% "
                  f"pf={r.profit_factor:.2f} sharpe={r.sharpe_ratio:.2f} "
                  f"ret={r.funding_adjusted_return_pct*100:.1f}%")
        else:
            print("skipped")

    # Combine and rank all results
    all_results = results1 + results2
    all_results.sort(key=lambda x: x[0], reverse=True)
    print_table(all_results, "FINAL RESULTS — All Combinations Ranked")

    best_score, best_params, best_report = all_results[0]
    print(f"\n{'='*60}")
    print("  WINNING CONFIG FOR XRP")
    print(f"{'='*60}")
    for k, v in best_params.items():
        print(f"  {k:<20} = {v}")
    print()
    best_report.print()

    print(f"\n{'='*60}")
    print("  UPDATE .env WITH:")
    print(f"{'='*60}")
    print(f"  TIMEFRAME={best_params['timeframe']}")
    print(f"  ATR_STOP_MULTIPLIER={best_params['atr_stop']}")
    print(f"  ATR_TRAIL_MULTIPLIER={best_params['atr_trail']}")
    print(f"  RSI_THRESHOLD={best_params['rsi_threshold']}")
    print(f"  ADX_THRESHOLD={best_params.get('adx_threshold', 0)}")
    print(f"  VOLUME_FILTER={best_params.get('volume_filter', 0)}")

    return best_params


if __name__ == "__main__":
    sweep()
