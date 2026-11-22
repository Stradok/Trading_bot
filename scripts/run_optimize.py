#!/usr/bin/env python3
"""Walk-forward parameter search over stored 1m history, then a profit
projection (24/7 server vs. a configurable local-hours MacBook run) built
from the out-of-sample results only.

Usage:
    uv run scripts/run_optimize.py
    uv run scripts/run_optimize.py --train-days 180 --test-days 60 --local-start-hour 8 --local-end-hour 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from gold_scalper.backtest.optimize import summarize_out_of_sample, walk_forward_optimize
from gold_scalper.backtest.projection import (
    PROJECTION_DISCLAIMER, format_projection, project_24_7, project_active_hours,
)
from gold_scalper.config import load_config
from gold_scalper.data import store

# Small, targeted grid over the parameters most responsible for the noise-vs-signal
# tradeoff (breakout threshold, exit arm buffer, trail distance) -- kept small
# because cost scales with len(grid) * n_windows * backtest_runtime.
DEFAULT_PARAM_GRID = {
    "signal.breakout_atr_mult": [0.2, 0.35, 0.5],
    "exit.min_net_profit_pct": [0.0003, 0.0005, 0.001],
    "exit.trail_atr_mult": [0.3, 0.5, 0.8],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--starting-cash", type=float, default=10.0)
    parser.add_argument("--local-start-hour", type=int, default=8)
    parser.add_argument("--local-end-hour", type=int, default=24)
    parser.add_argument("--local-timezone", default="UTC")
    parser.add_argument("--out-dir", default="data/reports")
    args = parser.parse_args()

    cfg = load_config("config/default.yaml", "config/backtest.yaml")
    raw = store.load_ohlcv(cfg.data.ohlcv_1m_path)
    if raw is None or raw.empty:
        print(f"No data at {cfg.data.ohlcv_1m_path} -- run scripts/fetch_history.py first.")
        return
    bars = store.to_datetime_indexed(raw)
    print(f"Loaded {len(bars)} bars, {bars.index[0]} .. {bars.index[-1]}")

    print(f"\nRunning walk-forward optimization (train={args.train_days}d, test={args.test_days}d, "
          f"grid={len(DEFAULT_PARAM_GRID['signal.breakout_atr_mult']) * len(DEFAULT_PARAM_GRID['exit.min_net_profit_pct']) * len(DEFAULT_PARAM_GRID['exit.trail_atr_mult'])} combos)...")
    results = walk_forward_optimize(
        bars, cfg, DEFAULT_PARAM_GRID,
        train_days=args.train_days, test_days=args.test_days,
        starting_cash=args.starting_cash,
    )

    if not results:
        print("No windows fit inside the available history -- reduce --train-days/--test-days.")
        return

    print(f"\n{len(results)} walk-forward windows evaluated.")
    rows = []
    for r in results:
        print(f"  test [{r.test_start.date()} .. {r.test_end.date()}): "
              f"{r.best_params} -> {r.test_metrics.trade_count} trades, "
              f"expectancy_net={r.test_metrics.expectancy_net:.4f}, "
              f"net_profit_per_day={r.test_metrics.net_profit_per_day:.4f}")
        rows.append({
            "test_start": r.test_start, "test_end": r.test_end, **r.best_params,
            "trade_count": r.test_metrics.trade_count,
            "expectancy_net": r.test_metrics.expectancy_net,
            "net_profit_per_day": r.test_metrics.net_profit_per_day,
            "max_drawdown_pct": r.test_metrics.max_drawdown_pct,
        })

    summary = summarize_out_of_sample(results)
    print("\nOut-of-sample summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print(format_projection(project_24_7(results)))
    print()
    print(format_projection(project_active_hours(
        results, args.local_start_hour, args.local_end_hour, args.local_timezone
    )))
    print("=" * 60)
    print(PROJECTION_DISCLAIMER)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "walk_forward_windows.csv", index=False)
    print(f"\nWrote {out_dir}/walk_forward_windows.csv")


if __name__ == "__main__":
    main()
