#!/usr/bin/env python3
"""Run the backtester over stored 1m OHLCV history and print a metrics report.

Usage:
    uv run scripts/run_backtest.py
    uv run scripts/run_backtest.py --starting-cash 100 --out-dir data/reports
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gold_scalper.backtest.metrics import compute_metrics
from gold_scalper.backtest.report import format_summary, plot_equity_curve, write_trade_csv
from gold_scalper.backtest.runner import run_backtest
from gold_scalper.config import load_config
from gold_scalper.data import store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starting-cash", type=float, default=10.0)
    parser.add_argument("--out-dir", default="data/reports")
    args = parser.parse_args()

    cfg = load_config("config/default.yaml", "config/backtest.yaml")
    raw = store.load_ohlcv(cfg.data.ohlcv_1m_path)
    if raw is None or raw.empty:
        print(f"No data at {cfg.data.ohlcv_1m_path} -- run scripts/fetch_history.py first.")
        return
    bars = store.to_datetime_indexed(raw)

    if cfg.backtest.start:
        bars = bars.loc[bars.index >= cfg.backtest.start]
    if cfg.backtest.end:
        bars = bars.loc[bars.index <= cfg.backtest.end]

    print(f"Running backtest over {len(bars)} bars, {bars.index[0]} .. {bars.index[-1]}...")
    result = run_backtest(bars, cfg, starting_cash=args.starting_cash)
    metrics = compute_metrics(result)

    print()
    print(format_summary(metrics))

    out_dir = Path(args.out_dir)
    write_trade_csv(result, str(out_dir / "trades.csv"))
    plot_equity_curve(result, str(out_dir / "equity_curve.png"))
    print(f"\nWrote {out_dir}/trades.csv and {out_dir}/equity_curve.png")


if __name__ == "__main__":
    main()
