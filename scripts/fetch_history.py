#!/usr/bin/env python3
"""Fetch/update historical OHLCV data for the configured symbol.

Usage:
    uv run scripts/fetch_history.py --timeframe 1m
    uv run scripts/fetch_history.py --timeframe 1s --window-days 60

Safe to re-run: resumes from the last stored timestamp instead of
redownloading everything (see gold_scalper/data/fetch.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gold_scalper.config import load_config
from gold_scalper.data import fetch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframe", choices=["1m", "1s"], default="1m")
    parser.add_argument("--window-days", type=int, default=60, help="only used for --timeframe 1s")
    args = parser.parse_args()

    cfg = load_config("config/default.yaml", "config/backtest.yaml", require_calibrated_spread=False)

    def progress(count: int, last_ts) -> None:
        print(f"\r  fetched {count} rows, up to {last_ts}", end="", flush=True)

    if args.timeframe == "1m":
        print(f"Fetching {cfg.exchange.symbol} 1m OHLCV from {cfg.exchange.id}...")
        df = fetch.fetch_ohlcv_history(
            cfg.exchange.id, cfg.exchange.symbol, "1m", cfg.data.ohlcv_1m_path,
            progress_callback=progress,
        )
    else:
        print(f"Fetching {cfg.exchange.symbol} 1s bars ({args.window_days}d window) from {cfg.exchange.id}...")
        df = fetch.fetch_recent_second_bars(
            cfg.exchange.id, cfg.exchange.symbol, cfg.data.ohlcv_1s_path,
            window_days=args.window_days, progress_callback=progress,
        )

    print()
    if df is None or df.empty:
        print("No data stored -- check symbol/exchange availability.")
        return
    print(f"Stored {len(df)} rows, {df['timestamp'].min()} .. {df['timestamp'].max()} (ms epoch)")


if __name__ == "__main__":
    main()
