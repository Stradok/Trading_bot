#!/usr/bin/env python3
"""Fetch/update raw trade-level data (with taker side) for order-flow features.

Usage:
    uv run scripts/fetch_trades.py --window-days 90

Safe to re-run: resumes from the last stored trade instead of redownloading
everything (see gold_scalper/data/fetch.py:fetch_recent_trades).
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
    parser.add_argument("--window-days", type=int, default=90)
    args = parser.parse_args()

    cfg = load_config("config/default.yaml", "config/backtest.yaml", require_calibrated_spread=False)
    if not cfg.data.trades_path:
        raise SystemExit("config.data.trades_path is not set")

    def progress(count: int, last_ts) -> None:
        print(f"\r  fetched {count} trades, up to {last_ts}", end="", flush=True)

    print(f"Fetching {cfg.exchange.symbol} trades ({args.window_days}d window) from {cfg.exchange.id}...")
    df = fetch.fetch_recent_trades(
        cfg.exchange.id, cfg.exchange.symbol, cfg.data.trades_path,
        window_days=args.window_days, progress_callback=progress,
    )

    print()
    if df is None or df.empty:
        print("No data stored -- check symbol/exchange availability.")
        return
    print(f"Stored {len(df)} trades, {df['timestamp'].min()} .. {df['timestamp'].max()} (ms epoch)")


if __name__ == "__main__":
    main()
