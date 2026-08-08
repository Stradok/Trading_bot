#!/usr/bin/env python3
"""Run paper trading against live Binance data. No real orders are ever
placed -- see gold_scalper/live/paper_runner.py. Runs until interrupted
(Ctrl-C), then prints a summary of whatever trades happened.

Usage:
    uv run scripts/run_paper.py
    uv run scripts/run_paper.py --starting-cash 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gold_scalper.backtest.metrics import compute_metrics
from gold_scalper.backtest.report import format_summary
from gold_scalper.backtest.runner import BacktestResult
from gold_scalper.config import load_config
from gold_scalper.live.paper_runner import PaperRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starting-cash", type=float, default=150.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config("config/default.yaml", "config/paper.yaml")
    runner = PaperRunner(cfg, starting_cash=args.starting_cash)

    try:
        runner.run_forever()
    except KeyboardInterrupt:
        print("\nStopped.\n")
        # lane_count_curve isn't tracked live (see engine/lane_manager.py's
        # open_lane_count for the current-moment equivalent); metrics that
        # depend on it (concurrent-lane drawdown, lane utilization) will read
        # as empty here -- everything trade-log-based is still accurate.
        result = BacktestResult(portfolio=runner.portfolio, manager=runner.manager, lane_count_curve=[])
        print(format_summary(compute_metrics(result)))


if __name__ == "__main__":
    main()
