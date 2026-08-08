#!/usr/bin/env python3
"""Poll the live order book to build a real bid/ask spread distribution,
then (optionally) write the resulting percentile into config/default.yaml.

This needs to run for a meaningful window (hours to days) to be trustworthy --
a 30-second smoke test is enough to prove the pipeline works, not enough to
calibrate a real threshold. Safe to re-run repeatedly; samples accumulate.

Usage:
    uv run scripts/calibrate_spread.py --duration-seconds 30          # smoke test
    uv run scripts/calibrate_spread.py --duration-seconds 86400 --write-config  # full day, then persist p75
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

from gold_scalper.config import load_config
from gold_scalper.data import spread_calibration

SPREAD_SAMPLES_PATH = "data/spread_calibration.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=10.0)
    parser.add_argument("--write-config", action="store_true",
                         help="write the resulting p75 into config/default.yaml")
    args = parser.parse_args()

    cfg = load_config("config/default.yaml", require_calibrated_spread=False)

    def on_sample(sample: dict) -> None:
        print(f"\r  bid={sample['best_bid']:.4f} ask={sample['best_ask']:.4f} "
              f"spread_bps={sample['spread_bps']:.2f}", end="", flush=True)

    print(f"Polling {cfg.exchange.symbol} order book on {cfg.exchange.id} "
          f"every {args.poll_interval_seconds}s for {args.duration_seconds}s...")
    spread_calibration.run_calibration(
        cfg.exchange.id, cfg.exchange.symbol, SPREAD_SAMPLES_PATH,
        duration_seconds=args.duration_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        sample_callback=on_sample,
    )
    print()

    p75 = spread_calibration.compute_percentile(SPREAD_SAMPLES_PATH, percentile=75)
    n_samples = len(pd.read_parquet(SPREAD_SAMPLES_PATH))
    print(f"spread_bps p75 = {p75:.2f} (from {n_samples} total samples)")

    if args.write_config:
        with open("config/default.yaml") as f:
            raw = yaml.safe_load(f)
        raw["spread"]["spread_bps_p75"] = round(p75, 4)
        with open("config/default.yaml", "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
        print("Wrote spread.spread_bps_p75 into config/default.yaml")
    else:
        print("Re-run with --write-config once you're happy with the sample size to persist this value.")


if __name__ == "__main__":
    main()
