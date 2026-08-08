"""Order-book polling to build a real spread distribution for the symbol.

The exit rule and fee model both need a realistic spread number instead of a
guessed constant (see gold_scalper/config.py: spread.spread_bps_p75). This
module builds that distribution by sampling the live order book over time;
scripts/calibrate_spread.py is the CLI entry point that actually runs it for
a multi-day window and writes the resulting p75 back into config.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import pandas as pd

from .exchange import make_exchange

SampleCallback = Callable[[dict], None]


def poll_order_book_once(exchange, symbol: str, depth: int = 5) -> dict | None:
    order_book = exchange.fetch_order_book(symbol, limit=depth)
    bids, asks = order_book.get("bids") or [], order_book.get("asks") or []
    if not bids or not asks:
        return None
    best_bid, best_ask = bids[0][0], asks[0][0]
    mid = (best_bid + best_ask) / 2
    return {
        "timestamp": exchange.milliseconds(),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread_bps": (best_ask - best_bid) / mid * 10_000,
    }


def _append_samples(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    df.to_parquet(path, index=False)
    return df


def run_calibration(
    exchange_id: str,
    symbol: str,
    path: str,
    duration_seconds: float,
    poll_interval_seconds: float = 10.0,
    sample_callback: SampleCallback | None = None,
) -> pd.DataFrame:
    """Poll the order book every `poll_interval_seconds` for `duration_seconds`,
    appending samples to `path`. Intended to be run for days, not minutes --
    call repeatedly (e.g. from cron) to build up a real distribution over time.
    """
    exchange = make_exchange(exchange_id)
    exchange.load_markets()

    rows: list[dict] = []
    deadline = time.monotonic() + duration_seconds
    while time.monotonic() < deadline:
        sample = poll_order_book_once(exchange, symbol)
        if sample:
            rows.append(sample)
            if sample_callback:
                sample_callback(sample)
        time.sleep(poll_interval_seconds)

    df = pd.DataFrame(rows)
    if df.empty:
        existing = pd.read_parquet(path) if Path(path).exists() else pd.DataFrame()
        return existing
    return _append_samples(df, path)


def compute_percentile(path: str, percentile: float = 75) -> float:
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"No spread samples found at {path}")
    return float(df["spread_bps"].quantile(percentile / 100))
