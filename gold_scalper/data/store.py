"""Parquet-backed storage for OHLCV bars, keyed on a UTC timestamp column.

Kept deliberately dumb: read whole file, merge, dedupe, write whole file.
At the data volumes here (single symbol, 1m/1s bars over a few years) that's
well within what pandas/parquet handles instantly -- no need for a real
time-series database.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def load_ohlcv(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df.sort_values("timestamp").reset_index(drop=True)


def save_ohlcv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    df = df.reset_index(drop=True)
    df.to_parquet(path, index=False)


def append_ohlcv(new_df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Merge new_df into whatever is already stored at path, dedupe, save, return the full result."""
    existing = load_ohlcv(path)
    combined = new_df if existing is None else pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    combined = combined.reset_index(drop=True)
    save_ohlcv(combined, path)
    return combined


def last_timestamp_ms(path: str | Path) -> int | None:
    df = load_ohlcv(path)
    if df is None or df.empty:
        return None
    return int(df["timestamp"].iloc[-1])


def to_datetime_indexed(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the millisecond-epoch 'timestamp' column into a UTC DatetimeIndex,
    which is what the engine (Lane/Portfolio/etc, all of which use row.name as
    the event time) expects.
    """
    df = df.copy()
    df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.index.name = "timestamp"
    return df.drop(columns=["timestamp"])
