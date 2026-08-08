"""Single entry point for building the full feature set used by every
consumer (backtester, walk-forward optimizer, paper runner) -- signal
features, circuit-breaker baseline features, and the spread column all come
from here so nothing computes its own divergent version.
"""

from __future__ import annotations

import pandas as pd

from gold_scalper.config import Config

from .signals import compute_features


def compute_all_features(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    bars = compute_features(bars, cfg.signal)

    window = cfg.circuit_breaker.baseline_window_bars
    bars["atr_baseline"] = bars["atr"].rolling(window).mean()
    bars["volume_baseline"] = bars["volume"].rolling(window).mean()

    # Spread is stored in price units (dollars), derived from the calibrated
    # bps figure -- everything downstream (exits, execution) expects dollars.
    bars["spread"] = bars["close"] * cfg.spread.spread_bps_p75 / 10_000

    return bars
