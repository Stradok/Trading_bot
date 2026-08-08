"""Entry signal: ATR-scaled breakout of a recent range, confirmed by a
positive short-EMA slope. Pure functions, no I/O -- same code path is used
by the backtester, the walk-forward optimizer, and the paper trading runner.
"""

from __future__ import annotations

import pandas as pd

from gold_scalper.config import SignalConfig


def true_range(bars: pd.DataFrame) -> pd.Series:
    prev_close = bars["close"].shift(1)
    ranges = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def compute_features(bars: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    bars = bars.copy()
    bars["ema_fast"] = bars["close"].ewm(span=cfg.ema_fast_periods, adjust=False).mean()
    # shift(1) before rolling: the breakout level must exclude the current
    # bar's own high, otherwise close <= high <= roll_high always holds and
    # the breakout condition can never fire.
    bars["roll_high"] = bars["high"].shift(1).rolling(cfg.breakout_lookback).max()
    bars["atr"] = true_range(bars).ewm(span=cfg.atr_periods, adjust=False).mean()
    bars["ema_slope"] = bars["ema_fast"].diff(cfg.slope_lookback)
    return bars


def entry_signal(row: pd.Series, cfg: SignalConfig) -> bool:
    if pd.isna(row["roll_high"]) or pd.isna(row["atr"]) or pd.isna(row["ema_slope"]):
        return False  # not enough warmup history yet
    buffer = cfg.breakout_atr_mult * row["atr"]
    return bool(row["close"] > row["roll_high"] + buffer and row["ema_slope"] > cfg.min_slope)
