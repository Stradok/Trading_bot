"""Event-driven backtest loop over stored historical bars.

Enforces no-lookahead: a decision evaluated on bar t's data fills against
bar t+1's open/spread, never bar t's own close (see gold_scalper/engine/lane.py
for why -- otherwise the backtest trades on information it doesn't have yet).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from gold_scalper.config import Config
from gold_scalper.engine.execution import ExecutionSimulator
from gold_scalper.engine.lane_manager import LaneManager
from gold_scalper.engine.portfolio import Portfolio
from gold_scalper.strategy.features import compute_all_features


@dataclass
class BacktestResult:
    portfolio: Portfolio
    manager: LaneManager
    lane_count_curve: list[tuple[pd.Timestamp, int]] = field(default_factory=list)


def run_backtest(bars: pd.DataFrame, cfg: Config, starting_cash: float = 200.0) -> BacktestResult:
    """bars must be indexed by a UTC DatetimeIndex (see store.to_datetime_indexed)
    and cover [cfg.backtest.start, cfg.backtest.end] already if that filtering
    is wanted -- this function runs over whatever it's given.
    """
    bars = compute_all_features(bars, cfg)
    portfolio = Portfolio(starting_cash)
    manager = LaneManager(cfg, ExecutionSimulator(cfg.execution), portfolio)
    lane_count_curve: list[tuple[pd.Timestamp, int]] = []

    for i in range(len(bars) - 1):  # last bar has no "next bar" to fill against
        decision_row = bars.iloc[i]
        fill_row = bars.iloc[i + 1]
        manager.on_bar(decision_row, fill_row=fill_row)
        lane_count_curve.append((decision_row.name, manager.open_lane_count))

    return BacktestResult(portfolio=portfolio, manager=manager, lane_count_curve=lane_count_curve)
