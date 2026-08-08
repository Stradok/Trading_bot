"""Owns the set of Lanes and decides how many should be active based on
current equity (see gold_scalper/config.py: lanes.lane_equity_steps) --
scaling concurrency, never per-trade size, and only within max_lanes.
"""

from __future__ import annotations

import pandas as pd

from gold_scalper.config import Config

from .circuit_breaker import CircuitBreaker
from .execution import ExecutionSimulator
from .lane import Lane, LaneStatus
from .portfolio import Portfolio


class LaneManager:
    def __init__(self, cfg: Config, execution: ExecutionSimulator, portfolio: Portfolio):
        self.cfg = cfg
        self.execution = execution
        self.portfolio = portfolio
        self.circuit_breaker = CircuitBreaker(cfg.circuit_breaker)
        self.lanes: list[Lane] = [Lane(0)]

    def desired_lane_count(self, equity: float) -> int:
        """lane_equity_steps thresholds are growth *multiples* of starting
        capital (e.g. 1.5 = equity has grown to 1.5x what we started with),
        not absolute dollar amounts -- otherwise a lane count would depend on
        how much capital you happened to start with rather than on profit
        actually earned, which contradicts the whole point of scaling lanes
        as a reward for accumulated profit (see Idea.md).
        """
        starting_cash = self.portfolio.starting_cash
        growth = equity / starting_cash if starting_cash > 0 else 1.0
        n = 1
        for threshold, count in self.cfg.lanes.lane_equity_steps:
            if growth >= threshold:
                n = count
        return min(n, self.cfg.lanes.max_lanes)

    def on_bar(self, decision_row: pd.Series, fill_row: pd.Series | None = None) -> None:
        """See Lane.on_bar for the decision_row vs fill_row distinction. Defaults
        fill_row to decision_row for live/paper trading; the backtester passes
        the next bar explicitly to avoid same-bar lookahead.
        """
        self.portfolio.mark_to_market(decision_row["close"], decision_row.name)

        if self.circuit_breaker.tripped(decision_row, self.portfolio):
            if self.cfg.circuit_breaker.flatten_on_trip:
                for lane in self.lanes:
                    lane.force_close(decision_row, self.cfg, self.execution, self.portfolio, fill_row=fill_row)
            return  # no new entries while tripped, regardless of flatten_on_trip

        target = self.desired_lane_count(self.portfolio.equity)
        while len(self.lanes) < target:
            self.lanes.append(Lane(len(self.lanes)))

        for lane in self.lanes[:target]:
            lane.on_bar(decision_row, self.cfg, self.execution, self.portfolio, fill_row=fill_row)
        # Lanes beyond `target` (equity dropped since they were created) are
        # left alone to exit via their own trailing/stop/timeout logic rather
        # than being force-closed -- avoids whipsawing a lane shut right as
        # equity dips and then having to reopen it a bar later.

    @property
    def open_lane_count(self) -> int:
        return sum(1 for lane in self.lanes if lane.status is LaneStatus.IN_POSITION)
