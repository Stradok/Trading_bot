"""A Lane is one independent watch/enter/exit/cooldown cycle. Concurrency is
achieved by running several Lanes in parallel (see lane_manager.py), each
trading a fixed position size -- scaling happens by adding lanes, never by
upsizing an individual trade.
"""

from __future__ import annotations

from enum import Enum, auto

import pandas as pd

from gold_scalper.config import Config
from gold_scalper.strategy.exits import TrailingExit
from gold_scalper.strategy.signals import entry_signal

from .execution import ExecutionSimulator
from .portfolio import Portfolio


class LaneStatus(Enum):
    WATCHING = auto()
    IN_POSITION = auto()
    COOLDOWN = auto()


class Lane:
    def __init__(self, lane_id: int):
        self.id = lane_id
        self.status = LaneStatus.WATCHING
        self.exit_model: TrailingExit | None = None
        self.cooldown_until: pd.Timestamp | None = None

    def on_bar(
        self,
        decision_row: pd.Series,
        cfg: Config,
        execution: ExecutionSimulator,
        portfolio: Portfolio,
        fill_row: pd.Series | None = None,
    ) -> None:
        """decision_row is what the signal/exit rule evaluates ("what we knew
        when we decided"); fill_row is what the order actually executes
        against ("what price we got"). In live/paper trading these collapse
        to the same row (no future bar exists yet). In backtesting, the
        runner passes bar t as decision_row and bar t+1 as fill_row so a
        decision never fills against information it couldn't have had --
        see backtest/runner.py.
        """
        fill_row = decision_row if fill_row is None else fill_row

        if self.status is LaneStatus.WATCHING:
            if entry_signal(decision_row, cfg.signal) and self._can_afford(cfg, portfolio):
                qty = cfg.lanes.position_size_usd / fill_row["open"]
                fill = execution.market_buy(fill_row["open"], fill_row["spread"], qty, fill_row.name)
                portfolio.apply_buy(self.id, fill)
                self.exit_model = TrailingExit(fill.price, fill_row.name, cfg.exit)
                self.status = LaneStatus.IN_POSITION

        elif self.status is LaneStatus.IN_POSITION:
            assert self.exit_model is not None
            reason = self.exit_model.update(
                decision_row["close"], decision_row["atr"], decision_row["spread"], decision_row.name
            )
            if reason:
                self._close(fill_row, cfg, execution, portfolio, reason)

        elif self.status is LaneStatus.COOLDOWN:
            if self.cooldown_until is not None and decision_row.name >= self.cooldown_until:
                self.status = LaneStatus.WATCHING

    def force_close(
        self,
        decision_row: pd.Series,
        cfg: Config,
        execution: ExecutionSimulator,
        portfolio: Portfolio,
        fill_row: pd.Series | None = None,
    ) -> None:
        """Used by the circuit breaker to flatten an open position outside the normal exit rule."""
        if self.status is LaneStatus.IN_POSITION:
            self._close(decision_row if fill_row is None else fill_row, cfg, execution, portfolio, "circuit_breaker")

    @staticmethod
    def _can_afford(cfg: Config, portfolio: Portfolio) -> bool:
        """Skip the entry rather than silently letting cash go negative --
        matters most at small starting capital (e.g. $10), where a fixed
        position_size_usd across several lanes can easily exceed what's
        actually available. Buffer covers fees/slippage on top of the raw
        notional so a fill doesn't overshoot available cash by a few cents.
        """
        buffer = 1 + cfg.execution.taker_fee_rate + cfg.execution.extra_slippage_bps / 10_000 + 0.005
        return portfolio.cash >= cfg.lanes.position_size_usd * buffer

    def _close(self, fill_row: pd.Series, cfg: Config, execution: ExecutionSimulator, portfolio: Portfolio, reason: str) -> None:
        position = portfolio.open_positions[self.id]
        fill = execution.market_sell(fill_row["open"], fill_row["spread"], position.qty, fill_row.name)
        portfolio.apply_sell(self.id, fill, reason)
        self.exit_model = None
        self.cooldown_until = fill_row.name + pd.Timedelta(seconds=cfg.lanes.cooldown_seconds)
        self.status = LaneStatus.COOLDOWN
