"""Cash/position ledger with mark-to-market equity, which is what drives lane
scaling (gold_scalper/engine/lane_manager.py) -- realized PnL alone isn't
enough, since a lane with an open position should still count toward equity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .execution import Fill


@dataclass
class Position:
    entry_price: float
    qty: float
    entry_ts: pd.Timestamp


@dataclass
class TradeRecord:
    lane_id: int
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    fees: float
    pnl: float
    close_reason: str


class Portfolio:
    def __init__(self, starting_cash: float):
        self.cash = starting_cash
        self.starting_cash = starting_cash
        self.open_positions: dict[int, Position] = {}
        self._open_entry_fee: dict[int, float] = {}
        self.trade_log: list[TradeRecord] = []
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []

    def apply_buy(self, lane_id: int, fill: Fill) -> None:
        self.cash -= fill.price * fill.qty + fill.fee
        self.open_positions[lane_id] = Position(fill.price, fill.qty, fill.ts)
        self._open_entry_fee[lane_id] = fill.fee

    def apply_sell(self, lane_id: int, fill: Fill, close_reason: str) -> TradeRecord:
        position = self.open_positions.pop(lane_id)
        entry_fee = self._open_entry_fee.pop(lane_id)
        proceeds = fill.price * fill.qty - fill.fee
        self.cash += proceeds
        pnl = proceeds - (position.entry_price * position.qty + entry_fee)
        record = TradeRecord(
            lane_id=lane_id, entry_ts=position.entry_ts, exit_ts=fill.ts,
            entry_price=position.entry_price, exit_price=fill.price, qty=position.qty,
            fees=entry_fee + fill.fee, pnl=pnl, close_reason=close_reason,
        )
        self.trade_log.append(record)
        return record

    def mark_to_market(self, current_price: float, now: pd.Timestamp) -> float:
        equity = self.cash + sum(p.qty * current_price for p in self.open_positions.values())
        self.equity_curve.append((now, equity))
        return equity

    @property
    def equity(self) -> float:
        return self.equity_curve[-1][1] if self.equity_curve else self.starting_cash

    def daily_drawdown_pct(self) -> float:
        """Drawdown from today's peak equity so far -- used by the circuit breaker."""
        if not self.equity_curve:
            return 0.0
        now = self.equity_curve[-1][0]
        today_start = now.normalize()
        todays_equity = [e for ts, e in self.equity_curve if ts >= today_start]
        if not todays_equity:
            return 0.0
        peak = max(todays_equity)
        if peak <= 0:
            return 0.0
        return max(0.0, (peak - todays_equity[-1]) / peak)
