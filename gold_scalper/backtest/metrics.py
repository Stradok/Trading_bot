"""Backtest performance metrics, computed net of the execution model's
fees/slippage by default (see engine/execution.py) -- gross figures are
reported alongside so fee drag is a visible number, not hidden in an
aggregate. Also breaks out concurrent-lane risk explicitly, since running
several lanes at once is correlated exposure, not diversification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from gold_scalper.backtest.runner import BacktestResult
from gold_scalper.engine.portfolio import Portfolio


@dataclass
class BacktestMetrics:
    trade_count: int
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy_net: float
    gross_pnl: float
    net_pnl: float
    total_fees: float
    profit_factor: float
    max_drawdown_pct: float
    max_concurrent_lane_drawdown_pct: float
    lane_utilization: dict[int, dict]  # lane_count -> {"bars": int, "net_pnl": float}
    circuit_breaker_trip_count: int
    circuit_breaker_halted_bars: int
    avg_holding_seconds: float
    span_days: float
    trades_per_day: float
    net_profit_per_day: float


def _drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return (running_max - equity) / running_max.replace(0, pd.NA)


def compute_metrics(result: BacktestResult) -> BacktestMetrics:
    portfolio = result.portfolio
    trades = portfolio.trade_log
    trade_count = len(trades)

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / trade_count if trade_count else 0.0
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0

    net_pnl = sum(t.pnl for t in trades)
    total_fees = sum(t.fees for t in trades)
    gross_pnl = net_pnl + total_fees
    expectancy_net = net_pnl / trade_count if trade_count else 0.0

    gross_win_sum = sum(t.pnl for t in wins)
    gross_loss_sum = -sum(t.pnl for t in losses)
    if gross_loss_sum > 0:
        profit_factor = gross_win_sum / gross_loss_sum
    else:
        profit_factor = float("inf") if gross_win_sum > 0 else 0.0

    if portfolio.equity_curve:
        ts_index = [ts for ts, _ in portfolio.equity_curve]
        equity = pd.Series([e for _, e in portfolio.equity_curve], index=ts_index)
        dd = _drawdown_series(equity)
        max_drawdown_pct = float(dd.max()) if not dd.empty else 0.0
    else:
        equity, dd, max_drawdown_pct = pd.Series(dtype=float), pd.Series(dtype=float), 0.0

    lane_counts = pd.Series(
        [c for _, c in result.lane_count_curve],
        index=[ts for ts, _ in result.lane_count_curve],
    )
    if not lane_counts.empty and not dd.empty:
        concurrent_mask = lane_counts.reindex(dd.index).fillna(0) > 1
        max_concurrent_lane_drawdown_pct = float(dd[concurrent_mask].max()) if concurrent_mask.any() else 0.0
    else:
        max_concurrent_lane_drawdown_pct = 0.0

    lane_utilization: dict[int, dict] = {}
    for count in sorted(set(lane_counts.tolist())):
        lane_utilization[int(count)] = {"bars": int((lane_counts == count).sum()), "net_pnl": 0.0}
    for t in trades:
        idx = lane_counts.index.searchsorted(t.entry_ts, side="right") - 1
        count_at_entry = int(lane_counts.iloc[max(0, min(idx, len(lane_counts) - 1))]) if len(lane_counts) else 0
        lane_utilization.setdefault(count_at_entry, {"bars": 0, "net_pnl": 0.0})
        lane_utilization[count_at_entry]["net_pnl"] += t.pnl

    avg_holding_seconds = (
        sum((t.exit_ts - t.entry_ts).total_seconds() for t in trades) / trade_count if trade_count else 0.0
    )

    if len(lane_counts) >= 2:
        span_days = (lane_counts.index[-1] - lane_counts.index[0]).total_seconds() / 86400
    else:
        span_days = 0.0
    trades_per_day = trade_count / span_days if span_days > 0 else 0.0
    net_profit_per_day = net_pnl / span_days if span_days > 0 else 0.0

    return BacktestMetrics(
        trade_count=trade_count,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy_net=expectancy_net,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        total_fees=total_fees,
        profit_factor=profit_factor,
        max_drawdown_pct=max_drawdown_pct,
        max_concurrent_lane_drawdown_pct=max_concurrent_lane_drawdown_pct,
        lane_utilization=lane_utilization,
        circuit_breaker_trip_count=result.manager.circuit_breaker.trip_count,
        circuit_breaker_halted_bars=result.manager.circuit_breaker.halted_bar_count,
        avg_holding_seconds=avg_holding_seconds,
        span_days=span_days,
        trades_per_day=trades_per_day,
        net_profit_per_day=net_profit_per_day,
    )
