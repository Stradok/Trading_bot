"""Revenue projection for two operating scenarios -- running locally on a
MacBook for a limited window per day, vs. running 24/7 on a server.

Built entirely from walk-forward *out-of-sample* results (optimize.py), never
from in-sample/optimistic numbers. Output is a range across test windows
(25th/50th/75th percentile of that window's realized net-profit/day), not a
single point estimate -- a point estimate would overstate confidence given
how fee- and parameter-sensitive this strategy is. Both scenarios are
implicitly bounded by the capital-constrained lane cap (lanes.max_lanes,
lanes.lane_equity_steps, lanes.min_notional_usd) because that's already
enforced inside the backtest engine that produced these results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_scalper.backtest.optimize import WindowResult


@dataclass
class ProjectionRange:
    p25: float
    p50: float
    p75: float


@dataclass
class ProfitProjection:
    scenario: str
    n_windows_used: int
    daily: ProjectionRange
    monthly: ProjectionRange
    annual: ProjectionRange


def _percentile_range(values: list[float]) -> ProjectionRange:
    if not values:
        return ProjectionRange(0.0, 0.0, 0.0)
    arr = np.array(values)
    return ProjectionRange(
        p25=float(np.percentile(arr, 25)),
        p50=float(np.percentile(arr, 50)),
        p75=float(np.percentile(arr, 75)),
    )


def _scale(rng: ProjectionRange, factor: float) -> ProjectionRange:
    return ProjectionRange(rng.p25 * factor, rng.p50 * factor, rng.p75 * factor)


def _in_active_window(ts: pd.Timestamp, start_hour: int, end_hour: int, timezone: str) -> bool:
    local_hour = ts.tz_convert(timezone).hour
    if start_hour <= end_hour:
        return start_hour <= local_hour < end_hour
    return local_hour >= start_hour or local_hour < end_hour  # window wraps past midnight


def project_24_7(results: list[WindowResult]) -> ProfitProjection:
    """The server scenario: use each test window's full-day net-profit/day figure directly."""
    daily_values = [r.test_metrics.net_profit_per_day for r in results if r.test_metrics.span_days > 0]
    daily = _percentile_range(daily_values)
    return ProfitProjection(
        scenario="24/7 server", n_windows_used=len(daily_values),
        daily=daily, monthly=_scale(daily, 30), annual=_scale(daily, 365),
    )


def project_active_hours(
    results: list[WindowResult], start_hour: int, end_hour: int, timezone: str = "UTC",
) -> ProfitProjection:
    """The local MacBook scenario: restrict each test window's trades to those
    whose *entry* fell inside [start_hour, end_hour) in `timezone`, using the
    actual per-trade timestamps from that window's out-of-sample run --
    PAXG/USDT liquidity and volatility vary by session, so this isn't just
    net_profit_per_day * hours/24, it reflects when trades actually happened.
    """
    daily_values = []
    for r in results:
        if r.test_metrics.span_days <= 0:
            continue
        trades = r.test_result.portfolio.trade_log
        window_trades = [t for t in trades if _in_active_window(t.entry_ts, start_hour, end_hour, timezone)]
        net_pnl = sum(t.pnl for t in window_trades)
        daily_values.append(net_pnl / r.test_metrics.span_days)

    daily = _percentile_range(daily_values)
    hours = (end_hour - start_hour) if start_hour <= end_hour else (24 - start_hour + end_hour)
    return ProfitProjection(
        scenario=f"local MacBook ({hours}h/day, {start_hour:02d}:00-{end_hour:02d}:00 {timezone})",
        n_windows_used=len(daily_values),
        daily=daily, monthly=_scale(daily, 30), annual=_scale(daily, 365),
    )


def format_projection(projection: ProfitProjection) -> str:
    def fmt(rng: ProjectionRange) -> str:
        return f"${rng.p25:+.2f} .. ${rng.p50:+.2f} .. ${rng.p75:+.2f}  (p25 / p50 / p75)"

    lines = [
        f"Scenario: {projection.scenario}  ({projection.n_windows_used} out-of-sample windows used)",
        f"  Daily:   {fmt(projection.daily)}",
        f"  Monthly: {fmt(projection.monthly)}",
        f"  Annual:  {fmt(projection.annual)}",
    ]
    return "\n".join(lines)


PROJECTION_DISCLAIMER = (
    "This projection is derived entirely from backtest out-of-sample results on historical data. "
    "It is not a guarantee -- live/paper trading performance (see gold_scalper/live/paper_runner.py) "
    "is the real test, and should be compared against this projection before trusting it with more capital."
)
