import pandas as pd
import pytest

from gold_scalper.backtest.optimize import walk_forward_optimize
from gold_scalper.backtest.projection import project_24_7, project_active_hours
from tests.test_optimize import make_cfg, make_multi_day_bars


def test_project_24_7_returns_a_percentile_range_per_horizon():
    bars = make_multi_day_bars(days=5)
    results = walk_forward_optimize(
        bars, make_cfg(), {"signal.breakout_atr_mult": [0.2, 0.5]},
        train_days=1, test_days=1, starting_cash=1000.0, min_trades=1,
    )
    projection = project_24_7(results)

    assert projection.n_windows_used == len(results)
    # monthly should be ~30x daily, annual ~365x -- exact multiple, since both
    # are derived from the same per-window daily figures scaled linearly
    assert projection.monthly.p50 == pytest.approx(projection.daily.p50 * 30)
    assert projection.annual.p50 == pytest.approx(projection.daily.p50 * 365)
    assert projection.daily.p25 <= projection.daily.p50 <= projection.daily.p75


def test_project_active_hours_only_counts_trades_entered_in_window():
    bars = make_multi_day_bars(days=5)
    results = walk_forward_optimize(
        bars, make_cfg(), {"signal.breakout_atr_mult": [0.2, 0.5]},
        train_days=1, test_days=1, starting_cash=1000.0, min_trades=1,
    )

    full_day = project_active_hours(results, start_hour=0, end_hour=24, timezone="UTC")
    half_day = project_active_hours(results, start_hour=0, end_hour=12, timezone="UTC")

    # restricting to half the day should never produce a larger |net pnl|,
    # magnitude-wise, than using the full day's trades (fewer trades counted)
    assert abs(half_day.daily.p50) <= abs(full_day.daily.p50) + 1e-9 or half_day.n_windows_used == 0


def test_project_active_hours_handles_wraparound_window():
    bars = make_multi_day_bars(days=5)
    results = walk_forward_optimize(
        bars, make_cfg(), {}, train_days=1, test_days=1, starting_cash=1000.0, min_trades=1,
    )
    # wraps past midnight (22:00 -> 04:00) -- must not raise
    projection = project_active_hours(results, start_hour=22, end_hour=4, timezone="UTC")
    assert "22:00-04:00" in projection.scenario
