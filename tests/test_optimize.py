import pandas as pd
import pytest

from gold_scalper.backtest.optimize import summarize_out_of_sample, walk_forward_optimize
from gold_scalper.config import (
    CircuitBreakerConfig, Config, ExchangeConfig, ExecutionConfig, ExitConfig,
    LaneConfig, SignalConfig, SpreadConfig,
)

# A repeating breakout/pullback cycle, tiled across several synthetic days so
# each 1-day train/test window has enough trades to clear optimize.py's
# min_trades filter.
_CYCLE = [100.0] * 6 + [101, 105, 106, 104, 102] + [101] * 4


def make_cfg() -> Config:
    cfg = Config(
        exchange=ExchangeConfig(id="binance", symbol="PAXG/USDT"),
        spread=SpreadConfig(spread_bps_p75=0.02),
        signal=SignalConfig(
            ema_fast_periods=3, breakout_lookback=5, atr_periods=3,
            breakout_atr_mult=0.2, slope_lookback=2, min_slope=0.0,
        ),
        exit=ExitConfig(
            taker_fee_rate=0.001, min_net_profit_pct=0.0005,
            trail_atr_mult=0.5, trail_spread_mult=1.5,
            stop_loss_atr_mult=1.0, max_hold_seconds=1800,
        ),
        execution=ExecutionConfig(taker_fee_rate=0.001, extra_slippage_bps=0.0),
        lanes=LaneConfig(
            position_size_usd=25.0, cooldown_seconds=1, max_lanes=1,
            lane_equity_steps=[(1.0, 1)], min_notional_usd=5.0,
        ),
        circuit_breaker=CircuitBreakerConfig(
            vol_spike_mult=1000.0, volume_spike_mult=1000.0, max_daily_drawdown_pct=0.5,
            flatten_on_trip=True, blackout_windows=[], baseline_window_bars=5,
        ),
    )
    cfg.validate()
    return cfg


def make_multi_day_bars(days: int) -> pd.DataFrame:
    bars_per_day = 1440
    repeats = -(-bars_per_day // len(_CYCLE))  # ceil
    one_day = (_CYCLE * repeats)[:bars_per_day]
    closes = one_day * days
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=len(closes), freq="1min")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [1.0] * len(closes)},
        index=idx,
    )


def test_walk_forward_optimize_produces_out_of_sample_results_for_each_window():
    bars = make_multi_day_bars(days=5)
    param_grid = {"signal.breakout_atr_mult": [0.2, 0.5]}

    results = walk_forward_optimize(
        bars, make_cfg(), param_grid, train_days=1, test_days=1,
        starting_cash=1000.0, min_trades=3,
    )

    # 5 days of bars (last bar lands 1 minute short of day 5), 1-day train +
    # 1-day test stepped by test_days=1 -> windows test on [1,2), [2,3), [3,4)
    assert len(results) == 3
    for r in results:
        assert r.train_end == r.test_start
        assert r.test_end - r.test_start == pd.Timedelta(days=1)
        assert "signal.breakout_atr_mult" in r.best_params
        assert r.best_params["signal.breakout_atr_mult"] in (0.2, 0.5)
        assert r.test_metrics.trade_count >= 0  # ran without crashing even if below min_trades

    summary = summarize_out_of_sample(results)
    assert summary["n_windows"] == 3
    assert "mean_expectancy_net" in summary


def test_walk_forward_optimize_handles_no_window_fitting_gracefully():
    bars = make_multi_day_bars(days=1)  # too short for train=2 + test=1
    results = walk_forward_optimize(bars, make_cfg(), {"signal.min_slope": [0.0]}, train_days=2, test_days=1)
    assert results == []
    assert summarize_out_of_sample(results) == {}
