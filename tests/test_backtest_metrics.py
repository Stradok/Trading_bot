import pandas as pd
import pytest

from gold_scalper.backtest.metrics import compute_metrics
from gold_scalper.backtest.runner import run_backtest
from gold_scalper.config import (
    CircuitBreakerConfig, Config, ExchangeConfig, ExecutionConfig, ExitConfig,
    LaneConfig, SignalConfig, SpreadConfig,
)

# Same synthetic breakout/pullback/fresh-breakout shape used in test_lane_manager.py,
# extended a bit so the fresh breakout also has room to run and pull back.
CLOSES = (
    [100.0] * 10
    + [100, 101, 108, 108.5, 109]
    + [104, 103, 102, 103, 104]
    + [115, 116, 116.5, 117]
    + [110, 108, 107, 108, 109]
)


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


def make_bars(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=len(closes), freq="1min")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [1.0] * len(closes)},
        index=idx,
    )


def test_metrics_are_internally_consistent():
    result = run_backtest(make_bars(CLOSES), make_cfg(), starting_cash=1000.0)
    metrics = compute_metrics(result)

    trades = result.portfolio.trade_log
    assert metrics.trade_count == len(trades)
    assert metrics.trade_count >= 1

    assert metrics.net_pnl == pytest.approx(sum(t.pnl for t in trades))
    assert metrics.total_fees == pytest.approx(sum(t.fees for t in trades))
    assert metrics.gross_pnl == pytest.approx(metrics.net_pnl + metrics.total_fees)
    assert 0.0 <= metrics.win_rate <= 1.0
    assert metrics.max_drawdown_pct >= 0.0
    assert metrics.max_concurrent_lane_drawdown_pct >= 0.0

    # lane_utilization bar counts should partition the full lane_count_curve
    assert sum(v["bars"] for v in metrics.lane_utilization.values()) == len(result.lane_count_curve)
    # with max_lanes=1, every trade's entry should be attributed to lane-count bucket 0 or 1
    assert set(metrics.lane_utilization.keys()) <= {0, 1}

    assert metrics.circuit_breaker_trip_count == result.manager.circuit_breaker.trip_count
    assert metrics.span_days > 0
    assert metrics.trades_per_day > 0


def test_no_trades_produces_zeroed_metrics_not_a_crash():
    # flat series never triggers a breakout -- must not divide by zero anywhere
    flat_cfg = make_cfg()
    result = run_backtest(make_bars([100.0] * 20), flat_cfg, starting_cash=1000.0)
    metrics = compute_metrics(result)
    assert metrics.trade_count == 0
    assert metrics.net_pnl == 0.0
    assert metrics.win_rate == 0.0
    assert metrics.expectancy_net == 0.0
