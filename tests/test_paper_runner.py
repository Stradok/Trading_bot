import pandas as pd
import pytest

from gold_scalper.config import (
    ActiveHoursConfig, CircuitBreakerConfig, Config, ExchangeConfig, ExecutionConfig,
    ExitConfig, LaneConfig, LiveConfig, SignalConfig, SpreadConfig,
)
from gold_scalper.live.paper_runner import PaperRunner

BREAKOUT_CLOSES = [100.0] * 10 + [100, 101, 108, 108.5, 109] + [104, 103, 102, 103, 104] + [115]


class FakeFeed:
    """Stub for LiveFeed -- PaperRunner only calls poll_spread() directly;
    bars are fed manually via on_new_bar() in these tests instead of feed.run().
    """

    def poll_spread(self) -> float:
        return 0.02


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
        live=LiveConfig(
            poll_interval_seconds=2.0, timeframe="1m", feature_window_bars=50,
            active_hours=ActiveHoursConfig(timezone="UTC", start_hour=None, end_hour=None),
        ),
    )
    cfg.validate()
    return cfg


def make_bar(i: int, close: float, start=pd.Timestamp("2026-01-01T00:00:00Z")) -> dict:
    ts = start + pd.Timedelta(minutes=i)
    return {"timestamp": int(ts.value // 1_000_000), "open": close, "high": close, "low": close, "close": close, "volume": 1.0}


def test_paper_runner_feeds_bars_through_same_engine_as_backtest():
    runner = PaperRunner(make_cfg(), starting_cash=1000.0, feed=FakeFeed())

    for i, close in enumerate(BREAKOUT_CLOSES):
        runner.on_new_bar(make_bar(i, close))

    # same breakout/pullback pattern used in test_lane_manager.py -- expect at
    # least one completed trade with a positive pnl despite the pullback
    assert len(runner.portfolio.trade_log) >= 1
    assert runner.portfolio.trade_log[0].pnl > 0


def test_paper_runner_window_never_exceeds_configured_size():
    cfg = make_cfg()
    cfg.live.feature_window_bars = 15
    runner = PaperRunner(cfg, starting_cash=1000.0, feed=FakeFeed())

    for i, close in enumerate(BREAKOUT_CLOSES):
        runner.on_new_bar(make_bar(i, close))

    assert len(runner._bars) <= 15


def test_paper_runner_uses_live_spread_override():
    class SpikeSpreadFeed(FakeFeed):
        def poll_spread(self) -> float:
            return 5.0  # deliberately unrealistic, to prove it's actually used

    runner = PaperRunner(make_cfg(), starting_cash=1000.0, feed=SpikeSpreadFeed())
    latest = runner.on_new_bar(make_bar(0, 100.0))
    assert latest["spread"] == 5.0
