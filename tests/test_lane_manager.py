import pandas as pd
import pytest

from gold_scalper.config import (
    CircuitBreakerConfig, Config, ExchangeConfig, ExecutionConfig, ExitConfig,
    LaneConfig, SignalConfig, SpreadConfig,
)
from gold_scalper.engine.circuit_breaker import CircuitBreaker
from gold_scalper.engine.execution import ExecutionSimulator
from gold_scalper.engine.lane import LaneStatus
from gold_scalper.engine.lane_manager import LaneManager
from gold_scalper.engine.portfolio import Portfolio
from gold_scalper.strategy.features import compute_all_features


def make_config(**overrides) -> Config:
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
            position_size_usd=25.0, cooldown_seconds=1, max_lanes=4,
            lane_equity_steps=[(1.0, 1), (1.2, 2), (1.5, 3), (2.0, 4)],
            min_notional_usd=5.0,
        ),
        circuit_breaker=CircuitBreakerConfig(
            vol_spike_mult=3.0, volume_spike_mult=4.0, max_daily_drawdown_pct=0.5,
            flatten_on_trip=True, blackout_windows=[], baseline_window_bars=5,
        ),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.validate()
    return cfg


def make_bars(closes: list[float], volume: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=len(closes), freq="1min")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [volume] * len(closes)},
        index=idx,
    )


# Same breakout(11-14) / pullback(15-20) shape as test_signals.py, chosen so
# the pullback is sharp enough to also trip the trailing exit after entry.
BREAKOUT_CLOSES = [100.0] * 10 + [100, 101, 108, 108.5, 109] + [104, 103, 102, 103, 104] + [115]


def _single_lane_cfg(**circuit_breaker_overrides) -> Config:
    base = make_config()
    cb_kwargs = dict(
        vol_spike_mult=3.0, volume_spike_mult=4.0, max_daily_drawdown_pct=0.5,
        flatten_on_trip=True, blackout_windows=[], baseline_window_bars=5,
    )
    cb_kwargs.update(circuit_breaker_overrides)
    return make_config(
        lanes=LaneConfig(
            position_size_usd=25.0, cooldown_seconds=1, max_lanes=1,
            lane_equity_steps=[(1.0, 1)], min_notional_usd=5.0,
        ),
        circuit_breaker=CircuitBreakerConfig(**cb_kwargs),
    )


def test_lane_opens_on_breakout_and_closes_on_trailing_exit():
    # vol/volume spike multipliers set very high: this test isolates lane
    # entry/exit logic, not circuit-breaker behavior (which has its own tests
    # below) -- a synthetic all-flat warmup gives the ATR baseline a near-zero
    # value right before the breakout, which would otherwise trip the breaker
    # on the first real bar of movement (real market data doesn't have this
    # degenerate all-zero-ATR warmup, so this isn't a production concern).
    cfg = _single_lane_cfg(vol_spike_mult=1000.0, volume_spike_mult=1000.0)
    bars = compute_all_features(make_bars(BREAKOUT_CLOSES), cfg)
    portfolio = Portfolio(starting_cash=1000.0)
    manager = LaneManager(cfg, ExecutionSimulator(cfg.execution), portfolio)

    for i in range(len(bars)):
        manager.on_bar(bars.iloc[i])

    assert len(portfolio.trade_log) >= 1
    first_trade = portfolio.trade_log[0]
    assert first_trade.entry_price == pytest.approx(101.0, abs=0.01)
    assert first_trade.close_reason == "exit_profit"
    assert first_trade.pnl > 0  # entered ~101, exited well above that despite the pullback


def test_desired_lane_count_scales_with_equity_and_respects_max_lanes():
    cfg = make_config()
    manager = LaneManager(cfg, ExecutionSimulator(cfg.execution), Portfolio(starting_cash=100.0))
    assert manager.desired_lane_count(equity=50) == 1
    assert manager.desired_lane_count(equity=120) == 2
    assert manager.desired_lane_count(equity=150) == 3
    assert manager.desired_lane_count(equity=10_000) == 4  # capped at max_lanes even with huge equity


def test_circuit_breaker_trips_on_volatility_spike_and_flattens_open_lane():
    cfg = make_config()
    breaker = CircuitBreaker(cfg.circuit_breaker)
    portfolio = Portfolio(starting_cash=1000.0)

    calm_row = pd.Series({"atr": 1.0, "atr_baseline": 1.0, "volume": 1.0, "volume_baseline": 1.0}, name=pd.Timestamp("2026-01-01"))
    assert breaker.tripped(calm_row, portfolio) is False

    spike_row = pd.Series({"atr": 5.0, "atr_baseline": 1.0, "volume": 1.0, "volume_baseline": 1.0}, name=pd.Timestamp("2026-01-01T00:01:00"))
    assert breaker.tripped(spike_row, portfolio) is True
    assert breaker.trip_count == 1
    assert breaker.halted_bar_count == 1


def test_lane_manager_blocks_new_entries_and_flattens_during_circuit_breaker_trip():
    # same reasoning as above: disable spurious trips from the flat warmup so
    # only the deliberately-injected spike below causes a trip.
    cfg = _single_lane_cfg(vol_spike_mult=1000.0, volume_spike_mult=1000.0)
    bars = compute_all_features(make_bars(BREAKOUT_CLOSES), cfg)
    portfolio = Portfolio(starting_cash=1000.0)
    manager = LaneManager(cfg, ExecutionSimulator(cfg.execution), portfolio)

    # run up to and including the entry bar (index 11) so a lane is open
    for i in range(12):
        manager.on_bar(bars.iloc[i])
    assert manager.lanes[0].status is LaneStatus.IN_POSITION

    # force a circuit-breaker trip on the next bar via an artificial atr spike
    tripped_row = bars.iloc[12].copy()
    tripped_row["atr_baseline"] = 0.0001  # makes atr/atr_baseline blow past vol_spike_mult (1000)
    manager.on_bar(tripped_row)

    assert manager.lanes[0].status is LaneStatus.COOLDOWN
    assert portfolio.trade_log[-1].close_reason == "circuit_breaker"
