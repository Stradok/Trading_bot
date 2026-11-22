"""YAML -> dataclass config loading with fail-fast validation.

Config is assembled by deep-merging one or more YAML files in order
(later files override earlier ones), e.g. default.yaml + backtest.yaml.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class ExchangeConfig:
    id: str
    symbol: str


@dataclass
class SpreadConfig:
    spread_bps_p75: float


@dataclass
class SignalConfig:
    ema_fast_periods: int
    breakout_lookback: int
    atr_periods: int
    breakout_atr_mult: float
    slope_lookback: int
    min_slope: float


@dataclass
class ExitConfig:
    taker_fee_rate: float
    min_net_profit_pct: float
    trail_atr_mult: float
    trail_spread_mult: float
    stop_loss_atr_mult: float
    max_hold_seconds: float


@dataclass
class ExecutionConfig:
    taker_fee_rate: float
    extra_slippage_bps: float


@dataclass
class LaneConfig:
    position_size_usd: float
    cooldown_seconds: float
    max_lanes: int
    lane_equity_steps: list[tuple[float, int]]
    min_notional_usd: float


@dataclass
class CircuitBreakerConfig:
    vol_spike_mult: float
    volume_spike_mult: float
    max_daily_drawdown_pct: float
    flatten_on_trip: bool
    blackout_windows: list[tuple[dt.datetime, dt.datetime]]
    baseline_window_bars: int


@dataclass
class DataPathsConfig:
    ohlcv_1m_path: str
    ohlcv_1s_path: str
    trades_path: str | None = None


@dataclass
class BacktestConfig:
    start: str | None
    end: str | None
    train_test_split: float


@dataclass
class ActiveHoursConfig:
    timezone: str
    start_hour: int | None
    end_hour: int | None


@dataclass
class LiveConfig:
    poll_interval_seconds: float
    timeframe: str
    feature_window_bars: int
    active_hours: ActiveHoursConfig


@dataclass
class Config:
    exchange: ExchangeConfig
    spread: SpreadConfig
    signal: SignalConfig
    exit: ExitConfig
    execution: ExecutionConfig
    lanes: LaneConfig
    circuit_breaker: CircuitBreakerConfig
    data: DataPathsConfig | None = None
    backtest: BacktestConfig | None = None
    live: LiveConfig | None = None

    def validate(self, *, require_calibrated_spread: bool = True) -> None:
        errors: list[str] = []

        if require_calibrated_spread and self.spread.spread_bps_p75 <= 0:
            errors.append(
                "spread.spread_bps_p75 is 0 (uncalibrated) -- run "
                "scripts/calibrate_spread.py before backtesting or paper "
                "trading, or pass require_calibrated_spread=False if this "
                "really is intentional (e.g. a unit test)."
            )

        if self.lanes.max_lanes < 1:
            errors.append("lanes.max_lanes must be >= 1")
        if self.lanes.position_size_usd <= 0:
            errors.append("lanes.position_size_usd must be > 0")
        steps = self.lanes.lane_equity_steps
        if not steps:
            errors.append("lanes.lane_equity_steps must not be empty")
        else:
            thresholds = [t for t, _ in steps]
            if thresholds != sorted(thresholds):
                errors.append("lanes.lane_equity_steps must be sorted ascending by threshold")
            if thresholds[0] != 1.0:
                errors.append(
                    "lanes.lane_equity_steps must start at threshold 1.0 (thresholds are growth "
                    "multiples of starting capital, not absolute dollar amounts -- 1.0 = no growth yet)"
                )
            if any(count > self.lanes.max_lanes for _, count in steps):
                errors.append("lanes.lane_equity_steps has a lane count exceeding lanes.max_lanes")

        if self.lanes.position_size_usd < self.lanes.min_notional_usd:
            errors.append(
                f"lanes.position_size_usd ({self.lanes.position_size_usd}) is below "
                f"lanes.min_notional_usd ({self.lanes.min_notional_usd}) -- the exchange would reject this order"
            )

        if self.execution.taker_fee_rate <= 0:
            errors.append("execution.taker_fee_rate must be > 0 (0 would hide real fee drag)")
        if self.exit.taker_fee_rate <= 0:
            errors.append("exit.taker_fee_rate must be > 0")
        if self.exit.trail_atr_mult <= 0 and self.exit.trail_spread_mult <= 0:
            errors.append(
                "exit.trail_atr_mult and exit.trail_spread_mult are both <= 0 -- "
                "the trailing exit would fire on any tick of movement"
            )
        if self.exit.stop_loss_atr_mult <= 0:
            errors.append("exit.stop_loss_atr_mult must be > 0 (0 would hide the stop-loss protection)")
        if self.exit.min_net_profit_pct < 0:
            errors.append("exit.min_net_profit_pct must be >= 0")

        if self.signal.breakout_atr_mult < 0:
            errors.append("signal.breakout_atr_mult must be >= 0")
        if self.signal.breakout_lookback < 1 or self.signal.ema_fast_periods < 1:
            errors.append("signal.breakout_lookback and signal.ema_fast_periods must be >= 1")

        if self.circuit_breaker.max_daily_drawdown_pct <= 0:
            errors.append("circuit_breaker.max_daily_drawdown_pct must be > 0")
        if self.circuit_breaker.baseline_window_bars < 1:
            errors.append("circuit_breaker.baseline_window_bars must be >= 1")
        if self.circuit_breaker.vol_spike_mult <= 1 or self.circuit_breaker.volume_spike_mult <= 1:
            errors.append(
                "circuit_breaker.vol_spike_mult and volume_spike_mult must be > 1 "
                "(a spike multiplier <= 1 would trip on every bar)"
            )

        if self.live is not None:
            min_window = max(
                self.signal.breakout_lookback + 1, self.signal.atr_periods,
                self.circuit_breaker.baseline_window_bars,
            ) + 5
            if self.live.feature_window_bars < min_window:
                errors.append(
                    f"live.feature_window_bars ({self.live.feature_window_bars}) is too small for "
                    f"indicator warmup (need at least ~{min_window} given current signal/circuit_breaker "
                    "lookback settings) -- features would stay NaN and no trades would ever fire"
                )

        if errors:
            raise ValueError("Invalid config:\n" + "\n".join(f"  - {e}" for e in errors))


def _parse_blackout_windows(raw: list) -> list[tuple[dt.datetime, dt.datetime]]:
    windows = []
    for start, end in raw:
        windows.append((dt.datetime.fromisoformat(start), dt.datetime.fromisoformat(end)))
    return windows


def load_config(*paths: str | Path, require_calibrated_spread: bool = True) -> Config:
    """Deep-merge one or more YAML files in order and build a validated Config.

    Typical usage: load_config("config/default.yaml", "config/backtest.yaml")
    """
    merged: dict[str, Any] = {}
    for path in paths:
        with open(path) as f:
            merged = _deep_merge(merged, yaml.safe_load(f) or {})

    cfg = Config(
        exchange=ExchangeConfig(**merged["exchange"]),
        spread=SpreadConfig(**merged["spread"]),
        signal=SignalConfig(**merged["signal"]),
        exit=ExitConfig(**merged["exit"]),
        execution=ExecutionConfig(**merged["execution"]),
        lanes=LaneConfig(
            position_size_usd=merged["lanes"]["position_size_usd"],
            cooldown_seconds=merged["lanes"]["cooldown_seconds"],
            max_lanes=merged["lanes"]["max_lanes"],
            lane_equity_steps=[tuple(step) for step in merged["lanes"]["lane_equity_steps"]],
            min_notional_usd=merged["lanes"]["min_notional_usd"],
        ),
        circuit_breaker=CircuitBreakerConfig(
            vol_spike_mult=merged["circuit_breaker"]["vol_spike_mult"],
            volume_spike_mult=merged["circuit_breaker"]["volume_spike_mult"],
            max_daily_drawdown_pct=merged["circuit_breaker"]["max_daily_drawdown_pct"],
            flatten_on_trip=merged["circuit_breaker"]["flatten_on_trip"],
            blackout_windows=_parse_blackout_windows(merged["circuit_breaker"].get("blackout_windows", [])),
            baseline_window_bars=merged["circuit_breaker"]["baseline_window_bars"],
        ),
        data=DataPathsConfig(**merged["data"]) if "data" in merged else None,
        backtest=BacktestConfig(**merged["backtest"]) if "backtest" in merged else None,
        live=LiveConfig(
            poll_interval_seconds=merged["live"]["poll_interval_seconds"],
            timeframe=merged["live"]["timeframe"],
            feature_window_bars=merged["live"]["feature_window_bars"],
            active_hours=ActiveHoursConfig(**merged["live"]["active_hours"]),
        )
        if "live" in merged
        else None,
    )
    cfg.validate(require_calibrated_spread=require_calibrated_spread)
    return cfg
