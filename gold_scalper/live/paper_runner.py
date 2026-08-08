"""Paper trading: drives the exact same LaneManager/ExecutionSimulator used
by the backtester, fed by a live data feed instead of stored history. No real
orders are ever placed -- ExecutionSimulator only *simulates* fills, so this
literally is the backtest engine, just pointed at gold_scalper/data/live_feed.py
instead of a parquet file. Trade log format matches the backtester's, so
paper performance can be diffed directly against backtest expectation over
the same period.
"""

from __future__ import annotations

import logging

import pandas as pd

from gold_scalper.config import Config
from gold_scalper.data.live_feed import LiveFeed
from gold_scalper.engine.execution import ExecutionSimulator
from gold_scalper.engine.lane_manager import LaneManager
from gold_scalper.engine.portfolio import Portfolio
from gold_scalper.strategy.features import compute_all_features

logger = logging.getLogger(__name__)

_EMPTY_BARS_COLUMNS = ["open", "high", "low", "close", "volume"]


class PaperRunner:
    def __init__(self, cfg: Config, starting_cash: float = 150.0, feed: LiveFeed | None = None):
        assert cfg.live is not None, "PaperRunner requires config/paper.yaml's [live] section"
        self.cfg = cfg
        self.feed = feed or LiveFeed(
            cfg.exchange.id, cfg.exchange.symbol,
            timeframe=cfg.live.timeframe, poll_interval_seconds=cfg.live.poll_interval_seconds,
        )
        self.portfolio = Portfolio(starting_cash)
        self.execution = ExecutionSimulator(cfg.execution)
        self.manager = LaneManager(cfg, self.execution, self.portfolio)
        self._bars = pd.DataFrame(columns=_EMPTY_BARS_COLUMNS)
        self._bars.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")

    def _append_bar(self, bar: dict) -> None:
        ts = pd.to_datetime(bar["timestamp"], unit="ms", utc=True)
        row = pd.DataFrame([{k: bar[k] for k in _EMPTY_BARS_COLUMNS}], index=[ts])
        self._bars = pd.concat([self._bars, row])
        window = self.cfg.live.feature_window_bars
        if len(self._bars) > window:
            self._bars = self._bars.iloc[-window:]

    def on_new_bar(self, bar: dict) -> pd.Series:
        """Processes one newly closed bar. Returns the feature-enriched row
        that was fed to the engine, mainly for logging/testing.
        """
        self._append_bar(bar)
        featured = compute_all_features(self._bars, self.cfg)
        latest = featured.iloc[-1].copy()

        # features.py derives 'spread' from the calibrated bps figure; prefer
        # a live top-of-book read when available since it's more current.
        live_spread = self.feed.poll_spread()
        if live_spread is not None:
            latest["spread"] = live_spread

        # decision_row == fill_row: no future bar exists yet in live mode,
        # unlike the backtester which fills against bar t+1 (see engine/lane.py).
        self.manager.on_bar(latest)

        logger.info(
            "bar %s close=%.4f equity=%.4f open_lanes=%d trades=%d",
            latest.name, latest["close"], self.portfolio.equity,
            self.manager.open_lane_count, len(self.portfolio.trade_log),
        )
        return latest

    def run_forever(self) -> None:
        logger.info(
            "Starting paper trading: %s on %s, timeframe=%s, starting_cash=%.2f",
            self.cfg.exchange.symbol, self.cfg.exchange.id, self.cfg.live.timeframe, self.portfolio.starting_cash,
        )
        for bar in self.feed.run():
            self.on_new_bar(bar)
