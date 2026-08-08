import pandas as pd
import pytest

from gold_scalper.config import ExecutionConfig
from gold_scalper.engine.execution import ExecutionSimulator

TS = pd.Timestamp("2026-01-01T00:00:00Z")


def test_market_buy_crosses_half_spread_and_adds_slippage():
    sim = ExecutionSimulator(ExecutionConfig(taker_fee_rate=0.001, extra_slippage_bps=2.0))
    fill = sim.market_buy(mid=100.0, spread=0.10, qty=2.0, ts=TS)
    expected_price = 100.0 + 0.05 + 100.0 * 2.0 / 10_000  # 100.05 + 0.02
    assert fill.price == pytest.approx(expected_price)
    assert fill.qty == 2.0
    assert fill.fee == pytest.approx(expected_price * 2.0 * 0.001)
    assert fill.ts == TS


def test_market_sell_crosses_half_spread_and_subtracts_slippage():
    sim = ExecutionSimulator(ExecutionConfig(taker_fee_rate=0.001, extra_slippage_bps=2.0))
    fill = sim.market_sell(mid=100.0, spread=0.10, qty=2.0, ts=TS)
    expected_price = 100.0 - 0.05 - 100.0 * 2.0 / 10_000  # 99.95 - 0.02
    assert fill.price == pytest.approx(expected_price)
    assert fill.fee == pytest.approx(expected_price * 2.0 * 0.001)


def test_round_trip_at_same_mid_always_loses_money_to_spread_and_fees():
    sim = ExecutionSimulator(ExecutionConfig(taker_fee_rate=0.001, extra_slippage_bps=1.0))
    buy = sim.market_buy(mid=100.0, spread=0.05, qty=1.0, ts=TS)
    sell = sim.market_sell(mid=100.0, spread=0.05, qty=1.0, ts=TS)
    round_trip_pnl = (sell.price - buy.price) - buy.fee - sell.fee
    assert round_trip_pnl < 0
