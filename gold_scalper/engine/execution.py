"""Shared fill/fee/slippage model, used identically by the backtester and the
paper trading runner -- this is what makes small profit targets fee-sensitive
in a realistic way instead of assuming free, frictionless fills.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gold_scalper.config import ExecutionConfig


@dataclass
class Fill:
    price: float
    qty: float
    fee: float
    ts: pd.Timestamp


class ExecutionSimulator:
    def __init__(self, cfg: ExecutionConfig):
        self.cfg = cfg

    def market_buy(self, mid: float, spread: float, qty: float, ts: pd.Timestamp) -> Fill:
        """spread is in price units (e.g. dollars), not basis points."""
        price = mid + spread / 2 + mid * self.cfg.extra_slippage_bps / 10_000
        fee = price * qty * self.cfg.taker_fee_rate
        return Fill(price, qty, fee, ts)

    def market_sell(self, mid: float, spread: float, qty: float, ts: pd.Timestamp) -> Fill:
        price = mid - spread / 2 - mid * self.cfg.extra_slippage_bps / 10_000
        fee = price * qty * self.cfg.taker_fee_rate
        return Fill(price, qty, fee, ts)
