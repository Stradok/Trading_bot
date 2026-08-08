"""Profit-lock trailing exit, redefined in real terms (fees + spread + ATR)
instead of the idea doc's raw "0.0001%" reversal threshold.

The exit only arms once unrealized profit clears the actual round-trip cost
(fees + spread) plus a buffer -- so it can never lock in a "profit" that's
actually a net loss. Once armed, the trail distance is scaled to whichever
is larger of recent volatility (ATR) or spread, so a single noisy tick can't
stop it out immediately after arming.
"""

from __future__ import annotations

import pandas as pd

from gold_scalper.config import ExitConfig

EXIT_PROFIT = "exit_profit"
EXIT_STOP_LOSS = "exit_stop_loss"
EXIT_TIMEOUT = "exit_timeout"


class TrailingExit:
    def __init__(self, entry_price: float, entry_ts: pd.Timestamp, cfg: ExitConfig):
        self.entry_price = entry_price
        self.entry_ts = entry_ts
        self.peak = entry_price
        self.armed = False
        self.cfg = cfg

    def update(self, price: float, atr: float, spread: float, now: pd.Timestamp) -> str | None:
        """spread is in price units (e.g. dollars), not basis points."""
        self.peak = max(self.peak, price)
        unreal_pct = (price - self.entry_price) / self.entry_price
        cost_pct = 2 * self.cfg.taker_fee_rate + spread / price  # round-trip fee + spread

        if not self.armed and unreal_pct >= cost_pct + self.cfg.min_net_profit_pct:
            self.armed = True

        if self.armed:
            trail_dist = max(self.cfg.trail_atr_mult * atr, self.cfg.trail_spread_mult * spread)
            if price <= self.peak - trail_dist:
                return EXIT_PROFIT

        if price <= self.entry_price - self.cfg.stop_loss_atr_mult * atr:
            return EXIT_STOP_LOSS

        if (now - self.entry_ts).total_seconds() > self.cfg.max_hold_seconds:
            return EXIT_TIMEOUT

        return None
