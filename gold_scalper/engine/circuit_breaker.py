"""Halts new entries (and optionally flattens open positions) during
conditions where running several correlated lanes at once is dangerous:
a volatility or volume spike, a daily drawdown breach, or a manually
configured blackout window (e.g. FOMC/NFP). This is what bounds the risk
of "more concurrent lanes" not being real diversification.
"""

from __future__ import annotations

import pandas as pd

from gold_scalper.config import CircuitBreakerConfig

from .portfolio import Portfolio


class CircuitBreaker:
    def __init__(self, cfg: CircuitBreakerConfig):
        self.cfg = cfg
        self.trip_count = 0        # number of distinct trip events (not-tripped -> tripped transitions)
        self.halted_bar_count = 0  # total bars spent tripped
        self._was_tripped = False

    def tripped(self, row: pd.Series, portfolio: Portfolio) -> bool:
        vol_spike = self._ratio_exceeds(row.get("atr"), row.get("atr_baseline"), self.cfg.vol_spike_mult)
        volume_spike = self._ratio_exceeds(row.get("volume"), row.get("volume_baseline"), self.cfg.volume_spike_mult)
        dd_breach = portfolio.daily_drawdown_pct() > self.cfg.max_daily_drawdown_pct
        blackout = any(start <= row.name <= end for start, end in self.cfg.blackout_windows)

        result = bool(vol_spike or volume_spike or dd_breach or blackout)
        if result:
            self.halted_bar_count += 1
            if not self._was_tripped:
                self.trip_count += 1
        self._was_tripped = result
        return result

    @staticmethod
    def _ratio_exceeds(value, baseline, mult: float) -> bool:
        if value is None or baseline is None or pd.isna(value) or pd.isna(baseline) or baseline <= 0:
            return False
        return (value / baseline) > mult
