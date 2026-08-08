"""Shared ccxt exchange construction, used by fetch/spread-calibration/live-feed."""

from __future__ import annotations

import ccxt


def make_exchange(exchange_id: str) -> ccxt.Exchange:
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": True})
