"""Live market data polling for paper trading.

Plain REST polling (not websockets/ccxt.pro) at a 1-2s cadence is enough at
this trading frequency and keeps the dependency set minimal -- see
gold_scalper/live/paper_runner.py for how this feeds the same engine the
backtester uses.
"""

from __future__ import annotations

import time
from typing import Iterator

from .exchange import make_exchange


class LiveFeed:
    def __init__(self, exchange_id: str, symbol: str, timeframe: str = "1m", poll_interval_seconds: float = 2.0):
        self.exchange = make_exchange(exchange_id)
        self.exchange.load_markets()
        if timeframe not in getattr(self.exchange, "timeframes", {}):
            raise ValueError(f"{exchange_id} does not expose timeframe {timeframe!r}")
        self.symbol = symbol
        self.timeframe = timeframe
        self.poll_interval_seconds = poll_interval_seconds
        self._last_bar_ts: int | None = None

    def poll_latest_closed_bar(self) -> dict | None:
        """Returns the most recently *closed* candle, or None if it's the same
        one already returned on a previous poll (still waiting on a new bar).
        """
        candles = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=2)
        if len(candles) < 2:
            return None
        closed = candles[-2]  # the last element is the still-forming current bar
        ts = closed[0]
        if ts == self._last_bar_ts:
            return None
        self._last_bar_ts = ts
        return {
            "timestamp": ts, "open": closed[1], "high": closed[2],
            "low": closed[3], "close": closed[4], "volume": closed[5],
        }

    def poll_spread(self) -> float | None:
        """Current top-of-book bid/ask spread in price units (dollars)."""
        order_book = self.exchange.fetch_order_book(self.symbol, limit=5)
        bids, asks = order_book.get("bids") or [], order_book.get("asks") or []
        if not bids or not asks:
            return None
        return asks[0][0] - bids[0][0]

    def run(self) -> Iterator[dict]:
        """Yields each newly closed bar as it becomes available."""
        while True:
            bar = self.poll_latest_closed_bar()
            if bar is not None:
                yield bar
            time.sleep(self.poll_interval_seconds)
