"""Paginated, resumable historical OHLCV fetching via ccxt.

Resumability: each call checks the last timestamp already stored at `path`
and picks up from there, so re-running the same command later (or after a
crash mid-pull) just fetches the gap instead of redownloading everything.
"""

from __future__ import annotations

import time
from typing import Callable

import pandas as pd

from . import store
from .exchange import make_exchange

ProgressCallback = Callable[[int, pd.Timestamp], None]


def _flush(rows: list, path: str) -> None:
    df = pd.DataFrame(rows, columns=store.OHLCV_COLUMNS)
    store.append_ohlcv(df, path)


def fetch_ohlcv_history(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    path: str,
    default_since_iso: str = "2019-01-01T00:00:00Z",
    limit: int = 1000,
    flush_every: int = 5000,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Fetch all available OHLCV candles for `timeframe`, resuming from whatever
    is already stored at `path`, up to "now". Returns the full stored DataFrame.
    """
    exchange = make_exchange(exchange_id)
    exchange.load_markets()
    interval_ms = exchange.parse_timeframe(timeframe) * 1000

    last_ts = store.last_timestamp_ms(path)
    since = last_ts + interval_ms if last_ts is not None else exchange.parse8601(default_since_iso)
    now_ms = exchange.milliseconds()

    pending: list = []
    total_fetched = 0
    while since < now_ms:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not candles:
            break
        pending.extend(candles)
        total_fetched += len(candles)

        new_since = candles[-1][0] + interval_ms
        if new_since <= since:
            break  # exchange returned a non-advancing page; stop rather than loop forever
        since = new_since

        if progress_callback:
            progress_callback(total_fetched, pd.to_datetime(candles[-1][0], unit="ms", utc=True))

        if len(pending) >= flush_every:
            _flush(pending, path)
            pending = []

        if len(candles) < limit:
            break  # fewer than a full page means we've caught up to "now"

        time.sleep(exchange.rateLimit / 1000)

    if pending:
        _flush(pending, path)

    return store.load_ohlcv(path)


def fetch_recent_trades(
    exchange_id: str,
    symbol: str,
    path: str,
    window_days: int = 90,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Fetch raw trades (with taker side) over a recent window, resuming from
    whatever is already stored. Unlike fetch_recent_second_bars, this keeps
    the buy/sell side per trade instead of collapsing into OHLCV -- needed
    for order-flow-imbalance features, which OHLCV bars can't express.
    """
    exchange = make_exchange(exchange_id)
    exchange.load_markets()

    last_ts = store.last_trade_timestamp_ms(path)
    window_start_ms = exchange.milliseconds() - window_days * 24 * 3600 * 1000
    since = max(last_ts, window_start_ms) if last_ts is not None else window_start_ms
    now_ms = exchange.milliseconds()

    pending: list = []
    total_fetched = 0
    while since < now_ms:
        trades = exchange.fetch_trades(symbol, since=since, limit=1000)
        if not trades:
            break
        pending.extend(
            {"timestamp": t["timestamp"], "id": t["id"], "price": t["price"],
             "amount": t["amount"], "side": t["side"]}
            for t in trades
        )
        total_fetched += len(trades)

        new_since = trades[-1]["timestamp"] + 1
        if new_since <= since:
            break
        since = new_since

        if progress_callback:
            progress_callback(total_fetched, pd.to_datetime(trades[-1]["timestamp"], unit="ms", utc=True))

        if len(pending) >= 20_000:
            store.append_trades(pd.DataFrame(pending, columns=store.TRADE_COLUMNS), path)
            pending = []

        # Unlike OHLCV candles, Binance's trades endpoint caps each `since`-based
        # page to ~1h of trades regardless of `limit` -- a short page does NOT
        # mean we've caught up to "now", so (unlike fetch_ohlcv_history) there's
        # no early-exit on a partial page here. Only `since >= now_ms` or an
        # empty response ends the loop.
        time.sleep(exchange.rateLimit / 1000)

    if pending:
        store.append_trades(pd.DataFrame(pending, columns=store.TRADE_COLUMNS), path)

    return store.load_trades(path)


def fetch_recent_second_bars(
    exchange_id: str,
    symbol: str,
    path: str,
    window_days: int = 60,
    bar_seconds: int = 1,
    limit: int = 1000,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Fetch a recent window of sub-minute bars for exit-rule validation.

    Uses the exchange's native '1s' timeframe if it exposes one; otherwise
    falls back to pulling raw trades and aggregating them locally into
    bar_seconds-wide OHLCV bars.
    """
    exchange = make_exchange(exchange_id)
    exchange.load_markets()

    if bar_seconds == 1 and "1s" in getattr(exchange, "timeframes", {}):
        since_iso = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)).isoformat()
        return fetch_ohlcv_history(
            exchange_id, symbol, "1s", path, default_since_iso=since_iso, limit=limit,
            progress_callback=progress_callback,
        )

    return _fetch_and_aggregate_trades(exchange, symbol, path, window_days, bar_seconds, progress_callback)


def _fetch_and_aggregate_trades(
    exchange,
    symbol: str,
    path: str,
    window_days: int,
    bar_seconds: int,
    progress_callback: ProgressCallback | None,
) -> pd.DataFrame:
    last_ts = store.last_timestamp_ms(path)
    window_start_ms = exchange.milliseconds() - window_days * 24 * 3600 * 1000
    since = max(last_ts, window_start_ms) if last_ts is not None else window_start_ms
    now_ms = exchange.milliseconds()

    all_trades: list = []
    while since < now_ms:
        trades = exchange.fetch_trades(symbol, since=since, limit=1000)
        if not trades:
            break
        all_trades.extend(trades)
        new_since = trades[-1]["timestamp"] + 1
        if new_since <= since:
            break
        since = new_since
        if progress_callback:
            progress_callback(len(all_trades), pd.to_datetime(trades[-1]["timestamp"], unit="ms", utc=True))
        if len(trades) < 1000:
            break
        time.sleep(exchange.rateLimit / 1000)

    if not all_trades:
        return store.load_ohlcv(path)

    trades_df = pd.DataFrame(
        [{"timestamp": t["timestamp"], "price": t["price"], "amount": t["amount"]} for t in all_trades]
    )
    bar_ms = bar_seconds * 1000
    trades_df["bar_ts"] = (trades_df["timestamp"] // bar_ms) * bar_ms
    ohlcv = (
        trades_df.groupby("bar_ts")
        .agg(open=("price", "first"), high=("price", "max"), low=("price", "min"),
             close=("price", "last"), volume=("amount", "sum"))
        .reset_index()
        .rename(columns={"bar_ts": "timestamp"})
    )
    return store.append_ohlcv(ohlcv[store.OHLCV_COLUMNS], path)
