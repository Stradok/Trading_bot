# Gold Scalper — Plan & Technology Reference

This is the reference document for the whole project: what we're building,
why, what technology it's built on, and where things currently stand. See
[`README.md`](README.md) for day-to-day usage commands.

## The idea

From [`Idea.md`](Idea.md) ("Idea 1"): open a small position, lock in a
profit once the price starts reversing, wait, re-enter on a fresh
micro-trend, repeat. As profit accumulates, run more *concurrent* attempts
at this loop (not bigger individual bets), so many small wins compound into
a meaningful total.

The raw idea had real gaps that shaped everything below: no defined entry
trigger, a reversal threshold (0.0001%) smaller than actual market noise,
no accounting for trading fees, and no acknowledgment that running several
concurrent trades on the same asset is correlated risk, not
diversification. Every piece of the architecture exists to fix one of
these.

## Hard constraints

- **Halal / Sharia-compliant, permanently.** Spot trading only. No margin,
  no futures, no leverage — ever. Not a "v1 simplification," a permanent
  design boundary. Leverage/CFDs were ruled out specifically because of
  riba (interest via funding/swap rates), gharar (excessive uncertainty),
  and maysir (gambling-like speculation). This is also why the strategy is
  long-only: spot has no short side.
- **Small starting capital**, roughly $10 to start, growing from there.
- **Paper trading first.** No real order placement exists in this codebase
  yet, on purpose — live trading is a distinct future phase, built only
  after paper trading validates the strategy.
- **Exchange: Binance spot, PAXG/USDT.** Chosen over MetaTrader/CFD-style
  brokers for two reasons: (1) crypto exchanges don't throttle or ban
  accounts for running profitable algorithmic strategies the way retail
  FX/CFD market-maker brokers sometimes do, and (2) MetaTrader's usual
  crypto-adjacent product (CFDs, margin) reintroduces the same riba/gharar
  problem that ruled out leverage in the first place. Kraken (also lists
  PAXG spot) is a documented fallback if needed.

## Technology stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12+ | Best ecosystem fit for both data work (pandas) and exchange APIs (ccxt); GPU acceleration isn't relevant here — see note below |
| Package/env management | [`uv`](https://github.com/astral-sh/uv) | Fast, reliable dependency resolution and virtualenv management |
| Exchange access | [`ccxt`](https://github.com/ccxt/ccxt) | Unified API across exchanges (Binance today, Kraken as a documented fallback) — same code works against either with a one-line config change |
| Data handling | `pandas` + `pyarrow` (parquet storage) | Standard for tabular time-series work; parquet is compact and fast to reload |
| Config | `PyYAML` + plain dataclasses | Human-editable config files, validated fail-fast in code (`gold_scalper/config.py`) rather than trusted blindly |
| Charting/reports | `matplotlib` | Equity curve plots, kept simple (static PNGs, not a live dashboard, for now) |
| Testing | `pytest` | 31 tests currently, covering every module below |
| Strategy approach | Rule-based signals + walk-forward parameter search | Deliberately *not* a black-box ML model for v1 — see "Why not ML yet" below |

**Why no GPU:** this is a sequential simulation, not a parallel numeric
workload. Each bar's outcome depends on the exact state left by the
previous bar (open positions, cooldown timers, trailing-stop peaks,
equity-driven lane count) — a chain of dependencies, which is the opposite
of what GPUs accelerate (many independent operations at once, like matrix
math). It runs on one CPU core; RAM usage is just the price history sitting
in memory as a pandas table. GPU would become relevant only if we later
train an actual ML model (see below).

**Why not an ML model yet:** the current strategy is interpretable —
breakout/momentum entry, fee-and-volatility-scaled exit — with thresholds
*fit from data* via walk-forward optimization rather than a black box. This
was a deliberate choice over jumping straight to ML: it's auditable (you
can see why a trade happened), a single symbol's history is a fairly small
and correlated dataset to train a robust model on without it just
memorizing noise, and it lets us find out cheaply whether the underlying
idea has any edge at all before investing in something more complex. If
walk-forward search finds no profitable parameter region across a properly
tuned rule structure, that's the signal to reach for something like a
gradient-boosted classifier as a next iteration — not before.

## Architecture

```
gold_scalper/
  config.py           # YAML -> dataclasses, fail-fast validation
  data/
    fetch.py            # resumable historical OHLCV puller
    store.py             # parquet read/append/dedupe
    spread_calibration.py # real bid/ask spread distribution from the order book
    live_feed.py          # REST polling for paper trading
  strategy/
    signals.py           # entry rule: ATR-scaled breakout + EMA slope
    exits.py              # exit rule: fee/spread-aware trailing profit-lock
    features.py            # combines the above + circuit-breaker baselines + spread column
  engine/
    execution.py          # fill/fee/slippage simulator
    portfolio.py            # cash/position ledger, mark-to-market equity
    lane.py                  # one watch/enter/exit/cooldown cycle (state machine)
    lane_manager.py           # owns all lanes, decides concurrency, runs the circuit breaker
    circuit_breaker.py         # halts new entries / flattens positions on vol/volume spikes or drawdown
  backtest/
    runner.py             # event-driven loop, no-lookahead (fills on bar t+1, never bar t's own close)
    metrics.py               # win rate, expectancy net of fees, drawdown, lane utilization
    optimize.py                # walk-forward parameter search, out-of-sample only
    projection.py                # revenue projection: local part-time vs 24/7, as a range not a point estimate
    report.py                     # CSV + printed summary + equity curve plot
  live/
    paper_runner.py       # drives the *same* engine objects as the backtester, fed by live data
scripts/                 # CLI entry points: fetch_history, calibrate_spread, run_backtest, run_optimize, run_paper
config/                  # default.yaml + backtest.yaml/paper.yaml overrides
tests/                   # one file per module above, 31 tests
```

The one deliberate abstraction in the whole codebase: `backtest/runner.py`
and `live/paper_runner.py` both drive the identical `engine/*` objects, so
fee/fill/risk logic is written once. Paper trading *is* the backtest
engine, just pointed at a live feed instead of stored history — this is
what makes a later live/paper divergence meaningful (it means a backtest
assumption was wrong, not that the code is different).

## Key design decisions made during implementation (not just planning)

A few things changed or were caught while actually building and testing
this, worth recording because they weren't obvious upfront:

- **Lane scaling is a growth *multiple* of starting capital, not an
  absolute dollar threshold.** Originally `lane_equity_steps` used raw
  dollar amounts (e.g. "$150 -> 3 lanes"), which meant a $150 starting
  balance would unlock 3 concurrent lanes immediately, on bar one — before
  any profit was actually earned. That contradicts the whole point of the
  original idea (scale up *because* you've made money). Fixed so
  thresholds are multiples of whatever you started with (e.g. `1.5` = grown
  to 1.5x starting capital), independent of the dollar amount.
- **Cash-sufficiency check before entering a trade.** The portfolio didn't
  originally check whether there was enough cash before buying — it would
  silently go negative. Matters most at small capital ($10): fixed so a
  lane simply skips an entry it can't afford.
- **Circuit breaker thresholds were recalibrated against real data.**
  Volume on PAXG/USDT is naturally bursty; a 4x-average spike threshold
  fired on ~5.5% of ordinary bars in real data (checked 2026-08-08) — way
  too often to represent a genuine anomaly. Raised to 10x based on the
  actual observed distribution (real p99 was ~11x).
- **No-lookahead is enforced structurally, not just by convention**: the
  backtest runner passes bar *t* as the "decision" row and bar *t+1* as the
  "fill" row into the engine, so a signal can never execute against a price
  it couldn't have had. Live/paper trading collapses these to the same row
  (no future bar exists yet in real time).

## Current status (as of 2026-08-08)

Built and tested:
- Full data pipeline: ~6 years of 1-minute PAXG/USDT history (~3.1M bars,
  2020-08 to today) and a 60-day window of 1-second bars, both fetched from
  the real Binance API. Spread calibration smoke-tested against the live
  order book.
- The full strategy/engine/backtest/paper-trading stack described above,
  31 passing tests.

In progress:
- Walk-forward parameter optimization (finding real, data-fit thresholds
  instead of the hand-picked seed values) is running now against the last
  12 months of data. A smoke test with the *unoptimized* seed parameters
  lost money over a week of real data — expected, and exactly why this
  step exists rather than trusting guessed numbers.

Not started yet:
- Paper trading run against live data (needs the optimizer's results
  first, so it's testing tuned parameters, not seed guesses).
- Live trading phase: real order placement, API key handling, rate-limit
  backoff beyond `ccxt`'s built-in throttling, crash recovery, alerting.
  Deliberately not designed in detail yet — the right design for this
  depends on what paper trading actually shows.

## Roadmap

1. Finish walk-forward optimization on historical data (in progress).
2. If a profitable, consistent (low-variance across test windows)
   parameter region exists: run paper trading live for a meaningful
   stretch (days, not minutes) and compare its results against what the
   backtest predicted for the same period. A mismatch means a backtest
   assumption (spread, slippage, fee) needs recalibrating.
3. Repeat paper trading validation across different market conditions
   before trusting it with any real capital — one good week isn't enough
   evidence for a strategy this parameter-sensitive.
4. Only after that: design the live-trading phase, with explicit
   safeguards (order-frequency caps, the existing lane cap, a kill switch,
   a trade-only/no-withdrawal API key) built in from the start rather than
   bolted on.
5. Start live with the smallest possible real capital (~$10) and scale up
   only as real (not backtested) performance earns it.

## Performance notes

The backtest engine processes bars sequentially at roughly 2,000-2,500
bars/second (Python-level iteration, not vectorized — a deliberate
tradeoff for correctness/readability of a genuinely sequential state
machine). A full pass over ~6 years of 1-minute data takes ~20-25 minutes;
walk-forward optimization multiplies that by the parameter grid size, so a
full multi-parameter search over the full history would take many hours
and should run as a long background job, not interactively. If this
becomes a real bottleneck later, the likely fix is switching the hot loop
from row-by-row pandas access to a columnar/numpy representation, not GPU
acceleration (see "Why no GPU" above) — that's a known, deferred
optimization, not something needed yet.
