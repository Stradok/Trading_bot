# Gold Scalper

A scalping bot for tokenized gold (PAXG/USDT) on Binance spot, implementing
"Idea 1" from [`Idea.md`](Idea.md): open small positions, lock in a profit
once it starts reversing, wait for a fresh micro-trend, repeat -- and scale
up the number of *concurrent* trades (not trade size) as profit accumulates,
so many small wins compound into a meaningful total.

## Constraints this is built around

- **Halal / Sharia-compliant, permanently**: spot only. No margin, no
  futures, no leverage, ever -- not a v1 simplification, a hard design
  constraint (leverage/CFDs involve riba via interest/funding-rate
  mechanisms, gharar, and maysir). This also means the strategy is
  inherently long-only, since spot has no short side.
- **Small capital**: designed to start around $10 and grow from there.
  Position size per lane is fixed and small; scaling happens only by adding
  more concurrent lanes, capped by `lanes.max_lanes` and Binance's minimum
  notional per trade.
- **Fee-and-spread aware**: every threshold (entry, exit, projected
  profitability) is computed against real fee/spread numbers, not
  guesses -- see [`config/default.yaml`](config/default.yaml).
- **Paper trading before live**: there is currently no code path anywhere in
  this repo that can place a real order. `ExecutionSimulator` only
  *simulates* fills. Live trading is a deliberate, not-yet-built future
  phase (see Status below).

## Status

All of the following is built and covered by automated tests
(`uv run pytest tests/ -q`):

| Phase | What it does |
|---|---|
| Data layer | Resumable historical OHLCV fetch, order-book spread calibration, live market data polling -- all smoke-tested against the real Binance API |
| Strategy | ATR/breakout entry signal, fee-and-spread-aware trailing exit (replaces the original idea's raw "0.0001%" threshold, which was smaller than real market noise) |
| Engine | Execution/fee/slippage simulator, mark-to-market portfolio, lane state machine, equity-scaled lane manager, circuit breaker for correlated-lane risk |
| Backtester | No-lookahead event loop (a decision on bar *t* fills against bar *t+1*, never its own close), performance metrics net of fees, walk-forward parameter optimization, profit projection |
| Paper trading | Runs the *exact same* engine as the backtester against live data, with zero real orders ever placed |

**Not built yet, on purpose:** real order placement, API key/secret
handling beyond public market-data endpoints, exchange rate-limit backoff
beyond `ccxt`'s built-in throttling, crash recovery/reconciliation with a
real exchange account, WebSocket streaming, alerting. These are a separate
future phase, deliberately scoped out until paper trading has validated the
strategy over a meaningful period.

**Where this actually stands right now:** historical data is fetched
(~6 years of 1-minute PAXG/USDT bars), and walk-forward parameter
optimization against that data is the current step -- finding out whether,
and under what parameters, this strategy is actually profitable net of
fees, before any paper-trading or live-trading capital is at risk. Early
smoke tests with hand-picked (non-optimized) parameters lost money, which
is expected and exactly why the optimizer exists rather than trusting
guessed thresholds.

## Setup

```bash
uv sync
```

Requires Python >=3.12. No API keys are needed for anything in this repo
today -- data fetching, spread calibration, backtesting, and paper trading
all use Binance's public (unauthenticated) market data endpoints.

## Usage

```bash
# 1. Pull historical OHLCV data (resumable -- safe to re-run, only fetches the gap)
uv run scripts/fetch_history.py --timeframe 1m
uv run scripts/fetch_history.py --timeframe 1s --window-days 60

# 2. Calibrate the real bid/ask spread (run for hours/days for a trustworthy number;
#    a short run is enough to smoke-test the pipeline)
uv run scripts/calibrate_spread.py --duration-seconds 3600 --write-config

# 3. Run a single backtest with the current config/default.yaml parameters
uv run scripts/run_backtest.py --starting-cash 10

# 4. Walk-forward parameter search: finds thresholds from data instead of
#    guessing them, evaluated out-of-sample on rolling train/test windows.
#    This is expensive -- see the runtime note below.
uv run scripts/run_optimize.py --train-days 90 --test-days 30 --starting-cash 10

# 5. Paper trade against live data (no real orders are ever placed)
uv run scripts/run_paper.py --starting-cash 10
```

Reports (equity curve PNG, trade CSV, walk-forward window CSV) are written
to `data/reports/`.

### A note on walk-forward runtime

The backtest engine processes bars sequentially in Python (not vectorized),
at roughly 2,000-2,500 bars/second. A full pass over the ~6 years of stored
1-minute data takes about 20-25 minutes; walk-forward optimization runs
train+test backtests for every parameter combination across every rolling
window, so runtime scales with `len(param_grid) x dataset_span`. A 3-value,
single-parameter grid over the last 12 months takes on the order of 30-45
minutes; the full multi-parameter grid over the full history would take
many hours and should be kicked off as a long background job, not run
interactively.

## Configuration

See [`config/default.yaml`](config/default.yaml) for all tunable
parameters, with [`config/backtest.yaml`](config/backtest.yaml) and
[`config/paper.yaml`](config/paper.yaml) layering overrides on top (see
`gold_scalper/config.py:load_config`). A few things worth understanding
before changing numbers:

- `lanes.lane_equity_steps` are **growth multiples of starting capital**,
  not dollar amounts (e.g. `1.2` means "equity has grown to 1.2x what you
  started with") -- this is what makes lane scaling reflect profit actually
  earned regardless of how much capital you start with.
- `lanes.position_size_usd` is fixed per lane; if it exceeds available cash,
  the lane simply skips the entry rather than overdrawing (see
  `engine/lane.py:_can_afford`). The default ($8) is sized to work with a
  $10 starting balance.
- `spread.spread_bps_p75` and `circuit_breaker.vol_spike_mult` /
  `volume_spike_mult` should be recalibrated against real data for whatever
  window you're trading in -- the shipped values were checked against real
  PAXG/USDT data on 2026-08-08 but markets change.
- `config.validate()` fails fast on internally inconsistent config (e.g. an
  uncalibrated spread, a lane count exceeding `max_lanes`, thresholds that
  wouldn't actually protect against anything) rather than running silently
  with numbers that don't make sense.

## Architecture

```
gold_scalper/
  config.py           # YAML -> dataclasses, fail-fast validation
  data/                # fetch, store, spread calibration, live feed
  strategy/            # entry signal, exit rule, feature computation
  engine/              # execution sim, portfolio, lane state machine, circuit breaker
  backtest/            # event-driven runner, metrics, walk-forward optimizer, projection
  live/                # paper trading runner (reuses the backtest engine)
scripts/                # CLI entry points for everything above
config/                 # default + backtest/paper overrides
tests/                  # one test file per module above
```

`backtest/runner.py` and `live/paper_runner.py` both drive the same
`engine/*` objects, so fee/fill/risk logic is single-sourced between
backtesting and paper trading -- paper trading *is* the backtest engine,
just pointed at a live feed instead of stored history.
