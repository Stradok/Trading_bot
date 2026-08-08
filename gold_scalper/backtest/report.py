"""CSV trade log + printed summary + equity curve plot for a backtest run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from gold_scalper.backtest.metrics import BacktestMetrics
from gold_scalper.backtest.runner import BacktestResult


def trades_to_dataframe(result: BacktestResult) -> pd.DataFrame:
    rows = [
        {
            "lane_id": t.lane_id, "entry_ts": t.entry_ts, "exit_ts": t.exit_ts,
            "entry_price": t.entry_price, "exit_price": t.exit_price, "qty": t.qty,
            "fees": t.fees, "pnl": t.pnl, "close_reason": t.close_reason,
        }
        for t in result.portfolio.trade_log
    ]
    return pd.DataFrame(rows)


def write_trade_csv(result: BacktestResult, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    trades_to_dataframe(result).to_csv(path, index=False)


def plot_equity_curve(result: BacktestResult, path: str) -> None:
    if not result.portfolio.equity_curve:
        return
    ts = [t for t, _ in result.portfolio.equity_curve]
    equity = [e for _, e in result.portfolio.equity_curve]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts, equity)
    ax.set_title("Equity curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity (USD)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def format_summary(metrics: BacktestMetrics) -> str:
    lines = [
        f"Trades:                       {metrics.trade_count}",
        f"Win rate:                     {metrics.win_rate:.1%}",
        f"Avg win / avg loss:           {metrics.avg_win:.4f} / {metrics.avg_loss:.4f}",
        f"Expectancy (net of fees):     {metrics.expectancy_net:.4f} per trade",
        f"Gross PnL:                    {metrics.gross_pnl:.2f}",
        f"Net PnL:                      {metrics.net_pnl:.2f}",
        f"Total fees:                   {metrics.total_fees:.2f}",
        f"Profit factor:                {metrics.profit_factor:.2f}",
        f"Max drawdown:                 {metrics.max_drawdown_pct:.2%}",
        f"Max concurrent-lane drawdown: {metrics.max_concurrent_lane_drawdown_pct:.2%}",
        f"Avg holding time:             {metrics.avg_holding_seconds:.0f}s",
        f"Span:                         {metrics.span_days:.1f} days",
        f"Trades/day:                   {metrics.trades_per_day:.2f}",
        f"Net profit/day:               {metrics.net_profit_per_day:.4f}",
        f"Circuit breaker trips:        {metrics.circuit_breaker_trip_count} ({metrics.circuit_breaker_halted_bars} bars halted)",
        "Lane utilization (bars / net pnl attributed at entry):",
    ]
    for count, stats in sorted(metrics.lane_utilization.items()):
        lines.append(f"  {count} lane(s): {stats['bars']} bars, {stats['net_pnl']:.2f} net pnl")
    return "\n".join(lines)
