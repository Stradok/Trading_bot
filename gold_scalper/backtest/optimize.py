"""Walk-forward parameter search: this is the actual mechanism behind
"thresholds learned from data" rather than hand-picked constants. Rolling
train/test windows are stepped across the full stored history; parameters
are only ever chosen on a train window and scored on the *next*, untouched
test window, so the reported performance is out-of-sample, not an in-sample
fit that would just be overfitting with extra steps.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from gold_scalper.backtest.metrics import BacktestMetrics, compute_metrics
from gold_scalper.backtest.runner import BacktestResult, run_backtest
from gold_scalper.config import Config


def _set_param(cfg: Config, dotted_path: str, value: Any) -> None:
    section, field = dotted_path.split(".")
    setattr(getattr(cfg, section), field, value)


def _score(metrics: BacktestMetrics, drawdown_penalty: float, min_trades: int) -> float:
    """Maximize net expectancy per trade, penalized for max drawdown so the
    search can't win by taking on more risk. A minimum trade count guards
    against picking a "great" expectancy that's really just 1-2 lucky trades.
    """
    if metrics.trade_count < min_trades:
        return float("-inf")
    return metrics.expectancy_net - drawdown_penalty * metrics.max_drawdown_pct


@dataclass
class WindowResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict[str, Any]
    train_metrics: BacktestMetrics | None
    test_metrics: BacktestMetrics
    test_result: "BacktestResult"  # kept for trade-level analysis, e.g. backtest/projection.py's active-hours scenario


def walk_forward_optimize(
    bars: pd.DataFrame,
    base_cfg: Config,
    param_grid: dict[str, list[Any]],
    train_days: int = 180,
    test_days: int = 60,
    starting_cash: float = 200.0,
    drawdown_penalty: float = 2.0,
    min_trades: int = 5,
    progress_callback: Callable[[str, dict], None] | None = None,
) -> list[WindowResult]:
    """progress_callback(event: str, info: dict), if given, is called after
    every train-combo backtest and after every window completes -- long
    walk-forward runs can take much longer than a naive bars/sec estimate
    suggests (real throughput depends heavily on trade frequency in the
    slice), so don't run one of these blind without a way to see how far
    along it actually is.
    """
    results: list[WindowResult] = []
    start, end = bars.index[0], bars.index[-1]
    train_delta, test_delta = pd.Timedelta(days=train_days), pd.Timedelta(days=test_days)

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values())) if param_grid else [()]
    total_windows = int((end - start - train_delta - test_delta) / test_delta) + 1 if end > start + train_delta + test_delta else 0

    window_start = start
    window_index = 0
    while window_start + train_delta + test_delta <= end:
        window_index += 1
        train_start, train_end = window_start, window_start + train_delta
        test_start, test_end = train_end, train_end + test_delta
        train_bars = bars.loc[train_start:train_end]
        test_bars = bars.loc[test_start:test_end]

        best_score, best_params, best_train_metrics = float("-inf"), {}, None
        for combo_i, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            cfg = copy.deepcopy(base_cfg)
            for path, value in params.items():
                _set_param(cfg, path, value)
            cfg.validate()

            metrics = compute_metrics(run_backtest(train_bars, cfg, starting_cash=starting_cash))
            score = _score(metrics, drawdown_penalty, min_trades)
            if score > best_score:
                best_score, best_params, best_train_metrics = score, params, metrics
            if progress_callback:
                progress_callback("train_combo", {
                    "window": window_index, "total_windows": total_windows,
                    "combo": combo_i + 1, "total_combos": len(combos), "params": params,
                })

        test_cfg = copy.deepcopy(base_cfg)
        for path, value in best_params.items():
            _set_param(test_cfg, path, value)
        test_cfg.validate()
        test_result = run_backtest(test_bars, test_cfg, starting_cash=starting_cash)
        test_metrics = compute_metrics(test_result)

        results.append(WindowResult(
            train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end,
            best_params=best_params, train_metrics=best_train_metrics, test_metrics=test_metrics,
            test_result=test_result,
        ))
        if progress_callback:
            progress_callback("window_done", {
                "window": window_index, "total_windows": total_windows, "result": results[-1],
            })
        window_start += test_delta  # rolling, non-overlapping test windows

    return results


def summarize_out_of_sample(results: list[WindowResult]) -> dict:
    """Aggregate out-of-sample (test-window) performance. Prefer this over any
    single window's number -- the point of walk-forward is that no individual
    window's result should be trusted in isolation.
    """
    if not results:
        return {}
    expectancies = [r.test_metrics.expectancy_net for r in results]
    profit_per_day = [r.test_metrics.net_profit_per_day for r in results]
    return {
        "n_windows": len(results),
        "mean_expectancy_net": sum(expectancies) / len(expectancies),
        "min_expectancy_net": min(expectancies),
        "max_expectancy_net": max(expectancies),
        "mean_net_profit_per_day": sum(profit_per_day) / len(profit_per_day),
        "windows_profitable": sum(1 for e in expectancies if e > 0),
    }
