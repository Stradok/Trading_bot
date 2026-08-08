import pandas as pd

from gold_scalper.config import ExitConfig
from gold_scalper.strategy.exits import EXIT_PROFIT, EXIT_STOP_LOSS, EXIT_TIMEOUT, TrailingExit

CFG = ExitConfig(
    taker_fee_rate=0.001, min_net_profit_pct=0.0005,
    trail_atr_mult=0.5, trail_spread_mult=1.5,
    stop_loss_atr_mult=1.0, max_hold_seconds=1800,
)
T0 = pd.Timestamp("2026-01-01T00:00:00Z")


def test_no_exit_before_arming_and_no_premature_trigger():
    # round-trip cost here is ~0.22% (2*0.1% fee + 0.02/100 spread), so the
    # exit shouldn't arm until unrealized profit clears ~0.27% (cost + buffer)
    exit_ = TrailingExit(entry_price=100.0, entry_ts=T0, cfg=CFG)
    assert exit_.update(price=100.1, atr=0.1, spread=0.02, now=T0) is None
    assert not exit_.armed
    assert exit_.update(price=100.2, atr=0.1, spread=0.02, now=T0) is None
    assert not exit_.armed


def test_arms_then_exits_on_pullback_from_peak():
    exit_ = TrailingExit(entry_price=100.0, entry_ts=T0, cfg=CFG)
    assert exit_.update(price=100.1, atr=0.1, spread=0.02, now=T0) is None
    assert exit_.update(price=100.2, atr=0.1, spread=0.02, now=T0) is None
    # 100.3 clears the ~0.27% arm threshold -- arms, but trail_dist=0.05 means
    # peak(100.3) - 0.05 = 100.25, which price hasn't breached yet this bar
    assert exit_.update(price=100.3, atr=0.1, spread=0.02, now=T0) is None
    assert exit_.armed
    assert exit_.update(price=100.4, atr=0.1, spread=0.02, now=T0) is None
    # pulls back to exactly peak(100.4) - trail_dist(0.05) = 100.35 -> exits
    assert exit_.update(price=100.35, atr=0.1, spread=0.02, now=T0) == EXIT_PROFIT


def test_stop_loss_fires_on_adverse_move_without_ever_arming():
    exit_ = TrailingExit(entry_price=100.0, entry_ts=T0, cfg=CFG)
    # stop level = entry(100) - stop_loss_atr_mult(1.0)*atr(0.1) = 99.9
    result = exit_.update(price=99.85, atr=0.1, spread=0.02, now=T0)
    assert result == EXIT_STOP_LOSS
    assert not exit_.armed


def test_timeout_fires_when_max_hold_exceeded_without_profit_or_stop():
    exit_ = TrailingExit(entry_price=100.0, entry_ts=T0, cfg=CFG)
    later = T0 + pd.Timedelta(seconds=CFG.max_hold_seconds + 1)
    result = exit_.update(price=100.05, atr=0.1, spread=0.02, now=later)
    assert result == EXIT_TIMEOUT
