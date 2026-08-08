import pandas as pd

from gold_scalper.config import SignalConfig
from gold_scalper.strategy.signals import compute_features, entry_signal

CFG = SignalConfig(
    ema_fast_periods=3, breakout_lookback=5, atr_periods=3,
    breakout_atr_mult=0.2, slope_lookback=2, min_slope=0.0,
)


def make_bars(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [1.0] * len(closes),
    })


def signal_series(closes: list[float], cfg: SignalConfig = CFG) -> list[bool]:
    bars = compute_features(make_bars(closes), cfg)
    return [entry_signal(bars.iloc[i], cfg) for i in range(len(bars))]


def test_no_entry_signal_during_flat_warmup():
    signals = signal_series([100.0] * 12)
    assert not any(signals)


def test_entry_signal_fires_on_breakout_quiets_on_pullback_then_fires_on_fresh_breakout():
    # flat warmup (0-9), breakout (10-14), pullback that stays below the prior
    # high (15-19), fresh breakout above the new high (20).
    closes = [100.0] * 10 + [100, 101, 108, 108.5, 109] + [104, 103, 102, 103, 104] + [115]
    signals = signal_series(closes)

    assert not any(signals[:11])          # warmup + first post-warmup bar: no signal yet
    assert all(signals[11:15])            # breakout run fires every bar
    assert not any(signals[15:20])        # pullback: quiet
    assert signals[20]                    # fresh breakout above the new high: fires


def test_entry_signal_false_during_insufficient_warmup():
    # fewer bars than breakout_lookback -- roll_high is NaN, must not raise or signal True
    signals = signal_series([100.0, 101.0, 102.0])
    assert not any(signals)
