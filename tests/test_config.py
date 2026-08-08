import pytest

from gold_scalper.config import load_config


def test_default_config_loads_with_calibrated_spread():
    # config/default.yaml has a real (if preliminary) spread_bps_p75 from
    # scripts/calibrate_spread.py, so this should load without disabling the check.
    cfg = load_config("config/default.yaml")
    assert cfg.spread.spread_bps_p75 > 0


def test_uncalibrated_spread_is_rejected():
    cfg = load_config("config/default.yaml", require_calibrated_spread=False)
    cfg.spread.spread_bps_p75 = 0.0
    with pytest.raises(ValueError, match="uncalibrated"):
        cfg.validate(require_calibrated_spread=True)


def test_default_config_loads_with_spread_check_disabled():
    cfg = load_config("config/default.yaml", require_calibrated_spread=False)
    assert cfg.exchange.symbol == "PAXG/USDT"
    assert cfg.lanes.max_lanes == 4
    assert cfg.lanes.lane_equity_steps[0] == (1.0, 1)


def test_backtest_config_merges_data_paths():
    cfg = load_config("config/default.yaml", "config/backtest.yaml", require_calibrated_spread=False)
    assert cfg.data is not None
    assert cfg.data.ohlcv_1m_path == "data/ohlcv/PAXG_USDT_1m.parquet"
    assert cfg.live is None


def test_paper_config_merges_live_settings():
    cfg = load_config("config/default.yaml", "config/paper.yaml", require_calibrated_spread=False)
    assert cfg.live is not None
    assert cfg.live.timeframe == "1m"
    assert cfg.live.active_hours.timezone == "UTC"


def test_bad_lane_equity_steps_rejected():
    cfg = load_config("config/default.yaml", require_calibrated_spread=False)
    cfg.lanes.lane_equity_steps = [(1.0, 1), (3.0, 2), (2.0, 3)]
    with pytest.raises(ValueError, match="sorted ascending"):
        cfg.validate(require_calibrated_spread=False)
