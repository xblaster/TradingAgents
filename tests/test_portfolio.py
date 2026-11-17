from datetime import date

import numpy as np
import pandas as pd
import pytest

from trading_desk.portfolio import (
    TrendConfig,
    equal_weight_targets,
    portfolio_metrics,
    simulate_portfolio,
    single_asset_targets,
    target_weights,
    trend_signals,
)

IDX = pd.date_range("2024-01-01", periods=400, freq="D")


def trending(drift: float, vol: float = 0.02, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(drift + rng.normal(0, vol, len(IDX)))), index=IDX)


def test_signal_is_one_in_uptrend_zero_in_downtrend():
    closes = pd.DataFrame({"UP": 100 * 1.01 ** np.arange(400), "DOWN": 100 * 0.99 ** np.arange(400)}, index=IDX)
    s = trend_signals(closes, (20, 60, 120, 250))
    assert s["UP"].iloc[-1] == 1 and s["DOWN"].iloc[-1] == 0
    assert s["UP"].iloc[:250].isna().all()


def test_weights_are_inverse_vol_times_signal():
    a = pd.Series(100 * 1.002 ** np.arange(400), index=IDX) * (1 + 0.01 * np.sin(np.arange(400)))
    b = pd.Series(100 * 1.002 ** np.arange(400), index=IDX) * (1 + 0.02 * np.sin(np.arange(400)))
    w = target_weights(pd.DataFrame({"A": a, "B": b}), TrendConfig()).iloc[-1]
    assert w.sum() == pytest.approx(1.0, abs=1e-9)  # both fully trending
    assert w["A"] / w["B"] == pytest.approx(2.0, rel=0.1)  # half the vol, twice the weight


def test_weights_never_exceed_full_exposure_and_respect_history():
    closes = pd.DataFrame({"A": trending(0.003, seed=1), "B": trending(-0.002, seed=2), "C": trending(0.001, seed=3)})
    closes.loc[: IDX[200], "C"] = np.nan  # listed later
    w = target_weights(closes, TrendConfig())
    assert (w.sum(axis=1) <= 1 + 1e-9).all() and (w >= 0).all().all()
    assert (w["C"] == 0).all()  # never reaches 250 bars of history
    assert (w.iloc[:250] == 0).all().all()


def test_targets_do_not_use_future_data():
    closes = pd.DataFrame({"A": trending(0.002, seed=4), "B": trending(0.001, seed=5)})
    base = target_weights(closes, TrendConfig())
    shocked = closes.copy()
    shocked.iloc[300:] *= 0.3  # crash after day 299
    pd.testing.assert_frame_equal(base.iloc[:300], target_weights(shocked, TrendConfig()).iloc[:300])


def frames(rows: dict[str, list[tuple[float, float]]]):
    idx = pd.date_range("2026-01-01", periods=len(next(iter(rows.values()))), freq="D")
    opens = pd.DataFrame({k: [r[0] for r in v] for k, v in rows.items()}, index=idx)
    closes = pd.DataFrame({k: [r[1] for r in v] for k, v in rows.items()}, index=idx)
    return opens, closes


def test_simulator_fills_next_open_and_charges_costs():
    opens, closes = frames({"X": [(100, 100), (102, 105), (105, 110), (110, 108)]})
    run = simulate_portfolio(opens, closes, single_asset_targets(closes, "X"), date(2026, 1, 1), date(2026, 1, 4), every=1, min_trade=0.02, cost_bps=10)
    assert run.equity[-1] == pytest.approx((1 - 0.001) * 108 / 102)
    assert run.trades == 1 and run.fees == pytest.approx(0.001)


def test_simulator_two_assets_half_half():
    opens, closes = frames({"X": [(100, 100), (100, 110), (110, 110)], "Y": [(50, 50), (50, 45), (45, 45)]})
    targets = equal_weight_targets(closes, TrendConfig(min_history=1))
    run = simulate_portfolio(opens, closes, targets, date(2026, 1, 1), date(2026, 1, 3), every=10, min_trade=0.02, cost_bps=0)
    assert run.equity[-1] == pytest.approx(0.5 * 1.1 + 0.5 * 0.9)
    assert run.exposure[-1] == pytest.approx(1.0)


def test_exit_to_cash_protects_from_crash():
    opens, closes = frames({"X": [(100, 100), (100, 100), (100, 50), (50, 50)]})
    targets = pd.DataFrame({"X": [1.0, 0.0, 0.0, 0.0]}, index=closes.index)
    run = simulate_portfolio(opens, closes, targets, date(2026, 1, 1), date(2026, 1, 4), every=1, min_trade=0.02, cost_bps=0)
    assert run.equity[-1] == pytest.approx(1.0)  # bought day 2 open at 100, sold day 3 open at 100


def test_metrics():
    from trading_desk.portfolio import PortfolioRun

    run = PortfolioRun(equity=[1.0, 1.2, 0.6, 0.9], exposure=[0, 1, 1, 1])
    m = portfolio_metrics(run)
    assert m["max_drawdown"] == pytest.approx(-0.5) and m["total_return"] == pytest.approx(-0.1)


def test_scaled_config():
    assert TrendConfig().scaled(0.5).lookbacks == (10, 30, 60, 125)
