import json
from datetime import date

import pandas as pd
import pytest

from trading_desk.backtest import BASELINES, Signal, baseline, metrics, run_backtest, simulate
from trading_desk.config import DeskSettings
from trading_desk.pipeline import TradingDesk

from .conftest import FakeBackend, FakeProvider, make_bars


def bars_from(rows):
    """rows: (open, high, low, close)"""
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="D")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx).assign(Volume=1.0)


def const(target, stop=None):
    async def decide(i, hist, w):
        return Signal(target, stop)
    return decide


ROWS = [(100, 101, 99, 100), (102, 106, 101, 105), (105, 111, 104, 110), (110, 112, 107, 108)]


async def test_full_position_fills_at_next_open():
    run = await simulate(bars_from(ROWS), 0, const(1.0), every=1, cost_bps=0)
    # invested from the open of bar 1 (102) to the close of bar 3 (108)
    assert run.curve[-1] == pytest.approx(108 / 102)
    assert run.trades == 1 and run.decisions == 3


async def test_costs_are_charged_on_turnover():
    run = await simulate(bars_from(ROWS), 0, const(0.5), every=10, cost_bps=10)
    w0 = 0.5
    expected = (1 - w0 * 0.001) * (1 + w0 * (105 / 102 - 1))  # bar 1 after the fill
    assert run.curve[1] == pytest.approx(expected)
    assert run.fees == pytest.approx(0.0005)


async def test_small_drift_is_not_rebalanced():
    rows = [(100, 100.5, 99.5, 100 + 0.2 * k) for k in range(6)]
    run = await simulate(bars_from(rows), 0, const(0.5), every=1, cost_bps=10)
    assert run.trades == 1


async def test_large_drift_is_rebalanced():
    rows = [(100, 101, 99, 100), (100, 131, 99, 130), (130, 131, 129, 130), (130, 131, 129, 130)]
    run = await simulate(bars_from(rows), 0, const(0.5), every=1, cost_bps=0)
    assert run.trades == 2  # initial fill, then trim the 30% winner back to 50%
    assert run.weights[-1] == pytest.approx(0.5)


async def test_flat_strategy_does_nothing():
    run = await simulate(bars_from(ROWS), 0, const(0.0), every=1, cost_bps=10)
    assert run.curve == [1.0] * 4 and run.trades == 0


async def test_stop_loss_exits_at_stop():
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 100, 90, 92), (92, 93, 91, 92)]
    run = await simulate(bars_from(rows), 0, const(1.0, stop=95), every=10, cost_bps=0)
    assert run.stop_hits == 1
    assert run.curve[-1] == pytest.approx(0.95)  # entered at 100, out at 95, then flat


async def test_gap_through_stop_exits_at_open():
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (90, 91, 88, 89), (89, 90, 88, 89)]
    run = await simulate(bars_from(rows), 0, const(1.0, stop=95), every=10, cost_bps=0)
    assert run.curve[-1] == pytest.approx(0.90)


async def test_deciders_never_see_the_future():
    seen = []

    async def spy(i, hist, w):
        seen.append((i, hist.index[-1], len(hist)))
        return Signal(0.0)

    bars = make_bars(50)
    await simulate(bars, 10, spy, every=3, cost_bps=0)
    assert all(last == bars.index[i] and n == i + 1 for i, last, n in seen)


async def test_trend_baseline_follows_the_trend():
    up = make_bars(200, drift=0.01, seed=1)
    down = make_bars(200, drift=-0.01, seed=1)
    trend = baseline("trend_sma", 0.5, 365)
    assert (await trend(199, up, 0)).target == 0.5
    assert (await trend(199, down, 0)).target == 0.0
    mom = baseline("momentum", 1.0, 365)
    assert (await mom(199, up, 0)).target == 1.0 and (await mom(199, down, 0)).target == 0.0


def test_metrics_on_known_curve():
    from trading_desk.backtest import StrategyRun

    m = metrics(StrategyRun(curve=[1.0, 1.1, 0.99, 1.2], weights=[0, 1, 1, 1]), 365)
    assert m["total_return"] == pytest.approx(0.2)
    assert m["max_drawdown"] == pytest.approx(0.99 / 1.1 - 1)
    assert m["cagr"] is None  # too short to annualize honestly


async def test_baselines_only_backtest_on_crypto():
    provider = FakeProvider(make_bars(700, start=date(2024, 1, 1)))
    res = await run_backtest(None, provider, "BTC-EUR", date(2025, 6, 1), date(2025, 11, 30), strategies=BASELINES, max_position_pct=50)
    assert res.asset_class == "crypto" and res.cost_bps == 15
    assert set(res.strategies) == {*BASELINES, "buy_hold_100"}
    assert res.strategies["buy_hold"]["avg_exposure"] == pytest.approx(0.5, abs=0.05)
    assert res.strategies["buy_hold_100"]["avg_exposure"] == pytest.approx(1.0, abs=0.05)
    assert res.warnings == [] and res.decisions == []
    assert len(res.equity) == len(provider.bars.loc["2025-06-01":"2025-11-30"])


async def test_llm_desk_backtest_with_cutoff(tmp_path):
    provider = FakeProvider(make_bars(700, start=date(2024, 1, 1)))
    s = DeskSettings()
    s.llm.knowledge_cutoff = date(2025, 5, 1)
    desk = TradingDesk(s, FakeBackend(), provider)
    res = await run_backtest(desk, provider, "ETH-EUR", date(2025, 6, 1), date(2025, 8, 31), every=14)
    llm = res.strategies["llm"]
    assert llm["decisions"] == len(res.decisions) > 0
    assert llm["avg_exposure"] > 0 and llm["llm_cost_usd"] > 0
    assert res.max_position_pct == 20
    assert any("uncontaminated" in w for w in res.warnings)

    folder = res.save(tmp_path)
    json.loads((folder / "backtest.json").read_text())  # valid JSON (no NaN)
    report = (folder / "report.md").read_text()
    assert "LLM desk" in report and "Momentum 3m (capped)" in report


async def test_contaminated_period_is_flagged():
    provider = FakeProvider(make_bars(700, start=date(2024, 1, 1)))
    s = DeskSettings()
    s.llm.knowledge_cutoff = date(2026, 1, 1)
    res = await run_backtest(TradingDesk(s, FakeBackend(), provider), provider, "ETH-EUR", date(2025, 6, 1), date(2025, 7, 1), every=14)
    assert any("contaminated" in w and "uncontaminated" not in w for w in res.warnings)


async def test_invalid_strategy_requests():
    provider = FakeProvider()
    with pytest.raises(ValueError, match="Unknown"):
        await run_backtest(None, provider, "X", date(2025, 6, 1), date(2025, 7, 1), strategies=("magic",))
    with pytest.raises(ValueError, match="needs a desk"):
        await run_backtest(None, provider, "X", date(2025, 6, 1), date(2025, 7, 1), strategies=("llm",))
