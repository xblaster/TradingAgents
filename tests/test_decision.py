from datetime import date

import pandas as pd
import pytest

from trading_desk.config import DeskSettings
from trading_desk.decision import HybridDesk, stats_verdict
from trading_desk.pipeline import TradingDesk
from trading_desk.screener import AssetStats

from .conftest import FakeBackend, FakeProvider, make_bars

AS_OF = date(2026, 6, 1)


def provider(drift: float, seed: int = 3) -> FakeProvider:
    return FakeProvider(make_bars(700, start=date(2024, 7, 1), drift=drift, seed=seed))


def hybrid(prov, backend=None, **kw):
    s = DeskSettings()
    llm = TradingDesk(s, backend, prov) if backend is not None else None
    return HybridDesk(s, prov, llm, universe=(), **kw)


async def test_clear_uptrend_is_decided_by_statistics_alone():
    backend = FakeBackend()
    h = hybrid(provider(0.004), backend, llm_on_gems=False)
    d = await h.decide("BTC-EUR", AS_OF)
    assert d.decided_by == "stats" and d.action == "BUY"
    assert d.target_position_pct == pytest.approx(20.0)  # cap x full trend signal
    assert d.stop_loss is not None and backend.calls == [] and d.llm_cost_usd == 0


async def test_clear_downtrend_exits_without_llm():
    backend = FakeBackend()
    d = await hybrid(provider(-0.004), backend).decide("BTC-EUR", AS_OF, current_position_pct=15)
    assert d.decided_by == "stats" and d.action == "SELL" and d.target_position_pct == 0
    assert backend.calls == []


async def test_gem_goes_to_llm_with_statistical_context():
    backend = FakeBackend()
    h = hybrid(provider(0.004), backend)  # strong trend + top rank -> G1
    d = await h.decide("BTC-EUR", AS_OF)
    assert d.decided_by == "llm_gem" and any(g["rule"] == "G1" for g in d.gems)
    assert h.llm_calls == 1 and d.llm_cost_usd > 0
    pm = next(c for c in backend.calls if c.label == "portfolio_manager")
    assert "why_you_are_consulted" in pm.prompt and "G1" in pm.prompt
    assert d.llm_run.snapshot.stats["statistical_verdict"]["clear"] is True


async def test_ambiguous_uses_llm_only_as_last_resort():
    prov = provider(0.0, seed=11)
    probe = await hybrid(prov, None).decide("BTC-EUR", AS_OF)
    assert probe.stats.clear is False, "fixture should be statistically ambiguous"
    assert probe.decided_by == "stats_hold" and probe.action == "HOLD"  # no LLM configured

    backend = FakeBackend()
    d = await hybrid(prov, backend).decide("BTC-EUR", AS_OF, current_position_pct=5)
    assert d.decided_by == "llm_last_resort" and backend.calls

    quiet = FakeBackend()
    d2 = await hybrid(prov, quiet, llm_last_resort=False).decide("BTC-EUR", AS_OF, current_position_pct=5)
    assert d2.decided_by == "stats_hold" and d2.target_position_pct == 5 and quiet.calls == []


def test_extreme_volatility_halves_the_position():
    pack = {"trend_ensemble_signal": 1.0, "volatility_regime": {"label": "extreme", "current_vol": 1.2}, "last_close": 100.0}
    v = stats_verdict(AssetStats("X", pack, 0.9), 0.0, DeskSettings())
    assert v.clear and v.target_position_pct == pytest.approx(10.0)
    assert 85.0 <= v.stop_loss < 100.0  # within the 15% max stop distance
