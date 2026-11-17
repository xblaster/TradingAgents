from datetime import date

import pytest

from trading_desk.agents.schemas import AnalystReport, DebateArgument, RiskAssessment, TradeDecision, json_schema
from trading_desk.backtest import run_backtest
from trading_desk.config import DeskSettings
from trading_desk.data.models import NewsItem
from trading_desk.llm import LLMError
from trading_desk.pipeline import TradingDesk

from .conftest import FakeBackend, FakeProvider, make_bars

AS_OF = date(2025, 11, 3)


def desk(backend, provider=None, **settings):
    return TradingDesk(DeskSettings(**settings), backend, provider or FakeProvider())


class NewsProvider(FakeProvider):
    async def snapshot(self, ticker, as_of, lookback_days=400):
        snap = await super().snapshot(ticker, as_of, lookback_days)
        snap.news = [NewsItem(title="Company beats estimates")]
        return snap


async def test_happy_path_runs_every_role_and_buys():
    backend = FakeBackend()
    run = await desk(backend, NewsProvider(), debate_rounds=2).run("TEST", AS_OF)
    labels = [c.label for c in backend.calls]
    assert sorted(labels[:3]) == ["fundamental_analyst", "news_analyst", "technical_analyst"]
    assert labels[3:] == ["bull_r1", "bear_r1", "bull_r2", "bear_r2", "risk_manager", "portfolio_manager"]
    assert run.decision.action == "BUY" and run.decision.target_position_pct == 10
    assert run.decision.stop_loss is not None and run.decision.stop_loss < run.snapshot.last_close
    assert run.cost_usd == pytest.approx(0.09)


async def test_news_analyst_skipped_without_news_or_web():
    backend = FakeBackend()
    run = await desk(backend).run("TEST", AS_OF)
    assert "news_analyst" not in [c.label for c in backend.calls]
    assert set(run.analysts) == {"technical", "fundamental"}


async def test_cached_steps_cost_nothing():
    run = await desk(FakeBackend()).run("TEST", AS_OF)
    paid = run.cost_usd
    run.steps[0].cached = True
    assert run.cost_usd == pytest.approx(paid - run.steps[0].cost_usd)


async def test_second_round_sees_the_opponent():
    backend = FakeBackend()
    await desk(backend, debate_rounds=2).run("TEST", AS_OF)
    bull_r2 = next(c for c in backend.calls if c.label == "bull_r2")
    assert '"thesis": "down"' in bull_r2.prompt


async def test_models_are_routed_by_tier():
    backend = FakeBackend()
    s = DeskSettings()
    s.llm.fast_model, s.llm.deep_model = "fast-x", "deep-y"
    await TradingDesk(s, backend, FakeProvider()).run("TEST", AS_OF)
    models = {c.label: c.model for c in backend.calls}
    assert models["technical_analyst"] == "fast-x" and models["bull_r1"] == "fast-x"
    assert models["risk_manager"] == "deep-y" and models["portfolio_manager"] == "deep-y"


async def test_historical_mode_never_gives_tools():
    backend = FakeBackend()
    await desk(backend).run("TEST", AS_OF)
    assert all(c.tools == () for c in backend.calls)


async def test_fails_closed_when_pm_errors():
    run = await desk(FakeBackend({"portfolio_manager": LLMError("boom")})).run("TEST", AS_OF, current_position_pct=7)
    assert run.decision.action == "HOLD" and run.decision.target_position_pct == 7
    assert any(not s.ok and s.agent == "portfolio_manager" for s in run.steps)


async def test_fails_closed_when_analysts_fail():
    backend = FakeBackend({"technical": LLMError("x"), "news": LLMError("y")})
    run = await desk(backend, NewsProvider()).run("TEST", AS_OF)
    assert run.decision.action == "HOLD"
    assert not any(c.label == "portfolio_manager" for c in backend.calls)


async def test_fails_closed_when_one_of_two_analysts_fails():
    backend = FakeBackend({"technical": LLMError("x")})
    run = await desk(backend).run("TEST", AS_OF)
    assert run.decision.action == "HOLD"
    assert not any(c.label == "portfolio_manager" for c in backend.calls)


async def test_no_risk_assessment_means_no_new_exposure():
    run = await desk(FakeBackend({"risk_manager": LLMError("x")})).run("TEST", AS_OF)
    assert run.decision.action == "HOLD" and run.decision.target_position_pct == 0


async def test_invalid_output_is_repaired_once():
    answers = iter([{"stance": "very bullish"}, {"stance": "neutral", "confidence": 0.4, "summary": "s", "key_points": [], "risks": [], "data_gaps": []}])
    backend = FakeBackend({"technical": lambda req: next(answers)})
    run = await desk(backend).run("TEST", AS_OF)
    assert run.analysts["technical"].stance == "neutral"
    assert sum(c.label == "technical_analyst" for c in backend.calls) == 2


@pytest.mark.parametrize("model", [AnalystReport, DebateArgument, RiskAssessment, TradeDecision])
def test_schemas_are_flat(model):
    schema = json_schema(model)
    assert "$ref" not in str(schema) and "$defs" not in schema
    assert schema["additionalProperties"] is False


async def test_run_saves_report(tmp_path):
    run = await desk(FakeBackend()).run("TEST", AS_OF)
    folder = run.save(tmp_path)
    assert (folder / "run.json").exists()
    assert "# TEST on 2025-11-03: **BUY**" in (folder / "report.md").read_text()


async def test_backtest_always_long_tracks_buy_and_hold():
    bars = make_bars(400, drift=0.001)
    provider = FakeProvider(bars)
    always_full = {"portfolio_manager": {**_pm(100)}, "risk_manager": _risk(100)}
    s = DeskSettings()
    s.risk.max_position_pct = 100
    s.risk.max_stop_distance_pct = 90
    d = TradingDesk(s, FakeBackend(always_full), provider)
    start, end = bars.index[300].date(), bars.index[-1].date()
    result = await run_backtest(d, provider, "TEST", start, end, every=10, cost_bps=0)
    m = result.metrics
    # Fully invested from the second day's open: same as buy & hold except the first overnight gap and day.
    assert m["avg_exposure"] > 0.98
    assert m["total_return"] == pytest.approx(m["buy_hold_return"], abs=0.05)
    assert m["trades"] == 1 and m["stop_hits"] == 0


async def test_backtest_flat_desk_has_zero_return():
    bars = make_bars(400)
    provider = FakeProvider(bars)
    d = TradingDesk(DeskSettings(), FakeBackend({"portfolio_manager": _pm(0, action="HOLD")}), provider)
    result = await run_backtest(d, provider, "TEST", bars.index[300].date(), bars.index[-1].date(), every=5)
    assert result.metrics["total_return"] == 0 and result.metrics["trades"] == 0
    assert result.warnings


def _pm(target, action="BUY"):
    return {"action": action, "conviction": 0.9, "target_position_pct": target, "stop_loss": 1.0, "take_profit": None,
            "horizon_days": 20, "rationale": "r", "invalidation": "i"}


def _risk(max_pct):
    return {"risk_level": "low", "max_position_pct": max_pct, "stop_distance_pct": 50, "key_risks": [], "veto": False, "veto_reason": ""}
