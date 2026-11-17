import pytest

from trading_desk.agents.schemas import RiskAssessment, TradeDecision
from trading_desk.config import RiskLimits
from trading_desk.guardrails import apply_guardrails

LIMITS = RiskLimits(max_position_pct=20, min_conviction=0.55, max_stop_distance_pct=15, default_stop_distance_pct=8)


def decision(**kw) -> TradeDecision:
    base = dict(action="BUY", conviction=0.8, target_position_pct=10, stop_loss=90.0, take_profit=120.0, horizon_days=20, rationale="r", invalidation="i")
    return TradeDecision(**{**base, **kw})


def risk(**kw) -> RiskAssessment:
    base = dict(risk_level="medium", max_position_pct=15, stop_distance_pct=6, key_risks=[], veto=False, veto_reason="")
    return RiskAssessment(**{**base, **kw})


def run(d, current=0.0, r=None):
    return apply_guardrails(d, last_close=100.0, current_position_pct=current, limits=LIMITS, risk=r)


def test_clean_buy_passes_through():
    out = run(decision(), r=risk())
    assert (out.action, out.target_position_pct, out.stop_loss, out.adjustments) == ("BUY", 10, 90.0, [])


def test_low_conviction_becomes_hold():
    out = run(decision(conviction=0.3), current=5)
    assert out.action == "HOLD" and out.target_position_pct == 5


def test_size_capped_by_risk_manager_then_hard_limit():
    assert run(decision(target_position_pct=50), r=risk(max_position_pct=12)).target_position_pct == 12
    assert run(decision(target_position_pct=50), r=risk(max_position_pct=80)).target_position_pct == 20


def test_cap_never_forces_a_sale():
    out = run(decision(target_position_pct=40), current=30, r=risk(max_position_pct=10))
    assert out.action == "HOLD" and out.target_position_pct == 30


def test_veto_blocks_adding_but_not_reducing():
    assert run(decision(), r=risk(veto=True, veto_reason="earnings tomorrow")).action == "HOLD"
    out = run(decision(action="SELL", target_position_pct=0), current=10, r=risk(veto=True, veto_reason="x"))
    assert out.action == "SELL" and out.stop_loss is None


def test_action_is_made_consistent_with_target():
    out = run(decision(action="BUY", target_position_pct=5), current=10, r=risk())
    assert out.action == "SELL"


@pytest.mark.parametrize("stop,expected", [(None, 94.0), (105.0, 94.0), (50.0, 85.0)])
def test_stop_loss_sanitized(stop, expected):
    out = run(decision(stop_loss=stop), r=risk(stop_distance_pct=6))
    assert out.stop_loss == pytest.approx(expected)


def test_take_profit_below_price_dropped():
    assert run(decision(take_profit=95.0), r=risk()).take_profit is None
