"""Deterministic checks applied after the portfolio manager. The LLM proposes, code disposes."""

from __future__ import annotations

from pydantic import BaseModel

from .agents.schemas import RiskAssessment, TradeDecision
from .config import RiskLimits


class FinalDecision(BaseModel):
    action: str
    target_position_pct: float
    current_position_pct: float
    stop_loss: float | None
    take_profit: float | None
    conviction: float
    horizon_days: int
    adjustments: list[str]
    proposed: TradeDecision | None


def hold(current_position_pct: float, reason: str, proposed: TradeDecision | None = None) -> FinalDecision:
    return FinalDecision(
        action="HOLD", target_position_pct=current_position_pct, current_position_pct=current_position_pct,
        stop_loss=proposed.stop_loss if proposed else None, take_profit=proposed.take_profit if proposed else None,
        conviction=proposed.conviction if proposed else 0.0, horizon_days=proposed.horizon_days if proposed else 1,
        adjustments=[reason], proposed=proposed,
    )


def apply_guardrails(
    proposed: TradeDecision,
    *,
    last_close: float,
    current_position_pct: float,
    limits: RiskLimits,
    risk: RiskAssessment | None,
) -> FinalDecision:
    notes: list[str] = []
    action = proposed.action
    target = proposed.target_position_pct

    if action == "HOLD":
        target = current_position_pct
    elif proposed.conviction < limits.min_conviction:
        return hold(current_position_pct, f"Conviction {proposed.conviction:.2f} below minimum {limits.min_conviction:.2f}: HOLD.", proposed)

    # Size caps: the configured hard limit, then the risk manager's.
    cap = limits.max_position_pct
    if risk is not None:
        cap = min(cap, risk.max_position_pct)
    if target > cap and target > current_position_pct:
        notes.append(f"Target {target:.1f}% capped to {cap:.1f}%.")
        target = max(cap, min(target, current_position_pct))  # never force a sale just to meet a new cap

    if risk is not None and risk.veto and target > current_position_pct:
        return hold(current_position_pct, f"Risk manager veto: {risk.veto_reason or 'no reason given'}", proposed)

    # Make the action consistent with the size change.
    if target > current_position_pct + 1e-9:
        action = "BUY"
    elif target < current_position_pct - 1e-9:
        action = "SELL"
    else:
        action = "HOLD"
    if action != proposed.action:
        notes.append(f"Action changed from {proposed.action} to {action} to match the target position.")

    # Stops and targets must make sense for a long position.
    stop, take = proposed.stop_loss, proposed.take_profit
    if target > 0:
        max_dist = limits.max_stop_distance_pct / 100
        floor = last_close * (1 - max_dist)
        if stop is None or stop >= last_close:
            dist = (risk.stop_distance_pct if risk else limits.default_stop_distance_pct) / 100
            stop = round(last_close * (1 - min(dist, max_dist)), 4)
            notes.append(f"Stop-loss missing or above price: set to {stop}.")
        elif stop < floor:
            notes.append(f"Stop-loss {stop} more than {limits.max_stop_distance_pct}% away: tightened to {floor:.4f}.")
            stop = round(floor, 4)
        if take is not None and take <= last_close:
            notes.append(f"Take-profit {take} not above price: dropped.")
            take = None
    else:
        stop, take = None, None

    return FinalDecision(
        action=action, target_position_pct=round(target, 4), current_position_pct=current_position_pct,
        stop_loss=stop, take_profit=take, conviction=proposed.conviction, horizon_days=proposed.horizon_days,
        adjustments=notes, proposed=proposed,
    )
