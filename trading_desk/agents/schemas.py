"""Typed outputs of every desk role. Each one doubles as the --json-schema sent to the CLI.

Schemas are kept flat (no nested models, no $ref) so any JSON-schema consumer handles them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Stance = Literal["bullish", "bearish", "neutral"]


class AnalystReport(BaseModel):
    stance: Stance
    confidence: float = Field(ge=0, le=1, description="0 = no idea, 1 = certain")
    summary: str = Field(description="3-6 sentences, grounded in the provided data")
    key_points: list[str] = Field(description="Facts that drive the stance, each citing a number from the data")
    risks: list[str] = Field(description="What could make this stance wrong")
    data_gaps: list[str] = Field(description="Missing or stale data that limits this analysis")


class DebateArgument(BaseModel):
    thesis: str = Field(description="One-sentence thesis")
    arguments: list[str] = Field(description="Strongest arguments, most important first")
    rebuttals: list[str] = Field(description="Answers to the other side's points; empty in the opening round")
    what_would_change_my_mind: str


class RiskAssessment(BaseModel):
    risk_level: Literal["low", "medium", "high", "extreme"]
    max_position_pct: float = Field(ge=0, le=100, description="Largest prudent position, % of portfolio")
    stop_distance_pct: float = Field(ge=0.5, le=50, description="Suggested stop-loss distance below entry, in %")
    key_risks: list[str]
    veto: bool = Field(description="True only if opening or adding to a position must be blocked")
    veto_reason: str = Field(description="Empty string when veto is false")


class TradeDecision(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"] = Field(description="BUY = open/increase, SELL = reduce/close, HOLD = no change")
    conviction: float = Field(ge=0, le=1)
    target_position_pct: float = Field(ge=0, le=100, description="Desired position after the trade, % of portfolio")
    stop_loss: float | None = Field(description="Absolute price; null if no position is held")
    take_profit: float | None = Field(description="Absolute price, or null")
    horizon_days: int = Field(ge=1, le=365)
    rationale: str
    invalidation: str = Field(description="Observable condition that would prove this decision wrong")


def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    schema["additionalProperties"] = False
    return schema
