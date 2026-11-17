"""What the agents see: a frozen, point-in-time snapshot of one ticker."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    title: str
    publisher: str | None = None
    published: datetime | None = None
    summary: str | None = None


class Snapshot(BaseModel):
    ticker: str
    as_of: date
    mode: Literal["live", "historical"]
    last_bar_date: date
    last_close: float
    currency: str | None = None
    technicals: dict[str, float | None]
    benchmark: dict[str, float | None] = Field(default_factory=dict)
    recent_bars: list[dict[str, Any]] = Field(default_factory=list)
    fundamentals: dict[str, Any] | None = None
    profile: dict[str, Any] | None = None
    news: list[NewsItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def block(self, *fields: str) -> str:
        """Render the requested fields as compact JSON for a prompt."""
        data = self.model_dump(mode="json", include=set(fields))
        return json.dumps(data, indent=1, default=str)
