"""What the agents see: a frozen, point-in-time snapshot of one ticker."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    title: str
    publisher: str | None = None
    published: datetime | None = None
    summary: str | None = None


AssetClass = Literal["equity", "crypto"]

_CRYPTO = re.compile(r"^[A-Z0-9]{2,10}-(USD|EUR|GBP|USDT|USDC|CHF)$")


def asset_class_of(ticker: str) -> AssetClass:
    """BTC-EUR, ETH-USD... are crypto pairs (Revolut X / Yahoo format); anything else is an equity."""
    return "crypto" if _CRYPTO.match(ticker.upper()) else "equity"


def periods_per_year(asset_class: AssetClass) -> int:
    return 365 if asset_class == "crypto" else 252


class Snapshot(BaseModel):
    ticker: str
    as_of: date
    mode: Literal["live", "historical"]
    asset_class: AssetClass = "equity"
    market: dict[str, Any] | None = None  # crypto: market structure + live order book
    last_bar_date: date
    last_close: float
    currency: str | None = None
    technicals: dict[str, float | None]
    benchmark: dict[str, float | str | None] = Field(default_factory=dict)
    recent_bars: list[dict[str, Any]] = Field(default_factory=list)
    fundamentals: dict[str, Any] | None = None
    profile: dict[str, Any] | None = None
    news: list[NewsItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def block(self, *fields: str) -> str:
        """Render the requested fields as compact JSON for a prompt."""
        data = self.model_dump(mode="json", include=set(fields))
        return json.dumps(data, indent=1, default=str)
