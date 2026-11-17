from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from trading_desk.data.indicators import technical_summary
from trading_desk.data.models import Snapshot
from trading_desk.llm import LLMError, LLMRequest, LLMResponse


def make_bars(n: int = 300, start: date = date(2025, 1, 1), drift: float = 0.0005, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1e6}, index=idx)


class FakeProvider:
    def __init__(self, bars: pd.DataFrame | None = None):
        self.bars = bars if bars is not None else make_bars()

    async def prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        return self.bars.loc[pd.Timestamp(start) : pd.Timestamp(end)]

    async def snapshot(self, ticker: str, as_of: date, lookback_days: int = 400) -> Snapshot:
        bars = self.bars.loc[pd.Timestamp(as_of - timedelta(days=lookback_days)) : pd.Timestamp(as_of)]
        return Snapshot(
            ticker=ticker, as_of=as_of, mode="historical", last_bar_date=bars.index[-1].date(),
            last_close=float(bars["Close"].iloc[-1]), technicals=technical_summary(bars),
        )


DEFAULTS: dict[str, dict[str, Any]] = {
    "analyst": {"stance": "bullish", "confidence": 0.7, "summary": "s", "key_points": ["k"], "risks": ["r"], "data_gaps": []},
    "bull": {"thesis": "up", "arguments": ["a"], "rebuttals": [], "what_would_change_my_mind": "x"},
    "bear": {"thesis": "down", "arguments": ["a"], "rebuttals": [], "what_would_change_my_mind": "x"},
    "risk_manager": {"risk_level": "medium", "max_position_pct": 15, "stop_distance_pct": 6, "key_risks": ["vol"], "veto": False, "veto_reason": ""},
    "portfolio_manager": {"action": "BUY", "conviction": 0.8, "target_position_pct": 10, "stop_loss": None, "take_profit": None, "horizon_days": 20, "rationale": "r", "invalidation": "i"},
}


class FakeBackend:
    """Answers by agent label. Override per label with a dict, an exception, or a callable."""

    def __init__(self, overrides: dict[str, Any] | None = None):
        self.overrides = overrides or {}
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        label = request.label
        key = next((k for k in self.overrides if label.startswith(k)), None)
        if key is not None:
            value = self.overrides[key]
            if isinstance(value, Exception):
                raise value
            data = value(request) if callable(value) else value
        elif label.endswith("_analyst"):
            data = DEFAULTS["analyst"]
        else:
            data = DEFAULTS[next(k for k in DEFAULTS if label.startswith(k))]
        return LLMResponse(text=json.dumps(data), data=data, model=request.model, cost_usd=0.01)


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()
