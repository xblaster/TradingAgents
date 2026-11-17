"""Data provider contract. Swap Yahoo for Polygon, IBKR, a CSV dump... by implementing this."""

from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd

from .models import Snapshot


class DataError(RuntimeError):
    pass


class DataProvider(Protocol):
    async def snapshot(self, ticker: str, as_of: date, lookback_days: int) -> Snapshot:
        """Everything known about `ticker` at the close of `as_of`, and nothing after."""
        ...

    async def prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Daily OHLCV bars between start and end (inclusive), indexed by date."""
        ...
