"""Yahoo Finance provider (via yfinance), with point-in-time discipline, for equities and crypto.

Rules that keep backtests honest:
  * price bars are cut at `as_of` (inclusive);
  * quarterly statements are only visible once `period_end + REPORTING_LAG` has passed;
  * the company profile (live ratios, analyst targets), Yahoo headlines and the Revolut X order
    book only describe *today*, so they are used in live mode only.

Crypto pairs use the Revolut X / Yahoo format (BTC-EUR, ETH-USD): 365 bars a year, BTC as the
benchmark, and market-structure metrics instead of financial statements.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from .indicators import horizons, market_structure, technical_summary, trailing_return
from .models import NewsItem, Snapshot, asset_class_of, periods_per_year
from .provider import DataError

log = logging.getLogger(__name__)

REPORTING_LAG = timedelta(days=45)
LIVE_WINDOW = timedelta(days=4)  # as_of within 4 days of today counts as live (weekends, holidays)
EQUITY_BENCHMARK = "SPY"

_STATEMENT_ROWS = {
    "income": ["Total Revenue", "Gross Profit", "Operating Income", "Net Income", "Diluted EPS"],
    "balance": ["Cash Cash Equivalents And Short Term Investments", "Total Debt", "Stockholders Equity"],
    "cashflow": ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"],
}
_PROFILE_KEYS = [
    "longName", "sector", "industry", "marketCap", "trailingPE", "forwardPE",
    "priceToSalesTrailing12Months", "enterpriseToEbitda", "profitMargins", "returnOnEquity",
    "debtToEquity", "beta", "dividendYield", "recommendationMean", "numberOfAnalystOpinions",
    "targetMeanPrice", "shortPercentOfFloat", "earningsTimestamp",
    # crypto
    "circulatingSupply", "maxSupply", "volume24Hr", "volumeAllCurrencies", "startDate",
]


def benchmark_for(ticker: str) -> str | None:
    if asset_class_of(ticker) == "equity":
        return EQUITY_BENCHMARK
    base, quote = ticker.upper().split("-", 1)
    return None if base == "BTC" else f"BTC-{quote}"


class YahooProvider:
    def __init__(self, today: date | None = None, max_news: int = 15, order_book: bool = True):
        self._today = today
        self.max_news = max_news
        self.order_book = order_book
        self._bars: dict[str, pd.DataFrame] = {}
        self._lock = threading.Lock()

    @property
    def today(self) -> date:
        return self._today or date.today()

    def is_live(self, as_of: date) -> bool:
        return self.today - LIVE_WINDOW <= as_of <= self.today

    # ------------------------------------------------------------------ public

    async def prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        return await asyncio.to_thread(self._prices, ticker.upper(), start, end)

    async def snapshot(self, ticker: str, as_of: date, lookback_days: int = 400) -> Snapshot:
        return await asyncio.to_thread(self._snapshot, ticker.upper(), as_of, lookback_days)

    # ----------------------------------------------------------------- private

    def _prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        bars = self._history(ticker, start)
        return bars.loc[pd.Timestamp(start) : pd.Timestamp(end)]

    def _history(self, ticker: str, start: date) -> pd.DataFrame:
        """Daily bars from at least `start` up to today, downloaded once per ticker."""
        import yfinance as yf

        with self._lock:
            cached = self._bars.get(ticker)
            if cached is not None and not cached.empty and cached.index[0] <= pd.Timestamp(start) + timedelta(days=7):
                return cached
            raw = yf.Ticker(ticker).history(start=start - timedelta(days=7), end=self.today + timedelta(days=1), auto_adjust=True)
            if raw.empty:
                raise DataError(f"No price data for {ticker} from Yahoo")
            raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
            bars = raw[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            bars = bars[~bars.index.duplicated(keep="last")]
            self._bars[ticker] = bars
            return bars

    def _snapshot(self, ticker: str, as_of: date, lookback_days: int) -> Snapshot:
        import yfinance as yf

        if as_of > self.today:
            raise DataError(f"as_of {as_of} is in the future")
        live = self.is_live(as_of)
        asset_class = asset_class_of(ticker)
        ppy = periods_per_year(asset_class)
        notes: list[str] = []

        bars = self._prices(ticker, as_of - timedelta(days=lookback_days), as_of)
        if len(bars) < 60:
            raise DataError(f"Only {len(bars)} bars for {ticker} before {as_of}; need at least 60")
        if len(bars) < 210:
            notes.append(f"Short price history ({len(bars)} bars): 200-day indicators may be missing.")

        bench_ticker = benchmark_for(ticker)
        bench_close = None
        if bench_ticker:
            try:
                bench_close = self._prices(bench_ticker, as_of - timedelta(days=lookback_days), as_of)["Close"]
            except DataError:
                notes.append(f"Benchmark {bench_ticker} unavailable.")

        recent = bars.tail(10).reset_index(names="date")
        recent["date"] = recent["date"].dt.date.astype(str)
        snapshot = Snapshot(
            ticker=ticker,
            as_of=as_of,
            mode="live" if live else "historical",
            asset_class=asset_class,
            last_bar_date=bars.index[-1].date(),
            last_close=round(float(bars["Close"].iloc[-1]), 4),
            currency=ticker.split("-", 1)[1] if asset_class == "crypto" else None,
            technicals=technical_summary(bars, ppy),
            benchmark=self._benchmark(bench_ticker, bench_close, bars["Close"], ppy),
            recent_bars=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()} for row in recent.to_dict("records")],
        )

        t = yf.Ticker(ticker)
        if asset_class == "crypto":
            snapshot.market = market_structure(bars, bench_close, ppy)
            if live and self.order_book:
                snapshot.market["revolut_x_order_book"] = self._order_book(ticker, notes)
        else:
            try:
                snapshot.fundamentals = self._fundamentals(t, as_of)
            except Exception as exc:  # yfinance raises a zoo of exceptions on missing data
                notes.append(f"Fundamentals unavailable: {exc}")
            if snapshot.fundamentals is None or not snapshot.fundamentals.get("quarters"):
                notes.append("No quarterly statements were public as of this date in the Yahoo dataset.")

        if live:
            try:
                info = t.info or {}
                snapshot.currency = info.get("currency") or snapshot.currency
                profile = {k: info.get(k) for k in _PROFILE_KEYS if info.get(k) is not None}
                if isinstance(profile.get("earningsTimestamp"), (int, float)):
                    profile["next_or_last_earnings_date"] = datetime.fromtimestamp(profile.pop("earningsTimestamp")).date().isoformat()
                snapshot.profile = profile or None
            except Exception as exc:
                notes.append(f"Profile unavailable: {exc}")
            try:
                snapshot.news = self._news(t, as_of)
            except Exception as exc:
                notes.append(f"News unavailable: {exc}")
        else:
            withheld = "order book and headlines" if asset_class == "crypto" else "live ratios, analyst targets and headlines"
            notes.append(f"Historical mode: {withheld} are withheld to avoid look-ahead.")
        snapshot.notes = notes
        return snapshot

    @staticmethod
    def _order_book(ticker: str, notes: list[str]) -> dict[str, Any] | None:
        from ..revolutx import RevolutXClient, RevolutXError, summarize_order_book

        client = RevolutXClient()
        try:
            book = client._send(client.build_request("GET", f"/public/order-book/{ticker}", auth=False))["data"]
            return summarize_order_book(book)
        except (RevolutXError, KeyError, TypeError) as exc:
            notes.append(f"Revolut X order book unavailable: {exc}")
            return None

    @staticmethod
    def _benchmark(name: str | None, bench: pd.Series | None, close: pd.Series, ppy: int) -> dict[str, Any]:
        if not name or bench is None:
            return {}
        out: dict[str, Any] = {"benchmark": name}
        for label, days in horizons(ppy).items():
            if label == "1w":
                continue
            r_b, r_s = trailing_return(bench, days), trailing_return(close, days)
            out[f"benchmark_return_{label}"] = None if r_b is None else round(r_b, 4)
            out[f"excess_return_{label}"] = None if r_b is None or r_s is None else round(r_s - r_b, 4)
        return out

    @staticmethod
    def _fundamentals(t: Any, as_of: date) -> dict[str, Any]:
        cutoff = pd.Timestamp(as_of - REPORTING_LAG)
        frames = {"income": t.quarterly_income_stmt, "balance": t.quarterly_balance_sheet, "cashflow": t.quarterly_cashflow}
        quarters: dict[str, dict[str, float]] = {}
        for kind, frame in frames.items():
            if frame is None or frame.empty:
                continue
            for period in frame.columns:
                if pd.Timestamp(period) > cutoff:
                    continue  # not yet reported as of this date
                q = quarters.setdefault(pd.Timestamp(period).date().isoformat(), {})
                for row in _STATEMENT_ROWS[kind]:
                    if row in frame.index and pd.notna(frame.at[row, period]):
                        q[row] = float(frame.at[row, period])
        ordered = dict(sorted(quarters.items(), reverse=True)[:5])

        derived: dict[str, float] = {}
        periods = list(ordered.values())
        revenue = periods[0].get("Total Revenue") if periods else None
        if revenue:
            latest = periods[0]
            for row, name in (("Gross Profit", "gross_margin"), ("Operating Income", "operating_margin"), ("Net Income", "net_margin"), ("Free Cash Flow", "fcf_margin")):
                if row in latest:
                    derived[name] = round(latest[row] / revenue, 4)
            if len(periods) >= 2 and periods[1].get("Total Revenue"):
                derived["revenue_growth_qoq"] = round(revenue / periods[1]["Total Revenue"] - 1, 4)
            if len(periods) >= 5 and periods[4].get("Total Revenue"):
                derived["revenue_growth_yoy"] = round(revenue / periods[4]["Total Revenue"] - 1, 4)
        return {"reporting_lag_days": REPORTING_LAG.days, "quarters": ordered, "derived": derived}

    def _news(self, t: Any, as_of: date) -> list[NewsItem]:
        end_of_day = datetime.combine(as_of, datetime.max.time())
        items: list[NewsItem] = []
        for raw in t.news or []:
            content = raw.get("content", raw)
            published = content.get("pubDate") or content.get("providerPublishTime")
            if isinstance(published, (int, float)):
                published = datetime.fromtimestamp(published)
            elif isinstance(published, str):
                published = datetime.fromisoformat(published.replace("Z", "+00:00")).replace(tzinfo=None)
            if published and published > end_of_day:
                continue
            provider = content.get("provider") or {}
            items.append(
                NewsItem(
                    title=content.get("title") or "",
                    publisher=provider.get("displayName") if isinstance(provider, dict) else content.get("publisher"),
                    published=published,
                    summary=(content.get("summary") or content.get("description") or None),
                )
            )
        return [i for i in items if i.title][: self.max_news]
