"""Forward paper trading on live Revolut X prices (STRATEGY.md, section 7).

Each run:
  1. computes target weights from *completed* daily bars (today's partial bar is dropped);
  2. fetches the live Revolut X order book of every pair;
  3. for every book due for a rebalance, buys at the best ask and sells at the best bid, paying
     the 9 bps taker fee, with the same 2% minimum trade as the backtest;
  4. marks every book to market at mid and appends one line to the journal.

The journal is append-only JSON lines, so the forward record can be audited (and committed).
One run per UTC day: a second run the same day is refused unless forced.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import pandas as pd

from .data import DataProvider
from .portfolio import EXTENDED, TrendConfig, equal_weight_targets, load_prices, target_weights, warmup_start

TAKER_FEE = 0.0009
MIN_ORDER_EUR = 10.0
STRATEGIES: dict[str, TrendConfig] = {"v1": TrendConfig(), "v2": TrendConfig().scaled(0.5)}
BENCHMARKS: dict[str, int] = {"ew_bh": 30, "btc_bh": 10**6}  # rebalance period in days
FORWARD_MIN_WEEKS = 26


@dataclass
class Quote:
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


QuoteFn = Callable[[str], Awaitable[Quote]]


@dataclass
class Order:
    pair: str
    side: str  # "buy" | "sell"
    eur: float  # quote amount (buy) or expected proceeds before fees (sell)
    qty: float  # base quantity
    price: float


@dataclass
class Book:
    cash: float
    holdings: dict[str, float] = field(default_factory=dict)
    last_rebalance: str | None = None

    def nav(self, quotes: dict[str, Quote]) -> float:
        return self.cash + sum(q * quotes[p].mid for p, q in self.holdings.items() if q > 0)

    def weights(self, quotes: dict[str, Quote]) -> dict[str, float]:
        nav = self.nav(quotes)
        return {p: q * quotes[p].mid / nav for p, q in self.holdings.items() if q > 0} if nav > 0 else {}


def plan_orders(book: Book, targets: dict[str, float], quotes: dict[str, Quote], *, min_trade: float = 0.02, min_order_eur: float = MIN_ORDER_EUR) -> list[Order]:
    """Orders that move `book` to `targets` (weights of NAV). Sells first, buys scaled to the cash they free."""
    nav = book.nav(quotes)
    current = book.weights(quotes)
    sells, buys = [], []
    for pair in sorted(set(targets) | set(current)):
        target, have = targets.get(pair, 0.0), current.get(pair, 0.0)
        diff = target - have
        full_exit = target == 0 and have > 0
        if abs(diff) < min_trade and not full_exit:
            continue
        q = quotes[pair]
        if diff < 0:
            qty = book.holdings[pair] if full_exit else min(book.holdings[pair], -diff * nav / q.mid)
            if qty * q.bid >= min_order_eur:
                sells.append(Order(pair, "sell", qty * q.bid, qty, q.bid))
        else:
            eur = diff * nav
            if eur >= min_order_eur:
                buys.append(Order(pair, "buy", eur, eur * (1 - TAKER_FEE) / q.ask, q.ask))
    cash_after_sells = book.cash + sum(o.eur * (1 - TAKER_FEE) for o in sells)
    wanted = sum(o.eur for o in buys)
    if wanted > cash_after_sells > 0:  # rounding/drift: never spend cash we do not have
        k = cash_after_sells / wanted
        buys = [Order(o.pair, o.side, o.eur * k, o.qty * k, o.price) for o in buys]
    return sells + buys


def execute(book: Book, orders: list[Order]) -> None:
    for o in orders:
        if o.side == "sell":
            book.holdings[o.pair] = max(0.0, book.holdings.get(o.pair, 0.0) - o.qty)
            book.cash += o.qty * o.price * (1 - TAKER_FEE)
        else:
            book.cash -= o.eur
            book.holdings[o.pair] = book.holdings.get(o.pair, 0.0) + o.eur * (1 - TAKER_FEE) / o.price
    book.holdings = {p: q for p, q in book.holdings.items() if q > 1e-12}


class PaperTrader:
    def __init__(self, folder: str | Path, provider: DataProvider, quote_fn: QuoteFn, *, universe: tuple[str, ...] = EXTENDED, initial_cash: float = 10_000.0):
        self.folder = Path(folder)
        self.provider = provider
        self.quote_fn = quote_fn
        self.universe = universe
        self.initial_cash = initial_cash

    @property
    def state_path(self) -> Path:
        return self.folder / "state.json"

    @property
    def journal_path(self) -> Path:
        return self.folder / "journal.jsonl"

    def load(self) -> dict[str, Book]:
        if not self.state_path.exists():
            return {name: Book(self.initial_cash) for name in (*STRATEGIES, *BENCHMARKS)}
        raw = json.loads(self.state_path.read_text())
        return {name: Book(**b) for name, b in raw["books"].items()}

    def journal(self) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        return [json.loads(line) for line in self.journal_path.read_text().splitlines() if line.strip()]

    async def targets(self, today: date) -> dict[str, dict[str, float]]:
        cfg_max = max(STRATEGIES.values(), key=lambda c: max(c.lookbacks))
        _, closes = await load_prices(self.provider, self.universe, warmup_start(today, cfg_max), today)
        closes = closes[closes.index < pd.Timestamp(today)]  # completed bars only
        out = {name: _last_row(target_weights(closes, cfg)) for name, cfg in STRATEGIES.items()}
        out["ew_bh"] = _last_row(equal_weight_targets(closes, TrendConfig()))
        out["btc_bh"] = {"BTC-EUR": 1.0}
        return out

    async def run(self, now: datetime | None = None, *, force: bool = False) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        today = now.date()
        journal = self.journal()
        if journal and journal[-1]["date"] == today.isoformat() and not force:
            raise RuntimeError(f"Already ran today ({today}); use --force to run again")

        books = self.load()
        targets = await self.targets(today)
        quotes = {p: await self.quote_fn(p) for p in self.universe}
        entry: dict[str, Any] = {"ts": now.isoformat(), "date": today.isoformat(), "quotes": {p: asdict(q) for p, q in quotes.items()}, "books": {}}
        for name, book in books.items():
            every = STRATEGIES[name].every if name in STRATEGIES else BENCHMARKS[name]
            due = book.last_rebalance is None or (today - date.fromisoformat(book.last_rebalance)).days >= every
            orders: list[Order] = []
            if due:
                orders = plan_orders(book, targets[name], quotes)
                execute(book, orders)
                book.last_rebalance = today.isoformat()
            entry["books"][name] = {
                "nav": round(book.nav(quotes), 2), "cash": round(book.cash, 2), "rebalanced": due,
                "targets": {k: round(v, 4) for k, v in targets[name].items()},
                "orders": [asdict(o) for o in orders],
            }

        self.folder.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        self.state_path.write_text(json.dumps({"updated": now.isoformat(), "books": {n: asdict(b) for n, b in books.items()}}, indent=1))
        return entry

    def status(self) -> dict[str, Any]:
        journal = self.journal()
        if not journal:
            return {"days": 0, "books": {}}
        navs = pd.DataFrame({e["date"]: {n: b["nav"] for n, b in e["books"].items()} for e in journal}).T
        navs.index = pd.to_datetime(navs.index)
        weeks = (navs.index[-1] - navs.index[0]).days / 7
        out: dict[str, Any] = {"start": journal[0]["date"], "last": journal[-1]["date"], "weeks": round(weeks, 1), "books": {}}
        for name in navs.columns:
            out["books"][name] = _stats(navs[name])
        ew = out["books"].get("ew_bh")
        for name in STRATEGIES:
            s = out["books"].get(name)
            if not s or not ew:
                continue
            s["P1"] = s["return"] >= 0 or s["return"] >= ew["return"]
            s["P2"] = s["sharpe"] is not None and ew["sharpe"] is not None and s["sharpe"] >= ew["sharpe"]
            s["P3"] = abs(s["max_drawdown"]) <= 0.6 * abs(ew["max_drawdown"]) if ew["max_drawdown"] < 0 else s["max_drawdown"] == 0
            s["verdict"] = ("PASS" if s["P1"] and s["P2"] and s["P3"] else "FAIL") if weeks >= FORWARD_MIN_WEEKS else f"PENDING ({weeks:.0f}/{FORWARD_MIN_WEEKS} weeks)"
        return out


def _last_row(frame: pd.DataFrame) -> dict[str, float]:
    row = frame.iloc[-1]
    return {k: float(v) for k, v in row.items() if v > 0}


def _stats(nav: pd.Series) -> dict[str, Any]:
    rets = nav.pct_change().dropna()
    std = float(rets.std()) if len(rets) > 1 else 0.0
    return {
        "nav": float(nav.iloc[-1]),
        "return": float(nav.iloc[-1] / nav.iloc[0] - 1),
        "max_drawdown": float((nav / nav.cummax() - 1).min()),
        "sharpe": float(rets.mean() / std * math.sqrt(365)) if std > 0 else None,
    }


_PUBLIC_INTERVAL_S = 0.6  # Revolut X public endpoints: 20 requests per 10 s
_last_public_call = 0.0
_public_lock: asyncio.Lock | None = None


async def revolutx_quote(pair: str, retries: int = 3) -> Quote:
    """Best bid/ask from the public Revolut X order book, throttled and retried on HTTP 429."""
    global _last_public_call, _public_lock
    from .revolutx import RevolutXClient, RevolutXError, summarize_order_book

    _public_lock = _public_lock or asyncio.Lock()
    for attempt in range(retries + 1):
        async with _public_lock:
            wait = _last_public_call + _PUBLIC_INTERVAL_S - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            _last_public_call = time.monotonic()
        try:
            book = summarize_order_book(await RevolutXClient().public_order_book(pair))
            break
        except RevolutXError as exc:
            if exc.status != 429 or attempt == retries:
                raise
            await asyncio.sleep(2 * 2**attempt)
    if book["best_bid"] is None or book["best_ask"] is None:
        raise RuntimeError(f"Empty Revolut X order book for {pair}")
    return Quote(book["best_bid"], book["best_ask"])
