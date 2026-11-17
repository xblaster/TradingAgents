from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from trading_desk.paper import TAKER_FEE, Book, PaperTrader, Quote, execute, plan_orders

Q = {"BTC-EUR": Quote(99.0, 101.0), "ETH-EUR": Quote(9.9, 10.1)}


def test_plan_from_cash_buys_at_ask_with_fee():
    book = Book(1000.0)
    orders = plan_orders(book, {"BTC-EUR": 0.6, "ETH-EUR": 0.3}, Q)
    assert [(o.pair, o.side, round(o.eur)) for o in orders] == [("BTC-EUR", "buy", 600), ("ETH-EUR", "buy", 300)]
    execute(book, orders)
    assert book.cash == pytest.approx(100.0)
    assert book.holdings["BTC-EUR"] == pytest.approx(600 * (1 - TAKER_FEE) / 101)


def test_full_exit_sells_everything_at_bid():
    book = Book(0.0, {"BTC-EUR": 5.0})
    orders = plan_orders(book, {}, Q)
    assert len(orders) == 1 and orders[0].side == "sell" and orders[0].qty == 5.0
    execute(book, orders)
    assert book.holdings == {} and book.cash == pytest.approx(5 * 99 * (1 - TAKER_FEE))


def test_small_drift_is_ignored_and_tiny_orders_skipped():
    book = Book(500.0, {"BTC-EUR": 5.0})  # BTC weight 500/1000 = 50%
    assert plan_orders(book, {"BTC-EUR": 0.51}, Q) == []
    assert plan_orders(Book(100.0), {"ETH-EUR": 0.05}, Q) == []  # 5 EUR < minimum order


def test_buys_never_exceed_available_cash():
    book = Book(100.0, {"BTC-EUR": 9.0})  # NAV 1000 at mid; switch everything to ETH
    orders = plan_orders(book, {"ETH-EUR": 1.0}, Q)
    execute(book, orders)
    assert book.cash >= -1e-9


class StubProvider:
    def __init__(self, n=700):
        idx = pd.date_range(end="2026-11-17", periods=n, freq="D")
        up = 100 * 1.003 ** np.arange(n)
        down = 100 * 0.997 ** np.arange(n)
        self.frames = {
            "BTC-EUR": pd.DataFrame({"Open": up, "Close": up}, index=idx),
            "ETH-EUR": pd.DataFrame({"Open": down, "Close": down}, index=idx),
        }

    async def prices(self, ticker, start, end):
        return self.frames[ticker].loc[pd.Timestamp(start) : pd.Timestamp(end)]


async def quote(pair):
    return Q[pair]


def trader(tmp_path):
    return PaperTrader(tmp_path, StubProvider(), quote, universe=("BTC-EUR", "ETH-EUR"))


NOW = datetime(2026, 11, 17, 12, tzinfo=timezone.utc)


async def test_run_rebalances_books_and_journals(tmp_path):
    t = trader(tmp_path)
    entry = await t.run(NOW)
    v1 = entry["books"]["v1"]
    # uptrend in, downtrend out; ETH's risk budget stays in EUR (w = budget x signal)
    assert set(v1["targets"]) == {"BTC-EUR"} and 0 < v1["targets"]["BTC-EUR"] < 1
    assert v1["rebalanced"] and v1["orders"][0]["side"] == "buy"
    assert entry["books"]["ew_bh"]["targets"] == {"BTC-EUR": 0.5, "ETH-EUR": 0.5}
    assert t.journal_path.exists() and t.state_path.exists()


async def test_one_run_per_day_and_weekly_cadence(tmp_path):
    t = trader(tmp_path)
    await t.run(NOW)
    with pytest.raises(RuntimeError, match="Already ran"):
        await t.run(NOW + timedelta(hours=3))
    second = await t.run(NOW + timedelta(days=1))
    assert not second["books"]["v1"]["rebalanced"]
    week = await t.run(NOW + timedelta(days=7))
    assert week["books"]["v1"]["rebalanced"] and not week["books"]["ew_bh"]["rebalanced"]


async def test_status_is_pending_before_26_weeks(tmp_path):
    t = trader(tmp_path)
    for d in range(3):
        await t.run(NOW + timedelta(days=d))
    st = t.status()
    assert st["weeks"] < 1 and st["books"]["v1"]["verdict"].startswith("PENDING")
    assert set(st["books"]) == {"v1", "v2", "ew_bh", "btc_bh"}
