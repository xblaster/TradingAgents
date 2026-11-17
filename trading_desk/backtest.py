"""Walk-forward backtest of the desk on one ticker.

Execution model (deliberately conservative):
  * the desk decides on the close of day t using data up to t only;
  * the trade fills at the open of t+1, paying `cost_bps` on turnover;
  * a long position whose low crosses its stop is exited at the stop (or at the open on a gap).

Warning: an LLM whose training data covers the backtest period may "remember" what happened.
Point-in-time data does not fix that. Only periods after the model's knowledge cutoff give an
uncontaminated estimate; everything else is a smoke test.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .data import DataProvider
from .data.indicators import TRADING_DAYS
from .pipeline import DeskRun, TradingDesk

log = logging.getLogger(__name__)

LOOKAHEAD_WARNING = (
    "Look-ahead risk: if the period predates the models' knowledge cutoff, the LLM may know how "
    "it ended. Treat results on such periods as a pipeline check, not as evidence of an edge."
)


class BacktestResult(BaseModel):
    ticker: str
    start: date
    end: date
    decisions: list[dict[str, Any]]
    equity: list[dict[str, Any]]
    metrics: dict[str, Any]
    warnings: list[str]

    def save(self, results_dir: Path) -> Path:
        from .report import render_backtest_markdown

        folder = results_dir / self.ticker / f"backtest_{self.start}_{self.end}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "backtest.json").write_text(self.model_dump_json(indent=2))
        (folder / "report.md").write_text(render_backtest_markdown(self))
        pd.DataFrame(self.equity).to_csv(folder / "equity.csv", index=False)
        return folder


def _drift(w: float, r: float) -> float:
    """New weight of a position at weight w after it returns r (rest is cash at 0%)."""
    denom = 1 + w * r
    return w * (1 + r) / denom if denom > 0 else 0.0


def _stats(equity: pd.Series) -> tuple[float, float, float]:
    rets = equity.pct_change().dropna()
    vol = float(rets.std() * math.sqrt(TRADING_DAYS)) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / rets.std() * math.sqrt(TRADING_DAYS)) if len(rets) > 1 and rets.std() > 0 else 0.0
    mdd = float((equity / equity.cummax() - 1).min())
    return vol, sharpe, mdd


async def run_backtest(
    desk: TradingDesk,
    provider: DataProvider,
    ticker: str,
    start: date,
    end: date,
    *,
    every: int = 5,
    cost_bps: float = 5.0,
    on_decision: Callable[[DeskRun], None] | None = None,
) -> BacktestResult:
    ticker = ticker.upper()
    bars = await provider.prices(ticker, start, end)
    if len(bars) < 2:
        raise ValueError(f"Not enough bars for {ticker} between {start} and {end}")
    dates = [ts.date() for ts in bars.index]
    o, h, l, c = (bars[col].to_numpy(dtype=float) for col in ("Open", "High", "Low", "Close"))
    cost = cost_bps / 10_000

    equity, w, stop = 1.0, 0.0, None
    pending: tuple[float, float | None] | None = None
    curve: list[dict[str, Any]] = [{"date": dates[0].isoformat(), "strategy": 1.0, "buy_hold": 1.0, "weight": 0.0}]
    decisions: list[dict[str, Any]] = []
    trades = stop_hits = 0
    llm_cost = 0.0
    exposure: list[float] = []

    for i in range(len(dates)):
        if i > 0:
            # Overnight gap on the position held at yesterday's close.
            r_gap = o[i] / c[i - 1] - 1
            equity *= 1 + w * r_gap
            w = _drift(w, r_gap)
            # Fill yesterday's decision at today's open.
            if pending is not None:
                new_w, new_stop = pending
                if abs(new_w - w) > 1e-9:
                    equity *= 1 - abs(new_w - w) * cost
                    trades += 1
                w, stop, pending = new_w, new_stop, None
            # Intraday: stop-loss, else open-to-close.
            if w > 0 and stop is not None and l[i] <= stop:
                exit_px = min(o[i], stop)
                equity *= (1 + w * (exit_px / o[i] - 1)) * (1 - w * cost)
                w, stop = 0.0, None
                stop_hits += 1
                trades += 1
            else:
                r_day = c[i] / o[i] - 1
                equity *= 1 + w * r_day
                w = _drift(w, r_day)
            exposure.append(w)
            curve.append({"date": dates[i].isoformat(), "strategy": round(equity, 6), "buy_hold": round(c[i] / c[0], 6), "weight": round(w, 4)})

        if i % every == 0 and i < len(dates) - 1:
            run = await desk.run(ticker, dates[i], current_position_pct=round(w * 100, 2))
            llm_cost += run.cost_usd
            d = run.decision
            pending = (d.target_position_pct / 100, d.stop_loss)
            decisions.append({
                "date": dates[i].isoformat(), "action": d.action, "target_position_pct": d.target_position_pct,
                "conviction": d.conviction, "stop_loss": d.stop_loss, "adjustments": d.adjustments, "cost_usd": run.cost_usd,
            })
            log.info("%s %s -> %s %.1f%% (equity %.4f)", ticker, dates[i], d.action, d.target_position_pct, equity)
            if on_decision:
                on_decision(run)

    eq = pd.Series([p["strategy"] for p in curve])
    bh = pd.Series([p["buy_hold"] for p in curve])
    vol, sharpe, mdd = _stats(eq)
    bh_vol, bh_sharpe, bh_mdd = _stats(bh)
    metrics = {
        "total_return": float(eq.iloc[-1] - 1), "buy_hold_return": float(bh.iloc[-1] - 1),
        "ann_vol": vol, "sharpe": sharpe, "max_drawdown": mdd,
        "buy_hold_ann_vol": bh_vol, "buy_hold_sharpe": bh_sharpe, "buy_hold_max_drawdown": bh_mdd,
        "avg_exposure": float(np.mean(exposure)) if exposure else 0.0,
        "trades": trades, "stop_hits": stop_hits, "decisions": len(decisions), "llm_cost_usd": round(llm_cost, 4),
    }
    return BacktestResult(ticker=ticker, start=dates[0], end=dates[-1], decisions=decisions, equity=curve, metrics=metrics, warnings=[LOOKAHEAD_WARNING])
