"""Walk-forward backtests: the LLM desk against simple rule-based baselines, on the same terms.

Execution model, identical for every strategy (deliberately conservative):
  * a strategy decides on the close of bar t, seeing bars up to t only;
  * the trade fills at the open of t+1 and pays `cost_bps` on turnover (fees + spread);
  * a long position whose low crosses its stop exits at the stop (or at the open on a gap);
  * decisions happen every `every` bars, and every strategy is capped at the same position size,
    so that "the desk beat buy & hold" is not just "the desk was more invested".

Warning: an LLM whose training data covers the backtest period may remember what happened.
Point-in-time data does not fix that. Only periods after the models' knowledge cutoff give an
uncontaminated estimate of the LLM desk; baselines have no such problem.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .data import DataProvider
from .data.indicators import horizons, sma, trailing_return
from .data.models import asset_class_of, periods_per_year
from .pipeline import DeskRun, TradingDesk

log = logging.getLogger(__name__)

LOOKAHEAD_WARNING = (
    "Look-ahead risk: if the period predates the models' knowledge cutoff, the LLM may know how "
    "it ended. Treat the LLM desk's results on such periods as a pipeline check, not as evidence of an edge."
)
DEFAULT_COST_BPS = {"crypto": 15.0, "equity": 5.0}  # crypto: Revolut X taker fee + spread + slippage
WARMUP_DAYS = 400
BASELINES = ("buy_hold", "trend_sma", "momentum")


@dataclass
class Signal:
    target: float  # desired weight, 0..1
    stop: float | None = None


Decider = Callable[[int, pd.DataFrame, float], Awaitable[Signal]]  # (bar index, bars so far, current weight)


@dataclass
class StrategyRun:
    curve: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    trades: int = 0
    stop_hits: int = 0
    fees: float = 0.0
    decisions: int = 0


class BacktestResult(BaseModel):
    ticker: str
    asset_class: str
    start: date
    end: date
    every: int
    cost_bps: float
    max_position_pct: float
    strategies: dict[str, dict[str, float | None]]
    equity: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    warnings: list[str]

    def save(self, results_dir: Path) -> Path:
        from .report import render_backtest_markdown

        folder = results_dir / self.ticker / f"backtest_{self.start}_{self.end}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "backtest.json").write_text(self.model_dump_json(indent=2))
        (folder / "report.md").write_text(render_backtest_markdown(self))
        pd.DataFrame(self.equity).to_csv(folder / "equity.csv", index=False)
        return folder


# ---------------------------------------------------------------- simulation


def _drift(w: float, r: float) -> float:
    """New weight of a position at weight w after it returns r (the rest is cash at 0%)."""
    denom = 1 + w * r
    return w * (1 + r) / denom if denom > 0 else 0.0


async def simulate(
    bars: pd.DataFrame, start_idx: int, decide: Decider, *, every: int, cost_bps: float, min_trade: float = 0.01,
) -> StrategyRun:
    """Run one strategy. Weight changes smaller than `min_trade` (1% of the portfolio) are skipped,
    so constant targets do not pay to rebalance price drift; exits to zero always execute."""
    o, l, c = (bars[col].to_numpy(dtype=float) for col in ("Open", "Low", "Close"))
    cost = cost_bps / 10_000
    run = StrategyRun(curve=[1.0], weights=[0.0])
    equity, w, stop = 1.0, 0.0, None
    pending: Signal | None = None

    for i in range(start_idx, len(bars)):
        if i > start_idx:
            r_gap = o[i] / c[i - 1] - 1  # overnight / since last close
            equity *= 1 + w * r_gap
            w = _drift(w, r_gap)
            if pending is not None:  # fill at the open
                turnover = abs(pending.target - w)
                if turnover >= min_trade or (pending.target == 0 and w > 0):
                    equity *= 1 - turnover * cost
                    run.fees += turnover * cost
                    run.trades += 1
                    w = pending.target
                stop = pending.stop if w > 0 else None
                pending = None
            if w > 0 and stop is not None and l[i] <= stop:
                exit_px = min(o[i], stop)
                equity *= 1 + w * (exit_px / o[i] - 1)
                equity *= 1 - w * cost
                run.fees += w * cost
                w, stop = 0.0, None
                run.stop_hits += 1
                run.trades += 1
            else:
                r_day = c[i] / o[i] - 1
                equity *= 1 + w * r_day
                w = _drift(w, r_day)
            run.curve.append(equity)
            run.weights.append(w)

        if (i - start_idx) % every == 0 and i < len(bars) - 1:
            pending = await decide(i, bars.iloc[: i + 1], w)
            pending.target = min(max(pending.target, 0.0), 1.0)
            run.decisions += 1
    return run


def metrics(run: StrategyRun, ppy: int) -> dict[str, float | None]:
    eq = pd.Series(run.curve)
    rets = eq.pct_change().dropna()
    std = float(rets.std()) if len(rets) > 1 else 0.0
    years = max(len(rets) / ppy, 1e-9)
    total = float(eq.iloc[-1] - 1)
    return {
        "total_return": total,
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1) if years >= 0.5 and eq.iloc[-1] > 0 else None,
        "ann_vol": std * math.sqrt(ppy),
        "sharpe": float(rets.mean() / std * math.sqrt(ppy)) if std > 0 else 0.0,
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "avg_exposure": float(np.mean(run.weights[1:])) if len(run.weights) > 1 else 0.0,
        "trades": float(run.trades),
        "stop_hits": float(run.stop_hits),
        "fees": run.fees,
        "psr": _psr(rets),
    }


def _psr(rets: pd.Series) -> float | None:
    from .stats.significance import probabilistic_sharpe_ratio

    if len(rets) < 30 or rets.std() == 0:
        return None
    value = probabilistic_sharpe_ratio(rets)
    return float(value) if np.isfinite(value) else None


# ------------------------------------------------------------------ baselines


def baseline(name: str, cap: float, ppy: int) -> Decider:
    """Rule-based strategies, using only bars up to the decision bar."""
    lookback = horizons(ppy)["3m"]

    async def buy_hold(i: int, hist: pd.DataFrame, w: float) -> Signal:
        return Signal(cap)

    async def trend_sma(i: int, hist: pd.DataFrame, w: float) -> Signal:
        close = hist["Close"]
        s20, s50 = sma(close, 20).iloc[-1], sma(close, 50).iloc[-1]
        return Signal(cap if pd.notna(s50) and close.iloc[-1] > s50 and s20 > s50 else 0.0)

    async def momentum(i: int, hist: pd.DataFrame, w: float) -> Signal:
        r = trailing_return(hist["Close"], lookback)
        return Signal(cap if r is not None and r > 0 else 0.0)

    return {"buy_hold": buy_hold, "trend_sma": trend_sma, "momentum": momentum}[name]


# ------------------------------------------------------------------ entry point


async def run_backtest(
    desk: TradingDesk | None,
    provider: DataProvider,
    ticker: str,
    start: date,
    end: date,
    *,
    every: int = 5,
    cost_bps: float | None = None,
    max_position_pct: float | None = None,
    strategies: tuple[str, ...] = ("llm", *BASELINES),
    on_decision: Callable[[DeskRun], None] | None = None,
) -> BacktestResult:
    ticker = ticker.upper()
    asset_class = asset_class_of(ticker)
    ppy = periods_per_year(asset_class)
    cost_bps = DEFAULT_COST_BPS[asset_class] if cost_bps is None else cost_bps
    if max_position_pct is None:
        max_position_pct = desk.settings.risk.max_position_pct if desk else 100.0
    cap = max_position_pct / 100
    unknown = set(strategies) - {"llm", "stats", "hybrid", *BASELINES}
    if unknown:
        raise ValueError(f"Unknown strategies: {sorted(unknown)}")
    if {"llm", "hybrid"} & set(strategies) and desk is None:
        raise ValueError("The llm and hybrid strategies need a desk")

    bars = await provider.prices(ticker, start - timedelta(days=WARMUP_DAYS), end)
    in_range = bars.index >= pd.Timestamp(start)
    if in_range.sum() < 2:
        raise ValueError(f"Not enough bars for {ticker} between {start} and {end}")
    start_idx = int(np.argmax(in_range))
    dates = [ts.date() for ts in bars.index[start_idx:]]

    decisions: list[dict[str, Any]] = []
    llm_cost = 0.0

    async def llm(i: int, hist: pd.DataFrame, w: float) -> Signal:
        nonlocal llm_cost
        run = await desk.run(ticker, hist.index[-1].date(), current_position_pct=round(w * 100, 2))
        llm_cost += run.cost_usd
        d = run.decision
        decisions.append({
            "date": hist.index[-1].date().isoformat(), "action": d.action, "target_position_pct": d.target_position_pct,
            "conviction": d.conviction, "stop_loss": d.stop_loss, "adjustments": d.adjustments, "cost_usd": run.cost_usd,
        })
        if on_decision:
            on_decision(run)
        return Signal(d.target_position_pct / 100, d.stop_loss)

    from .config import DeskSettings
    from .decision import HybridDesk

    routes: dict[str, dict[str, int]] = {}

    def hybrid_decider(name: str, hybrid: HybridDesk) -> Decider:
        async def decide(i: int, hist: pd.DataFrame, w: float) -> Signal:
            nonlocal llm_cost
            d = await hybrid.decide(ticker, hist.index[-1].date(), current_position_pct=round(w * 100, 2))
            routes.setdefault(name, {})
            routes[name][d.decided_by] = routes[name].get(d.decided_by, 0) + 1
            if name == "hybrid":
                llm_cost += d.llm_cost_usd
            return Signal(d.target_position_pct / 100, d.stop_loss)
        return decide

    settings = desk.settings if desk else DeskSettings()
    if desk is None:
        settings.risk.max_position_pct = max_position_pct

    runs: dict[str, StrategyRun] = {}
    for name in strategies:
        if name == "llm":
            decider = llm
        elif name == "stats":
            decider = hybrid_decider(name, HybridDesk(settings, provider, None, llm_on_gems=False, llm_last_resort=False))
        elif name == "hybrid":
            decider = hybrid_decider(name, HybridDesk(settings, provider, desk))
        else:
            decider = baseline(name, cap, ppy)
        runs[name] = await simulate(bars, start_idx, decider, every=every, cost_bps=cost_bps)
    # Reference: fully invested, no cap, no costs beyond the first fill.
    runs["buy_hold_100"] = await simulate(bars, start_idx, baseline("buy_hold", 1.0, ppy), every=every, cost_bps=cost_bps)

    stats = {name: metrics(run, ppy) for name, run in runs.items()}
    if "llm" in stats:
        stats["llm"]["decisions"] = float(runs["llm"].decisions)
        stats["llm"]["llm_cost_usd"] = round(llm_cost, 4)
    for name, counts in routes.items():
        stats[name]["decisions"] = float(runs[name].decisions)
        for route, n in counts.items():
            stats[name][f"route_{route}"] = float(n)
    if "hybrid" in stats:
        stats["hybrid"]["llm_cost_usd"] = round(llm_cost - stats.get("llm", {}).get("llm_cost_usd", 0.0), 4)

    equity = [
        {"date": d.isoformat(), **{name: round(run.curve[k], 6) for name, run in runs.items()}, "llm_weight": round(runs["llm"].weights[k], 4) if "llm" in runs else None}
        for k, d in enumerate(dates)
    ]
    uses_llm = bool({"llm", "hybrid"} & set(strategies))
    warnings = [LOOKAHEAD_WARNING] if uses_llm else []
    cutoff = desk.settings.llm.knowledge_cutoff if desk else None
    if uses_llm and cutoff:
        warnings.append(
            f"Whole period is after the declared knowledge cutoff ({cutoff}): an uncontaminated test."
            if start > cutoff else f"Period starts before the declared knowledge cutoff ({cutoff}): LLM results are contaminated."
        )
    return BacktestResult(
        ticker=ticker, asset_class=asset_class, start=dates[0], end=dates[-1], every=every, cost_bps=cost_bps,
        max_position_pct=max_position_pct, strategies=stats, equity=equity, decisions=decisions, warnings=warnings,
    )
