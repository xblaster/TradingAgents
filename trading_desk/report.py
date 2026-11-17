"""Human-readable Markdown for runs and backtests."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backtest import BacktestResult
    from .pipeline import DeskRun


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:+.1f}%"


def render_markdown(run: "DeskRun") -> str:
    d, snap = run.decision, run.snapshot
    out = [
        f"# {run.ticker} on {run.as_of}: **{d.action}**",
        "",
        f"- Target position: **{d.target_position_pct:.1f}%** (was {d.current_position_pct:.1f}%)",
        f"- Conviction: {d.conviction:.2f}, horizon {d.horizon_days} days",
        f"- Last close: {snap.last_close} ({snap.last_bar_date}), stop-loss: {d.stop_loss or 'none'}, take-profit: {d.take_profit or 'none'}",
        f"- Mode: {snap.mode}, LLM cost: ${run.cost_usd:.4f}",
    ]
    if d.adjustments:
        out += ["", "**Guardrail adjustments**", *[f"- {a}" for a in d.adjustments]]
    if d.proposed:
        out += ["", "## Portfolio manager", d.proposed.rationale, "", f"*Invalidation:* {d.proposed.invalidation}"]
    if run.risk:
        r = run.risk
        out += [
            "", f"## Risk manager: {r.risk_level}",
            f"Max position {r.max_position_pct:.1f}%, stop distance {r.stop_distance_pct:.1f}%" + (f", **VETO**: {r.veto_reason}" if r.veto else ""),
            *[f"- {k}" for k in r.key_risks],
        ]
    for i, turn in enumerate(run.debate, 1):
        out += ["", f"## Debate, round {i}"]
        for side, arg in turn.items():
            out += [f"**{side.capitalize()}**: {arg.thesis}", *[f"- {a}" for a in arg.arguments], ""]
    out += ["## Analysts"]
    for name, rep in run.analysts.items():
        out += [
            "", f"### {name.capitalize()}: {rep.stance} ({rep.confidence:.2f})", rep.summary, "",
            *[f"- {p}" for p in rep.key_points],
        ]
        if rep.risks:
            out += ["", "*Risks:* " + "; ".join(rep.risks)]
        if rep.data_gaps:
            out += ["*Data gaps:* " + "; ".join(rep.data_gaps)]
    failed = [s for s in run.steps if not s.ok]
    out += ["", "## Calls", "| agent | model | ok | cost | time | cached |", "|---|---|---|---|---|---|"]
    out += [f"| {s.agent} | {s.model} | {'yes' if s.ok else 'NO'} | ${s.cost_usd:.4f} | {s.duration_ms / 1000:.1f}s | {s.cached} |" for s in run.steps]
    if failed:
        out += ["", "**Failures**", *[f"- {s.agent}: {s.error}" for s in failed]]
    if snap.notes:
        out += ["", "## Data caveats", *[f"- {n}" for n in snap.notes]]
    return "\n".join(out) + "\n"


STRATEGY_LABELS = {
    "llm": "LLM desk",
    "buy_hold": "Buy & hold (capped)",
    "trend_sma": "Trend SMA20/50 (capped)",
    "momentum": "Momentum 3m (capped)",
    "buy_hold_100": "Buy & hold 100% (reference)",
}


def strategy_table(result: "BacktestResult") -> list[str]:
    rows = [
        "| strategy | return | CAGR | vol | Sharpe | max DD | exposure | trades | fees |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in result.strategies.items():
        rows.append(
            f"| {STRATEGY_LABELS.get(name, name)} | {_pct(m['total_return'])} | {_pct(m['cagr'])} | {_pct(m['ann_vol'])} | "
            f"{m['sharpe']:.2f} | {_pct(m['max_drawdown'])} | {m['avg_exposure'] * 100:.0f}% | {m['trades']:.0f} | {m['fees'] * 100:.2f}% |"
        )
    return rows


def render_backtest_markdown(result: "BacktestResult") -> str:
    out = [
        f"# Backtest {result.ticker} ({result.asset_class}): {result.start} to {result.end}",
        "",
        f"Decision every {result.every} bars, fills at next open, costs {result.cost_bps:g} bps per unit of turnover, "
        f"position cap {result.max_position_pct:g}% for every capped strategy.",
        "",
        *[f"> {w}" for w in result.warnings],
        "",
        *strategy_table(result),
    ]
    llm = result.strategies.get("llm")
    if llm:
        out += [
            "", f"LLM desk: {llm['decisions']:.0f} decisions, {llm['stop_hits']:.0f} stop hits, LLM cost ${llm['llm_cost_usd']:.2f}.",
            "", "## LLM decisions", "",
            "| date | action | target | conviction | stop | notes |",
            "|---|---|---|---|---|---|",
        ]
        out += [
            f"| {d['date']} | {d['action']} | {d['target_position_pct']:.1f}% | {d['conviction']:.2f} | {d['stop_loss'] or ''} | {'; '.join(d['adjustments'])[:120]} |"
            for d in result.decisions
        ]
    return "\n".join(out) + "\n"
