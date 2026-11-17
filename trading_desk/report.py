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


def render_backtest_markdown(result: "BacktestResult") -> str:
    m = result.metrics
    out = [
        f"# Backtest {result.ticker}: {result.start} to {result.end}",
        "",
        *[f"> {w}" for w in result.warnings],
        "",
        "| metric | strategy | buy & hold |",
        "|---|---|---|",
        f"| total return | {_pct(m['total_return'])} | {_pct(m['buy_hold_return'])} |",
        f"| annualized vol | {_pct(m['ann_vol'])} | {_pct(m['buy_hold_ann_vol'])} |",
        f"| Sharpe (rf=0) | {m['sharpe']:.2f} | {m['buy_hold_sharpe']:.2f} |",
        f"| max drawdown | {_pct(m['max_drawdown'])} | {_pct(m['buy_hold_max_drawdown'])} |",
        "",
        f"Average exposure {m['avg_exposure'] * 100:.1f}%, {m['trades']} trades, {m['stop_hits']} stop hits, "
        f"{m['decisions']} decisions, LLM cost ${m['llm_cost_usd']:.2f}.",
        "",
        "| date | action | target | conviction | stop | notes |",
        "|---|---|---|---|---|---|",
    ]
    out += [
        f"| {d['date']} | {d['action']} | {d['target_position_pct']:.1f}% | {d['conviction']:.2f} | {d['stop_loss'] or ''} | {'; '.join(d['adjustments'])[:120]} |"
        for d in result.decisions
    ]
    return "\n".join(out) + "\n"
