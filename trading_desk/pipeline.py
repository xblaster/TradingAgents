"""The desk: data snapshot -> analysts (parallel) -> bull/bear debate -> risk -> PM -> guardrails.

Fail-closed by construction: if the evidence is too thin or a decision-making step fails,
the outcome is HOLD with the reason recorded, never a trade built on an error message.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .agents import prompts
from .agents.runner import AgentResult, StepRecord, run_agent
from .agents.schemas import AnalystReport, DebateArgument, RiskAssessment, TradeDecision
from .config import DeskSettings
from .data import DataProvider, Snapshot
from .guardrails import FinalDecision, apply_guardrails, hold
from .llm import LLMBackend

log = logging.getLogger(__name__)

MIN_ANALYSTS = 2


class DeskRun(BaseModel):
    ticker: str
    as_of: date
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    snapshot: Snapshot
    analysts: dict[str, AnalystReport] = Field(default_factory=dict)
    debate: list[dict[str, DebateArgument]] = Field(default_factory=list)
    risk: RiskAssessment | None = None
    decision: FinalDecision
    steps: list[StepRecord] = Field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        """What this run actually spent: cached answers are free."""
        return round(sum(s.cost_usd for s in self.steps if not s.cached), 4)

    def save(self, results_dir: Path) -> Path:
        folder = results_dir / self.ticker / self.as_of.isoformat()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "run.json").write_text(self.model_dump_json(indent=2))
        from .report import render_markdown

        (folder / "report.md").write_text(render_markdown(self))
        return folder


class TradingDesk:
    def __init__(self, settings: DeskSettings, backend: LLMBackend, provider: DataProvider):
        self.settings = settings
        self.backend = backend
        self.provider = provider

    async def run(self, ticker: str, as_of: date, current_position_pct: float = 0.0) -> DeskRun:
        s = self.settings
        snap = await self.provider.snapshot(ticker.upper(), as_of, s.lookback_days)
        log.info("%s %s: snapshot ready (%s mode, close %.2f)", snap.ticker, as_of, snap.mode, snap.last_close)
        steps: list[StepRecord] = []

        crypto = snap.asset_class == "crypto"

        def sys(role: str, web: bool = False) -> str:
            return prompts.system_prompt(role, as_of=as_of.isoformat(), language=s.language, web=web, crypto=crypto)

        def track(result: AgentResult):
            steps.append(result.record)
            return result.value

        # 1. Analysts, in parallel.
        web = s.web_search and snap.mode == "live"
        analyst_jobs = {"technical": (sys(prompts.TECHNICAL), self._technical_prompt(snap), ())}
        if crypto:
            analyst_jobs["market_structure"] = (sys(prompts.MARKET_STRUCTURE), self._market_prompt(snap), ())
        else:
            analyst_jobs["fundamental"] = (sys(prompts.FUNDAMENTAL), self._fundamental_prompt(snap), ())
        if snap.news or web:  # an analyst with nothing to read would only add a low-confidence "neutral"
            analyst_jobs["news"] = (sys(prompts.NEWS, web=web), self._news_prompt(snap), ("WebSearch",) if web else ())
        results = await asyncio.gather(*(
            run_agent(self.backend, name=f"{name}_analyst", model=s.llm.fast_model, system=system, prompt=prompt, output=AnalystReport, tools=tools)
            for name, (system, prompt, tools) in analyst_jobs.items()
        ))
        analysts = {name: r for name, r in zip(analyst_jobs, (track(r) for r in results)) if r is not None}

        def finish(decision: FinalDecision, **extra) -> DeskRun:
            return DeskRun(ticker=snap.ticker, as_of=as_of, snapshot=snap, analysts=analysts, decision=decision, steps=steps, **extra)

        if len(analysts) < MIN_ANALYSTS:
            return finish(hold(current_position_pct, f"Only {len(analysts)} analyst report(s) succeeded (need {MIN_ANALYSTS}): HOLD."))

        # 2. Bull/bear debate. Both sides speak in parallel each round, replying to the other's last turn.
        reports = self._reports_block(analysts)
        debate: list[dict[str, DebateArgument]] = []
        for rnd in range(s.debate_rounds):
            previous = debate[-1] if debate else {}
            sides = {"bull": (prompts.BULL, previous.get("bear")), "bear": (prompts.BEAR, previous.get("bull"))}
            outs = await asyncio.gather(*(
                run_agent(self.backend, name=f"{side}_r{rnd + 1}", model=s.llm.fast_model, system=sys(role),
                          prompt=self._debate_prompt(snap, reports, opponent), output=DebateArgument)
                for side, (role, opponent) in sides.items()
            ))
            turn = {side: arg for side, arg in zip(sides, (track(o) for o in outs)) if arg is not None}
            if not turn:
                break
            debate.append(turn)

        # 3. Risk manager.
        risk = track(await run_agent(
            self.backend, name="risk_manager", model=s.llm.deep_model, system=sys(prompts.RISK),
            prompt=self._risk_prompt(snap, reports, debate, current_position_pct), output=RiskAssessment,
        ))

        # 4. Portfolio manager.
        proposed = track(await run_agent(
            self.backend, name="portfolio_manager", model=s.llm.deep_model, system=sys(prompts.PORTFOLIO_MANAGER),
            prompt=self._pm_prompt(snap, reports, debate, risk, current_position_pct), output=TradeDecision,
        ))
        if proposed is None:
            return finish(hold(current_position_pct, "Portfolio manager failed: HOLD."), debate=debate, risk=risk)
        if risk is None and proposed.target_position_pct > current_position_pct:
            return finish(hold(current_position_pct, "No risk assessment available: refusing to add exposure.", proposed), debate=debate)

        # 5. Guardrails.
        decision = apply_guardrails(proposed, last_close=snap.last_close, current_position_pct=current_position_pct, limits=s.risk, risk=risk)
        return finish(decision, debate=debate, risk=risk)

    # ------------------------------------------------------------ prompt builders

    @staticmethod
    def _header(snap: Snapshot) -> str:
        notes = "\n".join(f"- {n}" for n in snap.notes) or "- none"
        return f"Ticker: {snap.ticker} ({snap.asset_class})\nAnalysis date: {snap.as_of} ({snap.mode} mode, last bar {snap.last_bar_date}, close {snap.last_close})\nData caveats:\n{notes}\n"

    def _technical_prompt(self, snap: Snapshot) -> str:
        return f"{self._header(snap)}\nIndicators, benchmark comparison and the last 10 daily bars:\n{snap.block('technicals', 'benchmark', 'recent_bars')}\n\nGive your technical assessment."

    def _fundamental_prompt(self, snap: Snapshot) -> str:
        return f"{self._header(snap)}\nQuarterly statements public at the analysis date, and live profile if any:\n{snap.block('fundamentals', 'profile')}\n\nGive your fundamental assessment."

    def _market_prompt(self, snap: Snapshot) -> str:
        return f"{self._header(snap)}\nMarket structure, benchmark comparison and live profile if any:\n{snap.block('market', 'benchmark', 'profile')}\n\nGive your market-structure assessment."

    def _news_prompt(self, snap: Snapshot) -> str:
        if snap.news:
            news = snap.block("news")
        else:
            news = "(no headlines available)"
        profile = snap.block("profile") if snap.profile else "(no profile)"
        return f"{self._header(snap)}\nCompany profile:\n{profile}\n\nRecent headlines:\n{news}\n\nRecent price action for context:\n{snap.block('benchmark')}\n\nGive your news and sentiment assessment."

    @staticmethod
    def _reports_block(analysts: dict[str, AnalystReport]) -> str:
        return json.dumps({k: v.model_dump(mode="json") for k, v in analysts.items()}, indent=1)

    def _debate_prompt(self, snap: Snapshot, reports: str, opponent: DebateArgument | None) -> str:
        rebuttal = f"\nThe other side just argued:\n{opponent.model_dump_json(indent=1)}\n" if opponent else "\nThis is the opening round.\n"
        return f"{self._header(snap)}\nAnalyst reports:\n{reports}\n{rebuttal}\nMake your case."

    @staticmethod
    def _debate_block(debate: list[dict[str, DebateArgument]]) -> str:
        return json.dumps([{side: arg.model_dump(mode="json") for side, arg in turn.items()} for turn in debate], indent=1)

    def _risk_prompt(self, snap: Snapshot, reports: str, debate, current: float) -> str:
        lim = self.settings.risk
        vol = {k: snap.technicals.get(k) for k in ("atr_pct", "realized_vol_20d", "max_drawdown_1y", "rsi_14", "pct_from_52w_high")}
        return (
            f"{self._header(snap)}\nCurrent position: {current}% of portfolio. Hard limit: {lim.max_position_pct}%; "
            f"stops may not be more than {lim.max_stop_distance_pct}% away.\nVolatility profile:\n{json.dumps(vol)}\n"
            f"\nAnalyst reports:\n{reports}\n\nDebate:\n{self._debate_block(debate)}\n\nAssess the risk."
        )

    def _pm_prompt(self, snap: Snapshot, reports: str, debate, risk: RiskAssessment | None, current: float) -> str:
        lim = self.settings.risk
        risk_block = risk.model_dump_json(indent=1) if risk else "(risk assessment unavailable: do not add exposure)"
        return (
            f"{self._header(snap)}\nCurrent position: {current}% of portfolio. Hard limit: {lim.max_position_pct}%. "
            f"Trades with conviction below {lim.min_conviction} will be rejected.\n"
            f"\nAnalyst reports:\n{reports}\n\nDebate:\n{self._debate_block(debate)}\n\nRisk manager:\n{risk_block}\n\nMake the decision."
        )
