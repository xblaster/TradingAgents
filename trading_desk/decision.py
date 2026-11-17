"""Stats-first decisions. The LLM desk is consulted only for gems, or as a last resort.

    statistics clear  ──► decide from statistics (free, deterministic)
    gem flagged       ──► LLM desk, with the statistical pack and the gem evidence in its prompt
    ambiguous         ──► LLM desk as last resort (if allowed), otherwise hold the current position

"Clear" means the composite statistical score and the trend ensemble agree with some strength;
the exact thresholds are below and are deliberately simple.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .config import DeskSettings
from .data import DataProvider
from .data.models import asset_class_of
from .pipeline import DeskRun, TradingDesk
from .portfolio import EXTENDED
from .screener import RULES, AssetStats, screen

CLEAR_SCORE = 0.5  # |composite score| needed to act on statistics alone


@dataclass
class StatsVerdict:
    action: str
    target_position_pct: float
    stop_loss: float | None
    confidence: float
    clear: bool
    reasons: list[str]


@dataclass
class HybridDecision:
    ticker: str
    as_of: date
    decided_by: str  # "stats" | "llm_gem" | "llm_last_resort" | "stats_hold"
    action: str
    target_position_pct: float
    current_position_pct: float
    stop_loss: float | None
    stats: StatsVerdict
    gems: list[dict[str, Any]] = field(default_factory=list)
    llm_run: DeskRun | None = None
    llm_cost_usd: float = 0.0
    notes: list[str] = field(default_factory=list)


def stats_verdict(asset: AssetStats, current_pct: float, settings: DeskSettings) -> StatsVerdict:
    pack, score = asset.pack, asset.score
    signal = pack["trend_ensemble_signal"] or 0.0
    cap = settings.risk.max_position_pct
    vreg = pack["volatility_regime"].get("label")
    target = cap * signal * (0.5 if vreg == "extreme" else 1.0)
    reasons = [f"composite score {score:+.2f}", f"trend ensemble {signal:.2f}", f"volatility regime {vreg}"]

    agree_long = score >= CLEAR_SCORE and signal >= 0.5
    agree_flat = score <= -CLEAR_SCORE and signal <= 0.25
    clear = agree_long or agree_flat
    if not clear:
        reasons.append(f"ambiguous: |score| < {CLEAR_SCORE} or score and trend ensemble disagree")

    # Stop at roughly a 2-sigma 10-day move, within the configured limits.
    daily_vol = (pack["volatility_regime"].get("current_vol") or 0.6) / math.sqrt(365)
    stop_pct = min(max(2 * daily_vol * math.sqrt(10), 0.02), settings.risk.max_stop_distance_pct / 100)
    last = pack["last_close"]
    stop = round(last * (1 - stop_pct), 6) if target > 0 else None

    if not clear:
        action, target = "HOLD", current_pct
    elif abs(target - current_pct) < 2:
        action, target = "HOLD", current_pct
    else:
        action = "BUY" if target > current_pct else "SELL"
    return StatsVerdict(action, round(target, 2), stop if target > 0 else None, min(abs(score), 1.0), clear, reasons)


def compact_context(asset: AssetStats, verdict: StatsVerdict, why: str) -> dict[str, Any]:
    """What the LLM sees: the statistical facts, the statistical verdict and why it is consulted."""
    pk = asset.pack

    def r(x: Any, d: int = 3) -> Any:
        return round(x, d) if isinstance(x, float) else x

    return {
        "why_you_are_consulted": why,
        "gems": [{"rule": g.rule, "name": RULES[g.rule].name, "strength": r(g.strength, 2), "evidence": g.evidence} for g in asset.gems],
        "statistical_verdict": {"action": verdict.action, "target_position_pct": verdict.target_position_pct, "clear": verdict.clear, "reasons": verdict.reasons},
        "composite_score": r(asset.score, 2),
        "momentum_rank_in_universe": r(pk["momentum_rank"], 2),
        "trend_ensemble_signal": pk["trend_ensemble_signal"],
        "trend_60d": {k: r(v) for k, v in pk["trend_60d"].items()},
        "trend_120d": {k: r(v) for k, v in pk["trend_120d"].items()},
        "variance_ratio_5": {k: r(v) for k, v in pk["variance_ratio_5"].items()},
        "hurst_1y": r(pk["hurst_1y"]),
        "volatility_regime": {k: r(v) for k, v in pk["volatility_regime"].items()},
        "zscore_20d": r(pk["zscore_20d"]),
        "return_robust_z": r(pk["return_robust_z"]),
        "volume_robust_z": r(pk["volume_robust_z"]),
        "psr_1y": r(pk["psr_1y"]),
        "performance_1y": {k: r(v) for k, v in pk["performance_1y"].items() if k in ("sharpe", "sortino", "max_drawdown", "cvar_95", "skew", "excess_kurtosis")},
    }


class HybridDesk:
    def __init__(
        self,
        settings: DeskSettings,
        provider: DataProvider,
        llm_desk: TradingDesk | None = None,
        *,
        universe: tuple[str, ...] | None = None,
        llm_on_gems: bool = True,
        llm_last_resort: bool = True,
    ):
        self.settings = settings
        self.provider = provider
        self.llm_desk = llm_desk
        self.universe = universe
        self.llm_on_gems = llm_on_gems
        self.llm_last_resort = llm_last_resort
        self.llm_calls = 0

    def _universe_for(self, ticker: str) -> tuple[str, ...]:
        if self.universe is not None:  # an explicit empty tuple means "this asset alone"
            base = self.universe
        else:
            base = EXTENDED if asset_class_of(ticker) == "crypto" else ()
        return tuple(dict.fromkeys((ticker, *base)))

    async def decide(self, ticker: str, as_of: date, current_position_pct: float = 0.0) -> HybridDecision:
        ticker = ticker.upper()
        sc = await screen(self.provider, self._universe_for(ticker), as_of)
        if ticker not in sc.assets:
            raise ValueError(f"Not enough history to screen {ticker} on {as_of}")
        asset = sc.assets[ticker]
        verdict = stats_verdict(asset, current_position_pct, self.settings)
        gems = [{"rule": g.rule, "strength": g.strength, "evidence": g.evidence} for g in asset.gems]

        route, why = "stats", ""
        if asset.gems and self.llm_on_gems:
            route, why = "llm_gem", "Statistical screen flagged this asset: " + "; ".join(f"{g.rule} {RULES[g.rule].name}" for g in asset.gems)
        elif not verdict.clear:
            route, why = ("llm_last_resort", "Statistics are ambiguous; your judgement is the tie-breaker.") if self.llm_last_resort else ("stats_hold", "")

        if route.startswith("llm") and self.llm_desk is not None:
            run = await self.llm_desk.run(ticker, as_of, current_position_pct, stats_context=compact_context(asset, verdict, why))
            self.llm_calls += 1
            d = run.decision
            return HybridDecision(ticker, as_of, route, d.action, d.target_position_pct, current_position_pct, d.stop_loss, verdict, gems, run, run.cost_usd,
                                  notes=[why, *d.adjustments])
        if route.startswith("llm"):
            route = "stats_hold"  # no LLM configured: fall back to holding
        return HybridDecision(ticker, as_of, route if route != "stats" else "stats", verdict.action, verdict.target_position_pct, current_position_pct,
                              verdict.stop_loss, verdict, gems, notes=verdict.reasons)
