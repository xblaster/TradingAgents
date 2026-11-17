"""Statistical screener: every asset gets the full diagnostic pack; a few get flagged as "gems".

A gem is an asset where a statistical condition is unusually strong, defined by explicit rules
below. Gems are where the LLM's judgement is worth paying for; everything else is decided by
statistics alone (see decision.py).

Multiple-testing caveat: with N assets x 5 rules, some flags will be chance. The rules use strict
thresholds (|t| > 3, robust z > 3-4) to keep false positives rare, and the screen reports how many
tests were run so the reader can weigh the hits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .data import DataProvider
from .data.indicators import rsi
from .portfolio import TrendConfig, trend_signals
from .stats import cross_asset, diagnostics as dg, performance as pf, regimes, significance as sg

LOOKBACK_DAYS = 800
MIN_RANK_UNIVERSE = 3


@dataclass(frozen=True)
class GemRule:
    code: str
    name: str
    condition: str


RULES = {
    "G1": GemRule("G1", "Emerging significant trend", "60-day trend HAC t-stat > 3, momentum rank >= 70th percentile, price < 2.5 sd above its 20-day mean"),
    "G2": GemRule("G2", "Volume-confirmed breakout", "close at a 90-day high and 5-day volume >= 2x its 60-day median, or today's volume robust z > 3"),
    "G3": GemRule("G3", "Statistical capitulation", "today's return robust z < -4, or price < -2.5 sd under its 20-day mean with RSI < 25"),
    "G4": GemRule("G4", "Cointegration dislocation", "Engle-Granger p < 0.05 with another pair, spread |z| > 2.5, half-life < 30 days: this asset is the cheap leg"),
    "G5": GemRule("G5", "Volatility squeeze", "20-day volatility in the bottom 10% of its 1-year range while the 120-day trend is significant (|t| > 2)"),
}


@dataclass
class Gem:
    rule: str
    strength: float  # >= 1 means the threshold is met; higher is stronger
    evidence: str


@dataclass
class AssetStats:
    ticker: str
    pack: dict[str, Any]
    score: float  # -1 (statistically bearish) .. +1 (statistically bullish)
    gems: list[Gem] = field(default_factory=list)


@dataclass
class Screen:
    as_of: date
    assets: dict[str, AssetStats]
    universe: dict[str, Any]
    tests_run: int

    @property
    def gems(self) -> list[tuple[str, Gem]]:
        flat = [(t, g) for t, a in self.assets.items() for g in a.gems]
        return sorted(flat, key=lambda x: -x[1].strength)


def _num(x: Any) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def asset_pack(frame: pd.DataFrame, ppy: int = 365) -> dict[str, Any]:
    """Every statistic the desk knows about one asset, from its OHLCV history (completed bars)."""
    close, volume = frame["Close"].dropna(), frame["Volume"].fillna(0)
    simple = close.pct_change().dropna()
    last_year = simple.iloc[-365:]
    window_1y = close.iloc[-366:]
    prior_90 = close.iloc[-91:-1]
    vol_med_60 = float(volume.iloc[-65:-5].median()) if len(volume) > 65 else float("nan")
    signal = trend_signals(close.to_frame("x"), TrendConfig().lookbacks)["x"].iloc[-1]
    return {
        "last_close": float(close.iloc[-1]),
        "last_date": close.index[-1].date().isoformat(),
        "returns": {k: _num(close.iloc[-1] / close.iloc[-1 - d] - 1) if len(close) > d else None for k, d in (("1w", 7), ("1m", 30), ("3m", 91), ("6m", 182), ("1y", 365))},
        "trend_ensemble_signal": _num(signal),
        "trend_60d": dg.trend_tstat(close, 60, ppy),
        "trend_120d": dg.trend_tstat(close, 120, ppy),
        "variance_ratio_5": dg.variance_ratio(window_1y, 5),
        "hurst_1y": dg.hurst_exponent(window_1y),
        "ljung_box_1y": dg.ljung_box(np.log(window_1y).diff().dropna(), 10) if len(window_1y) > 60 else None,
        "volatility_regime": regimes.volatility_regime(close, ppy=ppy),
        "zscore_20d": dg.zscore_vs_mean(close, 20),
        "rsi_14": _num(rsi(close).iloc[-1]),
        "return_robust_z": dg.robust_zscore(simple, 60),
        "volume_robust_z": dg.robust_zscore(volume, 60),
        "volume_5d_vs_60d_median": _num(volume.iloc[-5:].mean() / vol_med_60) if vol_med_60 and vol_med_60 > 0 else None,
        "at_90d_high": bool(len(prior_90) == 90 and close.iloc[-1] >= prior_90.max()),
        "pct_from_90d_high": _num(close.iloc[-1] / close.iloc[-91:].max() - 1),
        "performance_1y": pf.summary(last_year, ppy),
        "psr_1y": _num(sg.probabilistic_sharpe_ratio(last_year)) if len(last_year) > 30 else None,
    }


def stat_score(pack: dict[str, Any], rank_avg: float | None) -> float:
    """Composite statistical view in [-1, 1]: trend ensemble, 120-day trend significance, relative strength."""
    parts = []
    if pack["trend_ensemble_signal"] is not None:
        parts.append(2 * pack["trend_ensemble_signal"] - 1)
    t = _num(pack["trend_120d"]["t_stat"])
    if t is not None:
        parts.append(math.tanh(t / 3))
    if rank_avg is not None and math.isfinite(rank_avg):
        parts.append(2 * rank_avg - 1)
    return float(np.mean(parts)) if parts else 0.0


def detect_gems(pack: dict[str, Any], rank_avg: float | None) -> list[Gem]:
    gems: list[Gem] = []
    t60 = _num(pack["trend_60d"]["t_stat"]) or 0.0
    z20 = pack["zscore_20d"]
    if t60 > 3 and (rank_avg is None or rank_avg >= 0.7) and (z20 is None or z20 < 2.5):
        rank_txt = "n/a (universe too small)" if rank_avg is None else f"{rank_avg:.0%}"
        gems.append(Gem("G1", t60 / 3, f"60d trend t={t60:.1f} (slope {pack['trend_60d']['slope_annual']:+.0%}/yr), momentum rank {rank_txt}, z20={z20:.1f}"))
    vol_ratio, vol_z = pack["volume_5d_vs_60d_median"], pack["volume_robust_z"]
    if pack["at_90d_high"] and ((vol_ratio or 0) >= 2 or (vol_z or 0) > 3):
        strength = max((vol_ratio or 0) / 2, (vol_z or 0) / 3)
        gems.append(Gem("G2", strength, f"90-day high, 5d volume {vol_ratio or float('nan'):.1f}x the 60d median, volume z={vol_z or float('nan'):.1f}"))
    ret_z, rsi_v = pack["return_robust_z"], pack["rsi_14"]
    if (ret_z is not None and ret_z < -4) or (z20 is not None and z20 < -2.5 and rsi_v is not None and rsi_v < 25):
        strength = max(-(ret_z or 0) / 4, -(z20 or 0) / 2.5)
        gems.append(Gem("G3", strength, f"daily return robust z={ret_z or float('nan'):.1f}, z20={z20 or float('nan'):.1f}, RSI {rsi_v or float('nan'):.0f}"))
    vreg = pack["volatility_regime"]
    t120 = _num(pack["trend_120d"]["t_stat"]) or 0.0
    if vreg.get("percentile_1y") is not None and vreg["percentile_1y"] < 0.10 and abs(t120) > 2:
        gems.append(Gem("G5", (0.10 / max(vreg["percentile_1y"], 0.01)) * abs(t120) / 2 / 2,
                        f"20d vol {vreg['current_vol']:.0%} at the {vreg['percentile_1y']:.0%} percentile of its year, 120d trend t={t120:.1f}"))
    return gems


async def screen(provider: DataProvider, tickers: tuple[str, ...], as_of: date | None = None) -> Screen:
    as_of = as_of or date.today()
    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        f = await provider.prices(t, as_of - timedelta(days=LOOKBACK_DAYS), as_of)
        f = f[f.index < pd.Timestamp(as_of)] if as_of == date.today() else f  # today's bar is incomplete
        if len(f) >= 200:
            frames[t] = f
    closes = pd.DataFrame({t: f["Close"] for t, f in frames.items()}).sort_index()
    ranks = cross_asset.momentum_ranks(closes)
    coint = cross_asset.cointegration_scan(closes) if closes.shape[1] >= 2 else []

    assets: dict[str, AssetStats] = {}
    for t, f in frames.items():
        pack = asset_pack(f)
        rank = _num(ranks.loc[t, "rank_avg"]) if len(frames) >= MIN_RANK_UNIVERSE else None  # a rank among 1-2 assets says nothing
        pack["momentum_rank"] = rank
        a = AssetStats(t, pack, stat_score(pack, rank), detect_gems(pack, rank))
        assets[t] = a
    for hit in coint:
        if abs(hit["spread_z"]) > 2.5 and hit["half_life"] is not None and hit["half_life"] < 30:
            leg = hit["cheap_leg"]
            other = hit["b"] if leg == hit["a"] else hit["a"]
            assets[leg].gems.append(Gem("G4", abs(hit["spread_z"]) / 2.5,
                                        f"cointegrated with {other} (p={hit['p_value']:.3f}), spread z={hit['spread_z']:+.1f}, half-life {hit['half_life']:.0f}d"))
    n = len(assets)
    universe = {
        "assets": n,
        "average_correlation_90d": _num(cross_asset.average_correlation(closes)) if n > 1 else None,
        "dispersion_30d": _num(cross_asset.dispersion(closes)) if n > 1 else None,
        "cointegrated_pairs": coint,
        "pairs_tested": math.comb(n, 2),
    }
    return Screen(as_of=as_of, assets=assets, universe=universe, tests_run=n * 4 + math.comb(n, 2))


def render_screen(s: Screen) -> str:
    def p(x: float | None, digits: int = 0) -> str:
        return "n/a" if x is None else f"{x * 100:+.{digits}f}%"

    out = [
        f"# Statistical screen {s.as_of}",
        "",
        f"{s.universe['assets']} assets, average 90d correlation {s.universe['average_correlation_90d'] or float('nan'):.2f}, "
        f"30d dispersion {p(s.universe['dispersion_30d'])}, {s.tests_run} statistical tests run.",
        "",
        "| asset | 1m | 3m | rank | trend t120 | VR(5) | Hurst | vol regime | ensemble | PSR 1y | score | gems |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t, a in sorted(s.assets.items(), key=lambda kv: -kv[1].score):
        pk = a.pack
        vr = pk["variance_ratio_5"]
        vr_txt = f"{vr['vr']:.2f}{'*' if vr['p_value'] < 0.05 else ''}"
        rank_txt = "n/a" if pk["momentum_rank"] is None else f"{pk['momentum_rank']:.0%}"
        out.append(
            f"| {t} | {p(pk['returns']['1m'])} | {p(pk['returns']['3m'])} | {rank_txt} | {pk['trend_120d']['t_stat']:+.1f} | "
            f"{vr_txt} | {pk['hurst_1y'] or float('nan'):.2f} | {pk['volatility_regime'].get('label')} | {pk['trend_ensemble_signal'] or 0:.2f} | "
            f"{pk['psr_1y'] or float('nan'):.2f} | {a.score:+.2f} | {', '.join(g.rule for g in a.gems) or '-'} |"
        )
    out += ["", "VR(5) marked * when significantly different from 1 (p < 0.05): > 1 momentum, < 1 mean reversion.", "", "## Gems"]
    if not s.gems:
        out.append("None today: no statistical condition is strong enough to justify an LLM review.")
    for t, g in s.gems:
        out.append(f"- **{t}**, {g.rule} {RULES[g.rule].name} (strength {g.strength:.1f}): {g.evidence}")
    out += ["", "## Rules", *[f"- {r.code} {r.name}: {r.condition}" for r in RULES.values()]]
    return "\n".join(out) + "\n"
