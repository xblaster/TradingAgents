"""Runs the pre-registered protocol of STRATEGY.md and returns a PASS/FAIL verdict.

Nothing here is tuned: periods, universes, variants, costs and thresholds are copied from the
protocol. Changing them means changing STRATEGY.md first, with a dated justification.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

import pandas as pd

from .data import DataProvider
from .portfolio import (
    UNIVERSES,
    TrendConfig,
    equal_weight_targets,
    load_prices,
    portfolio_metrics,
    simulate_portfolio,
    single_asset_targets,
    target_weights,
    warmup_start,
)

IN_SAMPLE = (date(2018, 1, 1), date(2022, 12, 31))
OUT_OF_SAMPLE = (date(2023, 1, 1), date(2026, 11, 17))
SCALES = (0.5, 0.75, 1.25, 1.5)
FREQUENCIES = (3, 7, 14)
EW_REBALANCE_DAYS = 30
DD_RATIO = 0.6
ROBUST_SHARE = 0.75


@dataclass
class Criterion:
    code: str
    label: str
    passed: bool
    detail: str


@dataclass
class Evaluation:
    criteria: list[Criterion]
    tables: dict[str, dict[str, dict[str, Any]]]
    variants: list[dict[str, Any]]
    yearly: dict[str, dict[str, float]]
    weights_now: dict[str, float]
    notes: list[str] = field(default_factory=list)
    significance: dict[str, Any] = field(default_factory=dict)

    @property
    def viable(self) -> bool:
        return all(c.passed for c in self.criteria)


class Lab:
    def __init__(self, opens: pd.DataFrame, closes: pd.DataFrame):
        self.opens, self.closes = opens, closes

    def run(self, kind: str, universe: str, period: tuple[date, date], cfg: TrendConfig = TrendConfig()):
        cols = [t for t in UNIVERSES[universe] if t in self.closes.columns]
        closes, opens = self.closes[cols], self.opens[cols]
        if kind == "strategy":
            targets, every = target_weights(closes, cfg), cfg.every
        elif kind == "ew_bh":
            targets, every = equal_weight_targets(closes, cfg), EW_REBALANCE_DAYS
        elif kind == "btc_bh":
            targets, every = single_asset_targets(closes, "BTC-EUR"), 10**9
        else:
            raise ValueError(kind)
        run = simulate_portfolio(opens, closes, targets, *period, every=every, min_trade=cfg.min_trade, cost_bps=cfg.cost_bps)
        return run, portfolio_metrics(run)


def _beats(strat: dict, bench: dict) -> tuple[bool, bool]:
    sharpe_ok = strat["sharpe"] >= bench["sharpe"]
    dd_ok = abs(strat["max_drawdown"]) <= DD_RATIO * abs(bench["max_drawdown"])
    return sharpe_ok, dd_ok


async def evaluate(provider: DataProvider, end: date | None = None) -> Evaluation:
    oos = (OUT_OF_SAMPLE[0], end or OUT_OF_SAMPLE[1])
    widest = TrendConfig().scaled(max(SCALES))
    opens, closes = await load_prices(provider, UNIVERSES["extended"], warmup_start(IN_SAMPLE[0], widest), oos[1])
    lab = Lab(opens, closes)
    base = TrendConfig()

    tables: dict[str, dict[str, dict[str, Any]]] = {}
    runs = {}
    for period_name, period in (("oos", oos), ("is", IN_SAMPLE)):
        for universe in ("core", "extended"):
            key = f"{period_name}_{universe}"
            tables[key] = {}
            for kind in ("strategy", "ew_bh", "btc_bh"):
                run, m = lab.run(kind, universe, period, base)
                runs[(key, kind)] = run
                tables[key][kind] = m
    costly = {u: lab.run("strategy", u, oos, replace(base, cost_bps=base.cost_bps * 2))[1] for u in ("core", "extended")}
    for u, m in costly.items():
        tables[f"oos_{u}"]["strategy_30bps"] = m

    variants = []
    ew_core = tables["oos_core"]["ew_bh"]
    for scale in SCALES:
        for every in FREQUENCIES:
            cfg = replace(base.scaled(scale), every=every)
            m = lab.run("strategy", "core", oos, cfg)[1]
            sharpe_ok, dd_ok = _beats(m, ew_core)
            variants.append({"scale": scale, "every": every, "lookbacks": cfg.lookbacks, "sharpe": m["sharpe"],
                             "max_drawdown": m["max_drawdown"], "cagr": m["cagr"], "pass": sharpe_ok and dd_ok})

    criteria: list[Criterion] = []
    oc, oe = tables["oos_core"], tables["oos_extended"]

    def pct(x: float) -> str:
        return f"{x * 100:+.1f}%"

    criteria.append(Criterion("C1", "CAGR > 0 after costs (core and extended)",
                              oc["strategy"]["cagr"] > 0 and oe["strategy"]["cagr"] > 0,
                              f"core {pct(oc['strategy']['cagr'])}, extended {pct(oe['strategy']['cagr'])}"))
    s_core, d_core = _beats(oc["strategy"], oc["ew_bh"])
    s_ext, d_ext = _beats(oe["strategy"], oe["ew_bh"])
    criteria.append(Criterion("C2", "Sharpe >= equal-weight buy & hold (core and extended)", s_core and s_ext,
                              f"core {oc['strategy']['sharpe']:.2f} vs {oc['ew_bh']['sharpe']:.2f}, extended {oe['strategy']['sharpe']:.2f} vs {oe['ew_bh']['sharpe']:.2f}"))
    criteria.append(Criterion("C3", f"Max drawdown <= {DD_RATIO} x equal-weight buy & hold (core and extended)", d_core and d_ext,
                              f"core {pct(oc['strategy']['max_drawdown'])} vs limit {pct(DD_RATIO * oc['ew_bh']['max_drawdown'])}, "
                              f"extended {pct(oe['strategy']['max_drawdown'])} vs limit {pct(DD_RATIO * oe['ew_bh']['max_drawdown'])}"))
    share = sum(v["pass"] for v in variants) / len(variants)
    criteria.append(Criterion("C4", f">= {ROBUST_SHARE:.0%} of {len(variants)} parameter variants pass C2 and C3 (core)",
                              share >= ROBUST_SHARE, f"{sum(v['pass'] for v in variants)}/{len(variants)} pass ({share:.0%})"))
    criteria.append(Criterion("C5", "CAGR > 0 with doubled costs, 30 bps (core and extended)",
                              costly["core"]["cagr"] > 0 and costly["extended"]["cagr"] > 0,
                              f"core {pct(costly['core']['cagr'])}, extended {pct(costly['extended']['cagr'])}"))
    ic = tables["is_core"]
    is_sharpe = ic["strategy"]["sharpe"] >= ic["ew_bh"]["sharpe"]
    is_dd = abs(ic["strategy"]["max_drawdown"]) < abs(ic["ew_bh"]["max_drawdown"])
    criteria.append(Criterion("C6", "In-sample 2018-2022 sanity: Sharpe >= and drawdown < equal-weight buy & hold (core)", is_sharpe and is_dd,
                              f"Sharpe {ic['strategy']['sharpe']:.2f} vs {ic['ew_bh']['sharpe']:.2f}, max DD {pct(ic['strategy']['max_drawdown'])} vs {pct(ic['ew_bh']['max_drawdown'])}"))

    yearly: dict[str, dict[str, float]] = {}
    for kind in ("strategy", "ew_bh", "btc_bh"):
        run = runs[("oos_core", kind)]
        eq = pd.Series(run.equity, index=pd.to_datetime(run.dates))
        by_year = eq.groupby(eq.index.year).last()
        prev = pd.concat([pd.Series([1.0]), by_year.iloc[:-1]]).to_numpy()
        yearly[kind] = {str(y): float(v / p - 1) for y, v, p in zip(by_year.index, by_year.to_numpy(), prev)}

    # Today's target weights, from the extended universe with the base config.
    cols = list(UNIVERSES["extended"])
    weights_now = {k: round(float(v), 4) for k, v in target_weights(closes[cols], base).iloc[-1].items() if v > 0}

    notes = [
        f"Data: Yahoo daily bars, {closes.index[0].date()} to {closes.index[-1].date()}; the out-of-sample window ends {oos[1]}.",
        "Universe chosen among pairs listed on Revolut X today: survivorship bias favours the extended universe.",
    ]
    significance = _significance(runs, variants)
    return Evaluation(criteria=criteria, tables=tables, variants=variants, yearly=yearly, weights_now=weights_now, notes=notes, significance=significance)


def _significance(runs: dict, variants: list[dict[str, Any]]) -> dict[str, Any]:
    """Informational statistics on the out-of-sample core results. They do not change the verdict."""
    import math

    from .stats import significance as sg

    def rets(key: str, kind: str) -> pd.Series:
        return pd.Series(runs[(key, kind)].equity).pct_change().dropna()

    strat, ew = rets("oos_core", "strategy"), rets("oos_core", "ew_bh")
    n_trials = 1 + len(variants)  # base configuration + robustness variants
    per_period = [v["sharpe"] / math.sqrt(365) for v in variants]
    return {
        "strategy": {
            "psr": sg.probabilistic_sharpe_ratio(strat),
            "deflated": sg.deflated_sharpe_ratio(strat, n_trials, float(pd.Series(per_period).var())),
            "sharpe_ci": sg.bootstrap_ci(strat, sg.annualized_sharpe_stat(365), n_boot=1000),
            "cagr_ci": sg.bootstrap_ci(strat, sg.cagr_stat(365), n_boot=1000),
            "min_track_record_days": sg.min_track_record_length(strat),
        },
        "ew_bh": {
            "psr": sg.probabilistic_sharpe_ratio(ew),
            "sharpe_ci": sg.bootstrap_ci(ew, sg.annualized_sharpe_stat(365), n_boot=1000),
        },
        "strategy_minus_ew_sharpe": sg.sharpe_difference_test(strat, ew, n_boot=1000),
        "n_trials": n_trials,
    }


# ------------------------------------------------------------------ report


LABELS = {"strategy": "Trend portfolio (15 bps)", "strategy_30bps": "Trend portfolio (30 bps)", "ew_bh": "Equal-weight buy & hold", "btc_bh": "BTC buy & hold"}
SECTIONS = {"oos_core": "Out-of-sample 2023 → today, core (BTC, ETH)", "oos_extended": "Out-of-sample 2023 → today, extended (10 pairs)",
            "is_core": "In-sample 2018-2022, core", "is_extended": "In-sample 2018-2022, extended"}


def _p(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:+.1f}%"


def render(ev: Evaluation) -> str:
    out = [
        "# Revolut X trend strategy: pre-registered evaluation",
        "",
        f"## Verdict: **{'VIABLE' if ev.viable else 'NOT VIABLE'}**",
        "",
        "Protocol, parameters and criteria: [STRATEGY.md](../STRATEGY.md), committed before this evaluation was run.",
        "",
        "| | Criterion | Result | Detail |",
        "|---|---|---|---|",
        *[f"| {c.code} | {c.label} | {'PASS' if c.passed else '**FAIL**'} | {c.detail} |" for c in ev.criteria],
    ]
    for key, title in SECTIONS.items():
        out += ["", f"## {title}", "", "| | CAGR | total | vol | Sharpe | max DD | Calmar | exposure | turnover/yr | fees |", "|---|---|---|---|---|---|---|---|---|---|"]
        for kind, m in ev.tables[key].items():
            calmar = "n/a" if m["calmar"] is None else f"{m['calmar']:.2f}"
            out.append(f"| {LABELS[kind]} | {_p(m['cagr'])} | {_p(m['total_return'])} | {_p(m['ann_vol'])} | {m['sharpe']:.2f} | {_p(m['max_drawdown'])} | "
                       f"{calmar} | {m['avg_exposure'] * 100:.0f}% | {m['turnover_per_year']:.1f}x | {_p(m['fees'])} |")
    years = sorted(next(iter(ev.yearly.values())))
    out += ["", "## Calendar years, out-of-sample, core", "", "| | " + " | ".join(years) + " |", "|---|" + "---|" * len(years)]
    out += [f"| {LABELS[k]} | " + " | ".join(_p(v.get(y)) for y in years) + " |" for k, v in ev.yearly.items()]
    sig = ev.significance
    if sig:
        s, e, d = sig["strategy"], sig["ew_bh"], sig["strategy_minus_ew_sharpe"]
        mintrl = s["min_track_record_days"]
        out += [
            "", "## Statistical significance, out-of-sample, core (informational: does not change the verdict)", "",
            "| | value |", "|---|---|",
            f"| Strategy PSR (P[true Sharpe > 0]) | {s['psr']:.2f} |",
            f"| Strategy Deflated Sharpe ({sig['n_trials']} configurations tested) | {s['deflated']['dsr']:.2f} |",
            f"| Strategy Sharpe, 90% stationary-bootstrap CI | {s['sharpe_ci']['estimate']:.2f} [{s['sharpe_ci']['low']:.2f}, {s['sharpe_ci']['high']:.2f}] |",
            f"| Strategy CAGR, 90% CI | {_p(s['cagr_ci']['estimate'])} [{_p(s['cagr_ci']['low'])}, {_p(s['cagr_ci']['high'])}] |",
            f"| Track record needed for 95% confidence in Sharpe > 0 | {'never at this Sharpe' if mintrl == float('inf') else f'{mintrl:.0f} days ({mintrl / 365:.1f} years)'} |",
            f"| EW buy & hold PSR | {e['psr']:.2f} |",
            f"| EW buy & hold Sharpe, 90% CI | {e['sharpe_ci']['estimate']:.2f} [{e['sharpe_ci']['low']:.2f}, {e['sharpe_ci']['high']:.2f}] |",
            f"| Sharpe(strategy) - Sharpe(EW), 90% CI | {d['difference']:+.2f} [{d['low_90']:+.2f}, {d['high_90']:+.2f}], P(strategy better) = {d['p_a_better']:.0%} |",
        ]
    out += ["", "## Parameter robustness (C4), out-of-sample, core", "", "| lookbacks | rebalance | CAGR | Sharpe | max DD | C2 & C3 |", "|---|---|---|---|---|---|"]
    out += [f"| {v['lookbacks']} | {v['every']}d | {_p(v['cagr'])} | {v['sharpe']:.2f} | {_p(v['max_drawdown'])} | {'pass' if v['pass'] else 'fail'} |" for v in ev.variants]
    out += ["", "## Target weights at the last close (extended universe)", ""]
    out += [f"- {k}: {v * 100:.1f}%" for k, v in sorted(ev.weights_now.items(), key=lambda kv: -kv[1])] or ["- 100% EUR (no asset in an uptrend)"]
    out += [f"- EUR: {(1 - sum(ev.weights_now.values())) * 100:.1f}%", "", "## Notes", "", *[f"- {n}" for n in ev.notes]]
    return "\n".join(out) + "\n"
