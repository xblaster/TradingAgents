"""Event study of the gem rules: do flagged assets outperform afterwards?

For every screening date (weekly by default) and every gem raised that day, we measure the
asset's forward return over each horizon, and its excess over the equal-weighted universe on the
same window. Per rule we report the count, mean forward and excess returns, a t-test of the mean
excess, and the hit rate. Signals are computed with data up to the screening date only.

Caveats printed with the results: events overlap (weekly screens, 30-day horizons), so the
effective sample is smaller than the count and t-stats are optimistic; 5 rules x N assets are
tested, so an isolated "significant" rule may be luck.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as st

from .data import DataProvider
from .screener import RULES, screen

HORIZONS = (7, 30)


@dataclass
class Event:
    when: date
    ticker: str
    rule: str
    forward: dict[int, float]
    excess: dict[int, float]


async def gem_event_study(
    provider: DataProvider, tickers: tuple[str, ...], start: date, end: date, *, step: int = 7, horizons: tuple[int, ...] = HORIZONS
) -> dict[str, Any]:
    closes = pd.DataFrame({t: (await provider.prices(t, start, end + timedelta(days=max(horizons) + 5)))["Close"] for t in tickers}).sort_index().ffill()
    events: list[Event] = []
    dates = pd.date_range(start, end, freq=f"{step}D")
    for d in dates:
        day = d.date()
        if pd.Timestamp(day) not in closes.index:
            continue
        sc = await screen(provider, tickers, day)
        i = closes.index.get_loc(pd.Timestamp(day))
        for ticker, gem in sc.gems:
            fwd, exc = {}, {}
            for h in horizons:
                if i + h >= len(closes):
                    continue
                row_now, row_fut = closes.iloc[i], closes.iloc[i + h]
                asset = row_fut[ticker] / row_now[ticker] - 1
                universe = float((row_fut / row_now - 1).dropna().mean())
                fwd[h], exc[h] = float(asset), float(asset - universe)
            if fwd:
                events.append(Event(day, ticker, gem.rule, fwd, exc))

    def summarize(evs: list[Event]) -> dict[str, Any]:
        out: dict[str, Any] = {"events": len(evs), "distinct_dates": len({e.when for e in evs})}
        for h in horizons:
            ex = np.array([e.excess[h] for e in evs if h in e.excess])
            fw = np.array([e.forward[h] for e in evs if h in e.forward])
            if len(ex) < 3:
                out[f"h{h}"] = None
                continue
            t, p = st.ttest_1samp(ex, 0.0)
            out[f"h{h}"] = {"n": int(len(ex)), "mean_forward": float(fw.mean()), "mean_excess": float(ex.mean()),
                            "median_excess": float(np.median(ex)), "t_stat": float(t), "p_value": float(p), "hit_rate": float((ex > 0).mean())}
        return out

    by_rule = {code: summarize([e for e in events if e.rule == code]) for code in RULES}
    return {
        "start": start.isoformat(), "end": end.isoformat(), "step_days": step, "screens": int(len(dates)), "assets": list(tickers),
        "horizons": list(horizons), "all_gems": summarize(events), "by_rule": by_rule,
        "events": [{"date": e.when.isoformat(), "ticker": e.ticker, "rule": e.rule, **{f"excess_{h}d": e.excess.get(h) for h in horizons}} for e in events],
    }


def render_gem_study(res: dict[str, Any]) -> str:
    hs = res["horizons"]
    out = [
        f"# Gem rules event study {res['start']} → {res['end']}",
        "",
        f"{res['screens']} screens every {res['step_days']} days over {len(res['assets'])} assets. Excess = asset forward return minus the equal-weighted universe on the same window.",
        "",
        "| rule | events | dates | " + " | ".join(f"excess {h}d | t | p | hit {h}d" for h in hs) + " |",
        "|---|---|---|" + "---|---|---|---|" * len(hs),
    ]
    rows = [("All gems", res["all_gems"])] + [(f"{c} {RULES[c].name}", s) for c, s in res["by_rule"].items()]
    for label, s in rows:
        cells = []
        for h in hs:
            v = s.get(f"h{h}")
            cells.append("n/a | | |" if not v else f"{v['mean_excess'] * 100:+.2f}% | {v['t_stat']:+.2f} | {v['p_value']:.3f} | {v['hit_rate']:.0%}")
        out.append(f"| {label} | {s['events']} | {s['distinct_dates']} | " + " | ".join(cells) + " |")
    out += [
        "",
        "Reading guide: a rule is informative if its mean excess is positive with |t| well above 2 at several horizons.",
        "Events overlap (weekly screens, 30-day horizons) so t-stats are optimistic; with 5 rules tested, expect one",
        "spurious p < 0.05 about a quarter of the time. Treat this as evidence for or against routing gems to the LLM,",
        "not as a trading signal on its own.",
    ]
    return "\n".join(out) + "\n"
