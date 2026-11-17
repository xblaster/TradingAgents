"""Cross-asset statistics across a universe: correlation, beta, ranks, cointegration."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from .diagnostics import half_life


def returns_frame(closes: pd.DataFrame) -> pd.DataFrame:
    return np.log(closes).diff()


def correlation_matrix(closes: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    return returns_frame(closes).iloc[-window:].corr()


def average_correlation(closes: pd.DataFrame, window: int = 90) -> float:
    """Mean pairwise correlation: close to 1 means the universe offers little diversification."""
    c = correlation_matrix(closes, window).to_numpy()
    iu = np.triu_indices_from(c, k=1)
    vals = c[iu]
    return float(np.nanmean(vals)) if vals.size else float("nan")


def beta(closes: pd.DataFrame, asset: str, benchmark: str, window: int = 90) -> float | None:
    r = returns_frame(closes[[asset, benchmark]]).iloc[-window:].dropna()
    var = r[benchmark].var()
    return float(r[asset].cov(r[benchmark]) / var) if var > 0 and len(r) > 20 else None


def momentum_ranks(closes: pd.DataFrame, horizons: dict[str, int] | None = None) -> pd.DataFrame:
    """Cross-sectional percentile rank (0 worst, 1 best) of trailing returns, plus their average."""
    horizons = horizons or {"1m": 30, "3m": 91, "6m": 182}
    last = closes.ffill().iloc[-1]
    out = pd.DataFrame(index=closes.columns)
    for label, days in horizons.items():
        past = closes.ffill().shift(days).iloc[-1]
        out[f"ret_{label}"] = last / past - 1
        out[f"rank_{label}"] = out[f"ret_{label}"].rank(pct=True)
    out["rank_avg"] = out[[c for c in out.columns if c.startswith("rank_")]].mean(axis=1)
    return out


def dispersion(closes: pd.DataFrame, days: int = 30) -> float:
    """Cross-sectional standard deviation of trailing returns: high when assets diverge."""
    ffilled = closes.ffill()
    rets = ffilled.iloc[-1] / ffilled.shift(days).iloc[-1] - 1
    return float(rets.std())


def cointegration_scan(closes: pd.DataFrame, window: int = 365, max_p: float = 0.05) -> list[dict[str, float | str]]:
    """Engle-Granger test on every pair of log prices; for cointegrated pairs, spread z-score and half-life.

    With n assets this runs n(n-1)/2 tests: at 5% about 1 in 20 pairs will look cointegrated by chance,
    so a hit is a lead to examine, not a proof."""
    from statsmodels.tsa.stattools import coint

    import warnings

    logp = np.log(closes.iloc[-window:]).dropna(axis=1, thresh=int(window * 0.9)).dropna()
    hits = []
    for a, b in itertools.combinations(logp.columns, 2):
        if logp[a].std() == 0 or logp[b].std() == 0 or abs(logp[a].corr(logp[b])) > 0.9999:
            continue  # constant or duplicate series: the test is meaningless
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p, _ = coint(logp[a], logp[b])
        except Exception:
            continue
        if p > max_p:
            continue
        hedge = np.polyfit(logp[b], logp[a], 1)[0]
        spread = logp[a] - hedge * logp[b]
        if spread.std() == 0:
            continue
        z = (spread.iloc[-1] - spread.mean()) / spread.std()
        hits.append({"a": a, "b": b, "p_value": float(p), "hedge_ratio": float(hedge), "spread_z": float(z),
                     "half_life": half_life(spread), "cheap_leg": a if z < 0 else b, "tests_run": math.comb(logp.shape[1], 2)})
    return sorted(hits, key=lambda h: h["p_value"])
