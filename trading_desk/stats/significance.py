"""Is a performance statistically real? Tests that account for fat tails and multiple testing.

References:
  Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier" (PSR, MinTRL).
  Bailey & López de Prado (2014), "The Deflated Sharpe Ratio" (DSR).
  Politis & Romano (1994), "The Stationary Bootstrap".
All Sharpe ratios inside the formulas are per period (not annualized).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as st

EULER_GAMMA = 0.5772156649015329


def _moments(returns: pd.Series) -> tuple[float, int, float, float]:
    r = pd.Series(returns, dtype=float).dropna()
    sd = r.std()
    sr = float(r.mean() / sd) if sd > 0 else 0.0
    return sr, len(r), float(st.skew(r)), float(st.kurtosis(r, fisher=False))  # raw kurtosis (normal = 3)


def mean_return_tstat(returns: pd.Series) -> dict[str, float]:
    """Classic t-test of a positive mean return (assumes i.i.d. returns)."""
    r = pd.Series(returns, dtype=float).dropna()
    t, p_two = st.ttest_1samp(r, 0.0)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    return {"t_stat": float(t), "p_value_positive_mean": float(p_one)}


def _sr_std(sr: float, skew: float, kurt: float) -> float:
    return math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark), with the benchmark as a per-period Sharpe ratio."""
    sr, n, skew, kurt = _moments(returns)
    if n < 3:
        return float("nan")
    return float(st.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / _sr_std(sr, skew, kurt)))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period Sharpe among `n_trials` unskilled strategies (all true SR = 0)."""
    if n_trials <= 1:
        return 0.0
    a = st.norm.ppf(1 - 1 / n_trials)
    b = st.norm.ppf(1 - 1 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int, sr_variance: float | None = None) -> dict[str, float]:
    """PSR against the Sharpe you would expect from the best of `n_trials` lucky strategies.

    `sr_variance` is the variance of the per-period Sharpe ratios across the trials; if unknown,
    the sampling variance of a Sharpe estimate under the null is used (1/n)."""
    sr, n, _, _ = _moments(returns)
    var = sr_variance if sr_variance is not None else 1 / max(n - 1, 1)
    sr0 = expected_max_sharpe(n_trials, var)
    return {"dsr": probabilistic_sharpe_ratio(returns, sr0), "sr0_per_period": sr0, "sharpe_per_period": sr, "n_trials": float(n_trials)}


def min_track_record_length(returns: pd.Series, sr_benchmark: float = 0.0, confidence: float = 0.95) -> float:
    """Periods of track record needed for PSR(sr_benchmark) to reach `confidence`."""
    sr, _, skew, kurt = _moments(returns)
    if sr <= sr_benchmark:
        return float("inf")
    z = st.norm.ppf(confidence)
    return float(1 + _sr_std(sr, skew, kurt) ** 2 * (z / (sr - sr_benchmark)) ** 2)


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano: blocks of geometric length (mean `mean_block`), wrapping around."""
    p = 1 / mean_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    restarts = rng.random(n) < p
    starts = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if restarts[t] else (idx[t - 1] + 1) % n
    return idx


def bootstrap_ci(
    returns: pd.Series,
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int = 1000,
    mean_block: float = 20.0,
    level: float = 0.90,
    seed: int = 7,
) -> dict[str, float]:
    """Confidence interval of `statistic` that preserves volatility clustering and autocorrelation."""
    r = pd.Series(returns, dtype=float).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.array([statistic(r[stationary_bootstrap_indices(len(r), mean_block, rng)]) for _ in range(n_boot)])
    draws = draws[np.isfinite(draws)]
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"estimate": float(statistic(r)), "low": float(lo), "high": float(hi), "level": level, "share_positive": float((draws > 0).mean())}


def annualized_sharpe_stat(ppy: int) -> Callable[[np.ndarray], float]:
    def stat(x: np.ndarray) -> float:
        sd = x.std(ddof=1)
        return float(x.mean() / sd * math.sqrt(ppy)) if sd > 0 else float("nan")
    return stat


def cagr_stat(ppy: int) -> Callable[[np.ndarray], float]:
    def stat(x: np.ndarray) -> float:
        growth = float(np.prod(1 + x))
        return growth ** (ppy / len(x)) - 1 if growth > 0 else -1.0
    return stat


def sharpe_difference_test(a: pd.Series, b: pd.Series, *, n_boot: int = 1000, mean_block: float = 20.0, ppy: int = 365, seed: int = 7) -> dict[str, float]:
    """Paired stationary bootstrap of Sharpe(a) - Sharpe(b) on aligned returns."""
    df = pd.concat([pd.Series(a, dtype=float), pd.Series(b, dtype=float)], axis=1, join="inner").dropna().to_numpy()
    rng = np.random.default_rng(seed)
    f = annualized_sharpe_stat(ppy)
    diff = f(df[:, 0]) - f(df[:, 1])
    draws = []
    for _ in range(n_boot):
        s = df[stationary_bootstrap_indices(len(df), mean_block, rng)]
        draws.append(f(s[:, 0]) - f(s[:, 1]))
    draws = np.array([d for d in draws if np.isfinite(d)])
    lo, hi = np.quantile(draws, [0.05, 0.95])
    return {"difference": float(diff), "low_90": float(lo), "high_90": float(hi), "p_a_better": float((draws > 0).mean())}
