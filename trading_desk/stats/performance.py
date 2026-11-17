"""Return and risk statistics on a series of periodic simple returns."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as st


EPS = 1e-12  # below this, a standard deviation is floating-point noise


def _clean(returns: pd.Series | np.ndarray) -> pd.Series:
    r = pd.Series(returns, dtype=float).dropna()
    return r[np.isfinite(r)]


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + _clean(returns)).cumprod()


def cagr(returns: pd.Series, ppy: int) -> float | None:
    r = _clean(returns)
    if r.empty:
        return None
    growth = float((1 + r).prod())
    years = len(r) / ppy
    return growth ** (1 / years) - 1 if growth > 0 and years > 0 else -1.0


def ann_vol(returns: pd.Series, ppy: int) -> float:
    return float(_clean(returns).std() * math.sqrt(ppy))


def sharpe(returns: pd.Series, ppy: int, rf: float = 0.0) -> float | None:
    r = _clean(returns) - rf / ppy
    sd = r.std()
    return float(r.mean() / sd * math.sqrt(ppy)) if sd > EPS else None


def sortino(returns: pd.Series, ppy: int, target: float = 0.0) -> float | None:
    r = _clean(returns)
    downside = np.minimum(r - target / ppy, 0)
    dd = math.sqrt(float((downside**2).mean()))
    return float((r.mean() - target / ppy) / dd * math.sqrt(ppy)) if dd > EPS else None


def omega(returns: pd.Series, threshold: float = 0.0) -> float | None:
    """Probability-weighted gains over losses relative to a per-period threshold."""
    r = _clean(returns) - threshold
    losses = -r[r < 0].sum()
    return float(r[r > 0].sum() / losses) if losses > 0 else None


def drawdown(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown(returns)
    return float(dd.min()) if not dd.empty else 0.0


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest number of periods spent below a previous equity peak."""
    underwater = (drawdown(returns) < 0).to_numpy()
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def ulcer_index(returns: pd.Series) -> float:
    """Root mean square of percentage drawdowns: penalizes both depth and duration."""
    dd = drawdown(returns)
    return float(math.sqrt((dd**2).mean())) if not dd.empty else 0.0


def calmar(returns: pd.Series, ppy: int) -> float | None:
    mdd = max_drawdown(returns)
    g = cagr(returns, ppy)
    return g / abs(mdd) if mdd < 0 and g is not None else None


def var_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """Per-period Value-at-Risk (a loss, reported as a negative return)."""
    return float(np.quantile(_clean(returns), alpha))


def cvar_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    r = _clean(returns)
    tail = r[r <= np.quantile(r, alpha)]
    return float(tail.mean()) if not tail.empty else float("nan")


def var_cornish_fisher(returns: pd.Series, alpha: float = 0.05) -> float:
    """Parametric VaR corrected for skewness and excess kurtosis (Cornish-Fisher expansion)."""
    r = _clean(returns)
    z = st.norm.ppf(alpha)
    s, k = float(st.skew(r)), float(st.kurtosis(r))  # excess kurtosis
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return float(r.mean() + z_cf * r.std())


def tail_ratio(returns: pd.Series) -> float | None:
    r = _clean(returns)
    lo = abs(np.quantile(r, 0.05))
    return float(np.quantile(r, 0.95) / lo) if lo > 0 else None


def hit_rate(returns: pd.Series) -> float:
    r = _clean(returns)
    r = r[r != 0]
    return float((r > 0).mean()) if len(r) else float("nan")


def profit_factor(returns: pd.Series) -> float | None:
    r = _clean(returns)
    losses = -r[r < 0].sum()
    return float(r[r > 0].sum() / losses) if losses > 0 else None


def summary(returns: pd.Series, ppy: int) -> dict[str, float | int | None]:
    r = _clean(returns)
    if len(r) < 3:
        return {"observations": len(r)}
    return {
        "observations": len(r),
        "cagr": cagr(r, ppy),
        "ann_vol": ann_vol(r, ppy),
        "sharpe": sharpe(r, ppy),
        "sortino": sortino(r, ppy),
        "calmar": calmar(r, ppy),
        "omega": omega(r),
        "max_drawdown": max_drawdown(r),
        "max_drawdown_duration": max_drawdown_duration(r),
        "ulcer_index": ulcer_index(r),
        "var_95": var_historical(r, 0.05),
        "cvar_95": cvar_historical(r, 0.05),
        "var_95_cornish_fisher": var_cornish_fisher(r, 0.05),
        "skew": float(st.skew(r)),
        "excess_kurtosis": float(st.kurtosis(r)),
        "tail_ratio": tail_ratio(r),
        "hit_rate": hit_rate(r),
        "profit_factor": profit_factor(r),
        "best_period": float(r.max()),
        "worst_period": float(r.min()),
    }
