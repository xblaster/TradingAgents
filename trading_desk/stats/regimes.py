"""Market regimes: volatility percentile, trend state and a two-state Markov-switching model."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from .diagnostics import log_returns, trend_tstat


def volatility_regime(prices: pd.Series, window: int = 20, history: int = 365, ppy: int = 365) -> dict[str, float | str]:
    r = log_returns(prices)
    vol = r.rolling(window, min_periods=window).std() * math.sqrt(ppy)
    hist = vol.dropna().iloc[-history:]
    if hist.empty:
        return {"label": "unknown"}
    current = float(hist.iloc[-1])
    pct = float((hist <= current).mean())
    label = "low" if pct < 0.25 else "normal" if pct < 0.75 else "high" if pct < 0.95 else "extreme"
    return {"label": label, "current_vol": current, "percentile_1y": pct}


def trend_regime(prices: pd.Series, ppy: int = 365) -> dict[str, float | str]:
    """Trend state from 60- and 120-day HAC t-stats (|t| > 2 is significant).

    Both significant in the same direction: uptrend/downtrend. Significant in opposite directions:
    transition (the recent move fights the longer trend). Otherwise: range."""
    t60, t120 = trend_tstat(prices, 60, ppy)["t_stat"], trend_tstat(prices, 120, ppy)
    long_t = t120["t_stat"]
    sig = lambda t: 1 if t > 2 else -1 if t < -2 else 0  # noqa: E731
    s60, s120 = sig(t60), sig(long_t)
    if s60 and s120 and s60 != s120:
        label = "transition"
    elif (s120 or s60) > 0:
        label = "uptrend"
    elif (s120 or s60) < 0:
        label = "downtrend"
    else:
        label = "range"
    return {"label": label, "t_stat_60d": t60, "t_stat_120d": long_t, "slope_annual_120d": t120["slope_annual"]}


def markov_switching(prices: pd.Series, window: int = 730) -> dict[str, float] | None:
    """Two-state Markov-switching model of daily returns with regime-specific mean and variance.

    Returns the smoothed probability of being in the high-volatility state today and each state's
    annualized mean and volatility. None if the model does not converge."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    r = log_returns(prices).iloc[-window:] * 100
    if len(r) < 200:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = MarkovRegression(r.to_numpy(), k_regimes=2, trend="c", switching_variance=True).fit(disp=False)
    except Exception:
        return None
    names = list(res.model.param_names)
    params = dict(zip(names, np.asarray(res.params)))
    variances = [params[f"sigma2[{k}]"] for k in range(2)]
    means = [params[f"const[{k}]"] for k in range(2)]
    hi = int(np.argmax(variances))
    probs = np.asarray(res.smoothed_marginal_probabilities)
    p_hi = float(probs[-1, hi] if probs.ndim == 2 else probs[hi][-1])
    lo = 1 - hi
    return {
        "p_high_vol_state": p_hi,
        "high_state_ann_vol": float(math.sqrt(variances[hi]) / 100 * math.sqrt(365)),
        "low_state_ann_vol": float(math.sqrt(variances[lo]) / 100 * math.sqrt(365)),
        "high_state_ann_mean": float(means[hi] / 100 * 365),
        "low_state_ann_mean": float(means[lo] / 100 * 365),
    }
