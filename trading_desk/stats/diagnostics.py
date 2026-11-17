"""Time-series diagnostics: is this market trending, mean-reverting, or a random walk?

  * Hurst exponent: ~0.5 random walk, > 0.5 persistent (trend), < 0.5 anti-persistent.
  * Lo-MacKinlay variance ratio with heteroskedasticity-robust z: VR > 1 momentum, < 1 reversal.
  * Ljung-Box: joint test that return autocorrelations are zero.
  * ADF / KPSS: stationarity (ADF null = unit root, KPSS null = stationary).
  * Trend t-stat: OLS of log price on time with Newey-West (HAC) errors.
  * Half-life: Ornstein-Uhlenbeck mean-reversion speed.
  * Jarque-Bera: normality of returns.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats as st


def _arr(x) -> np.ndarray:
    a = np.asarray(pd.Series(x, dtype=float).dropna(), dtype=float)
    return a[np.isfinite(a)]


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(pd.Series(prices, dtype=float)).diff().dropna()


def hurst_exponent(prices: pd.Series, max_lag: int = 100) -> float | None:
    """Scaling of the dispersion of log-price changes with the horizon: std(x_{t+k} - x_t) ~ k^H."""
    x = np.log(_arr(prices))
    max_lag = min(max_lag, len(x) // 4)
    if max_lag < 8:
        return None
    lags = np.unique(np.logspace(0, math.log10(max_lag), 20).astype(int))
    tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
    ok = tau > 0
    if ok.sum() < 4:
        return None
    return float(np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)[0])


def variance_ratio(prices: pd.Series, q: int = 5) -> dict[str, float]:
    """Lo-MacKinlay (1988) VR(q) on log prices, overlapping, with the heteroskedasticity-robust z*."""
    x = np.log(_arr(prices))
    r = np.diff(x)
    n = len(r)
    if n < 4 * q:
        return {"vr": float("nan"), "z": float("nan"), "p_value": float("nan"), "q": q}
    mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / (n - 1)
    rq = x[q:] - x[:-q]
    m = q * (n - q + 1) * (1 - q / n)
    varq = np.sum((rq - q * mu) ** 2) / m
    vr = varq / var1
    dev2 = (r - mu) ** 2
    denom = np.sum(dev2) ** 2
    theta = sum(
        (2 * (q - j) / q) ** 2 * (n * np.sum(dev2[j:] * dev2[:-j]) / denom)
        for j in range(1, q)
    )
    # theta estimates the asymptotic variance of sqrt(n) * (VR - 1)
    z = math.sqrt(n) * (vr - 1) / math.sqrt(theta) if theta > 0 else float("nan")
    return {"vr": float(vr), "z": float(z), "p_value": float(2 * (1 - st.norm.cdf(abs(z)))), "q": q}


def ljung_box(returns: pd.Series, lags: int = 10) -> dict[str, float]:
    from statsmodels.stats.diagnostic import acorr_ljungbox

    r = _arr(returns)
    res = acorr_ljungbox(r, lags=[lags], return_df=True)
    return {"stat": float(res["lb_stat"].iloc[0]), "p_value": float(res["lb_pvalue"].iloc[0]), "lags": lags}


def autocorrelations(returns: pd.Series, lags: tuple[int, ...] = (1, 2, 5, 10)) -> dict[str, float]:
    r = pd.Series(_arr(returns))
    return {f"acf_{k}": float(r.autocorr(k)) for k in lags if len(r) > k + 2}


def adf(series: pd.Series) -> dict[str, float]:
    from statsmodels.tsa.stattools import adfuller

    with warnings.catch_warnings():  # statsmodels 0.15 FutureWarning about the return type
        warnings.simplefilter("ignore", FutureWarning)
        stat, p, *_ = adfuller(_arr(series), autolag="AIC")
    return {"stat": float(stat), "p_value": float(p)}


def kpss(series: pd.Series) -> dict[str, float]:
    from statsmodels.tsa.stattools import kpss as _kpss

    with warnings.catch_warnings():  # p-values outside the lookup table trigger an InterpolationWarning
        warnings.simplefilter("ignore")
        stat, p, *_ = _kpss(_arr(series), regression="c", nlags="auto")
    return {"stat": float(stat), "p_value": float(p)}


def trend_tstat(prices: pd.Series, window: int | None = None, ppy: int = 365) -> dict[str, float]:
    """Slope of log price on time over the window, with Newey-West standard errors."""
    import statsmodels.api as sm

    y = np.log(_arr(prices))
    if window:
        y = y[-window:]
    n = len(y)
    if n < 20:
        return {"slope_annual": float("nan"), "t_stat": float("nan"), "p_value": float("nan"), "r2": float("nan")}
    X = sm.add_constant(np.arange(n, dtype=float))
    lags = int(math.floor(4 * (n / 100) ** (2 / 9)))
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(lags, 1)})
    return {
        "slope_annual": float(math.expm1(res.params[1] * ppy)),
        "t_stat": float(res.tvalues[1]),
        "p_value": float(res.pvalues[1]),
        "r2": float(res.rsquared),
        "window": float(n),
    }


def half_life(series: pd.Series) -> float | None:
    """Mean-reversion half-life (periods) from Δx_t = a + b·x_{t-1}; None when not mean-reverting."""
    x = _arr(series)
    if len(x) < 20:
        return None
    dx, lag = np.diff(x), x[:-1]
    b = np.polyfit(lag, dx, 1)[0]
    if b >= 0 or b <= -1:
        return None
    hl = -math.log(2) / math.log(1 + b)
    return float(hl) if hl <= len(x) else None  # longer than the sample: not measurable here


def jarque_bera(returns: pd.Series) -> dict[str, float]:
    res = st.jarque_bera(_arr(returns))
    return {"stat": float(res.statistic), "p_value": float(res.pvalue)}


def robust_zscore(series: pd.Series, window: int = 60) -> float | None:
    """How unusual the last value is versus the previous `window`, with median/MAD (outlier-proof)."""
    x = _arr(series)
    if len(x) < window + 1:
        return None
    hist, last = x[-window - 1 : -1], x[-1]
    med = np.median(hist)
    mad = np.median(np.abs(hist - med)) * 1.4826
    return float((last - med) / mad) if mad > 0 else None


def zscore_vs_mean(prices: pd.Series, window: int = 20) -> float | None:
    """Distance of the last price from its moving average, in standard deviations."""
    x = _arr(prices)
    if len(x) < window:
        return None
    w = x[-window:]
    sd = w.std(ddof=1)
    return float((x[-1] - w.mean()) / sd) if sd > 0 else None


def diagnose(prices: pd.Series, ppy: int = 365) -> dict[str, object]:
    """Full diagnostic pack on a price series, with a plain-language market character."""
    r = log_returns(prices)
    vr = variance_ratio(prices, 5)
    hurst = hurst_exponent(prices)
    lb = ljung_box(r, 10) if len(r) > 30 else None
    t60, t120 = trend_tstat(prices, 60, ppy), trend_tstat(prices, 120, ppy)
    if vr["p_value"] < 0.05:
        character = "trending (momentum)" if vr["vr"] > 1 else "mean-reverting"
    else:
        character = "close to a random walk"
    return {
        "character": character,
        "hurst": hurst,
        "variance_ratio_5": vr,
        "variance_ratio_20": variance_ratio(prices, 20),
        "ljung_box_10": lb,
        "autocorrelation": autocorrelations(r),
        "adf_log_price": adf(np.log(_arr(prices))) if len(r) > 30 else None,
        "kpss_returns": kpss(r) if len(r) > 30 else None,
        "trend_60d": t60,
        "trend_120d": t120,
        "half_life_log_price": half_life(np.log(_arr(prices))),
        "jarque_bera": jarque_bera(r) if len(r) > 30 else None,
    }
