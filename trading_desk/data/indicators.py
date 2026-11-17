"""Technical indicators as plain pandas functions. No hidden state, easy to test."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n, min_periods=n).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    out = out.where(loss != 0, 100.0)  # only gains: RSI saturates at 100
    return out.where(gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    std = close.rolling(n, min_periods=n).std()
    return pd.DataFrame({"mid": mid, "upper": mid + k * std, "lower": mid - k * std})


def realized_vol(close: pd.Series, n: int = 20, periods_per_year: int = TRADING_DAYS) -> pd.Series:
    return np.log(close).diff().rolling(n, min_periods=n).std() * math.sqrt(periods_per_year)


def max_drawdown(close: pd.Series) -> float:
    if close.empty:
        return float("nan")
    return float((close / close.cummax() - 1).min())


def trailing_return(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1)


def _last(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = series.iloc[-1]
    return None if pd.isna(value) else float(value)


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else round(value, digits)


CRYPTO_DAYS = 365  # crypto trades every day


def market_structure(bars: pd.DataFrame, benchmark: pd.Series | None = None, periods_per_year: int = CRYPTO_DAYS) -> dict[str, float | None]:
    """Regime, participation and co-movement metrics: the crypto stand-in for fundamentals."""
    close, high, low, volume = bars["Close"], bars["High"], bars["Low"], bars["Volume"]
    vol = realized_vol(close, 20, periods_per_year)
    vol_median_1y = vol.iloc[-periods_per_year:].median()
    out: dict[str, float | None] = {
        "vol_regime_20d_vs_1y_median": _r(_last(vol) / vol_median_1y if vol_median_1y and _last(vol) else None),
        "volume_7d_vs_30d": _r(volume.tail(7).mean() / volume.tail(30).mean() if volume.tail(30).mean() else None),
        "up_days_ratio_30d": _r(float((close.diff().tail(30) > 0).mean())),
        "avg_daily_range_14d": _r(float(((high - low) / close).tail(14).mean())),
        "pct_from_lookback_high": _r(float(close.iloc[-1] / close.max() - 1)),
        "days_since_lookback_high": float(len(close) - 1 - int(np.argmax(close.to_numpy()))),
    }
    if benchmark is not None and len(benchmark) > 30:
        joined = pd.concat([np.log(close).diff(), np.log(benchmark).diff()], axis=1, join="inner").dropna().tail(90)
        if len(joined) > 30:
            a, b = joined.iloc[:, 0], joined.iloc[:, 1]
            out["correlation_to_benchmark_90d"] = _r(float(a.corr(b)))
            out["beta_to_benchmark_90d"] = _r(float(a.cov(b) / b.var()) if b.var() else None)
    return out


def horizons(periods_per_year: int) -> dict[str, int]:
    """Bars in 1w/1m/3m/6m/1y for a market with `periods_per_year` daily bars."""
    if periods_per_year >= CRYPTO_DAYS:
        return {"1w": 7, "1m": 30, "3m": 91, "6m": 182, "1y": 365}
    return {"1w": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 252}


def technical_summary(bars: pd.DataFrame, periods_per_year: int = TRADING_DAYS) -> dict[str, float | None]:
    """Latest indicator values for a daily OHLCV frame (columns Open/High/Low/Close/Volume)."""
    close, high, low, volume = bars["Close"], bars["High"], bars["Low"], bars["Volume"]
    last = float(close.iloc[-1])
    h = horizons(periods_per_year)
    window_1y = close.iloc[-h["1y"]:]
    m = macd(close)
    bb = bollinger(close)
    sma50, sma200 = _last(sma(close, 50)), _last(sma(close, 200))
    atr14 = _last(atr(high, low, close))
    bb_upper, bb_lower = _last(bb["upper"]), _last(bb["lower"])
    vol20 = _last(volume.rolling(20, min_periods=20).mean())

    return {
        "last_close": _r(last),
        **{f"return_{label}": _r(trailing_return(close, n)) for label, n in h.items()},
        "sma_20": _r(_last(sma(close, 20))),
        "sma_50": _r(sma50),
        "sma_200": _r(sma200),
        "price_vs_sma50": _r(last / sma50 - 1 if sma50 else None),
        "price_vs_sma200": _r(last / sma200 - 1 if sma200 else None),
        "rsi_14": _r(_last(rsi(close)), 2),
        "macd": _r(_last(m["macd"])),
        "macd_signal": _r(_last(m["signal"])),
        "macd_hist": _r(_last(m["hist"])),
        "atr_14": _r(atr14),
        "atr_pct": _r(atr14 / last if atr14 else None),
        "bollinger_pct_b": _r((last - bb_lower) / (bb_upper - bb_lower) if bb_upper and bb_lower and bb_upper != bb_lower else None),
        "realized_vol_20d": _r(_last(realized_vol(close, 20, periods_per_year))),
        "max_drawdown_1y": _r(max_drawdown(window_1y)),
        "pct_from_52w_high": _r(last / float(window_1y.max()) - 1),
        "pct_from_52w_low": _r(last / float(window_1y.min()) - 1),
        "volume_vs_20d_avg": _r(float(volume.iloc[-1]) / vol20 if vol20 else None),
    }
