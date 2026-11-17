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


def realized_vol(close: pd.Series, n: int = 20) -> pd.Series:
    return np.log(close).diff().rolling(n, min_periods=n).std() * math.sqrt(TRADING_DAYS)


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


def technical_summary(bars: pd.DataFrame) -> dict[str, float | None]:
    """Latest indicator values for a daily OHLCV frame (columns Open/High/Low/Close/Volume)."""
    close, high, low, volume = bars["Close"], bars["High"], bars["Low"], bars["Volume"]
    last = float(close.iloc[-1])
    window_1y = close.iloc[-TRADING_DAYS:]
    m = macd(close)
    bb = bollinger(close)
    sma50, sma200 = _last(sma(close, 50)), _last(sma(close, 200))
    atr14 = _last(atr(high, low, close))
    bb_upper, bb_lower = _last(bb["upper"]), _last(bb["lower"])
    vol20 = _last(volume.rolling(20, min_periods=20).mean())

    return {
        "last_close": _r(last),
        "return_1w": _r(trailing_return(close, 5)),
        "return_1m": _r(trailing_return(close, 21)),
        "return_3m": _r(trailing_return(close, 63)),
        "return_6m": _r(trailing_return(close, 126)),
        "return_1y": _r(trailing_return(close, 252)),
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
        "realized_vol_20d": _r(_last(realized_vol(close))),
        "max_drawdown_1y": _r(max_drawdown(window_1y)),
        "pct_from_52w_high": _r(last / float(window_1y.max()) - 1),
        "pct_from_52w_low": _r(last / float(window_1y.min()) - 1),
        "volume_vs_20d_avg": _r(float(volume.iloc[-1]) / vol20 if vol20 else None),
    }
