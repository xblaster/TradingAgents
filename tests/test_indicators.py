import numpy as np
import pandas as pd
import pytest

from trading_desk.data import indicators as ind

from .conftest import make_bars


def test_sma_and_trailing_return():
    s = pd.Series(np.arange(1, 11, dtype=float))
    assert ind.sma(s, 3).iloc[-1] == pytest.approx(9.0)
    assert ind.trailing_return(s, 1) == pytest.approx(10 / 9 - 1)
    assert ind.trailing_return(s, 20) is None


def test_rsi_extremes():
    up = pd.Series(np.arange(1, 50, dtype=float))
    down = up[::-1].reset_index(drop=True)
    assert ind.rsi(up).iloc[-1] == pytest.approx(100.0)
    assert ind.rsi(down).iloc[-1] == pytest.approx(0.0)
    assert ind.rsi(up).iloc[:13].isna().all()


def test_atr_constant_range():
    n = 50
    close = pd.Series([100.0] * n)
    atr = ind.atr(close + 1, close - 1, close)
    assert atr.iloc[-1] == pytest.approx(2.0)


def test_max_drawdown():
    assert ind.max_drawdown(pd.Series([100, 120, 60, 130.0])) == pytest.approx(-0.5)


def test_technical_summary_is_complete_and_finite():
    summary = ind.technical_summary(make_bars(300))
    assert summary["sma_200"] is not None and summary["return_1y"] is not None
    assert 0 <= summary["rsi_14"] <= 100
    assert all(v is None or np.isfinite(v) for v in summary.values())


def test_technical_summary_short_history_has_gaps():
    summary = ind.technical_summary(make_bars(80))
    assert summary["sma_200"] is None and summary["price_vs_sma200"] is None
    assert summary["sma_50"] is not None
